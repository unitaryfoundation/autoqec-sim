#!/usr/bin/env bash
# Fresh-environment reproduction for autoqec-sim: new venv, install the
# package from a built wheel, run the test suite and both
# reduced-budget rediscovery regressions (evolutionary search and PPO),
# print a summary. docs/reproduce.md has the expected outputs.
#
#   tools/reproduce.sh                # about 3.5 min on a laptop CPU (201 s measured)
#   REPRODUCE_NO_RL=1 tools/reproduce.sh   # skip torch and the PPO regression
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${REPRODUCE_WORK:-$(mktemp -d)}"
VENV="$WORK/venv"
PY="${REPRODUCE_PYTHON:-3.12}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
t0=$(date +%s)

uv venv -q --python "$PY" "$VENV"
# Install from the built wheel of this checkout so the installed code, not
# the source tree, is what runs.
mkdir -p "$WORK/dist"
uv build -q --out-dir "$WORK/dist" "$ROOT"
uv pip install -q -p "$VENV/bin/python" pytest
if [ -n "${REPRODUCE_NO_RL:-}" ]; then
  uv pip install -q -p "$VENV/bin/python" "$WORK"/dist/autoqec_sim-*.whl
else
  # CPU torch first, from the pytorch-cpu index alone (mixing indexes at
  # install time would also pull build dependencies from it).
  uv pip install -q -p "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch>=2.14.0"
  uv pip install -q -p "$VENV/bin/python" "$("$VENV/bin/python" -c "import glob; print(glob.glob('$WORK/dist/autoqec_sim-*.whl')[0])")[rl]"
fi
t_install=$(date +%s)

cd "$WORK"
TEST_OUT=$("$VENV/bin/python" -m pytest "$ROOT/tests" -q -o addopts="" -p no:cacheprovider 2>&1 | tail -1)
t_tests=$(date +%s)

# Both CLIs exit 1 when a target fails its pass criterion; the status is
# reported in the summary rather than aborting the script.
REGRESS_STATUS=0
"$VENV/bin/autoqec-regress" --budget reduced --seed 0 --out "$WORK/regression_reduced.json" > "$WORK/regress.log" 2>&1 || REGRESS_STATUS=$?
t_regress=$(date +%s)
PPO_STATUS=skipped
if [ -z "${REPRODUCE_NO_RL:-}" ]; then
  PPO_STATUS=0
  "$VENV/bin/autoqec-ppo" --budget reduced --seed 0 --out "$WORK/regression_ppo_reduced.json" \
      --compare "$ROOT/experiments/regression/regression.json" > "$WORK/ppo.log" 2>&1 || PPO_STATUS=$?
fi
t_ppo=$(date +%s)

echo "autoqec_sim $("$VENV/bin/python" -c 'import autoqec_sim; print(autoqec_sim.__version__)')"
echo "python: $("$VENV/bin/python" -V), numpy: $("$VENV/bin/python" -c 'import numpy; print(numpy.__version__)'), torch: $("$VENV/bin/python" -c 'import importlib.util as u; print(__import__("torch").__version__ if u.find_spec("torch") else "not installed")')"
echo "installed from: $("$VENV/bin/python" -c 'import autoqec_sim; print(autoqec_sim.__file__)')"
echo "tests: $TEST_OUT ($((t_tests - t_install)) s)"
echo "exit status: autoqec-regress $REGRESS_STATUS, autoqec-ppo $PPO_STATUS (0 = every target met its pass criterion)"
"$VENV/bin/python" - "$WORK" <<'PYEOF'
import json, os, sys
work = sys.argv[1]
d = json.load(open(os.path.join(work, "regression_reduced.json")))
print("evolutionary regression, reduced budget, seed %d:" % d["seed"])
for name, t in d["targets"].items():
    v = (t.get("result") or {}).get("verification") or {}
    print("  %-32s passed=%-5s rediscovered=%-5s attempts=%d evaluations=%-7d wall=%6.1f s  d=%s w*=%s match=%s" % (
        name, t["passed"], t["rediscovered"], t["attempts_used"], t["total_evaluations"], t["wall_time_s"],
        v.get("distance_exact", "-"), v.get("min_max_generator_weight", "-"), v.get("enumerators_match_reference", "-")))
p = os.path.join(work, "regression_ppo_reduced.json")
if os.path.exists(p):
    d = json.load(open(p))
    print("PPO regression, reduced budget, seed %d:" % d["seed"])
    for name, t in d["targets"].items():
        v = (t.get("result") or {}).get("verification") or {}
        print("  %-32s rediscovered=%-5s env_steps=%-7d episodes=%-6d wall=%6.1f s  d=%s match=%s" % (
            name, t["rediscovered"], t["total_env_steps"], t["total_episodes"], t["wall_time_s"],
            v.get("distance_exact", "-"), v.get("enumerators_match_reference", "-")))
PYEOF
t1=$(date +%s)
echo "wall time: install $((t_install - t0)) s, tests $((t_tests - t_install)) s, evolutionary $((t_regress - t_tests)) s, ppo $((t_ppo - t_regress)) s, total $((t1 - t0)) s"
echo "outputs in $WORK (regression_reduced.json, regression_ppo_reduced.json, *.log)"
