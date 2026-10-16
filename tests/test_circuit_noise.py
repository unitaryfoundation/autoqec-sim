import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import numpy as np
import pytest
import stim
from _helpers import gates_from_stim

from autoqec_sim import graphs, known_codes
from autoqec_sim import noise as nm
from autoqec_sim.circuit_noise import (
    ConnectivityError,
    circuit_depth,
    circuit_level_reward,
    dem_undetected_mass,
    depolarizing_noise_model,
    direct_undetected_mass,
    fault_set,
    gates_to_layered_circuit,
    make_circuit_level_reward,
    resolve_noise_model,
)
from autoqec_sim.env import CodeDiscoveryEnv
from autoqec_sim.noise import presets
from autoqec_sim.regress import replay
from autoqec_sim.tableau import gates_to_stim_circuit


def _steane_encoder():
    code = known_codes.get("steane")
    return code, gates_from_stim(code.to_stim_tableau().to_circuit("elimination"))


# Two Steane encoders found by the evolutionary search (7 qubits, H and
# CNOT). DEEP has 14 CNOTs in 11 layers, SHALLOW 12 CNOTs in 8 layers.
STEANE_DEEP = [
    ("H", 6, -1), ("CNOT", 6, 4), ("H", 3, -1), ("CNOT", 2, 3), ("CNOT", 2, 5), ("CNOT", 3, 1),
    ("H", 1, -1), ("CNOT", 3, 0), ("CNOT", 4, 2), ("H", 6, -1), ("CNOT", 6, 0), ("CNOT", 6, 5),
    ("CNOT", 2, 5), ("CNOT", 6, 2), ("CNOT", 1, 2), ("CNOT", 4, 1), ("CNOT", 0, 2), ("CNOT", 2, 5),
]
STEANE_SHALLOW = [
    ("H", 3, -1), ("CNOT", 3, 0), ("H", 1, -1), ("H", 4, -1), ("CNOT", 1, 4), ("CNOT", 4, 5),
    ("CNOT", 5, 0), ("CNOT", 3, 6), ("H", 3, -1), ("CNOT", 1, 6), ("CNOT", 1, 5), ("CNOT", 0, 2),
    ("CNOT", 0, 3), ("CNOT", 4, 2), ("CNOT", 3, 1), ("CNOT", 1, 2),
]


def _fault_key(fs):
    return sorted(zip((bytes(r) for r in fs.paulis), np.round(fs.probs, 12)))


def test_fault_set_propagates_through_suffix():
    # X on control before a CNOT becomes XX on the output.
    gates = [("CNOT", 0, 1)]
    c = stim.Circuit("X_ERROR(0.1) 0\nCX 0 1")
    paulis, probs = fault_set(c, 2)
    assert probs.tolist() == [0.1]
    assert paulis[0].tolist() == [1, 1, 0, 0]
    noisy = depolarizing_noise_model(1e-3, 1e-2)(gates_to_stim_circuit(gates, 2))
    paulis, probs = fault_set(noisy, 2)
    assert paulis.shape == (15, 4)
    assert np.isclose(probs.sum(), 1e-2)


def test_layered_circuit_is_asap_and_pads_qubits():
    gates = [("H", 0, -1), ("H", 1, -1), ("CNOT", 0, 1), ("H", 2, -1), ("CNOT", 2, 3)]
    c = gates_to_layered_circuit(gates, 6)
    assert c.num_qubits == 6
    layers = str(c).split("TICK")
    assert len(layers) == 2 == circuit_depth(gates, 6)
    assert layers[0].endswith("H 0 1 2\n") and layers[1].strip() == "CX 0 1 2 3"
    # same unitary as the flat circuit
    assert stim.Tableau.from_circuit(c) == stim.Tableau.from_circuit(gates_to_stim_circuit(gates, 6))


def test_circuit_level_reward_bounds_and_perfect_case():
    # Noise only on the last gate of an encoding circuit for the [[5,1,3]]
    # code produces weight <= 2 faults on the code state, all detectable.
    code = known_codes.get("five_qubit")
    tab = code.to_tableau()
    gates = gates_from_stim(code.to_stim_tableau().to_circuit("elimination"))

    def last_gate_noise(circuit):
        out = stim.Circuit()
        insts = list(circuit.flattened())
        gate_idx = [i for i, ins in enumerate(insts) if ins.name not in ("I", "TICK", "QUBIT_COORDS")]
        for i, ins in enumerate(insts):
            out.append(ins)
            if i == gate_idx[-1]:
                tg = [t.value for t in ins.targets_copy()]
                if ins.name in ("CX", "CZ", "SWAP"):
                    out.append("DEPOLARIZE2", tg, 0.01)
                else:
                    out.append("DEPOLARIZE1", sorted(set(tg)), 0.01)
        return out

    r = circuit_level_reward(tab, 1, [gates], last_gate_noise)
    assert r[0] == 0.0
    r_full = circuit_level_reward(tab, 1, [gates], depolarizing_noise_model())
    assert -1.0 <= r_full[0] <= 0.0
    r_board = circuit_level_reward(tab, 1, [gates], presets.board())
    assert -1.0 <= r_board[0] < 0.0


def test_board_preset_equals_placeholder_at_matched_rates():
    # The board recipe has no single-qubit-gate channel, DEPOLARIZE2(p) on
    # every two-qubit gate, and DEPOLARIZE1(p) on every idle data qubit per
    # layer; the placeholder with p1=None, p2=p, idle=p is the same model.
    code, gates = _steane_encoder()
    sk = gates_to_layered_circuit(gates, 7)
    board = resolve_noise_model(presets.board(), n=7)(sk)
    placeholder = depolarizing_noise_model(None, 1e-3, idle=1e-3)(sk)
    fb, fp = fault_set(board, 7), fault_set(placeholder, 7)
    assert len(fb) == len(fp) > 0
    assert _fault_key(fb) == _fault_key(fp)
    tab = code.to_tableau()
    r1 = circuit_level_reward(tab, 1, [gates], presets.board())
    r2 = circuit_level_reward(tab, 1, [gates], depolarizing_noise_model(None, 1e-3, idle=1e-3))
    assert np.isclose(r1[0], r2[0])
    # by name and by scale
    r3 = circuit_level_reward(tab, 1, [gates], "board-rfc0001", scale=0.5, normalize=False)
    assert np.isclose(r3[0], 0.5 * circuit_level_reward(tab, 1, [gates], presets.board(), normalize=False)[0])


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_direct_enumeration_matches_stim_dem(basis):
    # Undetected mass from the direct mechanism enumeration against stim's
    # detector error model of the circuit followed by an ideal stabilizer
    # and logical readout; stim merges mechanisms by signature (XOR of
    # probabilities), so agreement is to O(p^2).
    code, gates = _steane_encoder()
    tab = code.to_tableau()
    sk = gates_to_layered_circuit(gates, 7)
    for prof in (presets.chain(), presets.forte(), presets.heron(graph=None)):
        noisy = resolve_noise_model(prof, n=7)(sk)
        fs = fault_set(noisy, 7)
        d = dem_undetected_mass(noisy, tab, 1, basis)
        dd = direct_undetected_mass(fs, tab, 1, basis)
        assert dd > 0
        assert abs(d - dd) / dd < 1e-2


def test_correlated_and_heralded_mechanisms():
    code, gates = _steane_encoder()
    sk = gates_to_layered_circuit(gates, 7)
    corr = presets.forte().with_params(
        correlated=nm.Correlated(gate_pairs=(("ZZ", 1e-3), ("XX", 5e-4)), exclusive=True)
    )
    fs = fault_set(resolve_noise_model(corr, n=7)(sk), 7)
    n_cnot = sum(g[0] == "CNOT" for g in gates)
    # each CNOT adds two chained terms; the noise module chooses the
    # conditional probability of the ELSE term so that its absolute
    # probability is the given rate
    base = fault_set(resolve_noise_model(presets.forte(), n=7)(sk), 7)
    assert len(fs) == len(base) + 2 * n_cnot
    assert np.isclose(fs.total - base.total, n_cnot * (1e-3 + 5e-4))
    assert np.isclose(np.sort(fs.probs)[np.searchsorted(np.sort(fs.probs), 5e-4)], 5e-4)
    leak = presets.chain().with_params(leakage=nm.Leakage(per_2q=1e-3, mode="heralded"))
    fs = fault_set(resolve_noise_model(leak, n=7)(sk), 7)
    assert fs.heralded.sum() == 3 * 2 * n_cnot
    tab = code.to_tableau()
    # heralded mechanisms never count as undetected
    from autoqec_sim.circuit_noise import undetected_fault_mass

    unher = fault_set(resolve_noise_model(presets.chain(), n=7)(sk), 7)
    assert np.isclose(undetected_fault_mass(tab, 1, fs), undetected_fault_mass(tab, 1, unher))


def test_ion_sequential_timing_penalizes_the_deeper_encoder():
    # Two Steane encoders. Under the Forte preset with parallel timing the
    # deeper one (14 CNOTs, 11 layers) has the smaller undetected fault
    # mass; under sequential timing every CNOT is its own slot with idle
    # noise on the other five qubits, and the ranking flips.
    code = known_codes.get("steane")
    for g in (STEANE_DEEP, STEANE_SHALLOW):
        c = replay(g, 7)
        assert c.distance_exact() == 3 and c.same_weight_enumerators(code)
    assert circuit_depth(STEANE_DEEP, 7) > circuit_depth(STEANE_SHALLOW, 7)
    tabs = np.concatenate([replay(STEANE_DEEP, 7).to_tableau(), replay(STEANE_SHALLOW, 7).to_tableau()])
    gl = [STEANE_DEEP, STEANE_SHALLOW]
    par = -circuit_level_reward(tabs, 1, gl, presets.forte("parallel"), normalize=False)
    seq = -circuit_level_reward(tabs, 1, gl, presets.forte("sequential"), normalize=False)
    assert par[0] < par[1]  # deep encoder better when gates run in parallel
    assert seq[0] > seq[1]  # and worse once every two-qubit gate is a slot
    # the extra undetected mass from serialization grows with the slot count
    assert seq[0] - par[0] > seq[1] - par[1] > 0
    # the chain preset (Table V per-slot idle, ten times smaller) shows the
    # same monotone increase, without flipping this pair
    cpar = -circuit_level_reward(tabs, 1, gl, presets.chain("parallel"), normalize=False)
    cseq = -circuit_level_reward(tabs, 1, gl, presets.chain("sequential"), normalize=False)
    assert cseq[0] - cpar[0] > cseq[1] - cpar[1] > 0


def test_heavy_hex_profile_rejects_non_edge():
    hh = graphs.heavy_hex(1, 1)  # 12-cycle, degree 2
    assert hh.n == 12
    a, b = hh.edges[0]
    non_edge = next((x, y) for x in range(hh.n) for y in range(x + 1, hh.n) if not hh.has_edge(x, y))
    tab_ok = replay([("H", a, -1), ("CNOT", a, b)], hh.n).to_tableau()
    prof = presets.heron(graph=hh.to_topology())
    r = circuit_level_reward(tab_ok, 1, [[("H", a, -1), ("CNOT", a, b)]], prof)
    assert -1.0 <= r[0] <= 0.0
    bad = [("H", non_edge[0], -1), ("CNOT", non_edge[0], non_edge[1])]
    tab_bad = replay(bad, hh.n).to_tableau()
    with pytest.raises(ConnectivityError):
        circuit_level_reward(tab_bad, 1, [bad], prof)
    r = circuit_level_reward(tab_bad, 1, [bad], prof, on_violation="penalize")
    assert r[0] == -1.0
    # a profile without a graph takes the environment's graph
    with pytest.raises(ConnectivityError):
        circuit_level_reward(tab_bad, 1, [bad], presets.heron(graph=None), graph=hh)
    # and a profile whose graph differs from the environment's is refused
    with pytest.raises(ValueError):
        resolve_noise_model(presets.heron(graph=nm.heavy_hex(2, 2)), graph=hh)


def test_env_graph_and_profile_graph_are_one_object():
    env = CodeDiscoveryEnv(n=9, k=1, graph=nm.square(3, 3), batch_size=4, gates=("H", "CNOT"), max_steps=6, seed=1)
    assert env.graph == graphs.square(3, 3)
    assert env.topology == nm.square(3, 3)
    env.reset()
    for _ in range(6):
        env.step(env.sample_actions())
    # every executed two-qubit gate is an edge of the topology, so the
    # profile bound to the environment never raises
    f = make_circuit_level_reward(env, presets.heron(graph=None), normalize=False)
    r = f(env.tab, 1)
    assert r.shape == (4,) and np.all(r <= 0)
    prof = resolve_noise_model(presets.heron(graph=None), graph=env.graph, n=9)
    assert prof.keywords["profile"].graph == env.topology
