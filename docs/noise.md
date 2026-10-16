# Interface

```python
import stim
from autoqec_sim import noise as nm
from autoqec_sim.noise import presets

noisy = nm.apply(skeleton, presets.forte(), scale=0.25)           # ion, parallel
noisy = nm.apply(skeleton, presets.chain(timing="sequential"), 1.0)
noisy = nm.apply(skeleton, presets.board(), 1.0, n_data=n)         # board recipe
noisy = nm.apply(skeleton, presets.heron(graph=nm.heavy_hex(4, 4)), 1.0)

prof = presets.heron().with_bias(10.0)                            # eta = pZ/(pX+pY)
prof = prof.with_timing(nm.partial(4))                            # 4 two-qubit gates per slot
prof = prof.with_params(correlated=nm.Correlated(crosstalk=1e-4))
f = nm.noisifier(prof, scale=0.5)                                 # circuit -> noisy circuit
dem = nm.dem(noisy)                                               # board options; approximate only if stim requires
```

`apply(circuit, profile, scale, *, n_data=None, idle_qubits=None,
check_graph=True)`. Every probability of the profile is multiplied by
`scale`; durations and coherence times are not. The input must be
noiseless (`nm.strip_noise` first for a committed board circuit). With a
graph on the profile, every two-qubit gate is checked against it and a
`ConnectivityError` listing the violations is raised; `check_connectivity`
returns the list without raising.

`Profile(name, params, timing, graph)` bundles:

| Field | Content |
|---|---|
| `Params.p2`, `bias2`, `pauli2` | two-qubit gate error, its bias, or an explicit 15-vector in stim `PAULI_CHANNEL_2` order |
| `Params.p1`, `bias1` | single-qubit unitary error and bias |
| `Params.p_reset`, `p_meas` | X_ERROR after R and Z_ERROR after RX; flip probability on M and MX |
| `Params.idle` | `Idle(kind="duration", T1, T2)` or `Idle(kind="slot", gate, meas)` or `kind="none"`; `scope` "all" or "data"; optional `bias` for scalar slot rates |
| `Params.leakage` | `Leakage(per_meas_layer, per_2q, mode="pauli"/"heralded", herald_detectors)` |
| `Params.correlated` | `Correlated(gate_pairs=((pauli2, rate), ...), exclusive, crosstalk, crosstalk_pauli)` |
| `Params.t_2q`, `t_1q`, `t_meas`, `t_reset` | layer durations by instruction class, seconds |
| `Profile.timing` | `PARALLEL`, `SEQUENTIAL`, or `partial(k)` (`Timing(gates_per_slot=k)`) |
| `Profile.graph` | `Graph` from `square`, `triangular`, `heavy_hex`, `complete` (ion chain), `path`, or None for unconstrained |

A rate set to `None` means no channel of that class (the board recipe has
no single-qubit-gate channel); `0.0` emits the channel with probability 0,
so scale sweeps keep the instruction structure (as `ion_noise` does).

### Placement per TICK segment

After the timing rewrite (`reslot`: the non-two-qubit instructions of a
layer stay together in a first slot, then the two-qubit gates `k` per
slot; `k=1` equals `ion_noise.serialize_two_qubit`), each segment emits
its instructions unchanged, each followed by its channel: two-qubit gate,
then `DEPOLARIZE2(p2)` or `PAULI_CHANNEL_2`, then the `gate_pairs` terms,
then crosstalk terms, then per-gate leakage; single-qubit unitary, then
`DEPOLARIZE1(p1)`, `PAULI_CHANNEL_1`, or `Z_ERROR` (infinite bias); reset,
then its flip; measurement with `p_meas` as the gate argument. If the
segment did anything, one idle channel goes on every in-scope qubit not
operated in it, and, if the segment measures, the per-measurement-layer
leakage term on the same qubits. The idle duration is the slowest
instruction class present in the layer.

### Channels

Single-qubit bias: `px = py = p / (2 (1 + eta))`, `pz = p eta / (1 + eta)`;
`eta = 1/2` is depolarizing and is emitted as `DEPOLARIZE1(p)`.

Two-qubit bias (product-weight construction): weights `w(I) = w(X) = w(Y)
= 1`, `w(Z) = 2 eta`, probability of the pair `P1 P2` proportional to
`w(P1) w(P2)` over the 15 non-identity pairs, total `p`. Each single-qubit
marginal has bias exactly `eta`; `eta = 1/2` is the uniform depolarizing
channel and is emitted as `DEPOLARIZE2(p)`. At large `eta` the mass sits
on ZZ (weight `4 eta^2`), then equally on the six pairs with one Z factor
(`2 eta`), then the eight Z-free pairs. This is a modelling choice with no
measured decomposition behind it (no IonQ or IBM Pauli-twirled two-qubit
decomposition is public); a measured 15-vector goes in `Params.pauli2`.

Correlated: `gate_pairs` are `CORRELATED_ERROR` lines after every two-qubit
gate on its two targets, independent by default; `exclusive=True` chains
them with `ELSE_CORRELATED_ERROR` using conditional probabilities
`q_i = p_i / (1 - sum_{j<i} p_j)`. Crosstalk: for every two-qubit gate
`(a, b)` and every graph neighbour `c` of `a` or `b` outside the gate, one
`CORRELATED_ERROR(rate)` with `crosstalk_pauli[0]` on the gate qubit and
`crosstalk_pauli[1]` on `c` (default ZZ, a stochastic stand-in for
always-on ZZ coupling or spectator drive). Spectators that are themselves
in another gate of the layer are included.

Leakage: `mode="pauli"` is the erasure-to-Pauli proxy, `DEPOLARIZE1(3/4
rate)` (the qubit replaced by the maximally mixed state with probability
`rate`, the simulation proxy of arXiv:2606.06455 Appendix E).
`mode="heralded"` is `HERALDED_ERASE(rate)`, stim's erasure with a flag bit
in the measurement record, plus a `DETECTOR` per flag when
`herald_detectors` (default), so the DEM carries the herald; the
skeleton's `rec[-k]` references are rewritten around the extra records
and `nm.strip_noise` removes the heralds again. Both are approximations:
stim has no leaked state, and a leaked qubit is neither erased nor
depolarized (its later gates apply non-Pauli errors on neighbours).

### DEM exactness

The board's pinned derivation (`derive_dem`, `approximate_disjoint_errors`
False) converts `DEPOLARIZE1/2`, `X_ERROR`, `Z_ERROR`, `PAULI_CHANNEL_1`
with three positive components, and independent `CORRELATED_ERROR` terms
exactly. It refuses `PAULI_CHANNEL_2` (any two-qubit bias other than 1/2,
or `pauli2`), `ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`, and a
`PAULI_CHANNEL_1` with a zero component. `Profile.dem_exact` says which
case a profile is in; `nm.dem(circuit)` tries the exact conversion and
falls back to `approximate_disjoint_errors=True` (error O(p^2) in the
merged probabilities, mechanism set unchanged), or raises with
`approximate=False`. Board conformance (`noise_recipe_errors`) accepts
only the board profile's output.

## Topologies

| Builder | Graph | Notes |
|---|---|---|
| `heavy_hex(rows, cols)` | honeycomb of rows x cols hexagons with a qubit on every edge; max degree 3 | IBM Eagle and Heron style. rows + 1 lines of 4 cols + 1 qubits plus connectors; 12 qubits per hexagon. IBM trims the outer lines differently (Eagle 127, Heron r2 156), so sizes match approximately |
| `square(rows, cols)` | degree 4 | Google Sycamore and Willow style |
| `triangular(rows, cols)` | square plus one diagonal per cell, degree 6 | |
| `complete(n)` (`ion_chain`) | all-to-all | one trapped-ion chain; `presets.forte_graph(36)` |
| `path(n)` | nearest-neighbour line | |

`check_connectivity(circuit, graph, layout=None)` returns one `Violation`
(layer, gate, qubits, reason) per two-qubit gate whose mapped pair is not
an edge or touches a node outside the graph; `apply` raises
`ConnectivityError` with the list. The [[9,1,3]] one-round circuits have
24 violations on any heavy-hex graph with the identity layout (the
circuit uses one ancilla per check with all-to-all placement); a layout
mapping is the caller's job.

## Use in the circuit-level reward

`circuit_noise.resolve_noise_model` accepts a `autoqec_sim.noise.Profile`
(`presets.forte()`, `presets.chain()`, `presets.board()`,
`presets.heron()`, `presets.depolarizing(p)`, or any `with_bias`,
`with_timing`, `with_params` variant), a preset name, or a bare
`stim.Circuit -> stim.Circuit` callable. The encoding gate list is laid out
as a TICK-layered skeleton (as-soon-as-possible scheduling) so the
profile's timing model and idle channel apply; every noise instruction the
profile emits becomes a fault mechanism, propagated through the rest of
the circuit (a GF(2) product against the symplectic map of the circuit
suffix, which `fault_set_reference` reproduces with stim's
`PauliString.after`) and scored with the Knill-Laflamme kernel as
`-(undetected mass)`, normalized by the total mass. A profile without a
graph takes the environment's graph; a profile whose graph differs is
refused, and a two-qubit gate off the graph raises `ConnectivityError` (or
scores `violation_reward` with `on_violation="penalize"`).
`dem_undetected_mass` cross-checks the result against stim's detector error
model to `O(p^2)`. Use through `make_circuit_level_reward(env, profile)` as
a terminal reward (`reward_mode="sparse"`); it is per circuit, not batched.
