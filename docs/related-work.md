# Related work

What exists around each part of this package, as of October 2026, from a
survey that walked the reference lists of the closest papers, the papers
citing them, and the surrounding topics. Every entry below was checked
against its arXiv or publisher record. Items with public code are marked.

## Automated code and encoder discovery

The gate-by-gate formulation this package's search layer uses, an agent
appending Clifford gates to a stabilizer tableau under a Knill-Laflamme
reward with the connectivity graph as the action space, is that of
[@olle2024], whose code, QDX, is public and reached distance-5 codes at 11
to 17 qubits and a [[20,13,3]] code; composite "gadget" actions extend it to
distance 7 [@olle2025; @yevtushenko2025]. The same simulator and agent were
turned to fault-tolerant state preparation for a fixed code in [@zen2024]
(code public). Earlier reinforcement learning for codes deformed a surface
code toward a target logical error rate [@nautrup2018] and discovered
few-qubit feedback strategies [@fosel2018]; later work searched tensor
network code geometries [@mauron2023; @su2023] and reduced stabilizer
weights toward LDPC codes [@he2025]. Variational discovery of codes and
encoders predates the tableau formulation: a Knill-Laflamme loss on
connectivity-constrained circuits [@cao2022; @xu2019], neural-network code
states [@bausch2018], autoencoders [@locher2022], state-distinguishability
losses with device runs [@meyer2025] and their extensions to concatenation
and logical operations [@meyer2026a; @meyer2026b] (code public), Riemannian
optimization over code subspaces [@casanova2024], and variational graphical
codes [@shao2024]. Equivariant policies over tableaux appear in Clifford
synthesis [@yeung2026]; the non-cumulative-objective RL of [@nagele2024]
is benchmarked on the same encoder task.

Evolutionary search for codes exists at the gate-sequence level
[@tandeitnik2022] and over binary stabilizer strings to 20 qubits
[@webster2024], and annealing reaches 50 qubits [@perlin2026]. Exact and
exhaustive methods frame the problem differently: additive codes over GF(4)
[@calderbank1998], the classification of self-dual additive codes
[@danielsen2006], every stabilizer code to 9 qubits [@cross2025], all
distance-3 codes [@yu2009], codeword-stabilized search as clique finding
[@chuang2008], and, in 2026, SAT: code design with hardware and
physics constraints, proved NP-complete and solved to 100 qubits
[@dalfavero2026] (code public), phantom codes [@koh2026], and
machine-checked distance certificates [@ehatamm2026]. Above 20 qubits the
field designs codes by program evolution with large language models
[@cruzbenito2026; @liu2026a; @qian2026] (the IBM code is public), by
multi-agent orchestration with circuit-level evaluation [@yan2026], by
Bayesian optimization with a learned logical-error predictor [@chengyu2026]
(code public), by game-theoretic search [@guerrero2025], with decoders in the
loop [@freire2025; @sella2026], with hardware calls in the reward
[@yanay2026], and by multi-agent RL that includes a syndrome-extraction agent
[@guatto2025]. Sparse codes can also be grown from seeds [@cao2025].

Where this package stands: its search layer is a reimplementation of
[@olle2024] in numpy with a torch path, at parity on the distance-3 targets
of QDX's published set and behind it on larger ones
(`experiments/qdx_targets/REPORT.md`). Its evolutionary agent is simpler
than [@webster2024]. Nothing in its discovery mode is new.

## Noise- and hardware-tailored codes

Codes tailored to biased noise are a mature line: the XY and XZZX surface
codes [@tuckett2018; @bonillaataides2021], Clifford deformations, uniform
and per-qubit [@dua2022; @tiurev2022], bias-tailored LDPC codes
[@roffe2022], and in 2026 per-qubit deformation optimized from calibration
data [@yun2026] (code public). Picking a small code from device error data
goes back to [@robertson2017] and was benchmarked against hardware in
[@finsterhoelzl2022]; distance choice from drifting calibration appears in
[@das2025], an ion-chain noise model with tailored extraction circuits in
[@ye2025], and a full-stack empirical comparison in [@swierkowska2025]. The
device presets here (Forte [@chen2024], a long ion chain [@tham2026], Heron
[@harper2026; @vezvaee2026]) and the heavy-hex lattice [@chamberland2020]
are the inputs this package evaluates codes against.

Where this package stands: its reward is the undetected single-fault mass of
the encoding circuit under such a profile, and its memory objective is a
multi-round experiment with a decoder. The nearest prior objectives are
code-capacity [@olle2024; @robertson2017; @yun2026] or circuit-level for a
fixed code [@zen2024]; [@yan2026] evaluates at circuit level inside a closed
LLM loop. The comparison in `experiments/sat_codes/REPORT.md` shows the two
levels ranking the same codes differently.

## Encoder synthesis and fault-tolerant preparation

For a fixed code, SAT-optimal fault-tolerant state preparation [@peham2024],
tableau-search encoders for arbitrary stabilizer codes [@peham2026],
noise-aware LDPC encoder resynthesis [@sodhani2026], SMT search for
fault-tolerant Clifford circuits [@shutty2022], architecture-aware
stabilizer synthesis [@winderl2023], and heuristic and optimal CNOT and
Clifford synthesis [@webster2025] (code public) all compete with
RL-found encoders on gate count, and [@webster2025] claims to win. A
benchmark for AI-assisted synthesis [@paz2026] and a FAIR circuit library
[@schmid2026] appeared in 2026.

## Syndrome-extraction schedules

Hook errors and CNOT ordering are as old as the surface code
[@dennis2001; @tomita2014]. Automated scheduling is a 2026 topic: Monte
Carlo tree search with decoder feedback [@liu2026b] (code public),
MaxSAT-guided rewrites [@viszlai2026] (code public), SMT depth-optimal
compilation [@zhang2026], RL rewarded by sampled logical error rate
[@ye2026], left-right circuits with effective-distance proxies
[@strikis2026], hook-free surface-code schedules [@hirai2026], short
Shor-style sequences [@delfosse2020], flag-style extraction for small codes
[@bhatnagar2023], detector-error-model-driven design [@derks2024], and the
result that hypergraph-product codes keep their distance under any schedule
[@manes2023]. The schedule synthesizer here is RL over the CX-layer
assignment with the exact circuit-level fault distance as the reward, which
none of these use; its one result, on the [[49,13,4]] entry of
[@qldpcchallenge], is a demonstration rather than a benchmark against them.

## Distance, verification, and simulation

Exact distance here is enumeration to 20 qubits. Public tools go further:
randomized information-set distance [@pryadko2023], parallel
Brouwer-Zimmermann [@hernando2024], and SAT, MILP, and heuristic code and
circuit distance in one package [@webster2026]; the problem is NP-hard
[@kapshikar2022], has been cast as quantum annealing [@ismail2024], and
for large LDPC codes is bounded from above by heuristic witness search
[@kasai2026]. Weight enumerators are not a complete invariant, and
local-Clifford equivalence has its own test [@danielsen2006; @vandennest2004];
the generator weight this package calls `w*` is the optimal generator
weight of [@wei2026]. Fault-counting sensitivities [@fu2026], Pauli
diagnostics [@iyer2021], and tensor-network enumerators [@pato2025] are
further evaluation tools. Stim [@gidney2021] underlies the noise and
detector-error-model work; Clifft [@chase2026] is the Unitary Foundation's
near-Clifford sampler; [@fur2026] reports hardware error-detection
experiments from the same group.

## Evaluating codes under circuit-level noise

The evaluator's proxy is a leading-order count of the minimum-weight
undetectable fault sets of the extraction circuit, decoded by maximum
likelihood over those sets alone, and its ingredients are standard.
Counting malignant pairs of fault locations bounded circuit-level failure
to leading order in the threshold proofs of [@aliferis2006];
[@fowler2013] enumerated the minimum-weight fault configurations of a
periodic surface-code circuit to give its low-rate logical error
analytically for arbitrary per-gate Pauli noise; and [@beverland2019]
showed that the number of minimum-weight failing configurations, the
entropy term, sets the low-rate logical error alongside the distance,
which is what separates codes of equal distance on the same footprint
here. Where the leading order ends, sampling takes over; the rare-event
splitting method of [@bravyi2013] reaches rates below what direct
sampling gives, and this package raises the noise scale instead of
implementing it. The reference decoders are minimum-weight matching in
its sparse-blossom implementation [@higgott2025], BP+OSD [@roffe2020;
@panteleev2021], and the Tesseract most-likely-error search [@beni2025].
Which of 39 Clifford deformations of the distance-3 surface code is best
depends on the decoder, with PyMatching against BP+OSD at Pearson 0.88
under one circuit-level noise model and 0.30 under another
[@bonnelykke2026], which is why the validation here
(`experiments/proxy_tesseract/REPORT.md`) is against a near-maximum-
likelihood decoder. Circuit noise models built from device parameters go
back to the superconducting-inspired SI1000 model of [@gidney2021b]; the
presets here are the same kind of object, assembled from public
specifications. Flag-qubit extraction with two extra qubits for
distance-3 codes [@chao2018] and with flag ancillas for arbitrary
distance [@chamberland2018] is what the one-flag option (one flag on each
generator of weight three or more) approximates. Subsystem codes are
measured through their gauge generators in the stabilizer formalism of
[@poulin2005], with the bare logical operators of [@bravyi2011]; the
Bacon-Shor code [@bacon2006] and the heavy-hexagon code [@chamberland2020]
are the two shipped. The published memory experiments the evaluator is
compared against (`experiments/hardware_comparison/REPORT.md`) are the
distance-3 surface codes of [@acharya2023; @acharya2025], the Steane code
on a trapped-ion processor [@ryananderson2021], the toric and bivariate
bicycle [@bravyi2024] codes on a 40-ion chain [@tham2026], and the
heavy-hexagon subsystem code [@sundaresan2023]; that comparison is of the
package's model on its own circuit against each paper's reported number,
not a prediction of the devices.

Where this package stands: the leading-order estimator applies the
counting of [@fowler2013] and [@beverland2019] to the extraction circuit
of an arbitrary stabilizer code through stim's detector error model
[@gidney2021], with the enumeration carried to weight-3 fault sets at
circuit distance 5 and 6; the decoders, noise models, and experiments it
is checked against are prior work.

## Reviews

AI for quantum error correction [@wang2024], reinforcement learning for
quantum technology [@bukov2026], and noise-adapted error correction
[@jayashankar2022]. No dedicated review of automated code design exists.
