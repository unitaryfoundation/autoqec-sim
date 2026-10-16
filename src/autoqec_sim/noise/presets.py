"""Named parameter sets. Every number is sourced or labelled ASSUMPTION
(see README.md, profiles table). Rates are at scale 1.

  forte-2308.05071   171Yb+ IonQ Forte (SPEC.md table 1)
  chain-2606.06455   133Ba+ 40-ion chain, arXiv:2606.06455 (SPEC.md table 2)
  board-rfc0001      qldpc-challenge canonical recipe, uniform p = 1e-3
  heron-r2           IBM Heron r2 superconducting default (heavy-hex)
"""

from __future__ import annotations

from .profile import Correlated, Idle, Leakage, Params, Profile
from .timing import PARALLEL, Timing
from .topology import Graph, complete, heavy_hex

# ------------------------------------------------------------------ trapped ion
# Constants copied from ion_noise.py; the equivalence tests pin them.
FORTE = Params(
    p2=4.0e-3,            # Forte page "0.4%"; arXiv:2308.05071 Sec. 3.1 median 46.4e-4
    p1=2.0e-4,            # Forte page "0.02%"; paper Sec. 3.1 median 2.0e-4
    p_reset=1.0e-3,       # ASSUMPTION: split of the 0.5% SPAM (SPEC.md)
    p_meas=4.0e-3,        # ASSUMPTION: same split
    idle=Idle(kind="duration", T1=10.0, T2=1.0, scope="all"),  # Forte page T2 ~1 s, T1 10-100 s
    t_2q=900e-6,          # paper Sec. 2 "about 900 us"
    t_1q=110e-6,          # paper Sec. 2
    t_meas=200e-6,        # ASSUMPTION (SPEC.md)
    t_reset=20e-6,        # ASSUMPTION (SPEC.md)
    notes="SPEC.md table 'forte-2308.05071'",
)

CHAIN = Params(
    p2=8.5e-3,            # arXiv:2606.06455 Fig. 7 reference line; ASSUMPTION +-1e-3
    p1=3.2e-4,            # Table V
    p_reset=3.2e-4,       # ASSUMPTION: not tabulated; single-qubit-op rate
    p_meas=9.0e-3,        # Table V measurement bit-flip
    idle=Idle(kind="slot",
              gate=(5.0e-6, 5.0e-6, 4.0e-5),   # Table V idle per gate slot; X/Y split ASSUMPTION
              meas=(5.0e-5, 5.0e-5, 4.0e-4),   # Table V idle per MCM round
              scope="all"),
    leakage=Leakage(per_meas_layer=1.7e-3, mode="pauli"),  # Table V; proxy of Appendix E
    t_2q=460e-6,          # DERIVED (SPEC.md)
    t_1q=460e-6,
    t_meas=9.7e-3,        # DERIVED (SPEC.md)
    t_reset=460e-6,       # ASSUMPTION
    notes="SPEC.md table 'chain-2606.06455'",
)

# --------------------------------------------------------------- board recipe
BOARD_P_REF = 1e-3        # qldpc-challenge verify/circuit_tools.P_REF

BOARD = Params(
    p2=BOARD_P_REF, p1=None, p_reset=BOARD_P_REF, p_meas=BOARD_P_REF,
    idle=Idle(kind="slot", gate=BOARD_P_REF, meas=BOARD_P_REF, scope="data"),
    notes="qldpc-challenge RFC 0001: one rate on every gate, reset, measurement "
          "and idle data qubit per layer; no single-qubit-gate channel",
)

# ------------------------------------------------------------ superconducting
# IBM Heron r2 (156 fixed-frequency transmons, tunable couplers, heavy-hex).
# Sources accessed 2026-09-20:
#   [H1] Harper et al., "Characterising the failure mechanisms of
#        error-corrected quantum logic gates", arXiv:2504.07258 (Nat. Commun.
#        2026), Sec. IV.1.2: two-qubit median error "approximately 0.35%" on
#        Heron-class devices; T1/T2 "on the order of 213/120 us on the best
#        Heron device"; measurement duration "between 3000 and 4000 ns".
#        Table 1 (fitted circuit-level model): single-qubit depolarizing
#        0.02%, two-qubit 0.41%, quantum measurement 1.2%, classical
#        measurement 4.2%, reset 7.5%, idling 1.2%.
#   [H2] Vezvaee et al., "Surface code scaling on heavy-hex superconducting
#        quantum processors", arXiv:2510.18847, appendix: single-qubit X
#        pulse 32 ns = 8 dt on ibm_aachen; native gate set CZ, RZ, SX, X,
#        reset (measurement followed by conditional X).
#   [H3] IBM Quantum Platform device table for ibm_fez (Heron r2), as
#        quoted in secondary sources on 2026-09-20: median two-qubit error
#        2.77e-3, median readout assignment error 1.55e-2. CZ duration
#        68 ns is quoted for Heron r2 in the same secondary sources
#        (attributed to arXiv:2410.00916); not verified against the
#        primary.
HERON = Params(
    p2=3.5e-3,            # [H1] Sec. IV.1.2 median ~0.35% (ibm_fez page: 2.77e-3, [H3])
    p1=2.0e-4,            # [H1] Table 1 fitted single-qubit depolarizing 0.02%
    p_reset=1.5e-2,       # ASSUMPTION: IBM reset is measure + conditional X [H2],
                          # so its flip rate is taken equal to readout error
    p_meas=1.5e-2,        # [H3] ibm_fez median readout assignment error 1.55e-2
    bias1=0.5, bias2=0.5, # ASSUMPTION: no public Pauli decomposition of CZ/SX
    idle=Idle(kind="duration", T1=213e-6, T2=120e-6, scope="all"),  # [H1]
    leakage=Leakage(per_meas_layer=0.0, per_2q=0.0),  # not sourced: CZ leakage
                          # on Heron is unpublished; left at 0 and stated
    correlated=Correlated(crosstalk=0.0, crosstalk_pauli="ZZ"),  # ZZ crosstalk
                          # is named in [H2] as the dominant coherent term,
                          # with no rate; set crosstalk per run
    t_2q=68e-9,           # [H3] CZ 68 ns (secondary)
    t_1q=32e-9,           # [H2] 32 ns pulse
    t_meas=3.5e-6,        # [H1] 3000 to 4000 ns; midpoint, ASSUMPTION
    t_reset=3.5e-6 + 32e-9,   # measure + X [H2]; ASSUMPTION on timing
    notes="IBM Heron r2 default; see README profiles table",
)

# Heavy-hex graph large enough for the [[9,1,3]] one-round circuits with
# any layout; the size is a default, not a device.
HERON_GRAPH = heavy_hex(4, 4)


def forte(timing="parallel", graph: Graph | None = None) -> Profile:
    """IonQ Forte profile (arXiv:2308.05071; ``SPEC.md``).

    Parameters
    ----------
    timing
        Timing model name (``"parallel"`` or ``"sequential"``).
    graph
        Connectivity graph; ``None`` for all-to-all.
    """
    return Profile("forte-2308.05071", FORTE, Timing.from_name(timing), graph,
                   source="IonQ Forte, arXiv:2308.05071; SPEC.md")


def chain(timing="sequential", graph: Graph | None = None) -> Profile:
    """The chain set (arXiv:2606.06455, Table V; ``SPEC.md``). Its device runs
    gates strictly sequentially, so the default timing is sequential.

    Parameters
    ----------
    timing
        Timing model name.
    graph
        Connectivity graph; ``None`` for all-to-all.
    """
    return Profile("chain-2606.06455", CHAIN, Timing.from_name(timing), graph,
                   source="arXiv:2606.06455 Table V; SPEC.md")


def board(p: float = BOARD_P_REF) -> Profile:
    """The qldpc-challenge canonical recipe.

    Parameters
    ----------
    p
        Error rate of the recipe (the profile's scale multiplies it).

    Notes
    -----
    :func:`autoqec_sim.noise.apply.apply` needs ``n_data`` for this profile,
    since its idle channel acts on data qubits only.
    """
    params = BOARD if p == BOARD_P_REF else BOARD.replace(
        p2=p, p_reset=p, p_meas=p, idle=Idle(kind="slot", gate=p, meas=p, scope="data"))
    return Profile("board-rfc0001", params, PARALLEL, None,
                   source="qldpc-challenge verify/circuit_tools.apply_noise")


def heron(timing="parallel", graph: Graph | None = HERON_GRAPH) -> Profile:
    """IBM Heron r2 profile (arXiv:2504.07258, arXiv:2510.18847, and the
    ``ibm_fez`` calibration table).

    Parameters
    ----------
    timing
        Timing model name.
    graph
        Connectivity graph; default a 4 x 4 heavy-hex lattice large enough for
        the one-round ``[[9,1,3]]`` circuits.
    """
    return Profile("heron-r2", HERON, Timing.from_name(timing), graph,
                   source="arXiv:2504.07258, arXiv:2510.18847, IBM ibm_fez table")


def depolarizing(p: float, timing="parallel", graph: Graph | None = None,
                 idle_scope: str = "all") -> Profile:
    """Uniform depolarizing profile: ``p`` on every gate, reset, measurement, and
    idle qubit per layer, including single-qubit gates (unlike the board
    recipe).

    Parameters
    ----------
    p
        Error probability of every channel.
    timing
        Timing model name.
    graph
        Connectivity graph; ``None`` for all-to-all.
    idle_scope
        ``"all"`` or ``"data"``: which qubits take the idle channel.
    """
    params = Params(p2=p, p1=p, p_reset=p, p_meas=p,
                    idle=Idle(kind="slot", gate=p, meas=p, scope=idle_scope),
                    notes="uniform depolarizing")
    return Profile(f"depolarizing-{p:g}", params, Timing.from_name(timing), graph)


PRESETS = {
    "forte-2308.05071": forte,
    "chain-2606.06455": chain,
    "board-rfc0001": board,
    "heron-r2": heron,
}


def preset(name: str, **kw) -> Profile:
    """Profile by preset name.

    Parameters
    ----------
    name
        One of ``PRESETS`` (``"forte-2308.05071"``, ``"chain-2606.06455"``,
        ``"board-rfc0001"``, ``"heron-r2"``).
    **kw
        Passed to the preset constructor.
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name](**kw)


# ion chain graph helper: the Forte set is all-to-all within one chain
def forte_graph(n_ions: int = 36) -> Graph:
    return complete(n_ions, name=f"ion-chain-{n_ions}")
