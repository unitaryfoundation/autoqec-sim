"""Helpers shared by the top-level tests."""

def gates_from_stim(circ):
    """``(gate, q1, q2)`` list of a stim circuit in the simulator's gate set
    (``SQRT_X`` expanded as ``H S H``)."""
    gates = []
    for inst in circ.flattened():
        tg = [t.value for t in inst.targets_copy()]
        if inst.name in ("H", "S", "S_DAG", "X", "Y", "Z"):
            gates += [(inst.name, q, -1) for q in tg]
        elif inst.name == "CX":
            gates += [("CNOT", tg[i], tg[i + 1]) for i in range(0, len(tg), 2)]
        elif inst.name in ("CZ", "SWAP"):
            gates += [(inst.name, tg[i], tg[i + 1]) for i in range(0, len(tg), 2)]
        elif inst.name == "SQRT_X":
            gates += [(g, q, -1) for q in tg for g in ("H", "S", "H")]
        elif inst.name == "SQRT_X_DAG":
            gates += [(g, q, -1) for q in tg for g in ("H", "S_DAG", "H")]
    return gates

