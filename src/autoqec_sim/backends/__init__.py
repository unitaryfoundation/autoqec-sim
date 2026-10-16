# Copyright 2026 Unitary Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Alternative array backends for the hot paths of the reward engine.

The package default is numpy (``autoqec_sim.tableau``, ``autoqec_sim.reward``).
``autoqec_sim.backends.torch_backend`` mirrors the gate kernels, the
symplectic product, and the parity masks on torch tensors on any device;
it is opt-in and imports torch only when imported itself.
"""
