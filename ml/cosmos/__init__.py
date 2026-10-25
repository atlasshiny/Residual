"""Real Cosmos 3 workers (ARCHITECTURE.md sections 4.3 and 7).

Three job entry points share one file contract (``docs/cosmos-contract.md``):

* :mod:`ml.cosmos.transfer_worker` -- Job A, transfer (``Cosmos3-Nano-Sim-Transfer``)
* :mod:`ml.cosmos.inverse_worker`  -- Job B, inverse dynamics (``Cosmos3-Nano``)
* :mod:`ml.cosmos.forward_worker`  -- Job C, forward dynamics (``Cosmos3-Nano``)

:mod:`ml.cosmos.common` holds the shared contract helpers and
:mod:`ml.cosmos.omni` the vLLM-Omni HTTP client.  Each worker also runs as a
script, for example::

    python ml/cosmos/inverse_worker.py jobs/flyby_w03
"""

from __future__ import annotations

__all__ = ["common", "forward_worker", "inverse_worker", "omni", "transfer_worker"]
