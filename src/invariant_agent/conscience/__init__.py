"""Functional conscience (IBAAM 6, *Conscience Within Controls*): history-calibrated appraisal inside the invariants.

The conscience weighs what the agent *should* do inside the boundary the invariants set on what it *may* do. It has its
own contract (``IConscience``, not ``IInvariant``), runs only after the relevant invariants pass, and can never
authorise anything they block.
"""

from .calibration import CalibrationBin, CalibrationReport
from .coordinator import ConscienceCoordinator
from .history_calibrated import HistoryCalibratedConscience
from .options import ConscienceOptions, ConscienceSession
from .precedent_index import PrecedentIndex

__all__ = [
    "CalibrationBin",
    "CalibrationReport",
    "ConscienceCoordinator",
    "ConscienceOptions",
    "ConscienceSession",
    "HistoryCalibratedConscience",
    "PrecedentIndex",
]
