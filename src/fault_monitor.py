"""
fault_monitor.py
Threshold-based fault detection and isolation using EKF estimates.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FaultEvent:
    step: int
    motor_id: int
    estimated_severity: float
    confidence: float


class FaultMonitor:
    """
    Monitors per-motor kf_factor estimates from EKF.
    Triggers fault events when estimates drop below threshold.

    Uses a sliding window with CUSUM-style detection to reduce false alarms.
    """

    def __init__(
        self,
        fault_threshold: float = 0.85,    # kf_factor below this = fault
        window_size: int = 20,            # steps to confirm fault
        confidence_min: float = 0.9,      # fraction of window steps below threshold
    ):
        self.fault_threshold = fault_threshold
        self.window_size = window_size
        self.confidence_min = confidence_min

        self._histories: List[deque] = [
            deque(maxlen=window_size) for _ in range(4)
        ]
        self._fault_events: List[FaultEvent] = []
        self._active_faults: np.ndarray = np.zeros(4, dtype=bool)
        self._step = 0

    def update(self, kf_factors: np.ndarray, step: int) -> np.ndarray:
        """
        Update monitor with latest EKF fault parameter estimates.

        Returns:
            fault_flags (np.ndarray, shape [4], bool): True if motor i is faulted.
        """
        self._step = step
        fault_flags = np.zeros(4, dtype=bool)

        for i, kf in enumerate(kf_factors):
            self._histories[i].append(kf)

            if len(self._histories[i]) < self.window_size:
                continue  # not enough data yet

            below_thresh = np.sum(
                np.array(self._histories[i]) < self.fault_threshold
            )
            confidence = below_thresh / self.window_size

            if confidence >= self.confidence_min:
                fault_flags[i] = True

                if not self._active_faults[i]:
                    # New fault event
                    severity = 1.0 - float(np.mean(self._histories[i]))
                    event = FaultEvent(
                        step=step,
                        motor_id=i,
                        estimated_severity=severity,
                        confidence=confidence,
                    )
                    self._fault_events.append(event)
                    self._active_faults[i] = True
                    print(
                        f"[FaultMonitor] FAULT DETECTED — Motor {i} | "
                        f"Step {step} | Severity ≈ {severity:.2f} | "
                        f"Confidence {confidence:.0%}"
                    )
            else:
                if self._active_faults[i] and confidence < 0.5:
                    # Recovery
                    self._active_faults[i] = False
                    print(f"[FaultMonitor] Recovery detected — Motor {i}")

        return fault_flags

    def any_fault(self) -> bool:
        return bool(np.any(self._active_faults))

    def get_active_faults(self) -> np.ndarray:
        return self._active_faults.copy()

    def get_fault_events(self) -> List[FaultEvent]:
        return list(self._fault_events)

    def reset(self):
        for h in self._histories:
            h.clear()
        self._fault_events.clear()
        self._active_faults[:] = False
        self._step = 0
