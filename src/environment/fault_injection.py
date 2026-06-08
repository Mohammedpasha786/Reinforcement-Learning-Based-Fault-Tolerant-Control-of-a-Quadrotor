"""
fault_injection.py
Motor fault models for quadrotor simulation.

Fault representation: equivalent resistance increase in motor winding,
which reduces the effective thrust coefficient (kf) for that motor.
Reference: Bhan et al., SysTol 2021.
"""

import numpy as np
from typing import Dict, Optional


FAULT_TYPES = {
    "motor_resistance": "Increased winding resistance → reduced kf",
    "propeller_damage": "Reduced blade efficiency → partial kf loss",
    "complete_failure":  "Total motor loss → kf = 0",
}


class FaultInjector:
    """
    Injects fault effects into motor thrust coefficients.

    Args:
        fault_type:  One of FAULT_TYPES keys, or None (nominal)
        motor_id:    Which motor (0-3) is affected
        severity:    0.0 = nominal, 1.0 = complete failure
    """

    def __init__(
        self,
        fault_type: Optional[str],
        motor_id: int = 0,
        severity: float = 0.0,
        gradual: bool = False,
        ramp_steps: int = 100,
    ):
        if fault_type is not None and fault_type not in FAULT_TYPES:
            raise ValueError(f"Unknown fault_type '{fault_type}'. Choose from: {list(FAULT_TYPES)}")

        self.fault_type = fault_type
        self.motor_id = motor_id
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.gradual = gradual
        self.ramp_steps = ramp_steps

        self._fault_params = np.zeros(4)  # one param per motor
        self._fault_step_counter = 0

    def reset(self):
        self._fault_params = np.zeros(4)
        self._fault_step_counter = 0

    def apply(
        self, state: Dict[str, np.ndarray], step: int, active: bool
    ) -> np.ndarray:
        """
        Returns fault_params vector (shape [4]) representing per-motor
        effective kf degradation factor in [0, 1].
        1.0 = fully nominal, 0.0 = complete failure.
        """
        params = np.ones(4)  # nominal: all motors at full thrust

        if not active or self.fault_type is None:
            self._fault_params = params
            return params

        self._fault_step_counter += 1

        # Compute effective severity (allow gradual ramp)
        if self.gradual:
            ramp_factor = min(self._fault_step_counter / self.ramp_steps, 1.0)
            eff_severity = self.severity * ramp_factor
        else:
            eff_severity = self.severity

        # Apply fault model
        if self.fault_type == "motor_resistance":
            # Equivalent resistance increase: thrust_factor = 1 / (1 + R_extra)
            # severity=1.0 → R_extra → ∞ → thrust_factor → 0
            # We model: thrust_factor = 1 - severity (linear approximation)
            thrust_factor = 1.0 - eff_severity
            params[self.motor_id] = np.clip(thrust_factor, 0.0, 1.0)

        elif self.fault_type == "propeller_damage":
            # Blade damage reduces kf quadratically with severity
            thrust_factor = (1.0 - eff_severity) ** 2
            params[self.motor_id] = np.clip(thrust_factor, 0.0, 1.0)

        elif self.fault_type == "complete_failure":
            params[self.motor_id] = 0.0

        self._fault_params = params
        return params.copy()

    @property
    def current_fault_params(self) -> np.ndarray:
        return self._fault_params.copy()

    def inject_noise(self, params: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """Add measurement noise to fault parameters (for realism)."""
        return params + np.random.normal(0, noise_std, size=params.shape)
