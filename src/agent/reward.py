"""
reward.py
Reward function components for the FTC RL agent.

Design philosophy:
  - Dense shaping for smooth gradient signal
  - Bonus for fault recovery (maintaining tracking despite fault)
  - Penalty for excessive control effort
  - Terminal penalty for crash/flip
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RewardWeights:
    position_tracking: float = 1.0      # weight on position error
    attitude_tracking: float = 0.5      # weight on euler angle error
    velocity_damping:  float = 0.05     # penalize excessive velocity
    control_effort:    float = 0.01     # penalize large thrust commands
    fault_recovery:    float = 5.0      # bonus for staying within tolerance during fault
    crash_penalty:     float = -50.0    # terminal crash/flip penalty
    recovery_radius:   float = 0.5      # [m] position error to earn recovery bonus


class RewardFunction:
    """
    Computes scalar reward for each environment step.
    """

    def __init__(self, weights: RewardWeights = None):
        self.w = weights or RewardWeights()

    def __call__(
        self,
        position: np.ndarray,
        target_position: np.ndarray,
        euler: np.ndarray,
        velocity: np.ndarray,
        motor_thrusts: np.ndarray,
        fault_active: bool,
        terminated: bool,
    ) -> float:

        if terminated:
            return self.w.crash_penalty

        pos_err  = np.linalg.norm(position - target_position)
        att_err  = np.linalg.norm(euler)               # euler should be near zero
        vel_pen  = np.linalg.norm(velocity)
        ctrl_pen = np.sum(motor_thrusts**2) / (4 * 8.0**2)   # normalized

        r = (
            - self.w.position_tracking * pos_err
            - self.w.attitude_tracking * att_err
            - self.w.velocity_damping  * vel_pen
            - self.w.control_effort    * ctrl_pen
        )

        # Fault recovery bonus: reward maintaining close position during fault
        if fault_active and pos_err < self.w.recovery_radius:
            r += self.w.fault_recovery

        return float(r)

    def components(
        self,
        position: np.ndarray,
        target_position: np.ndarray,
        euler: np.ndarray,
        velocity: np.ndarray,
        motor_thrusts: np.ndarray,
        fault_active: bool,
        terminated: bool,
    ) -> dict:
        """Return individual reward components for logging/debugging."""
        if terminated:
            return {"crash": self.w.crash_penalty}

        pos_err  = np.linalg.norm(position - target_position)
        att_err  = np.linalg.norm(euler)
        vel_pen  = np.linalg.norm(velocity)
        ctrl_pen = np.sum(motor_thrusts**2) / (4 * 8.0**2)
        recovery = float(fault_active and pos_err < self.w.recovery_radius)

        return {
            "position":  -self.w.position_tracking * pos_err,
            "attitude":  -self.w.attitude_tracking * att_err,
            "velocity":  -self.w.velocity_damping  * vel_pen,
            "control":   -self.w.control_effort    * ctrl_pen,
            "recovery":   self.w.fault_recovery    * recovery,
        }

