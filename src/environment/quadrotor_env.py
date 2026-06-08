"""
quadrotor_env.py
Gymnasium environment for a 6-DOF quadrotor with fault injection support.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

from .dynamics import QuadrotorDynamics
from .fault_injection import FaultInjector


@dataclass
class EnvConfig:
    sim_dt: float = 0.01          # simulation timestep [s]
    episode_length: int = 1000    # steps per episode
    max_tilt_deg: float = 60.0    # max allowed tilt before termination
    target_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 2.0])
    )
    fault_type: Optional[str] = None     # e.g. 'motor_resistance'
    fault_motor_id: int = 0
    fault_severity: float = 0.0          # 0=nominal, 1=total failure
    fault_start_step: int = 200          # when fault begins


class QuadrotorFTCEnv(gym.Env):
    """
    Gymnasium environment for fault-tolerant quadrotor control.

    Observation space:
        [pos(3), vel(3), euler(3), omega(3), pid_gains(12), fault_params(4)]
    
    Action space (RL agent output):
        Delta corrections to PID gains for position and attitude loops.
        Shape: (12,) — [Kp_pos(3), Ki_pos(3), Kd_pos(3), Kp_att(3)]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        self.dynamics = QuadrotorDynamics(dt=self.cfg.sim_dt)
        self.fault_injector = FaultInjector(
            fault_type=self.cfg.fault_type,
            motor_id=self.cfg.fault_motor_id,
            severity=self.cfg.fault_severity,
        )

        # --- Observation space ---
        obs_dim = 3 + 3 + 3 + 3 + 12 + 4  # pos, vel, euler, omega, gains, fault_params
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # --- Action space: gain deltas, normalized [-1, 1] ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # PID gain bounds [Kp_pos, Ki_pos, Kd_pos, Kp_att]
        self._gain_min = np.array([0.1]*3 + [0.0]*3 + [0.0]*3 + [0.1]*3)
        self._gain_max = np.array([10.]*3 + [2.0]*3 + [5.0]*3 + [8.0]*3)
        self._gain_nominal = np.array([2.0]*3 + [0.1]*3 + [0.5]*3 + [3.0]*3)

        self._step = 0
        self._state = None
        self._pid_gains = self._gain_nominal.copy()

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._pid_gains = self._gain_nominal.copy()

        init_pos = np.array([0.0, 0.0, 0.0])
        init_vel = np.zeros(3)
        init_euler = np.zeros(3)
        init_omega = np.zeros(3)

        self._state = self.dynamics.reset(init_pos, init_vel, init_euler, init_omega)
        self.fault_injector.reset()

        obs = self._get_obs(fault_params=np.zeros(4))
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step += 1

        # 1. Apply gain delta from RL agent
        gain_range = self._gain_max - self._gain_min
        delta = action * gain_range * 0.05   # max ±5% of range per step
        self._pid_gains = np.clip(
            self._pid_gains + delta, self._gain_min, self._gain_max
        )

        # 2. Apply fault model to motor thrust coefficients
        fault_active = (
            self.cfg.fault_type is not None
            and self._step >= self.cfg.fault_start_step
        )
        fault_params = self.fault_injector.apply(
            self._state, step=self._step, active=fault_active
        )

        # 3. Compute PID control output
        thrust_cmds = self._compute_pid(self._state, self._pid_gains)

        # 4. Step dynamics
        self._state = self.dynamics.step(thrust_cmds)

        # 5. Compute reward
        reward = self._compute_reward(self._state, fault_active)

        # 6. Termination conditions
        pos = self._state["position"]
        euler = self._state["euler"]
        terminated = (
            np.any(np.abs(euler[:2]) > np.deg2rad(self.cfg.max_tilt_deg))
            or pos[2] < 0.0
        )
        truncated = self._step >= self.cfg.episode_length

        obs = self._get_obs(fault_params)
        info = {
            "step": self._step,
            "fault_active": fault_active,
            "fault_params": fault_params,
            "pid_gains": self._pid_gains.copy(),
            "position_error": np.linalg.norm(pos - self.cfg.target_position),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self, fault_params: np.ndarray) -> np.ndarray:
        s = self._state
        obs = np.concatenate([
            s["position"],
            s["velocity"],
            s["euler"],
            s["omega"],
            self._pid_gains,
            fault_params,
        ]).astype(np.float32)
        return obs

    def _compute_pid(self, state: Dict, gains: np.ndarray) -> np.ndarray:
        """Cascade PID: position -> desired attitude -> motor thrusts."""
        Kp_pos, Ki_pos, Kd_pos = gains[:3], gains[3:6], gains[6:9]
        Kp_att = gains[9:12]

        pos_err = self.cfg.target_position - state["position"]
        vel_err = -state["velocity"]

        # Position loop -> desired acceleration
        des_acc = Kp_pos * pos_err + Kd_pos * vel_err

        # Map to motor thrusts (simplified mixer)
        thrust = np.clip(
            self.dynamics.mass * (9.81 + des_acc[2]), 0.0, 4 * self.dynamics.max_thrust
        )
        torques = Kp_att * (np.zeros(3) - state["euler"])
        motor_thrusts = self.dynamics.mixer(thrust, torques)
        return motor_thrusts

    def _compute_reward(self, state: Dict, fault_active: bool) -> float:
        pos_err = np.linalg.norm(state["position"] - self.cfg.target_position)
        att_err = np.linalg.norm(state["euler"])
        vel_pen = np.linalg.norm(state["velocity"])

        r = (
            -1.0 * pos_err
            - 0.5 * att_err
            - 0.01 * vel_pen
        )
        if fault_active and pos_err < 0.5:
            r += 5.0   # fault recovery bonus
        return float(r)

    def render(self):
        pass  # Visualization via external scripts/plotting utilities

    def close(self):
        pass
