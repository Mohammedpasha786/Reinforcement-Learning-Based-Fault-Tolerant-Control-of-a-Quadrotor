"""
tests/test_dynamics.py
Unit tests for quadrotor dynamics and fault injection.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.dynamics import QuadrotorDynamics
from src.environment.fault_injection import FaultInjector
from src.environment.quadrotor_env import QuadrotorFTCEnv, EnvConfig
from src.estimator.ekf import FaultEKF
from src.estimator.fault_monitor import FaultMonitor


# -----------------------------------------------------------------------
class TestQuadrotorDynamics:

    def setup_method(self):
        self.dyn = QuadrotorDynamics(dt=0.01)
        self.dyn.reset(
            position=np.array([0., 0., 2.]),
            velocity=np.zeros(3),
            euler=np.zeros(3),
            omega=np.zeros(3),
        )

    def test_reset_returns_dict(self):
        state = self.dyn.reset(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        assert isinstance(state, dict)
        for key in ["position", "velocity", "euler", "omega"]:
            assert key in state

    def test_hover_thrust(self):
        """With hover thrust, position should remain roughly constant for one step."""
        hover_force = self.dyn.mass * self.dyn.g / 4
        thrusts = np.full(4, hover_force)
        state0 = self.dyn._state.copy()
        self.dyn.step(thrusts)
        dz = abs(self.dyn._state["position"][2] - state0["position"][2])
        assert dz < 0.05, f"Altitude changed by {dz:.4f} m under hover thrust"

    def test_mixer_shape(self):
        torques = np.array([0.1, -0.1, 0.05])
        forces = self.dyn.mixer(4 * self.dyn.mass * self.dyn.g / 4, torques)
        assert forces.shape == (4,)

    def test_mixer_non_negative(self):
        torques = np.array([0.0, 0.0, 0.0])
        forces = self.dyn.mixer(self.dyn.mass * self.dyn.g, torques)
        assert np.all(forces >= 0.0)

    def test_step_returns_dict(self):
        thrusts = np.full(4, self.dyn.mass * self.dyn.g / 4)
        state = self.dyn.step(thrusts)
        assert isinstance(state, dict)
        for key in ["position", "velocity", "euler", "omega"]:
            assert key in state


# -----------------------------------------------------------------------
class TestFaultInjector:

    def test_nominal_returns_ones(self):
        fi = FaultInjector(fault_type=None)
        fi.reset()
        state = {"position": np.zeros(3), "velocity": np.zeros(3),
                 "euler": np.zeros(3), "omega": np.zeros(3)}
        params = fi.apply(state, step=1, active=False)
        np.testing.assert_array_almost_equal(params, np.ones(4))

    def test_inactive_fault_returns_ones(self):
        fi = FaultInjector(fault_type="motor_resistance", motor_id=0, severity=0.8)
        fi.reset()
        state = {"position": np.zeros(3), "velocity": np.zeros(3),
                 "euler": np.zeros(3), "omega": np.zeros(3)}
        params = fi.apply(state, step=1, active=False)
        np.testing.assert_array_almost_equal(params, np.ones(4))

    def test_motor_resistance_reduces_thrust(self):
        fi = FaultInjector(fault_type="motor_resistance", motor_id=2, severity=0.5)
        fi.reset()
        state = {}
        params = fi.apply(state, step=1, active=True)
        assert params[2] == pytest.approx(0.5, abs=1e-5)
        assert params[0] == pytest.approx(1.0)
        assert params[1] == pytest.approx(1.0)
        assert params[3] == pytest.approx(1.0)

    def test_complete_failure(self):
        fi = FaultInjector(fault_type="complete_failure", motor_id=1, severity=1.0)
        fi.reset()
        params = fi.apply({}, step=1, active=True)
        assert params[1] == pytest.approx(0.0)

    def test_severity_bounds(self):
        # severity > 1.0 should be clamped
        fi = FaultInjector(fault_type="motor_resistance", motor_id=0, severity=2.0)
        assert fi.severity == 1.0

    def test_unknown_fault_type_raises(self):
        with pytest.raises(ValueError):
            FaultInjector(fault_type="laser_malfunction", motor_id=0, severity=0.5)


# -----------------------------------------------------------------------
class TestFaultEKF:

    def setup_method(self):
        self.ekf = FaultEKF(dt=0.01)
        self.ekf.reset(initial_pos=np.array([0., 0., 2.]))

    def test_initial_kf_factors_nominal(self):
        kf = self.ekf.get_fault_params()
        np.testing.assert_array_almost_equal(kf, np.ones(4))

    def test_predict_update_cycle(self):
        thrusts = np.full(4, 9.81 / 4)
        self.ekf.predict(thrusts)
        measurement = np.zeros(12)
        self.ekf.update(measurement)
        # After one step, state should still be reasonable
        kf = self.ekf.get_fault_params()
        assert np.all(kf >= 0.0)
        assert np.all(kf <= 1.0)

    def test_covariance_decreases_with_updates(self):
        thrusts = np.full(4, 9.81 / 4)
        trace_initial = self.ekf.get_covariance_trace()
        for _ in range(50):
            self.ekf.predict(thrusts)
            meas = np.random.randn(12) * 0.01
            self.ekf.update(meas)
        trace_final = self.ekf.get_covariance_trace()
        assert trace_final < trace_initial

    def test_reset_restores_nominal(self):
        self.ekf.x[12] = 0.3  # corrupt kf_factor
        self.ekf.reset()
        kf = self.ekf.get_fault_params()
        np.testing.assert_array_almost_equal(kf, np.ones(4))


# -----------------------------------------------------------------------
class TestFaultMonitor:

    def test_no_fault_below_threshold(self):
        monitor = FaultMonitor(fault_threshold=0.85, window_size=20)
        nominal_kf = np.ones(4)
        for step in range(30):
            flags = monitor.update(nominal_kf, step=step)
        assert not np.any(flags)

    def test_detects_fault_above_threshold(self):
        monitor = FaultMonitor(fault_threshold=0.85, window_size=20, confidence_min=0.9)
        faulty_kf = np.array([0.5, 1.0, 1.0, 1.0])
        flags = None
        for step in range(30):
            flags = monitor.update(faulty_kf, step=step)
        assert flags[0] == True
        assert flags[1] == False

    def test_records_fault_event(self):
        monitor = FaultMonitor(fault_threshold=0.85, window_size=10, confidence_min=0.9)
        faulty_kf = np.array([0.4, 1.0, 1.0, 1.0])
        for step in range(15):
            monitor.update(faulty_kf, step=step)
        events = monitor.get_fault_events()
        assert len(events) >= 1
        assert events[0].motor_id == 0

    def test_reset_clears_state(self):
        monitor = FaultMonitor()
        faulty_kf = np.array([0.3, 1.0, 1.0, 1.0])
        for step in range(25):
            monitor.update(faulty_kf, step=step)
        monitor.reset()
        assert not monitor.any_fault()
        assert len(monitor.get_fault_events()) == 0


# -----------------------------------------------------------------------
class TestQuadrotorEnv:

    def test_reset_returns_obs_and_info(self):
        env = QuadrotorFTCEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_step_valid_action(self):
        env = QuadrotorFTCEnv()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_terminates_on_flip(self):
        cfg = EnvConfig(episode_length=2000, max_tilt_deg=5.0)
        env = QuadrotorFTCEnv(config=cfg)
        env.reset()
        # Apply maximum destabilizing action
        terminated = False
        for _ in range(200):
            action = env.action_space.high
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        # Should have terminated due to excessive tilt
        assert terminated

    def test_fault_config_propagates(self):
        cfg = EnvConfig(
            fault_type="motor_resistance",
            fault_motor_id=0,
            fault_severity=0.5,
            fault_start_step=5,
        )
        env = QuadrotorFTCEnv(config=cfg)
        env.reset()
        # Check fault injector configured correctly
        assert env.fault_injector.fault_type == "motor_resistance"
        assert env.fault_injector.severity == 0.5
