# System Architecture

## Overview

The RL-FTC framework is composed of four tightly integrated subsystems:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION LOOP (dt = 10ms)                  │
│                                                                     │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │  Reference  │    │   RL Agent       │    │  Fault Injector   │  │
│  │  Trajectory │    │ (PPO / SAC)      │    │  (Motor Model)    │  │
│  └──────┬──────┘    └────────┬─────────┘    └────────┬──────────┘  │
│         │ setpoint           │ gain deltas             │ kf_factors  │
│         ▼                   ▼                         ▼             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Cascade PID Controller                      │   │
│  │   Position Loop (x,y,z)  →  Attitude Loop (φ,θ,ψ)           │   │
│  │   Gains auto-tuned by RL agent                               │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                                │ motor thrusts [T1..T4]             │
│  ┌────────────────────────────▼────────────────────────────────┐   │
│  │                   Quadrotor Dynamics (RK4)                   │   │
│  │   6-DOF rigid body + motor model + aerodynamics              │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                                │ state [pos, vel, euler, omega]      │
│  ┌─────────────────────────────▼──────────────────────────────┐    │
│  │                Extended Kalman Filter (EKF)                  │    │
│  │   Augmented state: kinematics + kf_factors (4 motors)        │    │
│  │   Online estimation of fault-related thrust degradation      │    │
│  └──────────────────────┬─────────────────────────────────────┘    │
│                          │ estimated kf_factors                      │
│  ┌───────────────────────▼──────────────────────────────────────┐   │
│  │                     Fault Monitor                             │   │
│  │   Sliding-window threshold detection + CUSUM logic            │   │
│  │   Triggers fault events → feeds EKF output to RL observation  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Subsystem Details

### 1. Quadrotor Dynamics (`src/environment/dynamics.py`)

**Model:** 6-DOF Newton-Euler rigid body

**State vector:**
```
x = [px, py, pz,       (position, world frame)
     vx, vy, vz,       (velocity, world frame)
     φ,  θ,  ψ,        (roll, pitch, yaw — ZYX Euler)
     p,  q,  r]        (body angular rates)
```

**Motor layout (X-configuration):**
```
    M1 (CW)  ←arm→  M2 (CCW)
         \         /
          \       /
           [body]
          /       \
         /         \
    M3 (CCW)     M4 (CW)
```

**Integration:** 4th-order Runge-Kutta at dt=10ms

---

### 2. Fault Injection (`src/environment/fault_injection.py`)

| Fault Type | Model | Parameter |
|---|---|---|
| `motor_resistance` | `kf_factor = 1 - severity` | Winding resistance increase |
| `propeller_damage` | `kf_factor = (1 - severity)²` | Blade efficiency loss |
| `complete_failure` | `kf_factor = 0` | Total motor loss |

Faults can be **instantaneous** or **gradual** (linear ramp over N steps).

---

### 3. Extended Kalman Filter (`src/estimator/ekf.py`)

**Augmented state (16-dim):**
```
x_aug = [pos(3), vel(3), euler(3), omega(3), kf_factors(4)]
```

**Process model:** Nonlinear dynamics (same as simulation) + random-walk kf_factors

**Observation model:** Linear — direct measurement of kinematic state

**Jacobian:** Computed numerically via finite differences (5th-order accuracy)

**Covariance update:** Joseph form for numerical stability

---

### 4. RL Agent (`src/agent/`)

**Observation space (28-dim):**
```
obs = [pos(3), vel(3), euler(3), omega(3), pid_gains(12), fault_params(4)]
```

**Action space (12-dim):**
```
action = delta_gains ∈ [-1, 1]^12
```
Gains updated as: `gains_new = clip(gains + action * max_delta, gain_min, gain_max)`

**Algorithms supported:** PPO (on-policy), SAC (off-policy)

**Training curriculum:**
- Phase 1: No fault (severity=0.0) — learn nominal hovering
- Phase 2: Mild fault (severity=0.2) — begin adaptation
- Phase 3: Moderate fault (severity=0.4) — primary training target
- Phase 4: Severe fault (severity=0.6) — stress test

---

### 5. Reward Design

| Component | Formula | Weight |
|---|---|---|
| Position tracking | `-‖pos - target‖` | 1.0 |
| Attitude tracking | `-‖euler‖` | 0.5 |
| Velocity damping | `-‖vel‖` | 0.05 |
| Control effort | `-‖thrusts‖²/(4·T_max²)` | 0.01 |
| Fault recovery bonus | `+5.0` if fault active and `‖pos_err‖ < 0.5m` | — |
| Crash penalty | `-50.0` at termination | — |

---

## Data Flow Diagram

```
Sensors ──► EKF ──► Fault Monitor
                         │
                    fault_flags
                         │
                    RL Observation ◄─── kinematic state
                         │
                    RL Agent (PPO/SAC)
                         │
                    gain_deltas
                         │
                    PID Controller ◄─── setpoint
                         │
                    motor_thrusts (×kf_factors) ◄─── Fault Injector
                         │
                    Quadrotor Dynamics
                         │
                    new state ──────────────────────────────► loop
```

---

## Key Design Decisions

1. **Why augmented-state EKF?**
   Tracking kf_factors as latent variables avoids explicit fault detection triggers and provides a smooth, continuous fault severity signal to the RL agent.

2. **Why cascade PID + RL gain tuning (not end-to-end RL)?**
   Cascade PID provides a stable, interpretable baseline. RL tuning requires only 12-dimensional action space rather than raw motor commands, reducing exploration difficulty and improving safety.

3. **Why curriculum training?**
   Direct training on severe faults leads to degenerate policies. Progressive severity scheduling bootstraps from nominal flight and gradually shifts the distribution.

4. **Why PPO as primary algorithm?**
   PPO's sample efficiency and stability are well-suited to continuous control. SAC is provided as an alternative for offline/replay-buffer settings.
