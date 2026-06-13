#  RL-Based Fault-Tolerant Control of a Quadrotor
A model-based Reinforcement Learning framework for fault-tolerant PID gain tuning on quadrotor UAVs — detecting faults via Extended Kalman Filter and recovering via adaptive RL policy.

## Overview

Quadrotor UAVs operating in real-world environments (Urban Air Mobility, delivery, inspection) face faults such as motor degradation, propeller damage, or battery failure. This project implements a **Fault-Tolerant Controller (FTC)** that:

1. **Estimates** fault parameters online using an Extended Kalman Filter (EKF)
2. **Detects** fault events from parameter drift
3. **Adapts** PID controller gains using a trained RL agent (PPO/SAC)
4. **Maintains** stable flight despite degraded actuator performance

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Quadrotor Environment                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  Quadrotor  │◄──│  PID Stack   │◄──│  Gain Tuner  │  │
│  │  Dynamics   │   │ (Pos + Att)  │   │  (RL Agent)  │  │
│  └──────┬──────┘   └──────────────┘   └──────▲───────┘  │
│         │ state                               │gains      │
│  ┌──────▼──────┐   ┌──────────────┐          │           │
│  │    EKF      │──►│ Fault Detect │──────────┘           │
│  │  Estimator  │   │  & Monitor   │                      │
│  └─────────────┘   └──────────────┘                      │
└──────────────────────────────────────────────────────────┘
See [docs/architecture.md](docs/architecture.md) for full system design.


## Features

-  6-DOF quadrotor dynamics simulation (Python / Gymnasium)
-  Motor fault injection via equivalent resistance modeling
-  Extended Kalman Filter for online fault parameter estimation
-  PPO and SAC RL agents for adaptive PID gain tuning
-  Configurable reward shaping (tracking error + energy penalty)
-  Modular fault scenario library (single motor, multi-motor, gradual)
-  Training dashboard with W&B / TensorBoard integration
-  Evaluation harness with trajectory metrics (RMSE, max deviation)


## Installation

```bash
git clone https://github.com/yourusername/rl-ftc-quadrotor.git
cd rl-ftc-quadrotor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project package
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.x
- Gymnasium 0.29+
- Stable-Baselines3 2.x
- NumPy, SciPy, Matplotlib

## Usage

### Quick Start — Evaluate Pre-trained Agent

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/ppo_ftc_best.zip
```

### Train from Scratch

```bash
python scripts/train.py --config configs/train_ppo.yaml
```

### Run Fault Scenario Demo

```bash
python scripts/demo.py --fault motor_1_degradation --severity 0.5
```

## Project Structure

```
rl-ftc-quadrotor/
├── src/
│   ├── environment/          # Gymnasium quadrotor env + fault injection
│   │   ├── quadrotor_env.py
│   │   ├── dynamics.py
│   │   └── fault_injection.py
│   ├── estimator/            # EKF-based fault parameter estimator
│   │   ├── ekf.py
│   │   └── fault_monitor.py
│   ├── controllers/          # PID position & attitude controllers
│   │   ├── pid_controller.py
│   │   └── gain_scheduler.py
│   ├── agent/                # RL agent definitions (PPO, SAC)
│   │   ├── ppo_agent.py
│   │   ├── sac_agent.py
│   │   └── reward.py
│   └── utils/                # Logging, plotting, metrics
│       ├── logger.py
│       └── metrics.py
├── configs/                  # YAML experiment configs
├── docs/                     # Architecture, API, experiment notes
├── tests/                    # Unit and integration tests
├── scripts/                  # Train, evaluate, demo entry points
├── workflows/                # CI/CD GitHub Actions
├── requirements.txt
└── setup.py
```

## Configuration

All experiments are configured via YAML files in `configs/`:

```yaml
# configs/default.yaml
environment:
  sim_dt: 0.01           # simulation timestep (s)
  episode_length: 1000   # steps per episode
  fault:
    type: motor_resistance
    motor_id: 0
    severity: 0.4        # 0=nominal, 1=complete failure

agent:
  algorithm: PPO
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10

reward:
  position_weight: 1.0
  attitude_weight: 0.5
  energy_penalty: 0.01
  fault_recovery_bonus: 5.0
```

## Training

Training curves and metrics are logged to TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metrics tracked:
- Episode reward (mean ± std)
- Position RMSE (x, y, z)
- Attitude RMSE (roll, pitch, yaw)
- Fault detection latency
- PID gain adaptation trajectory


## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/ppo_ftc_best.zip \
  --scenarios configs/eval_scenarios.yaml \
  --render

Generates:
- Trajectory comparison plots (nominal vs fault vs FTC)
- Gain evolution over time
- EKF estimation accuracy report
## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. PRs welcome!

## References

1. Blanke et al., *Diagnosis and Fault-Tolerant Control*, Springer, 2006
2. Bhan et al., "Fault Tolerant Control combining Reinforcement Learning and Model-based Control," SysTol 2021
3. Daigle et al., "A Comparison of Filter-Based Approaches for Model-Based Prognostics," IEEE Aerospace 2012

## License

MIT License — see [LICENSE](LICENSE)

