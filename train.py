"""
train.py
Entry point for training the RL-based FTC agent.

Usage:
    python scripts/train.py --config configs/train_ppo.yaml
    python scripts/train.py --config configs/train_sac.yaml --run-name exp_01
"""

import argparse
import os
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.quadrotor_env import QuadrotorFTCEnv, EnvConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL-FTC agent")
    parser.add_argument("--config", type=str, default="configs/train_ppo.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, fault_severity: float = 0.0, seed: int = 0):
    env_cfg = EnvConfig(
        sim_dt=cfg["environment"]["sim_dt"],
        episode_length=cfg["environment"]["episode_length"],
        fault_type=cfg["environment"]["fault"].get("type"),
        fault_motor_id=cfg["environment"]["fault"].get("motor_id", 0),
        fault_severity=fault_severity,
        fault_start_step=cfg["environment"]["fault"].get("start_step", 200),
    )
    env = QuadrotorFTCEnv(config=env_cfg)
    env = Monitor(env)
    return env


def main():
    args = parse_args()
    cfg = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{cfg['agent']['algorithm'].lower()}_{timestamp}"
    log_dir = Path("logs") / run_name
    ckpt_dir = Path("checkpoints") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  RL-FTC Training — {run_name}")
    print(f"  Algorithm : {cfg['agent']['algorithm']}")
    print(f"  Fault type: {cfg['environment']['fault'].get('type', 'None')}")
    print(f"  Severity  : {cfg['environment']['fault'].get('severity', 0.0)}")
    print(f"{'='*50}\n")

    # --- Curriculum: start with low fault severity, increase over training ---
    severity_schedule = cfg.get("curriculum", {}).get(
        "severity_schedule", [0.0, 0.2, 0.4, 0.6]
    )

    best_model = None
    for phase, severity in enumerate(severity_schedule):
        print(f"\n[Phase {phase+1}/{len(severity_schedule)}] Fault severity = {severity:.1f}")

        train_env = make_vec_env(
            lambda: make_env(cfg, fault_severity=severity, seed=args.seed),
            n_envs=args.n_envs,
            seed=args.seed,
        )
        eval_env = Monitor(make_env(cfg, fault_severity=severity, seed=args.seed + 99))

        agent_cfg = cfg["agent"]
        algo = agent_cfg["algorithm"].upper()

        if algo == "PPO":
            model_cls = PPO
            model_kwargs = dict(
                learning_rate=agent_cfg["learning_rate"],
                n_steps=agent_cfg.get("n_steps", 2048),
                batch_size=agent_cfg.get("batch_size", 64),
                n_epochs=agent_cfg.get("n_epochs", 10),
                gamma=agent_cfg.get("gamma", 0.99),
                gae_lambda=agent_cfg.get("gae_lambda", 0.95),
                clip_range=agent_cfg.get("clip_range", 0.2),
                verbose=1,
                tensorboard_log=str(log_dir),
                seed=args.seed,
            )
        elif algo == "SAC":
            model_cls = SAC
            model_kwargs = dict(
                learning_rate=agent_cfg["learning_rate"],
                buffer_size=agent_cfg.get("buffer_size", 100_000),
                batch_size=agent_cfg.get("batch_size", 256),
                tau=agent_cfg.get("tau", 0.005),
                gamma=agent_cfg.get("gamma", 0.99),
                verbose=1,
                tensorboard_log=str(log_dir),
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        if best_model is not None:
            # Warm-start from previous phase
            model = model_cls.load(best_model, env=train_env)
            print("  Warm-starting from previous phase checkpoint.")
        else:
            model = model_cls("MlpPolicy", train_env, **model_kwargs)

        callbacks = CallbackList([
            EvalCallback(
                eval_env,
                best_model_save_path=str(ckpt_dir / f"phase_{phase}"),
                log_path=str(log_dir / f"phase_{phase}_eval"),
                eval_freq=max(5000 // args.n_envs, 1),
                n_eval_episodes=10,
                deterministic=True,
                verbose=1,
            ),
            CheckpointCallback(
                save_freq=max(20_000 // args.n_envs, 1),
                save_path=str(ckpt_dir),
                name_prefix=f"rl_ftc_phase{phase}",
            ),
        ])

        total_steps = cfg.get("training", {}).get("total_timesteps_per_phase", 200_000)
        model.learn(total_timesteps=total_steps, callback=callbacks, reset_num_timesteps=(phase == 0))

        best_model = str(ckpt_dir / f"phase_{phase}" / "best_model")
        train_env.close()
        eval_env.close()

    # Save final model
    final_path = str(ckpt_dir / "final_model")
    model.save(final_path)
    print(f"\n✅ Training complete. Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
