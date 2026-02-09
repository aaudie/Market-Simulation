"""
Training script for reinforcement learning trading agents.

Trains a PPO agent on historical price data with configurable environment
parameters, validates on train/test splits, and saves the best model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def evaluate_model(model: PPO, eval_env: DummyVecEnv, deterministic: bool = True):
    """
    Evaluate trained model on an environment.
    
    Args:
        model: Trained PPO model
        eval_env: Vectorized evaluation environment
        deterministic: If True, use deterministic policy
        
    Returns:
        tuple: (equity_curve list, final_equity float)
    """
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        # Handle both gym and gymnasium return formats
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity



def main():
    """Main training function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # =============================================================================
    # Data Loading
    # =============================================================================
    file_path = os.path.join(script_dir, "data/Equinix.csv")
    print(f"Loading data from: {file_path}")
    df, feature_cols = load_and_preprocess_data(file_path)

    # Split into train/test (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Training bars: {len(train_df):,}")
    print(f"Testing bars:  {len(test_df):,}\n")

    # =============================================================================
    # Environment Configuration
    # =============================================================================
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]  # Stop-loss options (pips)
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]  # Take-profit options (pips)
    WIN = 30  # Observation window size

    def make_train_env():
        """Create training environment with random starts."""
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0
        )

    def make_train_eval_env():
        """Create training evaluation environment (deterministic start)."""
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0
        )

    def make_test_eval_env():
        """Create test evaluation environment (deterministic start)."""
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=1.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,
            time_penalty_pips=0.00,
            unrealized_delta_weight=0.0
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    # =============================================================================
    # Model Configuration
    # =============================================================================
    print("Initializing PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )

    # Setup checkpointing
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="ppo_equinix"
    )

    # =============================================================================
    # Training
    # =============================================================================
    total_timesteps = 600_000
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print("="*60)
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    print("="*60)
    print("✓ Training complete!\n")

    # =============================================================================
    # Model Selection (Best Checkpoint by Out-of-Sample Performance)
    # =============================================================================
    print("="*60)
    print("EVALUATING CHECKPOINTS")
    print("="*60)
    
    equity_curve_test_last, final_equity_test_last = evaluate_model(
        model, test_eval_env
    )
    print(f"[OOS] Final model equity: ${final_equity_test_last:,.2f}")

    best_equity = -np.inf
    best_path = None

    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) 
         if f.endswith(".zip") and f.startswith("ppo_equinix")],
        key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(ckpt_dir, ck)
        try:
            m = PPO.load(ck_path, env=test_eval_env)
            _, final_eq = evaluate_model(m, test_eval_env)
            print(f"[OOS] {ck:40s} -> ${final_eq:,.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] {ck}: {e}")

    # Select best model
    if best_path is None or final_equity_test_last >= best_equity:
        print(f"\n✓ Using final model (best OOS equity: ${final_equity_test_last:,.2f})")
        best_model = model
    else:
        print(f"\n✓ Using checkpoint: {os.path.basename(best_path)}")
        print(f"  OOS equity: ${best_equity:,.2f}")
        best_model = PPO.load(best_path, env=train_vec_env)

    best_model.save("model_equinix_best")
    print(f"\n✓ Best model saved as: model_equinix_best.zip")

    # =============================================================================
    # Final Evaluation and Visualization
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    equity_curve_train, final_equity_train = evaluate_model(
        best_model, train_eval_env
    )
    equity_curve_test, final_equity_test = evaluate_model(
        best_model, test_eval_env
    )

    print(f"[In-Sample]      Final equity: ${final_equity_train:,.2f}")
    print(f"[Out-of-Sample]  Final equity: ${final_equity_test:,.2f}")
    print("="*60)

    # Plot equity curves
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve_train, label="Train (in-sample)", linewidth=2, color='#3498db')
    plt.plot(equity_curve_test, label="Test (out-of-sample)", linewidth=2, color='#e74c3c')
    plt.title("Equity Curves: In-Sample vs Out-of-Sample (Best Model)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
