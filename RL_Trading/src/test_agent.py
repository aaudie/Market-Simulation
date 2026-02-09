"""
Interactive testing script for trained trading agents.

Allows users to:
1. Select a CSV data file to test on
2. Choose a trained model
3. Run the agent and view performance metrics
4. Export trade history and visualize equity curve
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def run_one_episode(model, vec_env, deterministic=True):
    """
    Run one complete trading episode.
    
    Args:
        model: Trained PPO model
        vec_env: Vectorized trading environment
        deterministic: If True, use deterministic policy (no exploration)
        
    Returns:
        tuple: (equity_curve list, closed_trades list)
    """
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        # Handle both gym and gymnasium return formats
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        equity_curve.append(vec_env.get_attr("equity_usd")[0])

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def main():
    """Main interactive testing function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    # =============================================================================
    # Step 1: Select CSV Data File
    # =============================================================================
    print("\n" + "="*60)
    print("AVAILABLE CSV FILES FOR TESTING")
    print("="*60)
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    for idx, filename in enumerate(csv_files, 1):
        file_path = os.path.join(data_dir, filename)
        try:
            df_temp = pd.read_csv(file_path)
            num_rows = len(df_temp)
            print(f"{idx}. {filename} ({num_rows:,} rows)")
        except Exception:
            print(f"{idx}. {filename} (unable to read)")
    
    print("="*60)
    
    # User selects data file
    while True:
        try:
            choice = input("\nEnter the number of the CSV file you want to test on: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(csv_files):
                selected_file = csv_files[choice_idx]
                break
            print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return
    
    file_path = os.path.join(data_dir, selected_file)
    print(f"\nLoading data from: {selected_file}...")
    
    df, feature_cols = load_and_preprocess_data(file_path)
    print(f"✓ Data loaded: {len(df):,} rows after preprocessing\n")

    test_df = df.copy()

    # Environment parameters (must match training configuration)
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    WIN = 30

    test_env = ForexTradingEnv(
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
        time_penalty_pips=0.0,
        unrealized_delta_weight=0.0
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # =============================================================================
    # Step 2: Select Trained Model
    # =============================================================================
    print("="*60)
    print("AVAILABLE TRAINED MODELS")
    print("="*60)
    
    model_files = [
        f for f in os.listdir(script_dir) 
        if f.startswith("model_") and f.endswith("_best.zip")
    ]
    
    if not model_files:
        print("❌ No trained models found!")
        print("   Looking for files like 'model_*_best.zip'")
        return
    
    for idx, model_name in enumerate(model_files, 1):
        print(f"{idx}. {model_name}")
    
    print("="*60)
    
    # User selects model
    while True:
        try:
            choice = input("\nEnter the number of the model you want to test: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected_model = model_files[choice_idx]
                break
            print(f"Please enter a number between 1 and {len(model_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return
    
    model_path = os.path.join(script_dir, selected_model.replace('.zip', ''))
    print(f"\nLoading model: {selected_model}...")
    model = PPO.load(model_path, env=vec_test_env)
    print("✓ Model loaded successfully!\n")

    # =============================================================================
    # Step 3: Run Testing Episode
    # =============================================================================
    print("="*60)
    print("RUNNING TEST EPISODE")
    print("="*60)
    
    equity_curve, closed_trades = run_one_episode(
        model, 
        vec_test_env, 
        deterministic=True
    )
    
    # =============================================================================
    # Step 4: Display Results
    # =============================================================================
    initial_equity = equity_curve[0]
    final_equity = equity_curve[-1]
    pnl = final_equity - initial_equity
    return_pct = (final_equity / initial_equity - 1) * 100
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Data file:        {selected_file}")
    print(f"Model:            {selected_model}")
    print(f"Initial equity:   ${initial_equity:,.2f}")
    print(f"Final equity:     ${final_equity:,.2f}")
    print(f"Total P&L:        ${pnl:+,.2f} ({return_pct:+.2f}%)")
    print(f"Closed trades:    {len(closed_trades)}")
    print("="*60)

    # =============================================================================
    # Step 5: Export Trade History
    # =============================================================================
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = os.path.join(script_dir, "trade_history_output.csv")
        trades_df.to_csv(out_csv, index=False)
        print(f"\n✓ Trade history saved to: {out_csv}")
    else:
        print("\n⚠ No closed trades recorded.")

    # =============================================================================
    # Step 6: Visualize Equity Curve
    # =============================================================================
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve, label="Equity", linewidth=2, color='#2c3e50')
    plt.axhline(y=initial_equity, color='gray', linestyle='--', 
                label='Initial Equity', alpha=0.7)
    plt.title(f"Equity Curve: {selected_model} on {selected_file}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
