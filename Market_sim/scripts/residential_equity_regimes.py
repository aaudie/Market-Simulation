import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Read the data
df = pd.read_csv('../../RL_Trading/data/Residential_Equity.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort by date ascending
df = df.sort_values('datetime', ascending=True)

# Calculate returns
df['Returns'] = df['close'].pct_change()
df = df.dropna()

# Calculate rolling volatility
window = 20  # 20-period rolling window
df['Volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(52)  # Annualized (weekly data)

# Drop NaN values from rolling calculation
df = df.dropna()

# Volatility thresholds based on quantiles
vol_33 = df['Volatility'].quantile(0.33)
vol_66 = df['Volatility'].quantile(0.66)

# Assign volatility regimes
df['Regime'] = 'Medium'
df.loc[df['Volatility'] <= vol_33, 'Regime'] = 'Low'
df.loc[df['Volatility'] >= vol_66, 'Regime'] = 'High'

# Print regime statistics
print("\n=== Regime Statistics ===")
print(f"Low Volatility Threshold: {vol_33:.4f}")
print(f"High Volatility Threshold: {vol_66:.4f}")
print(f"\nRegime Counts:")
print(df['Regime'].value_counts())

# Calculate transition probabilities between regimes
def get_transition_matrix(regimes):
    states = ['Low', 'Medium', 'High']
    n_states = len(states)
    
    # Initialize transition counts matrix
    counts = np.zeros((n_states, n_states))
    
    # Count transitions
    for i in range(len(regimes)-1):
        current = regimes.iloc[i]
        next_state = regimes.iloc[i+1]
        current_idx = states.index(current)
        next_idx = states.index(next_state)
        counts[current_idx, next_idx] += 1
    
    # Convert to probabilities
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return probabilities, states

# Get transition matrix
trans_matrix, states = get_transition_matrix(df['Regime'])

# Print transition matrix
print("\n=== Transition Probability Matrix ===")
print("From/To      Low      Medium    High")
for i, state in enumerate(states):
    print(f"{state:10} {trans_matrix[i,0]:.3f}    {trans_matrix[i,1]:.3f}     {trans_matrix[i,2]:.3f}")

# Calculate regime statistics
regime_stats = {}
for regime in ['Low', 'Medium', 'High']:
    mask = df['Regime'] == regime
    returns = df.loc[mask, 'Returns']
    
    regime_stats[regime] = {
        'mean': returns.mean(),
        'std': returns.std(),
        'count': len(returns)
    }

print("\n=== Regime Return Statistics (Weekly) ===")
for regime, stats in regime_stats.items():
    print(f"\n{regime} Volatility Regime:")
    print(f"  Mean Return: {stats['mean']*100:.2f}%")
    print(f"  Std Dev: {stats['std']*100:.2f}%")
    print(f"  Number of Periods: {stats['count']}")

# Create figure
fig = go.Figure()

# Regime colors
regime_colors = {
    'Low': 'rgb(0,255,0)',      # Green
    'Medium': 'rgb(255,255,0)',  # Yellow
    'High': 'rgb(255,0,0)'       # Red
}

# Build continuous segments
dates = df['datetime'].values
prices = df['close'].values
regimes = df['Regime'].values

# Track which regimes have been added to legend
legend_added = set()

for i in range(len(df) - 1):
    x = [dates[i], dates[i+1]]
    y = [prices[i], prices[i+1]]
    regime = regimes[i]
    
    # Only show legend for first occurrence of each regime
    show_legend = regime not in legend_added
    if show_legend:
        legend_added.add(regime)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=regime_colors[regime], width=2),
            name=f"{regime} Volatility",
            showlegend=show_legend,
            legendgroup=regime
        )
    )

# Update layout
fig.update_layout(
    title='Residential Equity Price Colored by Volatility Regimes',
    height=600,
    width=1100,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        title='Price',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

# Save to PNG file
fig.write_image('residential_equity_regimes.png', width=1100, height=600)
print("\nChart saved to 'residential_equity_regimes.png'")
