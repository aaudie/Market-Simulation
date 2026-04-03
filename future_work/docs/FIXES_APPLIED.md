# Deep Markov Model Training Fixes

## Problem Diagnosed

The model was collapsing to always predict **"Panic" regime** for all observations. After data inspection, we found:

### Root Causes

1. **Extreme Volatility Mismatch** (12x difference):
   - Traditional CRE: σ = 0.52% monthly (very stable)
   - Tokenized REIT: σ = 6.6% monthly (includes 2008 crisis with -37% returns)

2. **No Data Normalization**: 
   - Raw returns fed directly to the model
   - Model couldn't distinguish between "calm tokenized market" and "panic traditional market"

3. **Class Imbalance**: 
   - 67 traditional sequences vs 14 tokenized sequences
   - Model biased toward traditional data patterns

4. **Weak Anti-Collapse Mechanisms**:
   - Original entropy regularization insufficient
   - No mechanism to encourage regime diversity

## Solutions Implemented

### 1. **Data Normalization** (`deep_markov_model.py` - `prepare_data()`)

Added **separate standardization** for traditional and tokenized data:

```python
# Standardize returns separately for traditional and tokenized
# This preserves regime differences while normalizing scales

if trad_mask.any():
    trad_returns = returns[trad_mask]
    returns_normalized[trad_mask] = (trad_returns - mean) / std

if token_mask.any():
    token_returns = returns[token_mask]
    returns_normalized[token_mask] = (token_returns - mean) / std
```

**Why this helps**: 
- Prevents high-volatility tokenized data from dominating
- Model can learn regime patterns within each data type
- Returns and volatility are on comparable scales

### 2. **Regime Diversity Loss** (`compute_elbo()`)

Added penalty for uneven regime usage across the batch:

```python
# Compute average regime usage across batch and time
avg_regime_usage = regime_probs.mean(dim=[0, 1])
uniform_target = torch.ones_like(avg_regime_usage) / n_regimes

# Penalize deviation from uniform distribution
diversity_loss = F.kl_div(log(avg_regime_usage), uniform_target)
```

**Why this helps**:
- Actively prevents collapse to single regime
- Encourages model to use all 4 regimes (calm/neutral/volatile/panic)
- Balances against the model's tendency to find easy solutions

### 3. **Better Network Initialization** (`InferenceNetwork`)

Changed output layer initialization:

```python
# Initialize with small weights and zero bias
nn.init.xavier_uniform_(self.output.weight, gain=0.1)
nn.init.zeros_(self.output.bias)
```

**Why this helps**:
- Prevents strong initial preferences for certain regimes
- Model starts with nearly uniform predictions
- Allows data to drive the learning, not initialization

### 4. **Enhanced Gradient Flow** (from previous fix)

Straight-through Gumbel-Softmax estimator:

```python
regime_samples_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
regime_samples_hard = F.gumbel_softmax(logits, tau=tau, hard=True)
regime_samples = hard - soft.detach() + soft  # Straight-through
```

**Why this helps**:
- Forward pass: discrete (hard) samples for clean regime assignments
- Backward pass: continuous (soft) gradients for inference network learning

### 5. **Slower KL Annealing**

Changed from reaching β=1.0 at 50% to 90% of training:

```python
beta = min(1.0, epoch / (epochs * 0.9))  # Was 0.5
```

**Why this helps**:
- Gives reconstruction loss more time to learn patterns
- Prevents KL term from dominating too early
- Allows model to explore before committing to regime structure

### 6. **Temperature Annealing**

Added Gumbel-Softmax temperature decay:

```python
tau = max(0.5, 1.0 - (epoch / epochs) * 0.5)  # 1.0 → 0.5
```

**Why this helps**:
- Early training: soft sampling (τ=1.0) for exploration
- Late training: hard sampling (τ=0.5) for exploitation
- Smooth transition between exploration and exploitation

## Training Metrics to Monitor

Now the training output shows:

```
Epoch  50/200 | Loss: -2.3456 | Recon: 1.2345 | KL: 0.1234 | Ent: 0.9876 | Div: 0.0543 | Beta: 0.556 | Tau: 0.750
```

### What to look for:

1. **KL Divergence should increase**: From ~0 to 0.1-0.5
   - Shows the model is learning structured latent representations
   
2. **Entropy should decrease**: From ~1.386 (uniform) to 0.5-0.8
   - Shows the model is making confident predictions
   
3. **Diversity Loss should stay low**: < 0.2
   - Shows the model is using all regimes, not collapsing
   
4. **Reconstruction Loss should decrease**: Steadily improving
   - Shows the model can predict observations from regimes

## Expected Results

After these fixes, you should see:

- ✅ **All 4 regimes used** (not just "Panic")
- ✅ **Meaningful transition matrices** (not all zeros except one column)
- ✅ **Different predictions for traditional vs tokenized** markets
- ✅ **Smooth regime transitions** over time
- ✅ **KL divergence > 0** (was stuck at 0)

## Files Modified

1. `Market_sim/dmm/core/deep_markov_model.py`:
   - `prepare_data()`: Added normalization
   - `compute_elbo()`: Added diversity loss
   - `InferenceNetwork.__init__()`: Better initialization
   - `train()`: Updated metrics tracking

2. `Market_sim/dmm/training/train_dmm_with_qfclient.py`:
   - Updated to use `slow_linear` beta schedule
   - Added explanation of improvements

3. `Market_sim/dmm/training/train_dmm.py`:
   - Updated to use `slow_linear` beta schedule
   - Updated visualization to show entropy and diversity

## Diagnostic Tool Created

`Market_sim/dmm/training/debug_data.py` - Run this to inspect data quality:

```bash
cd Market_Sim/Market_sim/dmm/training
python3 debug_data.py
```

This will show:
- Price and return statistics for both data types
- Volatility distributions
- Sample windows visualization
- Data quality warnings

## Next Steps

1. **Run training**:
   ```bash
   cd Market_Sim/Market_sim/dmm/training
   python3 train_dmm_with_qfclient.py
   ```

2. **Monitor the metrics** during training - all should now be non-zero and meaningful

3. **Check visualizations** - Should see diverse regime predictions, not all panic

4. **If still issues**, adjust hyperparameters:
   - Increase diversity penalty: `diversity_penalty = 0.2 * diversity_loss` (was 0.1)
   - Decrease entropy penalty: `entropy_penalty = -0.005 * entropy` (was -0.01)
   - Lower learning rate: `learning_rate=1e-4` (was 5e-4)

## Technical Details

### Why Separate Normalization?

Normalizing traditional and tokenized data separately allows the model to:
- Learn "calm" regime exists in both data types (just scaled differently)
- Distinguish true regime differences from scale differences
- Compare relative volatility within each market type

### Why Diversity Loss?

The diversity loss term explicitly penalizes the model for:
- Using only one regime (all predictions = panic)
- Ignoring certain regimes (e.g., never predicting "calm")
- Not exploring the full regime space

It's computed across the entire batch, so it encourages the model to use different regimes for different sequences, not the same regime for everything.

### Why Both Entropy and Diversity?

- **Entropy regularization**: Acts on individual predictions (per timestep)
  - Encourages: "Be confident in your prediction"
  - Prevents: "Predict uniform distribution"

- **Diversity loss**: Acts on batch-level statistics
  - Encourages: "Use all regimes across the dataset"
  - Prevents: "Predict the same regime for everything"

They work together: confident predictions (low entropy) that are diverse (balanced regime usage).

---

**Date**: 2026-02-11  
**Issue**: Posterior collapse → Single regime prediction (Panic)  
**Solution**: Multi-faceted: Normalization + Diversity loss + Better initialization + Annealing
