# Deep Markov Model Training - Progress Report

## 📊 Training Evolution

### **Attempt 1: Complete Collapse to "Panic"**
**Problem**: Model predicted "Panic" regime 100% of the time

**Diagnosis**:
- No data normalization → 12x volatility difference between traditional and tokenized
- Model saw tokenized data as "extremely volatile" compared to traditional
- All predictions collapsed to highest-volatility regime

### **Attempt 2: Uniform Predictions** 
**Problem**: Model predicted uniform distribution (25% each regime)

**Fixes Applied**:
- ✅ Data normalization (separate for traditional/tokenized)
- ✅ Diversity loss to prevent single-regime collapse
- ✅ Straight-through Gumbel-Softmax for gradients
- ✅ Slower KL annealing

**Result**: 
- Still stuck - model found new easy solution: uniform predictions
- Entropy = 1.4 (maximum entropy)
- KL = 0 (perfect match with prior)

### **Attempt 3: Collapse to "Neutral"** ✅ Progress!
**Problem**: Model confidently predicts "Neutral" 80-85% of the time

**What Improved**:
- ✅ KL divergence now > 0 (0.7-0.8) - model is learning!
- ✅ Entropy decreased to 0.7-0.9 - confident predictions!
- ✅ Stable training - no wild oscillations

**Remaining Issue**:
- Single-regime dominance (different regime, same problem)
- Transition matrices show 82-85% self-transition probability

### **Attempt 4: Current Fixes** (In Progress)

**New Losses Added**:

1. **Smarter Diversity Loss**
   - Activates at 50% usage (was 60%)
   - Scales with imbalance severity
   - At 80% usage → strong penalty (0.12)
   - At 50% usage → no penalty (0.0)

2. **Emission Separation Loss** (NEW!)
   - Forces each regime to have distinct emission distributions
   - Penalizes regimes with similar mean returns and volatilities
   - Prevents "everything is neutral" solution
   - Weight: 0.1

3. **Non-uniform Prior**
   - Initial prior favors: calm(2x) > neutral(2x) > volatile(1x) > panic(0.5x)
   - Breaks symmetry and guides learning

4. **Warmup Period**
   - First 20% of training: β=0 (only reconstruction)
   - Allows model to learn data representation before KL constraint

## 🎯 Expected Behavior After Latest Fixes

### Phase 1: Warmup (Epochs 0-40)
- **KL = 0**: Expected, β=0
- **Entropy ≈ 1.4**: Starting high, will decrease
- **Reconstruction**: Should decrease steadily
- **Regime usage**: Can be imbalanced, that's OK

### Phase 2: Learning (Epochs 40-120)
- **KL increases**: 0 → 0.3-0.5
- **Entropy decreases**: 1.4 → 0.8-1.0
- **Diversity loss kicks in**: If usage > 50%
- **Separation loss active**: Forces distinct emissions

### Phase 3: Refinement (Epochs 120-200)
- **KL stabilizes**: Around 0.4-0.6
- **Entropy stabilizes**: Around 0.7-0.9
- **Diverse regime predictions**: Each regime 10-40% usage
- **Meaningful transitions**: Not just self-loops

## 📈 Key Metrics to Monitor

```
Epoch 150/200 | Loss: 2.8456 | Recon: 2.1234 | KL: 0.4567
         Ent: 0.7890 | Div: 0.0234 | Sep: 0.1234 | Beta: 0.857 | Tau: 0.625
```

### What Each Metric Means:

- **Recon** (Reconstruction Loss): How well model predicts returns/volatility
  - Target: < 2.5 by end of training
  - Should decrease smoothly

- **KL** (KL Divergence): How much posterior differs from prior
  - Target: 0.3-0.6 (shows learning)
  - Should be 0 during warmup, then increase

- **Ent** (Entropy): Confidence of predictions
  - Target: 0.7-0.9 (moderately confident)
  - Too low (< 0.5): Overconfident, might be stuck
  - Too high (> 1.2): Too uncertain, not learning

- **Div** (Diversity Loss): Penalty for imbalanced regime usage
  - Target: 0.0-0.05 (low is good)
  - High (> 0.1): Severe imbalance, one regime dominates

- **Sep** (Separation Loss): How similar emission distributions are
  - Target: Decrease over time
  - High initially: Regimes have similar emissions
  - Low eventually: Each regime has distinct characteristics

## 🎲 What Success Looks Like

### Regime Predictions:
```
Traditional CRE:
  Calm: ████████████████████████████░░░░░░ (70%)
  Neutral: ████████░░░░░░░░ (20%)
  Volatile: ████░░ (8%)
  Panic: ░░ (2%)

Tokenized REIT:
  Calm: ████████████░░░░░░ (35%)
  Neutral: ████████████████░░ (45%)
  Volatile: ██████░░ (15%)
  Panic: ██░░ (5%)
```

**Good**: Different distributions, but not uniform!

### Transition Matrices:
```
Traditional:
  Calm→Calm: 0.75 (stays calm)
  Neutral→Neutral: 0.60 (moderately stable)
  Volatile→Neutral: 0.35 (tends to calm down)
  Panic→Volatile: 0.50 (recovers quickly)

Tokenized:
  Calm→Calm: 0.65 (less stable than traditional)
  Neutral→Volatile: 0.15 (more transitions)
  Volatile→Panic: 0.08 (higher crisis risk)
```

**Good**: Diagonal dominant but not overwhelming. Some transitions between states.

## 🔧 If Still Having Issues

### Issue: Still stuck on one regime (any regime)

**Try**:
1. Increase diversity penalty:
   ```python
   diversity_penalty = 0.3 * imbalance_severity * diversity_loss  # was 0.2
   ```

2. Increase separation penalty:
   ```python
   separation_penalty = 0.2 * separation_loss  # was 0.1
   ```

3. Lower target entropy (more confident):
   ```python
   target_entropy = 0.5  # was 0.7
   ```

### Issue: Wild oscillations in loss

**Try**:
1. Lower learning rate:
   ```python
   learning_rate=5e-4  # was 1e-3
   ```

2. Increase gradient clipping:
   ```python
   max_norm=2.0  # was 5.0
   ```

### Issue: KL stays at 0 even after warmup

**Try**:
1. Check if beta is actually increasing:
   - Print beta value each epoch
   - Should be > 0 after epoch 40

2. Verify inference network is updating:
   - Check gradient norms
   - Should be > 0.01

3. Reduce entropy penalty:
   ```python
   entropy_penalty = 0.1 * (entropy - target_entropy).pow(2)  # was 0.5
   ```

## 📝 Files Modified (Latest Round)

1. `deep_markov_model.py`:
   - Added emission separation loss
   - Made diversity loss more sensitive (50% threshold)
   - Improved logging (multi-line output)

2. `train_dmm_with_qfclient.py`:
   - Increased learning rate to 1e-3

3. `train_with_diagnostics.py`:
   - Added regime usage tracking per epoch
   - Added gradient norm monitoring

## 🚀 Run Training

```bash
cd Market_Sim/Market_sim/dmm/training

# Standard training
python3 train_dmm_with_qfclient.py

# With detailed diagnostics
python3 train_with_diagnostics.py
```

## 🎓 Key Lessons Learned

1. **Data normalization is critical** - Without it, scale differences dominate
2. **Multiple regularizers needed** - Each prevents a different type of collapse
3. **Warmup is essential** - Reconstruction must learn before KL constraint
4. **Regime imbalance is OK** - Real markets aren't uniform; calm > panic is natural
5. **Diversity ≠ Uniformity** - We want all regimes used, but not equally

---

**Last Updated**: 2026-02-11  
**Status**: Attempt 4 fixes applied, ready for testing  
**Expected**: Diverse regime predictions with meaningful transitions
