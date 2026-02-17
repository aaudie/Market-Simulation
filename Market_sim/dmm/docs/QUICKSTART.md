# Deep Markov Model - Quick Start Checklist

**Goal:** Train and deploy a Deep Markov Model for your tokenized market simulation in under 15 minutes.

## ✅ Pre-Flight Checklist

### 1. System Requirements
- [ ] Python 3.8+ installed
- [ ] Have 2GB free disk space
- [ ] Working internet connection (for data fetching)

### 2. Verify Current Setup
```bash
cd Market_Sim(wAI)/Market_Sim
ls Market_sim/sim/deep_markov_model.py  # Should exist
ls Market_sim/scripts/train_deep_markov_model.py  # Should exist
```

## 🚀 Installation (5 minutes)

### Step 1: Install PyTorch

**Choose your platform:**

**Option A - macOS (M1/M2 or Intel):**
```bash
pip3 install torch numpy pandas matplotlib
```

**Option B - Linux/Windows (CPU only):**
```bash
pip install torch numpy pandas matplotlib
```

**Option C - Linux with NVIDIA GPU:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib
```

### Step 2: Verify Installation
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

Expected output: `PyTorch 2.x.x installed successfully`

### Step 3: Install Additional Dependencies (Optional)
```bash
cd Market_Sim/Market_sim
pip install -r dmm/requirements.txt
```

## 🎯 Training (5-10 minutes)

### Step 1: Navigate to Project
```bash
cd Market_Sim/Market_sim
```

### Step 2: Start Training
```bash
python3 dmm/train_dmm.py
```

**What you'll see:**
```
======================================================================
 DEEP MARKOV MODEL TRAINING
======================================================================

Loading traditional CRE data...
✓ Loaded 864 monthly data points (Traditional CRE)

Loading tokenized REIT data (VNQ)...
✓ Loaded 252 monthly data points (REIT/Tokenized)

✓ Prepared training data:
  - 67 traditional CRE windows
  - 15 tokenized REIT windows
  - Window size: 72 months
  - Total sequences: 82

Model Configuration:
  - Device: cpu (or cuda if GPU available)
  - Hidden dimension: 128
  - Learning rate: 0.0005
  - Regimes: ['calm', 'neutral', 'volatile', 'panic']

Training for 200 epochs...
Epoch   0/200 | Loss: 2.8543 | Recon: 2.6421 | KL: 0.2122 | Beta: 0.000
Epoch  10/200 | Loss: 1.9234 | Recon: 1.7856 | KL: 0.1378 | Beta: 0.050
...
Epoch 200/200 | Loss: 0.8234 | Recon: 0.6812 | KL: 0.1422 | Beta: 1.000

✓ Model saved: outputs/deep_markov_model.pt
```

**Progress indicators:**
- Loss should decrease steadily
- Final loss typically: 0.7-1.2
- Training time: 5-10 minutes on CPU, 1-2 minutes on GPU

### Step 3: Check Results
```bash
ls outputs/deep_markov_model.pt  # Model checkpoint
ls outputs/deep_markov_model_results.png  # Visualizations
```

## 🧪 Testing (2 minutes)

### Run Minimal Examples
```bash
python3 dmm/examples.py
```

**Expected output:**
```
======================================================================
EXAMPLE 1: Load Model and Predict Next Regime
======================================================================
✓ Loaded model from outputs/deep_markov_model.pt

Traditional Market (calm regime):
  Current: calm
  Predicted next: calm
  Probabilities:
    - calm      : 0.8623 (86.2%)
    - neutral   : 0.1342 (13.4%)
    - volatile  : 0.0035 (0.4%)
    - panic     : 0.0000 (0.0%)

Tokenized Market (volatile regime):
  Current: volatile
  Predicted next: volatile
  Probabilities:
    - calm      : 0.0623 (6.2%)
    - neutral   : 0.1923 (19.2%)
    - volatile  : 0.7298 (73.0%)
    - panic     : 0.0156 (1.6%)
...
```

## 🎮 Integration (5 minutes)

### Compare DMM vs Fixed Matrix
```bash
python3 dmm/integrate_dmm.py
```

**What this does:**
1. Loads your trained DMM
2. Runs 10 Monte Carlo simulations with DMM
3. Runs 10 Monte Carlo simulations with fixed matrices
4. Compares results statistically
5. Generates visualization

**Output:**
```
======================================================================
RUNNING COMPARISON SIMULATIONS
======================================================================

Running 10 Monte Carlo simulations...
  Completed 10/10 runs
✓ Simulations complete

======================================================================
ANALYSIS OF RESULTS
======================================================================

1. Final Price Distribution:
   DMM:   Mean=145234.56, Std=12345.67
   Fixed: Mean=143567.89, Std=11234.56

2. Regime Distribution (Forecast Period):
   DMM:
   - calm      : 42.3%
   - neutral   : 38.7%
   - volatile  : 15.2%
   - panic     :  3.8%

✓ Visualization saved: outputs/dmm_vs_fixed_comparison.png
```

## 📊 Verify Success

### Checklist: Your DMM is Working If...

- [x] Training completed without errors
- [x] Final loss < 1.5
- [x] Model file exists: `outputs/deep_markov_model.pt`
- [x] Visualizations generated: `outputs/deep_markov_model_results.png`
- [x] Examples run successfully
- [x] Learned transition matrices similar to empirical ones
- [x] DMM captures different behavior for traditional vs tokenized

### Visual Checks

**Look at `outputs/deep_markov_model_results.png`:**

1. **Training Loss** (top-left): Should decrease smoothly
2. **Loss Components** (top-right): Both should converge
3. **Regime Predictions**: Should show sensible regime sequences
4. **Transition Matrices**: Should have diagonal dominance

**Look at `outputs/dmm_vs_fixed_comparison.png`:**

1. **Price Trajectories**: Should be realistic (no crazy spikes)
2. **Final Distribution**: Should be roughly normal
3. **Volatility**: Should cluster (periods of high/low)
4. **Regime Distribution**: Should match historical patterns

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip3 install torch
# Or for specific platform, see Installation section above
```

### Problem: "FileNotFoundError: cre_monthly.csv not found"
**Solution:**
```bash
cd Market_sim
python3 scripts/run_complete_analysis.py  # Generates the CSV from parent sim
```

### Problem: Training loss not decreasing
**Solution:**
Try lower learning rate:
```python
# Edit train_deep_markov_model.py, line ~220
model = train_model(
    data=data,
    epochs=200,
    hidden_dim=128,
    learning_rate=1e-4,  # Changed from 5e-4
)
```

### Problem: "CUDA out of memory"
**Solution:**
Use CPU instead:
```python
# Edit deep_markov_model.py, line ~260
self.device = torch.device("cpu")  # Force CPU
```

Or reduce batch size:
```python
# Edit train_deep_markov_model.py
model.train(..., batch_size=8)  # Reduced from 16
```

### Problem: Results look weird
**Solution:**
1. Check training curves - loss should converge
2. Verify data quality - no NaNs or extreme outliers
3. Try training longer: `epochs=500`
4. Inspect learned matrices vs empirical

## 🎓 Next Steps

### Beginner Track
1. ✅ Complete quick start (you are here!)
2. Read `DMM_README.md` sections 1-4
3. Experiment with examples: modify contexts, try different regimes
4. Run your own simulations with DMM

### Intermediate Track
1. Fine-tune hyperparameters (hidden_dim, learning_rate)
2. Add custom context features
3. Train on your own data sources
4. Compare DMM vs Fixed HMM thoroughly

### Advanced Track
1. Modify network architectures
2. Implement attention mechanisms
3. Multi-asset modeling
4. Online learning / continuous adaptation

## 📚 Documentation

**Quick Reference:**
- `QUICKSTART.md` ← You are here
- `README.md` - Comprehensive documentation
- `IMPLEMENTATION.md` - Technical details

**Code Reference:**
- `dmm/deep_markov_model.py` - Core implementation
- `dmm/train_dmm.py` - Training pipeline
- `dmm/integrate_dmm.py` - Integration example
- `dmm/examples.py` - Simple examples

## 🎉 Success!

If you've reached this point and all checks pass, congratulations! 🎊

You now have:
✅ A trained Deep Markov Model  
✅ Context-aware regime predictions  
✅ Better market dynamics than fixed matrices  
✅ Ready for production deployment  

**Time to experiment:** Try different contexts, longer horizons, ensemble models!

## 💡 Pro Tips

1. **Start simple**: Use default hyperparameters first
2. **Monitor training**: Loss curves tell you everything
3. **Compare results**: Always benchmark against fixed HMM
4. **Visualize**: Plots reveal insights that numbers hide
5. **Iterate**: ML is experimental - try different approaches

## 🚨 Common Mistakes to Avoid

❌ Training without sufficient data (need 50+ sequences)  
❌ Not checking training convergence  
❌ Using extreme learning rates (too high or low)  
❌ Ignoring context features  
❌ Not comparing with baseline (fixed HMM)  

✅ Use provided defaults first  
✅ Monitor loss curves  
✅ Start with learning_rate=5e-4  
✅ Include all context features  
✅ Always benchmark performance  

## 📞 Getting Help

**If stuck:**
1. Check troubleshooting section above
2. Review error messages carefully
3. Verify all files exist in correct locations
4. Try running examples first before custom code

**For debugging:**
```python
# Add to any script for verbose output
import logging
logging.basicConfig(level=logging.DEBUG)
```

## ⏱️ Time Investment Summary

| Task | Time | Difficulty |
|------|------|------------|
| Installation | 5 min | ⭐ Easy |
| Training | 5-10 min | ⭐ Easy |
| Testing | 2 min | ⭐ Easy |
| Integration | 5 min | ⭐⭐ Medium |
| **Total** | **~20 min** | ⭐ Easy |

## 🔄 Regular Workflow

Once set up, your workflow is:

```bash
# 1. Update data (if needed)
python3 scripts/run_complete_analysis.py

# 2. Retrain model
python3 dmm/train_dmm.py

# 3. Use in simulations
python3 your_custom_simulation.py  # Using DeepMarkovSimulator
```

---

**Ready to start?** Run the first command in the Installation section! 🚀

**Questions?** Read the detailed `README.md`

**Need help?** Check the Troubleshooting section above
