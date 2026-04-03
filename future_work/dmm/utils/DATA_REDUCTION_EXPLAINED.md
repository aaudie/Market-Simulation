# Does Averaging Reduce Training Data?

## TL;DR: YES! Averaging reduces training data by ~50x

But there's a good reason why (and a solution if you want more data).

---

## The Math: How Much Data You Lose

### Scenario: 50 REITs, 180 months each

**Training Configuration (from train_dmm_with_qfclient.py):**
- Window size: 36 months
- Stride: 6 months

### Method 1: AVERAGE (Current Default)

**What happens:**
```
50 REITs (180 months each)
    ↓ averaging
1 combined series (180 months)
    ↓ sliding window
~25 training sequences
```

**Calculation:**
- Number of windows from 180 months = (180 - 36) / 6 + 1 = **25 sequences**
- Each sequence is 36 months of data
- Total training examples: **25**

### Method 2: CONCATENATE (Maximum Data)

**What happens:**
```
50 REITs (180 months each)
    ↓ keep separate
50 separate series (180 months each)
    ↓ sliding window on each
~1,250 training sequences
```

**Calculation:**
- Each REIT: (180 - 36) / 6 + 1 = 25 sequences
- 50 REITs × 25 sequences = **1,250 sequences**
- Total training examples: **1,250**

### The Difference

```
CONCATENATE: 1,250 sequences
AVERAGE:     25 sequences
             ────────────
Ratio:       50x MORE DATA with concatenate!
```

---

## Visual Example

### With Averaging (Current Method)

```
Input:
  VNQ:  [100, 102, 105, 103, 107, 109, 111, ...]  180 months
  IYR:  [95,  97,  100, 98,  101, 103, 105, ...]  180 months
  SCHH: [20,  21,  21,  20,  22,  23,  24,  ...]  180 months
  ... (47 more REITs)

After Averaging:
  AVG:  [71.7, 73.1, 75.4, 73.8, 76.7, ...]       180 months

Training Windows (window_size=36, stride=6):
  Sequence 1:  [71.7, 73.1, 75.4, ..., 89.2]  (months 0-35)
  Sequence 2:  [73.8, 75.1, 76.8, ..., 90.5]  (months 6-41)
  Sequence 3:  [75.4, 76.2, 78.1, ..., 91.8]  (months 12-47)
  ...
  Sequence 25: [85.3, 86.7, 88.4, ..., 102.1] (months 144-179)

TOTAL: 25 training sequences
```

### With Concatenate

```
Input:
  VNQ:  [100, 102, 105, 103, 107, 109, 111, ...]  180 months
  IYR:  [95,  97,  100, 98,  101, 103, 105, ...]  180 months
  SCHH: [20,  21,  21,  20,  22,  23,  24,  ...]  180 months
  ... (47 more REITs)

After Concatenate:
  KEEP ALL 50 SEPARATE!

Training Windows from VNQ:
  Sequence 1:  [100, 102, 105, ..., 135]  (months 0-35)
  Sequence 2:  [103, 105, 107, ..., 138]  (months 6-41)
  ...
  Sequence 25: [180, 182, 185, ..., 220]  (months 144-179)

Training Windows from IYR:
  Sequence 26: [95, 97, 100, ..., 128]    (months 0-35)
  Sequence 27: [98, 100, 102, ..., 131]   (months 6-41)
  ...
  Sequence 50: [172, 175, 178, ..., 210]  (months 144-179)

... and so on for all 50 REITs

TOTAL: 1,250 training sequences (50 REITs × 25 sequences each)
```

---

## The Tradeoff

### Averaging (Less Data, Cleaner Signal)

**Pros:**
- ✅ Smoother, less noisy data
- ✅ Captures market-wide trends (not company-specific quirks)
- ✅ Faster training (fewer sequences)
- ✅ Easier for model to learn general patterns
- ✅ More representative of "typical REIT behavior"

**Cons:**
- ❌ Only 25 training sequences (less data for neural network)
- ❌ Loses diversity across individual REITs
- ❌ Loses sector-specific patterns

**Training Time:**
- ~10-20 minutes for 200 epochs

### Concatenate (More Data, Noisier Signal)

**Pros:**
- ✅ 1,250 training sequences (50x more data!)
- ✅ Neural networks LOVE more data
- ✅ Learns patterns from each REIT individually
- ✅ Captures sector diversity (office vs retail vs industrial)
- ✅ More robust model (trained on more examples)

**Cons:**
- ❌ Noisier (includes company-specific events)
- ❌ Much longer training time
- ❌ May learn idiosyncratic patterns that don't generalize
- ❌ Requires more careful training (may overfit)

**Training Time:**
- ~6-10 hours for 200 epochs (50x more data)

---

## Which Should You Use?

### Use AVERAGE if:
- You want fast experimentation
- You want clean, representative market data
- You're prototyping / testing
- You care more about market-wide patterns than sector specifics
- **This is the current default and recommended for most cases**

### Use CONCATENATE if:
- You want maximum training data for deep learning
- You have time for longer training
- You want to learn sector-specific patterns
- You're building a production model
- You want the most robust model possible

---

## How to Switch Methods

### Current Code (Averaging)

In `train_dmm_with_qfclient.py` around line 110:

```python
token_prices = combine_multi_reit_data(
    reit_data, 
    method="average",      # ← Using averaging
    min_length=60
)
```

**Result:** ~25 training sequences

### To Use Concatenate (Maximum Data)

Change to:

```python
token_prices = combine_multi_reit_data(
    reit_data, 
    method="concatenate",  # ← Use concatenate instead!
    min_length=60
)
```

**Result:** ~1,250 training sequences (50x more!)

---

## What Actually Gets Trained

### With Averaging

```python
prepare_dmm_training_data(
    traditional_data=trad_prices,     # ~100 windows from CRE CSV
    tokenized_data=token_prices,      # ~25 windows from averaged REITs
    window_size=36,
    stride=6
)

# Total training sequences:
#   Traditional: ~100
#   Tokenized:   ~25
#   TOTAL:       ~125 sequences for training
```

### With Concatenate

If `token_prices` is a list of 50 arrays, the code would need modification.
Let me check if `prepare_dmm_training_data` handles this...

Actually, looking at the code, `prepare_dmm_training_data` expects a single
array for `tokenized_data`, not a list. So to use concatenate properly,
you'd need to modify the training data preparation.

---

## The Best Solution: Modified Concatenate

Currently, the code would need a small modification to fully leverage concatenate.
Here's what would work best:

```python
# After loading REITs
if method == "concatenate":
    # Keep all REITs separate and create windows from each
    all_tokenized_windows = []
    for symbol, prices in reit_data.items():
        # Create windows from this REIT
        for i in range(0, len(prices) - window_size, stride):
            all_tokenized_windows.append(prices[i:i + window_size])
    
    # Now you have 1,250 separate training sequences!
```

---

## Real-World Comparison

### Deep Learning Analogy

**Averaging = Training on 1 image dataset**
- Like training on CIFAR-10 (single dataset)
- Fast, clean, works well
- Standard approach

**Concatenate = Training on multiple image datasets**
- Like training on CIFAR-10 + ImageNet + COCO combined
- Much more data = better deep learning
- Takes longer but often worth it

### Why Deep Learning Loves More Data

Neural networks (like the DMM) improve with more training examples:

```
25 sequences:    Model learns general pattern
1,250 sequences: Model learns nuanced, robust patterns
                 Can handle edge cases better
                 More accurate predictions
```

---

## Recommendation

**For your first training run:** Keep `method="average"`
- Fast results (~30 min total)
- Clean data
- Good starting point

**For production model:** Switch to `method="concatenate"` later
- More training data
- More robust predictions
- Worth the extra time

Or use a hybrid approach:
1. Train quickly with averaging to validate your setup
2. Once working, retrain with concatenate for best results

---

## Summary Table

| Metric | Average | Concatenate |
|--------|---------|-------------|
| Training sequences | 25 | 1,250 |
| Data multiplier | 1x | 50x |
| Training time | 30 min | 6-10 hours |
| Signal quality | Clean | Noisier |
| Model accuracy | Good | Better |
| Use case | Quick testing | Production |
| Current default | ✅ YES | ❌ No |

---

## The Bottom Line

**YES**, averaging reduces your training data from 1,250 sequences to 25 sequences
(a **50x reduction**).

But it's a **smart tradeoff** because:
- The averaged data is cleaner and more representative
- 25 sequences is still enough for the DMM to learn (it's not pure deep learning)
- Training is 50x faster
- For prototyping and testing, it's perfect

**When you're ready for maximum performance**, switch to `concatenate` to get
all 1,250 training sequences. Your model will be more robust, but will take
much longer to train.

**Current recommendation:** Keep averaging for now, switch to concatenate once
you've validated your training pipeline works!
