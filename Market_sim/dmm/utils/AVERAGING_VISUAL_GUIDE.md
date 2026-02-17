# Visual Guide: How REIT Averaging Works

## The Simple Version

Imagine you have 3 friends tracking the same stock:
- Friend A says: "It's $100"
- Friend B says: "It's $102"  
- Friend C says: "It's $98"

You average them: (100 + 102 + 98) / 3 = **$100** ✓

That's exactly what we do with REITs, but over time!

---

## The Step-by-Step Process

### Step 1: Load Multiple REITs

```
VNQ (Vanguard):    [$100, $102, $105, $103, $107, ...]  240 months
IYR (iShares):     [$ 95, $ 97, $100, $ 98, $101, ...]  240 months  
SCHH (Schwab):     [$ 20, $ 21, $ 21, $ 20, $ 22, ...]  240 months
PLD (Prologis):    [$ 80, $ 82, $ 85, $ 84, $ 87, ...]  220 months ⚠️
SPG (Simon):       [$150, $145, $148, $142, $146, ...]  240 months
```

Notice PLD only has 220 months of data!

---

### Step 2: Align All Series (Truncate to Shortest)

```
Shortest = 220 months (PLD's length)

VNQ:  [$100, $102, $105, ..., $107]  ✂️ Cut to 220
IYR:  [$ 95, $ 97, $100, ..., $101]  ✂️ Cut to 220
SCHH: [$ 20, $ 21, $ 21, ..., $ 22]  ✂️ Cut to 220
PLD:  [$ 80, $ 82, $ 85, ..., $ 87]  ✓  Already 220
SPG:  [$150, $145, $148, ..., $146]  ✂️ Cut to 220
```

Now all have exactly 220 months!

---

### Step 3: Stack into a Table

Think of it like a spreadsheet:

```
        Month 0  Month 1  Month 2  Month 3  ...  Month 219
VNQ     $100.0   $102.0   $105.0   $103.0  ...   $107.0
IYR     $ 95.0   $ 97.0   $100.0   $ 98.0  ...   $101.0
SCHH    $ 20.0   $ 21.0   $ 21.0   $ 20.0  ...   $ 22.0
PLD     $ 80.0   $ 82.0   $ 85.0   $ 84.0  ...   $ 87.0
SPG     $150.0   $145.0   $148.0   $142.0  ...   $146.0
```

---

### Step 4: Calculate Column Averages

For each month, average down the column:

```
Month 0: (100 + 95 + 20 + 80 + 150) / 5 = 89.0
         ↓    ↓    ↓    ↓     ↓
        VNQ  IYR  SCHH PLD   SPG

Month 1: (102 + 97 + 21 + 82 + 145) / 5 = 89.4

Month 2: (105 + 100 + 21 + 85 + 148) / 5 = 91.8

...continuing for all 220 months...

Month 219: (107 + 101 + 22 + 87 + 146) / 5 = 92.6
```

---

### Step 5: Result!

You now have ONE combined series:

```
Original (5 separate):
  VNQ:  [100, 102, 105, ..., 107]
  IYR:  [ 95,  97, 100, ..., 101]
  SCHH: [ 20,  21,  21, ...,  22]
  PLD:  [ 80,  82,  85, ...,  87]
  SPG:  [150, 145, 148, ..., 146]

Combined (1 averaged):
  AVG:  [89.0, 89.4, 91.8, ..., 92.6]  ← This goes to training!
```

---

## Why This is Better Than Using Just One REIT

### Example: Market Crash Scenario

Let's say the market crashes:

```
Individual REITs:
  VNQ:  -8%   (broad market movement)
  IYR:  -9%   (broad market movement)
  SCHH: -7%   (broad market movement)
  PLD:  -15%  (industrial + bad company news)
  SPG:  -25%  (malls hit extra hard + company issues)

If you use SPG alone:
  "Market crashed -25%!" ❌ Too extreme!

If you use averaged:
  Average = (-8 -9 -7 -15 -25) / 5 = -12.8%
  "Market crashed -13%" ✓ More representative!
```

---

## Benefits at a Glance

| Using 1 REIT | Using Averaged 50 REITs |
|-------------|------------------------|
| Noisy (company-specific events) | Smooth (market-wide trends) |
| Single point of failure | Robust to bad data |
| Sector-specific bias | Diversified representation |
| "Individual stock" | "Index fund" |

---

## The Math (For the Curious)

```python
# Step 1: Load data
reit_data = {
    'VNQ': np.array([100, 102, 105, ...]),
    'IYR': np.array([95, 97, 100, ...]),
    'SCHH': np.array([20, 21, 21, ...]),
}

# Step 2: Find minimum length
min_len = min(len(prices) for prices in reit_data.values())
# Result: 220

# Step 3: Truncate and stack
aligned = np.array([prices[:min_len] for prices in reit_data.values()])
# Shape: (5 REITs, 220 months)
#        ↑         ↑
#     num_reits  time_steps

# Step 4: Average across REITs (axis=0)
averaged = np.mean(aligned, axis=0)
# Shape: (220 months,)
# Each value is average of 5 REITs at that time
```

---

## Real Example from Your Training

When you run `train_dmm_with_qfclient.py`, you'll see:

```
2. Loading REIT data via qfclient...
  Loading 50 REITs...
  Progress: 10/50 REITs processed, 9 successful
  Progress: 20/50 REITs processed, 18 successful
  Progress: 30/50 REITs processed, 27 successful
  Progress: 40/50 REITs processed, 36 successful
  Progress: 50/50 REITs processed, 45 successful

✓ Successfully loaded 45/50 REITs (90.0%)
✓ Combined 45 REITs into 180 data points
✓ Successfully loaded: VNQ, IYR, SCHH, PLD, SPG, AMT, EQIX, PSA, CCI, WELL...
   ... and 35 more
```

That "Combined 45 REITs into 180 data points" means:
- Started with 45 separate price series
- Aligned them all to 180 months (shortest common length)
- Averaged across all 45 at each time point
- Result: 1 smooth series representing "typical REIT behavior"

---

## Alternative: Concatenate Method

Instead of averaging, you can keep them all separate:

```python
# Instead of:
token_prices = combine_multi_reit_data(reit_data, method="average")
# Result: 1 series of 180 points

# Use:
token_prices = combine_multi_reit_data(reit_data, method="concatenate")
# Result: 45 separate series of varying lengths
```

This gives your neural network 45x more training examples!
- Average: 1 series → ~20 training windows
- Concatenate: 45 series → ~900 training windows

But it takes longer to train.

---

## Summary

**What happens:** 50 REITs → Align → Average → 1 smooth series  
**Why:** Reduces noise, more representative, robust to outliers  
**When:** Every time you run `train_dmm_with_qfclient.py`  
**Where:** In `combine_multi_reit_data()` function  
**Result:** Better training data for your Deep Markov Model!

The averaged series captures the **collective behavior** of the REIT market,
not the idiosyncrasies of any single REIT. This makes your model learn
general patterns that apply to "tokenized real estate" as a whole.
