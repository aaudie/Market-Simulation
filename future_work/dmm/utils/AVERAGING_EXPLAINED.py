"""
How Automatic REIT Averaging Works
===================================

This document explains the averaging process used to combine multiple
REITs into a single representative price series.
"""

# =============================================================================
# THE AVERAGING PROCESS - STEP BY STEP
# =============================================================================

"""
STEP 1: LOAD MULTIPLE REITs
----------------------------
Each REIT is loaded as a separate price series:

VNQ:  [100.0, 102.5, 105.2, 103.8, 107.1, ...]  (240 months)
IYR:  [95.3,  97.8,  100.1, 98.5,  101.2, ...]  (240 months)
SCHH: [20.1,  20.5,  21.2,  20.8,  21.5,  ...]  (240 months)
PLD:  [80.5,  82.1,  85.3,  84.0,  86.8,  ...]  (220 months)
SPG:  [150.2, 145.8, 148.5, 142.0, 146.3, ...]  (240 months)

Note: REITs may have different history lengths!


STEP 2: ALIGN TO COMMON LENGTH
-------------------------------
Find the shortest series and truncate all others to match:

min_length = 220 months  (PLD has the least data)

VNQ:  [100.0, 102.5, 105.2, ..., 107.1]  (truncated to 220)
IYR:  [95.3,  97.8,  100.1, ..., 101.2]  (truncated to 220)
SCHH: [20.1,  20.5,  21.2,  ..., 21.5]   (truncated to 220)
PLD:  [80.5,  82.1,  85.3,  ..., 86.8]   (full 220)
SPG:  [150.2, 145.8, 148.5, ..., 146.3]  (truncated to 220)

Code:
    min_len = min(len(prices) for prices in reit_data.values())
    aligned_data = np.array([prices[:min_len] for prices in reit_data.values()])


STEP 3: STACK INTO 2D ARRAY
---------------------------
Create a 2D numpy array where each row is a REIT:

aligned_data = 
    [[100.0, 102.5, 105.2, ..., 107.1],   <- VNQ
     [95.3,  97.8,  100.1, ..., 101.2],   <- IYR
     [20.1,  20.5,  21.2,  ..., 21.5],    <- SCHH
     [80.5,  82.1,  85.3,  ..., 86.8],    <- PLD
     [150.2, 145.8, 148.5, ..., 146.3]]   <- SPG

Shape: (5 REITs, 220 months)


STEP 4: COMPUTE EQUAL-WEIGHTED AVERAGE
--------------------------------------
Calculate average across REITs at each time point (axis=0):

Month 0:  (100.0 + 95.3 + 20.1 + 80.5 + 150.2) / 5 = 89.22
Month 1:  (102.5 + 97.8 + 20.5 + 82.1 + 145.8) / 5 = 89.74
Month 2:  (105.2 + 100.1 + 21.2 + 85.3 + 148.5) / 5 = 92.06
...
Month 219: (107.1 + 101.2 + 21.5 + 86.8 + 146.3) / 5 = 92.58

Code:
    averaged_prices = np.mean(aligned_data, axis=0)

Result:
    averaged_prices = [89.22, 89.74, 92.06, ..., 92.58]  (220 months)


STEP 5: RETURN COMBINED SERIES
------------------------------
The averaged series represents the "typical" REIT behavior across all
input REITs.

This becomes your "tokenized data" for training the DMM!
"""

# =============================================================================
# VISUAL EXAMPLE WITH REAL NUMBERS
# =============================================================================

import numpy as np

def demonstrate_averaging():
    """Show a concrete example with 3 REITs over 12 months."""
    
    print("="*70)
    print("AVERAGING EXAMPLE: 3 REITs over 12 months")
    print("="*70)
    
    # Simulate 3 REITs with different price levels
    np.random.seed(42)
    
    vnq = np.array([100.0, 102.3, 101.8, 103.5, 105.2, 104.7, 
                     106.8, 107.2, 105.9, 108.3, 109.1, 110.5])
    
    iyr = np.array([95.0, 96.8, 95.5, 97.2, 99.1, 98.4, 
                    100.5, 101.2, 99.8, 102.1, 103.5, 104.8])
    
    schh = np.array([20.0, 20.3, 20.1, 20.5, 20.9, 20.7, 
                     21.2, 21.4, 21.0, 21.6, 21.9, 22.2])
    
    print("\nIndividual REIT Price Series:")
    print("-" * 70)
    print("Month | VNQ    | IYR    | SCHH   | Average")
    print("-" * 70)
    
    averaged = []
    for month in range(12):
        avg = (vnq[month] + iyr[month] + schh[month]) / 3
        averaged.append(avg)
        print(f"{month:5d} | ${vnq[month]:6.2f} | ${iyr[month]:6.2f} | "
              f"${schh[month]:6.2f} | ${avg:7.2f}")
    
    averaged = np.array(averaged)
    
    print("-" * 70)
    print(f"\nOriginal price ranges:")
    print(f"  VNQ:  ${vnq.min():.2f} - ${vnq.max():.2f} (range: ${vnq.max() - vnq.min():.2f})")
    print(f"  IYR:  ${iyr.min():.2f} - ${iyr.max():.2f} (range: ${iyr.max() - iyr.min():.2f})")
    print(f"  SCHH: ${schh.min():.2f} - ${schh.max():.2f} (range: ${schh.max() - schh.min():.2f})")
    
    print(f"\nAveraged series:")
    print(f"  Average: ${averaged.min():.2f} - ${averaged.max():.2f} (range: ${averaged.max() - averaged.min():.2f})")
    
    # Calculate volatility
    vnq_returns = np.diff(vnq) / vnq[:-1]
    iyr_returns = np.diff(iyr) / iyr[:-1]
    schh_returns = np.diff(schh) / schh[:-1]
    avg_returns = np.diff(averaged) / averaged[:-1]
    
    print(f"\nVolatility (std of returns):")
    print(f"  VNQ:     {vnq_returns.std():.4f}")
    print(f"  IYR:     {iyr_returns.std():.4f}")
    print(f"  SCHH:    {schh_returns.std():.4f}")
    print(f"  Average: {avg_returns.std():.4f} <- Smoother!")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: Averaging reduces noise and creates smoother series")
    print("="*70)


# =============================================================================
# WHY AVERAGING IS BENEFICIAL
# =============================================================================

"""
BENEFITS OF AVERAGING:
----------------------

1. REDUCES IDIOSYNCRATIC NOISE
   - Individual REITs have company-specific volatility
   - Averaging cancels out noise, keeps market-wide trends
   - Result: Smoother, more representative series

2. CREATES DIVERSIFIED PORTFOLIO
   - No single REIT dominates the signal
   - Better represents broad "tokenized real estate" market
   - Like an index fund vs. individual stock

3. ROBUST TO DATA QUALITY ISSUES
   - If one REIT has bad data, others compensate
   - Missing data or outliers have less impact
   - More reliable for training

4. MATHEMATICALLY CLEANER
   - Reduces extreme price movements
   - More stable for regime detection
   - Easier for neural networks to learn patterns

5. REPRESENTS MARKET CONSENSUS
   - Average reflects what multiple market participants are pricing
   - Better proxy for "typical" tokenized RE behavior
   - More generalizable predictions


EXAMPLE - CRISIS SCENARIO:
-------------------------
Imagine a market crash:

VNQ:  -8% (broad market)
IYR:  -9% (broad market)
SCHH: -7% (broad market)
PLD:  -15% (company-specific bad news + market)
SPG:  -20% (malls hit harder + market)

Individual:
- Using PLD alone: -15% (too pessimistic for broad market)
- Using SPG alone: -20% (way too pessimistic!)

Averaged:
- Average = (-8 -9 -7 -15 -20) / 5 = -11.8%
- Better represents typical REIT behavior during crisis
- Smooths out sector-specific and company-specific shocks


ALTERNATIVE: WHY NOT JUST USE VNQ (REIT ETF)?
---------------------------------------------
You could! VNQ already averages ~160 REITs internally.

But using multiple data sources:
1. Provides redundancy (if VNQ data is bad)
2. Captures slightly different compositions
3. More data points for deep learning (with concatenate method)
4. Can experiment with sector-specific models


WHEN TO USE OTHER METHODS:
--------------------------

MEDIAN:
- Use when you have outlier REITs
- More robust than average
- Example: If one REIT has data errors

LONGEST:
- Quick and simple
- Use when you trust one REIT's data quality
- Loses diversification benefits

CONCATENATE:
- Creates MORE training examples (50 REITs = 50x more data!)
- Best for deep learning (neural nets love more data)
- But increases training time
- See detailed explanation below
"""

# =============================================================================
# CONCATENATE METHOD - HOW IT REALLY WORKS
# =============================================================================

"""
CONCATENATE: MAXIMUM TRAINING DATA
===================================

This is the most powerful method but also the most misunderstood.
Let me explain exactly what happens.


WHAT ARE "TRAINING SEQUENCES"?
------------------------------

Each training sequence is a sliding window of price data.

Example with window_size=36 months, stride=6:

VNQ prices: [100, 102, 105, 103, 107, 109, 111, ..., 220]  (180 months)

Creates these sequences:
  Sequence 1: [100, 102, 105, ..., 135]  (months 0-35)
  Sequence 2: [103, 105, 107, ..., 138]  (months 6-41)   <- overlaps!
  Sequence 3: [107, 109, 111, ..., 142]  (months 12-47)  <- more overlap!
  ...
  Sequence 25: [185, 187, 189, ..., 220] (months 144-179)

Each sequence is 36 consecutive months that the neural network learns from.


WITH AVERAGE: 25 TOTAL SEQUENCES
---------------------------------

When you average 50 REITs into one series:

  50 REITs → Average → 1 combined series (180 months) → 25 sequences

Training data the neural network sees:
  - Sequence 1: [avg prices months 0-35]   → label: "tokenized"
  - Sequence 2: [avg prices months 6-41]   → label: "tokenized"
  - ...
  - Sequence 25: [avg prices months 144-179] → label: "tokenized"
  
  PLUS ~100 sequences from traditional CRE data
  
  Total: ~125 training examples

The neural network learns: 
  "When I see THIS pattern (averaged REIT behavior), predict THESE regimes"


WITH CONCATENATE: 1,250 TOTAL SEQUENCES
----------------------------------------

When you keep all 50 REITs separate:

  50 REITs → Keep separate → 50 series × 25 sequences each = 1,250 sequences

Training data the neural network sees:

  From VNQ (25 sequences):
    - Sequence 1: [VNQ prices months 0-35]   → label: "tokenized"
    - Sequence 2: [VNQ prices months 6-41]   → label: "tokenized"
    - ...
    - Sequence 25: [VNQ prices months 144-179] → label: "tokenized"
  
  From IYR (25 sequences):
    - Sequence 26: [IYR prices months 0-35]   → label: "tokenized"
    - Sequence 27: [IYR prices months 6-41]   → label: "tokenized"
    - ...
    - Sequence 50: [IYR prices months 144-179] → label: "tokenized"
  
  From SCHH (25 sequences):
    - Sequence 51: [SCHH prices months 0-35]  → label: "tokenized"
    - ...
  
  ... continuing for all 50 REITs ...
  
  From SPG (25 sequences):
    - Sequence 1226: [SPG prices months 0-35]  → label: "tokenized"
    - ...
    - Sequence 1250: [SPG prices months 144-179] → label: "tokenized"
  
  PLUS ~100 sequences from traditional CRE data
  
  Total: ~1,350 training examples


HOW THE NEURAL NETWORK LEARNS FROM 1,250 SEQUENCES
---------------------------------------------------

The DMM uses batch training. Here's what happens each epoch:

EPOCH 1:
--------
1. Shuffle all 1,350 sequences randomly

2. Create batches of 16 sequences each (1,350 / 16 = ~84 batches)

3. For each batch:
   
   Example Batch 1 might contain:
     - 3 sequences from VNQ  [105, 107, 109, ...]
     - 2 sequences from IYR  [98, 100, 102, ...]
     - 4 sequences from CRE  [1.2M, 1.3M, 1.4M, ...]
     - 5 sequences from SCHH [21, 22, 22, ...]
     - 2 sequences from SPG  [145, 148, 150, ...]
   
   Neural network processes:
     - Looks at price patterns in each sequence
     - Predicts regimes for each: [calm, calm, neutral, volatile, ...]
     - Compares predictions to actual regimes (calculated from volatility)
     - Updates its weights based on prediction errors
     - Uses backpropagation to improve
   
   Example Batch 2 might contain:
     - Different random mix of REITs
     - Network updates weights again
   
   ... process all 84 batches

EPOCH 2:
--------
- Shuffle again (different random order)
- Process 84 batches again
- Network keeps improving

... repeat for 200 epochs ...


AFTER 200 EPOCHS:
-----------------
The network has seen:
  - VNQ patterns 200 times (25 sequences × 200 epochs = 5,000 exposures)
  - IYR patterns 200 times (25 sequences × 200 epochs = 5,000 exposures)
  - SCHH patterns 200 times
  - ... all 50 REITs, 200 times each

It learns:
  "VNQ tends to have this volatility pattern"
  "IYR has that pattern"
  "SCHH has another pattern"
  "But they ALL share these common regime transition behaviors!"


THE KEY DIFFERENCE
------------------

AVERAGING (25 sequences):
  Network sees: "Here's the averaged/typical REIT behavior"
  Network learns: General pattern only
  Training steps per epoch: 25 sequences → ~2 batches
  Total training: 200 epochs × 2 batches = 400 weight updates

CONCATENATE (1,250 sequences):
  Network sees: "Here's VNQ, here's IYR, here's SCHH behavior..."
  Network learns: Individual patterns + what they share in common
  Training steps per epoch: 1,250 sequences → ~78 batches
  Total training: 200 epochs × 78 batches = 15,600 weight updates

The neural network automatically learns to find COMMON PATTERNS across 
all examples. This is called "implicit averaging" or "learning invariances."


WHY THIS WORKS BETTER
----------------------

Analogy: Learning What a "Face" Looks Like

AVERAGING FIRST:
  - Show AI a blurry averaged photo of 50 faces
  - It sees: "Two eyes, one nose, one mouth" (the average)
  - But it's seen only 1 example
  - Doesn't know about variations

CONCATENATE:
  - Show AI 50 clear photos of different faces
  - It sees: Round faces, long faces, dark skin, light skin, etc.
  - It learns: "Despite variations, ALL faces have two eyes, one nose, two ears"
  - It extracts the common pattern from variations
  - More robust to new faces it hasn't seen


IN PRACTICE: WHAT THE NETWORK SEES
-----------------------------------

During training, the neural network doesn't know or care which REIT 
a sequence came from. It just sees:

  Input: [price_t0, price_t1, price_t2, ..., price_t35]
  Context: is_tokenized=1.0, adoption_rate=0.5, time_normalized=0.3
  Target: [calm, calm, neutral, neutral, volatile, panic, ...]
  
  Task: Learn to predict regimes from this pattern

With 1,250 examples instead of 25:
  - It sees MORE VARIATIONS of regime transition patterns
  - It learns what's CONSISTENT across all REITs (the signal)
  - It learns to IGNORE what's unique to each REIT (the noise)
  - Result: Better generalization to new data


THE NEURAL NETWORK'S PERSPECTIVE
---------------------------------

Imagine you're the neural network:

AVERAGING (25 examples):
  Teacher shows you 25 examples of "tokenized market behavior"
  All examples look similar (because they're from same averaged series)
  You memorize: "This specific pattern = these regimes"
  Risk: Overfitting to the averaged pattern
  
CONCATENATE (1,250 examples):
  Teacher shows you 1,250 examples of "tokenized market behavior"
  Examples vary: Some volatile (SPG), some stable (VNQ), some in-between
  You learn: "Despite variations, calm→neutral happens when volatility rises 5%"
           "Neutral→volatile happens when volatility rises 15%"
           "Panic occurs when drops exceed 20% in 6 months"
  Result: You've learned the UNDERLYING RULES, not just memorized patterns


IMPLICIT AVERAGING THROUGH GRADIENT DESCENT
--------------------------------------------

The averaging happens mathematically through training:

Each weight update:
  weight_new = weight_old - learning_rate × gradient
  
  gradient = average of gradients from all examples in batch

Over many batches and epochs:
  - Weights that work well for VNQ patterns get reinforced
  - Weights that work well for IYR patterns get reinforced
  - Weights that work well for BOTH get reinforced MOST
  - Weights that only work for one REIT get averaged out

Final weights = averaged response across all REITs
BUT learned from raw individual data, not pre-averaged data!


PRACTICAL COMPARISON
--------------------

SCENARIO: Model needs to predict regimes for a new REIT (REXR)

WITH AVERAGING:
  - Trained on 25 examples of "average REIT behavior"
  - REXR is an industrial REIT (more volatile than average)
  - Model predicts based on average patterns
  - May miss REXR's higher volatility tendencies
  - Accuracy: 70%

WITH CONCATENATE:
  - Trained on 1,250 examples from diverse REITs
  - Has seen VNQ (stable), SPG (volatile malls), PLD (industrial)
  - Recognizes REXR is similar to PLD (both industrial)
  - Adjusts predictions based on learned volatility patterns
  - Accuracy: 85%


WHEN TO USE EACH METHOD
------------------------

USE AVERAGING IF:
  ✓ You're prototyping/testing
  ✓ You want fast results (30 min training)
  ✓ You need a baseline quickly
  ✓ You have limited compute resources
  ✓ "Good enough" accuracy is fine

USE CONCATENATE IF:
  ✓ You want maximum model performance
  ✓ You have time for longer training (6-10 hours)
  ✓ You're building a production model
  ✓ You want robustness to sector variations
  ✓ You can handle 50x more training time


THE BOTTOM LINE
---------------

CONCATENATE doesn't just give you more data.

It gives you:
  - More DIVERSE data
  - Natural regularization (prevents overfitting)
  - Better generalization
  - Learned invariances
  - Implicit ensemble learning

The neural network becomes an expert on "REIT behavior in general"
rather than an expert on one specific averaged pattern.
"""

# =============================================================================
# CODE WALKTHROUGH
# =============================================================================

"""
Here's the actual code with annotations:

```python
def combine_multi_reit_data(
    reit_data: Dict[str, np.ndarray],  # {'VNQ': [100, 102, ...], 'IYR': [95, 97, ...]}
    method: str = "average",
    min_length: int = None
) -> np.ndarray:
    
    # STEP 1: Optional filtering by minimum length
    if min_length:
        reit_data = {
            symbol: prices 
            for symbol, prices in reit_data.items() 
            if len(prices) >= min_length  # Keep only REITs with enough history
        }
    
    if method == "average" or method == "median":
        # STEP 2: Find shortest series
        min_len = min(len(prices) for prices in reit_data.values())
        # Example: VNQ=240, IYR=240, PLD=220 → min_len=220
        
        # STEP 3: Truncate all to same length and stack
        aligned_data = np.array([prices[:min_len] for prices in reit_data.values()])
        # Shape: (num_reits, min_len)
        # Example: (50 REITs, 220 months)
        
        # STEP 4: Average across REITs (axis=0)
        if method == "average":
            return np.mean(aligned_data, axis=0)
            # Returns: (220,) array - one value per month
        else:
            return np.median(aligned_data, axis=0)
```


WHAT AXIS=0 MEANS:
-----------------
Numpy array axes:

    aligned_data shape: (5 REITs, 220 months)
                         ↑         ↑
                       axis=0    axis=1

    np.mean(aligned_data, axis=0)
    → Average across axis 0 (the REIT dimension)
    → Result: (220 months,) - one average per month

    np.mean(aligned_data, axis=1)  # Don't do this!
    → Average across axis 1 (the time dimension)
    → Result: (5 REITs,) - one average per REIT (not useful!)
"""

# =============================================================================
# RUN EXAMPLE
# =============================================================================

if __name__ == "__main__":
    demonstrate_averaging()
    
    print("\n\nTo see this in action in your training:")
    print("  cd Market_sim/dmm/training")
    print("  python3 train_dmm_with_qfclient.py")
    print("\nLook for the line:")
    print("  '✓ Combined N REITs into M data points'")
    print("\nThat's the averaging in action!")
