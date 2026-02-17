"""
Guide: Training DMM with Multiple REITs

This guide explains how to train your Deep Markov Model with data from
many REITs instead of just a few.

Author: Market_Sim Development Team
"""

# =============================================================================
# QUICK START - 3 Options for Using Multiple REITs
# =============================================================================

"""
OPTION 1: Use Top 20 Liquid REITs (RECOMMENDED FOR STARTERS)
-------------------------------------------------------------
Fast, reliable, and covers major REIT sectors.

In train_dmm_with_qfclient.py, line 91-95, use:
    reit_symbols = TOP_20_LIQUID

Training time: ~5 minutes


OPTION 2: Use Top 50 Diversified REITs (BALANCED)
--------------------------------------------------
Good balance between diversity and speed.
Covers all major sectors with multiple representatives.

In train_dmm_with_qfclient.py, line 91-95, use:
    reit_symbols = get_recommended_reits(n=50, diversified=True)

Training time: ~10-15 minutes


OPTION 3: Use All 100+ REITs (COMPREHENSIVE)
---------------------------------------------
Maximum diversity and data coverage.
Best for production models.

In train_dmm_with_qfclient.py, line 91-95, use:
    reit_symbols = ALL_REITS

Training time: ~20-30 minutes
"""

# =============================================================================
# WHAT CHANGED IN THE CODE
# =============================================================================

"""
1. NEW FILE: dmm/utils/reit_symbols.py
   - Contains 100+ REIT symbols organized by category
   - Includes helper functions to get REITs by category or count

2. UPDATED FILE: dmm/utils/qfclient_data_loader.py
   - load_multi_reit_data() now has progress tracking
   - New function: combine_multi_reit_data() - combines multiple REITs
   - New function: load_reit_portfolio() - convenience wrapper

3. UPDATED FILE: dmm/training/train_dmm_with_qfclient.py
   - Now loads 50 REITs by default (up from 3)
   - Can easily be changed to 20, 50, or 100+ REITs
   - Better progress reporting
"""

# =============================================================================
# HOW IT WORKS
# =============================================================================

"""
DATA LOADING PROCESS:
---------------------

1. Script loads list of REIT symbols from reit_symbols.py
   
2. For each REIT:
   - Fetches 20 years of monthly price data via Yahoo Finance
   - Handles failures gracefully (some REITs may not have full history)
   - Tracks progress every 10 REITs
   
3. Combines all successful REIT data using one of these methods:
   
   a) AVERAGE (default):
      - Equal-weighted average of all REITs
      - Creates smoother, more representative data
      - Reduces noise from individual REIT volatility
      
   b) MEDIAN:
      - Median price across all REITs
      - More robust to outliers
      
   c) LONGEST:
      - Uses the single REIT with most data
      - Simple but loses diversification
      
   d) CONCATENATE:
      - Keeps all REITs separate
      - Creates MORE training sequences
      - Better for deep learning (more data!)

4. Uses combined data as "tokenized" market proxy for DMM training


WHY COMBINE MULTIPLE REITs?
---------------------------

1. Diversification: No single REIT represents the entire market
2. Noise reduction: Averaging smooths out idiosyncratic movements
3. Robustness: If one REIT has bad data, others compensate
4. Market representation: Broad portfolio better represents tokenized RE market
5. More data: Some methods create additional training sequences
"""

# =============================================================================
# CUSTOMIZATION EXAMPLES
# =============================================================================

"""
EXAMPLE 1: Load Only Residential REITs
---------------------------------------
"""
from dmm.utils.reit_symbols import RESIDENTIAL_REITS
from dmm.utils.qfclient_data_loader import load_multi_reit_data, combine_multi_reit_data

reit_data = load_multi_reit_data(RESIDENTIAL_REITS, years=10, interval="monthly")
combined = combine_multi_reit_data(reit_data, method="average")
print(f"Loaded {len(reit_data)} residential REITs")

"""
EXAMPLE 2: Load REITs by Category
----------------------------------
"""
from dmm.utils.reit_symbols import get_reits_by_category

healthcare_reits = get_reits_by_category('healthcare')
industrial_reits = get_reits_by_category('industrial')

# Load and combine
all_symbols = healthcare_reits + industrial_reits
reit_data = load_multi_reit_data(all_symbols, years=5)

"""
EXAMPLE 3: Create More Training Data with Concatenation
-------------------------------------------------------
"""
from dmm.utils.reit_symbols import TOP_50_DIVERSIFIED
from dmm.utils.qfclient_data_loader import load_multi_reit_data, combine_multi_reit_data

reit_data = load_multi_reit_data(TOP_50_DIVERSIFIED, years=15, interval="monthly")

# Keep all REITs separate (creates 50 separate price series)
all_series = combine_multi_reit_data(reit_data, method="concatenate")

print(f"Created {len(all_series)} separate REIT series")
print(f"Each series has: {[len(s) for s in all_series[:5]]} data points")

# This creates MORE training windows for the DMM!

"""
EXAMPLE 4: Filter by Data Quality
----------------------------------
"""
from dmm.utils.qfclient_data_loader import load_multi_reit_data, combine_multi_reit_data

# Only use REITs with at least 10 years of data
reit_data = load_multi_reit_data(ALL_REITS, years=20)
combined = combine_multi_reit_data(reit_data, method="average", min_length=120)  # 10 years monthly

"""
EXAMPLE 5: Load Specific REIT Portfolio
---------------------------------------
"""
# Custom portfolio focused on commercial real estate
custom_portfolio = [
    "SPG",   # Simon Property (malls)
    "BXP",   # Boston Properties (office)
    "PLD",   # Prologis (industrial)
    "PSA",   # Public Storage (self-storage)
    "EQIX",  # Equinix (data centers)
    "O",     # Realty Income (net lease)
]

from dmm.utils.qfclient_data_loader import load_reit_portfolio

combined = load_reit_portfolio(
    symbols=custom_portfolio,
    years=15,
    combination_method="average"
)

# =============================================================================
# MODIFYING THE TRAINING SCRIPT
# =============================================================================

"""
To change which REITs are used, edit train_dmm_with_qfclient.py:

Find this section (around line 85-95):

    # OPTION 1: Use all 100+ REITs (comprehensive but slower)
    # reit_symbols = ALL_REITS
    
    # OPTION 2: Use top 50 diversified REITs (good balance)
    reit_symbols = get_recommended_reits(n=50, diversified=True)
    
    # OPTION 3: Use top 20 liquid REITs (fast, reliable)
    # reit_symbols = TOP_20_LIQUID

Uncomment the option you want and comment out the others!


To change the combination method (around line 110-115):

    token_prices = combine_multi_reit_data(
        reit_data, 
        method="average",  # Change this!
        min_length=60
    )

Options:
- "average": Equal-weighted average (smooth, representative)
- "median": Median (robust to outliers)
- "longest": Single longest series (simple)
- "concatenate": Keep all separate (more training data)
"""

# =============================================================================
# PERFORMANCE CONSIDERATIONS
# =============================================================================

"""
DATA LOADING TIMES (approximate, depends on internet speed):
-------------------------------------------------------------
- 20 REITs: ~3-5 minutes
- 50 REITs: ~8-12 minutes  
- 100+ REITs: ~15-25 minutes

Yahoo Finance rate limits: ~2000 requests/hour
Monthly data for 20 years = 1 request per REIT
So you can load 100+ REITs without hitting limits


TRAINING TIMES (depends on combination method):
-----------------------------------------------
- Average/Median/Longest: Same as before (~10-20 min for 200 epochs)
- Concatenate: Longer (more sequences = more training data)

  Example: 50 REITs × 180 months = 9,000 individual data points
  vs. 1 averaged series × 180 months = 180 data points
  
  But creates 50x more training windows!


RECOMMENDED WORKFLOW:
--------------------
1. Start with TOP_20_LIQUID (fast, test your setup)
2. Move to TOP_50_DIVERSIFIED (good balance)
3. Use ALL_REITS for final production model
"""

# =============================================================================
# REIT CATEGORIES AVAILABLE
# =============================================================================

"""
BROAD_MARKET_ETFS (10):
- VNQ, IYR, SCHH, USRT, RWR, FREL, XLRE, ICF, RWO, REET

RESIDENTIAL_REITS (15):
- EQR, AVB, MAA, ESS, UDR, CPT, AIV, ELS, SUI, AMH, 
  INVH, ACC, CSR, IRT, NXRT

OFFICE_REITS (12):
- BXP, VNO, ARE, DEI, HIW, SLG, CLI, CUZ, PGRE, 
  PDM, OFC, JBGS

RETAIL_REITS (14):
- SPG, REG, FRT, KIM, BRX, ROIC, AKR, KRG, RPAI, 
  WPG, TCO, MAC, SKT, SITC

INDUSTRIAL_REITS (7):
- PLD, DRE, FR, STAG, TRNO, EGP, REXR

HEALTHCARE_REITS (12):
- WELL, PEAK, VTR, DOC, HR, OHI, SBRA, LTC, CTRE, 
  MPW, DHC, NHI

SPECIALTY_REITS (40+):
- Data Centers: EQIX, DLR, COR, QTS
- Cell Towers: AMT, CCI, SBAC
- Self-Storage: PSA, EXR, CUBE, LSI, NSA
- Net Lease: O, NNN, STOR, ADC, SRC, EPRT, FCPT, GTY
- Hotels: HST, RHP, PK, SHO, RLJ, APLE, INN
- Gaming: VICI, GLPI, MGP
- Timberland: WY, PCH
- And more!

Total: 100+ unique REITs
"""

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
PROBLEM: "Failed to load many REITs"
SOLUTION: 
- Some REITs may not have 20 years of data
- Reduce years parameter: years=10 or years=5
- Or use min_length in combine_multi_reit_data to filter

PROBLEM: "Takes too long to load"
SOLUTION:
- Use fewer REITs (TOP_20_LIQUID instead of ALL_REITS)
- Set max_failures parameter lower (e.g., max_failures=10)

PROBLEM: "No common data length for averaging"
SOLUTION:
- Use method="concatenate" instead of "average"
- Or set min_length to filter out short series

PROBLEM: "Not enough training data"
SOLUTION:
- Use method="concatenate" to create more sequences
- Or reduce window_size and stride in prepare_dmm_training_data()
"""

print(__doc__)
