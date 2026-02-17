"""
Comprehensive List of REIT Symbols for Training

This file contains categorized REIT symbols that can be used
for training the Deep Markov Model with diverse real estate data.

Categories:
- Broad Market ETFs: Diversified REIT exposure
- Residential: Apartment, single-family, student housing
- Commercial: Office, retail, mixed-use
- Industrial: Warehouses, distribution, logistics
- Healthcare: Medical offices, hospitals, senior housing
- Specialty: Data centers, cell towers, self-storage, etc.
"""

# =============================================================================
# BROAD MARKET REIT ETFs (Best for comprehensive coverage)
# =============================================================================
BROAD_MARKET_ETFS = [
    "VNQ",      # Vanguard Real Estate ETF - Largest and most liquid
    "IYR",      # iShares U.S. Real Estate ETF
    "SCHH",     # Schwab U.S. REIT ETF
    "USRT",     # iShares Core U.S. REIT ETF
    "RWR",      # SPDR Dow Jones REIT ETF
    "FREL",     # Fidelity MSCI Real Estate ETF
    "XLRE",     # Real Estate Select Sector SPDR Fund
    "ICF",      # iShares Cohen & Steers REALTY ETF
    "RWO",      # SPDR Dow Jones Global Real Estate ETF
    "REET",     # iShares Global REIT ETF
]

# =============================================================================
# RESIDENTIAL REITs (Apartments, Single-Family, Student Housing)
# =============================================================================
RESIDENTIAL_REITS = [
    "EQR",      # Equity Residential - Large cap apartments
    "AVB",      # AvalonBay Communities - Apartments
    "MAA",      # Mid-America Apartment Communities
    "ESS",      # Essex Property Trust - West Coast apartments
    "UDR",      # UDR Inc - Apartments
    "CPT",      # Camden Property Trust - Apartments
    "AIV",      # Apartment Investment and Management
    "ELS",      # Equity LifeStyle Properties - Manufactured homes
    "SUI",      # Sun Communities - Manufactured housing/RV
    "AMH",      # American Homes 4 Rent - Single family
    "INVH",     # Invitation Homes - Single family rental
    "ACC",      # American Campus Communities - Student housing
    "CSR",      # Centerspace - Apartments
    "IRT",      # Independence Realty Trust - Apartments
    "NXRT",     # NexPoint Residential Trust
]

# =============================================================================
# OFFICE REITs
# =============================================================================
OFFICE_REITS = [
    "BXP",      # Boston Properties - Premium office
    "VNO",      # Vornado Realty Trust - NYC office
    "ARE",      # Alexandria Real Estate - Life science
    "DEI",      # Douglas Emmett - Office/multifamily
    "HIW",      # Highwoods Properties - Office
    "SLG",      # SL Green Realty - Manhattan office
    "CLI",      # Mack-Cali Realty - Office
    "CUZ",      # Cousins Properties - Office
    "PGRE",     # Paramount Group - NYC office
    "PDM",      # Piedmont Office Realty Trust
    "OFC",      # Corporate Office Properties Trust
    "JBGS",     # JBG SMITH Properties
]

# =============================================================================
# RETAIL REITs (Malls, Shopping Centers, Outlets)
# =============================================================================
RETAIL_REITS = [
    "SPG",      # Simon Property Group - Malls (largest)
    "REG",      # Regency Centers - Shopping centers
    "FRT",      # Federal Realty Investment Trust
    "KIM",      # Kimco Realty - Shopping centers
    "BRX",      # Brixmor Property Group
    "ROIC",     # Retail Opportunity Investments
    "AKR",      # Acadia Realty Trust
    "KRG",      # Kite Realty Group Trust
    "RPAI",     # Retail Properties of America
    "WPG",      # Washington Prime Group
    "TCO",      # Taubman Centers - Malls
    "MAC",      # Macerich - Malls
    "SKT",      # Tanger Factory Outlet Centers
    "SITC",     # Site Centers - Shopping centers
]

# =============================================================================
# INDUSTRIAL REITs (Warehouses, Distribution, Logistics)
# =============================================================================
INDUSTRIAL_REITS = [
    "PLD",      # Prologis - Largest industrial REIT
    "DRE",      # Duke Realty
    "FR",       # First Industrial Realty Trust
    "STAG",     # STAG Industrial
    "TRNO",     # Terreno Realty
    "EGP",      # EastGroup Properties
    "REXR",     # Rexford Industrial Realty
]

# =============================================================================
# HEALTHCARE REITs (Medical, Senior Housing, Hospitals)
# =============================================================================
HEALTHCARE_REITS = [
    "WELL",     # Welltower - Senior housing/healthcare
    "PEAK",     # Healthpeak Properties
    "VTR",      # Ventas - Healthcare facilities
    "DOC",      # Physicians Realty Trust - Medical offices
    "HR",       # Healthcare Realty Trust
    "OHI",      # Omega Healthcare Investors
    "SBRA",     # Sabra Health Care REIT
    "LTC",      # LTC Properties
    "CTRE",     # CareTrust REIT
    "MPW",      # Medical Properties Trust
    "DHC",      # Diversified Healthcare Trust
    "NHI",      # National Health Investors
]

# =============================================================================
# SPECIALTY REITs (Data Centers, Cell Towers, Self-Storage, Timberland)
# =============================================================================
SPECIALTY_REITS = [
    # Data Centers
    "EQIX",     # Equinix - Data centers (largest)
    "DLR",      # Digital Realty Trust - Data centers
    "COR",      # CoreSite Realty - Data centers
    "QTS",      # QTS Realty Trust - Data centers
    
    # Cell Towers
    "AMT",      # American Tower - Cell towers
    "CCI",      # Crown Castle - Cell towers
    "SBAC",     # SBA Communications - Cell towers
    
    # Self-Storage
    "PSA",      # Public Storage - Self-storage (largest)
    "EXR",      # Extra Space Storage
    "CUBE",     # CubeSmart
    "LSI",      # Life Storage
    "NSA",      # National Storage Affiliates
    
    # Manufactured Housing
    "ELS",      # Equity LifeStyle Properties
    "SUI",      # Sun Communities
    
    # Timberland
    "WY",       # Weyerhaeuser - Timberland
    "PCH",      # PotlatchDeltic - Timberland
    
    # Net Lease
    "O",        # Realty Income - Net lease
    "NNN",      # National Retail Properties
    "STOR",     # STORE Capital
    "ADC",      # Agree Realty
    "SRC",      # Spirit Realty Capital
    "EPRT",     # Essential Properties Realty Trust
    "FCPT",     # Four Corners Property Trust
    "GTY",      # Getty Realty
    
    # Hotels/Resorts
    "HST",      # Host Hotels & Resorts
    "RHP",      # Ryman Hospitality Properties
    "PK",       # Park Hotels & Resorts
    "SHO",      # Sunstone Hotel Investors
    "RLJ",      # RLJ Lodging Trust
    "APLE",     # Apple Hospitality REIT
    "INN",      # Summit Hotel Properties
    
    # Gaming
    "VICI",     # VICI Properties - Gaming/entertainment
    "GLPI",     # Gaming and Leisure Properties
    "MGP",      # MGM Growth Properties
]

# =============================================================================
# AGGREGATE LISTS FOR EASY USAGE
# =============================================================================

# Top 20 most liquid REITs (recommended for reliable data)
TOP_20_LIQUID = [
    "VNQ", "IYR", "SCHH", "USRT", "RWR",      # ETFs
    "PLD", "AMT", "EQIX", "PSA", "CCI",       # Large cap individual
    "WELL", "DLR", "SPG", "O", "AVB",
    "EQR", "VICI", "ARE", "SBAC", "INVH"
]

# Top 50 diversified across sectors
TOP_50_DIVERSIFIED = (
    BROAD_MARKET_ETFS[:5] +
    RESIDENTIAL_REITS[:8] +
    OFFICE_REITS[:5] +
    RETAIL_REITS[:7] +
    INDUSTRIAL_REITS[:5] +
    HEALTHCARE_REITS[:8] +
    SPECIALTY_REITS[:12]
)

# All 100+ REITs (comprehensive dataset)
ALL_REITS = (
    BROAD_MARKET_ETFS +
    RESIDENTIAL_REITS +
    OFFICE_REITS +
    RETAIL_REITS +
    INDUSTRIAL_REITS +
    HEALTHCARE_REITS +
    SPECIALTY_REITS
)

# Remove duplicates while preserving order
ALL_REITS = list(dict.fromkeys(ALL_REITS))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_reits_by_category(category: str) -> list:
    """
    Get REITs by category.
    
    Args:
        category: One of 'broad', 'residential', 'office', 'retail',
                 'industrial', 'healthcare', 'specialty'
    
    Returns:
        List of REIT symbols
    """
    categories = {
        'broad': BROAD_MARKET_ETFS,
        'residential': RESIDENTIAL_REITS,
        'office': OFFICE_REITS,
        'retail': RETAIL_REITS,
        'industrial': INDUSTRIAL_REITS,
        'healthcare': HEALTHCARE_REITS,
        'specialty': SPECIALTY_REITS
    }
    
    return categories.get(category.lower(), [])


def get_recommended_reits(n: int = 20, diversified: bool = True) -> list:
    """
    Get recommended REITs for training.
    
    Args:
        n: Number of REITs to return (default 20)
        diversified: If True, returns diversified selection across sectors
                    If False, returns most liquid REITs
    
    Returns:
        List of REIT symbols
    """
    if diversified:
        return TOP_50_DIVERSIFIED[:n]
    else:
        return TOP_20_LIQUID[:n]


if __name__ == "__main__":
    print("="*70)
    print("REIT SYMBOLS FOR DMM TRAINING")
    print("="*70)
    
    print(f"\nTotal REITs available: {len(ALL_REITS)}")
    print(f"\nBy category:")
    print(f"  Broad Market ETFs: {len(BROAD_MARKET_ETFS)}")
    print(f"  Residential: {len(RESIDENTIAL_REITS)}")
    print(f"  Office: {len(OFFICE_REITS)}")
    print(f"  Retail: {len(RETAIL_REITS)}")
    print(f"  Industrial: {len(INDUSTRIAL_REITS)}")
    print(f"  Healthcare: {len(HEALTHCARE_REITS)}")
    print(f"  Specialty: {len(SPECIALTY_REITS)}")
    
    print(f"\n\nTop 20 Liquid REITs (recommended for starters):")
    for i, symbol in enumerate(TOP_20_LIQUID, 1):
        print(f"  {i:2d}. {symbol}")
    
    print(f"\n\nTo use in training:")
    print(f"  from dmm.utils.reit_symbols import TOP_20_LIQUID, ALL_REITS")
    print(f"  # Use TOP_20_LIQUID for fast training")
    print(f"  # Use ALL_REITS for comprehensive training")
