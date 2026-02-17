# Data Requirements & Collection Strategy

**Document Type:** Technical Guide  
**Thesis Relevance:** Medium (Methodology, Appendix)  
**Last Updated:** February 2026

---

## Executive Summary

This document provides a comprehensive guide to data requirements for training the market simulation models. See `dmm/DATA_REQUIREMENTS.md` for the full technical analysis.

**Quick Reference:**
- Current data: 83 sequences (insufficient for Deep Markov Model)
- Minimum viable: 1,500 sequences (hybrid approach recommended)
- Recommended: 5,000+ sequences (robust DMM training)
- Free sources can provide: 250-500 sequences
- Commercial data can provide: 1,000-2,000 sequences

---

## Quick Start Data Collection

### Free Data (0-2 weeks, $0)

**Script to download 200+ REIT sequences:**
```bash
cd Market_Sim/Market_sim
python3 dmm/data_collection/download_reit_data.py
```

This is documented in detail in:
- **[Current Limitations](01_CURRENT_LIMITATIONS.md)** - Section 1.1
- **[Future Improvements](02_FUTURE_IMPROVEMENTS.md)** - Section 1.1

---

## Data Sources Summary

| Source | Type | Cost | Sequences | Quality | Accessibility |
|--------|------|------|-----------|---------|---------------|
| **yfinance** | REIT prices | Free | 200-300 | Good | API (instant) |
| **FRED** | House prices | Free | 50-100 | Excellent | API (instant) |
| **NCREIF** | Institutional CRE | $2,500/yr | 800-1,000 | Excellent | Purchase |
| **CoStar** | Commercial RE | $6,000/yr | 1,500-2,000 | Excellent | Purchase |
| **Zillow** | Residential | Free (scraping) | 100-200 | Good | Web scraping |
| **Synthetic** | Generated | Free (compute) | Unlimited | Depends | Local generation |

---

## For More Information

**Detailed technical specifications**: See `dmm/DATA_REQUIREMENTS.md`

**Implementation guides**: See [Future Improvements](02_FUTURE_IMPROVEMENTS.md) - Section 1.1-1.2

**Data sufficiency checker**: Run `python3 dmm/check_data_sufficiency.py`

---

## Thesis Recommendation

For a thesis, the hybrid model approach is recommended, which requires **zero additional data**. If you want to implement the Deep Markov Model for comparison:

1. **Collect free data** (yfinance + FRED): 2 weeks, 250-500 sequences
2. **Use data augmentation**: 3 days, boost to 1,000-2,000 sequences
3. **Train simplified DMM**: 1 day, hidden_dim=32

This provides a meaningful comparison without major time/cost investment.
