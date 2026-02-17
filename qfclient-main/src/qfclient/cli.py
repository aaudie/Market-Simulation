"""
Command-line interface for qfclient.

Usage:
    qfclient list    - Show configured providers and their status
    qfclient test    - Run diagnostic tests on all providers
"""

import argparse
import sys
from typing import Callable


def list_providers(verbose: bool = False) -> dict:
    """
    List all providers and their configuration status.

    Args:
        verbose: If True, print detailed output

    Returns:
        Dict with 'market' and 'crypto' provider status
    """
    from . import MarketClient, CryptoClient

    market = MarketClient()
    crypto = CryptoClient()

    market_status = market.get_status()
    crypto_status = crypto.get_status()

    if verbose:
        print("=" * 50)
        print("qfclient Provider Status")
        print("=" * 50)

        print("\nMarket Data Providers:")
        print("-" * 30)
        configured_market = []
        unconfigured_market = []
        for name, info in sorted(market_status.items()):
            if info["configured"]:
                configured_market.append(name)
            else:
                unconfigured_market.append(name)

        for name in configured_market:
            print(f"  [x] {name}")
        for name in unconfigured_market:
            print(f"  [ ] {name}")

        print(f"\n  Configured: {len(configured_market)}/{len(market_status)}")

        print("\nCrypto Providers:")
        print("-" * 30)
        configured_crypto = []
        unconfigured_crypto = []
        for name, info in sorted(crypto_status.items()):
            if info["configured"]:
                configured_crypto.append(name)
            else:
                unconfigured_crypto.append(name)

        for name in configured_crypto:
            print(f"  [x] {name}")
        for name in unconfigured_crypto:
            print(f"  [ ] {name}")

        print(f"\n  Configured: {len(configured_crypto)}/{len(crypto_status)}")

        # Show which env vars are needed for unconfigured providers
        if unconfigured_market or unconfigured_crypto:
            print("\nTo configure providers, set these environment variables:")
            print("-" * 30)
            env_vars = {
                "alpaca": "ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY",
                "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
                "eodhd": "EODHD_API_KEY",
                "finnhub": "FINNHUB_API_KEY",
                "fmp": "FMP_API_KEY",
                "fred": "FRED_API_KEY",
                "marketstack": "MARKETSTACK_API_KEY",
                "polygon": "POLYGON_API_KEY",
                "sec": "(no key needed - SEC EDGAR is free)",
                "tiingo": "TIINGO_API_KEY",
                "tradier": "TRADIER_API_KEY",
                "twelve_data": "TWELVE_DATA_API_KEY",
                "yfinance": "(no key needed - install yfinance package)",
                "coingecko": "COINGECKO_API_KEY (optional)",
                "coinmarketcap": "COINMARKETCAP_API_KEY",
                "cryptocompare": "CRYPTOCOMPARE_API_KEY (optional)",
                "messari": "MESSARI_API_KEY (needs migration to new API)",
            }
            for name in unconfigured_market + unconfigured_crypto:
                if name in env_vars:
                    print(f"  {name}: {env_vars[name]}")

    return {
        "market": market_status,
        "crypto": crypto_status,
    }


def run_diagnostics(verbose: bool = True) -> dict:
    """
    Run diagnostic tests on all configured providers.

    Args:
        verbose: If True, print detailed output

    Returns:
        Dict with test results
    """
    import logging
    from datetime import date
    from . import MarketClient, CryptoClient, Interval
    from .common.base import ProviderError

    # Suppress provider failover warnings during tests
    logging.getLogger("qfclient.client").setLevel(logging.ERROR)
    logging.getLogger("qfclient.common.rate_limiter").setLevel(logging.ERROR)

    results = {
        "passed": [],
        "failed": [],
        "skipped": [],
    }

    def run_test(name: str, test_func: Callable, required_provider: str = None) -> bool:
        """Run a single test and record result."""
        try:
            result = test_func()
            results["passed"].append(name)
            if verbose:
                print(f"  [PASS] {name}")
                if result:
                    print(f"         {result}")
            return True
        except Exception as e:
            error_msg = str(e)
            if "No providers configured" in error_msg or "not support" in error_msg.lower():
                results["skipped"].append((name, error_msg))
                if verbose:
                    print(f"  [SKIP] {name}")
                    print(f"         {error_msg[:60]}")
            else:
                results["failed"].append((name, error_msg))
                if verbose:
                    print(f"  [FAIL] {name}")
                    print(f"         {error_msg[:60]}")
            return False

    if verbose:
        print("=" * 50)
        print("qfclient Diagnostic Tests")
        print("=" * 50)

    market = MarketClient()
    crypto = CryptoClient()

    # Market tests
    if verbose:
        print("\nMarket Data Tests:")
        print("-" * 30)

    run_test("Quote (AAPL)", lambda: (
        q := market.get_quote("AAPL"),
        f"${q.price:.2f}"
    )[1])

    run_test("OHLCV (AAPL)", lambda: (
        c := market.get_ohlcv("AAPL", limit=5),
        f"{len(c)} candles"
    )[1])

    run_test("Company Profile (AAPL)", lambda: (
        p := market.get_company_profile("AAPL"),
        f"{p.name}"
    )[1])

    run_test("Options Chain (AAPL)", lambda: (
        exp := market.get_option_expirations("AAPL"),
        chain := market.get_options_chain("AAPL", expiration=exp[0]),
        f"{len(chain.calls)} calls, {len(chain.puts)} puts"
    )[2])

    run_test("Dividends (AAPL)", lambda: (
        d := market.get_dividends("AAPL"),
        f"{len(d)} dividends"
    )[1])

    run_test("Stock Splits (AAPL)", lambda: (
        s := market.get_stock_splits("AAPL"),
        f"{len(s)} splits"
    )[1])

    run_test("Economic Data (FRED)", lambda: (
        r := market.get_economic_indicator("FEDFUNDS", limit=1),
        f"Fed Funds: {r[0].value}%"
    )[1])

    run_test("News (AAPL)", lambda: (
        n := market.get_news("AAPL", limit=3),
        f"{len(n)} articles"
    )[1])

    run_test("SEC Form 4 Summary (AAPL)", lambda: (
        s := market.get_insider_summary("AAPL"),
        f"{s.unique_insiders} insiders, NPR: {s.net_purchase_ratio:.2f}" if s.net_purchase_ratio else f"{s.unique_insiders} insiders"
    )[1])

    run_test("SEC Form 4 Filings (AAPL)", lambda: (
        f := market.get_sec_filings("AAPL", limit=3),
        f"{len(f)} filings" + (f", latest: {f[0].owner_name}" if f else "")
    )[1])

    run_test("SEC Form 4 Transactions (AAPL)", lambda: (
        t := market.get_sec_transactions("AAPL", limit=5),
        f"{len(t)} transactions" + (f", role: {t[0].role.role_type.value}" if t else "")
    )[1])

    # Crypto tests
    if verbose:
        print("\nCrypto Data Tests:")
        print("-" * 30)

    run_test("Crypto Quote (BTC)", lambda: (
        q := crypto.get_quote("BTC"),
        f"${q.price_usd:,.2f}"
    )[1])

    run_test("Crypto OHLCV (BTC)", lambda: (
        c := crypto.get_ohlcv("BTC", limit=5),
        f"{len(c)} candles"
    )[1])

    run_test("Top Coins", lambda: (
        t := crypto.get_top_coins(limit=5),
        f"Top: {[c.symbol for c in t[:3]]}"
    )[1])

    run_test("Global Market", lambda: (
        g := crypto.get_global_market(),
        f"MCap: ${g['total_market_cap_usd']:,.0f}"
    )[1])

    # Summary
    if verbose:
        print("\n" + "=" * 50)
        total = len(results["passed"]) + len(results["failed"]) + len(results["skipped"])
        print(f"Results: {len(results['passed'])}/{total} passed, "
              f"{len(results['failed'])} failed, {len(results['skipped'])} skipped")
        print("=" * 50)

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qfclient",
        description="qfclient - Quantitative Finance Data Client",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List all providers and their configuration status"
    )
    list_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - return status without printing"
    )

    # test command
    test_parser = subparsers.add_parser(
        "test",
        help="Run diagnostic tests on all providers"
    )
    test_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - return results without printing"
    )

    # version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "list":
        result = list_providers(verbose=not args.quiet)
        if args.quiet:
            import json
            print(json.dumps({
                "market": {k: v["configured"] for k, v in result["market"].items()},
                "crypto": {k: v["configured"] for k, v in result["crypto"].items()},
            }, indent=2))
        return 0

    elif args.command == "test":
        result = run_diagnostics(verbose=not args.quiet)
        if args.quiet:
            import json
            print(json.dumps({
                "passed": result["passed"],
                "failed": [{"test": t, "error": e} for t, e in result["failed"]],
                "skipped": [{"test": t, "reason": r} for t, r in result["skipped"]],
            }, indent=2))
        return 1 if result["failed"] else 0

    elif args.command == "version":
        from . import __version__
        print(f"qfclient {__version__}")
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
