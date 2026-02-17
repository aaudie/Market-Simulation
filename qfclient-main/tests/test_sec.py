"""
Tests for SEC EDGAR Form 4 provider.

Run with: pytest tests/test_sec.py -v
Run integration tests: pytest tests/test_sec.py -v -m integration
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch

from qfclient import (
    MarketClient,
    SECInsiderTransaction,
    Form4Filing,
    InsiderSummary,
    TransactionCode,
    InsiderRole,
    InsiderRoleType,
    OwnershipType,
)
from qfclient.market.providers.sec import SECProvider


# =============================================================================
# InsiderRole Tests
# =============================================================================

class TestInsiderRole:
    """Test InsiderRole model and role classification."""

    def test_role_defaults(self):
        """Default role should have OTHER type."""
        role = InsiderRole()
        assert role.is_director is False
        assert role.is_officer is False
        assert role.is_ten_percent_owner is False
        assert role.officer_title is None
        assert role.is_ceo is False
        assert role.is_cfo is False
        assert role.role_type == InsiderRoleType.OTHER
        assert role.role_description == "Insider"

    def test_ceo_role_type(self):
        """CEO should have CEO role type."""
        role = InsiderRole(is_officer=True, officer_title="Chief Executive Officer")
        assert role.is_ceo is True
        assert role.is_cfo is False
        assert role.role_type == InsiderRoleType.CEO
        assert "CEO" in role.role_description

    def test_ceo_alternate_title(self):
        """CEO with abbreviated title should be detected."""
        role = InsiderRole(is_officer=True, officer_title="CEO")
        assert role.is_ceo is True
        assert role.role_type == InsiderRoleType.CEO

    def test_cfo_role_type(self):
        """CFO should have CFO role type."""
        role = InsiderRole(is_officer=True, officer_title="Chief Financial Officer")
        assert role.is_cfo is True
        assert role.is_ceo is False
        assert role.role_type == InsiderRoleType.CFO
        assert "CFO" in role.role_description

    def test_cfo_alternate_title(self):
        """CFO with abbreviated title should be detected."""
        role = InsiderRole(is_officer=True, officer_title="CFO")
        assert role.is_cfo is True
        assert role.role_type == InsiderRoleType.CFO

    def test_coo_role_type(self):
        """COO should have COO role type."""
        role = InsiderRole(is_officer=True, officer_title="Chief Operating Officer")
        assert role.is_coo is True
        assert role.role_type == InsiderRoleType.COO

    def test_other_officer_role_type(self):
        """Other officers should have OTHER_OFFICER role type."""
        role = InsiderRole(is_officer=True, officer_title="General Counsel")
        assert role.is_ceo is False
        assert role.is_cfo is False
        assert role.role_type == InsiderRoleType.OTHER_OFFICER
        assert "General Counsel" in role.role_description

    def test_director_role_type(self):
        """Director should have DIRECTOR role type."""
        role = InsiderRole(is_director=True)
        assert role.role_type == InsiderRoleType.DIRECTOR
        assert "Director" in role.role_description

    def test_ten_percent_owner_role_type(self):
        """10% owner should have TEN_PERCENT_OWNER role type."""
        role = InsiderRole(is_ten_percent_owner=True)
        assert role.role_type == InsiderRoleType.TEN_PERCENT_OWNER
        assert "10% Owner" in role.role_description

    def test_multiple_roles_ceo_wins(self):
        """Multiple roles should return most senior role type (CEO)."""
        role = InsiderRole(
            is_director=True,
            is_officer=True,
            is_ten_percent_owner=True,
            officer_title="CEO"
        )
        assert role.role_type == InsiderRoleType.CEO

    def test_director_and_ten_percent(self):
        """Director + 10% owner should return DIRECTOR role type."""
        role = InsiderRole(is_director=True, is_ten_percent_owner=True)
        assert role.role_type == InsiderRoleType.DIRECTOR
        assert "Director" in role.role_description
        assert "10% Owner" in role.role_description

    def test_role_description_multiple(self):
        """Role description should list all applicable roles."""
        role = InsiderRole(
            is_director=True,
            is_officer=True,
            officer_title="VP Sales"
        )
        desc = role.role_description
        assert "VP Sales" in desc
        assert "Director" in desc

    def test_role_type_enum_values(self):
        """Test InsiderRoleType enum values."""
        assert InsiderRoleType.CEO.value == "ceo"
        assert InsiderRoleType.CFO.value == "cfo"
        assert InsiderRoleType.COO.value == "coo"
        assert InsiderRoleType.OTHER_OFFICER.value == "other_officer"
        assert InsiderRoleType.DIRECTOR.value == "director"
        assert InsiderRoleType.TEN_PERCENT_OWNER.value == "ten_percent_owner"
        assert InsiderRoleType.OTHER.value == "other"


# =============================================================================
# SECInsiderTransaction Tests
# =============================================================================

class TestSECInsiderTransaction:
    """Test SECInsiderTransaction model."""

    def test_purchase_transaction(self):
        """Test a basic purchase transaction."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Tim Cook",
            transaction_date=date(2024, 1, 15),
            transaction_code=TransactionCode.PURCHASE,
            shares=1000,
            price_per_share=185.50,
            acquired_or_disposed="A",
            shares_after=50000,
        )
        assert txn.symbol == "AAPL"
        assert txn.owner_name == "Tim Cook"
        assert txn.is_purchase is True
        assert txn.is_sale is False
        assert txn.is_open_market is True
        assert txn.net_shares == 1000
        assert txn.notional_value == 185500.0

    def test_sale_transaction(self):
        """Test a basic sale transaction."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Tim Cook",
            transaction_date=date(2024, 1, 15),
            transaction_code=TransactionCode.SALE,
            shares=5000,
            price_per_share=190.00,
            acquired_or_disposed="D",
            shares_after=45000,
        )
        assert txn.is_purchase is False
        assert txn.is_sale is True
        assert txn.is_open_market is True
        assert txn.net_shares == -5000
        assert txn.notional_value == 950000.0

    def test_position_change_buy(self):
        """Test position change calculation for buy."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.PURCHASE,
            shares=1000,
            price_per_share=150.0,
            acquired_or_disposed="A",
            shares_after=5000,
        )
        assert txn.shares_before == 4000  # 5000 - 1000
        assert txn.position_change_pct == 25.0  # 1000/4000 * 100

    def test_position_change_sell(self):
        """Test position change calculation for sell."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.SALE,
            shares=1000,
            price_per_share=150.0,
            acquired_or_disposed="D",
            shares_after=4000,
        )
        assert txn.shares_before == 5000  # 4000 + 1000
        assert txn.position_change_pct == -20.0  # -1000/5000 * 100

    def test_position_change_no_shares_after(self):
        """Position change should be None if shares_after is not provided."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.PURCHASE,
            shares=1000,
            price_per_share=150.0,
            acquired_or_disposed="A",
        )
        assert txn.shares_after is None
        assert txn.shares_before is None
        assert txn.position_change_pct is None

    def test_position_change_new_position(self):
        """Position change should be None for new positions (shares_before = 0)."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.PURCHASE,
            shares=1000,
            price_per_share=150.0,
            acquired_or_disposed="A",
            shares_after=1000,  # shares_before = 0
        )
        assert txn.shares_before == 0
        assert txn.position_change_pct is None  # Division by zero avoided

    def test_grant_transaction(self):
        """Test grant/award transaction (not open market)."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.ACQUISITION,
            shares=10000,
            price_per_share=0.0,
            acquired_or_disposed="A",
        )
        assert txn.is_open_market is False
        assert txn.is_purchase is True  # Acquiring, but not open market

    def test_derivative_flag(self):
        """Test derivative transaction flag."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.EXERCISE,
            shares=5000,
            price_per_share=100.0,
            acquired_or_disposed="A",
            is_derivative=True,
        )
        assert txn.is_derivative is True
        assert txn.is_open_market is False

    def test_ownership_type(self):
        """Test ownership type defaults and indirect ownership."""
        txn_direct = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.PURCHASE,
            shares=1000,
            price_per_share=150.0,
            acquired_or_disposed="A",
        )
        assert txn_direct.ownership_type == OwnershipType.DIRECT

        txn_indirect = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.PURCHASE,
            shares=1000,
            price_per_share=150.0,
            acquired_or_disposed="A",
            ownership_type=OwnershipType.INDIRECT,
            indirect_ownership_nature="By Trust",
        )
        assert txn_indirect.ownership_type == OwnershipType.INDIRECT
        assert txn_indirect.indirect_ownership_nature == "By Trust"

    def test_transaction_with_role(self):
        """Test transaction with insider role."""
        role = InsiderRole(is_officer=True, officer_title="CEO")
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Tim Cook",
            transaction_date=date.today(),
            transaction_code=TransactionCode.SALE,
            shares=50000,
            price_per_share=190.0,
            acquired_or_disposed="D",
            role=role,
        )
        assert txn.role.is_ceo is True
        assert txn.role.role_type == InsiderRoleType.CEO


# =============================================================================
# Form4Filing Tests
# =============================================================================

class TestForm4Filing:
    """Test Form4Filing model."""

    def test_filing_with_single_transaction(self):
        """Test filing with one transaction."""
        txn = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.PURCHASE,
            shares=1000,
            price_per_share=150.0,
            acquired_or_disposed="A",
        )
        filing = Form4Filing(
            accession_number="0001234567-24-000001",
            filing_date=date.today(),
            symbol="AAPL",
            owner_name="Test Insider",
            transactions=[txn],
        )
        assert filing.transaction_count == 1
        assert filing.total_shares_acquired == 1000
        assert filing.total_shares_disposed == 0
        assert filing.net_shares == 1000
        assert filing.total_value_acquired == 150000.0
        assert filing.net_value == 150000.0
        assert filing.is_net_buyer is True
        assert filing.has_open_market_transactions is True

    def test_filing_with_multiple_transactions(self):
        """Test filing with multiple transactions."""
        buy = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.PURCHASE,
            shares=2000,
            price_per_share=150.0,
            acquired_or_disposed="A",
        )
        sell = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.SALE,
            shares=500,
            price_per_share=155.0,
            acquired_or_disposed="D",
        )
        filing = Form4Filing(
            accession_number="0001234567-24-000002",
            filing_date=date.today(),
            symbol="AAPL",
            owner_name="Test Insider",
            transactions=[buy, sell],
        )
        assert filing.transaction_count == 2
        assert filing.total_shares_acquired == 2000
        assert filing.total_shares_disposed == 500
        assert filing.net_shares == 1500
        assert filing.total_value_acquired == 300000.0
        assert filing.total_value_disposed == 77500.0
        assert filing.net_value == 222500.0
        assert filing.is_net_buyer is True

    def test_filing_net_seller(self):
        """Test filing where insider is net seller."""
        sell = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.SALE,
            shares=5000,
            price_per_share=190.0,
            acquired_or_disposed="D",
        )
        filing = Form4Filing(
            accession_number="0001234567-24-000003",
            filing_date=date.today(),
            symbol="AAPL",
            owner_name="Test Insider",
            transactions=[sell],
        )
        assert filing.net_shares == -5000
        assert filing.is_net_buyer is False

    def test_filing_with_grant(self):
        """Test filing with non-open-market transaction."""
        grant = SECInsiderTransaction(
            symbol="AAPL",
            owner_name="Test",
            transaction_date=date.today(),
            transaction_code=TransactionCode.ACQUISITION,
            shares=10000,
            price_per_share=0.0,
            acquired_or_disposed="A",
        )
        filing = Form4Filing(
            accession_number="0001234567-24-000004",
            filing_date=date.today(),
            symbol="AAPL",
            owner_name="Test Insider",
            transactions=[grant],
        )
        assert filing.has_open_market_transactions is False


# =============================================================================
# InsiderSummary Tests
# =============================================================================

class TestInsiderSummary:
    """Test InsiderSummary model."""

    def test_net_purchase_ratio_bullish(self):
        """Test NPR for bullish scenario (more buying than selling)."""
        summary = InsiderSummary(
            symbol="AAPL",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
            open_market_shares_bought=10000,
            open_market_shares_sold=2000,
            num_buyers=5,
            num_sellers=1,
        )
        # NPR = (10000 - 2000) / (10000 + 2000) = 8000/12000 = 0.667
        assert summary.net_purchase_ratio == pytest.approx(0.667, rel=0.01)
        assert summary.insider_sentiment == "bullish"

    def test_net_purchase_ratio_bearish(self):
        """Test NPR for bearish scenario (more selling than buying)."""
        summary = InsiderSummary(
            symbol="AAPL",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
            open_market_shares_bought=1000,
            open_market_shares_sold=9000,
            num_buyers=1,
            num_sellers=8,
        )
        # NPR = (1000 - 9000) / (1000 + 9000) = -8000/10000 = -0.8
        assert summary.net_purchase_ratio == pytest.approx(-0.8, rel=0.01)
        assert summary.insider_sentiment == "bearish"

    def test_net_purchase_ratio_neutral(self):
        """Test NPR for neutral scenario (equal buying and selling)."""
        summary = InsiderSummary(
            symbol="AAPL",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
            open_market_shares_bought=5000,
            open_market_shares_sold=5000,
            num_buyers=3,
            num_sellers=3,
        )
        assert summary.net_purchase_ratio == 0.0
        assert summary.insider_sentiment == "slightly_bearish"  # 0 is in slightly_bearish range

    def test_net_purchase_ratio_no_activity(self):
        """Test NPR with no trading activity."""
        summary = InsiderSummary(
            symbol="AAPL",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
        )
        assert summary.net_purchase_ratio is None
        assert summary.insider_sentiment == "neutral"

    def test_buy_sell_ratio(self):
        """Test buyer/seller ratio calculation."""
        summary = InsiderSummary(
            symbol="AAPL",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
            num_buyers=3,
            num_sellers=1,
        )
        assert summary.buy_sell_ratio == 0.75  # 3 / (3 + 1)

    def test_buy_sell_ratio_no_traders(self):
        """Test buy/sell ratio with no traders."""
        summary = InsiderSummary(
            symbol="AAPL",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
        )
        assert summary.buy_sell_ratio is None

    def test_net_shares_and_value(self):
        """Test net shares and value calculations."""
        summary = InsiderSummary(
            symbol="AAPL",
            period_start=date.today() - timedelta(days=90),
            period_end=date.today(),
            total_shares_bought=10000,
            total_shares_sold=3000,
            total_value_bought=1500000.0,
            total_value_sold=480000.0,
        )
        assert summary.net_shares == 7000
        assert summary.net_value == 1020000.0


# =============================================================================
# TransactionCode Tests
# =============================================================================

class TestTransactionCode:
    """Test TransactionCode enum."""

    def test_common_codes(self):
        """Test common transaction codes."""
        assert TransactionCode.PURCHASE.value == "P"
        assert TransactionCode.SALE.value == "S"
        assert TransactionCode.ACQUISITION.value == "A"
        assert TransactionCode.GIFT.value == "G"
        assert TransactionCode.EXERCISE.value == "M"

    def test_code_from_value(self):
        """Test creating code from string value."""
        assert TransactionCode("P") == TransactionCode.PURCHASE
        assert TransactionCode("S") == TransactionCode.SALE


# =============================================================================
# SECProvider Tests
# =============================================================================

class TestSECProvider:
    """Test SEC EDGAR provider."""

    def test_provider_always_configured(self):
        """SEC provider should always be available (no API key needed)."""
        provider = SECProvider()
        assert provider.is_configured() is True
        assert provider.provider_name == "sec"

    def test_provider_capabilities(self):
        """SEC provider should support insider transactions only."""
        provider = SECProvider()
        assert provider.supports_insider_transactions is True
        assert provider.supports_quotes is False
        assert provider.supports_ohlcv is False
        assert provider.supports_company_profile is False

    def test_custom_user_agent(self):
        """Test custom user agent configuration."""
        provider = SECProvider(
            user_agent="MyApp/1.0 (test@example.com)",
            email="test@example.com"
        )
        headers = provider._get_headers()
        assert "MyApp/1.0" in headers["User-Agent"]

    def test_default_user_agent(self):
        """Test default user agent."""
        provider = SECProvider()
        headers = provider._get_headers()
        assert "qfclient" in headers["User-Agent"]


# =============================================================================
# MarketClient SEC Methods Tests
# =============================================================================

class TestMarketClientSEC:
    """Test SEC methods on MarketClient."""

    def test_client_has_sec_methods(self):
        """MarketClient should have SEC-specific methods."""
        client = MarketClient()
        assert hasattr(client, "get_sec_filings")
        assert hasattr(client, "get_sec_transactions")
        assert hasattr(client, "get_insider_summary")
        assert callable(client.get_sec_filings)
        assert callable(client.get_sec_transactions)
        assert callable(client.get_insider_summary)

    def test_sec_provider_in_registry(self):
        """SEC provider should be in the provider registry."""
        client = MarketClient()
        provider = client._get_provider("sec")
        assert provider.provider_name == "sec"
        assert provider.is_configured() is True


# =============================================================================
# Integration Tests (require network access)
# =============================================================================

@pytest.mark.integration
class TestSECIntegration:
    """
    Integration tests for SEC EDGAR API.

    These tests make real network requests to SEC EDGAR.
    Run with: pytest tests/test_sec.py -v -m integration
    """

    def test_get_insider_summary(self):
        """Test fetching insider summary from SEC."""
        client = MarketClient()
        summary = client.get_insider_summary("AAPL")

        assert isinstance(summary, InsiderSummary)
        assert summary.symbol == "AAPL"
        assert summary.period_start is not None
        assert summary.period_end is not None

    def test_get_sec_transactions(self):
        """Test fetching SEC transactions."""
        client = MarketClient()
        transactions = client.get_sec_transactions("AAPL", limit=5)

        assert hasattr(transactions, "__iter__")
        assert hasattr(transactions, "provider")

    def test_get_sec_filings(self):
        """Test fetching complete Form 4 filings."""
        client = MarketClient()
        filings = client.get_sec_filings("AAPL", limit=5)

        assert hasattr(filings, "__iter__")
        assert hasattr(filings, "provider")

    def test_transaction_has_required_fields(self):
        """Test that transactions have required fields populated."""
        client = MarketClient()
        transactions = client.get_sec_transactions("MSFT", limit=10)

        for txn in transactions:
            assert txn.symbol == "MSFT"
            assert txn.owner_name is not None
            assert txn.transaction_date is not None
            assert txn.shares >= 0
            assert txn.acquired_or_disposed in ("A", "D")

    def test_filing_aggregations(self):
        """Test that filing aggregations are computed correctly."""
        client = MarketClient()
        filings = client.get_sec_filings("GOOGL", limit=5)

        for filing in filings:
            assert filing.symbol == "GOOGL"
            assert filing.transaction_count == len(filing.transactions)
            # Net shares should equal acquired - disposed
            computed_net = filing.total_shares_acquired - filing.total_shares_disposed
            assert filing.net_shares == computed_net

    def test_role_type_populated(self):
        """Test that insider roles have role types."""
        client = MarketClient()
        transactions = client.get_sec_transactions("NVDA", limit=20)

        for txn in transactions:
            assert txn.role is not None
            assert isinstance(txn.role.role_type, InsiderRoleType)
