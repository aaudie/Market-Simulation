"""Shared pytest fixtures for qfclient tests."""

import pytest


@pytest.fixture
def market_client():
    """Create a MarketClient instance for testing."""
    from qfclient import MarketClient
    return MarketClient()


@pytest.fixture
def crypto_client():
    """Create a CryptoClient instance for testing."""
    from qfclient import CryptoClient
    return CryptoClient()


@pytest.fixture
def sec_provider():
    """Create an SECProvider instance for testing."""
    from qfclient.market.providers.sec import SECProvider
    return SECProvider()


@pytest.fixture
def sample_insider_role():
    """Create a sample InsiderRole for testing."""
    from qfclient import InsiderRole, InsiderRoleType
    role = InsiderRole(
        is_officer=True,
        is_director=True,
        officer_title="Chief Executive Officer"
    )
    assert role.role_type == InsiderRoleType.CEO
    return role


@pytest.fixture
def sample_transaction():
    """Create a sample SECInsiderTransaction for testing."""
    from datetime import date
    from qfclient import SECInsiderTransaction, TransactionCode, InsiderRole

    role = InsiderRole(is_officer=True, officer_title="CEO")
    return SECInsiderTransaction(
        symbol="AAPL",
        owner_name="Tim Cook",
        owner_cik="0001234567",
        role=role,
        transaction_date=date(2024, 1, 15),
        transaction_code=TransactionCode.PURCHASE,
        shares=10000,
        price_per_share=185.50,
        acquired_or_disposed="A",
        shares_after=1000000,
    )
