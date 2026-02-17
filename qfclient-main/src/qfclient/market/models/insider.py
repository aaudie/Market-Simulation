"""
Insider trading models for SEC Form 4 data.

Provides comprehensive Pydantic models for insider transactions with
rich data extraction including position changes and role classification.
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class TransactionCode(str, Enum):
    """SEC Form 4 transaction codes."""

    PURCHASE = "P"  # Open market purchase
    SALE = "S"  # Open market sale
    ACQUISITION = "A"  # Grant, award, or other acquisition
    DISPOSITION = "D"  # Disposition to issuer
    CONVERSION = "C"  # Conversion of derivative
    EXERCISE = "M"  # Exercise of derivative
    GIFT = "G"  # Gift
    INHERITANCE = "I"  # Inheritance
    DISCRETIONARY = "L"  # Small acquisition under Rule 16a-6
    EXEMPT = "W"  # Acquisition or disposition pursuant to a tender offer
    PLAN = "F"  # Tax withholding
    OTHER = "U"  # Other


class OwnershipType(str, Enum):
    """Direct or indirect ownership."""

    DIRECT = "D"
    INDIRECT = "I"


class InsiderRoleType(str, Enum):
    """
    Insider role classification.

    Raw enum for role type - users can define their own importance scoring.
    """
    CEO = "ceo"
    CFO = "cfo"
    COO = "coo"
    OTHER_OFFICER = "other_officer"
    DIRECTOR = "director"
    TEN_PERCENT_OWNER = "ten_percent_owner"
    OTHER = "other"


class InsiderRole(BaseModel):
    """
    Insider role information from SEC Form 4.

    Provides raw role flags as reported in the filing.
    Users can implement their own importance scoring based on these flags.
    """

    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    officer_title: Optional[str] = None

    def is_ceo(self) -> bool:
        """Check if insider is CEO based on title."""
        if not self.officer_title:
            return False
        title = self.officer_title.lower()
        return "chief executive" in title or "ceo" in title

    def is_cfo(self) -> bool:
        """Check if insider is CFO based on title."""
        if not self.officer_title:
            return False
        title = self.officer_title.lower()
        return "chief financial" in title or "cfo" in title

    def is_coo(self) -> bool:
        """Check if insider is COO based on title."""
        if not self.officer_title:
            return False
        title = self.officer_title.lower()
        return "chief operating" in title or "coo" in title

    def role_type(self) -> InsiderRoleType:
        """
        Get the primary role type as an enum.

        Returns the most senior role if multiple apply.
        """
        if self.is_ceo():
            return InsiderRoleType.CEO
        if self.is_cfo():
            return InsiderRoleType.CFO
        if self.is_coo():
            return InsiderRoleType.COO
        if self.is_officer:
            return InsiderRoleType.OTHER_OFFICER
        if self.is_director:
            return InsiderRoleType.DIRECTOR
        if self.is_ten_percent_owner:
            return InsiderRoleType.TEN_PERCENT_OWNER
        return InsiderRoleType.OTHER

    def role_description(self) -> str:
        """Human-readable role description."""
        roles = []
        if self.is_ceo():
            roles.append("CEO")
        elif self.is_cfo():
            roles.append("CFO")
        elif self.is_coo():
            roles.append("COO")
        elif self.is_officer and self.officer_title:
            roles.append(self.officer_title)
        elif self.is_officer:
            roles.append("Officer")

        if self.is_director:
            roles.append("Director")
        if self.is_ten_percent_owner:
            roles.append("10% Owner")

        return ", ".join(roles) if roles else "Insider"


class InsiderTransaction(BaseModel):
    """
    A single insider transaction from SEC Form 4.

    Contains all fields parsed from the Form 4 XML including
    computed fields for position changes and trade direction.
    """

    # Issuer (company) information
    symbol: str = Field(..., description="Trading symbol")
    issuer_cik: Optional[str] = Field(None, description="Issuer CIK number")
    company_name: Optional[str] = Field(None, description="Company name")

    # Reporting owner (insider) information
    owner_name: str = Field(..., description="Name of the reporting owner")
    owner_cik: Optional[str] = Field(None, description="Owner's CIK number")
    role: InsiderRole = Field(default_factory=InsiderRole, description="Insider's role")

    # Transaction details
    transaction_date: date = Field(..., description="Date of the transaction")
    transaction_code: TransactionCode = Field(..., description="Transaction type code")
    shares: float = Field(..., description="Number of shares transacted")
    price_per_share: float = Field(0.0, description="Price per share")
    acquired_or_disposed: str = Field(
        ..., description="A=acquired, D=disposed"
    )

    # Derivative flag
    is_derivative: bool = Field(False, description="Whether this is a derivative transaction")

    # Ownership details
    ownership_type: OwnershipType = Field(
        OwnershipType.DIRECT, description="Direct or indirect ownership"
    )
    indirect_ownership_nature: Optional[str] = Field(
        None, description="Nature of indirect ownership"
    )

    # Post-transaction position
    shares_after: Optional[float] = Field(
        None, description="Shares owned after transaction"
    )

    # Filing metadata
    filing_date: Optional[date] = Field(None, description="Date the form was filed")
    accession_number: Optional[str] = Field(None, description="SEC accession number")

    def notional_value(self) -> float:
        """Total dollar value of the transaction."""
        return self.shares * self.price_per_share

    def is_purchase(self) -> bool:
        """Whether this is a purchase (acquiring shares)."""
        return self.acquired_or_disposed == "A" and self.transaction_code in (
            TransactionCode.PURCHASE,
            TransactionCode.ACQUISITION,
        )

    def is_sale(self) -> bool:
        """Whether this is a sale (disposing shares)."""
        return self.acquired_or_disposed == "D" and self.transaction_code in (
            TransactionCode.SALE,
            TransactionCode.DISPOSITION,
        )

    def is_open_market(self) -> bool:
        """Whether this is an open market transaction (P or S code)."""
        return self.transaction_code in (
            TransactionCode.PURCHASE,
            TransactionCode.SALE,
        )

    def shares_before(self) -> Optional[float]:
        """Shares owned before transaction (computed from shares_after)."""
        if self.shares_after is None:
            return None

        if self.acquired_or_disposed == "A":
            return self.shares_after - self.shares
        elif self.acquired_or_disposed == "D":
            return self.shares_after + self.shares
        return None

    def position_change_pct(self) -> Optional[float]:
        """
        Percentage change in position.

        A key signal from research - larger position changes
        from insiders with small holdings are often more informative.
        """
        shares_before = self.shares_before()
        if shares_before is None or shares_before <= 0:
            return None

        if self.shares_after is None:
            return None

        return ((self.shares_after - shares_before) / shares_before) * 100

    def net_shares(self) -> float:
        """Net shares change (positive for buys, negative for sells)."""
        if self.acquired_or_disposed == "A":
            return self.shares
        elif self.acquired_or_disposed == "D":
            return -self.shares
        return 0


class Form4Filing(BaseModel):
    """
    A complete SEC Form 4 filing.

    Aggregates all transactions from a single filing with
    summary statistics.
    """

    # Filing identification
    accession_number: str = Field(..., description="SEC accession number")
    filing_date: date = Field(..., description="Date the form was filed")

    # Issuer information
    symbol: str = Field(..., description="Trading symbol")
    issuer_cik: Optional[str] = Field(None, description="Issuer CIK")
    company_name: Optional[str] = Field(None, description="Company name")

    # Owner information
    owner_name: str = Field(..., description="Reporting owner name")
    owner_cik: Optional[str] = Field(None, description="Owner CIK")
    role: InsiderRole = Field(default_factory=InsiderRole)

    # Transactions in this filing
    transactions: list[InsiderTransaction] = Field(
        default_factory=list, description="All transactions in this filing"
    )

    def total_shares_acquired(self) -> float:
        """Total shares acquired in this filing."""
        return sum(
            t.shares for t in self.transactions if t.acquired_or_disposed == "A"
        )

    def total_shares_disposed(self) -> float:
        """Total shares disposed in this filing."""
        return sum(
            t.shares for t in self.transactions if t.acquired_or_disposed == "D"
        )

    def net_shares(self) -> float:
        """Net change in shares (positive = net buyer)."""
        return self.total_shares_acquired - self.total_shares_disposed

    def total_value_acquired(self) -> float:
        """Total value of shares acquired."""
        return sum(
            t.notional_value()
            for t in self.transactions
            if t.acquired_or_disposed == "A"
        )

    def total_value_disposed(self) -> float:
        """Total value of shares disposed."""
        return sum(
            t.notional_value()
            for t in self.transactions
            if t.acquired_or_disposed == "D"
        )

    def net_value(self) -> float:
        """Net dollar value change."""
        return self.total_value_acquired() - self.total_value_disposed()

    def is_net_buyer(self) -> bool:
        """Whether the insider is a net buyer in this filing."""
        return self.net_shares() > 0

    def has_open_market_transactions(self) -> bool:
        """Whether this filing contains open market purchases/sales."""
        return any(t.is_open_market() for t in self.transactions)

    def transaction_count(self) -> int:
        """Number of transactions in this filing."""
        return len(self.transactions)


class InsiderSummary(BaseModel):
    """
    Summary of insider trading activity for a symbol.

    Aggregates multiple filings to provide an overview of
    insider sentiment.
    """

    symbol: str
    period_start: date
    period_end: date

    # Counts
    total_filings: int = 0
    total_transactions: int = 0
    unique_insiders: int = 0

    # Volume
    total_shares_bought: float = 0.0
    total_shares_sold: float = 0.0
    total_value_bought: float = 0.0
    total_value_sold: float = 0.0

    # Buyer/seller counts
    num_buyers: int = 0
    num_sellers: int = 0

    # Open market only
    open_market_shares_bought: float = 0.0
    open_market_shares_sold: float = 0.0
    open_market_value_bought: float = 0.0
    open_market_value_sold: float = 0.0

    def net_shares(self) -> float:
        """Net shares change."""
        return self.total_shares_bought - self.total_shares_sold

    def net_value(self) -> float:
        """Net dollar value change."""
        return self.total_value_bought - self.total_value_sold

    def buy_sell_ratio(self) -> Optional[float]:
        """Ratio of buyers to total traders."""
        total = self.num_buyers + self.num_sellers
        if total == 0:
            return None
        return self.num_buyers / total

    def net_purchase_ratio(self) -> Optional[float]:
        """
        Net Purchase Ratio (Lakonishok & Lee 2001).

        NPR = (Buy shares - Sell shares) / (Buy shares + Sell shares)
        Ranges from -1 (all selling) to +1 (all buying).
        """
        total = self.open_market_shares_bought + self.open_market_shares_sold
        if total == 0:
            return None
        return (self.open_market_shares_bought - self.open_market_shares_sold) / total

    def insider_sentiment(self) -> str:
        """Categorical insider sentiment."""
        npr = self.net_purchase_ratio()
        if npr is None:
            return "neutral"
        if npr > 0.3:
            return "bullish"
        elif npr > 0:
            return "slightly_bullish"
        elif npr > -0.3:
            return "slightly_bearish"
        else:
            return "bearish"
