"""
SEC EDGAR provider for Form 4 insider trading data.

Fetches and parses SEC Form 4 filings directly from EDGAR.
No API key required, but requires proper User-Agent header.

Rate limits: 10 requests/second (SEC enforced)
Features: Insider transactions, Form 4 filings, insider summaries
"""

import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Optional

from lxml import etree

from ...common.base import ProviderError, ResultList
from ..models.insider import (
    Form4Filing,
    InsiderRole,
    InsiderSummary,
    InsiderTransaction,
    OwnershipType,
    TransactionCode,
)
from .base import BaseProvider

logger = logging.getLogger(__name__)

# Cache for ticker -> CIK mapping
_TICKER_CIK_CACHE: dict[str, str] = {}


def _text(el) -> Optional[str]:
    """Extract text from an XML element safely."""
    return el.text.strip() if el is not None and el.text else None


def _float(el, default: float = 0.0) -> float:
    """Extract float from an XML element safely."""
    text = _text(el)
    if text is None:
        return default
    try:
        return float(text)
    except (ValueError, TypeError):
        return default


def _date(el) -> Optional[date]:
    """Extract date from an XML element safely."""
    text = _text(el)
    if text is None:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


class SECProvider(BaseProvider):
    """
    SEC EDGAR provider for Form 4 insider trading data.

    Fetches Form 4 filings directly from SEC EDGAR and parses
    them into structured Pydantic models with rich data extraction.

    Features:
    - Insider transaction details (shares, price, date)
    - Position change calculations (shares before/after, % change)
    - Insider role classification (CEO, CFO, Director, 10% owner)
    - Insider importance scoring based on research
    - Open market vs grant/award classification

    Rate limiting: SEC requires <= 10 requests/second.
    User-Agent: SEC requires company/email format.
    """

    provider_name = "sec"
    base_url = "https://www.sec.gov"

    # SEC requires specific User-Agent format
    SEC_USER_AGENT = "qfclient/0.1 (github.com/oregonquantgroup/qfclient)"

    # Regex to extract XML from .txt filings
    XML_DOC_PATTERN = re.compile(
        r"(<\?xml.*?</ownershipDocument>)", re.DOTALL | re.IGNORECASE
    )

    def __init__(self, user_agent: Optional[str] = None, email: Optional[str] = None):
        """
        Initialize SEC provider.

        Args:
            user_agent: Custom User-Agent string (SEC requires company/email format)
            email: Contact email for SEC compliance
        """
        super().__init__()

        # Build User-Agent header as SEC requires
        email = email or os.getenv("SEC_CONTACT_EMAIL", "user@example.com")
        if user_agent:
            self._user_agent = user_agent
        else:
            self._user_agent = f"qfclient/0.1 ({email})"

        self._last_request_time = 0.0

    def is_configured(self) -> bool:
        """SEC EDGAR is always available (no API key required)."""
        return True

    def _get_headers(self) -> dict[str, str]:
        """Get headers with SEC-compliant User-Agent."""
        return {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/xml, text/xml, */*",
        }

    def _rate_limit(self) -> None:
        """Enforce SEC rate limit (10 requests/second)."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 0.12:  # 100ms + 20ms buffer
            time.sleep(0.12 - elapsed)
        self._last_request_time = time.time()

    def _get_cik_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Look up the CIK (Central Index Key) for a ticker symbol.

        Uses SEC's company_tickers.json file which maps tickers to CIKs.
        Results are cached to avoid repeated lookups.
        """
        global _TICKER_CIK_CACHE

        ticker = ticker.upper()

        # Check cache first
        if ticker in _TICKER_CIK_CACHE:
            return _TICKER_CIK_CACHE[ticker]

        # Load the ticker -> CIK mapping from SEC
        try:
            content = self.get("/files/company_tickers.json")
            data = json.loads(content.decode("utf-8"))

            # Build cache from response
            # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            # Note: cik_str can be int or str depending on SEC API version
            for entry in data.values():
                t = str(entry.get("ticker", "")).upper()
                cik = entry.get("cik_str", "")
                if t and cik:
                    _TICKER_CIK_CACHE[t] = str(cik).zfill(10)  # Pad to 10 digits

            return _TICKER_CIK_CACHE.get(ticker)

        except Exception as e:
            logger.warning(f"Failed to load SEC ticker mapping: {e}")
            return None

    def get(self, path: str, params: dict | None = None) -> bytes:
        """
        Make a GET request to SEC EDGAR.

        Returns raw bytes to handle both text and binary responses.
        """
        import httpx

        self._rate_limit()

        url = f"{self.base_url}{path}"
        try:
            response = httpx.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=30.0,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            raise ProviderError(self.provider_name, f"HTTP {e.response.status_code}: {url}")
        except httpx.RequestError as e:
            raise ProviderError(self.provider_name, f"Request failed: {e}")

    # =========================================================================
    # Provider capability flags
    # =========================================================================

    @property
    def supports_quotes(self) -> bool:
        return False

    @property
    def supports_ohlcv(self) -> bool:
        return False

    @property
    def supports_company_profile(self) -> bool:
        return False

    @property
    def supports_insider_transactions(self) -> bool:
        return True

    # =========================================================================
    # Master index methods
    # =========================================================================

    def _master_idx_url(self, dt: date) -> str:
        """Get URL for daily master index file."""
        quarter = (dt.month - 1) // 3 + 1
        return (
            f"/Archives/edgar/daily-index/{dt.year}/QTR{quarter}/"
            f"master.{dt.strftime('%Y%m%d')}.idx"
        )

    def _parse_master_index(self, content: bytes) -> list[dict]:
        """Parse a master index file and return Form 4 rows."""
        rows = []
        lines = content.decode("utf-8", errors="ignore").split("\n")

        # Skip header until separator
        in_data = False
        for line in lines:
            if line.startswith("----"):
                in_data = True
                continue
            if not in_data:
                continue

            parts = line.strip().split("|")
            if len(parts) != 5:
                continue

            cik, company, form_type, date_filed, file_name = parts
            if form_type.strip() == "4":
                rows.append({
                    "cik": cik.strip(),
                    "company": company.strip(),
                    "date_filed": date_filed.strip(),
                    "file_name": file_name.strip(),
                })

        return rows

    def _fetch_form4_xml(self, file_name: str) -> Optional[bytes]:
        """
        Fetch Form 4 XML from SEC EDGAR.

        Tries direct XML URL first, then falls back to extracting
        from the .txt filing.
        """
        # Try direct XML endpoint first
        if file_name.endswith(".txt"):
            xml_path = "/Archives/" + file_name[:-4] + ".xml"
        else:
            xml_path = "/Archives/" + file_name

        try:
            content = self.get(xml_path)
            if b"<ownershipDocument" in content:
                return content
        except ProviderError:
            pass

        # Fallback: fetch .txt and extract XML
        txt_path = "/Archives/" + file_name
        try:
            content = self.get(txt_path)
            match = self.XML_DOC_PATTERN.search(content.decode("utf-8", errors="ignore"))
            if match:
                return match.group(1).encode("utf-8")
        except ProviderError:
            pass

        return None

    # =========================================================================
    # XML parsing
    # =========================================================================

    def _parse_form4_xml(
        self,
        xml_content: bytes,
        filing_date: Optional[date] = None,
        accession_number: Optional[str] = None,
    ) -> Optional[Form4Filing]:
        """
        Parse Form 4 XML into a Form4Filing model.

        Extracts all transaction data with rich metadata.
        """
        try:
            tree = etree.parse(BytesIO(xml_content))
        except Exception as e:
            logger.warning("Failed to parse Form 4 XML: %s", e)
            return None

        # Issuer information
        issuer_sym = _text(tree.find(".//issuerTradingSymbol"))
        issuer_cik = _text(tree.find(".//issuerCik"))
        issuer_name = _text(tree.find(".//issuerName"))

        if not issuer_sym:
            return None

        # Reporting owner information
        owner_name = _text(
            tree.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
        ) or ""
        owner_cik = _text(
            tree.find(".//reportingOwner/reportingOwnerId/rptOwnerCik")
        )

        # Parse owner relationship/role
        is_director = False
        is_officer = False
        is_ten_percent = False
        officer_title = None

        for rel in tree.findall(".//reportingOwner/reportingOwnerRelationship"):
            if (_text(rel.find("isDirector")) or "").lower() == "true":
                is_director = True
            if (_text(rel.find("isOfficer")) or "").lower() == "true":
                is_officer = True
            if (_text(rel.find("isTenPercentOwner")) or "").lower() == "true":
                is_ten_percent = True
            title = _text(rel.find("officerTitle"))
            if title:
                officer_title = title

        role = InsiderRole(
            is_director=is_director,
            is_officer=is_officer,
            is_ten_percent_owner=is_ten_percent,
            officer_title=officer_title,
        )

        transactions: list[InsiderTransaction] = []

        # Parse non-derivative transactions
        for txn in tree.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
            transaction = self._parse_transaction(
                txn,
                symbol=issuer_sym,
                issuer_cik=issuer_cik,
                company_name=issuer_name,
                owner_name=owner_name,
                owner_cik=owner_cik,
                role=role,
                is_derivative=False,
                filing_date=filing_date,
                accession_number=accession_number,
            )
            if transaction:
                transactions.append(transaction)

        # Parse derivative transactions
        for txn in tree.findall(".//derivativeTable/derivativeTransaction"):
            transaction = self._parse_transaction(
                txn,
                symbol=issuer_sym,
                issuer_cik=issuer_cik,
                company_name=issuer_name,
                owner_name=owner_name,
                owner_cik=owner_cik,
                role=role,
                is_derivative=True,
                filing_date=filing_date,
                accession_number=accession_number,
            )
            if transaction:
                transactions.append(transaction)

        if not transactions:
            return None

        return Form4Filing(
            accession_number=accession_number or "",
            filing_date=filing_date or date.today(),
            symbol=issuer_sym.upper(),
            issuer_cik=issuer_cik,
            company_name=issuer_name,
            owner_name=owner_name,
            owner_cik=owner_cik,
            role=role,
            transactions=transactions,
        )

    def _parse_transaction(
        self,
        txn,
        symbol: str,
        issuer_cik: Optional[str],
        company_name: Optional[str],
        owner_name: str,
        owner_cik: Optional[str],
        role: InsiderRole,
        is_derivative: bool,
        filing_date: Optional[date],
        accession_number: Optional[str],
    ) -> Optional[InsiderTransaction]:
        """Parse a single transaction element from Form 4 XML."""
        # Transaction date
        txn_date = _date(txn.find(".//transactionDate/value"))
        if not txn_date:
            return None

        # Transaction code
        code_text = _text(txn.find(".//transactionCoding/transactionCode"))
        try:
            code = TransactionCode(code_text) if code_text else TransactionCode.OTHER
        except ValueError:
            code = TransactionCode.OTHER

        # Shares and price
        shares = _float(txn.find(".//transactionAmounts/transactionShares/value"))
        price = _float(txn.find(".//transactionAmounts/transactionPricePerShare/value"))

        # Acquired or disposed
        acq_disp = _text(
            txn.find(".//transactionAmounts/transactionAcquiredDisposedCode/value")
        ) or ""

        # Shares after transaction
        shares_after = _float(
            txn.find(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
            default=None,
        )

        # Ownership type
        ownership_text = _text(
            txn.find(".//ownershipNature/directOrIndirectOwnership/value")
        )
        try:
            ownership = (
                OwnershipType(ownership_text)
                if ownership_text
                else OwnershipType.DIRECT
            )
        except ValueError:
            ownership = OwnershipType.DIRECT

        indirect_nature = _text(
            txn.find(".//ownershipNature/natureOfOwnership/value")
        )

        return InsiderTransaction(
            symbol=symbol.upper(),
            issuer_cik=issuer_cik,
            company_name=company_name,
            owner_name=owner_name,
            owner_cik=owner_cik,
            role=role,
            transaction_date=txn_date,
            transaction_code=code,
            shares=shares,
            price_per_share=price,
            acquired_or_disposed=acq_disp,
            is_derivative=is_derivative,
            ownership_type=ownership,
            indirect_ownership_nature=indirect_nature,
            shares_after=shares_after,
            filing_date=filing_date,
            accession_number=accession_number,
        )

    # =========================================================================
    # Public API methods
    # =========================================================================

    def get_insider_transactions(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 100,
    ) -> ResultList[InsiderTransaction]:
        """
        Get insider transactions for a symbol.

        Fetches Form 4 filings from SEC EDGAR and extracts all
        transactions with rich metadata.

        Args:
            symbol: Stock symbol
            start: Start date (default: 90 days ago)
            end: End date (default: today)
            limit: Maximum number of transactions to return

        Returns:
            ResultList of InsiderTransaction objects
        """
        if end is None:
            end = date.today()
        if start is None:
            start = end - timedelta(days=90)

        filings = self._fetch_form4_filings(symbol, start, end)
        transactions = ResultList(provider=self.provider_name)

        for filing in filings:
            for txn in filing.transactions:
                transactions.append(txn)
                if len(transactions) >= limit:
                    break
            if len(transactions) >= limit:
                break

        return transactions

    def get_form4_filings(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        limit: int = 50,
    ) -> ResultList[Form4Filing]:
        """
        Get Form 4 filings for a symbol.

        Returns complete filings with all transactions aggregated.

        Args:
            symbol: Stock symbol
            start: Start date (default: 90 days ago)
            end: End date (default: today)
            limit: Maximum number of filings to return

        Returns:
            ResultList of Form4Filing objects
        """
        if end is None:
            end = date.today()
        if start is None:
            start = end - timedelta(days=90)

        filings = self._fetch_form4_filings(symbol, start, end, limit=limit)
        result = ResultList(provider=self.provider_name)
        for filing in filings:
            result.append(filing)
        return result

    def get_insider_summary(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> InsiderSummary:
        """
        Get a summary of insider trading activity for a symbol.

        Aggregates all Form 4 filings to compute summary statistics
        like net purchase ratio and insider sentiment.

        Args:
            symbol: Stock symbol
            start: Start date (default: 90 days ago)
            end: End date (default: today)

        Returns:
            InsiderSummary with aggregated statistics
        """
        if end is None:
            end = date.today()
        if start is None:
            start = end - timedelta(days=90)

        filings = self._fetch_form4_filings(symbol, start, end)

        summary = InsiderSummary(
            symbol=symbol.upper(),
            period_start=start,
            period_end=end,
        )

        unique_owners = set()
        buyers = set()
        sellers = set()

        for filing in filings:
            summary.total_filings += 1
            unique_owners.add(filing.owner_cik or filing.owner_name)

            for txn in filing.transactions:
                summary.total_transactions += 1

                if txn.acquired_or_disposed == "A":
                    summary.total_shares_bought += txn.shares
                    summary.total_value_bought += txn.notional_value
                    if txn.is_open_market:
                        summary.open_market_shares_bought += txn.shares
                        summary.open_market_value_bought += txn.notional_value
                        buyers.add(filing.owner_cik or filing.owner_name)
                elif txn.acquired_or_disposed == "D":
                    summary.total_shares_sold += txn.shares
                    summary.total_value_sold += txn.notional_value
                    if txn.is_open_market:
                        summary.open_market_shares_sold += txn.shares
                        summary.open_market_value_sold += txn.notional_value
                        sellers.add(filing.owner_cik or filing.owner_name)

        summary.unique_insiders = len(unique_owners)
        summary.num_buyers = len(buyers)
        summary.num_sellers = len(sellers)

        return summary

    def _fetch_form4_filings(
        self,
        symbol: str,
        start: date,
        end: date,
        limit: int = 100,
    ) -> list[Form4Filing]:
        """
        Fetch Form 4 filings for a symbol within a date range.

        Uses SEC's company submissions API to efficiently find Form 4 filings
        for the company, then fetches and parses each Form 4 XML.
        """
        symbol = symbol.upper()
        filings: list[Form4Filing] = []

        # Look up the CIK for this ticker
        cik = self._get_cik_for_ticker(symbol)
        if not cik:
            logger.warning(f"Could not find CIK for ticker {symbol}")
            return filings

        # Fetch the company's filing history from SEC
        try:
            submissions_url = f"/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count={limit * 2}&output=atom"
            content = self.get(submissions_url)

            # Parse the Atom feed to get Form 4 filing URLs
            # The feed contains entries with links to each filing
            from xml.etree import ElementTree as ET

            root = ET.fromstring(content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entries = root.findall(".//atom:entry", ns)

            for entry in entries:
                if len(filings) >= limit:
                    break

                # Get filing date
                updated = entry.find("atom:updated", ns)
                if updated is not None and updated.text:
                    try:
                        filing_date = datetime.fromisoformat(
                            updated.text.replace("Z", "+00:00")
                        ).date()
                    except ValueError:
                        continue

                    # Filter by date range
                    if filing_date < start or filing_date > end:
                        continue

                # Get the filing link
                link = entry.find("atom:link[@rel='alternate']", ns)
                if link is None:
                    continue

                href = link.get("href", "")
                if not href:
                    continue

                # Extract accession number from URL
                # URL format: https://www.sec.gov/Archives/edgar/data/CIK/ACCESSION/...
                accession_match = re.search(r"/(\d{10}-\d{2}-\d{6})", href)
                if not accession_match:
                    continue

                accession = accession_match.group(1)
                accession_path = accession.replace("-", "")

                # Construct the Form 4 XML URL
                # Try primary document format first
                xml_url = f"/Archives/edgar/data/{cik}/{accession_path}/{accession}.txt"

                xml_content = self._fetch_form4_xml_from_path(xml_url, cik, accession_path)
                if not xml_content:
                    continue

                filing = self._parse_form4_xml(
                    xml_content,
                    filing_date=filing_date,
                    accession_number=accession,
                )

                if filing and filing.symbol.upper() == symbol:
                    filings.append(filing)

        except Exception as e:
            logger.warning(f"Error fetching Form 4 filings for {symbol}: {e}")

        return filings

    def _fetch_form4_xml_from_path(
        self, txt_path: str, cik: str, accession_path: str
    ) -> Optional[bytes]:
        """
        Fetch Form 4 XML content from SEC EDGAR.

        Tries multiple approaches:
        1. Direct XML file in the filing directory
        2. Extract from the .txt filing document
        """
        # Try to find the XML file in the filing directory
        try:
            # List the filing directory to find the XML file
            index_url = f"/Archives/edgar/data/{cik}/{accession_path}/index.json"
            content = self.get(index_url)
            index_data = json.loads(content.decode("utf-8"))

            # Look for the primary XML document
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith(".xml") and "primary_doc" in name.lower():
                    xml_url = f"/Archives/edgar/data/{cik}/{accession_path}/{name}"
                    xml_content = self.get(xml_url)
                    if b"<ownershipDocument" in xml_content:
                        return xml_content

            # If no primary_doc, try any XML file
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith(".xml"):
                    xml_url = f"/Archives/edgar/data/{cik}/{accession_path}/{name}"
                    try:
                        xml_content = self.get(xml_url)
                        if b"<ownershipDocument" in xml_content:
                            return xml_content
                    except ProviderError:
                        continue

        except Exception:
            pass

        # Fallback: try the .txt file and extract XML
        try:
            txt_content = self.get(txt_path)
            match = self.XML_DOC_PATTERN.search(txt_content.decode("utf-8", errors="ignore"))
            if match:
                return match.group(1).encode("utf-8")
        except Exception:
            pass

        return None
