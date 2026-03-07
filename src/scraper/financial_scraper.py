"""
Financial PDF Scraper
=======================
Downloads financial filings from SEC EDGAR using their REST API (no
Playwright needed).  Also supports Playwright-based scraping for other
portals via an isolated subprocess.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

RAW_PDFS_DIR = "data/raw_pdfs"

# SEC EDGAR requires a legitimate User-Agent header
SEC_HEADERS = {
    "User-Agent": "FinanceRAGProject research@financerag.local",
    "Accept-Encoding": "gzip, deflate",
}


class FinancialScraper:
    """
    Multi-strategy financial filing downloader.

    - SEC EDGAR: Uses the SEC REST API (no browser needed)
    - Other portals: Uses Playwright in an isolated subprocess
    """

    def __init__(self, output_dir: str = RAW_PDFS_DIR, headless: bool = True):
        self.output_dir = output_dir
        self.headless = headless
        os.makedirs(output_dir, exist_ok=True)
        logger.info("FinancialScraper initialized (output_dir=%s)", output_dir)

    # ================================================================== #
    #  Main Entry Point
    # ================================================================== #
    def scrape_pdfs(
        self,
        ticker: str,
        year: str = "",
        report_type: str = "10-K",
        portal: str = "SEC_EDGAR",
        max_downloads: int = 5,
        custom_url: str = "",
    ) -> List[str]:
        """
        Download financial filings.

        Args:
            ticker: Company ticker (e.g. 'AAPL').
            year: Filing year (e.g. '2024'). Empty = any year.
            report_type: Filing form type (e.g. '10-K', '10-Q', '8-K').
            portal: 'SEC_EDGAR' or 'PLAYWRIGHT' for generic scraping.
            max_downloads: Maximum files to download.
            custom_url: Custom URL for Playwright-based scraping.

        Returns:
            List of downloaded file paths.
        """
        logger.info("Starting scrape: ticker=%s, year=%s, type=%s, portal=%s",
                     ticker, year, report_type, portal)

        if portal in ("SEC_EDGAR", "SEC_EFTS"):
            return self._scrape_sec_edgar(ticker, year, report_type, max_downloads)
        else:
            return self._scrape_with_playwright(
                ticker, year, report_type, max_downloads, custom_url,
            )

    # ================================================================== #
    #  SEC EDGAR — REST API (Reliable, No Browser Needed)
    # ================================================================== #
    def _scrape_sec_edgar(
        self,
        ticker: str,
        year: str,
        report_type: str,
        max_downloads: int,
    ) -> List[str]:
        """Download filings from SEC EDGAR using their official REST API."""
        downloaded: List[str] = []

        try:
            # ------ Step 1: Map ticker → CIK number ------
            logger.info("Looking up CIK for ticker: %s", ticker)
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(tickers_url, headers=SEC_HEADERS, timeout=15)
            resp.raise_for_status()
            tickers_data = resp.json()

            cik = None
            company_name = ticker
            for entry in tickers_data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    company_name = entry.get("title", ticker)
                    break

            if not cik:
                logger.error("CIK not found for ticker '%s'. Verify the ticker symbol.", ticker)
                return []

            logger.info("Found CIK %s for %s (%s)", cik, ticker, company_name)

            # ------ Step 2: Get company submissions ------
            submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = requests.get(submissions_url, headers=SEC_HEADERS, timeout=15)
            resp.raise_for_status()
            submissions = resp.json()

            recent = submissions.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accession_numbers = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            primary_docs = recent.get("primaryDocument", [])
            primary_desc = recent.get("primaryDocDescription", [])

            if not forms:
                logger.warning("No recent filings found for %s.", ticker)
                return []

            logger.info("Found %d total recent filings for %s.", len(forms), ticker)

            # ------ Step 3: Filter and download matching filings ------
            for i, form in enumerate(forms):
                if len(downloaded) >= max_downloads:
                    break

                # Match form type
                if form.upper() != report_type.upper():
                    continue

                # Match year if specified
                filing_date = filing_dates[i] if i < len(filing_dates) else ""
                if year and not filing_date.startswith(year):
                    continue

                # Build document URL
                accession = accession_numbers[i] if i < len(accession_numbers) else ""
                accession_clean = accession.replace("-", "")
                doc_name = primary_docs[i] if i < len(primary_docs) else ""

                if not accession or not doc_name:
                    continue

                cik_int = int(cik)
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{accession_clean}/{doc_name}"
                )

                # Build local filename
                safe_ticker = ticker.upper().replace("/", "_")
                safe_date = filing_date.replace("-", "")
                ext = Path(doc_name).suffix or ".htm"
                filename = f"{safe_ticker}_{safe_date}_{form}{ext}"
                filepath = os.path.join(self.output_dir, filename)

                # Skip if already downloaded
                if os.path.exists(filepath):
                    logger.info("Already exists, skipping: %s", filename)
                    downloaded.append(filepath)
                    continue

                # Download the document
                logger.info("Downloading filing: %s -> %s", doc_url, filename)
                try:
                    doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=30)
                    doc_resp.raise_for_status()

                    with open(filepath, "wb") as f:
                        f.write(doc_resp.content)

                    size_kb = len(doc_resp.content) / 1024
                    desc = primary_desc[i] if i < len(primary_desc) else ""
                    logger.info(
                        "✓ Downloaded: %s (%.1f KB) — '%s' filed %s",
                        filename, size_kb, desc, filing_date,
                    )
                    downloaded.append(filepath)

                except requests.RequestException as e:
                    logger.error("Failed to download %s: %s", doc_url, e)

        except requests.RequestException as e:
            logger.error("SEC EDGAR API request failed: %s", e)
        except Exception as e:
            logger.error("SEC EDGAR scraping error: %s", e, exc_info=True)

        logger.info("SEC EDGAR scrape complete: %d files downloaded.", len(downloaded))
        return downloaded

    # ================================================================== #
    #  Generic Playwright Scraper (Isolated Subprocess)
    # ================================================================== #
    def _scrape_with_playwright(
        self,
        ticker: str,
        year: str,
        report_type: str,
        max_downloads: int,
        custom_url: str,
    ) -> List[str]:
        """Scrape using Playwright in an isolated subprocess."""
        if not custom_url:
            logger.error("Playwright scraping requires a custom_url.")
            return []

        args = {
            "target_url": custom_url,
            "output_dir": os.path.abspath(self.output_dir),
            "ticker": ticker,
            "year": year,
            "report_type": report_type,
            "max_downloads": max_downloads,
            "headless": self.headless,
        }

        args_file = None
        output_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_args.json", delete=False, dir=tempfile.gettempdir()
            ) as af:
                json.dump(args, af)
                args_file = af.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix="_output.json", delete=False, dir=tempfile.gettempdir()
            ) as of:
                json.dump([], of)
                output_file = of.name

            worker_module = "src.scraper._playwright_worker"
            project_root = str(Path(__file__).parent.parent.parent.resolve())

            logger.info("Launching Playwright subprocess for: %s", custom_url)
            result = subprocess.run(
                [sys.executable, "-m", worker_module, args_file, output_file],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.error("Playwright worker failed: %s",
                             result.stderr[-500:] if result.stderr else "")

            with open(output_file, "r") as f:
                downloaded = json.load(f)

            logger.info("Playwright scrape complete: %d files.", len(downloaded))
            return downloaded

        except subprocess.TimeoutExpired:
            logger.error("Playwright subprocess timed out.")
            return []
        except Exception as e:
            logger.error("Playwright scraping failed: %s", e, exc_info=True)
            return []
        finally:
            for tmp in [args_file, output_file]:
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass


# ------------------------------------------------------------------ #
#  Module-Level Convenience Function
# ------------------------------------------------------------------ #
def scrape_financial_pdfs(
    ticker: str,
    year: str = "",
    report_type: str = "10-K",
    portal: str = "SEC_EDGAR",
    max_downloads: int = 5,
    custom_url: str = "",
) -> List[str]:
    """Convenience function — creates a scraper and runs it."""
    scraper = FinancialScraper()
    return scraper.scrape_pdfs(
        ticker=ticker,
        year=year,
        report_type=report_type,
        portal=portal,
        max_downloads=max_downloads,
        custom_url=custom_url,
    )
