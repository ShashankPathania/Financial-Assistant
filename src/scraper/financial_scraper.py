"""
Financial PDF Scraper
=======================
Downloads financial filings from multiple international portals:
  - SEC EDGAR (US) — REST API
  - Companies House (UK) — REST API
  - BSE India — HTTP scraping
  - Direct URL — Download any PDF/HTML from a URL

Also supports Playwright-based scraping for other portals via subprocess.
"""

import hashlib
import json
import logging
import os
import re
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

# Generic headers for other portals
GENERIC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class FinancialScraper:
    """
    Multi-strategy financial filing downloader.

    Supported portals:
      - SEC_EDGAR: US filings via SEC REST API
      - COMPANIES_HOUSE: UK filings via Companies House REST API
      - BSE_INDIA: Indian filings via BSE website
      - DIRECT_URL: Download any URL directly
      - PLAYWRIGHT: Generic browser-based scraping via subprocess
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
        company_number: str = "",
    ) -> List[str]:
        """
        Download financial filings.

        Args:
            ticker: Company ticker (e.g. 'AAPL') or name.
            year: Filing year (e.g. '2024'). Empty = any year.
            report_type: Filing form type (e.g. '10-K', '10-Q', '8-K', 'annual-report').
            portal: 'SEC_EDGAR', 'COMPANIES_HOUSE', 'BSE_INDIA', 'DIRECT_URL', or 'PLAYWRIGHT'.
            max_downloads: Maximum files to download.
            custom_url: URL for direct download or Playwright scraping.
            company_number: UK company number for Companies House.

        Returns:
            List of downloaded file paths.
        """
        logger.info("Starting scrape: ticker=%s, year=%s, type=%s, portal=%s",
                     ticker, year, report_type, portal)

        if portal == "SEC_EDGAR":
            return self._scrape_sec_edgar(ticker, year, report_type, max_downloads)
        elif portal == "COMPANIES_HOUSE":
            return self._scrape_companies_house(ticker, year, report_type, max_downloads, company_number)
        elif portal == "BSE_INDIA":
            return self._scrape_moneycontrol_india(ticker, year, report_type, max_downloads)
        elif portal == "DIRECT_URL":
            return self._download_direct_url(custom_url, ticker)
        else:
            return self._scrape_with_playwright(
                ticker, year, report_type, max_downloads, custom_url,
            )

    # ================================================================== #
    #  SEC EDGAR — REST API (US)
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

                if form.upper() != report_type.upper():
                    continue

                filing_date = filing_dates[i] if i < len(filing_dates) else ""
                if year and not filing_date.startswith(year):
                    continue

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

                safe_ticker = ticker.upper().replace("/", "_")
                safe_date = filing_date.replace("-", "")
                ext = Path(doc_name).suffix or ".htm"
                filename = f"{safe_ticker}_{safe_date}_{form}{ext}"
                filepath = os.path.join(self.output_dir, filename)

                if os.path.exists(filepath):
                    logger.info("Already exists, skipping: %s", filename)
                    downloaded.append(filepath)
                    continue

                logger.info("Downloading filing: %s -> %s", doc_url, filename)
                try:
                    doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=30)
                    doc_resp.raise_for_status()

                    with open(filepath, "wb") as f:
                        f.write(doc_resp.content)

                    # Save metadata sidecar for the parser to use (e.g. for image fetching)
                    meta_path = f"{filepath}.meta.json"
                    try:
                        with open(meta_path, "w", encoding="utf-8") as mf:
                            json.dump({
                                "source_url": doc_url,
                                "ticker": ticker,
                                "filing_date": filing_date,
                                "form": form
                            }, mf, indent=2)
                    except Exception as me:
                        logger.warning("Could not save metadata sidecar for %s: %s", filename, me)

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
    #  Companies House — REST API (UK)
    # ================================================================== #
    def _scrape_companies_house(
        self,
        company_name: str,
        year: str,
        report_type: str,
        max_downloads: int,
        company_number: str = "",
    ) -> List[str]:
        """
        Download filings from UK Companies House.

        Uses the free Companies House API. If no company_number is provided,
        searches by company name first.
        """
        downloaded: List[str] = []
        api_key = os.getenv("COMPANIES_HOUSE_API_KEY", "")

        if not api_key:
            logger.warning(
                "No COMPANIES_HOUSE_API_KEY set. "
                "Register free at https://developer.company-information.service.gov.uk "
                "Trying direct download approach..."
            )
            # Fallback: try direct filing page without API
            return self._scrape_companies_house_direct(company_name, year, max_downloads, company_number)

        auth = (api_key, "")  # Basic auth with API key as username

        try:
            # Step 1: Resolve company number
            if not company_number:
                search_url = "https://api.company-information.service.gov.uk/search/companies"
                resp = requests.get(
                    search_url,
                    params={"q": company_name, "items_per_page": 5},
                    auth=auth,
                    timeout=15,
                )
                resp.raise_for_status()
                results = resp.json().get("items", [])

                if not results:
                    logger.error("No UK companies found for '%s'.", company_name)
                    return []

                # Take first match
                company_number = results[0].get("company_number", "")
                resolved_name = results[0].get("title", company_name)
                logger.info("Resolved '%s' → %s (%s)", company_name, company_number, resolved_name)

            # Step 2: Get filing history
            filings_url = f"https://api.company-information.service.gov.uk/company/{company_number}/filing-history"
            resp = requests.get(
                filings_url,
                params={"items_per_page": 50, "category": "accounts"},
                auth=auth,
                timeout=15,
            )
            resp.raise_for_status()
            filings = resp.json().get("items", [])

            if not filings:
                logger.warning("No filings found for company %s.", company_number)
                return []

            logger.info("Found %d filing(s) for company %s.", len(filings), company_number)

            # Step 3: Download matching filings
            for filing in filings:
                if len(downloaded) >= max_downloads:
                    break

                filing_date = filing.get("date", "")
                if year and not filing_date.startswith(year):
                    continue

                # Get document metadata
                doc_links = filing.get("links", {})
                doc_url = doc_links.get("document_metadata", "")

                if not doc_url:
                    continue

                # Fetch document metadata to get download URL
                if not doc_url.startswith("http"):
                    doc_url = f"https://api.company-information.service.gov.uk{doc_url}"

                try:
                    meta_resp = requests.get(doc_url, auth=auth, timeout=15)
                    meta_resp.raise_for_status()
                    doc_meta = meta_resp.json()

                    # Get first available document format
                    resources = doc_meta.get("resources", {})
                    download_url = None
                    ext = ".pdf"

                    for content_type, resource in resources.items():
                        if "application/pdf" in content_type:
                            download_url = doc_meta.get("links", {}).get("document", "")
                            ext = ".pdf"
                            break
                        elif "application/xhtml" in content_type:
                            download_url = doc_meta.get("links", {}).get("document", "")
                            ext = ".html"
                            break

                    if not download_url:
                        continue

                    if not download_url.startswith("http"):
                        download_url = f"https://document-api.company-information.service.gov.uk{download_url}"

                    # Download
                    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", company_name)[:30]
                    safe_date = filing_date.replace("-", "")
                    filename = f"UK_{safe_name}_{safe_date}_accounts{ext}"
                    filepath = os.path.join(self.output_dir, filename)

                    if os.path.exists(filepath):
                        downloaded.append(filepath)
                        continue

                    doc_resp = requests.get(
                        download_url,
                        auth=auth,
                        headers={"Accept": "application/pdf,application/xhtml+xml,*/*"},
                        timeout=30,
                    )
                    doc_resp.raise_for_status()

                    with open(filepath, "wb") as f:
                        f.write(doc_resp.content)

                    logger.info("✓ Downloaded UK filing: %s (%.1f KB)", filename, len(doc_resp.content) / 1024)
                    downloaded.append(filepath)

                except Exception as e:
                    logger.error("Failed to download UK filing: %s", e)

        except requests.RequestException as e:
            logger.error("Companies House API failed: %s", e)
        except Exception as e:
            logger.error("Companies House scraping error: %s", e, exc_info=True)

        logger.info("Companies House scrape complete: %d files.", len(downloaded))
        return downloaded

    def _scrape_companies_house_direct(
        self,
        company_name: str,
        year: str,
        max_downloads: int,
        company_number: str,
    ) -> List[str]:
        """Fallback: scrape Companies House without API key via web."""
        logger.info("Attempting direct Companies House web scrape for '%s'", company_name)

        if not company_number:
            logger.error(
                "Company number required for Companies House without API key. "
                "Please provide the UK company number (e.g. '01234567')."
            )
            return []

        # Try to download the filing history page
        url = f"https://find-and-update.company-information.service.gov.uk/company/{company_number}/filing-history"
        logger.info("Companies House filing history URL: %s", url)

        # Use Playwright for dynamic pages
        return self._scrape_with_playwright(
            company_name, year, "accounts", max_downloads, url,
        )

    # ================================================================== #
    #  Moneycontrol India — HTTP Scraping
    # ================================================================== #
    def _scrape_moneycontrol_india(
        self,
        company_name: str,
        year: str,
        report_type: str,
        max_downloads: int,
    ) -> List[str]:
        """
        Download annual reports for Indian companies via Moneycontrol.
        
        BSE India blocks automated scraping aggressively. Moneycontrol 
        provides the exact same Annual Report PDFs through accessible APIs.
        """
        downloaded: List[str] = []
        logger.info("Starting Moneycontrol scrape for '%s'", company_name)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        try:
            # Step 1: Search Moneycontrol for the company Link/Scrip ID
            search_url = f"https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php?query={requests.utils.quote(company_name)}&type=1&format=json"
            search_resp = requests.get(search_url, headers=headers, timeout=15)
            search_resp.raise_for_status()
            
            results = search_resp.json()
            if not results or not isinstance(results, list):
                logger.error("No Moneycontrol results for '%s'.", company_name)
                return []
                
            match = results[0]
            link_src = match.get("link_src", "")
            resolved_name = match.get("pdt_dis_nm", company_name)
            sc_id = match.get("sc_id", "")
            
            if not link_src or not sc_id:
                logger.error("Moneycontrol search match missing data: %s", match)
                return []
                
            logger.info("Moneycontrol resolved '%s' -> %s (%s)", company_name, resolved_name, sc_id)
            
            # Step 2: Build Annual Reports Page URL
            # The stock URL looks like: https://www.moneycontrol.com/india/stockpricequote/computers-software/infosys/IT
            # The Annual Reports URL is: https://www.moneycontrol.com/financials/infosys/annual-report/IT
            parts = link_src.strip("/").split("/")
            if len(parts) < 2:
                logger.error("Unrecognized Moneycontrol link format: %s", link_src)
                return []
                
            stock_code = parts[-2]
            annual_report_url = f"https://www.moneycontrol.com/financials/{stock_code}/annual-report/{sc_id}"
            logger.info("Fetching Annual Reports page: %s", annual_report_url)
            
            reports_resp = session.get(annual_report_url, timeout=15)
            reports_resp.raise_for_status()
            
            # Step 3: Parse PDF links using BeautifulSoup
            soup = BeautifulSoup(reports_resp.content, "html.parser")
            
            # Find the table containing reports
            tables = soup.find_all("table", class_="mctable1")
            if not tables:
                logger.warning("No financial tables found on Moneycontrol page.")
                return []
                
            # Iterate and find PDF links
            pdf_links = []
            for row in tables[0].find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    td_title = cells[0].text.strip()
                    td_link = cells[1].find("a", href=True)
                    if td_link and ".pdf" in td_link["href"].lower():
                        pdf_links.append((td_title, td_link["href"]))
                        
            if not pdf_links:
                logger.warning("No PDF report links found on Moneycontrol.")
                return []
                
            logger.info("Found %d annual report PDF(s) on Moneycontrol.", len(pdf_links))
            
            # Step 4: Download the matched PDFs
            for title, pdf_url in pdf_links:
                if len(downloaded) >= max_downloads:
                    break
                    
                # Title usually looks like "Mar 2024" or "Dec 2023"
                if year and year not in title:
                    continue
                    
                if not pdf_url.startswith("http"):
                    pdf_url = f"https://www.moneycontrol.com{pdf_url}"
                    
                # Clean title for filename
                safe_title = re.sub(r"[^a-zA-Z0-9]", "", title)
                safe_name = re.sub(r"[^a-zA-Z0-9]", "_", resolved_name)[:20]
                filename = f"MC_{safe_name}_{safe_title}_annual_report.pdf"
                filepath = os.path.join(self.output_dir, filename)
                
                if os.path.exists(filepath):
                    downloaded.append(filepath)
                    continue
                    
                logger.info("Downloading Moneycontrol report: %s -> %s", pdf_url, filename)
                try:
                    doc_resp = session.get(pdf_url, timeout=30)
                    doc_resp.raise_for_status()
                    
                    with open(filepath, "wb") as f:
                        f.write(doc_resp.content)
                        
                    logger.info("✓ Downloaded Moneycontrol filing: %s (%.1f KB)", filename, len(doc_resp.content) / 1024)
                    downloaded.append(filepath)
                except Exception as e:
                    logger.error("Failed to download Moneycontrol report: %s", e)
                    
        except requests.RequestException as e:
            logger.error("Moneycontrol scraping request failed: %s", e)
        except Exception as e:
            logger.error("Moneycontrol scraping error: %s", e, exc_info=True)
            
        logger.info("Moneycontrol scrape complete: %d files.", len(downloaded))
        return downloaded

    # ================================================================== #
    #  Direct URL Download
    # ================================================================== #
    def _download_direct_url(self, url: str, label: str = "document") -> List[str]:
        """Download a file from any URL (PDF, HTML, etc.)."""
        if not url:
            logger.error("No URL provided for direct download.")
            return []

        try:
            logger.info("Direct download: %s", url)
            resp = requests.get(url, headers=GENERIC_HEADERS, timeout=60, stream=True)
            resp.raise_for_status()

            # Determine filename from URL or Content-Disposition header
            content_disp = resp.headers.get("Content-Disposition", "")
            if "filename=" in content_disp:
                filename = content_disp.split("filename=")[-1].strip('"').strip("'")
            else:
                url_path = url.split("?")[0].split("#")[0]
                filename = Path(url_path).name or "downloaded_document"

            # Ensure the file has an extension
            if not Path(filename).suffix:
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" in content_type:
                    filename += ".pdf"
                elif "html" in content_type:
                    filename += ".html"
                else:
                    filename += ".pdf"

            # Sanitize filename
            safe_label = re.sub(r"[^a-zA-Z0-9]", "_", label)[:20]
            filename = f"{safe_label}_{filename}"
            filepath = os.path.join(self.output_dir, filename)

            if os.path.exists(filepath):
                logger.info("File already exists: %s", filename)
                return [filepath]

            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = os.path.getsize(filepath) / 1024
            logger.info("✓ Direct download complete: %s (%.1f KB)", filename, size_kb)
            return [filepath]

        except requests.RequestException as e:
            logger.error("Direct URL download failed: %s", e)
            return []
        except Exception as e:
            logger.error("Direct download error: %s", e, exc_info=True)
            return []

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
        portal: str = "",
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
            "portal": portal,
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

            logger.info("Launching Playwright subprocess for: %s (portal: %s)", custom_url, portal)
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
    company_number: str = "",
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
        company_number=company_number,
    )
