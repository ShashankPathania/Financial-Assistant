"""
Playwright Worker Subprocess
===============================
Standalone script that runs Playwright in its own process with a clean
event loop.  Invoked by financial_scraper.py via subprocess.run().

Usage (not called directly by users):
    python -m src.scraper._playwright_worker <json_args_file> <json_output_file>
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("playwright_worker")


async def run_scrape(args: dict) -> List[str]:
    """Perform the actual Playwright-based scraping."""
    from playwright.async_api import async_playwright

    target_url = args["target_url"]
    output_dir = args["output_dir"]
    ticker = args["ticker"]
    year = args["year"]
    report_type = args["report_type"]
    max_downloads = args.get("max_downloads", 5)
    headless = args.get("headless", True)

    downloaded_files: List[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            accept_downloads=True,
        )
        page = await context.new_page()

        # Network interception for PDF discovery
        intercepted_pdfs: Set[str] = set()

        async def handle_response(response):
            ct = response.headers.get("content-type", "")
            if "application/pdf" in ct:
                intercepted_pdfs.add(response.url)

        page.on("response", handle_response)

        # Navigate
        logger.info("Navigating to: %s", target_url)
        try:
            await page.goto(target_url, wait_until="networkidle", timeout=30000)
        except Exception as e:
            logger.warning("Navigation timeout/error (continuing): %s", e)

        await page.wait_for_timeout(2000)

        # Strategy 1: <a> tags with .pdf hrefs
        pdf_links: Set[str] = set()
        try:
            anchors = await page.query_selector_all("a[href]")
            base_url = page.url
            for anchor in anchors:
                href = await anchor.get_attribute("href")
                if href and (href.lower().endswith(".pdf") or ".pdf" in href.lower()):
                    pdf_links.add(urljoin(base_url, href))
        except Exception as e:
            logger.error("Error scanning PDF links: %s", e)

        logger.info("Found %d PDF links in page.", len(pdf_links))

        # Strategy 2: <iframe> with embedded PDFs
        try:
            iframes = await page.query_selector_all("iframe")
            for iframe in iframes:
                src = await iframe.get_attribute("src")
                if src and (".pdf" in src.lower() or "viewer" in src.lower()):
                    parsed = urlparse(src)
                    if "url=" in src:
                        import urllib.parse
                        params = urllib.parse.parse_qs(parsed.query)
                        if "url" in params:
                            pdf_links.add(params["url"][0])
                        elif "file" in params:
                            pdf_links.add(params["file"][0])
                    else:
                        pdf_links.add(urljoin(page.url, src))
        except Exception as e:
            logger.error("Error scanning iframes: %s", e)

        # Strategy 3: Network-intercepted PDFs
        pdf_links.update(intercepted_pdfs)
        logger.info("Total PDF URLs found: %d", len(pdf_links))

        # Download
        for url in list(pdf_links)[:max_downloads]:
            filepath = await _download_pdf(page, url, output_dir, ticker, year, report_type)
            if filepath:
                downloaded_files.append(filepath)

        await browser.close()

    return downloaded_files


async def _download_pdf(page, url, output_dir, ticker, year, report_type) -> Optional[str]:
    """Download and validate a single PDF."""
    filename = f"{ticker}_{year}_{report_type}_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
    filepath = os.path.join(output_dir, filename)

    try:
        logger.info("Downloading: %s -> %s", url, filename)
        dl_page = await page.context.new_page()
        try:
            response = await dl_page.goto(url, timeout=30000)
            if response:
                content = await response.body()

                # Failsafe: detect HTML
                if content[:50].strip().lower().startswith(b"<!doctype") or \
                   content[:50].strip().lower().startswith(b"<html"):
                    logger.warning("Downloaded file is HTML, not PDF: %s", filename)
                    await dl_page.close()
                    return None

                # Validate PDF magic bytes
                if b"%PDF" not in content[:100]:
                    logger.error("Not a PDF. Skipping: %s", filename)
                    await dl_page.close()
                    return None

                with open(filepath, "wb") as f:
                    f.write(content)

                logger.info("Downloaded: %s (%d bytes)", filename, len(content))
                await dl_page.close()
                return filepath

        except Exception as e:
            logger.error("Download failed: %s", e)
            await dl_page.close()
            return None

    except Exception as e:
        logger.error("Error: %s", e)
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.scraper._playwright_worker <args.json> <output.json>")
        sys.exit(1)

    args_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(args_file, "r") as f:
        args = json.load(f)

    # Run with a fresh ProactorEventLoop on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        files = asyncio.run(run_scrape(args))
    except Exception as e:
        logger.error("Scraping failed: %s", e, exc_info=True)
        files = []

    with open(output_file, "w") as f:
        json.dump(files, f)


if __name__ == "__main__":
    main()
