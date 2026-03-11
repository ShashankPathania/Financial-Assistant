import asyncio
import sys

async def main():
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = await context.new_page()
            
            print("1. Visiting BSE homepage...")
            await page.goto("https://www.bseindia.com/", wait_until="networkidle", timeout=30000)
            print("Title:", await page.title())
            
            search_url = "https://api.bseindia.com/BseIndiaAPI/api/Suggest/v1/SuggestScrip?Type=equity&text=Infosys"
            print(f"2. Searching: {search_url}")
            resp = await page.goto(search_url)
            text = str(await resp.text()).strip()
            print("Search Response Prefix:", text[:500])
            
            await browser.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
