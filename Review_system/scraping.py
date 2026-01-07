from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

URL = "https://play.google.com/store/apps/details?id=in.swiggy.android&hl=en_IN"
WAIT_SECONDS = 120
TARGET_COUNT = 2000
OUTFILE = "swiggy_reviews_manual_scroll_200.xlsx"


def extract_reviews(html: str):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for block in soup.select("div.RHo1pe"):
        header = block.select_one("header.c1bOId")
        review_id = header.get("data-review-id") if header else None

        date_tag = block.select_one("span.bp9Aid")
        feedback_tag = block.select_one("div.h3YV2d")

        if not date_tag or not feedback_tag:
            continue

        date_text = date_tag.get_text(strip=True)
        feedback = feedback_tag.get_text(strip=True)

        # Make a fallback unique key if review id missing
        rid = review_id or (date_text + "::" + feedback[:80])

        rows.append({"review_id": rid, "date": date_text, "feedback": feedback})

    return rows


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(URL, timeout=60000)
    page.wait_for_load_state("networkidle")

    # Click "See all reviews"
    page.get_by_role("button", name=re.compile("See all reviews", re.I)).click(timeout=20000)
    page.wait_for_selector("div[role='dialog']", timeout=20000)
    time.sleep(1)

    # Open sort dropdown ("Most relevant") and select "Newest"
    try:
        page.locator("div[role='dialog'] span:has-text('Most relevant')").first.click(force=True, timeout=8000)
    except Exception:
        page.locator("div[role='dialog'] :has-text('Most relevant')").first.click(force=True, timeout=8000)

    time.sleep(0.5)
    page.locator("span[role='menuitemradio']:has-text('Newest')").first.click(force=True, timeout=10000)
    time.sleep(2)

    print(f" Now manually scroll in the opened browser for {WAIT_SECONDS} seconds to load more reviews...")
    time.sleep(WAIT_SECONDS)

    html = page.content()
    browser.close()

# Parse after your manual scrolling
all_rows = extract_reviews(html)

# Deduplicate
dedup = {}
for r in all_rows:
    dedup[r["review_id"]] = {"date": r["date"], "feedback": r["feedback"]}

final_rows = list(dedup.values())[:TARGET_COUNT]

df = pd.DataFrame(final_rows)
df.to_excel(OUTFILE, index=False)

print(f" Extracted {len(df)} reviews and saved to {OUTFILE}")
