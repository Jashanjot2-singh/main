# Swiggy Google Play Reviews – Topic Trend Analysis
## Overview

This project is built to analyze real user reviews from the Swiggy Android app on Google Play Store and convert them into clean, meaningful topic trends over time.

The main goal was to align closely with the assignment requirements, while also keeping the system realistic, scalable, and accurate.

Because of limited Gemini API tokens and rate limits, the analysis is intentionally scoped to 4 specific days:

- 6 January 2026
- 5 January 2026
- 4 January 2026
- 3 January 2026

Even with this limitation, the dataset is still large and real — roughly 1300+ real user reviews were available across these dates, which is enough to demonstrate a production-style workflow.

### Why only 4 days?

- Gemini has rate limits and token constraints
- Processing a full 30–60 day window with LLM calls would not be practical in this setup
- Instead of using dummy data, I chose to:
  - Use real scraped reviews
  - Limit the time window
  - Keep the logic identical to a full-scale system

This makes the solution realistic, honest, and aligned with real-world constraints, which is exactly what the assessment focuses on.

## Tech Stack Used

- Playwright – for browser automation (Google Play Store is dynamic)

- BeautifulSoup – for parsing review HTML

- Pandas – for data processing and Excel handling

- Google Gemini (GenAI) – for intelligent topic classification

- Python – core implementation language

### High-Level Strategy

The system is split into two clear phases:

- Data Collection (Scraping)

- Topic Classification & Trend Analysis

This separation keeps the system clean, debuggable, and easy to scale later.

## Phase 1: Collecting Reviews from Google Play
#### What happens here?

1) Open Swiggy’s Google Play Store page using Playwright

2) Click “See all reviews”

3) Sort reviews by Newest

4) Scroll to load as many reviews as possible

#### Extract only:
- date
- feedback text

#### Why Playwright + BeautifulSoup?

1) Google Play Store content is JS-rendered

2) Static scraping alone doesn’t work

3) Playwright handles the browser behavior

4) BeautifulSoup keeps parsing simple and readable

### Output of Phase 1

All extracted reviews are stored locally as an Excel file for clarity and transparency:

``` date | feedback```


Using Excel here makes the data:

1) Easy to inspect

2) Easy to debug

3) Easy for reviewers to understand

## Phase 2: Reading Reviews & Preparing Data

1) The Excel file is loaded using pandas
2) Dates are normalized:
   - Example: 06-Jan-26 → 6 January 2026

3) Reviews are filtered day by day:
   - First process 3 January
   - Then 4 January
   - Then 5 January
   - Finally 6 January

Each day is processed independently, which matches a real daily batch processing system.

## Phase 3: Batch Processing

For each day:

- All feedback for that day is collected

- Feedback is split into batches of 30–50 reviews

- This avoids token overload

- Keeps Gemini calls stable and predictable

Batching also mirrors how this would work in a real pipeline.

## Phase 4: Topic Classification using Gemini
### Initial Topics

The system starts with a small predefined topic list, for example:

- Delivery issue

- Food stale

- Delivery partner rude

- Maps not working properly

- Instamart should be open all night

### How Gemini is used

For each batch, Gemini receives:

1) The list of feedback

2) The current topic list

3) A very strict prompt

The prompt enforces:

- Positive-only feedback → ignored

- Food-related problems → always mapped to “Food stale”

- Pricing / discount / value issues → mapped to a single price topic

- App, login, map, UI issues → mapped to app performance

- No semantic duplicates allowed

- New topics are created only if nothing fits

Gemini returns one JSON object per batch, containing counts per topic, not per review.

## Phase 5: Counting & Topic Growth

For each batch:
- JSON response is parsed

- Counts are added to integers

- If Gemini introduces a new topic:

    - It’s added to the global topic list

    - Used in all future batches and days

Important detail:

- Topic list grows globally

- Counts reset daily

This avoids topic fragmentation while still discovering new themes.

## Phase 6: Final Output

At the end, a final Excel report is generated:

- Rows → Topics

- Columns → Dates (3–6 January 2026)

- Values → Number of feedback per topic per day

This output:

- Is clean and readable

- Contains no duplicate topics

- Shows clear daily trends

- Matches what the assignment expects

#### In One Line

Real reviews are scraped → stored in Excel → processed day by day → classified in batches using Gemini with strict rules → counted into clean topics → exported as a final topic-vs-date Excel trend report.

## What I’d improve next (if I had more time / budget)
#### 1) Support the full window (June 2024 → till date) + true T-30 report 

Right now I demonstrated the pipeline on 4 days because of Gemini rate limits. Next I’d:
- scrape/store reviews continuously (or in chunks)
- accept any target date T
- generate the final trend table for T-30 … T automatically

This makes it fully spec-compliant and production-ready.

#### 2) Replace Excel intermediate storage with a database

- Excel is great for debugging and reviewer visibility, but in a real system I’d switch to:
   - SQLite/Postgres for raw reviews
   separate tables for: reviews, topics, review→topic mapping, daily topic counts
   That makes updates, backfills, and trend queries much easier and faster.

#### 3) Add a “Topic Manager” layer (stronger dedup than prompt-only)

My prompt already prevents many semantic duplicates, but next I’d add a dedicated step:

- embeddings-based similarity check against existing topics

- automatic merge suggestions (e.g., “Food quality issue” vs “Food stale”)

- keep a canonical taxonomy + aliases

This improves trend quality over time and reduces topic drift.

#### 4) Multi-label classification (optional but more accurate)

Some reviews mention multiple issues (delivery late + refund + rude agent).
Next improvement:

- allow up to 2–3 topics per review

- count all applicable topics

This increases recall and makes trends more realistic.






