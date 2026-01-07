import os
import json
import time
import difflib
from typing import List, Dict, Any, Iterable

import pandas as pd
from google import genai
from google.genai import types

INPUT_EXCEL = "swiggy_reviews_manual_scroll_200.xlsx"
DATE_COL = "date"
FEEDBACK_COL = "feedback"

TARGET_DATE_LABELS = [
    "6 January 2026",
    "5 January 2026",
    "4 January 2026",
    "3 January 2026",
]

BATCH_SIZE = 50
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SEED_TOPICS = [
    "Delivery issue",
    "Food stale",
    "Delivery partner rude",
    "Maps not working properly",
    "Instamart should be open all night",
    "Bring back 10 minute bolt delivery",
]

OUTPUT_EXCEL = "final_topic_trend_3_to_6_jan_2026_2.xlsx"


def batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_excel_as_df(path: str) -> pd.DataFrame:
    """
    Always returns a DataFrame even if the Excel has multiple sheets.
    """
    obj = pd.read_excel(path, sheet_name=None)
    if isinstance(obj, dict):
        first_key = list(obj.keys())[0]
        return obj[first_key]
    return obj


def normalize_date_to_label(series: pd.Series) -> pd.Series:
    """
    Handles '06-Jan-26' and '6 January 2026' -> '6 January 2026'
    """
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt.dt.strftime("%d %B %Y").str.lstrip("0")


def canonicalize_topic(name: str, topic_list: List[str], cutoff: float = 0.90) -> str:
    """
    Small dedup: if a new topic is very similar to an existing one, merge it.
    """
    if not name:
        return ""
    n = name.strip()

    for t in topic_list:
        if t.lower() == n.lower():
            return t

    close = difflib.get_close_matches(n, topic_list, n=1, cutoff=cutoff)
    if close:
        return close[0]

    return n


def gemini_count_batch(
    client: genai.Client,
    feedback_batch: List[str],
    topic_list: List[str],
) -> Dict[str, Any]:
    """
    Returns ONE dict like:
    {
      "Delivery issue": 2,
      "Food stale": 5,
      ...
      "new_topics": {"Refund not received": 3, "App crash": 1}
    }
    """
    prompt = f"""
You are a strict JSON-only classifier and counter.

Predefined topics (use EXACTLY these strings, do NOT rename them):
{json.dumps(topic_list, ensure_ascii=False)}

CRITICAL CANONICAL MAPPING RULES (MUST FOLLOW):
- ANY food-related complaint (food stale, bad quality, taste issue, cold food, low quantity,
  damaged food, wrong food item, packaging issue) MUST be counted ONLY under:
  "Food stale"
  → Do NOT create any new food-related topic.

- ANY price / money related complaint (high charges, expensive, not worth money,
  discount issue, coupon issue, offers missing, pricing problem) MUST be counted ONLY under:
  "Not value for money"

- ANY payment, refund, wallet, COD related complaint MUST be counted ONLY under:
  "Payment/Refund issue"

- ANY app / technical issue (login, crash, slow app, map issue, UI problem) MUST be counted ONLY under:
  "App performance issue"

- ANY customer support related issue (unhelpful staff, automated replies,
  language problem, restaurant communication) MUST be counted ONLY under:
  "Unhelpful staff/app"

IMPORTANT:
Before creating a new topic, you MUST first try to map the feedback to ONE of the predefined topics
using semantic similarity and the above rules.
ONLY create a new topic if NONE of the predefined topics logically fit.

TASK:
You will be given a list of feedback strings.
Your job is to COUNT how many feedbacks belong to each predefined topic.

RULES:
1) If feedback is ONLY positive / appreciation / praise / no complaint → IGNORE it completely.
2) If feedback matches or is semantically similar to a predefined topic → count it there.
3) DO NOT create semantic duplicates of existing topics.
4) New topics must be short, generic, and non-overlapping.

OUTPUT FORMAT (VERY IMPORTANT):
Return ONLY ONE plain JSON object with:
- ALL predefined topics as keys (even if count is 0)
- Values must be integers
- One extra key "new_topics" whose value is a JSON object:
  {{"topic name": integer}}
  (use empty {{}} if no new topics)

NO explanation.
NO markdown.
NO extra text.
ONLY valid JSON.

Inputs:
{json.dumps(feedback_batch, ensure_ascii=False)}
""".strip()

    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    data = json.loads(resp.text)
    if not isinstance(data, dict):
        raise ValueError("Gemini did not return a JSON object.")
    if "new_topics" not in data or not isinstance(data["new_topics"], dict):
        # enforce structure
        data["new_topics"] = {}
    return data


def main():
    api_key = ""
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY environment variable first.")

    client = genai.Client(api_key=api_key)

    df = load_excel_as_df(INPUT_EXCEL)

    if DATE_COL not in df.columns or FEEDBACK_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{DATE_COL}' and '{FEEDBACK_COL}'. Found: {list(df.columns)}"
        )

    df["_date_label"] = normalize_date_to_label(df[DATE_COL])
    df[FEEDBACK_COL] = df[FEEDBACK_COL].astype(str).fillna("").map(lambda x: x.strip())

    # Global topics list grows (seed + discovered)
    topics: List[str] = list(SEED_TOPICS)

    # day -> dict(topic->count)
    counts_by_day: Dict[str, Dict[str, int]] = {}

    for day_label in TARGET_DATE_LABELS:
        day_feedbacks = df.loc[df["_date_label"] == day_label, FEEDBACK_COL].tolist()
        day_feedbacks = [x for x in day_feedbacks if x]

        print(f"\n=== {day_label} ===")
        print(f"Feedback rows found: {len(day_feedbacks)}")

        # Reset counts for the day (but include all known topics so far)
        day_counts: Dict[str, int] = {t: 0 for t in topics}

        for batch_idx, fb_batch in enumerate(batches(day_feedbacks, BATCH_SIZE), start=1):
            # Ensure day_counts has any new topics discovered earlier
            for t in topics:
                day_counts.setdefault(t, 0)

            print(f"Batch {batch_idx}: {len(fb_batch)} items | topics={len(topics)}")

            # Gemini (retry once)
            try:
                result = gemini_count_batch(client, fb_batch, topics)
            except Exception as e:
                print(f"Gemini failed: {e} -> retrying once...")
                time.sleep(2)
                result = gemini_count_batch(client, fb_batch, topics)

            # 1) Add counts for predefined topics
            for t in topics:
                val = result.get(t, 0)
                try:
                    day_counts[t] += int(val)
                except Exception:
                    pass

            # 2) Handle new topics
            new_topics_obj = result.get("new_topics", {}) or {}
            if isinstance(new_topics_obj, dict):
                for new_name_raw, cnt in new_topics_obj.items():
                    try:
                        cnt_int = int(cnt)
                    except Exception:
                        continue
                    if cnt_int <= 0:
                        continue

                    new_name = canonicalize_topic(str(new_name_raw), topics)

                    # Prevent food duplicates (extra safety)
                    if "food" in new_name.lower():
                        # force into Food stale
                        day_counts["Food stale"] = day_counts.get("Food stale", 0) + cnt_int
                        continue

                    if new_name and new_name not in topics:
                        topics.append(new_name)

                    day_counts.setdefault(new_name, 0)
                    day_counts[new_name] += cnt_int

        counts_by_day[day_label] = day_counts

    # Build final report: rows=topics, columns=dates
    report = pd.DataFrame(0, index=topics, columns=TARGET_DATE_LABELS)
    for day_label in TARGET_DATE_LABELS:
        dc = counts_by_day.get(day_label, {})
        for t in topics:
            report.loc[t, day_label] = int(dc.get(t, 0))

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="trend_report")
        pd.DataFrame({"topic": topics}).to_excel(writer, sheet_name="all_topics", index=False)

    print(f"\n Saved final report to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()

