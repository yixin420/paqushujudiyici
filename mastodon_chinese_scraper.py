#!/usr/bin/env python3
"""
Mastodon Chinese Posts Weekly Scraper
=====================================

This script collects the top 100 public Chinese-language posts (Simplified + Traditional)
per ISO week from a Mastodon instance within a specified UTC time range. It respects
Mastodon API rate limits, supports optional Bearer token authentication, and outputs
weekly JSON files containing the required fields only.

Default configuration targets https://m.cmx.im.

Usage:
    python mastodon_chinese_scraper.py

Adjust global constants near the top of this file to change the domain, time range,
weekly limit, output directory, or authentication token. See the README or the
`print_usage_instructions` function for detailed guidance.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
from requests import RequestException
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout
from bs4 import BeautifulSoup
import schedule


# -----------------------------
# Configuration (edit as needed)
# -----------------------------
BASE_URL = "https://m.cmx.im"
# UTC time range for scraping
START_TIME_UTC = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME_UTC = datetime(2025, 11, 9, 23, 59, 59, tzinfo=timezone.utc)
# Weekly post cap
WEEKLY_LIMIT = 100
# Directory to store JSON outputs
OUTPUT_DIR = "data"
# Optional Mastodon API Bearer token; set to None for anonymous access
AUTH_BEARER_TOKEN: Optional[str] = None
# Daily scheduler UTC trigger time (24-hour format)
SCHEDULED_UTC_HOUR = 23
SCHEDULED_UTC_MINUTE = 55
# Request settings
REQUEST_LIMIT_PER_CALL = 40
REQUEST_RETRIES = 3
REQUEST_RETRY_DELAY_SECONDS = 2
MIN_REQUEST_INTERVAL_SECONDS = 1
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 10
MIN_PAGE_LIMIT = 5


CHINESE_PATTERN = re.compile(
    r"[\u4e00-\u9fa5\u814a-\u9fff\u3105-\u312f\u3400-\u4dbf]"
)


@dataclass(frozen=True)
class WeekWindow:
    """Represents a single ISO week window."""

    iso_year: int
    iso_week: int
    start: datetime
    end: datetime

    @property
    def key(self) -> str:
        return f"{self.iso_year}_week{self.iso_week:02d}"

    @property
    def filename(self) -> str:
        return f"{self.key}_chinese_posts.json"


class MastodonChineseScraper:
    """Collects weekly top N Chinese-language posts from a Mastodon instance."""

    def __init__(
        self,
        base_url: str,
        start_time: datetime,
        end_time: datetime,
        weekly_limit: int = 100,
        output_dir: str = "data",
        auth_token: Optional[str] = None,
    ) -> None:
        if start_time.tzinfo is None or end_time.tzinfo is None:
            raise ValueError("start_time and end_time must be timezone-aware (UTC).")
        if start_time > end_time:
            raise ValueError("start_time must be earlier than end_time.")

        self.base_url = base_url.rstrip("/")
        self.start_time = start_time
        self.end_time = end_time
        self.weekly_limit = weekly_limit
        self.auth_token = auth_token
        self.output_dir = output_dir

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "MastodonChineseScraper/1.0 "
                    "(https://github.com/cursor-ai/mastodon-chinese-scraper)"
                )
            }
        )
        if self.auth_token:
            self.session.headers["Authorization"] = f"Bearer {self.auth_token}"

        os.makedirs(self.output_dir, exist_ok=True)
        self._last_request_ts: float = 0.0
        # Some instances reject the maximum limit=40; start with the configured value
        # and reduce dynamically if we encounter 422 errors.
        self.page_limit: int = REQUEST_LIMIT_PER_CALL

    # -----------------------------
    # Public entry points
    # -----------------------------

    def run(self) -> Dict[str, List[dict]]:
        """
        Execute the scraper and return a mapping of week key to collected posts.
        JSON files are saved for each week encountered within the time range.
        """
        week_windows = self._prepare_week_windows()
        remaining_weeks = {w.key for w in week_windows}
        weekly_posts: Dict[str, List[dict]] = {w.key: [] for w in week_windows}
        week_lookup: Dict[str, WeekWindow] = {w.key: w for w in week_windows}
        seen_ids: set[str] = set()

        max_id: Optional[str] = None
        reached_start = False

        while not reached_start and remaining_weeks:
            statuses = self._fetch_public_timeline_page(max_id=max_id)
            if not statuses:
                break

            max_id = statuses[-1]["id"]
            for status in statuses:
                created_at = self._parse_datetime(status.get("created_at"))
                if created_at is None:
                    continue

                if created_at > self.end_time:
                    # Newer than our window; skip but continue processing.
                    continue
                if created_at < self.start_time:
                    # We've gone past our desired start time; we can exit after this page.
                    reached_start = True
                    continue

                if status.get("visibility") != "public":
                    continue

                post_id = status.get("id")
                if not post_id or post_id in seen_ids:
                    continue

                plain_content = self._html_to_text(status.get("content", ""))
                if not self._contains_chinese(plain_content):
                    continue

                week_key = self._week_key_for_datetime(created_at)
                if week_key not in remaining_weeks:
                    continue

                # Prepare record with required fields only.
                account = status.get("account", {}) or {}
                record = {
                    "id": post_id,
                    "account_id": account.get("id"),
                    "username": account.get("username"),
                    "display_name": self._html_to_text(account.get("display_name", "")),
                    "note": self._html_to_text(account.get("note", "")),
                    "created_at": status.get("created_at"),
                    "url": status.get("url"),
                    "content": plain_content,
                }

                weekly_posts[week_key].append(record)
                seen_ids.add(post_id)

                if len(weekly_posts[week_key]) >= self.weekly_limit:
                    remaining_weeks.discard(week_key)

            # Safety: sort each week's posts in descending chronological order.
            for key, posts in weekly_posts.items():
                posts.sort(key=lambda item: item["created_at"], reverse=True)

        self._save_weekly_outputs(weekly_posts, week_lookup)
        return weekly_posts

    def print_sample_posts(
        self, weekly_posts: Dict[str, List[dict]], sample_size: int = 5
    ) -> None:
        """Print up to `sample_size` posts across all weeks for quick inspection."""
        flattened: List[Tuple[str, dict]] = []
        for week_key, posts in weekly_posts.items():
            for post in posts:
                flattened.append((week_key, post))

        flattened.sort(
            key=lambda pair: pair[1]["created_at"] or "", reverse=True
        )
        print(f"\nSample of up to {sample_size} Chinese-language posts:")
        for week_key, post in flattened[:sample_size]:
            created_at = post.get("created_at", "")
            username = post.get("username", "")
            display = post.get("display_name", "")
            print(
                f"[{week_key}] {created_at} — {username} ({display})\n"
                f"URL: {post.get('url', '')}\n"
                f"Content: {post.get('content', '')}\n"
                "-" * 60
            )

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _fetch_public_timeline_page(
        self, max_id: Optional[str] = None
    ) -> List[dict]:
        """Fetch a single page from the public timeline with retries and rate limiting."""
        url = f"{self.base_url}/api/v1/timelines/public"
        params = {"limit": self.page_limit}
        if max_id:
            params["max_id"] = max_id

        for attempt in range(REQUEST_RETRIES):
            self._respect_rate_limit()
            try:
                response = self.session.get(url, params=params, timeout=30)
            except (Timeout, RequestsConnectionError) as exc:
                print(
                    f"Network timeout/connection error on attempt {attempt + 1} "
                    f"of {REQUEST_RETRIES}: {exc}. Retrying in "
                    f"{REQUEST_RETRY_DELAY_SECONDS}s..."
                )
                time.sleep(REQUEST_RETRY_DELAY_SECONDS)
                continue
            except RequestException as exc:
                raise RuntimeError(f"Request failed due to an unexpected error: {exc}") from exc

            if response.status_code == 200:
                self._update_rate_limit_state(response.headers)
                return response.json()

            if response.status_code == 401:
                raise PermissionError(
                    "Received 401 Unauthorized. Please configure AUTH_BEARER_TOKEN."
                )

            if response.status_code == 429:
                wait_seconds = self._wait_seconds_from_rate_limit(response.headers)
                print(
                    f"Rate limit exceeded (429). Waiting {wait_seconds:.1f}s "
                    "before retrying..."
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code >= 500:
                print(
                    f"Server error (status {response.status_code}). "
                    f"Retrying in {REQUEST_RETRY_DELAY_SECONDS}s..."
                )
                time.sleep(REQUEST_RETRY_DELAY_SECONDS)
                continue

            if response.status_code == 404:
                raise RuntimeError(
                    "Endpoint /api/v1/timelines/public not available on this instance."
                )

            if response.status_code == 503:
                print("Service unavailable (503). Retrying after short delay...")
                time.sleep(REQUEST_RETRY_DELAY_SECONDS)
                continue

            if response.status_code == 422:
                # Some instances restrict the maximum allowed limit value.
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"error": response.text}

                detail_str = json.dumps(payload, ensure_ascii=False)

                if self.page_limit > MIN_PAGE_LIMIT:
                    self.page_limit = max(self.page_limit // 2, MIN_PAGE_LIMIT)
                    params["limit"] = self.page_limit
                    print(
                        "Received 422 response, likely due to request limits. "
                        f"Reducing per-request limit to {self.page_limit} and retrying..."
                    )
                    time.sleep(MIN_REQUEST_INTERVAL_SECONDS)
                    continue

                raise RuntimeError(
                    "Received 422 Unprocessable Entity response even with minimal "
                    f"page size ({self.page_limit}). Details: {detail_str}"
                )

            # For other status codes, raise the error immediately.
            response.raise_for_status()

        raise RuntimeError("Failed to fetch public timeline after multiple attempts.")

    def _respect_rate_limit(self) -> None:
        """Ensure at least MIN_REQUEST_INTERVAL_SECONDS between requests."""
        elapsed = time.time() - self._last_request_ts
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    def _update_rate_limit_state(self, headers: dict) -> None:
        """Update the last request timestamp and optionally log remaining quota."""
        self._last_request_ts = time.time()
        remaining = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")
        if remaining is not None and reset is not None:
            try:
                reset_time = datetime.fromtimestamp(int(reset), tz=timezone.utc)
                print(
                    f"Remaining requests: {remaining}, "
                    f"reset at {reset_time.isoformat()}"
                )
            except (ValueError, OSError):
                print(f"Remaining requests: {remaining}")

    def _wait_seconds_from_rate_limit(self, headers: dict) -> float:
        """Determine backoff duration after a 429 response."""
        reset_header = headers.get("X-RateLimit-Reset")
        if reset_header:
            try:
                reset_epoch = int(reset_header)
                now_epoch = int(time.time())
                wait = max(reset_epoch - now_epoch, DEFAULT_RATE_LIMIT_BACKOFF_SECONDS)
                return float(wait)
            except (ValueError, OSError):
                pass
        return float(DEFAULT_RATE_LIMIT_BACKOFF_SECONDS)

    def _save_weekly_outputs(
        self,
        weekly_posts: Dict[str, List[dict]],
        week_lookup: Dict[str, WeekWindow],
    ) -> None:
        """Persist weekly JSON files with UTF-8 encoding."""
        for week_key, posts in weekly_posts.items():
            if not posts:
                continue

            filename = week_lookup[week_key].filename
            path = os.path.join(self.output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            print(f"Wrote {len(posts)} posts to {path}")

    def _prepare_week_windows(self) -> List[WeekWindow]:
        """Generate ISO week windows spanning the configured time range."""
        windows: List[WeekWindow] = []
        cursor = self._floor_to_week_start(self.start_time)

        while cursor <= self.end_time:
            iso_year, iso_week, _ = cursor.isocalendar()
            week_start = cursor
            week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            week_end = min(week_end, self.end_time)
            windows.append(
                WeekWindow(
                    iso_year=iso_year,
                    iso_week=iso_week,
                    start=week_start,
                    end=week_end,
                )
            )
            cursor = week_start + timedelta(days=7)

        return windows

    def _floor_to_week_start(self, dt: datetime) -> datetime:
        """Return the Monday 00:00 UTC of the week containing `dt`."""
        monday = dt - timedelta(days=dt.weekday())
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def _week_key_for_datetime(self, dt: datetime) -> str:
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}_week{iso_week:02d}"

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean plain text."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Simplified or Traditional Chinese characters."""
        if not text:
            return False
        return bool(CHINESE_PATTERN.search(text))

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse ISO 8601 timestamps into aware datetime objects."""
        if not value:
            return None
        try:
            sanitized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(sanitized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None


def print_usage_instructions() -> None:
    """Display configuration guidance for users."""
    instructions = f"""
Configuration Tips
------------------
1. Changing target instance:
   - Update BASE_URL at the top of this script
     e.g. BASE_URL = "https://mastodon.social"

2. Adjusting time range or weekly quota:
   - Modify START_TIME_UTC and END_TIME_UTC (keep timezone=timezone.utc)
   - Adjust WEEKLY_LIMIT as needed (e.g. WEEKLY_LIMIT = 50)

3. Providing a Bearer token:
   - Set AUTH_BEARER_TOKEN = "YOUR_ACCESS_TOKEN"
   - Token is required if the instance enforces authentication (401 responses)

4. Handling common errors:
   - 401 Unauthorized: set AUTH_BEARER_TOKEN
   - 429 Too Many Requests: the script will wait automatically; increase MIN_REQUEST_INTERVAL_SECONDS if needed
   - 503 Service Unavailable: the script retries; you may rerun later if the issue persists
   - No Chinese results: try a different instance or expand the time range

5. Output:
   - JSON files are saved under {os.path.abspath(OUTPUT_DIR)}
   - Filenames follow the pattern 2025_week01_chinese_posts.json
   - Each record includes only the required fields.
"""
    print(instructions)


def run_scraper_once() -> None:
    """Instantiate the scraper, execute a crawl, and print sample posts."""
    scraper = MastodonChineseScraper(
        base_url=BASE_URL,
        start_time=START_TIME_UTC,
        end_time=END_TIME_UTC,
        weekly_limit=WEEKLY_LIMIT,
        output_dir=OUTPUT_DIR,
        auth_token=AUTH_BEARER_TOKEN,
    )
    weekly_posts = scraper.run()
    scraper.print_sample_posts(weekly_posts, sample_size=5)


def start_daily_scheduler() -> None:
    """Keep the process alive and trigger the scraper once per day at the target UTC time."""

    last_run_date: Optional[date] = None

    def maybe_run_today() -> None:
        nonlocal last_run_date
        now_utc = datetime.now(timezone.utc)
        if (
            now_utc.hour == SCHEDULED_UTC_HOUR
            and now_utc.minute == SCHEDULED_UTC_MINUTE
        ):
            if last_run_date == now_utc.date():
                return
            print(
                f"[{now_utc.isoformat()}] Scheduled run triggered at "
                f"{SCHEDULED_UTC_HOUR:02d}:{SCHEDULED_UTC_MINUTE:02d} UTC."
            )
            run_scraper_once()
            last_run_date = now_utc.date()

    schedule.every().minute.do(maybe_run_today)
    print(
        "Scheduler active: waiting for daily run at "
        f"{SCHEDULED_UTC_HOUR:02d}:{SCHEDULED_UTC_MINUTE:02d} UTC. "
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Scheduler stopped by user request.")


def main() -> None:
    print_usage_instructions()
    print("Running immediate scrape...")
    run_scraper_once()
    start_daily_scheduler()


if __name__ == "__main__":
    main()
