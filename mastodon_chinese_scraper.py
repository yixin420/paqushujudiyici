#!/usr/bin/env python3
"""
Mastodon Chinese Posts Daily Scraper
====================================

This script collects all public Chinese-language posts (Simplified + Traditional)
for the current UTC day from a Mastodon instance. It respects Mastodon API rate
limits, supports optional Bearer token authentication, and outputs daily JSON
files along with a rolling CSV summary of per-day totals.

By default, the script automatically sets the scraping window to today's UTC date.
Default configuration targets https://m.cmx.im.

Usage:
    python mastodon_chinese_scraper.py

Adjust global constants near the top of this file to change the domain, output
directory, or authentication token. See the README or the
`print_usage_instructions` function for detailed guidance.
"""
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests import RequestException
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout
from bs4 import BeautifulSoup


# -----------------------------
# Configuration (edit as needed)
# -----------------------------
BASE_URL = "https://m.cmx.im"

# Automatically set the time window to cover today's UTC date.
_TODAY_UTC = datetime.now(timezone.utc).date()
START_TIME_UTC = datetime(
    _TODAY_UTC.year,
    _TODAY_UTC.month,
    _TODAY_UTC.day,
    0,
    0,
    0,
    tzinfo=timezone.utc,
)
END_TIME_UTC = datetime(
    _TODAY_UTC.year,
    _TODAY_UTC.month,
    _TODAY_UTC.day,
    23,
    59,
    59,
    tzinfo=timezone.utc,
)

# Directory to store per-day outputs
OUTPUT_DIR = "data"
# Aggregated summary CSV filename (stored inside OUTPUT_DIR)
DAILY_SUMMARY_FILENAME = "daily_summary.csv"
# Optional Mastodon API Bearer token; set to None for anonymous access
AUTH_BEARER_TOKEN: Optional[str] = "XlgSKGYdYD2C4JpblzqSSnvLK4Z3g69oRRYRet36Gjw"
# Request settings
REQUEST_LIMIT_PER_CALL = 40  # Mastodon API caps this at 40 per request
REQUEST_RETRIES = 4
REQUEST_RETRY_DELAY_SECONDS = 3
REQUEST_DELAY_SECONDS = 2  # delay between successive pagination requests
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 10
MIN_PAGE_LIMIT = 5
MAX_CONSECUTIVE_FETCH_FAILURES = 5
FETCH_FAILURE_BACKOFF_SECONDS = 30
WAIT_FOR_DAY_COMPLETION = True
LIVE_POLL_INTERVAL_SECONDS = 60


CHINESE_PATTERN = re.compile(
    r"[\u4e00-\u9fa5\u814a-\u9fff\u3105-\u312f\u3400-\u4dbf]"
)


@dataclass(frozen=True)
class DayWindow:
    """Represents a single UTC day window."""

    start: datetime
    end: datetime

    @property
    def key(self) -> str:
        return self.start.strftime("%Y-%m-%d")

    @property
    def filename(self) -> str:
        return f"{self.start.strftime('%Y%m%d')}_data.json"


class MastodonChineseScraper:
    """Collects all Chinese-language posts per UTC day from a Mastodon instance."""

    def __init__(
        self,
        base_url: str,
        start_time: datetime,
        end_time: datetime,
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
        self.auth_token = auth_token
        self.output_dir = output_dir
        self.summary_path = os.path.join(self.output_dir, DAILY_SUMMARY_FILENAME)
        self.instance_slug = self._derive_instance_slug(self.base_url)

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
        Execute the scraper and return a mapping of day key to collected posts.
        JSON files are saved for each day encountered within the time range.
        """
        day_windows = self._prepare_day_windows()
        if not day_windows:
            return {}

        daily_posts: Dict[str, List[dict]] = {w.key: [] for w in day_windows}
        day_lookup: Dict[str, DayWindow] = {w.key: w for w in day_windows}
        seen_ids: set[str] = set()
        latest_seen_id: Optional[str] = None

        max_id: Optional[str] = None
        reached_start = False
        consecutive_failures = 0

        while not reached_start:
            try:
                statuses = self._fetch_timeline_page(max_id=max_id)
                consecutive_failures = 0
            except RuntimeError as exc:
                consecutive_failures += 1
                print(
                    "获取公开时间线失败: "
                    f"{exc}. 已连续失败 {consecutive_failures} 次，"
                    f"将在 {FETCH_FAILURE_BACKOFF_SECONDS}s 后重试..."
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FETCH_FAILURES:
                    raise RuntimeError(
                        "多次连续失败，停止抓取。请检查网络连接、令牌或目标实例。"
                    ) from exc
                time.sleep(FETCH_FAILURE_BACKOFF_SECONDS)
                continue

            if not statuses:
                break

            if statuses[-1].get("id"):
                max_id = statuses[-1]["id"]
            latest_seen_id, crossed_start = self._process_status_batch(
                statuses=statuses,
                daily_posts=daily_posts,
                day_lookup=day_lookup,
                seen_ids=seen_ids,
                latest_seen_id=latest_seen_id,
                stop_on_start=True,
            )
            if crossed_start:
                reached_start = True

        now_utc = datetime.now(timezone.utc)
        if WAIT_FOR_DAY_COMPLETION and now_utc < self.end_time:
            print(
                f"当前 UTC 时间 {now_utc.isoformat()} 尚未到达结束时间 "
                f"{self.end_time.isoformat()}，将进入实时监听模式以捕获全天数据。"
            )
            latest_seen_id = self._follow_day_forward(
                latest_seen_id=latest_seen_id,
                daily_posts=daily_posts,
                day_lookup=day_lookup,
                seen_ids=seen_ids,
            )

        self._save_daily_outputs(daily_posts, day_lookup)
        return daily_posts

    def print_sample_posts(
        self, daily_posts: Dict[str, List[dict]], sample_size: int = 5
    ) -> None:
        """Print up to `sample_size` posts across all collected days for quick inspection."""
        flattened: List[Tuple[str, dict]] = []
        for day_key, posts in daily_posts.items():
            for post in posts:
                flattened.append((day_key, post))

        flattened.sort(
            key=lambda pair: pair[1]["created_at"] or "", reverse=True
        )
        print(f"\nSample of up to {sample_size} Chinese-language posts:")
        for day_key, post in flattened[:sample_size]:
            created_at = post.get("created_at", "")
            username = post.get("username", "")
            display = post.get("display_name", "")
            print(
                f"[{day_key}] {created_at} — {username} ({display})\n"
                f"URL: {post.get('url', '')}\n"
                f"Content: {post.get('content', '')}\n"
                "-" * 60
            )

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _fetch_timeline_page(
        self, max_id: Optional[str] = None, min_id: Optional[str] = None
    ) -> List[dict]:
        """
        Fetch a single page from the configured timeline with retries and rate limiting.

        Uses `/api/v1/timelines/public` by default and supports both backward pagination
        via `max_id` and forward pagination via `min_id`, matching the Mastodon
        timeline API behaviors described in the official documentation.
        """
        url = f"{self.base_url}/api/v1/timelines/public"
        params = {"limit": self.page_limit}
        if max_id:
            params["max_id"] = max_id
        if min_id:
            params["min_id"] = min_id

        for attempt in range(REQUEST_RETRIES):
            self._respect_rate_limit()
            try:
                response = self.session.get(url, params=params, timeout=30)
                self._last_request_ts = time.time()
            except (Timeout, RequestsConnectionError) as exc:
                self._last_request_ts = time.time()
                print(
                    f"Network timeout/connection error on attempt {attempt + 1} "
                    f"of {REQUEST_RETRIES}: {exc}. Retrying in "
                    f"{REQUEST_RETRY_DELAY_SECONDS}s..."
                )
                time.sleep(REQUEST_RETRY_DELAY_SECONDS)
                continue
            except RequestException as exc:
                self._last_request_ts = time.time()
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
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                raise RuntimeError(
                    "Received 422 Unprocessable Entity response even with minimal "
                    f"page size ({self.page_limit}). Details: {detail_str}"
                )

            # For other status codes, raise the error immediately.
            response.raise_for_status()

        raise RuntimeError("Failed to fetch public timeline after multiple attempts.")

    def _respect_rate_limit(self) -> None:
        """Ensure at least REQUEST_DELAY_SECONDS between requests."""
        elapsed = time.time() - self._last_request_ts
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)

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

    def _follow_day_forward(
        self,
        latest_seen_id: Optional[str],
        daily_posts: Dict[str, List[dict]],
        day_lookup: Dict[str, DayWindow],
        seen_ids: set[str],
    ) -> Optional[str]:
        """
        Continue polling the public timeline until END_TIME_UTC in order to capture
        every post published within the current UTC day.
        """
        if latest_seen_id is None:
            print(
                "尚未记录最新的时间线 ID，实时监听将从当前最新公开时间线开始。"
            )

        poll_round = 0
        while datetime.now(timezone.utc) < self.end_time:
            poll_round += 1
            statuses = self._fetch_timeline_page(min_id=latest_seen_id)
            if statuses:
                print(
                    f"[实时轮询 #{poll_round}] 获取到 {len(statuses)} 条新增数据，正在过滤..."
                )
            latest_seen_id, _ = self._process_status_batch(
                statuses=statuses,
                daily_posts=daily_posts,
                day_lookup=day_lookup,
                seen_ids=seen_ids,
                latest_seen_id=latest_seen_id,
                stop_on_start=False,
            )

            remaining_seconds = (
                self.end_time - datetime.now(timezone.utc)
            ).total_seconds()
            if remaining_seconds <= 0:
                break

            sleep_seconds = min(LIVE_POLL_INTERVAL_SECONDS, remaining_seconds)
            if sleep_seconds > 0:
                print(
                    f"[实时轮询 #{poll_round}] 距离当天结束还有 "
                    f"{remaining_seconds/60:.1f} 分钟，休眠 {sleep_seconds:.0f} 秒后继续..."
                )
                time.sleep(sleep_seconds)

        # Final sweep to cover posts published right before END_TIME_UTC.
        final_statuses = self._fetch_timeline_page(min_id=latest_seen_id)
        if final_statuses:
            print("执行日终补抓，确保包含收盘前的所有帖子...")
            latest_seen_id, _ = self._process_status_batch(
                statuses=final_statuses,
                daily_posts=daily_posts,
                day_lookup=day_lookup,
                seen_ids=seen_ids,
                latest_seen_id=latest_seen_id,
                stop_on_start=False,
            )

        return latest_seen_id

    def _save_daily_outputs(
        self,
        daily_posts: Dict[str, List[dict]],
        day_lookup: Dict[str, DayWindow],
    ) -> None:
        """Persist per-day JSON files and update the summary CSV."""
        summary_rows: List[Tuple[str, int]] = []
        for day_key in sorted(day_lookup.keys()):
            posts = daily_posts.get(day_key, [])
            filename = self._build_daily_filename(day_key, len(posts))
            path = os.path.join(self.output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            print(f"{day_key} 共收集到 {len(posts)} 条数据，写入 {path}")
            summary_rows.append((day_key, len(posts)))

        self._append_daily_summary(summary_rows)

    def _append_daily_summary(self, rows: List[Tuple[str, int]]) -> None:
        """Create or update the aggregated daily summary CSV."""
        if not rows:
            return

        existing: Dict[str, int] = {}
        if os.path.exists(self.summary_path):
            with open(self.summary_path, "r", encoding="utf-8", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for entry in reader:
                    date_key = entry.get("date")
                    count_val = entry.get("count")
                    if not date_key:
                        continue
                    try:
                        existing[date_key] = int(count_val) if count_val is not None else 0
                    except ValueError:
                        existing[date_key] = 0

        for day_key, count in rows:
            existing[day_key] = count

        with open(self.summary_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["date", "count"])
            for day_key in sorted(existing.keys()):
                writer.writerow([day_key, existing[day_key]])

    def _build_daily_filename(self, day_key: str, count: int) -> str:
        """Return the configured storage filename for a given day."""
        date_str = day_key.replace("-", "")
        safe_count = max(count, 0)
        return f"{date_str}_{self.instance_slug}_({safe_count}).json"

    def _prepare_day_windows(self) -> List[DayWindow]:
        """Generate per-day windows spanning the configured time range."""
        windows: List[DayWindow] = []
        cursor = self._floor_to_day_start(self.start_time)

        while cursor <= self.end_time:
            day_start = cursor
            day_end = min(
                day_start + timedelta(days=1) - timedelta(seconds=1),
                self.end_time,
            )
            windows.append(
                DayWindow(
                    start=day_start,
                    end=day_end,
                )
            )
            cursor = day_start + timedelta(days=1)

        return windows

    def _floor_to_day_start(self, dt: datetime) -> datetime:
        """Return 00:00 UTC of the day containing `dt`."""
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    def _day_key_for_datetime(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    def _derive_instance_slug(self, base_url: str) -> str:
        """Derive a filesystem-friendly slug from the target instance URL."""
        parsed = urlparse(base_url)
        host = parsed.netloc or parsed.path or base_url
        slug = host.strip().lower().replace(".", "-").replace(":", "-").strip("-")
        return slug or "mastodon-instance"

    def _process_status_batch(
        self,
        statuses: List[dict],
        daily_posts: Dict[str, List[dict]],
        day_lookup: Dict[str, DayWindow],
        seen_ids: set[str],
        latest_seen_id: Optional[str],
        stop_on_start: bool,
    ) -> Tuple[Optional[str], bool]:
        """Process a list of statuses and return the updated cursors."""
        reached_start = False
        for status in statuses:
            post_id = status.get("id")
            latest_seen_id = self._update_latest_seen_id(latest_seen_id, post_id)

            created_at = self._parse_datetime(status.get("created_at"))
            if created_at is None:
                continue

            if created_at > self.end_time:
                continue

            if created_at < self.start_time:
                if stop_on_start:
                    reached_start = True
                continue

            self._ingest_status(
                status=status,
                created_at=created_at,
                daily_posts=daily_posts,
                day_lookup=day_lookup,
                seen_ids=seen_ids,
            )

        return latest_seen_id, reached_start

    def _ingest_status(
        self,
        status: dict,
        created_at: datetime,
        daily_posts: Dict[str, List[dict]],
        day_lookup: Dict[str, DayWindow],
        seen_ids: set[str],
    ) -> bool:
        """Normalize, deduplicate, and append a status to the per-day bucket."""
        if status.get("visibility") != "public":
            return False

        post_id = status.get("id")
        if not post_id or post_id in seen_ids:
            return False

        plain_content = self._html_to_text(status.get("content", ""))
        if not self._contains_chinese(plain_content):
            return False

        day_key = self._day_key_for_datetime(created_at)
        if day_key not in daily_posts:
            return False

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

        daily_posts[day_key].append(record)
        seen_ids.add(post_id)
        self._sort_day_posts(daily_posts[day_key])
        return True

    def _sort_day_posts(self, posts: List[dict]) -> None:
        """Sort posts for a given day in reverse chronological order."""
        posts.sort(key=lambda item: item.get("created_at") or "", reverse=True)

    def _update_latest_seen_id(
        self, current_latest: Optional[str], candidate: Optional[str]
    ) -> Optional[str]:
        """Track the greatest Mastodon status ID encountered so far."""
        if candidate is None:
            return current_latest
        if current_latest is None:
            return candidate
        try:
            if int(candidate) > int(current_latest):
                return candidate
        except ValueError:
            if candidate > current_latest:
                return candidate
        return current_latest

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

2. Adjusting time range:
   - By default the script targets today's UTC day.
   - Modify START_TIME_UTC and END_TIME_UTC manually if you need a different window (keep timezone=timezone.utc).

3. Providing a Bearer token:
   - Set AUTH_BEARER_TOKEN = "YOUR_ACCESS_TOKEN"
   - Token is required if the instance enforces authentication (401 responses)

4. Handling common errors:
   - 401 Unauthorized: set AUTH_BEARER_TOKEN
   - 429 Too Many Requests: the script will wait automatically; increase REQUEST_DELAY_SECONDS if needed
   - 503 Service Unavailable: the script retries; you may rerun later if the issue persists
   - No Chinese results: try a different instance or expand the time range

5. Output:
   - JSON files are saved under {os.path.abspath(OUTPUT_DIR)}
   - Filenames follow the pattern YYYYMMDD_instance-slug_(<daily_count>).json
   - A rolling {DAILY_SUMMARY_FILENAME} file stores the total count per day.
   - Each record includes only the required fields.

6. Full-day capture mode:
   - When WAIT_FOR_DAY_COMPLETION is True the scraper keeps polling with min_id until END_TIME_UTC.
   - Adjust LIVE_POLL_INTERVAL_SECONDS to control how often new statuses are requested during this mode.
"""
    print(instructions)


def main() -> None:
    print_usage_instructions()
    scraper = MastodonChineseScraper(
        base_url=BASE_URL,
        start_time=START_TIME_UTC,
        end_time=END_TIME_UTC,
        output_dir=OUTPUT_DIR,
        auth_token=AUTH_BEARER_TOKEN,
    )
    daily_posts = scraper.run()
    scraper.print_sample_posts(daily_posts, sample_size=5)


if __name__ == "__main__":
    main()
