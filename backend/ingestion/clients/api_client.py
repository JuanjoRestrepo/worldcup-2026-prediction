import logging
import time
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)


class FootballAPIClient:
    """
    Client for football-data.org API.

    Handles:
    - retries
    - rate limiting
    - basic validation
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.football-data.org/v4",
        timeout: int = 10,
        retries: int = 3,
        backoff: int = 5,
    ):
        self.base_url = base_url
        self.headers = {"X-Auth-Token": api_key}
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Internal request handler with retry and rate limit handling.
        """
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(self.retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout,
                )

                # ✅ Success
                if response.status_code == 200:
                    return response.json()  # type: ignore[no-any-return]

                # ⚠️ Rate limit (429) or Request limit reached (403)
                if response.status_code in [429, 403]:
                    wait_time = self.backoff * (2**attempt)  # Exponential backoff
                    if response.status_code == 403:
                        logger.warning(
                            f"Request limit reached (403). Waiting {wait_time}s before retry..."
                        )
                    else:
                        logger.warning(
                            f"Rate limit hit (429). Waiting {wait_time}s before retry..."
                        )
                    time.sleep(wait_time)
                    continue

                # ⚠️ Other API errors
                logger.warning(f"API error {response.status_code}: {response.text}")

            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")

            time.sleep(self.backoff)

        raise Exception(f"Failed request after {self.retries} retries")

    def get_matches(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        competition: str | None = None,
    ) -> dict[str, Any]:
        """
        Fetch matches with optional filters.

        Automatically handles date ranges exceeding 10 days by splitting into chunks.

        Args:
            date_from: YYYY-MM-DD
            date_to: YYYY-MM-DD
            competition: competition code (e.g., 'WC')

        Returns:
            dict with 'matches' key containing all matches from the period
        """
        # If no date range specified, just make a single request
        if not date_from or not date_to:
            params = {}
            if date_from:
                params["dateFrom"] = date_from
            if date_to:
                params["dateTo"] = date_to
            if competition:
                params["competitions"] = competition

            data = self._request("matches", params)
            if "matches" not in data:
                raise ValueError("Invalid API response: 'matches' key missing")
            logger.info(f"Fetched {len(data['matches'])} matches from API")
            return data

        # Parse dates
        start_date = datetime.strptime(date_from, "%Y-%m-%d")
        end_date = datetime.strptime(date_to, "%Y-%m-%d")
        date_diff = (end_date - start_date).days

        # API only allows max 10 days per request
        MAX_DAYS = 10

        # If within 10 days, make single request
        if date_diff <= MAX_DAYS:
            params = {
                "dateFrom": date_from,
                "dateTo": date_to,
            }
            if competition:
                params["competitions"] = competition

            data = self._request("matches", params)
            if "matches" not in data:
                raise ValueError("Invalid API response: 'matches' key missing")
            logger.info(f"Fetched {len(data['matches'])} matches from API")
            return data

        # Split into 10-day chunks and fetch all
        logger.info(
            f"Date range ({date_diff} days) exceeds API limit ({MAX_DAYS} days). Splitting into chunks..."
        )

        all_matches = []
        current_date = start_date

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=MAX_DAYS), end_date)

            chunk_from = current_date.strftime("%Y-%m-%d")
            chunk_to = chunk_end.strftime("%Y-%m-%d")

            logger.info(f"Fetching matches for {chunk_from} to {chunk_to}")

            params = {
                "dateFrom": chunk_from,
                "dateTo": chunk_to,
            }
            if competition:
                params["competitions"] = competition

            try:
                data = self._request("matches", params)
                if "matches" in data:
                    all_matches.extend(data["matches"])
                    logger.info(f"  → Fetched {len(data['matches'])} matches")
            except Exception as e:
                logger.error(f"Error fetching chunk {chunk_from} to {chunk_to}: {e}")
                raise

            # Move to next chunk
            current_date = chunk_end + timedelta(days=1)

        logger.info(f"Fetched total {len(all_matches)} matches from API")

        if len(all_matches) == 0:
            logger.warning(
                "API returned zero matches. Check your filters or API status."
            )

        return {"matches": all_matches}
