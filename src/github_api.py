"""Thin HTTP client wrappers for GitHub API calls."""

from __future__ import annotations

from typing import Any, cast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
DEFAULT_TIMEOUT = 30
DEFAULT_PER_PAGE = 100


def create_session(token: str) -> requests.Session:
    """Create a configured requests session for GitHub API calls."""
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
    )
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, str | int] | None = None,
) -> Any:
    """GET JSON response and raise on HTTP errors."""
    response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_paginated(
    session: requests.Session,
    url: str,
    params: dict[str, str | int] | None = None,
) -> list[Any]:
    """GET paginated JSON responses and flatten into one list."""
    all_items: list[Any] = []
    first_params: dict[str, str | int] = {"per_page": DEFAULT_PER_PAGE}
    if params is not None:
        first_params.update(params)

    next_url: str | None = url
    next_params: dict[str, str | int] | None = first_params

    while next_url is not None:
        response = session.get(next_url, params=next_params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            all_items.extend(cast(list[Any], payload))
        next_link = response.links.get("next", {}).get("url")
        next_url = next_link if isinstance(next_link, str) else None
        next_params = None

    return all_items


def post_json(
    session: requests.Session,
    url: str,
    body: dict[str, Any],
) -> requests.Response:
    """POST JSON body and return raw response without raising."""
    return session.post(url, json=body, timeout=DEFAULT_TIMEOUT)


def get_text(
    session: requests.Session,
    url: str,
    accept: str = "application/vnd.github.raw+json",
) -> str | None:
    """GET text content with an Accept override; best-effort on errors."""
    response = session.get(url, headers={"Accept": accept}, timeout=DEFAULT_TIMEOUT)
    if response.ok:
        return response.text
    return None
