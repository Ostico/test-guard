# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tests for GitHub API thin client utilities."""

from unittest.mock import MagicMock, patch

from src.github_api import (
    DEFAULT_PER_PAGE,
    DEFAULT_TIMEOUT,
    GITHUB_API_URL,
    GITHUB_API_VERSION,
    create_session,
    get_json,
    get_paginated,
    get_text,
    post_json,
)


def test_module_constants_are_defined() -> None:
    assert GITHUB_API_URL == "https://api.github.com"
    assert GITHUB_API_VERSION == "2022-11-28"
    assert DEFAULT_TIMEOUT == 30
    assert DEFAULT_PER_PAGE == 100


@patch("src.github_api.HTTPAdapter")
@patch("src.github_api.Retry")
@patch("src.github_api.requests.Session")
def test_create_session_sets_headers_and_mounts_retries(
    mock_session_cls: MagicMock,
    mock_retry_cls: MagicMock,
    mock_adapter_cls: MagicMock,
) -> None:
    session = MagicMock()
    session.headers = {}
    mock_session_cls.return_value = session
    retry = MagicMock()
    mock_retry_cls.return_value = retry
    adapter = MagicMock()
    mock_adapter_cls.return_value = adapter

    created = create_session("ghp_test_token")

    assert created is session
    assert session.headers["Authorization"] == "Bearer ghp_test_token"
    assert session.headers["Accept"] == "application/vnd.github+json"
    assert session.headers["X-GitHub-Api-Version"] == "2022-11-28"
    mock_retry_cls.assert_called_once_with(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    mock_adapter_cls.assert_called_once_with(max_retries=retry)
    session.mount.assert_any_call("https://", adapter)
    session.mount.assert_any_call("http://", adapter)


def test_get_json_calls_get_and_raises_for_status() -> None:
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = {"ok": True}
    session.get.return_value = response

    payload = get_json(session, "https://api.github.com/repos/o/r", {"page": 1})

    session.get.assert_called_once_with(
        "https://api.github.com/repos/o/r",
        params={"page": 1},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status.assert_called_once_with()
    assert payload == {"ok": True}


def test_get_paginated_accumulates_pages_via_next_link() -> None:
    session = MagicMock()

    first = MagicMock()
    first.json.return_value = [{"id": 1}]
    first.links = {"next": {"url": "https://api.github.com/next?page=2"}}

    second = MagicMock()
    second.json.return_value = [{"id": 2}, {"id": 3}]
    second.links = {}

    session.get.side_effect = [first, second]

    data = get_paginated(session, f"{GITHUB_API_URL}/repos/o/r/pulls/1/files", {"state": "open"})

    assert data == [{"id": 1}, {"id": 2}, {"id": 3}]
    session.get.assert_any_call(
        f"{GITHUB_API_URL}/repos/o/r/pulls/1/files",
        params={"state": "open", "per_page": DEFAULT_PER_PAGE},
        timeout=DEFAULT_TIMEOUT,
    )
    session.get.assert_any_call(
        "https://api.github.com/next?page=2",
        params=None,
        timeout=DEFAULT_TIMEOUT,
    )
    first.raise_for_status.assert_called_once_with()
    second.raise_for_status.assert_called_once_with()


def test_post_json_returns_response_without_raise_for_status() -> None:
    session = MagicMock()
    response = MagicMock()
    session.post.return_value = response
    body = {"body": "hello"}

    result = post_json(session, f"{GITHUB_API_URL}/repos/o/r/issues/1/comments", body)

    assert result is response
    session.post.assert_called_once_with(
        f"{GITHUB_API_URL}/repos/o/r/issues/1/comments",
        json=body,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status.assert_not_called()


def test_get_text_returns_text_when_ok() -> None:
    session = MagicMock()
    response = MagicMock()
    response.ok = True
    response.text = "raw-file-content"
    session.get.return_value = response

    text = get_text(session, f"{GITHUB_API_URL}/repos/o/r/contents/file.py")

    assert text == "raw-file-content"
    session.get.assert_called_once_with(
        f"{GITHUB_API_URL}/repos/o/r/contents/file.py",
        headers={"Accept": "application/vnd.github.raw+json"},
        timeout=DEFAULT_TIMEOUT,
    )


def test_get_text_returns_none_when_not_ok() -> None:
    session = MagicMock()
    response = MagicMock()
    response.ok = False
    response.text = "error"
    session.get.return_value = response

    text = get_text(session, f"{GITHUB_API_URL}/repos/o/r/contents/file.py", "application/json")

    assert text is None
    session.get.assert_called_once_with(
        f"{GITHUB_API_URL}/repos/o/r/contents/file.py",
        headers={"Accept": "application/json"},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status.assert_not_called()
