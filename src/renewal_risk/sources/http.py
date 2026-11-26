"""A small, dependency-free JSON-over-HTTPS client that add-ons share.

It does the dull things every integration gets wrong at least once: timeouts, honouring ``Retry-After`` on a
429, backing off on 5xx, turning 401/403/404 into errors a person can act on, and never putting a credential
in an error message. The network call itself is a swappable ``transport`` so tests (and the simulated demo
mode) run the exact same code path without a network.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .base import ErrorKind, SourceError

# (url, headers, timeout_seconds) -> (status, response_headers, body)
Transport = Callable[[str, dict[str, str], float], tuple[int, dict[str, str], bytes]]

USER_AGENT = "renewal-risk-scorer (+https://github.com/GlassLightLabs/renewal-risk-scorer)"


def urllib_transport(url: str, headers: dict[str, str], timeout: float) -> tuple[int, dict[str, str], bytes]:
    if not url.startswith("https://"):
        raise SourceError("config", "Only https:// addresses are allowed.")
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https enforced above)
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read() or b""
    except urllib.error.URLError as e:
        raise SourceError("network", f"Could not reach {urllib.parse.urlsplit(url).netloc}: {e.reason}.") from None
    except TimeoutError:
        raise SourceError("network", f"{urllib.parse.urlsplit(url).netloc} did not answer in time.") from None


class JsonClient:
    """GET JSON from one host with retries. Paths are resolved against ``base_url``; full URLs must stay on it."""

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        transport: Transport | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        max_wait: float = 90.0,
        sleep: Callable[[float], None] = time.sleep,
        service: str = "The service",
    ):
        self.base_url = base_url.rstrip("/")
        self._host = urllib.parse.urlsplit(self.base_url).netloc
        self._headers = {"Accept": "application/json", "User-Agent": USER_AGENT, **(headers or {})}
        self._transport = transport or urllib_transport
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._sleep = sleep
        self.service = service
        self.requests = 0
        self.waited_seconds = 0.0

    def url(self, path_or_url: str, params: dict[str, Any] | None = None) -> str:
        if path_or_url.startswith(("http://", "https://")):
            url = path_or_url
            if urllib.parse.urlsplit(url).netloc != self._host:
                raise SourceError("bad_response", f"{self.service} pointed to another host; refusing to follow it.")
        else:
            url = f"{self.base_url}/{path_or_url.lstrip('/')}"
        if params:
            sep = "&" if "?" in url else "?"
            url += sep + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        return url

    def get(self, path_or_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.url(path_or_url, params)
        attempt = 0
        while True:
            self.requests += 1
            status, headers, body = self._transport(url, self._headers, self.timeout)
            if 200 <= status < 300:
                try:
                    data = json.loads(body or b"{}")
                except ValueError:
                    raise SourceError("bad_response", f"{self.service} returned something that is not JSON.") from None
                if not isinstance(data, dict):
                    raise SourceError("bad_response", f"{self.service} returned an unexpected JSON shape.")
                return data
            if status in (401, 403):
                raise SourceError(
                    "auth",
                    f"{self.service} refused the credentials (HTTP {status}).",
                    status=status,
                )
            if status == 404:
                raise SourceError("not_found", f"{self.service} has nothing at that address (HTTP 404).", status=404)
            if status == 429 or status >= 500:
                attempt += 1
                wait = _retry_after(headers) if status == 429 else None
                if wait is None:
                    wait = min(2.0 ** (attempt - 1), 30.0)
                if attempt > self.max_retries or wait > self.max_wait:
                    kind: ErrorKind = "rate_limit" if status == 429 else "network"
                    what = "is rate-limiting requests" if status == 429 else f"kept failing (HTTP {status})"
                    raise SourceError(kind, f"{self.service} {what} after {attempt} tries.", status=status)
                self.waited_seconds += wait
                self._sleep(wait)
                continue
            raise SourceError("bad_response", f"{self.service} answered HTTP {status}.", status=status)


def _retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None
