"""Polite, cached access to CourtListener's public v4 search endpoint."""

from __future__ import annotations

import json
import math
import os
import posixpath
import stat
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from eyecite import get_citations

from .util import canonical_json, project_root, sha256_bytes, write_json

SearchParamValue = str | int | float | bool | None | Sequence[str | int | float | bool]
COURTLISTENER_CACHE_SCHEMA_VERSION = "sanctionbench.courtlistener_cache.v3"
DEFAULT_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_RESULTS_PER_PAGE = 1_000
DEFAULT_MAX_PAGES = 100
DEFAULT_MAX_AGGREGATE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_AGGREGATE_RESULTS = 10_000
DEFAULT_MAX_RETRY_AFTER_SECONDS = 30.0
MAX_HTTP_ATTEMPTS_PER_SEARCH = 6


def build_exact_query(exact_clauses: Sequence[str]) -> str:
    """Join already-fielded exact clauses into one parenthesized OR query."""

    clauses = [clause.strip() for clause in exact_clauses]
    if not clauses or any(not clause for clause in clauses):
        raise ValueError("exact_clauses must contain at least one non-empty clause")
    return " OR ".join(f"({clause})" for clause in clauses)


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse either form of the standard Retry-After header."""

    if value is None:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _validated_base_url(base_url: str, *, token_present: bool) -> str:
    """Validate an authenticated origin before constructing the HTTP client."""

    parsed = urlsplit(base_url)
    decoded_path = unquote(parsed.path)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.port not in {None, 443}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or "\\" in decoded_path
        or not decoded_path.startswith("/")
    ):
        raise ValueError(
            "CourtListener base_url must use HTTPS with a hostname, the default port, and no "
            "credentials, query parameters, fragments, or unsafe path"
        )
    if token_present and parsed.hostname.casefold() != "www.courtlistener.com":
        raise ValueError(
            "COURTLISTENER_API_TOKEN may be sent only to www.courtlistener.com; unset it for an "
            "explicit custom test endpoint"
        )
    return base_url.rstrip("/") + "/"


def _bounded_evidence_text(value: object, *, maximum: int) -> str:
    """Project remote metadata into one control-character-free bounded field."""

    return " ".join(str(value).split())[:maximum]


def _validated_search_payload(payload: object) -> dict[str, Any]:
    """Reject structurally unbounded or malformed CourtListener responses."""

    if not isinstance(payload, dict):
        raise ValueError("Unexpected CourtListener search response")
    count = payload.get("count")
    results = payload.get("results")
    next_url = payload.get("next")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or not isinstance(results, list)
        or len(results) > DEFAULT_MAX_RESULTS_PER_PAGE
        or not all(isinstance(result, dict) for result in results)
        or (next_url is not None and (not isinstance(next_url, str) or len(next_url) > 4_096))
    ):
        raise ValueError("Unexpected CourtListener search response")
    return payload


def _read_bounded_cache_json(path: Path, *, maximum: int) -> object:
    """Read one regular cache file through a single no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"CourtListener cache entry is not a readable regular file: {path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ValueError("CourtListener cache record exceeds the byte limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read(maximum + 1)
        if len(encoded) > maximum:
            raise ValueError("CourtListener cache record exceeds the byte limit")
    finally:
        os.close(descriptor)
    try:
        return json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CourtListener cache record is not valid UTF-8 JSON") from error


def _page_budget_bytes(page: Mapping[str, Any]) -> int:
    payload = {key: value for key, value in page.items() if key != "_sanctionbench_retrieval"}
    return len(canonical_json(payload).encode("utf-8"))


def _citation_search_key(value: str) -> str:
    citations = list(get_citations(value))
    if citations:
        matched = getattr(citations[0], "matched_text", None)
        if callable(matched):
            return str(matched()).strip()
    upper = value.upper()
    if " WL " in upper or " U.S. " in upper and " LEXIS " in upper:
        return value.split("(", 1)[0].strip()
    return value.strip()


def _normalize_identity(value: str) -> str:
    aliases = {"cnty": "county", "cty": "city", "dept": "department"}
    words: list[str] = []
    current: list[str] = []
    for character in value.lower():
        if character.isalnum():
            current.append(character)
        elif current:
            word = "".join(current)
            words.append(aliases.get(word, word))
            current = []
    if current:
        word = "".join(current)
        words.append(aliases.get(word, word))
    return " ".join(words)


def _case_name_similarity(expected: str, actual: str) -> float:
    from difflib import SequenceMatcher

    stop = {"v", "vs", "the", "of", "and", "inc", "llc", "ltd", "co", "corp"}
    left = {word for word in _normalize_identity(expected).split() if word not in stop}
    right = {word for word in _normalize_identity(actual).split() if word not in stop}
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    containment = overlap / min(len(left), len(right))
    jaccard = overlap / len(left | right)
    sequence = SequenceMatcher(None, " ".join(sorted(left)), " ".join(sorted(right))).ratio()
    return max(jaccard, sequence if containment >= 0.5 else 0.0)


def _citation_matches(expected: str, reported: str) -> bool:
    left = "".join(character for character in expected.lower() if character.isalnum())
    right = "".join(character for character in reported.lower() if character.isalnum())
    return bool(left and right and (left in right or right in left))


def _validated_pagination_url(base_url: str, value: str) -> str:
    """Keep server-provided cursors inside the configured API origin and path."""

    resolved = urljoin(base_url, value)
    base = urlsplit(base_url)
    cursor = urlsplit(resolved)
    base_origin = (base.scheme.casefold(), base.hostname, base.port)
    cursor_origin = (cursor.scheme.casefold(), cursor.hostname, cursor.port)
    if base.scheme.casefold() not in {"http", "https"} or cursor_origin != base_origin:
        raise ValueError("CourtListener pagination URL escaped the configured API origin")
    if cursor.username or cursor.password or cursor.fragment:
        raise ValueError("CourtListener pagination URL contains prohibited URL components")
    decoded_base_path = unquote(base.path)
    decoded_cursor_path = unquote(cursor.path)
    if "\\" in decoded_cursor_path:
        raise ValueError("CourtListener pagination URL contains an unsafe path")
    base_path = posixpath.normpath(decoded_base_path)
    cursor_path = posixpath.normpath(decoded_cursor_path)
    if base_path != "/" and not (
        cursor_path == base_path or cursor_path.startswith(base_path.rstrip("/") + "/")
    ):
        raise ValueError("CourtListener pagination URL escaped the configured API path")
    return resolved


class CourtListenerClient:
    """A narrow CourtListener client used for retrieval and deterministic tools.

    The v4 search endpoint is currently usable anonymously. A token, when
    present, is read from COURTLISTENER_API_TOKEN and never persisted.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://www.courtlistener.com/api/rest/v4/",
        timeout: float = 30.0,
        delay_seconds: float = 0.35,
        cache_dir: Path | None = None,
        user_agent: str = "SanctionBench/1.0.0 public legal benchmark",
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_retry_after_seconds: float = DEFAULT_MAX_RETRY_AFTER_SECONDS,
        before_http_request: Callable[[], None] | None = None,
    ) -> None:
        token = os.environ.get("COURTLISTENER_API_TOKEN")
        self.base_url = _validated_base_url(base_url, token_present=bool(token))
        self.authentication_mode = "token" if token else "anonymous"
        self.endpoint_sha256 = sha256_bytes(self.base_url.encode("utf-8"))
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if not math.isfinite(max_retry_after_seconds) or max_retry_after_seconds < 0:
            raise ValueError("max_retry_after_seconds must be finite and nonnegative")
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Token {token}"
        # Redirects are intentionally not followed. Pagination cursors are
        # validated before each request; an automatic redirect would bypass
        # that origin-and-path boundary after validation.
        self.client = httpx.Client(timeout=timeout, follow_redirects=False, headers=headers)
        self.delay_seconds = delay_seconds
        self.cache_dir = cache_dir or project_root() / "data/cache/courtlistener"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_response_bytes = max_response_bytes
        self.max_cache_bytes = max_response_bytes + 1024 * 1024
        self.max_retry_after_seconds = max_retry_after_seconds
        self.before_http_request = before_http_request
        self._last_request = 0.0
        self.last_retry_after: str | None = None

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> CourtListenerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(
        self,
        query: str,
        *,
        search_type: str = "o",
        order_by: str = "score desc",
        force: bool = False,
        params: Mapping[str, SearchParamValue] | None = None,
    ) -> dict[str, Any]:
        """Search with the original convenience arguments plus optional parameters.

        Values in ``params`` override the convenience arguments. Existing callers
        that only pass ``query``, ``search_type``, and ``order_by`` retain the same
        request and cache key.
        """

        request_params: dict[str, SearchParamValue] = {
            "q": query,
            "type": search_type,
            "order_by": order_by,
        }
        if params:
            request_params.update(params)
        return self.search_params(request_params, force=force)

    def search_params(
        self,
        params: Mapping[str, SearchParamValue],
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run and cache a Search API request with arbitrary GET parameters."""

        request_params = dict(params)
        return self._search_request(params=request_params, force=force)

    def _before_http_request(self) -> None:
        """Hook for callers that need to account for each wire attempt."""

        if self.before_http_request is not None:
            self.before_http_request()

    def _wait_before_retry(self, wait_seconds: float) -> None:
        """Wait after a retryable response; specialized clients may checkpoint instead."""

        time.sleep(min(wait_seconds, self.max_retry_after_seconds))

    def _bounded_get(
        self,
        url: str,
        *,
        params: Mapping[str, SearchParamValue] | None = None,
    ) -> httpx.Response:
        """Buffer at most the configured response-byte budget."""

        request_params = dict(params) if params is not None else None
        with self.client.stream("GET", url, params=request_params) as streamed:
            content_length = streamed.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as error:
                    raise ValueError("CourtListener returned an invalid Content-Length") from error
                if declared_length < 0 or declared_length > self.max_response_bytes:
                    raise ValueError("CourtListener response exceeds the byte limit")
            content = bytearray()
            for chunk in streamed.iter_bytes():
                content.extend(chunk)
                if len(content) > self.max_response_bytes:
                    raise ValueError("CourtListener response exceeds the byte limit")
            decoded_headers = dict(streamed.headers)
            decoded_headers.pop("content-encoding", None)
            decoded_headers.pop("content-length", None)
            return httpx.Response(
                status_code=streamed.status_code,
                headers=decoded_headers,
                content=bytes(content),
                request=streamed.request,
            )

    def _search_request(
        self,
        *,
        params: Mapping[str, SearchParamValue] | None = None,
        absolute_url: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if (params is None) == (absolute_url is None):
            raise ValueError("Provide exactly one of params or absolute_url")

        request_target: dict[str, Any]
        if absolute_url is not None:
            request_target = {"url": absolute_url}
        else:
            request_target = {"params": dict(params or {})}
        request_identity: dict[str, Any] = {
            "endpoint_sha256": self.endpoint_sha256,
            "authentication_mode": self.authentication_mode,
            "target": request_target,
        }
        cache_key = sha256_bytes(canonical_json(request_identity).encode())
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not force:
            cached = _read_bounded_cache_json(cache_path, maximum=self.max_cache_bytes)
            if (
                not isinstance(cached, dict)
                or cached.get("schema_version") != COURTLISTENER_CACHE_SCHEMA_VERSION
                or cached.get("request") != request_identity
                or not isinstance(cached.get("retrieved_at"), str)
                or len(cached["retrieved_at"]) > 64
                or not set(cached).issubset(
                    {"schema_version", "retrieved_at", "request", "response", "retry_after"}
                )
            ):
                raise ValueError("CourtListener cache record does not match the requested query")
            payload = _validated_search_payload(cached.get("response"))
            retrieved_at = cached["retrieved_at"]
            output = {
                **payload,
                "_sanctionbench_retrieval": {
                    "retrieved_at": retrieved_at,
                    "from_cache": True,
                    "cache_key": cache_key,
                    "cache_schema_version": COURTLISTENER_CACHE_SCHEMA_VERSION,
                },
            }
            retry_after = cached.get("retry_after") if isinstance(cached, dict) else None
            if isinstance(retry_after, list):
                output["_sanctionbench_retrieval"]["retry_after"] = retry_after
            return output

        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        response: httpx.Response | None = None
        retry_after_values: list[str] = []
        for attempt in range(MAX_HTTP_ATTEMPTS_PER_SEARCH):
            self._before_http_request()
            try:
                if absolute_url is not None:
                    # CourtListener's `next` value is an opaque cursor URL. Pass it
                    # back verbatim instead of rebuilding or merging query params.
                    response = self._bounded_get(absolute_url)
                else:
                    response = self._bounded_get(urljoin(self.base_url, "search/"), params=params)
            except httpx.TransportError:
                if attempt == MAX_HTTP_ATTEMPTS_PER_SEARCH - 1:
                    raise
                time.sleep(min(20.0, 3.0 * (attempt + 1)))
                continue
            self._last_request = time.monotonic()
            if response.is_redirect:
                raise ValueError("CourtListener search redirects are not followed")
            if response.status_code != 429:
                break
            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                self.last_retry_after = retry_after
                retry_after_values.append(retry_after)
            wait_seconds = _retry_after_seconds(retry_after)
            if wait_seconds is None:
                wait_seconds = min(30.0, 5.0 * (attempt + 1))
            self._wait_before_retry(wait_seconds)
        if response is None:
            raise RuntimeError("CourtListener request loop produced no response")
        response.raise_for_status()
        payload = _validated_search_payload(response.json())
        retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        cache_record: dict[str, Any] = {
            "schema_version": COURTLISTENER_CACHE_SCHEMA_VERSION,
            "retrieved_at": retrieved_at,
            "request": request_identity,
            "response": payload,
        }
        if retry_after_values:
            cache_record["retry_after"] = retry_after_values
        write_json(
            cache_path,
            cache_record,
            indent=None,
            max_bytes=self.max_cache_bytes,
        )
        output = {
            **payload,
            "_sanctionbench_retrieval": {
                "retrieved_at": retrieved_at,
                "from_cache": False,
                "cache_key": cache_key,
            },
        }
        if retry_after_values:
            output["_sanctionbench_retrieval"]["retry_after"] = retry_after_values
        return output

    def iter_search_pages(
        self,
        params: Mapping[str, SearchParamValue],
        *,
        force: bool = False,
        max_pages: int | None = DEFAULT_MAX_PAGES,
    ) -> Iterator[dict[str, Any]]:
        """Yield Search API pages, following each server-provided ``next`` URL."""

        if max_pages is None:
            max_pages = DEFAULT_MAX_PAGES
        if max_pages < 1:
            raise ValueError("max_pages must be positive when provided")

        page = self.search_params(params, force=force)
        pages_seen = 0
        next_urls_seen: set[str] = set()
        while True:
            yield page
            pages_seen += 1
            if max_pages is not None and pages_seen >= max_pages:
                return

            next_url = page.get("next")
            if next_url is None:
                return
            if not isinstance(next_url, str) or not next_url:
                raise ValueError("CourtListener search response contained an invalid next URL")
            next_url = _validated_pagination_url(self.base_url, next_url)
            if next_url in next_urls_seen:
                raise ValueError("CourtListener search pagination repeated a next URL")
            next_urls_seen.add(next_url)
            page = self._search_request(absolute_url=next_url, force=force)

    def search_all(
        self,
        params: Mapping[str, SearchParamValue],
        *,
        force: bool = False,
        max_pages: int | None = DEFAULT_MAX_PAGES,
        max_aggregate_bytes: int = DEFAULT_MAX_AGGREGATE_BYTES,
        max_results: int = DEFAULT_MAX_AGGREGATE_RESULTS,
    ) -> dict[str, Any]:
        """Return one response with results accumulated across cursor pages."""

        if max_aggregate_bytes < 1 or max_results < 1:
            raise ValueError("aggregate CourtListener budgets must be positive")
        first: dict[str, Any] | None = None
        results: list[dict[str, Any]] = []
        retrievals: list[dict[str, Any]] = []
        aggregate_bytes = 0
        page_count = 0
        last_next: object = None
        for page in self.iter_search_pages(params, force=force, max_pages=max_pages):
            page_count += 1
            aggregate_bytes += _page_budget_bytes(page)
            page_results = page["results"]
            if (
                aggregate_bytes > max_aggregate_bytes
                or len(results) + len(page_results) > max_results
            ):
                raise ValueError("CourtListener aggregate search exceeds its local budget")
            if first is None:
                first = dict(page)
            results.extend(page_results)
            retrievals.append(page.get("_sanctionbench_retrieval", {}))
            last_next = page.get("next")
        if first is None:
            raise RuntimeError("CourtListener pagination returned no pages")
        first["results"] = results
        first["next"] = last_next
        first["_sanctionbench_pagination"] = {
            "page_count": page_count,
            "aggregate_response_bytes": aggregate_bytes,
            "retrievals": retrievals,
        }
        return first

    def iter_batch_search_pages(
        self,
        exact_clauses: Sequence[str],
        *,
        search_type: str,
        order_by: str | None = None,
        common_params: Mapping[str, SearchParamValue] | None = None,
        batch_size: int = 20,
        force: bool = False,
        max_pages_per_batch: int | None = DEFAULT_MAX_PAGES,
    ) -> Iterator[dict[str, Any]]:
        """Search parenthesized exact clauses in request-sized OR batches."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        clauses = [clause.strip() for clause in exact_clauses]
        if not clauses or any(not clause for clause in clauses):
            raise ValueError("exact_clauses must contain at least one non-empty clause")

        for start in range(0, len(clauses), batch_size):
            params: dict[str, SearchParamValue] = dict(common_params or {})
            params["q"] = build_exact_query(clauses[start : start + batch_size])
            params["type"] = search_type
            if order_by is not None:
                params["order_by"] = order_by
            yield from self.iter_search_pages(
                params,
                force=force,
                max_pages=max_pages_per_batch,
            )

    def batch_search(
        self,
        exact_clauses: Sequence[str],
        *,
        search_type: str,
        order_by: str | None = None,
        common_params: Mapping[str, SearchParamValue] | None = None,
        batch_size: int = 20,
        force: bool = False,
        max_pages_per_batch: int | None = DEFAULT_MAX_PAGES,
        max_aggregate_bytes: int = DEFAULT_MAX_AGGREGATE_BYTES,
        max_results: int = DEFAULT_MAX_AGGREGATE_RESULTS,
    ) -> list[dict[str, Any]]:
        """Return result records for all exact-clause batches and cursor pages."""

        if max_aggregate_bytes < 1 or max_results < 1:
            raise ValueError("aggregate CourtListener budgets must be positive")
        results: list[dict[str, Any]] = []
        aggregate_bytes = 0
        for page in self.iter_batch_search_pages(
            exact_clauses,
            search_type=search_type,
            order_by=order_by,
            common_params=common_params,
            batch_size=batch_size,
            force=force,
            max_pages_per_batch=max_pages_per_batch,
        ):
            aggregate_bytes += _page_budget_bytes(page)
            page_results = page["results"]
            if (
                aggregate_bytes > max_aggregate_bytes
                or len(results) + len(page_results) > max_results
            ):
                raise ValueError("CourtListener aggregate search exceeds its local budget")
            results.extend(page_results)
        return results

    def citation_lookup(
        self, citation: str, case_name: str = "", proposition: str = ""
    ) -> dict[str, Any]:
        """Return compact search evidence suitable for a model tool call."""

        citation_key = _citation_search_key(citation)
        query = f'"{citation_key}"'
        payload = self.search(query, search_type="o")
        matches = []
        for result in payload.get("results", [])[:5]:
            raw_citations = result.get("citation") or []
            if isinstance(raw_citations, str):
                raw_citations = [raw_citations]
            if not isinstance(raw_citations, list):
                raw_citations = []
            citations = [
                _bounded_evidence_text(reported, maximum=200) for reported in raw_citations[:20]
            ]
            reported_case_name = _bounded_evidence_text(result.get("caseName", ""), maximum=500)
            citation_match = any(
                _citation_matches(citation_key, str(reported)) for reported in citations
            )
            case_similarity = _case_name_similarity(case_name, reported_case_name)
            absolute_url = _bounded_evidence_text(result.get("absolute_url", ""), maximum=2_000)
            safe_url = (
                urljoin("https://www.courtlistener.com", absolute_url)
                if absolute_url.startswith("/") and not absolute_url.startswith("//")
                else None
            )
            opinions = result.get("opinions") or []
            opinion_record_count = len(opinions) if isinstance(opinions, list) else 0
            matches.append(
                {
                    "case_name": reported_case_name,
                    "citations": citations,
                    "citation_match": citation_match,
                    "case_name_similarity": round(case_similarity, 6),
                    "identity_match": citation_match and case_similarity >= 0.58,
                    "court": _bounded_evidence_text(
                        result.get("court_citation_string") or result.get("court") or "",
                        maximum=200,
                    ),
                    "date_filed": _bounded_evidence_text(result.get("dateFiled") or "", maximum=64),
                    "url": safe_url,
                    "opinion_record_count": min(opinion_record_count, 1_000),
                }
            )
        output = {
            "query": query,
            "requested_case_name": case_name,
            "requested_citation": citation,
            "citation_search_key": citation_key,
            "reported_count": int(payload.get("count", 0)),
            "matches": matches,
            "exact_identity_match_count": sum(bool(match["identity_match"]) for match in matches),
            "opinion_and_snippet_text_excluded": True,
            "source": str(
                httpx.URL(urljoin(self.base_url, "search/")).copy_merge_params(
                    {"q": query, "type": "o"}
                )
            ),
            "warning": "A zero-result search is evidence of absence in CourtListener, not proof that no authority exists in every legal corpus.",
        }
        generic_identity = "exists under this case name and citation" in proposition.lower()
        if proposition and not generic_identity:
            claim_query = f'"{citation}" "{proposition}"'
            claim_payload = self.search(claim_query, search_type="o")
            output["claim_search"] = {
                "query": claim_query,
                "reported_count": int(claim_payload.get("count", 0)),
                "matching_case_names": [
                    _bounded_evidence_text(result.get("caseName", ""), maximum=500)
                    for result in claim_payload.get("results", [])[:5]
                ],
                "response_sha256": response_digest(claim_payload),
                "warning": (
                    "Exact proposition search can miss paraphrases; a zero result requires review "
                    "rather than proving the proposition false."
                ),
            }
        return output


def response_digest(payload: dict[str, Any]) -> str:
    normalized = {key: value for key, value in payload.items() if key != "_sanctionbench_retrieval"}
    return sha256_bytes(canonical_json(normalized).encode())
