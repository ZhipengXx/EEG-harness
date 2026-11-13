"""Paper search and reading through GPT Researcher, without its LLM workflow.

The supported upstream interface is GPT Researcher 0.16.0 (commit
3582a6841fe7dc801e2a34e34ee408641aea9bce). Imports are optional and lazy so
ordinary EEG research does not require the web-research dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from .literature_contract import build_literature_search_text


class LiteratureSourceError(RuntimeError):
    """Search or reading failed, rather than finding no relevant papers."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def prepare_paper(paper: dict[str, Any], research_input: dict[str, Any],
                  max_tokens: int) -> dict[str, Any]:
    """Fit reader context using complete source paragraphs, never a text prefix.

    PDF text without paragraph/page separators may have no usable unit within
    the allowance; report that instead of cutting a scientific statement. The
    source artifact stays untouched and ranges refer to that original content.
    """
    from .context.budget import token_count

    if max_tokens < 1:
        raise ValueError("paper_input_tokens_must_be_positive")
    content = paper["content"]
    original_tokens, precision, tokenizer = token_count(content)
    if original_tokens <= max_tokens:
        return dict(paper)

    terms = set(re.findall(r"[^\W_]{3,}", build_literature_search_text(research_input).lower()))
    paragraphs = []
    # Blank lines and form feeds are source boundaries. A wrapped PDF line or
    # an arbitrary token count is not a paragraph boundary.
    for index, match in enumerate(re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\f|\Z)", content)):
        text = match.group()
        count = token_count(text)[0]
        words = set(re.findall(r"[^\W_]{3,}", text.lower()))
        score = len(terms & words) / max(1, len(words)) ** 0.5
        paragraphs.append((index, match.start(), match.end(), text, count, score))
    selected, used = [], 0
    separator_tokens = token_count("\n\n")[0]
    for paragraph in sorted(paragraphs, key=lambda row: (-row[5], row[0])):
        cost = paragraph[4] + (separator_tokens if selected else 0)
        if used + cost <= max_tokens:
            selected.append(paragraph)
            used += cost
    if not selected:
        raise LiteratureSourceError(
            f"paper_has_no_complete_passage_within_budget:{paper['source_id']}"
        )
    selected.sort(key=lambda row: row[0])
    text = "\n\n".join(row[3] for row in selected)
    prepared = dict(paper)
    # content_path describes the unabridged fetched artifact. Do not leave it
    # attached as if it were the path/hash of the selected model input.
    original_path = prepared.pop("content_path", None)
    prepared.update(content=text, coverage="selected_passages", content_sha256=_digest(text), read_scope={
        "selection": "complete paragraphs/pages, lexical relevance, original order",
        "original_coverage": paper.get("coverage"),
        "original_read_scope": paper.get("read_scope"),
        "original_content_path": original_path,
        "original_content_sha256": paper.get("content_sha256") or _digest(content),
        "selected_char_ranges": [{"paragraph_index": row[0], "start": row[1], "end": row[2]} for row in selected],
        "original_tokens": original_tokens, "selected_tokens": token_count(text)[0],
        "token_precision": precision, "tokenizer_version": tokenizer,
    })
    return prepared


def _components() -> tuple[Any, Any, Any, Any]:
    try:
        from gpt_researcher.retrievers.openalex.openalex import OpenAlexSearch
        from gpt_researcher.scraper import PyMuPDFScraper, Scraper
        from gpt_researcher.utils.workers import WorkerPool
    except ImportError as exc:
        raise LiteratureSourceError(
            "literature_dependencies_missing: install GPT Researcher 0.16.0 and its "
            "dependencies in the literature runtime; no packages are installed automatically"
        ) from exc
    return OpenAlexSearch, Scraper, PyMuPDFScraper, WorkerPool


def _search(query: str, limit: int, retriever_class: Any) -> list[dict[str, Any]]:
    import requests

    class StrictOpenAlexSearch(retriever_class):
        """Keep upstream normalization, but do not turn HTTP failures into []."""

        def search(self, max_results: int = 20) -> list[dict[str, Any]]:
            params = {"search": self.query, "per_page": min(max_results, 25), "sort": self.sort}
            headers = {}
            if self.api_key:
                # Header auth keeps a key out of HTTP-error URLs and cached queries.
                headers["Authorization"] = "Bearer " + self.api_key
            try:
                response = requests.get(self.BASE_URL, params=params, headers=headers, timeout=20)
            except requests.RequestException as exc:
                raise LiteratureSourceError(f"openalex_search_failed:{type(exc).__name__}") from exc
            if not response.ok:
                raise LiteratureSourceError(f"openalex_search_http_{response.status_code}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise LiteratureSourceError("openalex_search_invalid_json") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise LiteratureSourceError("openalex_search_missing_results")
            papers = []
            for work in payload["results"]:
                href = self._pick_href(work)
                if not href:
                    continue
                oa = work.get("best_oa_location") or {}
                papers.append({
                    "title": work.get("title") or "",
                    "url": _canonical_url(href),
                    "doi": work.get("doi"),
                    "openalex_id": work.get("id"),
                    "abstract": self._reconstruct_abstract(work.get("abstract_inverted_index")) or "",
                    "pdf_url": oa.get("pdf_url"),
                })
            return papers

    return StrictOpenAlexSearch(query).search(max_results=limit)


def _read_pdf_pages(link: str, session: Any) -> tuple[str, list, str]:
    """Use the upstream PDF loader once, retaining its actual page boundaries.

    GPT Researcher's PDF scraper joins pages with a plain newline. Keeping a
    form feed here makes later whole-page selection possible without downloading
    or parsing the PDF again.
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    with session.get(link, timeout=(5, 30), stream=True) as response:
        response.raise_for_status()
        with NamedTemporaryFile(suffix=".pdf") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
            handle.flush()
            pages = PyMuPDFLoader(handle.name, mode="page").load()
    content = "\n\f\n".join(page.page_content for page in pages)
    title = pages[0].metadata.get("title", "") if pages else ""
    return content, [], title


async def _read_batch(papers: list[dict[str, Any]], components: tuple[Any, ...],
                      max_workers: int) -> list[dict[str, Any]]:
    _, scraper_class, pdf_scraper, worker_class = components
    pdf_urls = {paper["url"] for paper in papers if paper.get("pdf_url")}

    class PageAwarePDFScraper(pdf_scraper):
        def scrape(self) -> tuple[str, list, str]:
            return _read_pdf_pages(self.link, self.session)

    class PaperScraper(scraper_class):
        def get_scraper(self, link: str) -> Any:
            # arxiv.org/pdf/<id> has no .pdf suffix; upstream otherwise reads
            # only the arXiv abstract. OpenAlex explicitly labels PDF locations.
            selected = pdf_scraper if link in pdf_urls else super().get_scraper(link)
            return PageAwarePDFScraper if selected is pdf_scraper else selected

    pool = worker_class(max_workers=max_workers)
    scraper = PaperScraper(
        [paper["url"] for paper in papers],
        "EEGagent-Literature/1.0", "bs", worker_pool=pool,
    )
    try:
        rows = await asyncio.gather(*(
            scraper.extract_data_from_url(paper["url"], scraper.session) for paper in papers
        ))
        result = []
        for paper, row in zip(papers, rows):
            content = (row or {}).get("raw_content") or ""
            backend = scraper.get_scraper(paper["url"])
            if content:
                coverage = "full_text" if backend is PageAwarePDFScraper else (
                    "abstract" if backend.__name__ == "ArxivScraper" else "excerpt"
                )
                scope = "all PDF pages: extracted text" if coverage == "full_text" else (
                    "arXiv abstract" if coverage == "abstract" else "extracted HTML; article completeness unverified"
                )
            else:
                content = paper["abstract"]
                coverage, scope = "abstract", "OpenAlex abstract; source-page reading failed"
            result.append({**paper, "content": content, "coverage": coverage, "read_scope": scope})
        return result
    finally:
        scraper.session.close()
        pool.executor.shutdown(wait=True)


def _cached_paper(output_dir: Path, source_id: str) -> dict[str, Any] | None:
    path = output_dir / "papers" / (source_id + ".json")
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    content_path = Path(saved["content_path"])
    if not content_path.is_file():
        return None
    return {**saved, "content": content_path.read_text(encoding="utf-8")}


def fetch_papers(queries: list[str], *, max_papers: int, output_dir: Path,
                 retriever: str = "openalex", max_workers: int = 3,
                 search_results_per_query: int = 20,
                 rank_candidates: Callable[[list[dict[str, Any]]],
                                           tuple[list[dict[str, Any]], dict[str, Any]]] | None = None,
                 ranking_context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Search metadata broadly, then return up to ``max_papers`` readable papers.

    Successful query responses and source text are cached independently. This
    function makes no model calls, performs no character truncation, and adds no
    retry layer around GPT Researcher's existing fetch behavior. The manifest
    reuses a completed selection for matching inputs, and distinguishes an empty
    successful search from search/read failure.
    """
    if retriever != "openalex":
        raise ValueError("literature_retriever_must_be_openalex")
    if max_papers < 1 or max_workers < 1:
        raise ValueError("max_papers_and_max_workers_must_be_positive")
    if not 1 <= search_results_per_query <= 25:
        raise ValueError("search_results_per_query_must_be_between_1_and_25")
    queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    if not queries:
        raise ValueError("literature_queries_empty")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_key = _digest(json.dumps({
        "queries": queries, "retriever": retriever, "max_papers": max_papers,
        "search_results_per_query": search_results_per_query,
        "rank_candidates": rank_candidates is not None, "ranking_context": ranking_context,
    }, ensure_ascii=False, sort_keys=True))
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and manifest.get("selection_key") == selection_key:
            restored = [_cached_paper(output_dir, source_id) for source_id in manifest["source_ids"]]
            if all(paper is not None for paper in restored):
                return restored
    components = None
    hits, query_records = [], []
    limit = search_results_per_query
    for query in queries:
        key = _digest(json.dumps([retriever, query, limit], ensure_ascii=False))
        path = output_dir / "queries" / (key + ".json")
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if cached.get("status") == "completed":
            papers = cached["papers"]
        else:
            try:
                components = components or _components()
                papers = _search(query, limit, components[0])
            except Exception as exc:
                _write_json(path, {"query": query, "retriever": retriever, "status": "failed", "error": str(exc)})
                _write_json(manifest_path, {"status": "search_failed", "query": query,
                                           "selection_key": selection_key, "error": str(exc)})
                raise
            _write_json(path, {"query": query, "retriever": retriever, "status": "completed", "papers": papers})
        hits.append(papers)
        query_records.append({"query": query, "results": len(papers), "cache_path": str(path)})

    # Interleave query ranks so a broad first query does not monopolize the batch.
    candidates, identities, urls = [], {}, {}
    for rank in range(max((len(rows) for rows in hits), default=0)):
        for query, rows in zip(queries, hits):
            if rank >= len(rows):
                continue
            paper = dict(rows[rank])
            identity = paper.get("doi") or paper.get("openalex_id") or paper["url"]
            existing = identities.get(identity) or urls.get(paper["url"])
            if existing is not None:
                if query not in existing["retrieval_queries"]:
                    existing["retrieval_queries"].append(query)
                continue
            paper.update(source_id="paper_" + _digest(identity)[:24], retrieval_queries=[query])
            identities[identity] = urls[paper["url"]] = paper
            candidates.append(paper)

    ranking_report = {"status": "query_rank_order"}
    if candidates and rank_candidates is not None:
        candidates, ranking_report = rank_candidates(candidates)
    records, failures = [], []
    while candidates and len(records) < max_papers:
        batch = candidates[:max_papers - len(records)]
        candidates = candidates[len(batch):]
        pending, recovered = [], {}
        for paper in batch:
            saved = _cached_paper(output_dir, paper["source_id"])
            if saved is not None:
                recovered[paper["source_id"]] = {**saved, "retrieval_queries": paper["retrieval_queries"]}
                continue
            pending.append(paper)
        if pending:
            components = components or _components()
            for paper in asyncio.run(_read_batch(pending, components, max_workers)):
                if not paper["content"].strip():
                    failures.append({"source_id": paper["source_id"], "url": paper["url"], "reason": "no_readable_text_or_abstract"})
                    continue
                content = paper.pop("content")
                paper.pop("abstract", None)
                content_path = output_dir / "papers" / (paper["source_id"] + ".txt")
                content_path.parent.mkdir(parents=True, exist_ok=True)
                content_path.write_text(content, encoding="utf-8")
                paper.update(content_path=str(content_path), content_sha256=_digest(content),
                             fetched_at=datetime.now(timezone.utc).isoformat())
                _write_json(content_path.with_suffix(".json"), paper)
                recovered[paper["source_id"]] = {**paper, "content": content}
        records.extend(recovered[paper["source_id"]] for paper in batch if paper["source_id"] in recovered)
    status = "completed" if records or not any(hits) else "read_failed"
    _write_json(manifest_path, {
        "status": status, "queries": query_records, "source_ids": [paper["source_id"] for paper in records],
        "failures": failures, "retriever": retriever, "selection_key": selection_key,
        "search_results_per_query": search_results_per_query, "max_papers": max_papers,
        "ranking": ranking_report,
    })
    if status == "read_failed":
        raise LiteratureSourceError("literature_sources_unreadable: see source manifest")
    return records
