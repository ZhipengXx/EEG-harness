"""Paper findings in a separate collection, using the shared MiniLM backend.

Only the three scientific values count against the literature delivery budget.
Storage, retrieval and delivery never make a language-model request.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .context.budget import token_count
from .context.sources import ContextSourceRegistry, atomic_immutable
from .embedding import EmbeddingUnavailable, MemoryConfig, backend_for, canonical, content_hash
from .literature_contract import build_literature_search_text, literature_config
from .semantic_memory import SidecarStore, index_records

MMR_RELEVANCE_WEIGHT = 0.8


def finding_text(finding: dict) -> str:
    """Index every scientific field without indexing local provenance metadata."""
    setup = finding["reported_setup"]
    return "\n".join(value for value in (
        finding["topic"], finding["statement"],
        setup["task"], setup["training"], setup["evaluation"],
    ) if value is not None)


def save_paper_findings(camp: Path, paper: dict, findings: list[dict]) -> dict:
    """Publish an already validated extraction with locally owned provenance.

    The caller validates the model response once using LiteratureExtraction.
    Immutable content identities make replaying a saved response idempotent.
    """
    camp = Path(camp)
    if not literature_config(camp).enabled:
        return {"status": "disabled", "record_ids": []}
    root = camp / "literature_memory"
    source_payload = {key: paper[key] for key in ("title", "url", "content", "coverage")}
    for key in ("doi", "read_scope"):
        if paper.get(key) is not None:
            source_payload[key] = paper[key]
    source_hash = content_hash(source_payload)
    source_id = paper.get("source_id") or "paper_" + source_hash[:24]
    registry = ContextSourceRegistry(camp)
    source = registry.capture(source_payload,
        block_id="literature-paper:" + source_id + ":" + source_hash[:24],
        kind="literature_source", role="literature_researcher",
        identity={"source_id": source_id})
    # The registry owns the immutable raw payload. The literature collection
    # keeps its location, rather than writing the full paper a second time.
    provenance = {key: value for key, value in source_payload.items() if key != "content"}
    provenance.update(source_id=source_id, source_ref=source.source_ref,
                      source_hash=source.source_hash, payload_ref=source.payload_ref)
    atomic_immutable(root / "sources" / (source_hash + ".json"), canonical(provenance))
    record_ids = []
    for finding in findings:
        record_id = "literature_" + content_hash({"source": source.source_ref, "finding": finding})[:24]
        body = {"record_id": record_id, "source_id": source_id,
                "title": source_payload["title"], "url": source_payload["url"],
                "coverage": source_payload["coverage"], "paper_source_ref": source.source_ref,
                **finding}
        if paper.get("doi"):
            body["doi"] = paper["doi"]
        card_source = registry.capture(body, block_id=record_id, kind="literature_finding",
            role="literature_researcher", identity={"record_id": record_id, "source_id": source_id},
            dependency_refs=(source.source_ref,))
        card = {**body, "source_ref": card_source.source_ref}
        atomic_immutable(root / "cards" / (record_id + ".json"), canonical(card))
        record_ids.append(record_id)
    return {"status": "saved" if record_ids else "no_findings", "source_id": source_id,
            "source_ref": source.source_ref, "record_ids": record_ids}


def _record(record_type: str, record_id: str, text: str, body: dict) -> dict:
    return {"record_type": record_type, "record_id": record_id,
            "version": 1, "content_hash": content_hash(body), "search_text": text,
            "text_hash": hashlib.sha256(text.encode()).hexdigest(), "body": body}


def mmr_select(rows: list[dict], *, limit: int, token_budget: int | None = None) -> tuple[list[dict], list[dict]]:
    """Select relevant, less repetitive items without imposing source quotas.

    Rows contain locally computed relevance, diversity vectors and optional
    token costs. An item that cannot fit never influences later selections.
    """
    pending = list(rows)
    selected, omitted = [], []
    redundancy = {row["record_id"]: 0.0 for row in rows}
    used = 0
    while pending and len(selected) < limit:
        eligible = []
        for row in pending:
            cost = row.get("content_tokens", 0)
            if token_budget is not None and used + cost > token_budget:
                omitted.append({"record_id": row["record_id"], "reason": "content_token_budget",
                                "content_tokens": cost})
            else:
                eligible.append(row)
        if not eligible:
            pending = []
            break
        def score(row):
            return (MMR_RELEVANCE_WEIGHT * row["cosine_score"]
                    - (1 - MMR_RELEVANCE_WEIGHT) * redundancy[row["record_id"]])
        chosen = min(eligible, key=lambda row: (-score(row), -row["cosine_score"], row["record_id"]))
        selected.append({**chosen, "mmr_score": score(chosen),
                         "redundancy_score": redundancy[chosen["record_id"]]})
        used += chosen.get("content_tokens", 0)
        pending = [row for row in eligible if row["record_id"] != chosen["record_id"]]
        for row in pending:
            redundancy[row["record_id"]] = max(redundancy[row["record_id"]],
                float(row["diversity_vector"] @ chosen["diversity_vector"]))
    omitted.extend({"record_id": row["record_id"], "reason": "top_k"} for row in pending)
    return selected, omitted


def _embedding_context(camp: Path):
    # Literature remains independently enabled when experiment memory is off.
    goal = json.loads((camp / "goal.json").read_text(encoding="utf-8"))
    config = MemoryConfig.model_validate({**(goal.get("memory") or {}), "enabled": True})
    backend = backend_for(config)
    root = camp / "literature_memory"
    root.mkdir(parents=True, exist_ok=True)
    return backend, SidecarStore(camp, path=root / "index.sqlite")


def _index(store, backend, records: list[dict], report: dict) -> None:
    indexing = index_records(store, records, backend)
    report.update(indexing=indexing, model=backend.metadata["model_id"],
                  revision=backend.metadata["model_revision"], fingerprint=backend.fingerprint)
    if indexing["failed"]:
        raise EmbeddingUnavailable(indexing["reason"])


def _query_vector(store, backend, prepared: dict, report: dict):
    text = prepared["text"]
    digest = hashlib.sha256(text.encode()).hexdigest()
    record = _record("literature_query", "query_" + digest[:24], text, {})
    vector = store.cached(record, backend.fingerprint, backend.metadata)
    report.update(query_cache_reused=vector is not None, query_representation=prepared["metadata"])
    if vector is None:
        vectors, metadata = backend.encode([text])
        store.save_vectors([record], vectors, metadata, backend)
        vector = vectors[0]
    return vector


def _chunk_vectors(store, backend, record: dict) -> list[tuple]:
    result = []
    for chunk in record["embedding_chunks"]:
        vector = store.cached(chunk, backend.fingerprint, backend.metadata)
        span = chunk.get("encoding_scope", {}).get("token_range", (0, 1))
        result.append((vector, span[1] - span[0]))
    return result


def _pooled(chunks: list[tuple]):
    # A short final chunk contributes in proportion to its original token span.
    vector = sum(vector * weight for vector, weight in chunks)
    norm = float((vector @ vector) ** 0.5)
    return vector / norm if norm else chunks[0][0]


def _selection_scores(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("record_id", "cosine_score", "mmr_score", "redundancy_score")}
            for row in rows]


def rank_literature_candidates(camp: Path, science: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    """Order deduplicated search metadata before any full paper is fetched.

    The caller stores the report locally. Original candidate dictionaries are
    preserved, including the search order when the local encoder is unavailable.
    """
    query = {"question": science.get("research_question", ""),
             "protocol": build_literature_search_text({key: science.get(key)
                 for key in ("task_setting", "evaluation_protocol")})}
    report = {"query": query, "candidate_count": len(candidates),
              "source_count": len({row["source_id"] for row in candidates}),
              "mmr_relevance_weight": MMR_RELEVANCE_WEIGHT, "scores": [], "omitted": []}
    if not candidates:
        return [], {**report, "status": "no_candidates", "ordered_ids": []}
    try:
        backend, store = _embedding_context(Path(camp))
        prepared = backend.prepare_query(query)
        if not prepared["text"].strip():
            return list(candidates), {**report, "status": "empty_query",
                                     "ordered_ids": [row["source_id"] for row in candidates]}
        records, without_text = [], []
        for candidate in candidates:
            text = "\n".join(candidate.get(key) or "" for key in ("title", "abstract")).strip()
            if text:
                records.append(_record("literature_candidate", candidate["source_id"], text, candidate))
            else:
                without_text.append(candidate)
        _index(store, backend, records, report)
        query_vector = _query_vector(store, backend, prepared, report)
        rows = []
        for record in records:
            chunks = _chunk_vectors(store, backend, record)
            rows.append({"record_id": record["record_id"], "body": record["body"],
                "cosine_score": max(float(vector @ query_vector) for vector, _ in chunks),
                "diversity_vector": _pooled(chunks)})
        selected, _ = mmr_select(rows, limit=len(rows))
        ordered = [row["body"] for row in selected] + without_text
        report.update(status="ranked", ordered_ids=[row["source_id"] for row in ordered],
                      scores=_selection_scores(selected),
                      unranked_ids=[row["source_id"] for row in without_text])
        return ordered, report
    except (EmbeddingUnavailable, OSError, ValueError, ImportError) as exc:
        return list(candidates), {**report, "status": "embedding_unavailable", "reason": str(exc),
                                 "ordered_ids": [row["source_id"] for row in candidates]}


def _report(root: Path, report: dict) -> None:
    atomic_immutable(root / "reports" / (content_hash(report) + ".json"), canonical(report))


def retrieve_literature(camp: Path, query: str | dict) -> dict:
    """Return complete ranked cards; operational diagnostics stay on disk.

    Literature uses its own top-k and aggregate content-token allowance. It
    does not consume an experiment observation's allowance or minimum score.
    """
    camp = Path(camp)
    config = literature_config(camp)
    if not config.enabled:
        return {"status": "disabled", "items": []}
    root = camp / "literature_memory"
    cards = [json.loads(path.read_text(encoding="utf-8"))
             for path in sorted((root / "cards").glob("*.json"))]
    report = {"query": query, "candidate_count": len(cards), "top_k": config.top_k,
              "prompt_content_tokens": config.prompt_content_tokens,
              "candidate_source_count": len({card["source_id"] for card in cards}),
              "selected_source_count": 0, "mmr_relevance_weight": MMR_RELEVANCE_WEIGHT,
              "selected_ids": [], "content_tokens": 0, "omitted": []}
    if not cards:
        _report(root, {**report, "status": "no_literature"})
        return {"status": "no_literature", "items": []}

    try:
        backend, store = _embedding_context(camp)
        prepared = backend.prepare_query(query)
        if not prepared["text"].strip():
            _report(root, {**report, "status": "empty_query", "query_representation": prepared["metadata"]})
            return {"status": "empty_query", "items": []}
        records = [_record("literature_finding", card["record_id"], finding_text(card), card) for card in cards]
        claims = [_record("literature_claim", card["record_id"],
                          card["topic"] + "\n" + card["statement"], card) for card in cards]
        # Complete conditions drive relevance; only claims drive the soft
        # redundancy penalty. Both views share one pending-encoding batch.
        _index(store, backend, records + claims, report)
        query_vector = _query_vector(store, backend, prepared, report)
        rows = []
        for record, claim in zip(records, claims):
            chunks = _chunk_vectors(store, backend, record)
            tokens, precision, version = token_count(record["search_text"])
            report.update(token_precision=precision, tokenizer_version=version)
            rows.append({"record_id": record["record_id"], "body": record["body"],
                "cosine_score": max(float(vector @ query_vector) for vector, _ in chunks),
                "diversity_vector": _pooled(_chunk_vectors(store, backend, claim)), "content_tokens": tokens})
    except (EmbeddingUnavailable, OSError, ValueError, ImportError) as exc:
        _report(root, {**report, "status": "embedding_unavailable", "reason": str(exc)})
        return {"status": "embedding_unavailable", "items": []}

    chosen, omitted = mmr_select(rows, limit=config.top_k, token_budget=config.prompt_content_tokens)
    report.update(scores=[{"record_id": row["record_id"], "cosine_score": row["cosine_score"]}
                         for row in sorted(rows, key=lambda row: (-row["cosine_score"], row["record_id"]))],
                  selection_scores=_selection_scores(chosen), omitted=omitted,
                  selected_ids=[row["record_id"] for row in chosen],
                  selected_source_count=len({row["body"]["source_id"] for row in chosen}),
                  content_tokens=sum(row["content_tokens"] for row in chosen))
    # Ranking metadata stays in the local report. Every delivered card retains
    # its complete scientific conditions and existing provenance handle.
    selected = [{key: row["body"][key] for key in (
        "topic", "statement", "reported_setup", "title", "url", "coverage", "source_ref", "doi"
    ) if key in row["body"]} for row in chosen]
    status = "ready" if selected else "no_cards_within_budget"
    _report(root, {**report, "status": status})
    return {"status": status, "items": selected}
