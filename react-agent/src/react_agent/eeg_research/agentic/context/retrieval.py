"""Bounded query construction and explicit MiniLM representations.

The original record is never shortened.  Only the derived embedding view is
segmented, and every segment records its original identity and token range.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

VERSION = "eeg_research.context.retrieval.v1"
MAX_DELIVERY_CHARS = 8000
MAX_CANDIDATES = 128
MAX_CHUNKS = 32


def prepare_query(*, question: Any = "", action: Any = "", error: Any = "",
                  method: Any = "", protocol: Any = "", identities: Any = (), feedback: Any = "",
                  tokenizer=None, max_tokens: int | None = None,
                  max_chars: int | None = None) -> dict:
    """Build one encoding view from original fields, with truthful coverage.

    The tokenizer owns the production query limit. Callers can request a
    character excerpt explicitly, but intermediate callers never shorten and
    rewrap a query before this boundary.
    """
    def words(value):
        if isinstance(value, dict):
            return " ".join(str(value[key]).strip() for key in (
                "method", "method_revision", "intervention", "scope", "regime",
                "dataset", "evaluation_identity", "error_type", "code")
                if value.get(key) not in (None, ""))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return value.strip() if isinstance(value, str) else ""

    fields = [("Feedback", words(feedback)), ("Question", words(question)),
              ("Action", words(action)), ("Error", words(error)), ("Method", words(method)),
              ("Protocol", words(protocol))]
    if isinstance(identities, (list, tuple)):
        fields.append(("IDs", " ".join(str(value).strip() for value in identities if value not in (None, ""))))
    active = [(key, value) for key, value in fields if value]
    selected = dict(active)
    bare_question = len(active) == 1 and active[0][0] == "Question"

    def render():
        return "\n".join(("" if bare_question else label + ": ") + selected[label]
                         for label, _ in active if selected[label])

    def shorten(label, target, current):
        value = selected[label]
        end = max(0, min(len(value) - 1, len(value) * max(0, target) // current))
        boundary = value.rfind(" ", 0, end + 1)
        selected[label] = value[:boundary if boundary > end // 2 else end].rstrip()

    def fit(limit, size, field_size):
        # Reduce the longest fields together. A long question therefore does
        # not crowd out a short method/feedback field; spare capacity is shared.
        text = render()
        while size(text) > limit and any(selected.values()):
            counts = {label: field_size(value) for label, value in selected.items() if value}
            longest = max(counts.values())
            group = [label for label, count in counts.items() if count == longest]
            next_length = max((count for count in counts.values() if count < longest), default=0)
            excess = size(text) - limit
            reduction = min(longest - next_length, max(1, (excess + len(group) - 1) // len(group)))
            for label in group:
                shorten(label, longest - reduction, longest)
            text = render()
        return text

    complete = render()
    metadata = {"builder_version": VERSION, "query_hash": hashlib.sha256(complete.encode()).hexdigest(),
                "representation": "bounded_query_card"}
    text = complete
    if max_chars is not None:
        text = fit(max(0, max_chars), len, len)
    if tokenizer is not None:
        if max_tokens is None or max_tokens < 4:
            raise ValueError("query_token_budget_missing")
        count = lambda value: len(tokenizer(value, truncation=False, add_special_tokens=True)["input_ids"])
        content_count = lambda value: max(1, len(tokenizer(value, truncation=False, add_special_tokens=False)["input_ids"]))
        text = fit(max_tokens, count, content_count)
        metadata.update(original_token_count=count(complete), encoded_token_count=count(text),
                        max_sequence_length=max_tokens)
        if count(text) > max_tokens:
            raise ValueError("query_special_tokens_exceed_budget")
    metadata.update(omitted=text != complete, chars=len(text), selected_fields={
        label.lower(): {"original_chars": len(value), "selected_char_range": [0, len(selected[label])],
                        "complete": selected[label] == value}
        for label, value in active})
    return {"text": text, "metadata": metadata}


def prepare_embedding_chunks(text: str, *, tokenizer, max_tokens: int,
                             record_id: str, max_chunks: int = MAX_CHUNKS) -> list[dict]:
    """Return stable, length-verified chunks bound to the untouched full text."""
    if max_tokens < 4 or not 1 <= max_chunks <= MAX_CHUNKS:
        raise ValueError("invalid_embedding_chunk_budget")
    full_hash = hashlib.sha256(text.encode()).hexdigest()
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    special = len(tokenizer("", truncation=False, add_special_tokens=True)["input_ids"])
    capacity = max_tokens - special
    if capacity < 1:
        raise ValueError("embedding_special_tokens_exceed_budget")
    chunks = []
    if not ids:
        return []
    start = 0
    while start < len(ids) and len(chunks) < max_chunks:
        end = min(len(ids), start + capacity)
        segment = text if start == 0 and end == len(ids) else tokenizer.decode(
            ids[start:end], skip_special_tokens=True)
        encoded = tokenizer(segment, truncation=False, add_special_tokens=True)["input_ids"]
        # Decode/re-encode can add tokens; shrink monotonically and validate.
        while len(encoded) > max_tokens and end > start:
            end -= 1
            segment = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            encoded = tokenizer(segment, truncation=False, add_special_tokens=True)["input_ids"]
        if end <= start:
            raise ValueError("embedding_chunk_cannot_fit")
        digest = hashlib.sha256(segment.encode()).hexdigest()
        chunks.append({"chunk_id": "emb_" + digest[:24], "record_id": record_id,
                       "text": segment, "text_hash": digest, "full_text_hash": full_hash,
                       "token_range": [start, end], "full_token_count": len(ids),
                       "encoded_token_count": len(encoded), "max_sequence_length": max_tokens,
                       "representation": "full" if start == 0 and end == len(ids) else "token_chunk",
                       "builder_version": VERSION})
        start = end
    for chunk in chunks:
        chunk.update(chunk_count=len(chunks), complete_index_coverage=start == len(ids),
                     omitted_token_range=None if start == len(ids) else [start, len(ids)])
    return chunks


def source_signature(item: dict) -> str:
    """Group shared support deterministically without conflating record scope."""
    refs = item.get("source_refs") or []
    return hashlib.sha256(json.dumps(refs, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode()).hexdigest()


def scientific_support_key(item: dict) -> tuple:
    """Local deduplication identity; it never enters a model card."""
    body = item.get("body") or {}
    facts = body.get("native_facts") or {}
    science_scope = {"evaluation_scope": body.get("evaluation_scope"),
        "conditions": body.get("conditions") or body.get("preconditions") or body.get("applicability"),
        "facts": {key: facts[key] for key in ("scope", "seed", "screening_fold_ids", "screening_subjects")
                  if key in facts}}
    key = (source_signature(item), item.get("applicability"),
           json.dumps(item.get("condition_differences") or {}, sort_keys=True),
           json.dumps(science_scope, sort_keys=True))
    if not item.get("source_refs"):
        key += (item.get("record_type"), item.get("record_id"))
    return key


def deduplicate_hits(items: list[dict], role: str = "research_planner") -> list[dict]:
    """Choose a role-appropriate representative only for identical support/scope."""
    prefer = "skill" if role in {"candidate_coder", "candidate_reviewer"} else "lesson"
    seen = {}; result = []
    for item in items:
        key = scientific_support_key(item)
        if key not in seen:
            seen[key] = len(result)
            result.append(item)
        elif item.get("record_type") == prefer and result[seen[key]].get("record_type") != prefer:
            result[seen[key]] = item
    return result
