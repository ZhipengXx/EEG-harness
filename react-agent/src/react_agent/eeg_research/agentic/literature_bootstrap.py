"""Initialize paper memory once; later planning rounds only retrieve its cards.

Paid calls use the existing role backend and its single repair/recovery loop.
This controller saves completed steps, never re-summarizes their outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context.sources import atomic_immutable
from .embedding import canonical
from .literature_contract import build_literature_input, literature_config


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def literature_input(camp: Path) -> dict:
    """Persist the compact scientific projection before the first paid call."""
    path = Path(camp) / "literature_memory" / "research_input.json"
    if path.is_file():
        return _read(path)
    value = build_literature_input(Path(camp))
    atomic_immutable(path, canonical(value))
    return value


def ensure_literature(camp: Path, *, query_backend=None, reader_backend=None,
                      fetch=None) -> dict:
    """Resume cached queries, papers and replies before admitting the Planner.

    Backends return schema-validated replies. No second validation/repair loop
    is introduced here; a restart can recover a paid reply through role_backend.
    """
    camp = Path(camp)
    config = literature_config(camp)
    if not config.enabled:
        return {"status": "disabled"}
    root = camp / "literature_memory"
    completed = root / "bootstrap.json"
    if completed.is_file():
        return _read(completed)

    from .llm import role_backend
    from .owner_budget import auxiliary_owner
    from .literature_memory import MMR_RELEVANCE_WEIGHT, rank_literature_candidates, save_paper_findings
    from .literature_sources import fetch_papers, prepare_paper

    science = literature_input(camp)
    with auxiliary_owner(camp, purpose="literature_bootstrap"):
        query_path = root / "query_plan.json"
        if query_path.is_file():
            query_plan = _read(query_path)
        else:
            query_backend = query_backend or role_backend(camp, "literature_query_planner")
            query_plan = query_backend(science)
            atomic_immutable(query_path, canonical(query_plan))
        papers = (fetch or fetch_papers)(query_plan["queries"][:config.max_queries],
            max_papers=config.max_papers, search_results_per_query=config.search_results_per_query,
            output_dir=root / "web", retriever=config.retriever,
            rank_candidates=lambda rows: rank_literature_candidates(camp, science, rows),
            ranking_context={"method": "mmr", "relevance_weight": MMR_RELEVANCE_WEIGHT,
                             "research_input": science})
        published = []
        for original in papers:
            source_id = original["source_id"]
            paper_path = root / "prepared_papers" / (source_id + ".json")
            if paper_path.is_file():
                paper = _read(paper_path)
            else:
                paper = prepare_paper(original, science, config.paper_input_tokens)
                atomic_immutable(paper_path, canonical(paper))
            # Exact source ranges and file hashes remain local. The reader sees
            # the text and its coverage; its request is archived by role_backend.
            request = {**science, "paper": {key: paper[key] for key in
                ("title", "content", "coverage")}}
            response_path = root / "extractions" / (source_id + ".json")
            if response_path.is_file():
                extraction = _read(response_path)
            else:
                reader_backend = reader_backend or role_backend(camp, "literature_reader")
                extraction = reader_backend(request)
                atomic_immutable(response_path, canonical(extraction))
            published.append(save_paper_findings(camp, paper, extraction["findings"]))
    result = {"status": "completed", "paper_count": len(papers),
              "finding_count": sum(len(row["record_ids"]) for row in published),
              "papers": published}
    atomic_immutable(completed, canonical(result))
    return result


def planning_literature_query(science: dict, delivered: dict) -> dict[str, str]:
    """Build a search view from science and an already delivered current card.

    A previous question is a retrieval hint, not the next selected method or
    experimental evidence. Full observations remain in the Planner's input;
    only this small query is subject to the embedding backend's usual limits.
    """
    task, protocol = science["task_setting"], science["evaluation_protocol"]
    task_fields = task if isinstance(task, dict) else {}
    protocol_fields = protocol if isinstance(protocol, dict) else {}
    scope = [task_fields.get("task") if isinstance(task, dict) else task,
             task_fields.get("dataset"),
             (protocol_fields.get("split") or protocol_fields.get("generalization_target")
              or task_fields.get("generalization_target")) if isinstance(protocol, dict) else protocol]
    if protocol_fields.get("gallery_size") is not None:
        scope.append(f"{protocol_fields['gallery_size']}-image gallery")
    question = delivered.get("current_question")
    if not isinstance(question, str) or not question.strip():
        pointer = delivered.get("selected_plan") or {}
        terminal_id = delivered.get("terminal_observation_id") or pointer.get("terminal_observation_id")
        cards = (delivered.get("experiment_observations") or {}).get("cards") or []
        current = next((card["body"] for card in cards
                        if terminal_id and card.get("record_id") == terminal_id), {})
        question = current.get("question")
    scope_text = "; ".join(dict.fromkeys(" ".join(value.replace("_", " ").split())
        for value in scope if isinstance(value, str) and value.strip()))
    query = {"question": " ".join(science["research_question"].split()), "protocol": scope_text}
    if isinstance(question, str) and question.strip():
        focus = " ".join(question.split())
        if focus.casefold() != query["question"].casefold():
            query["feedback"] = "Current research focus: " + focus
    return query


def planning_literature(camp: Path, delivered: dict) -> dict:
    """Retrieve once after experiment memory delivery, without rereading it."""
    if not literature_config(Path(camp)).enabled:
        return {"status": "disabled", "items": []}
    from .literature_memory import retrieve_literature

    return retrieve_literature(camp, planning_literature_query(literature_input(camp), delivered))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Search papers and call DeepSeek; default only previews input")
    parser.add_argument("--retrieve", action="store_true", help="Preview the separately budgeted literature memory after extraction")
    args = parser.parse_args(argv)
    if args.execute:
        result = ensure_literature(args.campaign)
    else:
        result = {"research_input": build_literature_input(args.campaign),
                  "literature_memory": literature_config(args.campaign).model_dump()}
    if args.retrieve:
        result = {**result, "literature_memory_delivery": planning_literature(args.campaign, {})}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
