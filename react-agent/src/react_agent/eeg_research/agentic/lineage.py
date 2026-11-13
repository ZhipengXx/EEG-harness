"""Parent source snapshots. parent_id is code lineage, not the experimental control."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from react_agent.eeg_research.agentic.binding import file_sha256

BASELINE_ENTRY = Path(__file__).resolve().parent / "baseline.py"


class LineageError(ValueError):
    """Raised when a parent snapshot cannot be materialized."""


def parent_entry(camp: Path, parent_id: str | None) -> tuple[Path, str]:
    """Resolve the parent source file. Missing parents are an error."""
    name = parent_id or "baseline"
    if name == "baseline":
        return BASELINE_ENTRY, "baseline"
    entry = camp / "candidates" / name / "extension" / "eeg_candidate.py"
    if not entry.is_file():
        raise LineageError(f"parent_source_missing:{name}")
    return entry, name


def materialize(camp: Path, workspace: Path, experiment: dict[str, Any] | None) -> dict[str, Any]:
    """Copy the parent snapshot into reference/parent.py and record lineage.json."""
    experiment = experiment or {}
    source, parent_id = parent_entry(camp, experiment.get("parent_candidate_id"))
    digest = file_sha256(source)
    reference = workspace / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    (reference / "parent.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {
        "parent_candidate_id": parent_id,
        "parent_source_hash": digest,
        "control_id": experiment.get("control_id"),
        "hypothesis_id": experiment.get("hypothesis_id") or experiment.get("id"),
        "parent_is_not_control": True,
    }
    (workspace / "lineage.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def inherited_source(workspace: Path, experiment: dict[str, Any]) -> str | None:
    """Resolve a recipe-only child from its already materialized parent snapshot."""
    from .design_contract import SECTIONS, is_training_only_change
    if (experiment.get("design_contract_version") != "v2"
            or not is_training_only_change(experiment.get("design_brief"))
            or experiment.get("implementation_requirements")
            or any((experiment.get("hook_modes") or {}).get(name) != "inherit" for name in SECTIONS)):
        return None
    source = workspace / "reference/parent.py"
    expected = (experiment.get("inheritance") or {}).get("source_hash")
    if not expected:
        return None
    if file_sha256(source) != expected:
        raise LineageError("inherited_parent_source_changed")
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    if not any(isinstance(node, ast.ClassDef) and node.name == "EEGCandidate" for node in tree.body):
        return None
    parent, _ = parent_entry(workspace.parent.parent, experiment.get("parent_candidate_id"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                return None
            names = [node.module or ""]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        if any((parent.parent / (name.split(".")[0] + ".py")).exists()
               or (parent.parent / name.split(".")[0]).is_dir() for name in names):
            return None
    assignment = "EEGCandidate.candidate_id = " + repr(workspace.name) + "\n"
    last = tree.body[-1]
    if (isinstance(last, ast.Assign) and len(last.targets) == 1
            and isinstance(last.targets[0], ast.Attribute) and last.targets[0].attr == "candidate_id"
            and isinstance(last.targets[0].value, ast.Name) and last.targets[0].value.id == "EEGCandidate"
            and last.col_offset == 0):
        lines = text.splitlines(keepends=True)
        lines[last.lineno - 1:last.end_lineno] = [assignment]
        return "".join(lines)
    return text.rstrip() + "\n\n" + assignment
