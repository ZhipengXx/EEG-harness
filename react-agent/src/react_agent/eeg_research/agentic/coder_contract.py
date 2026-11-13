"""Derive the Coder's concrete tool contract from its approved input.

Hashes and artifact metadata belong to the runtime. The Coder supplies code
locations in a flat requirement map; final source checks still resolve them.
"""
from __future__ import annotations

import copy
from typing import Any

from .coder import (RUNTIME_BOUND_WRITE_ARGS, TOOL_ARGUMENTS, bound_implementation_requirements, tool_catalog,
                    validate_tool_request)


def _request_value(payload: dict[str, Any], key: str) -> Any:
    """Read the frozen request, including its exact request-local pointers."""
    from .context.representations import resolve_request_pointer
    from .context.sources import pointer_escape

    def expand(value: Any, pointer: str) -> Any:
        if isinstance(value, dict):
            if "$context_pointer" in value:
                return expand(resolve_request_pointer(payload, pointer), pointer)
            return {name: expand(child, pointer + "/" + pointer_escape(name)) for name, child in value.items()}
        if isinstance(value, list):
            return [expand(child, pointer + "/" + str(index)) for index, child in enumerate(value)]
        return value

    try:
        return expand(resolve_request_pointer(payload, "/" + key), "/" + key)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _declarations(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    # A frozen compact request must carry the exact same declaration roster
    # used to generate its system schema. In particular, a missing roster must
    # never silently fall back to the legacy unconstrained map on paid recovery.
    if payload.get("coder_context_version") == "coder_implementation_input.v2":
        design = _request_value(payload, "approved_design")
        rows = design.get("implementation_requirements") if isinstance(design, dict) else None
    else:
        spec = _request_value(payload, "experiment_spec")
        spec = spec if isinstance(spec, dict) else {}
        experiment = spec.get("experiment") or spec
        if "implementation_requirements" not in experiment and experiment.get("design_contract_version") != "v2":
            return [], False
        rows = experiment.get("implementation_requirements")
    if not isinstance(rows, list):
        raise ValueError("coder_native_input_invalid:missing_implementation_requirements")
    identifiers = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("requirement_id"), str)
                or not row["requirement_id"].strip()
                or (row.get("probe") is not None and not isinstance(row["probe"], dict))):
            raise ValueError("coder_native_input_invalid:invalid_implementation_requirements")
        identifiers.append(row["requirement_id"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("coder_native_input_invalid:duplicate_requirement_id")
    return rows, True


def empty_implementation_map_is_local(payload: dict[str, Any]) -> bool:
    """Only an explicitly empty approved roster makes the whole map native-owned."""
    declarations, known = _declarations(payload)
    return known and not declarations


_MODULE_PATHS = ("input_module_path", "representation_module_path")


def _probe_schema(requirement: dict[str, Any]) -> dict[str, Any]:
    declared = requirement.get("probe") or {}
    properties = {key: {"const": copy.deepcopy(value)} for key, value in declared.items()}
    for key in _MODULE_PATHS:
        if not declared.get(key):
            properties[key] = {"type": ["string", "null"]}
    return {"type": ["object", "null"], "properties": properties,
            # The existing binder permits an undeclared key only when its
            # supplied value is None; preserve that edge case without inviting
            # the model to invent scientific probe parameters.
            "additionalProperties": {"type": "null"}}


def _map_schema(declarations: list[dict[str, Any]], *, known: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "description": (
        "Flat requirement_id to source_refs/optional probe. Runtime supplies artifact and hash metadata. "
        "Only include bindings being supplied; omission preserves the existing native behavior.")}
    if known:
        result["properties"] = {row["requirement_id"]: {
            "type": "object", "properties": {
                "source_refs": {"type": ["array", "null"], "items": {"type": "string"}},
                "probe": _probe_schema(row),
            }, "additionalProperties": False,
        } for row in declarations}
        result["additionalProperties"] = False
    else:
        # Legacy callers without an approved requirement roster retain their
        # existing generic map argument; a roster is never guessed from history.
        result["additionalProperties"] = True
    return result


def coder_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """One task-specific catalog, JSON Schema and fillable finish template.

    Requirement IDs are allowed properties rather than required properties:
    optional declarations and existing source bindings retain their semantics.
    """
    declarations, known = _declarations(payload)
    local_map = known and not declarations
    mapping_schema = _map_schema(declarations, known=known)
    catalog = tool_catalog()
    branches = []
    for tool, fields in TOOL_ARGUMENTS.items():
        if tool == "finish_patch" and local_map:
            fields = {name: spec for name, spec in fields.items() if name != "implementation_map"}
            catalog[tool]["args"].pop("implementation_map", None)
        if tool in {"apply_candidate_patch", "edit_candidate_code"}:
            # The runtime binds the write target and its CAS base; the model never sends them.
            fields = {name: spec for name, spec in fields.items() if name not in RUNTIME_BOUND_WRITE_ARGS}
            catalog[tool]["args"] = {name: spec for name, spec in catalog[tool]["args"].items()
                                     if name not in RUNTIME_BOUND_WRITE_ARGS}
        args_schema: dict[str, Any] = {"type": "object"}
        if fields is None:
            args_schema["additionalProperties"] = True
        else:
            properties = {name: {"type": kind} for name, (kind, _required) in fields.items()}
            args_schema.update(properties=properties, additionalProperties=False)
            required = [name for name, (_kind, required) in fields.items() if required]
            if required:
                args_schema["required"] = required
            if tool == "read_code":
                for name in ("start", "end"):
                    properties[name]["minimum"] = 1
            elif tool == "edit_candidate_code":
                properties["edits"].update(minItems=1, maxItems=20, items={"oneOf": [
                    {"type": "object", "properties": {"old": {"type": "string", "minLength": 1},
                                                          "new": {"type": "string"}},
                     "required": ["old", "new"], "additionalProperties": False},
                    {"type": "object", "properties": {"start": {"type": "integer", "minimum": 1},
                                                          "end": {"type": "integer", "minimum": 1},
                                                          "new": {"type": "string"}},
                     "required": ["start", "end", "new"], "additionalProperties": False},
                ]})
            elif tool == "finish_patch" and not local_map:
                properties["implementation_map"] = copy.deepcopy(mapping_schema)
            elif tool == "request_design_revision":
                properties["kind"]["enum"] = ["design_conflict", "scientific_replan"]
                for name in ("requirement_ids", "evidence_refs"):
                    properties[name]["items"] = {"type": "string", "minLength": 1}
        catalog[tool]["args_schema"] = copy.deepcopy(args_schema)
        branches.append({"type": "object", "properties": {"tool": {"const": tool}, "args": args_schema},
                         "required": ["tool", "args"], "additionalProperties": False})
    template = {}
    for requirement in declarations:
        row: dict[str, Any] = {"source_refs": []}
        declared_probe = requirement.get("probe") or {}
        if declared_probe or requirement.get("verification") == "ordered_window_slots":
            pending = {key: "" for key in _MODULE_PATHS if not declared_probe.get(key)}
            if pending:
                row["probe"] = pending
        template[requirement["requirement_id"]] = row
    finish_args: dict[str, Any] = {"summary": ""}
    if not local_map:
        finish_args["implementation_map"] = template
    return {
        "tools": catalog,
        "output_schema": {"title": "CoderToolRequest", "oneOf": branches},
        "finish_patch_template": {"tool": "finish_patch", "args": finish_args},
        "tool_request_example": {"tool": "read_code", "args": {
            "path": "reference/parent.py", "start": 1, "end": 120}},
    }


def validate_coder_request(reply: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Validate model-owned fields and bind native arguments without mutating the reply."""
    try:
        declarations, known = _declarations(payload)
    except ValueError as exc:
        tool, args, _ = validate_tool_request(reply)
        return tool, args, {"ok": False, "error": "coder_native_input_invalid", "detail": str(exc)}
    local_map = known and not declarations
    wire_reply = reply
    if (local_map and isinstance(reply, dict) and reply.get("tool") == "finish_patch"
            and isinstance(reply.get("args"), dict)):
        # Ignore only this locally determined field, before checking its model-supplied type.
        wire_reply = {**reply, "args": {key: value for key, value in reply["args"].items()
                                        if key != "implementation_map"}}
    tool, args, error = validate_tool_request(wire_reply)
    if error is None and tool == "finish_patch" and local_map:
        return tool, {**args, "implementation_map": {}}, None
    if error is not None or tool != "finish_patch" or "implementation_map" not in args:
        return tool, args, error
    if not known:
        return tool, args, None
    mapping = args["implementation_map"]
    declared_ids = {row["requirement_id"] for row in declarations}
    if set(mapping) - declared_ids:
        return tool, args, {"ok": False, "error": "implementation_map_unknown_requirement",
                            "allowed_requirement_ids": [row["requirement_id"] for row in declarations],
                            "unknown_requirement_ids": sorted(set(mapping) - declared_ids),
                            "field": "args.implementation_map",
                            "detail": "Use the flat map in finish_patch_template; runtime writes hash/artifact metadata."}
    for requirement_id, row in mapping.items():
        field = "args.implementation_map." + requirement_id
        if not isinstance(row, dict) or set(row) - {"source_refs", "probe"}:
            return tool, args, {"ok": False, "error": "implementation_map_invalid_row", "field": field}
        refs = row.get("source_refs") or []
        if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
            return tool, args, {"ok": False, "error": "implementation_map_invalid_source_reference", "field": field + ".source_refs"}
        if row.get("probe") is not None and not isinstance(row["probe"], dict):
            return tool, args, {"ok": False, "error": "implementation_map_invalid_probe", "field": field + ".probe"}
    # Reuse the existing semantic binder. It only derives a private copy, does
    # not check source locations and cannot overwrite the approved declaration.
    try:
        bound_implementation_requirements(declarations,
            {"requirements": mapping, "source_sha256": "request", "approved_spec_hash": "request"},
            source_sha256="request", spec_hash="request")
    except (TypeError, ValueError) as exc:
        return tool, args, {"ok": False, "error": str(exc)}
    return tool, args, None


def coder_request_normalization(reply: Any, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Describe a successful native binding separately from the original model response."""
    normalized = {"tool": tool, "args": args}
    if tool == "finish_patch" and args.get("implementation_map") == {} and normalized != reply:
        return {"kind": "native_empty_implementation_map", "normalized_reply": normalized,
                "raw_model_reply_preserved": True}
    return None


def previous_empty_map_contract_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Reconstruct only the former derived empty-map fields for exact paid replay.

    All source, design, history and native identity fields remain unchanged. This
    payload is used only to match an existing digest, never to send a new request.
    """
    if not empty_implementation_map_is_local(payload):
        return None
    contract = coder_contract(payload)
    if (payload.get("tools") != contract["tools"]
            or payload.get("finish_patch_template") != contract["finish_patch_template"]):
        return None
    previous = copy.deepcopy(payload)
    finish = previous["tools"]["finish_patch"]
    finish["args"]["implementation_map"] = {"type": "object", "required": False}
    finish["args_schema"]["properties"]["implementation_map"] = _map_schema([], known=True)
    previous["finish_patch_template"]["args"]["implementation_map"] = {}
    return previous


def coder_request_feedback(reply: Any, error: dict[str, Any]) -> dict[str, Any]:
    """Point to the already delivered schema instead of resending the tool catalog."""
    tool = reply.get("tool") if isinstance(reply, dict) else None
    feedback = {"failure_kind": "format_error", "validation_error": error,
        "instruction": "Correct only the reported tool envelope and parameter format against OUTPUT_SCHEMA. Return one tool/args object for the same current step; no tool executed or source changed."}
    if tool == "finish_patch":
        feedback["instruction"] += " Use FINISH_PATCH_TEMPLATE already supplied in the system message. Preserve valid source_refs and probe bindings; correct only the reported fields."
    return feedback
