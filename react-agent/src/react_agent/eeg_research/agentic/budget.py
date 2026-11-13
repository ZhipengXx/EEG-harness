"""Campaign resource ledger. Attempts charge; JSON repair does not bypass the cap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COST = "cost.json"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def snapshot(camp: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Used, reserved, and remaining resources. Missing files do not reset counters."""
    cost = _read(camp / COST)
    gpu_limit = float(state.get("max_gpu_seconds") or cost.get("max_gpu_seconds") or 0.0)
    gpu_used = float(cost.get("gpu_seconds_used") or 0.0)
    gpu_reserved = 0.0
    llm_limit = int(state.get("max_llm_calls") or 0)
    llm_used = int(cost.get("llm_calls") if isinstance(cost.get("llm_calls"), int) else state.get("llm_calls") or 0)
    jobs_limit = int(state.get("max_training_jobs") or 0)
    jobs_used = int(state.get("training_jobs") or cost.get("training_jobs") or 0)
    gpu_left = gpu_limit - gpu_used
    result = {
        "gpu_seconds_used": gpu_used,
        "gpu_seconds_reserved": gpu_reserved,
        "gpu_seconds_left": gpu_left,
        "gpu_seconds_limit": gpu_limit,
        "llm_calls_used": llm_used,
        "llm_calls_left": llm_limit - llm_used,
        "llm_calls_limit": llm_limit,
        "training_jobs_used": jobs_used,
        "training_jobs_left": jobs_limit - jobs_used,
        "training_jobs_limit": jobs_limit,
        "baseline_jobs": int(state.get("baseline_jobs") or 0),
    }
    from .owner_budget import enabled, snapshot as owner_snapshot, current_owner
    if enabled(camp):
        active = current_owner(camp)
        pointer = state.get("selected_plan") or {}
        owner_id = active.get("owner_id") if active else pointer.get("continuation_owner") or pointer.get("budget_owner")
        result["owner_budget"] = owner_snapshot(camp,state,owner_id=owner_id)
    return result


def charge_gpu(camp: Path, state: dict[str, Any], seconds: float, *, job_id: str | None = None) -> bool:
    """Record actual GPU seconds. One job settles once."""
    cost = _read(camp / COST)
    settled = [str(item) for item in cost.get("settled_job_ids") or []]
    if job_id and job_id in settled:
        return False
    used = max(0.0, float(seconds))
    state["gpu_seconds_reserved"] = 0.0
    state["gpu_seconds_left"] = float(state.get("gpu_seconds_left") or 0.0) - used
    cost["gpu_seconds_used"] = float(cost.get("gpu_seconds_used") or 0.0) + used
    cost["training_jobs"] = int(state.get("training_jobs") or 0)
    if job_id:
        cost["settled_job_ids"] = settled + [job_id]
    cost.setdefault("api_usd", None)
    if job_id:
        from .owner_budget import settle_training
        settle_training(camp, job_id, used)
    _write(camp / COST, cost)
    return True


def wall_deadline(gpu_seconds_left: float, n_gpu: int, *, now: float) -> float | None:
    """Wall-clock deadline from remaining GPU-seconds and assigned cards."""
    cards = max(1, int(n_gpu))
    remaining = float(gpu_seconds_left)
    if remaining <= 0:
        return now
    return now + remaining / cards


def reserve_phase(camp: Path, state: dict[str, Any], request: dict[str, Any], *, replace_receipt_id: str | None = None) -> dict[str, Any]:
    """Bind immutable phase work and ownership without reserving future cost.

    Actual jobs and GPU usage are admitted at the training launch boundary.
    Existing receipt identities remain readable, including historical estimates.
    """
    import fcntl
    from react_agent.eeg_research.agentic.loso_study import identity
    required = {"phase", "candidate_suite_id", "suite_hash", "method_revision",
                "selection_manifest_hash", "fold_work"}
    if not required <= request.keys():
        raise ValueError("phase_reservation_request_incomplete")
    work = request["fold_work"]
    if (not isinstance(work, dict) or set(work) != {"candidate", "baseline"}
            or any(not isinstance(ids, list) or any(not isinstance(fold, str) for fold in ids)
                   or len(ids) != len(set(ids)) for ids in work.values())):
        raise ValueError("phase_reservation_fold_work_invalid")
    path = camp / "phase_reservations.json"
    camp.mkdir(parents=True, exist_ok=True)
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(path) or {"schema_version": "eeg_research.phase_reservations.v1", "receipts": {}}
        receipt_id = "phase_" + identity(request)
        existing = ledger["receipts"].get(receipt_id)
        if existing and existing["status"] == "active":
            state["phase_reservation_id"] = receipt_id
            state["gpu_seconds_reserved"] = 0.0
            state["method_jobs_reserved"] = 0
            state.pop("method_gpu_reservation", None)
            return existing
        if existing:
            raise ValueError("phase_reservation_already_released")
        if state.get("status") in {"paused", "cancelled", "completed"} or state.get("pause_after_step"):
            raise ValueError("phase_admission_user_stopped")
        previous = ledger["receipts"].get(replace_receipt_id) if replace_receipt_id else None
        if replace_receipt_id and (not previous or previous["status"] != "active"
                or set(previous["consumed_job_ids"]) - set(previous.get("settled_phase_job_ids", []))):
            raise ValueError("phase_re_admission_requires_settled_previous_receipt")
        from .owner_budget import enabled, reserve_phase_owner
        owner_phase = None
        if enabled(camp):
            owner_phase = reserve_phase_owner(camp, receipt_id,
                replace_owner_id=(previous or {}).get("budget_owner"))
        receipt = {"receipt_id": receipt_id, "request": request, "request_hash": identity(request),
                   "status": "active", "consumed_job_ids": [], "released_reason": None,
                   "settled_phase_job_ids": []}
        if owner_phase:
            receipt.update(budget_owner=owner_phase["owner_id"])
        ledger["receipts"][receipt_id] = receipt
        if previous:
            previous.update(status="released", released_reason="superseded_by_explicit_failed_process_retry",
                            superseded_by_receipt_id=receipt_id)
        _write(path, ledger)
        state["phase_reservation_id"] = receipt_id
        state["gpu_seconds_reserved"] = 0.0
        state["method_jobs_reserved"] = 0
        state.pop("method_gpu_reservation", None)
        return receipt


def phase_reservation(camp: Path, receipt_id: str) -> dict[str, Any]:
    from react_agent.eeg_research.agentic.loso_study import identity
    row = (_read(camp / "phase_reservations.json").get("receipts") or {}).get(receipt_id)
    if not row or row["request_hash"] != identity(row["request"]) or receipt_id != "phase_" + row["request_hash"]:
        raise ValueError("phase_reservation_receipt_invalid")
    return row


def consume_phase_job(camp: Path, receipt_id: str, job_id: str) -> None:
    """Claim an upcoming job and bind it to its phase before publishing intent."""
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        if job_id in row["consumed_job_ids"]:
            return
        if row["status"] != "active":
            raise ValueError("phase_launch_reservation_released")
        from .owner_budget import enabled, claim_training
        if enabled(camp):
            claim_training(camp, job_id, phase_owner_id=row.get("budget_owner"))
        row["consumed_job_ids"].append(job_id)
        ledger["receipts"][receipt_id] = row
        _write(camp / "phase_reservations.json", ledger)


def settle_phase_job(camp: Path, state: dict[str, Any], receipt_id: str, job_id: str) -> None:
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        if job_id not in row["consumed_job_ids"]:
            raise ValueError("phase_job_settlement_missing_intent")
        if job_id not in row.setdefault("settled_phase_job_ids", []):
            row["settled_phase_job_ids"].append(job_id)
            ledger["receipts"][receipt_id] = row
            _write(camp / "phase_reservations.json", ledger)
        state["gpu_seconds_reserved"] = 0.0
        state["method_jobs_reserved"] = 0


def settle_phase_feedback(camp: Path, receipt_id: str, evidence_id: str, allocation: int | None = None) -> None:
    """Record completed method feedback; actual API charges settle separately."""
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        completed = row.setdefault("feedback_completed_evidence_ids", [])
        if evidence_id in completed:
            return
        if row["status"] != "active":
            raise ValueError("phase_feedback_reservation_released")
        completed.append(evidence_id)
        ledger["receipts"][receipt_id] = row
        _write(camp / "phase_reservations.json", ledger)


def release_phase(camp: Path, state: dict[str, Any], receipt_id: str, reason: str) -> dict[str, Any]:
    """Close reconciled phase work. Never change API/job/GPU charges."""
    import fcntl
    with (camp / "phase_reservations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _read(camp / "phase_reservations.json")
        row = phase_reservation(camp, receipt_id)
        if row["status"] == "active":
            unsettled = set(row["consumed_job_ids"]) - set(row.get("settled_phase_job_ids", []))
            if unsettled:
                raise ValueError("phase_release_requires_process_cost_reconciliation")
            if row.get("budget_owner"):
                from .owner_budget import release_phase_owner
                release_phase_owner(camp,row["budget_owner"])
            row.update(status="released", released_reason=reason)
            ledger["receipts"][receipt_id] = row
            _write(camp / "phase_reservations.json", ledger)
        state["gpu_seconds_reserved"] = 0.0
        state["method_jobs_reserved"] = 0
        state.pop("method_gpu_reservation", None)
        if state.get("phase_reservation_id") == receipt_id:
            state["phase_reservation_id"] = None
        return row
