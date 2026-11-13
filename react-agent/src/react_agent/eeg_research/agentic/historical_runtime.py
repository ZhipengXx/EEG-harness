"""Historical source verification and exact native dispatch-patch receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _dispatch_paths() -> set[str]:
    root = Path(__file__).resolve().parent
    return {str(root / name) for name in ("jobs.py", "method_suite.py", "screening.py", "multigpu_check.py")}


def _patch_manifest(camp: Path) -> dict:
    from .loso_study import identity, read
    manifest = read(camp / "method_search_manifest.json")
    if (not manifest.get("framework_dependencies") or manifest.get("manifest_hash") !=
            identity({key: value for key, value in manifest.items() if key != "manifest_hash"})):
        raise ValueError("runtime_dispatch_patch_manifest_invalid")
    return manifest


def dispatch_patch_receipts(camp: Path) -> tuple[Path, ...]:
    """Immutable registrations; later fixes leave previously cited receipts intact."""
    original = camp / "runtime_dispatch_patch.json"
    return ((original,) if original.is_file() else ()) + tuple(sorted(
        (camp / "runtime_dispatch_patches").glob("*.json")))


def record_dispatch_patch(camp: Path, replacements: list[dict], reason: str) -> dict:
    """Record exact dispatch-file replacements without rewriting scientific records."""
    import fcntl
    from .loso_study import digest, identity, read, write_once

    camp = Path(camp).resolve()
    if not reason.strip() or not replacements:
        raise ValueError("runtime_dispatch_patch_reason_and_replacements_required")
    with (camp / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("runtime_dispatch_patch_worker_active") from exc
        manifest = _patch_manifest(camp)
        dependencies = {str(Path(row["path"]).resolve()): row["sha256"]
                        for row in manifest["framework_dependencies"]}
        selected = {}
        for row in replacements:
            path = str(Path(row["path"]).resolve())
            if path not in _dispatch_paths() or path in selected:
                raise ValueError("runtime_dispatch_patch_path_not_allowed:" + path)
            if (path not in dependencies or row.get("old_sha256") != dependencies[path] or not row.get("new_sha256")
                    or row["new_sha256"] == row["old_sha256"]):
                raise ValueError("runtime_dispatch_patch_frozen_hash_mismatch:" + path)
            selected[path] = {"path": path, "old_sha256": row["old_sha256"], "new_sha256": row["new_sha256"]}
        for path, expected in dependencies.items():
            expected = selected[path]["new_sha256"] if path in selected else expected
            if digest(Path(path)) != expected:
                raise ValueError("runtime_dispatch_patch_dependency_changed:" + path)
        receipt = {"campaign_root": str(camp), "manifest_hash": manifest["manifest_hash"],
                   "replacements": [selected[path] for path in sorted(selected)], "reason": reason.strip()}
        receipt["receipt_hash"] = identity(receipt)
        target = camp / "runtime_dispatch_patch.json"
        for prior_path in dispatch_patch_receipts(camp):
            prior = read(prior_path)
            if prior == receipt:
                return receipt
            if prior.get("replacements") == receipt["replacements"]:
                raise ValueError("runtime_dispatch_patch_conflict")
        if target.exists():
            target = camp / "runtime_dispatch_patches" / (receipt["receipt_hash"] + ".json")
        write_once(target, receipt)
        return receipt


def dispatch_patch_matches(camp: Path, path: Path, expected: str, actual: str) -> bool:
    """Accept only an exact old/new pair bound to this campaign's original manifest."""
    from .loso_study import identity, read
    from .verification_cache import file_memo

    camp = Path(camp).resolve()
    receipt_paths = dispatch_patch_receipts(camp)
    if not isinstance(expected, str) or not isinstance(actual, str) or not receipt_paths:
        return False

    def entries():
        try:
            manifest = _patch_manifest(camp)
            dependencies = {str(Path(row["path"]).resolve()): row["sha256"]
                            for row in manifest["framework_dependencies"]}
            result = set()
            for receipt_path in receipt_paths:
                receipt = read(receipt_path)
                if (receipt.get("campaign_root") != str(camp)
                        or receipt.get("manifest_hash") != manifest["manifest_hash"]
                        or receipt.get("receipt_hash") != identity({key: value for key, value in receipt.items()
                                                                   if key != "receipt_hash"})):
                    return frozenset()
                for row in receipt["replacements"]:
                    if (row["path"] not in _dispatch_paths()
                            or dependencies.get(row["path"]) != row["old_sha256"]):
                        return frozenset()
                    result.add((row["path"], row["old_sha256"], row["new_sha256"]))
            return frozenset(result)
        except (OSError, ValueError, KeyError, TypeError):
            return frozenset()

    allowed = file_memo("runtime_dispatch_patch", (*receipt_paths, camp / "method_search_manifest.json"),
                        (str(camp), str(Path(__file__).resolve().parent)), entries)
    return (str(Path(path).resolve()), expected, actual) in allowed


def checkout_preview(camp: Path, checkout: Path, original_project: Path) -> dict:
    manifest = json.loads((camp / "method_search_manifest.json").read_text())
    dependencies = manifest.get("framework_dependencies") or []
    if not dependencies:
        raise ValueError("historical_frozen_framework_dependencies_missing")
    rows = []
    for dependency in dependencies:
        original = Path(dependency["path"])
        if not original.is_absolute():
            raise ValueError("historical_dependency_absolute_path_required")
        try:
            relative = original.relative_to(original_project.resolve())
        except ValueError:
            rows.append({"original_path": str(original), "status": "external_dependency_requires_original_environment"})
            continue
        proposed = (checkout.resolve() / relative).resolve()
        if not proposed.is_relative_to(checkout.resolve()):
            raise ValueError("historical_checkout_dependency_escape")
        actual = hashlib.sha256(proposed.read_bytes()).hexdigest() if proposed.is_file() else None
        rows.append({"original_path": str(original), "checkout_path": str(proposed),
                     "expected_sha256": dependency["sha256"], "actual_sha256": actual,
                     "status": "matches" if actual == dependency["sha256"] else "missing" if actual is None else "mismatch"})
    return {"preview_only": True, "campaign": str(camp.resolve()), "checkout": str(checkout.resolve()),
            "framework_files": rows, "source_checkout_matches": all(row["status"] == "matches" for row in rows),
            "native_resume_authorized": False,
            "required_original_absolute_mount": str(original_project.resolve()),
            "data_environment_and_process_verification": "required_in_original_runtime_before_resume",
            "instruction": "Map matching old source into its original absolute path inside an isolated mount/container; keep original data/campaign paths and native verification. This preview writes nothing and starts no worker."}
