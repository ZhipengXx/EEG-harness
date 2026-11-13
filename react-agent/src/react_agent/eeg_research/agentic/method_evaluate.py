"""Authorized CPU benchmark inference for one immutable method fold."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from react_agent.eeg_research.agentic.loso_study import digest, identity, read, write_once
from react_agent.eeg_research.agentic.method_suite import aggregate_records, manifest, validate_suite, verify_dependencies


def evaluate(camp: Path, suite_id: str, fold_id: str, *, diagnostics: bool = False) -> dict:
    from .verification_cache import verification_scope
    with verification_scope():
        return _evaluate(camp, suite_id, fold_id, diagnostics=diagnostics)


def _evaluate(camp: Path, suite_id: str, fold_id: str, *, diagnostics: bool = False) -> dict:
    suite = validate_suite(camp, suite_id, verify_data=True)
    fold = next(row for row in suite["folds"] if row["fold_id"] == fold_id)
    directory = camp / "suites" / suite_id / fold_id
    freeze = read(directory / "benchmark_freeze.json")
    if (freeze["suite_hash"] != suite["suite_hash"] or freeze["method_revision"] != suite["method"]["method_revision"]
            or freeze["permission"] != manifest(camp)["benchmark_permission"]
            or freeze["fold_id"] != fold_id or freeze["protocol_hash"] != fold["protocol"]["fingerprint"]):
        raise ValueError("method_benchmark_frozen_identity_mismatch")
    if freeze["evaluation_key"] != identity({k: v for k, v in freeze.items() if k != "evaluation_key"}):
        raise ValueError("method_benchmark_freeze_changed")
    verify_dependencies(freeze["dependencies"])
    target = directory / "benchmark.json"
    diagnostic_path = directory / "benchmark.diagnostics.json"
    existing = None
    if target.exists():
        score = read(target)
        from react_agent.eeg_research.agentic.method_suite import verify_score
        verify_score(camp, suite, score)
        receipt_path = directory / "benchmark.receipt.json"
        if not receipt_path.exists():
            write_once(receipt_path, {"sha256": digest(target), "evaluation_key": freeze["evaluation_key"]})
        receipt = read(receipt_path)
        if score["evaluation_key"] != freeze["evaluation_key"] or receipt["sha256"] != digest(target):
            raise ValueError("method_benchmark_existing_score_changed")
        aggregate_records(suite, [score])
        if not diagnostics or diagnostic_path.exists():
            if diagnostics:
                read_diagnostics(directory, score, restore_receipt=True)
            return score
        existing = score
    os.environ.update(CUDA_VISIBLE_DEVICES="", EEG_TRAIN_DEVICE="cpu", EEG_FINAL_TEST="0")
    for name in ("EEG_CANDIDATE_MODULE", "EEG_CANDIDATE_PATH", "EEG_EVALUATION_IDENTITY"):
        os.environ.pop(name, None)
    from react_agent.eeg_research.agentic.execution_protocol import design_from_protocol
    from react_agent.eeg_training.protocol import split_plan
    from react_agent.eeg_training.train_entry import score_held_out
    from react_agent.eeg_training.inference_consistency import validate_reconstruction_context
    design = design_from_protocol(fold["protocol"], training_seed=0)
    base = Path(fold["protocol"]["data_root"])
    plan = split_plan(base, design)
    if (list(plan.train_subjects) != fold["train_subjects"] or list(map(str, plan.train_files)) != fold["train_files"]
            or list(map(str, plan.val_files)) != fold["development_files"]
            or tuple(map(str, plan.forbidden_files)) != (fold["benchmark_file"],)):
        raise ValueError("method_benchmark_actual_split_mismatch")
    job = Path(freeze["job_dir"])
    if read(job / "job.json").get("status") != "finished" or not validate_reconstruction_context(job, job / "last.ckpt"):
        raise ValueError("method_benchmark_requires_finished_bound_checkpoint")
    started = time.time()
    capture = {}
    metrics = score_held_out(design, base, job, diagnostic_capture=capture)
    if metrics is None:
        raise ValueError("method_benchmark_score_missing")
    if existing is not None:
        if metrics != existing["metrics"]:
            raise ValueError("method_diagnostic_reconstruction_changed_score")
        verify_dependencies(freeze["dependencies"])
        _publish_diagnostics(directory, existing, capture)
        return existing
    body = {"status": "valid", "suite_hash": suite["suite_hash"], "method_revision": suite["method"]["method_revision"],
            "candidate_id": suite["method"]["candidate_id"], "fold_id": fold_id,
            "held_out_subject": fold["held_out_subject"], "train_subjects": fold["train_subjects"],
            "seed": 0, "fidelity": "full", "protocol_hash": fold["protocol"]["fingerprint"],
            "evaluation_key": freeze["evaluation_key"], "evaluator_hash": freeze["evaluator_hash"],
            "checkpoint_hash": freeze["checkpoint_hash"], "metrics": metrics,
            "scope": "method_development_benchmark", "device": "cpu", "precision": "fp32",
            "gpu_seconds": 0.0, "started_at": started, "ended_at": time.time(),
            "freeze_ref": str(directory / "benchmark_freeze.json"), "freeze_sha256": digest(directory / "benchmark_freeze.json")}
    if suite.get("evaluation_schedule") == "screen_then_confirm":
        body["canonical_fold_hash"] = fold["canonical_fold_identity"]["canonical_hash"]
    aggregate_records(suite, [body])
    from react_agent.eeg_research.agentic.method_suite import verify_score
    verify_score(camp, suite, body)
    validate_suite(camp, suite_id, verify_data=True)
    verify_dependencies(freeze["dependencies"])
    # Neither metrics nor checkpoint/state are modified. If publication crashes
    # after score creation, the missing receipt is reconstructed from its exact
    # immutable key on native recovery, never by rescoring for a better value.
    write_once(target, body)
    write_once(directory / "benchmark.receipt.json", {"sha256": digest(target), "evaluation_key": freeze["evaluation_key"]})
    _publish_diagnostics(directory, body, capture)
    return body


def _publish_diagnostics(directory: Path, score: dict, capture: dict) -> None:
    """Persist observations of the same scoring pass without changing its score."""
    from react_agent.eeg_training.diagnostics import representation_from_vectors

    rows, gallery = capture["queries"], capture["gallery_ids"]
    metrics = score["metrics"]
    if (len(rows) != capture["total_query_count"] or len(rows) != metrics["query_count"]
            or len(gallery) != metrics["candidate_count"]
            or sum(row["top1_hit"] for row in rows) / len(rows) != metrics["top1"]
            or sum(row["top5_hit"] for row in rows) / len(rows) != metrics["top5"]):
        raise ValueError("method_diagnostic_scoring_coverage_mismatch")
    coverage = {"population": score["scope"], "queries": len(rows),
        "total_queries": capture["total_query_count"], "gallery_count": len(gallery),
        "embedding_count": len(capture["embeddings"])}
    hub_counts = [{"image_id": image_id, "top1_count": 0, "top5_count": 0} for image_id in gallery]
    for row in rows:
        for j, index in enumerate(row["top_indices"]):
            hub_counts[index]["top5_count"] += 1
            if j == 0:
                hub_counts[index]["top1_count"] += 1
    margins = [row["margin"] for row in rows if row["margin"] is not None]
    representation = representation_from_vectors(capture["embeddings"])
    representation.setdefault("payload", {})["coverage"] = coverage
    report = {"evaluation_key": score["evaluation_key"], "benchmark_sha256": digest(directory / "benchmark.json"),
        "inference_fingerprint": metrics.get("inference_fingerprint"),
        "checkpoint_hash": score["checkpoint_hash"], "evaluator_hash": score["evaluator_hash"],
        "coverage": coverage, "retrieval_errors": {"status": "observed", "payload": {
            "scope": score["scope"], "coverage": coverage, "query_count": len(rows),
            "gallery_size": len(gallery), "top1": metrics["top1"], "top5": metrics["top5"],
            "mean_margin": sum(margins) / len(margins) if margins else None,
            "rank_interval_mean": [sum(row["positive_rank_interval"][i] for row in rows) / len(rows) for i in (0, 1)],
            "hubness": sorted(hub_counts, key=lambda row: (-row["top1_count"], -row["top5_count"]))[:10],
            "definition": "Same torch topk decisions as benchmark; positive rank interval includes ties."}},
        "representation": representation, "query_results": rows, "gallery_counts": hub_counts,
        "embeddings": capture["embeddings"]}
    write_once(directory / "benchmark.diagnostics.json", report)
    write_once(directory / "benchmark.diagnostics.receipt.json", {
        "sha256": digest(directory / "benchmark.diagnostics.json"), "evaluation_key": score["evaluation_key"]})


def read_diagnostics(directory: Path, score: dict, *, restore_receipt: bool = False) -> dict:
    """Read a sidecar against an already accepted score; no suite revalidation."""
    path = directory / "benchmark.diagnostics.json"
    report = read(path)
    if (report["evaluation_key"] != score["evaluation_key"]
            or report["benchmark_sha256"] != digest(directory / "benchmark.json")
            or report["inference_fingerprint"] != score["metrics"].get("inference_fingerprint")):
        raise ValueError("method_diagnostic_binding_changed")
    expected = {"sha256": digest(path), "evaluation_key": score["evaluation_key"]}
    receipt_path = directory / "benchmark.diagnostics.receipt.json"
    if restore_receipt and not receipt_path.exists():
        write_once(receipt_path, expected)
    if read(receipt_path) != expected:
        raise ValueError("method_diagnostic_binding_changed")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camp", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args(argv)
    evaluate(args.camp.resolve(), args.suite, args.fold, diagnostics=args.diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
