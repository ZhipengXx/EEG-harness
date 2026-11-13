"""Scientific literature inputs and replies; provenance belongs to local storage.

Input construction reads frozen configuration only. The same Pydantic types
generate model schemas and validate replies, including required nullable values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LITERATURE_ROLES = frozenset({"literature_query_planner", "literature_reader"})


class LiteratureMemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    prompt_content_tokens: int = Field(default=6000, ge=1)
    top_k: int = Field(default=8, ge=1)
    max_papers: int = Field(default=10, ge=1)
    search_results_per_query: int = Field(default=20, ge=1, le=25)
    max_queries: int = Field(default=3, ge=1, le=3)
    paper_input_tokens: int = Field(default=24000, ge=1)
    retriever: Literal["openalex"] = "openalex"


class _LiteratureReply(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LiteratureReportedSetup(_LiteratureReply):
    task: Text | None
    training: Text | None
    evaluation: Text | None


class LiteratureFinding(_LiteratureReply):
    topic: Text
    statement: Text
    reported_setup: LiteratureReportedSetup


class LiteratureExtraction(_LiteratureReply):
    findings: list[LiteratureFinding]


class LiteratureQueryPlan(_LiteratureReply):
    queries: list[Text] = Field(min_length=1, max_length=3)


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def literature_config(camp: Path) -> LiteratureMemoryConfig:
    return LiteratureMemoryConfig.model_validate(_read(camp / "goal.json").get("literature_memory") or {})


def literature_prompt(kind: Literal["query", "extract"]) -> str:
    name = {"query": "literature_query_planner.txt", "extract": "literature_reader.txt"}[kind]
    return (Path(__file__).with_name("prompts") / "runtime_v2" / name).read_text(encoding="utf-8")


def _pick(source: dict, keys: tuple[str, ...]) -> dict:
    return {key: source[key] for key in keys if key in source}


def build_literature_input(camp: Path) -> dict:
    """Build three scientific inputs once, without importing a dataset loader.

    Explicit scientific input is required for an unknown task. EEG retrieval
    uses the native protocol rather than the goal's mixed operational objective.
    """
    goal = _read(camp / "goal.json")
    task_type = goal.get("task_type")
    question = goal.get("research_question")
    if question is not None and (not isinstance(question, str) or not question.strip()):
        raise ValueError("literature_research_question_must_be_nonempty_text")
    if task_type != "eeg_image_retrieval":
        if not question or not goal.get("task_setting") or not goal.get("evaluation_protocol"):
            raise ValueError("literature_unknown_task_requires_research_question_task_setting_evaluation_protocol")
        return {"research_question": question.strip(), "task_setting": goal["task_setting"],
                "evaluation_protocol": goal["evaluation_protocol"]}

    protocol = _read(camp / "execution_protocol.json")
    evaluation = _read(camp / "evaluation_contract.json")

    task = {"task": "EEG-to-image retrieval"}
    dataset = protocol.get("dataset")
    if dataset:
        # Names of the existing project adapters; no dataset release is inferred.
        built_in = not (protocol.get("train_dir") or protocol.get("test_dir"))
        task["dataset"] = {"eeg": "THINGS-EEG", "meg": "THINGS-MEG"}.get(dataset, dataset) if built_in else dataset
    mode = goal.get("evaluation_mode")
    task.update(_pick(protocol, ("training_strategy",)))
    if mode != "loso_method_search":
        task.update(_pick(protocol, ("exp_setting",)))
    if question is None:
        target = evaluation.get("generalization_target")
        if target == "held_out_subject" or goal.get("evaluation_mode") == "loso_method_search":
            question = "Which methods and training practices have papers reported to improve cross-subject generalization in EEG-to-image retrieval?"
        elif target == "seen_subject_unseen_stimulus":
            question = "Which methods and training practices have papers reported to improve EEG-to-image retrieval for unseen stimuli from known subjects?"
        else:
            question = "Which methods and training practices have papers reported to improve EEG-to-image retrieval under the supplied task and evaluation setting?"

    conditions = _pick(evaluation, ("primary_metric", "subject_aggregation", "benchmark_scope",
        "generalization_target", "independent_final_test"))
    if "primary_metric" not in conditions and goal.get("primary_metric"):
        conditions["primary_metric"] = goal["primary_metric"]
    gallery = protocol.get("gallery_image_ids")
    if isinstance(gallery, list):
        conditions["gallery_size"] = len(gallery)
    if mode == "loso_method_search":
        conditions["split"] = "leave-one-subject-out"
        conditions["fold_count"] = goal["required_fold_count"]
        # These are native method_suite mode semantics, not a claim inferred
        # from filenames. Avoid re-reading its 10 duplicated fold protocols.
        conditions["checkpoint_selection"] = "source_subject_development_only"
        conditions["held_out_eeg_fit_access"] = False
    else:
        conditions.update(_pick(evaluation, ("val_mode",)))
    if goal.get("evaluation_schedule") == "screen_then_confirm":
        from .screening import SCIENCE_SCOPE

        conditions.update(_pick(SCIENCE_SCOPE, ("checkpoint_selection", "held_out_eeg_fit_access",
            "benchmark_image_features_used_in_development", "independent_final_test")))
    return {"research_question": question.strip(), "task_setting": task, "evaluation_protocol": conditions}


def build_literature_search_text(research_input: dict) -> str:
    """Render scientific meaning for passage matching, without JSON field tokens.

    Native enums and metric identifiers are implementation vocabulary. Translate
    them to paper terminology, retaining the labels that give numbers and boolean
    conditions meaning. Explicit inputs for other tasks can use prose or nested
    scientific fields; this does not validate or change their reader payload.
    """
    terminology = {
        "pooled_subjects": "training pooled across subjects",
        "per_subject": "training separately for each subject",
        "held_out_subject": "cross-subject generalization to held-out subjects",
        "seen_subject_unseen_stimulus": "generalization to unseen stimuli from known subjects",
        "within_subject": "within-subject evaluation",
        "macro_mean": "macro-average across subjects",
        "method_development_benchmark": "method development benchmark",
        "source_subject_development_only": "checkpoint selection using source-subject development data only",
        "other_subjects_test": "validation on source-subject test stimuli",
        "train_holdout": "validation on held-out training stimuli",
        "all_subjects_holdout": "validation on held-out training stimuli across all subjects",
        "custom_train_holdout": "validation on held-out custom training stimuli",
    }
    metrics = {
        "validation.fixed_gallery_top1": "fixed-gallery top-1 retrieval accuracy",
        "benchmark.loso_mean_fixed_gallery_top1": "mean leave-one-subject-out fixed-gallery top-1 retrieval accuracy",
    }
    boolean_conditions = {
        "independent_final_test": ("without an independent final test", "independent final test"),
        "held_out_eeg_fit_access": (
            "held-out subject EEG excluded from training and model fitting",
            "held-out subject EEG available for training and model fitting",
        ),
        "benchmark_image_features_used_in_development": (
            "benchmark image features excluded from development",
            "benchmark image features used during development",
        ),
    }

    def phrases(value: object, field: str = "") -> list[str]:
        if isinstance(value, dict):
            return [phrase for key, item in value.items() for phrase in phrases(item, str(key))]
        if isinstance(value, (list, tuple)):
            return [phrase for item in value for phrase in phrases(item, field)]
        if isinstance(value, bool):
            if field in boolean_conditions:
                return [boolean_conditions[field][int(value)]]
            return [f"{field.replace('_', ' ')}: {'yes' if value else 'no'}"]
        if isinstance(value, (int, float)):
            if field == "gallery_size":
                return [f"retrieval gallery containing {value} images"]
            if field == "fold_count":
                return [f"evaluation across {value} folds"]
            return [f"{field.replace('_', ' ')}: {value}"]
        if isinstance(value, str) and value.strip():
            value = value.strip()
            # exp_setting names a native training regime, not the held-out
            # population: inter-subject with subject="all" still evaluates
            # unseen stimuli from known subjects. Keep the actual enum text;
            # generalization_target supplies the scientific interpretation.
            if field == "primary_metric":
                return [metrics.get(value, value.rsplit(".", 1)[-1].replace("_", " "))]
            return [terminology.get(value, value.replace("_", " "))]
        return []

    # Only the three scientific inputs participate, even if a caller also has
    # local storage/provenance fields alongside them.
    text = [phrase for field in ("research_question", "task_setting", "evaluation_protocol")
            for phrase in phrases(research_input.get(field))]
    return "\n".join(dict.fromkeys(text))
