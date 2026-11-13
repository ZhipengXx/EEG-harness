"""What a candidate can and cannot use. Shared by the coder, the planner and the reviewer."""

from __future__ import annotations

from typing import Any

CANDIDATE_INTERFACE = {
    "input": "[batch, 17, 250] float32 EEG; channels and time window are fixed by the protocol",
    "output": "[batch, 1024] finite embedding; the frozen RN50 image feature is the target",
    "build_encoder": (
        "build_encoder(self, input_spec, model_config=None) -> any torch.nn.Module defined in "
        "extension/eeg_candidate.py; it may replace EEGProjectLayer entirely with convolution, "
        "temporal pooling, attention or a new projection head"
    ),
    "fit_statistics": (
        "optional fit_statistics(self, encoder, stats); called once before training with per-channel "
        "mean and std computed over the training files only; store them as registered buffers, "
        "not parameters; validation reuses them"
    ),
    "build_training_transform": (
        "optional build_training_transform(self, transform_config=None); train-only EEG transform; "
        "validation and evaluate-only must not call a stochastic transform"
    ),
    "build_training_objective": (
        "optional build_training_objective(self, objective_config=None); receives EEG embedding, "
        "frozen image embedding and runtime positive relations; learnable parameters enter AdamW. "
        "Return a scalar loss Tensor, or {'loss': total_loss, 'loss_terms': {'infonce': base_loss, "
        "'covariance_weighted': lambda_value * penalty}} for named weighted contributions. "
        "Terms are scalar Tensors on the loss device with the same names on every replica; "
        "runtime detaches and records them from this forward without changing the total loss. "
        "The same dict may optionally contain 'diagnostics': {'main.effective_scale': scale_used, "
        "'main.temperature': temperature_used}, with named scalar Tensors on the loss device and stable "
        "keys across replicas; use branch names for multiple scales. Runtime samples these actual "
        "forward values separately from averaged loss terms; no diagnostic export is required."
    ),
    "training_configuration": {
        "fields": ["epochs", "lr", "batch_size", "weight_decay"],
        "optimizer": "AdamW", "scheduler": "none",
        "usage": "Designer selects or inherits the training configuration; the runtime applies it. Coder may reuse unchanged hook code.",
    },
    "implementation_requirements": {
        "purpose": "Approved core implementation promises are checked individually; predicted Top-1 gains are scientific hypotheses, not pretraining implementation gates.",
        "fields": "requirement_id, kind (implementation/scientific_hypothesis), core, expected_behavior, code_locations, verification, probe",
        "verification": {
            "source_review": "Reviewer identifies the current execution path and exact current source/check receipt references.",
            "ordered_window_slots": "Only for an explicitly promised ordered temporal representation. Designer declares only slot_count, window_axis and layout; Coder binds real input_module_path at the pool input and representation_module_path at its downstream consumer. The checker injects distinct window values at that boundary and observes the real consumer input.",
        },
        "limits": "No universal global-pooling ban, raw-EEG permutation rule or predicted improvement requirement. Missing core implementation evidence stays unverified and cannot authorize training.",
    },
    "not_available": [
        "subject id or any per-subject metadata as model input",
        "image id as encoder input",
        "per-subject statistics",
        "validation or test data before evaluation",
        "changes to the loss target, evaluator, gallery, split or image features",
    ],
    "runtime_guarantees": [
        "the validation gallery and query ids are fixed by the contract fingerprint",
        "image features come from the frozen DirectT RN50 cache",
        "train, validation and held-out files are assigned by the frozen split manifest",
        "the trainer imports only extension/eeg_candidate.py and records its hash",
        "the runtime executes the approved candidate training configuration; fidelity and checkpoint selection follow the evaluation protocol",
        "negative_sampling_policy selects data_parallel_local or global_batch; custom objectives inherit the declared policy",
    ],
}


def training_batch_sizes(protocol: dict[str, Any], *, sample_count: int | None = None) -> tuple[int, int]:
    """Use the production loader's capped batch and its actual final batch."""
    samples = len(protocol.get("train_query_ids") or []) if sample_count is None else sample_count
    batch = min(int(protocol["batch_size"]), samples)
    return (batch, (samples % batch) or batch) if batch > 0 else (0, 0)


def training_batch_partition(batch_size: int, n_cards: int) -> list[int]:
    """Sizes of DataParallel's tensor.chunk shards, including a short final shard."""
    if batch_size < 1 or n_cards < 1:
        return []
    chunk_size = (batch_size + n_cards - 1) // n_cards
    return [min(chunk_size, batch_size - start) for start in range(0, batch_size, chunk_size)]


def training_replica_count(first_batch: int, n_cards: int) -> int:
    return len(training_batch_partition(first_batch, n_cards))


def candidate_interface(protocol: dict[str, Any] | None) -> dict[str, Any]:
    """Build the coder/reviewer input spec from the frozen protocol. MEG must not use EEG shapes."""
    payload = dict(CANDIDATE_INTERFACE)
    geom = (protocol or {}).get("input_geometry") if isinstance(protocol, dict) else None
    if not isinstance(geom, dict) or geom.get("c_num") is None:
        if isinstance(protocol, dict) and protocol.get("dataset"):
            from react_agent.eeg_training.protocol import geometry

            geom = geometry(str(protocol["dataset"]))
        else:
            geom = {"c_num": 17, "timesteps": [0, 250], "channels": None}
    c_num = int(geom["c_num"])
    timesteps = list(geom["timesteps"])
    length = int(timesteps[1]) - int(timesteps[0])
    gallery = (protocol or {}).get("gallery_image_ids") or (protocol or {}).get("validation_image_ids") or []
    payload["c_num"] = c_num
    payload["timesteps"] = timesteps
    payload["channels"] = geom.get("channels")
    payload["dataset"] = None if protocol is None else protocol.get("dataset")
    payload["input"] = (
        f"[batch, {c_num}, {length}] float32; channels and time window are fixed by the protocol"
    )
    payload["gallery_size"] = len(gallery) if gallery else None
    payload["runtime_guarantees"] = list(CANDIDATE_INTERFACE["runtime_guarantees"])
    if gallery:
        payload["runtime_guarantees"] = [
            f"the validation gallery ({len(gallery)} images) and query ids are fixed by the contract fingerprint",
            *CANDIDATE_INTERFACE["runtime_guarantees"][1:],
        ]
    if isinstance(protocol, dict) and protocol.get("evaluation_mode") == "loso_method_search":
        total_batch, tail_batch = training_batch_sizes(protocol)
        samples = len(protocol.get("train_query_ids") or [])
        payload["training_configuration"] = {
            **CANDIDATE_INTERFACE["training_configuration"], "min_batch_size": len(protocol["gpu"]),
            "batch_partition": ("Single GPU: the entire batch executes on one device; local negatives cover the whole batch."
                if len(protocol["gpu"]) == 1 else
                "DataParallel tensor.chunk; the first batch must populate every declared GPU."),
        }
        payload["multigpu_check"] = {
            "enabled": True, "evaluation_mode": "loso_method_search", "gpu": list(protocol["gpu"]),
            "batch_size": total_batch, "training_sample_count": samples,
            "tail_batch_size": tail_batch,
            "lr": protocol["lr"], "weight_decay": protocol["weight_decay"],
            "negative_sampling_policy": protocol["negative_sampling_policy"],
            "checkpoint_policy": protocol["checkpoint_policy"], "protocol_fingerprint": protocol["fingerprint"],
            "data_scope": "synthetic_training_inputs_only;no_held_out_EEG_or_images",
        }
        payload["runtime_guarantees"][-1] = "Custom objectives must explicitly support data_parallel_local; the same frozen devices, total batch and actual tail partition are tested by the native synthetic GPU checker before Reviewer approval."
    return payload
