# Running EEG-harness

This guide starts a current `single_plan_v1` research campaign. The [architecture map](architecture.md) explains stage responsibilities and the [runtime README](../react-agent/README.md) covers optional paper and experiment memory.

## Install and open the workbench

Use Python **3.11+** for the base project, or **3.12+** when installing the optional literature reader dependencies. From a fresh checkout:

```bash
git clone https://github.com/ZhipengXx/EEG-harness.git
cd EEG-harness/react-agent
uv sync --group dev
cp .env.example .env
```

Commands below run from `react-agent/` and set `PYTHONPATH=src` to use the checkout's source and resources.

```bash
PYTHONPATH=src uv run python -m react_agent.fmri.workbench
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The workbench shows research progress, experiment comparisons, candidate code and recorded artifacts. The [explicit platform demo](http://127.0.0.1:8765/?demo=1#research/demo_ui_workspace) creates synthetic records under `runs/eeg_research_ui_demo/` without an LLM call or training job. The [platform gallery](platform.md) identifies the older screenshots separately from current features.

## Create and run a research campaign

Prepare preprocessed THINGS-EEG / THINGS-MEG data, RN50 image-feature caches, a compatible Torch training interpreter, DeepSeek credentials, and a goal with your evaluation setting and resource limits. Export machine-specific values for CLI processes:

```bash
export EEG_DATA_ROOT=/path/to/data
export EEG_TRAIN_PYTHON=/path/to/training/env/bin/python
```

Inspect a training design and validate a goal before creating the campaign:

```bash
PYTHONPATH=src uv run python -m react_agent.eeg_training.cli run \
  --dataset eeg --exp-setting intra-subject --subject sub-01 \
  --out runs/eeg_training/probe --dry-run

PYTHONPATH=src uv run python -m react_agent.eeg_research.agentic.cli validate-goal \
  --goal configs/goals/eeg_retrieval_v1_9.yaml
```

The bundled goal is a configuration example. Use your own goal for the actual question, evaluation protocol, training seeds and budgets. Both YAML and JSON are accepted. Explicitly select the current execution mode:

```bash
PYTHONPATH=src uv run python -m react_agent.eeg_research.agentic.cli create \
  --campaign my_retrieval_study --goal /path/to/goal.yaml \
  --execution-version single_plan_v1 --gpu 0

PYTHONPATH=src uv run python -m react_agent.eeg_research.agentic.cli start \
  --campaign my_retrieval_study

PYTHONPATH=src uv run python -m react_agent.eeg_research.agentic.cli status \
  --campaign my_retrieval_study
```

`--gpu` accepts a comma-separated list such as `0,1,2,3`; scheduling also follows the goal's concurrency setting. `create` freezes the protocol and writes campaign state. `start` launches the worker and can consume API and GPU budget. Use `pause`, `resume` or `stop` with the same campaign name to control it.

The equivalent execution setting inside a goal is:

```yaml
research_execution:
  version: single_plan_v1
  planning_mode: propose_then_critic
```

The Planner proposes 2–4 alternatives (default target: 3). A separate Critic compares the pool and selects one plan. This normally takes **two planning API calls**, excluding schema corrections, transport retries and optional literature initialization. New-method tasks continue through Designer, Coder and Reviewer. Validation and confirmation tasks reuse existing candidate implementations; diagnostic tasks collect and analyze the requested evidence. Those tasks do not repeat design and coding.

The selected task fixes the **current experiment's** fold scope and completion condition. For example, a two-fold screen can precede ten-fold confirmation; it does not become a ten-fold run merely because the overall research goal requests confirmation. The `screen_then_confirm` schedule and its configuration are described in the [schedule guide](screen-then-confirm.md).

After analysis, local code saves an ExperimentObservation for the next planning round. A separate paid Auditor / Curator pass is not required for this normal transition: `paid_maintenance` defaults to `off` in single-plan campaigns.

## Optional paper and experiment memory

Enable these independently in the goal. [The runtime README](../react-agent/README.md#paper-memory-before-the-first-planner) gives the configuration and reader installation commands.

- **Paper memory:** GPT Researcher provides search and document readers; DeepSeek extracts findings. Initialization happens before the first Planner, with completed work reused on restart. The paper store and retrieval allowance are separate from experiment memory.
- **Experiment memory:** saved observations provide measured facts, analysis and limitations. Planner retrieves current and relevant experiment history. Critic reuses that context and can add proposal-specific experiment matches before choosing a plan.
- **Stage context:** Designer receives the selected task, current experiment setting, parent implementation and interfaces. It does not receive the general history-memory bundle or Critic's selection bookkeeping. Later stages receive their own implementation or measurement inputs.

Paper and experiment retrieval use local embeddings when configured. Retrieval itself does not add an LLM summarization call. Optional literature initialization and enabled maintenance have their own calls in the campaign ledger.

## Current experiment scope

| Area | Current implementation |
| --- | --- |
| Task | EEG / MEG image retrieval; intra-subject and inter-subject settings |
| Candidate changes | Encoder / projection, objective, training transforms and statistics through the candidate hooks |
| Training recipe | Method-specific learning rate, epochs, batch size and weight decay |
| Image target | Precomputed RN50 image features remain fixed; swapping or fine-tuning the image backbone is not implemented |
| Evaluation | Each selected task has its own folds and measurement scope under the campaign protocol |
| Memory | Separate paper findings and experiment observations, with role-specific retrieval delivery |
| Role configuration | Nine core roles, including Critic, plus the optional literature query planner and reader; each has model, thinking, output and timeout settings |

## Other entry points

### Offline numeric screening

This fMRI example uses local numeric checks and the rule policy:

```bash
PYTHONPATH=src uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo

PYTHONPATH=src uv run python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok
```

### Generate from an image and screen the prediction

Configure the external TRIBE checkout, model resources and `TRIBE_PYTHON`, then run:

```bash
PYTHONPATH=src uv run python -m react_agent.fmri.cli check-image \
  --config configs/fmri_check_tribe_image16.yaml \
  --image /path/to/image.jpg --out runs/image_check
```

For planned screening, use [`fmri_check_tribe_image16_agentic.yaml`](../react-agent/configs/fmri_check_tribe_image16_agentic.yaml) with a configured backend. Optional CortexMAE and CLIP resources are described in [asset setup](../assets/README.md). fMRI screening keeps its own state, reports and memory.

### Inspect an exported candidate pack

```bash
PYTHONPATH=src uv run python -m react_agent.eeg_research.agentic.cli evaluate-pack \
  --pack /path/to/candidate_pack --data-root /path/to/data --out runs/pack_eval
```

The default prints the evaluation command. To run it, set `EEG_ALLOW_EVALUATE_PACK=1` and add `--execute`.

## Configuration reference

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | DeepSeek role calls |
| `EEG_<ROLE>_MODEL`, `EEG_<ROLE>_THINKING`, `EEG_<ROLE>_TIMEOUT_S` | Per-role settings; explicit CLI and goal overrides take priority |
| `EEG_DATA_ROOT` | Preprocessed data and image-feature caches |
| `EEG_TRAIN_PYTHON` | Torch interpreter for training and candidate checks |
| `TRIBE_PYTHON` | Interpreter for external TRIBE generation |
| `MODEL` / `TAVILY_API_KEY` | Separate ReAct conversation graph |

See [`.env.example`](../react-agent/.env.example) and [`role_config.py`](../react-agent/src/react_agent/eeg_research/agentic/role_config.py) for available settings. Keep actual credentials local.

<details>
<summary>Legacy action planning</summary>

Existing goals without `research_execution.version: single_plan_v1` retain the legacy path unless `--execution-version single_plan_v1` is supplied at creation. Legacy `planner_mode: compare_options` compares and selects next actions in one response; `single_action` is also available. These modes are distinct from the current Planner → Critic protocol. Persisted campaigns keep their submitted execution configuration.

</details>
