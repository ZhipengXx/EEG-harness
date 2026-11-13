# EEG-harness runtime

Technical reference for the EEG / MEG image-retrieval runtime, its memory stores, the training worker, and the fMRI workbench. Start with the [project overview](../README.md) for the research loop and the [running guide](../docs/running.md) for campaign creation and execution.

## Setup and entry points

Run commands from this directory. The examples set `PYTHONPATH=src` so the selected interpreter loads this checkout's source.

```bash
uv sync --group dev
cp .env.example .env
PYTHONPATH=src uv run python -m react_agent.fmri.workbench
```

Open [localhost:8765](http://127.0.0.1:8765).

| Module / graph | Purpose |
| --- | --- |
| `react_agent.eeg_research.agentic.cli` / `eeg_research` | Create, control and inspect research campaigns |
| `react_agent.eeg_training.cli` | Inspect or run a retrieval design |
| `react_agent.fmri.workbench` | Local interface for research and screening |
| `react_agent.fmri.cli` / `fmri_check` | Screen an existing fMRI series |
| `react_agent.fmri.cli check-image` / `fmri_pipeline` | Generate a response with a configured backend, then screen it |
| `react_agent.graph` / `agent` | Separate ReAct conversation entry |

[`langgraph.json`](langgraph.json) registers the Studio views. The EEG graph wraps an existing campaign tick as load → step → summarize. The conversation entry originated from the [LangGraph ReAct template](https://github.com/langchain-ai/react-agent).

<details>
<summary>Offline fMRI screening example</summary>

```bash
PYTHONPATH=src uv run python -m react_agent.fmri.cli make-demo --out examples/fmri_demo
PYTHONPATH=src uv run python -m react_agent.fmri.cli check \
  --sample examples/fmri_demo/ok_numeric.json \
  --config configs/fmri_check.yaml \
  --backend none --policy rule \
  --out runs/rule_ok
```

This example uses synthetic data and the rule policy. Inspect the report in the workbench's screening view.

</details>

## Memory across the research loop

Paper memory stores findings from published work. Experiment memory stores this campaign's measured outcomes, analysis and conditional lessons. Both can inform planning, with separate storage and delivery allowances.

| Stage | Context it receives |
| --- | --- |
| **Planner** | Current experiment feedback, selected relevant observations and knowledge, a compact history index, and retrieved paper findings when enabled |
| **Critic** | The proposal pool and the Planner's scientific context; local retrieval adds experiment observations relevant to the proposed methods, retaining cited evidence. Paper findings and selected knowledge are reused. |
| **Designer** | The selected method, this task's evaluation scope, parent implementation/configuration, interface and explicitly supplied materials |
| **Coder / Reviewer** | The approved design, implementation requirements, source and current tool/review feedback |
| **Analysis / Memory** | The executed task's measured facts and diagnostics, followed by the resulting interpretation and source references |

Historical memory is not automatically attached to every stage. Selected-plan citations are retained locally as provenance; they do not automatically inject paper or experiment cards into Designer or Coder prompts.

The Planner's experiment query prioritizes the current question, method and latest terminal feedback. Persisted unresolved questions in `plan.json` supply a fallback. The Critic then queries by proposal. MiniLM's tokenizer determines the encoding length; long memory records are represented by explicit chunks with source ranges. This derived search representation does not rewrite the stored observation or its analysis.

### Delivery allowances

| Content | Configuration | Default |
| --- | --- | --- |
| Paper findings selected for a round | `literature_memory.prompt_content_tokens` | **6000 tokens total**, up to `literature_memory.top_k = 8` findings |
| Each experiment Observation | `memory.per_observation_tokens_budget` | **6000 tokens per observation** |
| Other optional historical cards, such as lessons and knowledge | `memory.content_chars_budget` | **6000 Unicode characters total** |

These allowances count scientific content, excluding fixed schema keys, JSON punctuation and citation identifiers. Complete cards are selected; an oversized optional observation can remain as a compact history entry. Current terminal feedback is mandatory and can exceed the optional per-observation allowance. The complete API request still has its own context and output budget.

## Paper memory before the first Planner

Add this block when creating a new `single_plan_v1` campaign:

```json
{
  "literature_memory": {
    "enabled": true,
    "max_papers": 10,
    "search_results_per_query": 20,
    "max_queries": 3,
    "top_k": 8,
    "prompt_content_tokens": 6000,
    "paper_input_tokens": 24000
  }
}
```

Initialization runs before the first Planner. GPT Researcher supplies OpenAlex search and document readers; DeepSeek generates search queries and extracts each paper's findings. Each query searches up to 20 metadata candidates, then local relevance and diversity ranking select up to 10 readable papers. Later rounds retrieve from the saved collection using the research goal and current focus.

The query-planning input contains `research_question`, `task_setting` and `evaluation_protocol`, constructed locally from the goal and protocol. Set `research_question` in the goal to specify the scientific question. The paper reader additionally receives the paper's title, text and reading coverage. It returns a `findings` array with three scientific fields per item:

```json
{
  "topic": "The paper's research topic",
  "statement": "A finding supported by the supplied paper text",
  "reported_setup": {
    "task": "Reported task and data setting",
    "training": null,
    "evaluation": "Reported evaluation protocol"
  }
}
```

`reported_setup` values are null when unreported. URLs, source identities, hashes and text ranges are attached locally. Oversized paper input uses complete paragraphs/pages and records the delivered coverage; the original text stays on disk.

**API usage:** a fresh successful initialization uses one query-planning call plus one extraction call per paper: **11 calls for 10 papers**, before any corrections. The existing schema-correction path allows up to five corrections per response, subject to campaign budgets. Cached queries and extractions are reused on resume. Subsequent paper and experiment retrieval use local embeddings and add **no LLM calls**. These counts describe literature initialization, not the separate Planner, Critic, Analysis or Curator calls.

### Reader dependencies and commands

Set `AGENT_PYTHON` to the agent's Python 3.12+ interpreter and `CAMPAIGN_DIR` to the campaign directory. Install the optional reader dependencies there:

```bash
uv pip install --python "$AGENT_PYTHON" -r requirements-literature.txt
```

The reader dependencies are separate from the legacy fMRI `p2` extra because their constraints conflict in a combined lock. Paper retrieval also needs the MiniLM dependency and prepared model cache described below. For paper-only retrieval, set `memory.enabled` to false but keep `model_id` and the pinned `model_revision` in the goal's `memory` block. Paper retrieval uses this encoder configuration independently of semantic experiment retrieval.

Preview the scientific input without calling an LLM:

```bash
PYTHONPATH=src "$AGENT_PYTHON" -m react_agent.eeg_research.agentic.literature_bootstrap \
  --campaign "$CAMPAIGN_DIR"
```

Initialize explicitly and inspect the selected findings:

```bash
PYTHONPATH=src "$AGENT_PYTHON" -m react_agent.eeg_research.agentic.literature_bootstrap \
  --campaign "$CAMPAIGN_DIR" --execute --retrieve
```

## Experiment memory and MiniLM

Completed task feedback is saved as an Observation containing its question, intervention, execution status, native measurements and references to the Analyst's interpretation. Failed engineering tasks can also produce observations, with their actual status. Local code constructs the card and its identities; it does not ask another model to regenerate measured results.

Enable semantic retrieval with a `memory` block in the new campaign goal:

```json
{
  "memory": {
    "enabled": true,
    "skills_enabled": false,
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "device": "cpu",
    "batch_size": 32,
    "top_k": 5,
    "min_similarity": 0.30,
    "per_observation_tokens_budget": 6000,
    "content_chars_budget": 6000,
    "allow_model_download": false,
    "fallback": "legacy"
  }
}
```

Install `react-agent[memory]` in the agent environment (`uv sync --extra memory` for a fresh project environment). Set `MEMORY_PYTHON` to an interpreter with that dependency; paper initialization and the campaign worker need access to the same prepared model cache. MiniLM runs on CPU. Original record languages are retained; the default model is English-focused.

Applicability conditions such as dataset, fold coverage and training setting are considered alongside cosine similarity. The current implementation retrieves within one EEG / MEG campaign. Final-test/final-holdout, fMRI and foreign-campaign sources are excluded from this experiment-memory collection.

Setting `memory.enabled` to false disables semantic experiment retrieval, while saved observations and current task feedback remain. Paper retrieval is controlled separately by `literature_memory.enabled`.

<details>
<summary>Prepare, query or rebuild the local index</summary>

Set `CAMPAIGN_DIR` to an existing campaign. The following commands maintain the derived index without starting a research worker or making an LLM call.

```bash
# Prepare the model cache and index existing records.
PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  index --campaign "$CAMPAIGN_DIR" --allow-model-download --require-embedding

# Index newly saved records; reuse unchanged vectors.
PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  backfill --campaign "$CAMPAIGN_DIR" --require-embedding

# Inspect retrieval for an explicit query.
PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  query --campaign "$CAMPAIGN_DIR" --text "Diagnose EEG and image feature alignment" \
  --require-embedding

# Recompute vectors when diagnosing an index problem.
PYTHONPATH=src "$MEMORY_PYTHON" -m react_agent.eeg_research.agentic.semantic_memory \
  rebuild --campaign "$CAMPAIGN_DIR" --require-embedding
```

Normal requests use the local model cache unless downloads are explicitly enabled. For offline use after preparation, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. `--model-revision` selects a revision for a maintenance command; `--retry` retries a model load that failed in the current process.

</details>

<details>
<summary>Optional procedural skills</summary>

With both `memory.enabled` and `skills_enabled` set to true, the existing Curator response can also supply structured procedures, preconditions and limits. Local code saves them with source references. This extends the Curator output rather than adding a separate summarizer call. Stored skills are not automatically inserted into every Designer, Coder or Reviewer request.

</details>

## Saved records and troubleshooting

Paths below are relative to `CAMPAIGN_DIR`.

| Location | Contents |
| --- | --- |
| `experiment_observations/obs_*.json` | Saved task outcomes: native facts, analysis interpretation and source references |
| `memory.sqlite` | Episodes, lessons and curation records |
| `semantic_memory.sqlite` | Experiment-observation index, embedding cache and optional skills |
| `memory_deliveries/` | Local delivery reports, including selected and omitted cards |
| `literature_memory/` | Scientific input, query plan, acquired text, prepared reader inputs, extractions and finding cards |
| `literature_memory/index.sqlite` | Separate paper embedding index |
| `literature_memory/reports/` | Paper retrieval scores, selection and encoding diagnostics |
| `llm_requests/`, `llm_responses/` | Archived model requests and responses |

Experiment retrieval reports missing dependencies, missing model weights or index failures as `backend=unavailable`, `degraded=true` with a reason. Paper retrieval reports `embedding_unavailable`. Maintenance commands with `--require-embedding` fail explicitly; inspect the reason and `indexed/reused/pending/failed` counters before retrying. Current campaign state and delivery reports show whether a runtime step is waiting for recovery.

For a missing historical finding, inspect the retrieval query, applicability, selected/omitted IDs and content allowance. For an incomplete paper finding, inspect the saved reader input and its coverage. Indexing and retrieval operate on saved source records; successful API responses and completed paper extractions are reused on resume.

See the [architecture map](../docs/architecture.md) for implementation responsibilities and the [running guide](../docs/running.md) for campaign status and recovery commands.
