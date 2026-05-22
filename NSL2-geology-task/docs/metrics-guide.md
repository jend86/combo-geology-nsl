# Metrics Guide

This document describes the metrics collected during training loop runs, where they are stored, and how to analyze them.

## Overview

The training loop produces metrics at three levels of granularity:

1. **Per-call** (`metrics_*.jsonl`) -- every LLM inference call and agent phase execution
2. **Per-generation** (`metadata.json`) -- aggregated episode-level stats for one generation
3. **Per-run** (`orchestration_summary.json`) -- full run totals with nested generation metrics

## How Metrics Are Produced

```
MetricsGenner (wraps inference backend, captures every LLM call)
  -> MetricsCollector (stores InferenceMetric + PhaseMetric in memory)
    -> run_single_episode (aggregates per-episode via helpers)
      -> EpisodeTrajectory (episode-level fields)
        -> GenerationData.to_metadata_dict() (generation-level aggregation)
          -> run_train_loop (writes orchestration_summary.json)
```

Key source files:

| File | Role |
|------|------|
| `src/observability/genner_wrapper.py` | Intercepts inference calls, records `InferenceMetric` |
| `src/observability/collector.py` | `MetricsCollector` -- stores, flushes to JSONL, computes summary |
| `src/observability/types.py` | `InferenceMetric`, `PhaseMetric`, `UsageInfo` dataclasses |
| `scripts/generate_training_data.py` | Episode-level aggregation (`_compute_episode_inference_metrics`, `_compute_context_length_stats`) |
| `src/typing/trajectory.py` | `EpisodeTrajectory` fields, `GenerationData.to_metadata_dict()` |
| `scripts/run_train_loop.py` | Orchestration summary construction |

## Output Files

### 1. Raw Per-Call Metrics (`metrics_*.jsonl`)

**Location:** `data/<project>/train_data/metrics_<run_id>.jsonl`

Each line is a JSON object. Two record types:

#### Inference records (`metric_type: "inference"`)

| Field | Type | Description |
|-------|------|-------------|
| `inference_id` | string | Unique ID for this call |
| `run_id` | string | Run identifier |
| `episode_id` | string | Episode this call belongs to |
| `backend` | string | Inference backend (e.g. `vllm`) |
| `phase` | string | Agent phase (`orchestrator`, `explorer`, `investigator`, `recorder`) |
| `success` | bool | Whether the call succeeded |
| `content` | string | LLM output text |
| `error_message` | string? | Error if `success=false` |
| `model` | string | Model name |
| `usage.prompt_tokens` | int | Input tokens |
| `usage.completion_tokens` | int | Output tokens |
| `usage.total_tokens` | int | Total tokens |
| `usage.stop_reason` | string | Why generation stopped |
| `latency_ms` | float | Call latency in milliseconds |
| `prompt_tokens_per_second` | float | Input processing rate |
| `output_tokens_per_second` | float | Output generation rate |
| `total_tokens_per_second` | float | Combined throughput |
| `gpu_memory_mb` | float | GPU memory at time of call |
| `host_memory_mb` | float | Host memory at time of call |

#### Phase records (`metric_type: "phase"`)

| Field | Type | Description |
|-------|------|-------------|
| `phase_name` | string | Phase name (e.g. `recorder`) |
| `run_id` | string | Run identifier |
| `episode_id` | string | Episode this phase belongs to |
| `duration_ms` | float | Phase execution time |
| `success` | bool | Whether the phase succeeded |
| `retry_count` | int | Number of retries |
| `error_message` | string? | Error if `success=false` |

### 2. Generation Metadata (`metadata.json`)

**Location:** `data/<project>/generations/<run_id>/generation_<N>/metadata.json`

One file per generation. Contains aggregated metrics across all episodes in that generation.

| Field | Type | Description |
|-------|------|-------------|
| `generation_id` | int | Generation index |
| `run_id` | string | Run identifier |
| `total_episodes_run` | int | Episodes attempted |
| `total_successful` | int | Episodes that succeeded |
| `total_rows_collected` | int | SFT training rows collected |
| `total_space_freed_kb` | float | Storage freed |
| `success_rate` | float | `total_successful / total_episodes_run` |
| `started_at` | string | ISO timestamp |
| `completed_at` | string | ISO timestamp |
| **Throughput** | | |
| `episodes_per_hour` | float | Episode completion rate |
| `episodes_per_minute` | float | Episode completion rate |
| `successful_rows_per_hour` | float | Successful row collection rate |
| `successful_rows_per_minute` | float | Successful row collection rate |
| **Tokens** | | |
| `total_input_tokens` | int | Sum of prompt tokens |
| `total_output_tokens` | int | Sum of completion tokens |
| `total_tokens` | int | `total_input_tokens + total_output_tokens` |
| `tokens_per_successful_episode` | float | `total_tokens / total_successful` |
| **Context window** | | |
| `peak_context_tokens` | int | Max prompt tokens in any single call |
| `avg_context_tokens` | float | Mean prompt tokens across calls |
| `median_context_tokens` | float | Median prompt tokens (median of episode medians) |
| **Utilization** | | |
| `peak_gpu_utilization_pct` | float | Peak GPU utilization % |
| `average_gpu_utilization_pct` | float | Average GPU utilization % |
| `peak_cpu_utilization_pct` | float | Peak CPU utilization % |
| `average_cpu_utilization_pct` | float | Average CPU utilization % |
| **Timing** | | |
| `total_agent_seconds` | float | Sum of episode durations (parallelism-independent) |
| `total_inference_seconds` | float | Sum of inference time across episodes |
| `total_episode_execution_seconds` | float | Sum of episode execution time |
| `total_container_overhead_seconds` | float | Container setup/teardown overhead |
| `average_inference_duty_cycle` | float | Fraction of episode time spent on inference |
| `average_output_tokens_per_second` | float | Mean output generation rate |

### 3. Orchestration Summary (`orchestration_summary.json`)

**Location:** `data/<project>/generations/<run_id>/orchestration_summary.json`

One file per multi-generation run. Contains run-level totals and per-generation metrics.

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Run identifier |
| `num_generations` | int | Number of generations configured |
| `training_window_size` | int | Sliding window for training data |
| `training_export_format` | string | Export format (e.g. `lora`) |
| `total_wall_clock_seconds` | float | End-to-end run time |
| `total_agent_seconds` | float | Sum of all generation agent seconds |
| `total_tokens` | float | Sum of all generation tokens |
| `generations` | array | Per-generation objects (see below) |

Each generation object contains:

| Field | Type | Description |
|-------|------|-------------|
| `generation_id` | int | Generation index |
| `generation_dir` | string | Path to generation output directory |
| `served_adapter_dir` | string? | Adapter served during this generation |
| `trained_adapter_dir` | string? | Adapter produced by training |
| `training_data_paths` | array | JSONL files used for training |
| `metrics` | object | Full generation metadata (same fields as `metadata.json`) |

## Analyzing Metrics

Use `scripts/analyze_metrics.py` to analyze metrics across runs.

### Basic usage

```bash
# Analyze all metrics in a project directory (text/markdown output)
uv run python scripts/analyze_metrics.py data/test-loop/

# Analyze a specific run by ID
uv run python scripts/analyze_metrics.py data/test-loop/ --run-id 20260408-aqhx5j

# JSON output for programmatic consumption
uv run python scripts/analyze_metrics.py data/test-loop/ --format json

# Save report to file
uv run python scripts/analyze_metrics.py data/test-loop/ --output report.md

# Analyze specific files or multiple directories
uv run python scripts/analyze_metrics.py data/test-loop/train_data/ data/test-loop/generations/
```

### CLI options

| Flag | Description |
|------|-------------|
| `paths` (positional) | One or more file/directory paths to scan for metrics |
| `--run-id RUN_ID` | Filter all results to a specific run ID |
| `--format {text,json}` | Output format (default: `text`, which produces markdown) |
| `--output FILE` | Write output to file instead of stdout |
| `--indent N` | JSON indentation level (default: 2) |

### What the analyzer reports

The analyzer discovers and processes three file types:
- `metrics_*.jsonl` -- raw inference/phase metrics
- `LAST_RUN_LOG.json` / `*_full_run_*.json` -- run metadata (commit, config, hardware tags)
- `orchestration_summary.json` -- orchestration-level metrics

The text report includes these sections:

| Section | Source | What it shows |
|---------|--------|---------------|
| Runs | JSONL | Per-run inference stats (latency, throughput, tokens) |
| Latency by model | JSONL | Average latency and output token rate per model |
| Generation benchmarks by backend | JSONL | Throughput percentiles per backend |
| Generation benchmarks by hardware/load tag | JSONL + run logs | Throughput by hardware configuration |
| Retry rates by phase | JSONL | Phase-level retry and failure rates |
| Memory profiles by model | JSONL | Host/GPU memory usage per model |
| Run Summary table | Orchestration summary | Markdown table showing cross-generation trends (episodes, success rate, tokens, GPU%, duty cycle, context tokens) |

## Interpreting Results

### Cross-generation trends

The **Generation comparison** section is key for evaluating whether LoRA fine-tuning is improving the model. Look for:

- **Success rate increasing** across generations -- the model is learning
- **Tokens per successful episode decreasing** -- the model is becoming more efficient
- **Episodes per minute increasing** -- faster convergence per generation

### Duty cycle

`average_inference_duty_cycle` measures what fraction of episode time is spent on actual LLM inference vs. overhead (container setup, code execution, etc.).

- Values near 1.0 mean the system is inference-bound (GPU is the bottleneck)
- Low values indicate overhead-bound workloads (container, I/O, or orchestration bottlenecks)

### Context window utilization

- `peak_context_tokens` shows maximum prompt size -- important for ensuring you don't exceed model context limits
- A rising `avg_context_tokens` across generations may indicate prompt bloat
- Large gaps between `avg_context_tokens` and `median_context_tokens` suggest skewed distributions (a few very long prompts)

### GPU utilization

- `peak_gpu_utilization_pct` near 100% is expected during inference
- Low `average_gpu_utilization_pct` suggests the GPU is idle between calls (overhead-bound)
- Compare across generations to identify if fine-tuning changes inference patterns
