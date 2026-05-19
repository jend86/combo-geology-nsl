import warnings
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.parsing.repetition_guard import RepetitionGuardConfig

CudagraphMode = Literal["full", "piecewise", "full_and_piecewise", "none"]


class ModeConfig(BaseModel):
    """One delegated mode in :class:`OrchestratorModeHarness`.

    Modes are a harness-internal vocabulary — they are NOT task capabilities.
    A mode is a prompt-prefix the orchestrator uses to delegate work to the
    LLM; if the delegated turn produces a code fence, the harness invokes a
    task-owned MCP capability (``code_capability``) to execute it.

    Fields:
      prompt: User-template for the delegated turn. Must include
        ``{instruction}`` and ``{scratchpad_content}`` placeholders.
      timeout_s: Per-mode code-execution timeout (seconds). Passed to the
        task's code-execution capability via the invocation input.
      scratchpad_label: When set, the harness extracts the labelled section
        from the LLM's response and writes it to the scratchpad.
      runs_code: When True, the harness extracts a fenced code block from
        the response and dispatches it to ``code_capability``.
      publishes_metric: When True, scratchpad entries for this mode are
        prefixed with the task's metric name/value (parsed from the
        ``parse_response`` invocations).
      writes_scratchpad: When True, the mode's content is appended to the
        cross-episode scratchpad. At least one mode must opt in.
      code_capability: Name of the task-declared MCP capability invoked
        when ``runs_code=True``. The harness dispatches this invocation
        through the framework capability bridge.
        Required when ``runs_code=True``; ignored otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    timeout_s: int = 30
    scratchpad_label: Optional[str] = None
    runs_code: bool = False
    publishes_metric: bool = False
    writes_scratchpad: bool = False
    code_capability: Optional[str] = None

    @model_validator(mode="after")
    def _validate_runs_code(self) -> "ModeConfig":
        if self.runs_code and not self.code_capability:
            raise ValueError(
                "ModeConfig.runs_code=True requires code_capability "
                "(name of the task-declared MCP capability to invoke)"
            )
        return self


class OrchestratorModesConfig(BaseModel):
    """Typed configuration for :class:`OrchestratorModeHarness`.

    Modes are owned wholly by the harness — task capabilities are a separate
    concept (real MCP tools the harness invokes). ``modes`` is a single typed
    block; the previous parallel ``capability_prompts`` / ``capability_timeouts``
    dicts have been replaced.
    """

    model_config = ConfigDict(extra="forbid")

    max_harness_iterations: int = 12
    scratchpad_max_chars: int = 32000
    tool_output_max_chars: int = 20000
    orchestrator_prompt: str
    modes: Dict[str, ModeConfig] = Field(default_factory=dict)
    repetition_guard: RepetitionGuardConfig = Field(
        default_factory=RepetitionGuardConfig
    )

    @model_validator(mode="after")
    def _at_least_one_scratchpad_writer(self) -> "OrchestratorModesConfig":
        # Empty modes block is allowed for autofill / programmatic
        # construction; only enforce the scratchpad-writer rule when modes
        # are populated.
        if self.modes and not any(m.writes_scratchpad for m in self.modes.values()):
            raise ValueError(
                "OrchestratorModesConfig: at least one mode must set "
                "writes_scratchpad=True. Otherwise the cross-episode "
                "scratchpad runs empty across the entire episode."
            )
        return self


class ContainerHarnessBuildConfig(BaseModel):
    """Local build spec for a ContainerHarness image.

    Populated when the harness image is produced from a Dockerfile inside
    this repo rather than pulled from a registry. Mirrors ``docker
    compose``'s ``build:`` block: ``context`` is the directory whose
    contents become the build context, ``dockerfile`` is the filename
    inside that context, and ``build_args`` become ``--build-arg`` pairs.

    ``force=True`` bypasses the ``images.get`` short-circuit in
    ``ensure_harness_image`` AND passes ``nocache=True`` to the build —
    so a forced rebuild actually rebuilds, not just re-tags.
    """

    model_config = ConfigDict(extra="forbid")

    context: str
    dockerfile: str = "Dockerfile"
    build_args: Dict[str, str] = Field(default_factory=dict)
    force: bool = False


class ContainerHarnessConfig(BaseModel):
    """Typed configuration for :class:`ContainerHarness`.

    ``profile_config`` stays a generic dict so adding a new profile does
    not require modifying this file; the loader late-validates it against
    the registered profile's ``profile_config_class`` model, so typos
    surface at ``AppConfig`` load time anyway.

    ``build`` is optional. When set, ``ensure_harness_image`` builds the
    image locally from ``build.context`` and tags it as ``image``; when
    unset, ``image`` is pulled from whatever registry its prefix names.
    """

    model_config = ConfigDict(extra="forbid")

    profile: str
    image: str
    build: Optional[ContainerHarnessBuildConfig] = None
    entrypoint: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env: Dict[str, str] = Field(default_factory=dict)
    max_wall_seconds: int = 600
    mem_limit: str = "2g"
    network_mode: Literal["bridge", "none", "host"] = "bridge"
    inference_transport: Literal["tcp"] = "tcp"
    profile_config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_image_vs_build(self) -> "ContainerHarnessConfig":
        # A registry-prefixed image (``ghcr.io/foo/bar:tag``) combined
        # with a local build context is almost certainly a mistake — the
        # user wanted either a bare local tag ("nsl/foo:0.1.0") or a
        # registry pull (with build=None). Catch at load time, not when
        # the daemon rejects an anonymous push at build-tag time.
        #
        # Per Docker reference grammar, only the first path segment is
        # a hostname — and only when it contains a ``.`` or ``:``. A
        # bare prefix like ``nsl/`` is an organisation namespace under
        # the implicit Docker Hub, not a registry, and is a valid local
        # tag target.
        if self.build is not None:
            first_segment = self.image.split("/", 1)[0]
            if "." in first_segment or ":" in first_segment:
                raise ValueError(
                    f"image={self.image!r} looks like a registry reference but "
                    f"build is set — local builds should use an un-prefixed tag "
                    f"like 'nsl/<name>:<ver>'. Either remove the registry prefix "
                    f"or drop the [harness.container.build] block."
                )
        return self

    @model_validator(mode="after")
    def _validate_profile_config(self) -> "ContainerHarnessConfig":
        # Late import: the profile registry imports HarnessProfile subclasses
        # that live alongside the harness implementation. Importing at module
        # top-level would create a circular import through
        # src.harness.profiles -> src.harness.context -> src.typing.config.
        from src.harness.profiles import REGISTRY

        if self.profile not in REGISTRY:
            raise ValueError(
                f"unknown harness profile {self.profile!r}; "
                f"registered: {sorted(REGISTRY)}"
            )
        # Only run the inner Pydantic validation when the user supplied
        # profile_config. An empty dict means "the outer ContainerHarnessConfig
        # is being constructed in isolation" (tests, programmatic callers);
        # full validation still fires at AppConfig load, where profile_config
        # is always populated by the TOML. This keeps the isolated-construction
        # surface ergonomic without weakening the typo-catch guarantee.
        if self.profile_config:
            profile_cls = REGISTRY[self.profile]
            profile_cls.profile_config_class.model_validate(self.profile_config)
        return self


_DEFAULT_ORCHESTRATOR_PROMPT = (
    "Default orchestrator prompt (not configured). Set "
    "[harness.orchestrator_modes].orchestrator_prompt in config. "
    "Scratchpad:\n{scratchpad_content}\n"
    "Budget {budget_remaining}/{total_budget}"
)


class HarnessConfig(BaseModel):
    """Configuration for the agent loop harness.

    The harness is user-swappable: the framework owns inference, containers,
    and trajectory capture; the harness drives the agent loop. Exactly one
    of the typed per-harness sections (``orchestrator_modes``, ``container``)
    must be populated, matching ``name``. Class-override dotted paths let
    users wrap or subclass the framework-provided TracedGenner /
    EventRecorder / harness class without touching the ABC.

    Full-default construction (``HarnessConfig()`` with no kwargs) is valid
    for programmatic callers — the loader auto-populates
    ``orchestrator_modes`` with a placeholder-prompt default so test
    fixtures that don't care about harness config keep working. Explicitly
    setting ``name`` or populating a section commits to strict validation:
    typos like ``HarnessConfig(name="container")`` without a matching
    ``container`` section still raise.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "orchestrator_modes"

    orchestrator_modes: Optional[OrchestratorModesConfig] = None
    container: Optional[ContainerHarnessConfig] = None

    traced_genner_class: Optional[str] = None
    event_recorder_class: Optional[str] = None
    harness_class: Optional[str] = None

    # N consecutive HarnessError occurrences trip the circuit breaker with
    # a distinct alarm so systematic harness breakage surfaces.
    consecutive_harness_error_limit: int = 3

    # Wall-clock timeout for a single episode's harness.run_episode(ctx). When
    # exceeded the framework sets HarnessContext.cancel_event; the harness is
    # expected to unwind cooperatively at the next TracedGenner / recorder
    # tick. None = no wall-clock bound (rely on action budget + inference
    # timeout instead).
    episode_wall_clock_seconds: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _autofill_bare_default(cls, values):
        # Ergonomic default: when the caller doesn't specify a harness
        # section AND doesn't override ``name``, auto-populate the default
        # orchestrator_modes section with a sentinel prompt. This keeps
        # programmatic construction usable from tests that only care about
        # non-section knobs (e.g. ``consecutive_harness_error_limit``).
        # Explicit ``name`` or explicit section entries skip the autofill
        # and fall through to the strict after-validator.
        if not isinstance(values, dict):
            return values
        has_explicit_name = "name" in values and values["name"] != "orchestrator_modes"
        has_section = any(
            values.get(s) is not None for s in ("orchestrator_modes", "container")
        )
        if not has_explicit_name and not has_section:
            values = dict(values)
            values["orchestrator_modes"] = {
                "orchestrator_prompt": _DEFAULT_ORCHESTRATOR_PROMPT,
            }
        return values

    @model_validator(mode="after")
    def _exactly_one_section(self) -> "HarnessConfig":
        sections = ("orchestrator_modes", "container")
        populated = {s for s in sections if getattr(self, s) is not None}
        if self.name not in sections:
            raise ValueError(
                f"harness.name={self.name!r} is not a known built-in section. "
                f"Set harness.harness_class for custom harnesses."
            )
        if self.name not in populated:
            raise ValueError(
                f"harness.{self.name} section must be populated when "
                f"harness.name={self.name!r}"
            )
        if len(populated) > 1:
            raise ValueError(
                "only one harness section may be populated at a time; got "
                f"{sorted(populated)}"
            )
        return self


class ToolContractConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enforcement: Literal["fail", "warn", "skip"] = "fail"
    unsafe_reason: Optional[str] = None

    @model_validator(mode="after")
    def _unsafe_modes_require_reason(self) -> "ToolContractConfig":
        if self.enforcement in {"warn", "skip"} and not self.unsafe_reason:
            raise ValueError(
                "tool_contract.unsafe_reason is required when enforcement is "
                f"{self.enforcement!r}"
            )
        return self


class AppConfig(BaseModel):
    model_name: str
    gpu_memory_utilization: float = 0.85
    code_host_cache_path: str
    container_ids: List[str]
    main_container_idx: int = 0

    dynamic_container: bool = False
    docker_compose_dir: Optional[str] = None  # legacy — migrating to task.config

    train_data_save_folder: str

    class TaskConfig(BaseModel):
        model_config = ConfigDict(populate_by_name=True)

        class_: str = Field(
            default="tasks.memory_cleanup.MemoryCleanupTask",
            alias="class",
        )
        config: Dict[str, Any] = Field(default_factory=dict)

    task: TaskConfig = Field(default_factory=TaskConfig)

    class PeftConfig(BaseModel):
        base_model_path: str
        checkpoint_path: str
        device: str = "auto"

    peft: Optional[PeftConfig] = None

    class InferenceConfig(BaseModel):
        temperature: float = 0.5
        # Per-call output token cap forwarded to the backend
        # (chat.completions max_tokens / Anthropic max_tokens).
        # None means "omit max_tokens from the request and let the
        # backend use its own default". For vLLM that resolves to
        # (max_model_len - prompt_tokens) per request — the right
        # semantic for "effectively uncapped: use whatever output
        # budget remains after the prompt" without having to keep
        # max_tokens + prompt_tokens <= max_model_len ourselves.
        # Claude requires a positive int; ClaudeGenner falls back to
        # a Claude-safe default if this is None.
        max_tokens: Optional[int] = None
        timeout: int = 300  # seconds
        frequency_penalty: Optional[float] = None
        presence_penalty: Optional[float] = None

    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    class VllmConfig(BaseModel):
        chat_template_path: Optional[str] = None
        local_model_path: Optional[str] = None
        lora_adapter_path: Optional[str] = None
        served_model_name: Optional[str] = None
        enable_auto_tool_choice: Optional[bool] = None
        tool_call_parser: Optional[str] = None
        # Opt-in only. Set for hybrid-thinking models (e.g. Qwen3 series) so
        # vLLM strips <think>...</think> into reasoning_content before the
        # tool-call parser sees the message body. Not auto-inferred —
        # passing it for non-thinking models silently breaks tool calls.
        reasoning_parser: Optional[str] = None
        compile_cache_dir: Optional[str] = None
        max_model_len: Optional[int] = None
        max_num_seqs: Optional[int] = None
        max_num_batched_tokens: Optional[int] = None
        enable_chunked_prefill: bool = False
        enable_prefix_caching: bool = False
        tensor_parallel_size: Optional[int] = None
        data_parallel_size: Optional[int] = None
        pipeline_parallel_size: Optional[int] = None
        kv_cache_dtype: Optional[Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]] = None
        disable_custom_all_reduce: bool = True
        startup_timeout: Optional[int] = None
        enforce_eager: bool = False
        cudagraph_mode: Optional[CudagraphMode] = None
        extra_env: Dict[str, str] = Field(default_factory=dict)

    vllm: Optional[VllmConfig] = None

    class SglangConfig(BaseModel):
        image: Optional[str] = None
        startup_timeout: Optional[int] = None
        extra_env: Dict[str, str] = Field(default_factory=dict)

        local_model_path: Optional[str] = None
        served_model_name: Optional[str] = None
        chat_template_path: Optional[str] = None

        lora_adapters: Dict[str, str] = Field(default_factory=dict)
        pinned_lora_names: List[str] = Field(default_factory=list)
        max_loras_per_batch: Optional[int] = None
        max_loaded_loras: Optional[int] = None
        max_lora_rank: Optional[int] = None
        lora_target_modules: Optional[List[str]] = None
        lora_backend: Optional[Literal["csgmv", "triton"]] = None
        enable_lora_overlap_loading: bool = False
        lora_eviction_policy: Optional[Literal["lru", "fifo"]] = None
        lora_routing_enabled: bool = False
        default_lora_name: Optional[str] = None

        quantization: Optional[
            Literal[
                "fp8",
                "mxfp4",
                "blockwise_int8",
                "w8a8_int8",
                "w8a8_fp8",
                "awq",
                "awq_marlin",
                "gptq",
                "gptq_marlin",
                "compressed-tensors",
                "modelopt_fp8",
                "modelopt_fp4",
                "quark",
                "auto-round",
                "torchao",
                "bitsandbytes",
                "gguf",
            ]
        ] = None
        fp8_gemm_backend: Optional[str] = None
        attention_backend: Optional[Literal["flashinfer", "triton"]] = None

        tensor_parallel_size: Optional[int] = None
        data_parallel_size: Optional[int] = None
        max_model_len: Optional[int] = None
        max_running_requests: Optional[int] = None
        mem_fraction_static: Optional[float] = None
        kv_cache_dtype: Optional[Literal["auto", "fp8_e5m2", "fp8_e4m3"]] = None

        disable_radix_cache: bool = False
        radix_eviction_policy: Optional[Literal["lru", "lfu", "fifo"]] = None
        disable_cuda_graph: bool = False
        cuda_graph_max_bs: Optional[int] = None
        cuda_graph_bs: Optional[List[int]] = None
        enable_torch_compile: bool = False
        torch_compile_max_bs: Optional[int] = None
        enable_chunked_prefill: bool = True
        enable_mixed_chunk: bool = False

        tool_call_parser: Optional[str] = None
        reasoning_parser: Optional[str] = None
        grammar_backend: Literal["xgrammar", "outlines", "llguidance"] = "xgrammar"

        speculative_algorithm: Optional[
            Literal["EAGLE", "EAGLE3", "MTP", "NEXTN", "NGRAM"]
        ] = None
        speculative_draft_model_path: Optional[str] = None
        speculative_num_steps: Optional[int] = None
        speculative_eagle_topk: Optional[int] = None
        speculative_num_draft_tokens: Optional[int] = None

        @model_validator(mode="after")
        def _validate_sglang_flags(self) -> "AppConfig.SglangConfig":
            lora_enabled = bool(self.lora_adapters)
            if lora_enabled and self.max_lora_rank is None:
                raise ValueError("sglang.lora_adapters requires max_lora_rank")
            if lora_enabled and self.lora_target_modules is None:
                warnings.warn(
                    "sglang.lora_adapters set without lora_target_modules; "
                    "SGLang can infer this today, but explicit modules are recommended.",
                    UserWarning,
                    stacklevel=2,
                )

            effective_max_loras_per_batch = self.max_loras_per_batch
            if effective_max_loras_per_batch is None and self.lora_adapters:
                effective_max_loras_per_batch = len(self.lora_adapters)
            if (
                effective_max_loras_per_batch is not None
                and len(self.pinned_lora_names) >= effective_max_loras_per_batch
            ):
                raise ValueError(
                    "sglang.pinned_lora_names must leave at least one unpinned "
                    "LoRA slot; reduce pinned_lora_names or raise max_loras_per_batch"
                )

            if (
                self.enable_lora_overlap_loading
                and self.max_loaded_loras is not None
                and effective_max_loras_per_batch is not None
                and self.max_loaded_loras > 2 * effective_max_loras_per_batch
            ):
                raise ValueError(
                    "sglang.enable_lora_overlap_loading requires max_loaded_loras "
                    "<= 2 * max_loras_per_batch"
                )

            if self.quantization == "bitsandbytes":
                warnings.warn(
                    "bitsandbytes on SGLang is supported but not Marlin-grade; "
                    "prefer FP8 on Hopper+ or AWQ/GPTQ-Marlin on Ampere.",
                    UserWarning,
                    stacklevel=2,
                )
                if lora_enabled:
                    warnings.warn(
                        "bitsandbytes + LoRA on SGLang is not optimized; "
                        "FP8 base weights plus FP16 LoRA is recommended.",
                        UserWarning,
                        stacklevel=2,
                    )
            if self.quantization == "gguf":
                warnings.warn(
                    "GGUF on SGLang is a compatibility shim; prefer the llama: backend.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.quantization == "torchao" and not self.disable_cuda_graph:
                raise ValueError(
                    "torchao quantization is incompatible with CUDA graphs; "
                    "set disable_cuda_graph=true or choose another format"
                )

            if self.enable_torch_compile and self.torch_compile_max_bs is None:
                warnings.warn(
                    "sglang.enable_torch_compile is set without torch_compile_max_bs; "
                    "startup may compile more shapes than intended.",
                    UserWarning,
                    stacklevel=2,
                )

            if self.speculative_algorithm and self.enable_chunked_prefill:
                raise ValueError(
                    "SGLang speculative decoding is incompatible with chunked prefill; "
                    "set enable_chunked_prefill=false when speculative_algorithm is set"
                )
            if self.speculative_algorithm in {"EAGLE", "EAGLE3"} and self.quantization in {
                "bitsandbytes",
                "gguf",
                "torchao",
            }:
                raise ValueError(
                    "SGLang EAGLE/EAGLE3 speculative decoding requires dense FP/FP8 "
                    "weights, not bitsandbytes, gguf, or torchao quantization"
                )
            if self.speculative_algorithm and self.cuda_graph_bs is None:
                warnings.warn(
                    "sglang.speculative_algorithm set without cuda_graph_bs; ensure "
                    "CUDA graph batch sizes cover speculative draft widening.",
                    UserWarning,
                    stacklevel=2,
                )
            return self

    sglang: Optional[SglangConfig] = None

    class LlamaConfig(BaseModel):
        lora_adapter_path: Optional[str] = None
        startup_timeout: Optional[int] = None

    llama: Optional[LlamaConfig] = None

    class ObservabilityConfig(BaseModel):
        enabled: bool = True
        record_inference: bool = True
        record_phases: bool = True
        record_resources: bool = True
        metrics_output_path: Optional[str] = None
        hardware_tags: List[str] = Field(default_factory=list)
        load_tags: List[str] = Field(default_factory=list)
        detect_hardware: bool = True

    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    tool_contract: ToolContractConfig = Field(default_factory=ToolContractConfig)

    class GenerationConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        target_training_rows: int = 4500
        max_episodes: int = 10000
        max_bootstrap_episodes: int | None = None
        generation_timeout_s: int | None = None
        container_restart_interval: int = 10
        container_rebuild_interval: int = 10
        variation_strategy: Literal["round_robin", "random"] = "round_robin"
        variation_random_seed: Optional[int] = None
        post_rebuild_wait_seconds: int = 10
        checkpoint_every_episode: bool = True
        resume_from_checkpoint: bool = True
        show_progress: bool = True
        resource_snapshot_interval_episodes: int = 1
        generation_output_dir: str = "./data/generations"
        max_consecutive_verification_failures: int = 15
        parallel_episodes: int = 1  # N=1 = current sequential behavior

    generation: Optional[GenerationConfig] = None

    class TrainingConfig(BaseModel):
        base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
        max_steps: int = 50
        per_device_train_batch_size: int = 2
        gradient_accumulation_steps: int = 4
        learning_rate: float = 2e-4
        warmup_steps: int = 5
        max_seq_length: int = 2048
        adapter_output_dir: str = "./models/adapters"
        wandb_project: Optional[str] = None
        export_format: Literal["auto", "lora", "merged_16bit", "gguf"] = "auto"
        gguf_quantize: str = "f16"
        gpu_wait_timeout_seconds: int = 120
        gpu_wait_min_free_memory_fraction: float = 0.9

    training: Optional[TrainingConfig] = None

    class OrchestrationConfig(BaseModel):
        num_generations: int = 1
        training_window_size: int = 3

    orchestration: Optional[OrchestrationConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty_strings(cls, values):
        if values.get("docker_compose_dir") == "":
            values["docker_compose_dir"] = None
        return values

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_sections(cls, values):
        # Pre-harness-abstraction configs put harness knobs under [episode] and
        # [repetition_guard] at the top level. Pydantic's default extra-ignore
        # would silently drop those sections, regressing agent behavior. Fail
        # loudly with a migration hint instead.
        if not isinstance(values, dict):
            return values
        if "episode" in values:
            raise ValueError(
                "Config has legacy [episode] section. Move these keys under "
                "[harness.orchestrator_modes] instead:\n"
                "  [harness]\n"
                '  name = "orchestrator_modes"\n'
                "  [harness.orchestrator_modes]\n"
                "  max_harness_iterations = ...\n"
                "  scratchpad_max_chars = ...\n"
                "  tool_output_max_chars = ..."
            )
        if "repetition_guard" in values:
            raise ValueError(
                "Config has legacy top-level [repetition_guard] section. "
                "Move it under [harness.orchestrator_modes.repetition_guard]:\n"
                "  [harness.orchestrator_modes.repetition_guard]\n"
                "  min_paragraphs = 8\n  ..."
            )
        return values

    @model_validator(mode="after")
    def _check_dynamic_container(self) -> "AppConfig":
        if self.dynamic_container and not self.docker_compose_dir:
            raise ValueError(
                "docker_compose_dir is required when dynamic_container=True"
            )
        return self

    @model_validator(mode="after")
    def _validate_sglang_model_constraints(self) -> "AppConfig":
        cfg = self.sglang
        if cfg is None:
            return self

        source_model = (
            self.model_name.split(":", 1)[1].strip()
            if self.model_name.startswith("sglang:")
            else self.model_name.strip()
        )
        normalized = source_model.lower()
        is_deepseek_v3_or_r1 = "deepseek-v3" in normalized or "deepseek-r1" in normalized

        if is_deepseek_v3_or_r1 and cfg.quantization is not None:
            raise ValueError("DeepSeek V3/R1 ships native FP8; do not set sglang.quantization")
        if normalized.endswith("-bnb-4bit") and cfg.quantization != "bitsandbytes":
            raise ValueError(
                "Model name implies bitsandbytes (-bnb-4bit), but "
                "sglang.quantization is not 'bitsandbytes'"
            )
        if cfg.speculative_algorithm == "MTP" and not is_deepseek_v3_or_r1:
            raise ValueError("SGLang MTP speculative decoding requires a DeepSeek V3/R1 model")
        if (
            cfg.local_model_path
            and (cfg.lora_adapters or cfg.lora_routing_enabled)
            and cfg.served_model_name is None
        ):
            raise ValueError(
                "sglang.served_model_name is required when local_model_path is used "
                "with LoRA adapters or LoRA routing"
            )
        return self

    @model_validator(mode="after")
    def _validate_backend_gpu_coexistence(self) -> "AppConfig":
        if self.vllm is None or self.sglang is None:
            return self

        vllm_devices = self.vllm.extra_env.get("CUDA_VISIBLE_DEVICES")
        sglang_devices = self.sglang.extra_env.get("CUDA_VISIBLE_DEVICES")
        if not vllm_devices or not sglang_devices:
            raise ValueError(
                "Configs with both [vllm] and [sglang] must set disjoint "
                "CUDA_VISIBLE_DEVICES in each section's extra_env"
            )

        vllm_set = _parse_cuda_visible_devices(vllm_devices)
        sglang_set = _parse_cuda_visible_devices(sglang_devices)
        if vllm_set & sglang_set:
            raise ValueError(
                "[vllm] and [sglang] CUDA_VISIBLE_DEVICES overlap; concurrent "
                "managed inference backends must use disjoint GPU sets"
            )
        return self


def _parse_cuda_visible_devices(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}
