# """Origin vLLM: Extended vLLM implementation using inheritance

# This module provides extended functionality by inheriting from the base vLLM implementation.
# It maintains compatibility with the original vLLM API while adding custom features.
# """

# import sys
# import os
# import warnings

# # Core Engine Components
# from .engine.arg_utils import AsyncEngineArgs as BaseAsyncEngineArgs, EngineArgs as BaseEngineArgs
# from .engine.arg_utils import AsyncEngineArgs , EngineArgs 
# from .engine.llm_engine import LLMEngine as BaseLLMEngine, _load_generation_config_dict
# from .worker.worker import Worker, _check_if_gpu_supports_dtype
# from .worker.model_runner import ModelRunner, CUDAGraphRunner, GPUModelRunnerBase 
# from .outputs import CompletionOutput, EmbeddingOutput, EmbeddingRequestOutput, RequestOutput
# from .sequence import (
#     SamplerOutput, Sequence, SequenceData, SequenceGroup, 
#     SequenceGroupMetadata, SequenceGroupOutput, SequenceOutput, SequenceStatus,
#     ExecuteModelRequest, IntermediateTensors
# )
# from .model_executor.models.gemma import GemmaForCausalLM
# from ._custom_ops import init_custom_ar
# from .transformers_utils.detokenizer import detokenize_incrementally
# from .utils import (
#     Counter, LRUCache, make_async, is_hip, get_cpu_memory,
#     in_wsl, str_to_int_tuple, CudaMemoryProfiler,
#     is_pin_memory_available, init_cached_hf_modules, FlexibleArgumentParser,
#     deprecate_kwargs, print_warning_once
# )
# from .tracing import SpanAttributes, SpanKind, extract_trace_context, init_tracer
# from .engine.llm_engine import LLMEngine, _load_generation_config_dict
# from .engine.output_processor.utils import create_output_by_sequence_group


# __all__ = [
#     # Core Engine Components (Base Classes)
#     "BaseAsyncEngineArgs",
#     "AsyncEngineArgs",
#     "BaseEngineArgs",
#     "EngineArgs",
#     "BaseAsyncLLMEngine", 
#     "BaseLLMEngine",
#     "BaseLLM",
#     "_load_generation_config_dict",
    
#     # Configuration Classes
#     "CacheConfig",
#     "DecodingConfig",
#     "DeviceConfig", 
#     "EngineConfig",
#     "LoadConfig",
#     "LoRAConfig",
#     "ModelConfig",
#     "MultiModalConfig",
#     "ObservabilityConfig",
#     "ParallelConfig",
#     "PromptAdapterConfig",
#     "SchedulerConfig",
#     "SpeculativeConfig",
#     "TokenizerPoolConfig",
#     "VisionLanguageConfig",
#     "_get_and_verify_dtype",
#     "_get_and_verify_max_len",
#     "get_served_model_name",
    
#     # Core Processing Components
#     "Scheduler",
#     "SchedulerOutputs",
#     "SequenceGroupOutputProcessor",
#     "StopChecker",
#     "LoggingStatLogger",
#     "PrometheusStatLogger",
#     "StatLogger",
#     "Stats",
#     "StatLoggerBase",
    
#     # Executor and Worker Components
#     "ExecutorBase",
#     "ExecutorAsyncBase",
#     "initialize_ray_cluster",
#     "Worker",
#     "_check_if_gpu_supports_dtype",
#     "WorkerInput",
#     "ModelRunner",
#     "CUDAGraphRunner",
#     "GPUModelRunnerBase",
#     "_async_h2d",
#     "ModelRunnerBase", 
#     "ModelRunnerInputBase",
#     "EmbeddingModelRunner",
#     "CacheEngine",
    
#     # Input/Output and Sequence Components
#     "INPUT_REGISTRY",
#     "InputRegistry",
#     "LLMInputs",
#     "PromptInputs",
#     "TextPrompt",
#     "TokensPrompt",
#     "parse_and_batch_prompt",
#     "InputPreprocessor",
#     "CompletionOutput",
#     "EmbeddingOutput",
#     "EmbeddingRequestOutput",
#     "RequestOutput",
#     "MultiModalData",
#     "SamplerOutput",
#     "Sequence",
#     "SequenceData",
#     "SequenceGroup",
#     "SequenceGroupMetadata",
#     "SequenceGroupOutput",
#     "SequenceOutput",
#     "SequenceStatus",
#     "ExecuteModelRequest",
#     "IntermediateTensors",
#     "PoolingParams",
#     "SamplingParams",
#     "SamplingType",
    
#     # Model Executor Components
#     "InputMetadata",
#     "SamplingMetadata", 
#     "set_random_seed",
#     "BaseModelRegistry",
#     "BaseModelLoader",
#     "get_architecture_class_name",
#     "_initialize_model",
#     "set_default_torch_dtype",
#     "default_weight_loader",
#     "get_quant_config",
#     "initialize_dummy_weights",
#     "is_pp_missing_parameter",
#     "supports_lora",
#     "supports_vision",
#     "GemmaForCausalLM",
    
#     # Model Executor Layers
#     "VocabParallelEmbedding",
#     "ParallelLMHead",
#     "ScaledActivation",
#     "Sampler",
#     "_prune_hidden_states",
#     "_apply_logits_processors",
#     "_apply_penalties",
#     "_apply_top_k_top_p",
#     "_apply_min_p",
#     "_sample",
#     "_get_logprobs",
#     "_build_sampler_output",
#     "QUANTIZATION_METHODS",
#     "get_quantization_config",
#     "LogitsProcessor",
#     "FusedMoE",
#     "SamplingTensors",
    
#     # Attention Components
#     "AttentionMetadata",
#     "get_attn_backend",
    
#     # Distributed and Parallel Components
#     "ps",
#     "get_pp_group",
#     "get_world_group", 
#     "init_distributed_environment",
#     "init_model_parallel_group",
#     "get_tensor_model_parallel_group",
#     "get_tensor_model_parallel_cpu_group",
#     "tensor_model_parallel_all_gather",
#     "pynccl_utils",
#     "init_custom_ar",
#     "set_custom_all_reduce",
#     "init_custom_ar_old",
#     "initialize_model_parallel",
#     "get_tensor_model_parallel_group_old",
    
#     # LoRA Components
#     "LoRARequest",
#     "LoRAMapping",
#     "LRUCacheWorkerLoRAManager",
    
#     # Prompt Adapter Components
#     "PromptAdapterRequest",
#     "LRUCacheWorkerPromptAdapterManager",
    
#     # Tokenizer Components
#     "detokenize_incrementally",
#     "get_cached_tokenizer",
#     "AnyTokenizer",
#     "TokenizerGroup",
#     "BaseTokenizerGroup",
#     "Detokenizer",
#     "get_config",
#     "get_hf_text_config",
    
#     # Guided Decoding Components
#     "GuidedDecodingRequest",
#     "get_local_guided_decoding_logits_processor",
#     "LLMGuidedOptions",
    
#     # Utility Components
#     "init_logger",
#     "Counter",
#     "LRUCache",
#     "make_async",
#     "is_hip",
#     "get_cpu_memory",
#     "get_nvcc_cuda_version",
#     "in_wsl",
#     "str_to_int_tuple",
#     "CudaMemoryProfiler",
#     "DeviceMemoryProfiler",
#     "is_pin_memory_available",
#     "init_cached_hf_modules",
#     "FlexibleArgumentParser",
#     "deprecate_kwargs",
#     "weak_bind",
#     "print_warning_once",
#     "supports_dynamo",
#     "UsageContext",
#     "is_usage_stats_enabled",
#     "usage_message",
#     "VLLM_VERSION",
#     "SpanAttributes",
#     "SpanKind",
#     "extract_trace_context",
#     "init_tracer",
#     "envs",
    
#     # Compilation Components
#     "CompilationLevel",
    
#     # Multimodal Components
#     "MULTIMODAL_REGISTRY",
#     "MultiModalRegistry",
    
#     # Plugin Components
#     "get_torch_compile_backend",
    
#     # Custom Extended Components
#     "SingleStepOutputProcessor",
#     "create_output_by_sequence_group",

#     "VLLM_VERSION",
# ]
