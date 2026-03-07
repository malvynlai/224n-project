"""
vLLM-based model manager — drop-in replacement for LocalModelManager.

Key advantages over the HuggingFace transformers backend:
  - PagedAttention for efficient KV-cache management
  - Optimized CUDA kernels (FlashAttention, fused ops)
  - Batch generation: process many prompts in one call
  - Prefix caching: shared prompt prefixes are computed once

Usage:
  manager = VLLMModelManager(quantization="4bit")
  text = manager.generate(messages, model_name="Qwen/Qwen2.5-7B-Instruct")
  texts = manager.batch_generate(messages_list, model_name="Qwen/Qwen2.5-7B-Instruct")

Requires: pip install vllm
"""

import gc
import os
from typing import List, Dict, Optional


class MockChoice:
    """Mimics litellm response choice for interface compatibility."""
    def __init__(self, text):
        self.message = {"content": text}


class MockResponse:
    """Mimics litellm completion response for interface compatibility."""
    def __init__(self, text):
        self.choices = [MockChoice(text)]


class VLLMModelManager:
    """
    Manages vLLM-based model loading and inference.

    Holds one model at a time by default (max_models=1) to maximize GPU memory
    for KV-cache. Set max_models > 1 to keep multiple small models loaded
    simultaneously (each gets gpu_memory_utilization / max_models of GPU memory).
    """

    def __init__(
        self,
        quantization: str = "4bit",
        gpu_memory_utilization: float = 0.90,
        max_models: int = 1,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
    ):
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_models = max_models
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self._engines: Dict[str, object] = {}
        self._tokenizers: Dict[str, object] = {}
        self._load_order: List[str] = []

    def _per_model_gpu_memory(self) -> float:
        if self.max_models > 1:
            return self.gpu_memory_utilization / self.max_models
        return self.gpu_memory_utilization

    def load_model(self, model_name: str):
        """Load a model via vLLM (cached; LRU eviction when at capacity)."""
        if model_name in self._engines:
            if model_name in self._load_order:
                self._load_order.remove(model_name)
            self._load_order.append(model_name)
            return

        while len(self._engines) >= self.max_models and self._load_order:
            self.unload(self._load_order[0])

        # vLLM V1 engine's multiprocess architecture is incompatible with
        # bitsandbytes (engine core subprocess dies during bnb weight loading).
        # Force V0 engine for bnb quantization.
        if self.quantization in ("4bit", "8bit"):
            os.environ["VLLM_USE_V1"] = "0"

        from vllm import LLM
        from transformers import AutoTokenizer

        print(f"[VLLMModelManager] Loading {model_name} "
              f"(quantization={self.quantization}, "
              f"gpu_mem={self._per_model_gpu_memory():.0%})...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm_kwargs: Dict = dict(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=self._per_model_gpu_memory(),
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
        )

        if self.quantization in ("4bit", "8bit"):
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
            llm_kwargs["enforce_eager"] = True
        elif self.quantization in ("awq", "gptq"):
            llm_kwargs["quantization"] = self.quantization

        engine = LLM(**llm_kwargs)

        self._engines[model_name] = engine
        self._tokenizers[model_name] = tokenizer
        self._load_order.append(model_name)
        print(f"[VLLMModelManager] Loaded {model_name}")

    def unload(self, model_name: str):
        """Free GPU memory for a specific model."""
        if model_name in self._engines:
            engine = self._engines.pop(model_name)
            del engine
            self._tokenizers.pop(model_name, None)
            if model_name in self._load_order:
                self._load_order.remove(model_name)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print(f"[VLLMModelManager] Unloaded {model_name}")

    def unload_all(self):
        """Free GPU memory for all cached models."""
        names = list(self._engines.keys())
        for name in list(self._engines):
            del self._engines[name]
        self._engines.clear()
        self._tokenizers.clear()
        self._load_order.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        for name in names:
            print(f"[VLLMModelManager] Unloaded {name}")

    def _format_chat(self, messages: List[dict], model_name: str) -> str:
        """Apply the model's chat template to format messages into a prompt string."""
        tokenizer = self._tokenizers[model_name]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _make_sampling_params(self, temperature: float, max_tokens: int):
        from vllm import SamplingParams
        if temperature > 0:
            return SamplingParams(
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.9,
            )
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def generate(
        self,
        messages: List[dict],
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Single-request generation (same interface as LocalModelManager.generate).

        Uses vLLM under the hood — still faster than HF transformers due to
        optimized kernels and PagedAttention even for batch_size=1.
        """
        self.load_model(model_name)
        prompt = self._format_chat(messages, model_name)
        params = self._make_sampling_params(temperature, max_tokens)
        outputs = self._engines[model_name].generate([prompt], params)
        return outputs[0].outputs[0].text

    def batch_generate(
        self,
        messages_list: List[List[dict]],
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> List[str]:
        """
        Batch generation — process many prompts in one vLLM call.

        This is the primary throughput advantage: vLLM's continuous batching
        and PagedAttention process N prompts far faster than N sequential calls.
        """
        if not messages_list:
            return []
        self.load_model(model_name)
        prompts = [self._format_chat(msgs, model_name) for msgs in messages_list]
        params = self._make_sampling_params(temperature, max_tokens)
        outputs = self._engines[model_name].generate(prompts, params)
        return [out.outputs[0].text for out in outputs]


class VLLMCompletionWrapper:
    """
    Wraps VLLMModelManager to match the litellm completion() call signature.

    Drop-in replacement for LocalCompletionWrapper — lets generate_with_client()
    work identically for HF, vLLM, and API backends.
    """

    def __init__(self, manager: VLLMModelManager, model_name: str):
        self.manager = manager
        self.model_name = model_name

    def __call__(self, messages, model=None, temperature=0.1,
                 max_completion_tokens=2048, **kwargs):
        model_name = model or self.model_name
        response_text = self.manager.generate(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )
        return MockResponse(response_text)
