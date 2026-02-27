import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class MockChoice:
    """Mimics litellm response choice for interface compatibility."""
    def __init__(self, text):
        self.message = {"content": text}


class MockResponse:
    """Mimics litellm completion response for interface compatibility."""
    def __init__(self, text):
        self.choices = [MockChoice(text)]


class LocalModelManager:
    """
    Manages loading, caching, and unloading of local HuggingFace models.

    Only one model is kept in GPU memory at a time to conserve VRAM.
    When a different model is requested, the current one is unloaded first.
    If the same model is requested consecutively, it stays loaded (no-op).
    """

    def __init__(self, quantization="4bit", device_map="auto"):
        self.quantization = quantization
        self.device_map = device_map
        self._model = None
        self._tokenizer = None
        self._current_model_name = None

        if quantization == "4bit":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            self.bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.bnb_config = None

    def load_model(self, model_name: str):
        """Load a model into GPU memory, unloading any previous model first."""
        if self._current_model_name == model_name:
            return

        self.unload()
        print(f"[LocalModelManager] Loading {model_name} ({self.quantization})...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if self.bnb_config is not None:
            load_kwargs["quantization_config"] = self.bnb_config

        self._model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self._model.eval()
        self._current_model_name = model_name
        print(f"[LocalModelManager] Loaded {model_name}")

    def unload(self):
        """Free GPU memory by deleting the current model."""
        if self._model is not None:
            model_name = self._current_model_name
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._current_model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[LocalModelManager] Unloaded {model_name}")

    def generate(self, messages, model_name, temperature=0.1, max_tokens=2048):
        """
        Generate a chat completion from a local model.

        Args:
            messages: list of dicts with 'role' and 'content' keys
            model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B-Instruct")
            temperature: sampling temperature (0 = greedy)
            max_tokens: max new tokens to generate
        """
        self.load_model(model_name)

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(temperature, 0.01)
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response


class LocalCompletionWrapper:
    """
    Wraps LocalModelManager to match the litellm completion() call signature.

    This lets generate_with_client() work identically for both API and local models —
    it calls client(messages=..., model=..., temperature=..., max_completion_tokens=...)
    and accesses .choices[0].message["content"].
    """

    def __init__(self, manager: LocalModelManager, model_name: str):
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
