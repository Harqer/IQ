from __future__ import annotations

"""Remote inference scaffold for IQ experiments on Hugging Face ZeroGPU.

The initial model is Phi-4-mini so the deployment path can be validated before a
full IQ checkpoint exists.  Once Stage 3 produces a complete IQ model, replace the
loader with the IQ PyTorch reference implementation and checkpoint.
"""

import os

import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("IQ_MODEL_ID", "microsoft/Phi-4-mini-instruct")

# ZeroGPU provides CUDA emulation outside @spaces.GPU and recommends placing the
# model on CUDA at module load time so transfers are optimized when a real GPU is
# attached to the decorated function.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda")
model.eval()


@spaces.GPU(duration=60)
def respond(message: str, history: list[dict] | None = None) -> str:
    history = history or []
    messages: list[dict[str, str]] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )
    generated = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="IQ remote inference prototype",
    description="Remote GPU inference path for the IQ transfer experiments.",
)


if __name__ == "__main__":
    demo.launch()
