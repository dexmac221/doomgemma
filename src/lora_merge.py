"""Merge the LoRA adapter into the base model (in RAM, bf16) and save a complete
Hugging Face model, ready for llama.cpp's convert_hf_to_gguf.py.
Usage: lora_merge.py <base model dir> <adapter dir> <output dir>"""
import os, shutil, sys, torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText
base, lora, out = sys.argv[1:4]
try:
    m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
except Exception:          # text-only models (e.g. Gemma 3 270M)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, lora).merge_and_unload()
m.save_pretrained(out, max_shard_size="20GB")
for f in os.listdir(base):          # tokenizer, chat template, processor: same as the base
    if f.endswith((".json", ".jinja", ".model")) and not f.startswith(("config", "model.safetensors")) \
            and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(base, f), out)
print("merged into", out, os.listdir(out))
