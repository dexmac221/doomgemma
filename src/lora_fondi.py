"""Fonde la LoRA nel modello base (in RAM, bf16) e salva un modello HF completo,
pronto per convert_hf_to_gguf.py. Uso: lora_fondi.py <base> <lora> <uscita>"""
import shutil, sys, os, torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText
base, lora, out = sys.argv[1:4]
try:
    m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
except Exception:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, lora).merge_and_unload()
m.save_pretrained(out, max_shard_size="20GB")
for f in os.listdir(base):          # tokenizer, template, processore: identici al base
    if f.endswith((".json", ".jinja", ".model")) and not f.startswith(("config", "model.safetensors")) \
            and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(base, f), out)
print("fuso in", out, os.listdir(out))
