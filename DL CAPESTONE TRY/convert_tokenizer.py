# convert_tokenizer.py
from transformers import PreTrainedTokenizerFast
import os

TOKENIZER_DIR = "tokenizer"
SP_MODEL = os.path.join(TOKENIZER_DIR, "indlegal_spm.model")

# Build a Hugging Face-compatible tokenizer wrapper
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=SP_MODEL,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>"
)

# Save it in the Hugging Face format (creates tokenizer.json, config.json, etc.)
tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"✅ Hugging Face tokenizer saved to {TOKENIZER_DIR}")