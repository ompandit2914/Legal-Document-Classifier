# pretrain_bart_from_scratch.py
import os
from pathlib import Path
import random
import json
import math
import argparse
from datasets import load_dataset, Dataset
import torch
from transformers import (
    BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset as TorchDataset

# ---------- Helper: simple BART-like span masking (text-infilling) ----------
def mask_spans(tokens, mask_ratio=0.3, avg_span_len=3, mask_token_id=1):  # pad id used temporarily
    # tokens: list of token ids (integers)
    n = len(tokens)
    num_to_mask = max(1, int(n * mask_ratio))
    masked = tokens.copy()
    covered = [False] * n
    spans = []
    i = 0
    while sum(covered) < num_to_mask and i < n:
        if covered[i]:
            i += 1; continue
        if random.random() < 0.02:  # small chance to start a span here
            span_len = max(1, int(random.expovariate(1.0/avg_span_len)))
            j = i
            cur = []
            while j < n and len(cur) < span_len and not covered[j]:
                covered[j] = True
                cur.append(j)
                j += 1
            if cur:
                spans.append((cur[0], cur[-1]))
            i = j
        else:
            i += 1
    # Replace spans with a single sentinel token id per span (use unique ids in real BART, but we'll use mask_token_id)
    out_tokens = []
    i = 0
    while i < n:
        if covered[i]:
            # skip continuous covered region
            j = i
            while j < n and covered[j]:
                j += 1
            out_tokens.append(mask_token_id)  # sentinel
            i = j
        else:
            out_tokens.append(tokens[i])
            i += 1
    return out_tokens

# ---------- Torch Dataset wrapper ----------
class RawTextDataset(TorchDataset):
    def __init__(self, jsonl_path, tokenizer: PreTrainedTokenizerFast, max_len=1024):
        self.lines = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for L in f:
                try:
                    j = json.loads(L)
                    txt = j.get("text") or j.get("document_text") or j.get("judgement")
                    if txt and len(txt.strip())>50:
                        self.lines.append(txt.strip())
                except:
                    continue
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        txt = self.lines[idx]
        tok = self.tokenizer(txt, truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = tok["input_ids"].squeeze().tolist()
        # create noised input
        noised = mask_spans(input_ids, mask_ratio=0.3, avg_span_len=3, mask_token_id=self.tokenizer.mask_token_id or self.tokenizer.pad_token_id)
        # decoder target is the original piece (we'll shift in model's loss)
        return {
            "input_ids": torch.tensor(noised, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(noised), dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/raw/merged.jsonl")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--output_dir", default="out/pretrained-bart")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=6)
    args = parser.parse_args()

    # Load tokenizer from SentencePiece or HF tokenizer; adapt as needed
    tokenizer_path = os.path.join(args.tokenizer_dir, "indlegal_spm.model")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    # Build a BartConfig for a small/medium model (scale to your compute)
    config = BartConfig(
        d_model=args.model_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        vocab_size=tokenizer.vocab_size or 52000,
        max_position_embeddings=args.max_len,
    )
    model = BartForConditionalGeneration(config)

    dataset = RawTextDataset(args.jsonl, tokenizer, max_len=args.max_len)
    print("Dataset size:", len(dataset))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=8,
        fp16=torch.cuda.is_available(),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    data_collator = lambda features: {
        "input_ids": torch.nn.utils.rnn.pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100),
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved pretrained model to", args.output_dir)
