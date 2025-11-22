# train_classifier.py
import argparse
import os
import json
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (
    PreTrainedTokenizerFast, BartConfig, BartForSequenceClassification,
    TrainingArguments, Trainer
)
import torch

def load_labelled_csv(csv_path, text_col="text", label_col="label"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    # expecting single-label classification; adapt for multi-label
    labels = sorted(df[label_col].unique())
    label2id = {l:i for i,l in enumerate(labels)}
    df['label_id'] = df[label_col].map(label2id)
    ds = Dataset.from_pandas(df[[text_col, 'label_id']].rename(columns={text_col:'text','label_id':'label'}))
    return ds, label2id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--pretrained_dir", default="out/pretrained-bart")
    parser.add_argument("--output_dir", default="out/bart-classifier")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    from datasets import Dataset
    import pandas as pd

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    labels = sorted(train_df['label'].unique())
    label2id = {l:i for i,l in enumerate(labels)}
    num_labels = len(labels)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_dir)
    config = BartConfig.from_pretrained(args.pretrained_dir)
    classifier_config = BartConfig.from_pretrained(args.pretrained_dir, num_labels=num_labels)
    model = BartForSequenceClassification.from_pretrained(args.pretrained_dir, config=classifier_config)

    def preprocess(examples):
        res = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=args.max_len)
        res['labels'] = [label2id[x] for x in examples['label']]
        return res

    from datasets import Dataset
    train_ds = Dataset.from_pandas(train_df).map(preprocess, batched=True, remove_columns=train_df.columns)
    val_ds = Dataset.from_pandas(val_df).map(preprocess, batched=True, remove_columns=val_df.columns)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics(p):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "precision_macro": precision_score(labels, preds, average="macro"),
            "recall_macro": recall_score(labels, preds, average="macro"),
        }

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved classifier to", args.output_dir)
