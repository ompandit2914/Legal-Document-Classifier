# prepare_data.py
import os
from pathlib import Path
import json
import shutil
import argparse
from datasets import load_dataset

def download_from_hf(dataset_name, config_name=None, out_dir="data/raw", limit=5000):
    """
    Download a dataset from Hugging Face (optionally limited to N examples),
    flatten fields, and save as JSONL.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load with config if provided
    if config_name:
        print(f"🔽 Downloading {dataset_name} ({config_name}) from Hugging Face...")
        ds = load_dataset(dataset_name, config_name, split="train")
    else:
        print(f"🔽 Downloading {dataset_name} from Hugging Face...")
        ds = load_dataset(dataset_name, split="train")

    # Limit dataset size for memory safety
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
        print(f"⚙️ Using only first {limit} samples to reduce memory load.")

    fout = out_dir / f"{dataset_name.replace('/','_')}_{config_name or 'default'}_limited.jsonl"
    with fout.open("w", encoding="utf-8") as f:
        for ex in ds:
            text = ex.get("text") or ex.get("full_text") or ex.get("document_text") or ex.get("judgement") or ex.get("content")
            if text:
                json.dump({"text": text, **{k: v for k, v in ex.items() if k not in ["text", "full_text", "document_text", "judgement", "content"]}}, f)
                f.write("\n")

    print(f"✅ Saved {len(ds)} records to {fout}")
    return fout


def merge_jsonl_files(in_dir, out_file):
    """
    Merge JSONL files safely, line-by-line, to prevent I/O overflow.
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fw:
        for p in Path(in_dir).rglob("*.jsonl"):
            with p.open("r", encoding="utf-8") as fr:
                for line in fr:
                    fw.write(line)
    print(f"🧩 Merged all JSONL files into {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare legal text datasets.")
    parser.add_argument("--out-dir", default="data/raw", help="Directory to save datasets")
    parser.add_argument("--lexglue-config", default="ledgar", help="LexGLUE subset to download")
    parser.add_argument("--include-indian", action="store_true", help="Also include Indian Legal Documents Corpus (ILDC)")
    parser.add_argument("--limit", type=int, default=5000, help="Limit number of samples per dataset to save space")
    args = parser.parse_args()

    out_dir = args.out_dir

    # --- Step 1: Download LexGLUE subset ---
    try:
        lex_path = download_from_hf("coastalcph/lex_glue", args.lexglue_config, out_dir, limit=args.limit)
    except Exception as e:
        print("❌ Failed to download LexGLUE via HF:", e)
        print("Please choose one of: ['case_hold', 'ecthr_a', 'ecthr_b', 'eurlex', 'ledgar', 'scotus', 'unfair_tos']")

    # --- Step 2: Optionally include Indian dataset ---
    if args.include_indian:
        try:
            ildc_path = download_from_hf("manorma29/ildc", None, out_dir, limit=args.limit)
        except Exception as e:
            print("❌ Failed to download ILDC dataset:", e)

    # --- Step 3: Merge ---
    merged_file = Path(out_dir) / "combined_legal_corpus_limited.jsonl"
    merge_jsonl_files(out_dir, merged_file)

    print("\n✅ Data preparation completed.")
    print(f"📁 Merged dataset saved to: {merged_file}")
    print("💡 Ready for tokenization and training.")
