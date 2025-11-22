# train_tokenizer.py
import sentencepiece as spm
from pathlib import Path
import argparse

def write_corpus(jsonl_path, out_txt):
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(out_txt, "w", encoding="utf-8") as fout:
        for line in fin:
            import json
            obj = json.loads(line)
            txt = obj.get("text") or obj.get("document_text") or obj.get("judgement")
            if txt:
                # very simple cleaning; you can improve
                fout.write(txt.replace("\n", " ").strip() + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="/Users/ompandit/Desktop/DL CAPESTONE TRY/data/raw/combined_legal_corpus_limited.jsonl", help="merged raw jsonl with 'text' field")
    parser.add_argument("--model_prefix", default="tokenizer/indlegal_spm")
    parser.add_argument("--vocab_size", type=int, default=52000)
    args = parser.parse_args()

    Path("tokenizer").mkdir(parents=True, exist_ok=True)
    tmp_corpus_txt = "tokenizer/corpus.txt"
    write_corpus(args.jsonl, tmp_corpus_txt)

    spm.SentencePieceTrainer.Train(
        input=tmp_corpus_txt,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        hard_vocab_limit=False,
        unk_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3,
    )
    print("Trained tokenizer at:", args.model_prefix + ".model")
