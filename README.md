# 🧠 LEGAL DOCUMENT CLASSIFIER

---

## Table of Contents
1. Project Overview
2. Objective & Motivation
3. Abstract
4. Terminology & Concepts
5. Repository Structure
6. Quickstart (Run locally)
7. Environment & Requirements
8. Data — Source, Format & Preparation
9. Data Preprocessing (Detailed)
10. Labeling & MultiLabelBinarizer
11. Model Architecture & Rationale
12. Training Pipeline (Step-by-step)
13. Hyperparameters & Tips
14. Evaluation, Metrics & Reporting
15. Inference & Chunking Strategy
16. Streamlit App (Usage & Features)
17. Exporting & Using the Saved Model
18. Optimization & Production-readiness
19. Reproducibility & Experiments
20. Troubleshooting & Common Errors
21. Security, Privacy & Ethics
22. Extensions & Next Steps
23. References & Resources
24. License
25. Acknowledgements
26. Appendix
    - Example queries / sample inputs
    - Example outputs (format)
    - Useful commands

---

## 1. Project Overview
This project provides a robust pipeline to train, evaluate, and deploy a **multi-label transformer-based classifier** for long-form documents (designed for legal texts but adaptable to other domains). It supports long-document chunking, probability aggregation, threshold tuning, and a Streamlit UI for user interaction.

---

## 2. Objective & Motivation
- Automate classification of documents that can belong to multiple labels (e.g., legal topics).
- Improve speed and consistency of document triage for legal professionals and researchers.
- Provide an interpretable, reproducible, and deployable solution.

---

## 3. Abstract
A transformer model is fine-tuned to perform multi-label classification on long-form documents. Long texts are chunked into overlapping windows; each chunk is classified and chunk predictions are aggregated. The system provides probability scores per label and a global tuned threshold to decide final labels. A Streamlit application allows easy upload and classification.

---

## 4. Terminology & Concepts
- **Multi-label classification**: each example may have zero, one, or many labels.
- **Transformer**: self-attention architecture (BERT, LegalBERT, etc.).
- **Chunking / sliding window**: breaking long text into overlapping token windows.
- **Aggregation**: combining chunk-wise probabilities (mean, max) to produce document-level probabilities.
- **Threshold tuning**: choosing a probability cutoff to convert probabilities into binary labels.
- **MLB (MultiLabelBinarizer)**: scikit-learn utility to encode labels into multi-hot vectors.

---

## 6. Quickstart (Run Locally)

1.  **Create & activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train (example):**
    ```bash
    python eurlex_train.py
    ```

4.  **Run Streamlit app:**
    ```bash
    streamlit run eurlex_streamlit.py
    ```

5.  **Predict via CLI:**
    ```bash
    python eurlex_inference.py --text-file examples/sample1.txt
    ```

---

## 7. Environment & Requirements

The project requires the following packages (see `requirements.txt`):

| Package | Suggested Version |
| :--- | :--- |
| `torch` | `>=1.13` |
| `transformers` | `>=4.30` |
| `datasets` | |
| `scikit-learn` | |
| `numpy` | |
| `pandas` | |
| `tqdm` | |
| `streamlit` | |
| `PyPDF2` | |
| `python-docx` | |
| `accelerate` | |
| `evaluate` | |

* Use a **GPU (CUDA)** for training. If using MPS/Apple Silicon, confirm compatibility.
* For large label sets or datasets, allocate sufficient **disk (50+ GB)** and **RAM**.

---

## 8. Data — Source, Format & Preparation

* **Primary dataset:** EUR-Lex / multi\_eurlex (public legal documents with EuroVoc labels).
* **Typical fields:** `text` (string or dict per language), `eurovoc_concepts` (list of string IDs).
* **Splits:** `train` / `validation` (dev) / `test`. If a dataset lacks a validation split, create one from the training data (e.g., 90/10 split).
* **Recommended file formats:**
    * Raw scraped documents: `.txt` or `.jsonl`
    * Labels: present in dataset; otherwise, maintain `labels.csv` or `mlb_classes.json`.

---

## 9. Data Preprocessing (Detailed)

* **Normalization:** Extract text from dataset fields; prefer **English** if multi-lingual.
* **Cleaning (optional):**
    * Remove boilerplate, HTML tags, or OCR artifacts.
    * Normalize whitespace and punctuation where appropriate.
    * Lowercasing is optional depending on the tokenizer (pretrained cased/uncased model).
* **Tokenization:**
    * Use `AutoTokenizer` compatible with the chosen model.
    * For training: use `truncation=True` and a fixed `max_length` (e.g., **512**).
    * For inference: **chunk long texts** into overlapping windows with `stride` for coverage.
* **Label Encoding:**
    * Fit `MultiLabelBinarizer` on training labels—this forms the mapping of class indices.
    * Save `mlb_classes.json` (ordered list) to ensure consistent indexing.

---

## 10. Labeling & MultiLabelBinarizer

* `mlb.classes_` gives the **canonical label order**. Do not shuffle this order; save it.
* Labels in the dataset may be numeric or text: `str()` them for consistency.
* For inference mapping, save:
    * `mlb_classes.json` (list)
    * `id2label.json` (index → original label id)
    * `id2name.json` (label id → human-readable name; loaded from `eurovoc_labels.json`)

---

## 11. Model Architecture & Rationale

* **Base model:** `nlpaueb/legal-bert-base-uncased` (recommended for legal domain) or `bert-base-uncased` as fallback.
* **Final head:** Single linear layer with `num_labels` outputs; use `problem_type="multi_label_classification"`.
* **Loss:** `BCEWithLogitsLoss` (binary cross-entropy with logits)—use per-label binary loss.
* **Optimizer:** `AdamW` (via `transformers Trainer`) with weight decay.
* **Rationale:** Pretrained transformer captures semantics and syntax critical to legal documents; the multi-label head allows independent probability per label.

---

## 12. Training Pipeline (Step-by-step)

1.  Load dataset via `datasets.load_dataset`.
2.  Normalize examples to `{"text": ..., "labels": [...]}`.
3.  Fit `MultiLabelBinarizer` on train labels.
4.  Add `labels_bin` column to dataset (multi-hot float vectors).
5.  Tokenize train/validation with truncation for training (**512 tokens**).
6.  Create `AutoModelForSequenceClassification` with `num_labels`.
7.  Use `CustomTrainer` to cast labels to `float` for BCE loss.
8.  Train with `TrainingArguments`. Example:

    ```python
    TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_total_limit=2,
    )
    ```
9.  After training, run **chunked evaluation** on validation/test splits for accurate long-document metrics.
10. Tune a **global threshold** on validation to maximize **micro-F1**.

---

## 13. Hyperparameters & Tips

| Hyperparameter | Typical Value/Range | Notes |
| :--- | :--- | :--- |
| `max_length` for chunk | **512** | Standard BERT input size. |
| `chunk_stride` | **128–256** | Overlap to avoid missing context. |
| `batch_size` | Depends on GPU memory | Reduce if OOM. |
| `learning_rate` | 1e-5 to 5e-5 | 2e-5 is typical for fine-tuning. |
| `num_train_epochs` | 2–5 | |

* Use `gradient_accumulation_steps` for effective large-batch training on small GPUs.
* Use `fp16=True` if GPU supports mixed precision (saves memory & speeds up).

---

## 14. Evaluation, Metrics & Reporting

* **Micro-F1:** (Primary for multi-label) Balances across labels by instance.
* **Macro-F1:** Gives per-label quality, emphasizes rare classes.
* **Precision@k** or **Precision@5:** Useful for retrieval-like tasks.
* **Per-label F1:** To see which labels perform poorly—useful for error analysis.
* **Report:** Training/validation loss, micro-F1 per epoch, final tuned threshold, and validation micro-F1.

---

## 15. Inference & Chunking Strategy

1.  Chunk entire document into token windows of `max_length` with `stride` overlap (ensures continuity).
2.  Classify each chunk independently, producing `num_labels` probabilities per chunk.
3.  Aggregate per-label probabilities across chunks (**max pooling** or **mean pooling**).
4.  Threshold aggregated probabilities using tuned `best_threshold` to decide final positive labels.
5.  Store `best_threshold.json` in saved artifacts.

---

## 16. Streamlit App (Usage & Features)

* **Input:** Paste text or upload supported files: `.txt`, `.pdf`, `.docx`.
    * PDF parsing using `PyPDF2` or `pdfplumber` (optional).
    * DOCX parsing via `python-docx`.
* **Provides:**
    * Predicted labels (human-friendly names)
    * Probabilities (sorted top-N)
    * JSON & CSV download of results
    * Threshold slider in sidebar for user tuning
* **Caching:** Model and tokenizer loaded with `@st.cache_resource` to avoid reloads.

---

## 17. Exporting & Using the Saved Model

Saved artifacts in `legal_doc_classifier/`:

* `pytorch_model.bin` & `config.json` — weights & config
* `tokenizer.*` — tokenizer files
* `mlb_classes.json` — label order
* `id2label.json`, `label2id.json` — index mappings
* `id2name.json` — human-readable names
* `best_threshold.json` — tuned threshold

**Loading example (inference):**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json, torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("legal_doc_classifier")
tokenizer = AutoTokenizer.from_pretrained("legal_doc_classifier")

# Load mappings and threshold
with open("legal_doc_classifier/mlb_classes.json") as f: 
    labels = json.load(f)
with open("legal_doc_classifier/best_threshold.json") as f: 
    thr = json.load(f)["best_threshold"]
```

---

## 18. Optimization & Production-readiness

* **Quantization:** Dynamic/static quantization for CPU inference (`torch.quantization`) or ONNX export + quantization.
* **Distillation:** Distil a smaller DistilBERT-like model for faster inference.
* **Batching & caching:** For inference server, bundle documents and reuse tokenized chunks.
* **Serving:** FastAPI or Flask endpoints with GPU support; containerize with Docker. 
* **Autoscaling:** Deploy to Kubernetes or serverless inference (e.g., AWS SageMaker, Azure ML).
* **Monitoring:** Track model drift and concept shift; log predictions for audits.

---

## 19. Reproducibility & Experiments

* Save **random seeds** (Python, NumPy, torch, CUDA/MPS).
* Log experiment config (hyperparameters, dataset commit hash, model version).
* Use **Weights & Biases** or **MLflow** for experiment tracking.
* Save final environment (`pip freeze > requirements.txt`) and Dockerfile for exact reproducibility.

---

## 20. Troubleshooting & Common Errors

| Error | Fix/Solution |
| :--- | :--- |
| Disk write error / PytorchStreamWriter failed | Check free disk space and `save_total_limit`. |
| OOM on GPU | Reduce `batch_size`, use `fp16`, or decrease sequence length. |
| MPS errors (Apple silicon) | Ensure correct torch + MPS support, or fallback to CPU. |
| `ValueError`: Unable to create tensor | Enable padding/truncation consistently in tokenizer calls. |
| `RuntimeError` about dtype | Ensure labels are `float` for BCE loss: `inputs["labels"] = inputs["labels"].float()` in trainer. |

---

## 21. Security, Privacy & Ethics

* Be mindful of **sensitive data** in legal documents. Avoid uploading private or PII-containing documents to untrusted servers.
* Document model limitations and potential **bias**; do not rely solely on automated labels for legal decisions.
* Add **disclaimers** in UI regarding the model being an assistant, not legal advice.

---

## 22. Extensions & Next Steps

* **Explainability:** LIME/SHAP or attention visualization to show which phrases influenced predictions. 
* **Threshold Tuning:** Implement per-label threshold tuning for imbalanced labels.
* **Label Management:** Implement hierarchical classification or two-stage label pruning for thousands of labels.
* **Multilingual:** Add multilingual support (use language-specific BERT or XLM-R).
* **Integration:** Integrate semantic search, summarization, or case retrieval pipelines.

---

## 23. References & Resources

* Hugging Face Transformers (docs)
* EUR-Lex / EuroVoc resources
* Research papers on legal NLP and multi-label classification (e.g., LexGLUE benchmark)
* scikit-learn MultiLabelBinarizer docs

---

## 24. License

This repository is provided under the **MIT License** (or your preferred license). Confirm the license file in the repo root.

---

## 25. Acknowledgements

Thanks to Hugging Face, PyTorch, the creators of the EUR-Lex / EuroVoc datasets, and the open-source community.

---

## 26. Appendix

### Example CLI commands

| Task | Command |
| :--- | :--- |
| Train | `python eurlex_train.py` |
| Inference (example) | `python eurlex_inference.py --text-file examples/sample1.txt` |
| Run Streamlit | `streamlit run eurlex_streamlit.py` |

### Example Input & Output

**Example Input (short):**

> The Regulation sets forth obligations for data controllers and processors and provides rights to individuals, including access, rectification, and erasure.

**Example Output:**

> Predicted Labels:
>   Criminal Law Suite
>   Civil Law Suite

---

## Useful Tips

* Keep model artifacts and mappings together in a single folder for reproducibility.
* Always test chunking on sample long documents and inspect per-chunk predictions for debugging.
* Save `best_threshold.json` to keep inference decisions stable across deployments.
