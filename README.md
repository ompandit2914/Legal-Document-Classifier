# 🧠 Multi-Label Document Classification using Transformers

A deep learning project built to classify documents into multiple categories using transformer-based models.  
This system can automatically understand and categorize long textual documents with high accuracy — ideal for domains such as legal, research, or business document processing.

---

## 🚀 Project Overview

This project implements a **multi-label text classification system** that uses **transformer-based architectures** to identify multiple relevant categories for each document.  
It leverages contextual embeddings from pre-trained transformer models, enabling it to capture the semantic depth and relationships between words and phrases in complex text data.

The workflow includes data preprocessing, label binarization, model fine-tuning, and an inference pipeline. A Streamlit-based interface allows users to upload and classify documents interactively.

---

## 🎯 Objectives

- Develop an intelligent model capable of **multi-label document classification**.  
- Fine-tune a transformer model to understand **domain-specific terminology and long-form text**.  
- Enable **real-time document analysis** and **interactive classification** through a web interface.  
- Provide a scalable, reproducible, and explainable pipeline for future research and enterprise use.

---

## 🧩 Abstract

This project demonstrates how large language models can be applied for automated document understanding.  
By training on labeled data and fine-tuning a pre-trained transformer model, the system learns to predict multiple relevant tags or topics per document.  
A custom chunking and aggregation mechanism allows it to process long documents efficiently, while probability-based thresholding ensures optimal label predictions.  
The final model integrates with Streamlit for an easy-to-use interface, allowing document uploads and instant classification feedback.

---

## 🧠 Introduction

Manual document classification is a time-consuming and error-prone task, especially when documents belong to multiple thematic categories.  
This project uses **transformer-based natural language understanding** to automate that process.  
By leveraging contextual embeddings, the model effectively identifies semantic relationships and assigns appropriate labels, offering a powerful AI-driven solution for document management, legal tech, and research indexing.

---

## 🧰 Project Definition

A **multi-label document classifier** that processes text inputs, extracts meaningful patterns, and outputs multiple label probabilities.  
The model handles variable-length documents using dynamic chunking and aggregates predictions through statistical pooling methods (e.g., mean or max).  
The pipeline includes:
- Dataset loading and preprocessing
- Tokenization and label binarization
- Transformer model fine-tuning
- Evaluation with F1-score metrics
- Model saving and Streamlit-based deployment

---

## 🧑‍💻 Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python |
| **Frameworks** | PyTorch, Hugging Face Transformers |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualization & UI** | Streamlit |
| **Evaluation Metrics** | F1-Score (Micro & Macro), Precision, Recall |
| **Deployment** | Streamlit, local or cloud (Heroku/Render) |
| **Environment** | Virtualenv / Conda |

---

## ⚙️ How It Works

1. **Dataset Loading** – Loads multi-label text datasets and prepares train-validation splits.  
2. **Preprocessing** – Extracts text fields, cleans data, and binarizes labels.  
3. **Tokenization** – Converts raw text into transformer-compatible tokens.  
4. **Model Training** – Fine-tunes a pre-trained transformer for multi-label classification.  
5. **Evaluation** – Computes micro and macro F1 scores to assess model performance.  
6. **Inference** – Uses chunked document input to generate label probabilities.  
7. **Interface** – A Streamlit app allows users to upload documents and visualize predictions interactively.

---

## 📦 Output

- Trained Transformer Model (`/legal_doc_classifier/`)  
- Tokenizer  
- Label Mapping (`mlb_classes.json`)  
- Logs and metrics  
- Streamlit interface for document upload and classification

---

## 📈 Example Use Case

Upload a PDF or text file via the Streamlit UI.  
The model processes the document, identifies its semantic meaning, and returns top relevant categories with confidence percentages.  
This can help automate workflows in:
- Legal document tagging  
- Research paper categorization  
- Policy analysis  
- Business intelligence

---

## 🧾 Key Features

- Multi-label text classification  
- Long document chunking and aggregation  
- Customizable probability thresholding  
- Explainable and interpretable outputs  
- Streamlit web interface for non-technical users  

---

## 🧮 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Micro-F1** | Captures performance over all labels equally |
| **Macro-F1** | Measures performance per label and averages |
| **Precision/Recall** | Evaluates accuracy and completeness of label predictions |

---

## 🧠 Future Enhancements

- Integrate explainability using SHAP or LIME  
- Add multilingual document support  
- Deploy using Docker or Hugging Face Spaces  
- Build REST API endpoints for external integration  

---

## 🤝 Contributing

Pull requests, feature suggestions, and improvements are welcome.  
Fork the repository, create a feature branch, and submit a PR for review.

---

## 📜 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it for educational or research purposes.

---

## 💡 Acknowledgements

This work draws on advances in **transformer-based NLP models**, **multi-label classification research**, and the open-source contributions of the **Hugging Face** and **PyTorch** communities.

---
