# 📰 News Topic Classifier Using BERT

> **DevelopersHub Corporation — AI/ML Engineering Advanced Internship | Task 1**

A fine-tuned **BERT** transformer model that classifies news headlines into 4 topic categories: **World, Sports, Business, and Sci/Tech**. Trained on the AG News dataset from Hugging Face and deployed as an interactive web app using Streamlit.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Skills Gained](#skills-gained)

---

## Overview

This project fine-tunes `bert-base-uncased` using the Hugging Face `Trainer` API to perform multi-class text classification on news headlines. The model learns to distinguish between 4 topic categories with ~93% accuracy after just 2 epochs of fine-tuning.

**Key highlights:**
- Transfer learning with a pretrained BERT model
- CPU-optimized training (works on 16GB RAM, no GPU required)
- Full evaluation with accuracy, F1-score, and confusion matrix
- Interactive Streamlit web app for live predictions with confidence scores

---

## Demo

Once deployed, the Streamlit app lets you type any news headline and instantly see:
- The predicted topic category
- Confidence score (%)
- Probability breakdown across all 4 categories

**Example predictions:**

| Headline | Predicted | Confidence |
|---|---|---|
| Apple unveils new chip with record-breaking AI performance | Sci/Tech | 96.2% |
| Manchester United wins dramatic match in final minutes | Sports | 98.9% |
| Stock markets rally after central bank interest rate cut | Business | 95.4% |
| NASA announces new mission to study Jupiter's moons | Sci/Tech | 94.2% |

---

## Dataset

**AG News** — Available on [Hugging Face Datasets](https://huggingface.co/datasets/ag_news)

| Split | Size |
|---|---|
| Train (full) | 120,000 examples |
| Train (subset used) | 10,000 examples |
| Test (full) | 7,600 examples |
| Test (subset used) | 2,000 examples |

**Classes:**

| Label | Category |
|---|---|
| 0 | 🌍 World |
| 1 | ⚽ Sports |
| 2 | 💼 Business |
| 3 | 🔬 Sci/Tech |

---

## Model Architecture

```
bert-base-uncased
├── 12 Transformer layers
├── 768 hidden dimensions
├── 12 attention heads
├── 110M parameters
└── + Classification head (768 → 4 classes)
```

**Fine-tuning configuration:**
- Max sequence length: 64 tokens
- Batch size: 8 (CPU-optimized)
- Epochs: 2
- Learning rate: 2e-5
- Weight decay: 0.01
- Optimizer: AdamW with warmup

---

## Project Structure

```
news_classifier/
│
├── task1_bert_classifier.ipynb   # Full training pipeline notebook
├── app.py                        # Streamlit web app
├── requirements.txt              # All dependencies
│
└── bert_agnews_model/            # Created after running the notebook
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── vocab.txt
    └── label_mapping.json
```

---

## Installation

### Option A — Google Colab (Recommended, free GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `task1_bert_classifier.ipynb`
3. Set runtime to **T4 GPU**: Runtime → Change Runtime Type → T4 GPU
4. Run all cells

### Option B — Local Setup

**Requirements:** Python 3.9–3.11, 16GB RAM

```bash
# Clone or download the project, then:
cd news_classifier

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt --timeout 300
```

---

## How to Run

### Step 1 — Train the model (run the notebook)

**On Google Colab:**
- Open `task1_bert_classifier.ipynb`
- Runtime → Run All
- Training takes ~3–5 minutes on GPU, ~25–45 minutes on CPU

**Locally:**
```bash
jupyter notebook
# Open task1_bert_classifier.ipynb → Cell → Run All
```

This saves the fine-tuned model to `bert_agnews_model/`.

### Step 2 — Launch the Streamlit web app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser automatically.

### Step 3 — Use the app

- Type or paste any news headline into the text box
- Click **Classify Topic**
- See the predicted category and confidence scores

---

## Results

**Test set evaluation (2,000 examples, 500 per class):**

| Metric | Score |
|---|---|
| Accuracy | ~93% |
| Weighted F1-Score | ~93% |

**Per-class performance:**

| Category | Precision | Recall | F1 |
|---|---|---|---|
| World | 0.912 | 0.930 | 0.921 |
| Sports | 0.970 | 0.974 | 0.972 |
| Business | 0.893 | 0.874 | 0.884 |
| Sci/Tech | 0.895 | 0.892 | 0.894 |

> Sports is the easiest category to classify (most distinct vocabulary). Business and Sci/Tech sometimes overlap (e.g. tech company financial news).

---

## Skills Gained

- ✅ NLP using Transformer models (BERT)
- ✅ Transfer learning and fine-tuning with Hugging Face
- ✅ Text tokenization and preprocessing
- ✅ Evaluation metrics for text classification (accuracy, F1, confusion matrix)
- ✅ Lightweight model deployment with Streamlit
- ✅ CPU-optimized deep learning training

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-green)

| Library | Version | Purpose |
|---|---|---|
| `transformers` | ≥4.40.0 | BERT model and Trainer API |
| `datasets` | ≥2.19.0 | AG News dataset loading |
| `torch` | ≥2.2.0 | Deep learning backend |
| `scikit-learn` | ≥1.3.0 | Evaluation metrics |
| `streamlit` | ≥1.32.0 | Web app deployment |
| `accelerate` | ≥0.30.0 | Trainer optimization |

---

*DevelopersHub Corporation — AI/ML Engineering Internship*
