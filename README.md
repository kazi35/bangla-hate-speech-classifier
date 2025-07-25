# 🇧🇩 Bangla Hate Speech Classifier (PyTorch)

A lightweight, transformer-based hate speech classifier trained on 10,000 balanced Bengali text samples using PyTorch and Hugging Face Transformers.

---

## 🧠 Project Overview

This project aims to detect **hate speech in Bengali language** using a fine-tuned transformer model. It was trained on a manually balanced dataset with 5,000 hate and 5,000 non-hate examples.

- ✅ Trained using `prajjwal1/bert-tiny` (for low-resource environments)
- ✅ Built in PyTorch using Hugging Face's `Trainer`
- ✅ Fully CPU-compatible (no GPU required)
- ✅ Designed to run locally in VS Code

---

## 📁 Dataset Structure

The dataset is a CSV file named `balanced_10000.csv` with:

| Column   | Description              |
|----------|--------------------------|
| `sentence` | The Bengali text input  |
| `hate`     | Target label: `1` = hate, `0` = not hate |

---

## 🛠️ How to Train

### 🔧 Install dependencies

```bash
pip install -r requirements.txt
