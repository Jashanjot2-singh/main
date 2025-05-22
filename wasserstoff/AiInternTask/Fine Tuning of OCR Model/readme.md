# ✍️ Fine-Tuned TrOCR OCR Pipeline

## Overview

This project focuses on fine-tuning Microsoft’s **TrOCR** model (`trocr-large-handwritten`) on the **IAM Handwriting Dataset**, optionally combined with the **Imgur5K** dataset. It enables high-accuracy OCR (Optical Character Recognition) for handwritten and scanned image documents.

This fine-tuned model supports and significantly improves the performance of the **RAG Chatbot + OCR** system built for the **Wasserstoff Gen-AI Internship Task**, by extracting text more accurately from challenging real-world scanned documents.

---

## 🚀 Features

- Fine-tunes TrOCR on the IAM handwriting dataset and optionally on Imgur5K.
- Supports noisy image preprocessing (denoising, resizing, grayscale normalization).
- Tracks performance using **Character Error Rate (CER)** and **Word Error Rate (WER)**.
- Implements **early stopping** for stable training.
- Saves the best model and processor for deployment.
- Includes an **inference function** for batch prediction on new images.

---

## 🔍 Why This Is Important for the Task

The Wasserstoff internship task required the ability to:

> “Convert and preprocess scanned documents using OCR (Optical Character Recognition). Extract text content accurately, ensuring high fidelity for research purposes.”

Fine-tuning TrOCR helps in:
- Reducing OCR errors in noisy or handwritten scanned documents.
- Improving downstream **embedding quality** for semantic search.
- Enhancing the **accuracy of responses** returned by the RAG pipeline.
- Supporting documents that are **not natively digital**, like handwritten notes or old reports.

This directly improves the **context quality** fed into the vector database and ultimately boosts the **relevance of generated answers** in the RAG chatbot.

---

## 🧠 Model & Dataset

- **Base Model:** `microsoft/trocr-large-handwritten`
- **Fine-Tuned On:**
  - [✅] IAM Handwriting Dataset (11,000+ lines of text from 700 writers)
  - [Optional] Imgur5K if available in CSV format (`annotations.csv`)
- **Preprocessing Includes:**
  - Denoising with OpenCV
  - Resizing to 384x384
  - Grayscale normalization

---

## 🛠 Usage

### 🔧 Training

```bash
python trocr_ocr.py --train --imgur_path path/to/imgur_dataset
If no --imgur_path is specified, only the IAM dataset is used.
```
### 🔍 Inference
```bash
python trocr_ocr.py --predict path/to/image1.png path/to/image2.jpg
Model must be trained or downloaded to fine_tuned_trocr/ folder before running predictions.
```
## 📊 Evaluation Metrics
#### CER (Character Error Rate): Measures character-level OCR accuracy.

#### WER (Word Error Rate): Measures word-level accuracy of the transcribed output.

A CER ≤ 7% and WER ≤ 15% are typically considered excellent for OCR tasks.

## ✅ Benefits Recap (in Internship Context)

| **Challenge**                                   | **How TrOCR Fine-Tuning Helps**                                  |
|--------------------------------------------------|-------------------------------------------------------------------|
| Scanned image text is noisy and error-prone     | Image denoising + model adaptation improves recognition           |
| Handwritten text not supported by generic OCR   | Fine-tuned TrOCR excels at handwriting                            |
| Downstream QA quality suffers from poor OCR     | Improved OCR → better embeddings → better answers                 |
| Need for citation-level precision               | Accurate OCR enables clean chunking and indexing                  |

## 📦 Dependencies
```bash
pip install torch torchvision transformers datasets opencv-python Pillow pandas tqdm