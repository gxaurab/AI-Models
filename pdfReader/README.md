# 📘 PDF Question Answering App

This is a simple Python script that allows you to ask questions about a PDF document using a Hugging Face transformer model (`distilbert-base-cased-distilled-squad`).

---

## 🚀 Features

- Loads and reads PDF files using `PyPDF`
- Extracts all text from the PDF
- Uses Hugging Face's `transformers` pipeline for question-answering
- Answers user questions based on the document's content

---

## 🛠️ Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
