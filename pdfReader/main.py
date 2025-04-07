from transformers import pipeline
from pypdf import PdfReader

def read_pdf_text(file_path):
    document = PdfReader(file_path)
    text = ""
    for page in document.pages:
        text += page.extract_text() + "\n"
    return text

question = input("Type your question about the research paper: ")
context = read_pdf_text('./stock.pdf')  # Make sure the path is correct

QApipeline = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad",  framework="pt")
answer = QApipeline(question=question, context=context)

print(f"Ans: {answer['answer']}")
