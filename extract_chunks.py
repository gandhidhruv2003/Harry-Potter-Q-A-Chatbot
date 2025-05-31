import fitz
import os
import textwrap

OUTPUT_DIR = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text):
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(text[i:i + CHUNK_SIZE])
    return chunks

def save_chunks(chunks):
    for i, chunk in enumerate(chunks):
        with open(os.path.join(OUTPUT_DIR, f"chunk_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(textwrap.fill(chunk))

if __name__ == "__main__":
    pdf_path = "harrypotter.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    save_chunks(chunks)
    print(f"Saved {len(chunks)} text chunks to '{OUTPUT_DIR}/'")