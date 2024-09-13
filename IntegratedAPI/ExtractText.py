# text_extraction.py
import re
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    pdf_document = fitz.open(file_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    # Basic preprocessing: remove extra whitespaces and split into segments
    text = re.sub(r'\s+', ' ', text)
    segments = text.split('\n')  # Assuming each segment is separated by a newline
    return [segment.strip() for segment in segments if segment.strip()]

if __name__ == "__main__":
    book_text = extract_text_from_pdf(r'../Books/Proceedings-66th-Annual-Convention-2020.pdf')
    preprocessed_segments = preprocess_text(book_text)
    # Save or use the preprocessed_segments
    with open('preprocessed_segments.txt', 'w', encoding='utf-8') as f:
        for segment in preprocessed_segments:
            f.write(segment + '\n')
