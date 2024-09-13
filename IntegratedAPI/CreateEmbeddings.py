import torch
import warnings
from transformers import BertTokenizer, BertModel
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class BookEmbeddingExtractor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to GPU
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move tensors back to CPU

def create_embeddings(segments):
    extractor = BookEmbeddingExtractor()
    embeddings = []
    for segment in segments:
        embedding = extractor.get_embedding(segment)
        embeddings.append(embedding)
    return np.vstack(embeddings)

if __name__ == "__main__":
    # Open file with utf-8 encoding
    with open('preprocessed_segments.txt', 'r', encoding='utf-8') as file:
        segments = file.readlines()

    embeddings = create_embeddings(segments)
    np.save('book_embeddings.npy', embeddings)
