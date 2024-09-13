import faiss
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BookEmbeddingExtractor:
    def __init__(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            logging.info(f"Model and tokenizer loaded. Using device: {self.device}")
        except Exception as e:
            logging.error(f"Error initializing BookEmbeddingExtractor: {e}")
            raise

    def get_embedding(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            raise

def query_index(query_text, faiss_index):
    try:
        extractor = BookEmbeddingExtractor()
        query_embedding = extractor.get_embedding(query_text)
        logging.info("Query embedding computed.")
        
        # Ensure the query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = faiss_index.search(query_embedding, k=5)
        return indices[0], distances[0]  # Return only the first row
    except Exception as e:
        logging.error(f"Error in query_index: {e}")
        raise

def main():
    try:
        # Set OpenMP environment variable
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        faiss_index_path = 'faiss_index.index'
        if not os.path.exists(faiss_index_path):
            logging.error(f"FAISS index file not found: {faiss_index_path}")
            return

        faiss_index = faiss.read_index(faiss_index_path)
        logging.info("FAISS index loaded successfully.")

        query_text = "Fever, lethargy, loss of appetite"
        logging.info(f"Querying index with: '{query_text}'")
        
        indices, distances = query_index(query_text, faiss_index)

        logging.info(f"Indices of relevant segments: {indices}")
        logging.info(f"Distances to relevant segments: {distances}")

    except Exception as e:
        logging.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()