import streamlit as st
import os
import faiss
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from groq import Groq
from Key import API_KEY
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Groq client
client = Groq(api_key=API_KEY)

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
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = faiss_index.search(query_embedding, k=5)
        return indices[0], distances[0]
    except Exception as e:
        logging.error(f"Error in query_index: {e}")
        raise

def load_faiss_index():
    try:
        faiss_index_path = 'faiss_index.index'
        if not os.path.exists(faiss_index_path):
            st.error(f"FAISS index file not found: {faiss_index_path}")
            return None
        faiss_index = faiss.read_index(faiss_index_path)
        logging.info("FAISS index loaded successfully.")
        return faiss_index
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

def main():
    st.title("Book Search and Q&A App")
    st.header("Search for books and ask questions!")

    # Load FAISS index
    faiss_index = load_faiss_index()
    if faiss_index is None:
        st.stop()

    # Book search
    search_query = st.text_input("Enter a book search query:")
    if st.button("Search Books"):
        if search_query:
            indices, distances = query_index(search_query, faiss_index)
            st.write("Search Results:")
            for idx, dist in zip(indices, distances):
                if idx != -1:
                    st.write(f"Book ID: {idx}, Distance: {dist}")
                else:
                    st.write("No more relevant results found.")
        else:
            st.warning("Please enter a search query.")

    # Q&A with Groq
    st.header("Ask a question about books")
    user_question = st.text_input("Enter your question:")
    if st.button("Ask Question"):
        if user_question:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": user_question,
                        }
                    ],
                    model="llama3-8b-8192",
                )
                st.write("LLM Response:")
                st.write(chat_completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Error querying Groq: {e}")
        else:
            st.warning("Please enter a question.")

    # Show a footer message
    st.write("Thank you for using the Book Search and Q&A App!")

if __name__ == "__main__":
    # Set OpenMP environment variable
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()