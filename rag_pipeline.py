from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.faiss import FaissVectorStore

from model import model, tokenizer

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load text chunks
documents = SimpleDirectoryReader("data").load_data()

# Build vector store index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def query_llm(question: str) -> str:
    response = query_engine.query(question)
    print(f"response: {response}")
    text = str(response).strip()
    if not text:
        return "Sorry, I couldn't find an answer."
    return text[:2000] + "..." if len(text) > 2000 else text