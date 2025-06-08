from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.faiss import FaissVectorStore

from model import model, tokenizer

# Embedding model (unchanged)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    generate_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.7
    },
    is_chat_model=True
)

# Load and split text
documents = SimpleDirectoryReader("data").load_data()
splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=200)
nodes = splitter.get_nodes_from_documents(documents)

# Build FAISS vector store and index
faiss_store = FaissVectorStore.from_documents(documents, embed_model=Settings.embed_model)
storage_context = StorageContext.from_defaults(vector_store=faiss_store)

index = VectorStoreIndex(nodes, storage_context=storage_context)
query_engine = index.as_query_engine()

def query_llm(question: str) -> str:
    response = query_engine.query(question)
    print(f"response: {response}")
    text = str(response).strip()
    if not text:
        return "Sorry, I couldn't find an answer."
    return text[:2000] + "..." if len(text) > 2000 else text
