# app.py

from data_loader import process_all_pdfs, split_documents
from embeddings import EmbeddingManager
from vectorstore import VectorStore
from rag_retriever import RAGRetriever
from rag import rag, rag_advance

# 1) Load PDFs
all_pdf_documents = process_all_pdfs("data/")
if all_pdf_documents:
    print("pdfs processed")

# 2) Split into chunks
chunks = split_documents(all_pdf_documents)
if chunks:
    print("chunks loaded")

# 3) Convert chunks to text
texts = [doc.page_content for doc in chunks]
if texts:
    print("texts loded")

# 4) Generate Embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(texts)

if embeddings.all():
    print("embeddings loaded")

# 5) Store in Chroma
vectorstore = VectorStore()
vectorstore.add_documents(chunks, embeddings)
print("embeddings added to vector store")

# 6) Create Retriever
rag_retriever = RAGRetriever(vectorstore, embedding_manager)
if rag_retriever:
    print("rag retriver ready")


query = "What is attention is all you need?"

result = rag_advance(query, rag_retriever, top_k=3, return_context=True)

print("\nAnswer:", result['answer'])
print("\nSources:", result['sources'])
print("\nConfidence:", result['confidence'])
print("\nContext Preview:", result['context'][:200],"...")
