import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings # Match retrieval
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_document():
    print("🚀 Starting Cloud-Optimized Ingestion...")
    file_path = "data/WEG-WMO-Installation-Operation-and-Maintenance-Manual-of-Electric-Motors.pdf"
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Use Inference API for ingestion too
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("☁️ Uploading to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name="vulcan-manuals"
    )
    print("🎉 Ingestion Complete!")

if __name__ == "__main__":
    ingest_document()