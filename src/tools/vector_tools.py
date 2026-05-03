import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore

def get_vectorstore():
    """Connects to the Pinecone cloud database using the Inference API."""
    
    # 1. Use the API for lightweight queries (Uses Cloud RAM, not yours!)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L6-v2" 
    )
    
    # 2. Connect to the existing Pinecone index
    vectorstore = PineconeVectorStore(
        index_name="vulcan-manuals",
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    return vectorstore

def search_manual(query):
    """Searches the indexed PDF manual for technical repair steps."""
    vectorstore = get_vectorstore()
    # Retrieve the top 3 most relevant paragraphs
    results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([res.page_content for res in results])
    return context