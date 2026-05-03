import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore

def get_vectorstore():
    """Connects to the Pinecone cloud database with a more robust embedding call."""
    
    # Ensure the key is present
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_key:
        raise ValueError("HUGGINGFACE_API_KEY is missing from environment variables.")

    # 1. Use the Inference API (Standardized Model Name)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_key,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 2. Connect to Pinecone
    vectorstore = PineconeVectorStore(
        index_name="vulcan-manuals",
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    return vectorstore

def search_manual(query):
    """Searches the indexed PDF manual for technical repair steps."""
    try:
        vectorstore = get_vectorstore()
        # Retrieve the top 3 most relevant paragraphs
        results = vectorstore.similarity_search(query, k=3)
        context = "\n".join([res.page_content for res in results])
        return context
    except Exception as e:
        return f"Error searching technical manual: {str(e)}"