import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if the directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The specified directory {docs_path} does not exist. Please create the 'docs' directory and add your company documents in .txt format.")
    
    # Load all text files from the directory
    loader = DirectoryLoader(
        path=docs_path, 
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True,
        }
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise ValueError(f"No text files found in {docs_path}. Please add your company documents in .txt format to the 'docs' directory.")

    for i, doc in enumerate(documents[:2]):  # Print the first 2 documents for verification
        print(f"Document {i+1}")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")  # Print the first 100 characters of the content
        print(f"  Metadata: {doc.metadata}")
        
    return documents
    
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):  # Print the first 5 chunks for verification
            print(f"\n--- Chunk {i+1} ---")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  Length: {len(chunk.page_content)} characters")
            print(f"  Content :")
            print(chunk.page_content)
            print("-" * 50)
    
        if len(chunks) > 5:
            print(f"\n...and {len(chunks) - 5} more chunks.")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create a persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create a Chroma vector store
    print("--- Creating Chroma vector store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("--- Finished creating Chroma vector store ---")

    print(f"Vector store created and saved to {persist_directory}.")
    
    return vector_store
    
def main():
    print("Main Function")
    
    #1 Load documents from the specified directory
    documents = load_documents(docs_path="docs")
    
    #2 Split documents into chunks
    chunks = split_documents(documents)
    
    #3 Create embeddings for the chunks
    vector_store = create_vector_store(chunks)
    
if __name__ == "__main__":
    main()