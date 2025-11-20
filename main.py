import os
import sys
import time

# Modern LangChain imports (Standard for 2024/2025)
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Configuration Constants
# We use "speech.txt" as required by Assignment 1 [cite: 8]
SOURCE_DOCUMENT = "speech.txt"
VECTOR_STORE_PATH = "output/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

def main():
    """
    Main execution function for the AmbedkarGPT RAG Pipeline.
    """
    print("--- Starting AmbedkarGPT Initialization ---")

    # 1. Load the provided text file [cite: 8]
    if not os.path.exists(SOURCE_DOCUMENT):
        print(f"Error: {SOURCE_DOCUMENT} not found. Please run setup_data.py first.")
        sys.exit(1)

    print(f"Loading {SOURCE_DOCUMENT}...")
    loader = TextLoader(SOURCE_DOCUMENT, encoding="utf-8")
    documents = loader.load()

    # 2. Split the text into manageable chunks [cite: 9]
    # We use RecursiveCharacterTextSplitter (industry standard) instead of simple CharacterTextSplitter
    # because it respects sentence boundaries better, leading to higher quality answers.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Medium chunk size (good starting point)
        chunk_overlap=50      # Overlap ensures context isn't lost between cuts
    )
    texts = text_splitter.split_documents(documents)
    print(f"Document split into {len(texts)} chunks.")

    # 3. Create Embeddings & Store in Vector Database [cite: 10, 16, 17]
    print("Initializing Embeddings (HuggingFace)...")
    # Using the CPU-friendly all-MiniLM-L6-v2 as requested [cite: 17]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating/Loading Vector Store (ChromaDB)...")
    # We persist the DB to 'output/' to keep the repo clean
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    # 4. Initialize the Retrieval QA Chain [cite: 15]
    print(f"Initializing LLM ({LLM_MODEL})...")
    try:
        # We use Ollama locally as requested [cite: 18]
        llm = Ollama(model=LLM_MODEL)
        
        # Create the retriever (The interface to find relevant chunks) [cite: 11]
        retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
        
        # Create the Chain [cite: 12]
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "Stuff" simply stuffs the context into the prompt
            retriever=retriever,
            return_source_documents=True # Useful for debugging
        )
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Ensure Ollama is running: 'ollama serve' in a separate terminal.")
        sys.exit(1)

    print("\n--- System Ready! ---\n")
    print("Ask a question about Dr. Ambedkar's speech (or type 'exit' to quit).")

    # 5. Interactive Loop
    while True:
        try:
            query = input("\nUser Question: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            if not query.strip():
                continue

            print("Thinking...")
            start_time = time.time()
            
            # Run the query through the chain
            response = qa_chain.invoke({"query": query})
            
            end_time = time.time()
            
            # Output the result
            print(f"\nAnswer: {response['result']}")
            print(f"\n(Response time: {end_time - start_time:.2f}s)")
            
            # Optional: Show where the info came from (Makes you look pro)
            # print("\n[Source Context Used]:")
            # for doc in response['source_documents']:
            #     print(f"- {doc.page_content[:100]}...")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

