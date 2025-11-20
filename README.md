# AmbedkarGPT - Intern Task

This is a local RAG (Retrieval-Augmented Generation) system built for the AI Intern assignment. It answers questions based on Dr. B.R. Ambedkar's speeches using LangChain, ChromaDB, and Ollama (Mistral 7B).

## Project Structure
* main.py: The interactive CLI for asking questions.
* evaluation.py: Testing script that runs the 25-question dataset and calculates metrics.
* data/: Contains the speech corpus.
* output/: Stores the vector database and test results.
* results_analysis.md: Analysis of performance and chunking strategies.

## Setup & Installation

1. Clone and Setup Environment
    git clone https://github.com/anvitvermaa/AmbedkarGPT-Intern-Task.git
    cd AmbedkarGPT-Intern-Task
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

2. Initialize Data
    python setup_data.py

3. Setup Ollama
    ollama pull mistral
    ollama serve

## Usage

To run the chatbot:
python main.py

To run the evaluation:
python evaluation.py

## Results
After optimization (k=5, overlap=100), the system achieved a 100% Hit Rate across all chunk sizes, with the 550-character strategy providing the highest answer quality (0.60 Cosine Similarity). See results_analysis.md for details.