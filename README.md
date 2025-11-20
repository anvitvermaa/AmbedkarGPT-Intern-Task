# AmbedkarGPT - Intern Task

This repository contains a local RAG (Retrieval-Augmented Generation) system designed to answer questions based on Dr. B.R. Ambedkar's speeches. The project consists of a functional command-line prototype (Phase 1) and a comprehensive evaluation suite (Phase 2).

## Project Overview
The goal was to build a system that allows users to query specific historical documents without relying on external paid APIs. The system ingests text, creates embeddings, and retrieves relevant context for a local LLM (Mistral 7B) to generate accurate answers.

## Evaluation & Optimization
I developed a custom testing script (evaluation.py) to benchmark the system against a dataset of 25 verified questions using three different chunking strategies: 250, 550, and 900 characters.

### Optimization from 95% to 100% Accuracy
During the initial testing phase, the system achieved a 95% Hit Rate. I identified that the remaining 5% of errors came from "Comparative Questions" (e.g., comparing views in Document 1 vs. Document 6). The default retrieval depth (k=3) often retrieved three chunks from the first document but failed to fetch the necessary context from the second document.

To resolve this, I implemented two specific engineering changes:
1. Increased Retrieval Depth (k=5): Retrieving 5 chunks instead of 3 ensured that context from multiple documents was captured.
2. Increased Overlap (100 chars): This prevented list-based answers from being cut mid-sentence.

These optimizations resulted in a perfect 1.00 (100%) Hit Rate across all chunking strategies in the final run.

### Note on BM25 / Hybrid Search
While implementing a Hybrid Search (BM25 + Vector) would be the standard approach for production systems to ensure exact keyword matching, I intentionally restricted this implementation to Pure Vector Search.

This decision was made to strictly adhere to the assignment's request for a "Comparative Chunking Analysis." Introducing keyword search would have masked the performance differences between the chunk sizes (Small/Medium/Large), making the evaluation data less useful for the specific analysis requested in the brief.

## Setup & Installation

1. Clone the repository and install dependencies:
   git clone 
   cd AmbedkarGPT-Intern-Task
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Generate the data files:
   python setup_data.py

3. Start the Ollama server:
   ollama pull mistral
   ollama serve

## How to Run

Phase 1: Run the Q&A Chatbot
python main.py

Phase 2: Run the Evaluation
python evaluation.py

## Results Summary
After optimization, the system achieved a 100% Hit Rate. The evaluation data suggests that the Medium Chunk strategy (550 characters) offers the best balance, providing the highest semantic similarity scores (answer quality) while maintaining perfect retrieval accuracy.
