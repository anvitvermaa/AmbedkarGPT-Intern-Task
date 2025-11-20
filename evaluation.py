import os
import json
import shutil
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# --- Configuration ---
CORPUS_PATH = "data/corpus"
TEST_DATA_PATH = "test_dataset.json"
OUTPUT_FILE = "test_results.json"
MODEL_NAME = "mistral"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Silent NLTK download
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_test_data():
    with open(TEST_DATA_PATH, 'r') as f:
        data = json.load(f)
        return data['test_questions']

def calculate_retrieval_metrics(retrieved_docs, true_filenames):
    hits = 0
    reciprocal_rank = 0
    
    retrieved_filenames = [os.path.basename(doc.metadata['source']) for doc in retrieved_docs]
    
    # HIT RATE: Did we find the right document?
    if any(f in true_filenames for f in retrieved_filenames):
        hits = 1
    
    # MRR: How high up was the right document?
    for i, fname in enumerate(retrieved_filenames):
        if fname in true_filenames:
            reciprocal_rank = 1 / (i + 1)
            break
            
    return hits, reciprocal_rank

def calculate_text_metrics(generated_answer, ground_truth, embedding_model):
    # ROUGE (Overlap)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, generated_answer)
    rouge_l = scores['rougeL'].fmeasure

    # BLEU (Translation Quality)
    try:
        reference = [nltk.word_tokenize(ground_truth.lower())]
        candidate = nltk.word_tokenize(generated_answer.lower())
        bleu = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
    except:
        bleu = 0.0

    # COSINE SIMILARITY (Meaning)
    embeddings = embedding_model.embed_documents([generated_answer, ground_truth])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return rouge_l, bleu, similarity

def run_evaluation_for_chunk_size(chunk_size, questions, embedding_model):
    print(f"\n---  Testing Chunk Strategy: {chunk_size} chars ---")
    
    db_path = f"output/chroma_{chunk_size}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # 1. Split Documents (Increased overlap to prevent cut-off sentences)
    loader = DirectoryLoader(CORPUS_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # OPTIMIZATION 1: Increase overlap to 100 to save small chunks like Q22
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    
    # 2. Embed & Store
    db = Chroma.from_documents(texts, embedding_model, persist_directory=db_path)
    
    # OPTIMIZATION 2: Increase k to 5 (Top-5 retrieval ensures we find the doc)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # 3. Setup Chain
    llm = Ollama(model=MODEL_NAME)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )
    
    results = []
    
    # 4. Run Questions
    for item in questions:
        if not item['answerable']:
            continue

        try:
            response = qa_chain.invoke({"query": item['question']})
            hits, mrr = calculate_retrieval_metrics(response['source_documents'], item['source_documents'])
            rouge, bleu, cos = calculate_text_metrics(response['result'], item['ground_truth'], embedding_model)
            
            results.append({
                "question_id": item['id'],
                "hit_rate": hits,
                "mrr": mrr,
                "rouge_l": rouge,
                "cosine_sim": cos
            })
            print(f"  [Q{item['id']}] Hit: {hits} | Cosine: {cos:.2f}")
            
        except Exception as e:
            print(f"  [Error Q{item['id']}]: {e}")

    avg_hit = np.mean([r['hit_rate'] for r in results])
    print(f" Result for {chunk_size}: Avg Hit Rate={avg_hit:.2f}")
    
    return {"chunk_size": chunk_size, "avg_hit_rate": avg_hit, "detailed_results": results}

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    questions = load_test_data()
    
    # Strategies required by Assignment-2 PDF (Source: 27-29)
    strategies = [250, 550, 900] 
    
    final_report = {"model": MODEL_NAME, "strategies": []}
    
    for size in strategies:
        strategy_result = run_evaluation_for_chunk_size(size, questions, embeddings)
        final_report["strategies"].append(strategy_result)
        
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\n Evaluation Complete! File saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()