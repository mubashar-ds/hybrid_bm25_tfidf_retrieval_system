import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from .search_engine import SearchEngine


def recall_at_k(rank_list, relevant_id, k):
    return precision_at_k(rank_list, relevant_id, k)

def precision_at_k(rank_list, relevant_id, k):

    topk = rank_list[:k]
    return 1.0 if relevant_id in topk else 0.0

def NDCG_at_k(rank_list, relevant_id, k):
    
    topk = rank_list[:k]
    
    DCG = 0.0
    for idx, doc_id in enumerate(topk, start = 1):
        if doc_id == relevant_id:
            DCG = 1.0 / np.log2(idx + 1)

    IDCG = 1.0  

    return DCG / IDCG

def reciprocal_rank(rank_list, relevant_id):
    
    for i, doc_id in enumerate(rank_list, start = 1):
        if doc_id == relevant_id:
            return 1.0 / i
    return 0.0

# baseline 1 : pure tf-idf cosine...

def tfidf_baseline(engine, query_text, top_k = 25):
    
    q_vec = engine.vectorizer.transform([query_text])
    sim = cosine_similarity(engine.tfidf_matrix, q_vec).ravel()
    return np.argsort(sim)[::-1][:top_k].tolist()

# baseline 2 : pure bm25 ...

def bm25_baseline(engine, query_tokens, top_k = 25):

    scores = engine.bm25.get_scores(query_tokens)
    return np.argsort(scores)[::-1][:top_k].tolist()

# complete system : hybrid engine that i already implemented...

def hybrid_system(engine, query, top_k = 25):

    results = engine.search(query, top_k = top_k)
    return [r['doc_id'] for r in results]

# finaly the evaluation pipeline...

def run_evaluation(num_queries = 50, k = 10):

    project_root = Path(__file__).resolve().parents[1]

    datasets_dir = project_root / 'datasets'

    cleaned_data_path = datasets_dir / r'Articles_Cleaned.csv'

    df = pd.read_csv(cleaned_data_path, encoding = 'latin-1')

    # so finally initializing the engine just once..

    engine = SearchEngine()

    # to choose the first N headings as test the queries..

    testing_queries = df['Heading'].fillna('').astype(str).tolist()[:num_queries]

    print(f'\n--> Running the evaluation on {num_queries} queries...')

    # below are the containers for the metrics..

    bm25_precision, tfidf_precision, hybrid_precision = [], [], []
    bm25_recall, tfidf_recall, hybrid_recall = [], [], []
    bm25_mrr, tfidf_mrr, hybrid_mrr = [], [], []
    bm25_ndgc, tfidf_ndgc, hybrid_ndgc = [], [], []

    for q_idx, query in enumerate(testing_queries):

        relevant_documents = q_idx  

        # tf-idf baseline...

        tfidf_ranked = tfidf_baseline(engine, query.lower(), top_k = k)

        tfidf_mrr.append(reciprocal_rank(tfidf_ranked, relevant_documents))
        tfidf_ndgc.append(NDCG_at_k(tfidf_ranked, relevant_documents, k))

        tfidf_recall.append(recall_at_k(tfidf_ranked, relevant_documents, k))
        tfidf_precision.append(precision_at_k(tfidf_ranked, relevant_documents, k))

        # bm25  baseline..

        bm25_ranked = bm25_baseline(engine, query.lower().split(), top_k = k)

        bm25_mrr.append(reciprocal_rank(bm25_ranked, relevant_documents))
        bm25_ndgc.append(NDCG_at_k(bm25_ranked, relevant_documents, k))

        bm25_recall.append(recall_at_k(bm25_ranked, relevant_documents, k))
        bm25_precision.append(precision_at_k(bm25_ranked, relevant_documents, k))

        # hybrid system ....

        hybrid_ranked = hybrid_system(engine, query, top_k = k)

        hybrid_mrr.append(reciprocal_rank(hybrid_ranked, relevant_documents))
        hybrid_ndgc.append(NDCG_at_k(hybrid_ranked, relevant_documents, k))

        hybrid_recall.append(recall_at_k(hybrid_ranked, relevant_documents, k))
        hybrid_precision.append(precision_at_k(hybrid_ranked, relevant_documents, k))
    
    # Final Results.........

    print('\n--------> EVALUATION RESULTS\n')


    print(f'Recall@{k} : ')

    print(f'TF-IDF :    {np.mean(tfidf_recall) : .3f}')
    print(f'BM25 :      {np.mean(bm25_recall) : .3f}')
    print(f'Hybrid :    {np.mean(hybrid_recall) : .3f}\n')

    print(f'Precision@{k} : ')

    print(f'TF-IDF :     {np.mean(tfidf_precision) : .3f}')
    print(f'BM25 :       {np.mean(bm25_precision) : .3f}')
    print(f'Hybrid :    {np.mean(hybrid_precision) : .3f}\n')

    print(f'nDCG@{k} : ')

    print(f'TF-IDF :    {np.mean(tfidf_ndgc) : .3f}')
    print(f'BM25 :      {np.mean(bm25_ndgc) : .3f}')
    print(f'Hybrid :    {np.mean(hybrid_ndgc ) : .3f}')


    print('Mean Reciprocal Rank (MRR) : ')

    print(f'TF-IDF :    {np.mean(tfidf_mrr) : .3f}')
    print(f'BM25 :      {np.mean(bm25_mrr) : .3f}')
    print(f'Hybrid :    {np.mean(hybrid_mrr) : .3f}\n')

    print('\nFinally the evaluation is completed!')

if __name__ == '__main__':
    run_evaluation(num_queries = 50, k = 10)
