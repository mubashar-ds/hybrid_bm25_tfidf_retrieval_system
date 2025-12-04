import pickle
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import scipy.sparse as sp
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]

datasets_dir = project_root / 'datasets'

indexes_dir = project_root / 'indexes'

cleaned_data_path = datasets_dir / r'Articles_Cleaned.csv'

bm25_path = indexes_dir / 'bm25_index.pkl'
tfidf_vector_path = indexes_dir / 'tfidf_vectorizer.pkl'
tfidf_matix_path = indexes_dir / 'tfidf_matrix.npz'

def scatter_plot(num_samples = 200):

    df = pd.read_csv(cleaned_data_path, encoding = 'latin-1')

    with open(bm25_path, 'rb') as f:
        bm25 = pickle.load(f)

    with open(tfidf_vector_path, 'rb') as f:
        vectorizer = pickle.load(f)

    tfidf_matrix = sp.load_npz(tfidf_matix_path)

    # now pick a random query (its heading)..

    query = random.choice(df['Heading'].astype(str).tolist())
    q_vec = vectorizer.transform([query.lower()])

    # computing scores...

    bm25_scores = bm25.get_scores(query.lower().split())
    tfidf_scores = cosine_similarity(tfidf_matrix, q_vec).ravel()

    # sample n points to plot..

    idx = np.random.choice(len(df), num_samples, replace = False)
    bm25_sample = bm25_scores[idx]
    tfidf_sample = tfidf_scores[idx]

    plt.figure(figsize = (10,7))
    plt.scatter(bm25_sample, tfidf_sample, alpha = 0.7)
    plt.title('BM25 vs TF-IDF Scores (Random Sample)')
    plt.xlabel('BM25 Score')
    plt.ylabel('TF–IDF Cosine Score')
    plt.show()

if __name__ == '__main__':
    scatter_plot()
