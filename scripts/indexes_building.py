import pandas as pd
from tqdm import tqdm

import pickle
from pathlib import Path

import scipy.sparse as sp
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

project_root = Path(__file__).resolve().parents[1]

datasets_dir = project_root / 'datasets'

indexes_dir = project_root / 'indexes'

cleaned_data_path = datasets_dir / r'Articles_Cleaned.csv'

bm25_path = indexes_dir / 'bm25_index.pkl'
tfidf_vector_path = indexes_dir / 'tfidf_vectorizer.pkl'
tfidf_matix_path = indexes_dir / 'tfidf_matrix.npz'

def loading_cleaned_data():

    if not cleaned_data_path.exists():
        raise FileNotFoundError(f'\n! ----> Cleaned dataset is not found : {cleaned_data_path}')

    print(f'\n--> Loading cleaned dataset from: {cleaned_data_path}')
    df = pd.read_csv(cleaned_data_path, encoding = 'latin-1')

    # here it expecte the cleaned text columns from the a_preprocessing.py ...

    if 'full_text_tokens' not in df.columns:
        raise ValueError('\n! ----> Expected column (full_text_tokens) is not found in the cleaned dataset.')

    return df

# bm25 indexing...
 
def building_bm25_index(token_lists):
   
    print('\n--> Now building the BM25 index..')
    tokenized_docs = [text.split() for text in token_lists]

    # i using BM25Okapi for my use case.

    bm25 = BM25Okapi(tokenized_docs)
    print('\n--> BM25 index is built successfully!')
    return bm25


# tf-idf Indexing...

def building_tfidf_index(raw_docs):
   
    print('\n--> Now building the TF-IDF vectorizer...')

    vectorizer = TfidfVectorizer(ngram_range = (1, 2), min_df = 3, max_df = 0.80, stop_words = None)

    tfidf_matrix = vectorizer.fit_transform(raw_docs)

    print(f'\n--> TF-IDF matrix shape is : {tfidf_matrix.shape}')
    print('\n--> So, TF-IDF index is built successfully!')

    return vectorizer, tfidf_matrix

# ,ain build function..

def building_all_indexes():
    
    df = loading_cleaned_data()

    # tokenized the text for bm25 that come from spacy output...

    token_texts = df['full_text_tokens'].astype(str).tolist()

    # raw cleaned text for tf-idf (it is also token based but kept as strings)..
    
    tfidf_docs = df['full_text_tokens'].astype(str).tolist()

    # now creating the indexes directory...

    indexes_dir.mkdir(parents = True, exist_ok = True)

    # bm25 Index...

    bm25 = building_bm25_index(token_texts)

    print(f'\n--> Saving the BM25 index to the path : {bm25_path}')
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)

    # building tfidf Index...

    vectorizer, tfidf_matrix = building_tfidf_index(tfidf_docs)

    print(f'\n--> Saving the TF-IDF vectorizer to the path: {tfidf_vector_path}')
    with open(tfidf_vector_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f'\n--> Saving the TF-IDF matrix to the path : {tfidf_matix_path}')
    sp.save_npz(tfidf_matix_path, tfidf_matrix)

    print('\n--> All indexes are built and saved successfully!\n')

if __name__ == '__main__':
    building_all_indexes()
