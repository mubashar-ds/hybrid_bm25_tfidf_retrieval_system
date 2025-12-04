import pandas as pd
import numpy as np

import pickle
from pathlib import Path
import re

from sklearn.metrics.pairwise import cosine_similarity

import scipy.sparse as sp

from .queries_parsing import parse_query   
from .correct_spelling import correcting_query_terms


project_root = Path(__file__).resolve().parents[1]

datasets_dir = project_root / 'datasets'

indexes_dir = project_root / 'indexes'

cleaned_data_path = datasets_dir / r'Articles_Cleaned.csv'

bm25_path = indexes_dir / 'bm25_index.pkl'
tfidf_vector_path = indexes_dir / 'tfidf_vectorizer.pkl'
tfidf_matix_path = indexes_dir / 'tfidf_matrix.npz'

def cleaning_query(q: str) -> str:
    
    if not isinstance(q, str):
        return ''
    q = q.lower().strip()
    q = re.sub(r'http\\S+|www\\.\\S+', ' ', q)
    q = re.sub(r'\\s+', ' ', q)
    return q

def extracting_phrases(q: str):
   
    return re.findall(r'"([^"]+)"', q)

def removing_phrases_from_query(q: str, phrases):
    
    for ph in phrases:
        q = q.replace(f'"{ph}"', '')
    return q.strip()


class SearchEngine:
    def __init__(self):

        if not cleaned_data_path.exists():
            raise FileNotFoundError(f'\n----> Could not find the cleaned dataset : {cleaned_data_path}')
        
        self.df = pd.read_csv(cleaned_data_path, encoding = 'latin-1')

        # loding the bm25..

        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        # Loading teh tf-idf vectorizer and matrix...

        with open(tfidf_vector_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.tfidf_matrix = sp.load_npz(tfidf_matix_path)

        print('\n--> Finally the Search engine is initialized successfully!')

    def search(self, user_query: str, top_k = 10, bm25_k = 200):
        
        if not isinstance(user_query, str) or not user_query.strip():
            return []

        # step 1 : parse the user query..

        parsed = parse_query(user_query)

        tokens   = parsed['tokens']
        phrases  = parsed['phrases']
        must_have_terms = parsed['AND']
        banned_terms    = parsed['NOT']
        category_filter = parsed['category']
        date_filter     = parsed['date_filter']

        # if no tokens (only phrases), then provide the fallback...

        if not tokens and not phrases:
            return []
        
        tokens = correcting_query_terms(tokens)
        must_have_terms = correcting_query_terms(must_have_terms)

        # step 2 : bm25 on positive tokens..

        bm25_scores = self.bm25.get_scores(tokens)
        top_ids = np.argsort(bm25_scores)[::-1][:bm25_k]

        # step 3 : basic filtering (AND, NOT, category, date)..

        filtered_ids = []

        for idx in top_ids:
            row = self.df.loc[idx]

            # it will provide full clean text tokens...

            doc_tokens = str(row['full_text_tokens']).split()

            ok = True

             # NOT filter..
            
            for t in banned_terms:
                if t in doc_tokens:
                    ok = False
                    break
            if not ok:
                continue

            # AND filter..

            ok = True

            for t in must_have_terms:
                if t not in doc_tokens:
                    ok = False
                    break
            if not ok:
                continue

            # Date filter..

            if date_filter:
                doc_date = str(row['Date'])  

                # convert to (YYYY-MM) format for comparison..

                try:
                    month, day, year = doc_date.split('/')
                    doc_ym = f'{year}-{month.zfill(2)}'
                except:
                    doc_ym = ''

                if date_filter['type'] == 'year':
                    if date_filter['value'] != year:
                        continue
                else:
                    start = date_filter['start']
                    end   = date_filter['end']
                    if not (start <= doc_ym <= end):
                        continue

            #  Category filter ...

            if category_filter is not None:
                if str(row['NewsType']).lower() != category_filter:
                    continue

            filtered_ids.append(idx)

        # if filters removed everything, then fallback to bm25 top_ids..

        if not filtered_ids:
            filtered_ids = top_ids

        # step 4, tf-idf cosine similarity...

        if tokens:
            q_vec = self.vectorizer.transform([' '.join(tokens)])
            cand_matrix = self.tfidf_matrix[filtered_ids]
            cosine_scores = cosine_similarity(cand_matrix, q_vec).ravel()
        else:
            cosine_scores = np.zeros(len(filtered_ids))

        # step 5, then heading boosting + Phrase boosting...

        heading_boost = []
        phrase_boost = []

        for idx in filtered_ids:

            row = self.df.loc[idx]
            head = str(row['heading_tokens']).split()

            # count the overlapping query tokens in heading..

            overlap = sum(1 for t in tokens if t in head)
            heading_boost.append(overlap)

            # phrase match boosting ...

            full_text = str(row['cleaned_full_text']).lower()
            match = any(ph.lower() in full_text for ph in phrases)
            phrase_boost.append(1 if match else 0)

        heading_boost = np.array(heading_boost, dtype = float)
        phrase_boost  = np.array(phrase_boost, dtype = float)

        if heading_boost.max() > 0:
            heading_boost /= heading_boost.max()

        # step 6, then combine scores..

        bm25_used = bm25_scores[filtered_ids]
        bm25_used = bm25_used / (np.max(bm25_used) + 1e-6)

        final_scores = (0.45 * bm25_used + 0.35 * cosine_scores + 0.15 * heading_boost + 0.05 * phrase_boost)

        # step 7, sorting and return top k documents...

        # sorted_ids = np.array(filtered_ids)[np.argsort(final_scores)[::-1]]

        # combinignthe ids and scores, then will sort by score..

        combined_results = list(zip(top_ids, final_scores))
        combined_results.sort(key = lambda x: x[1], reverse = True)
        
        # now taaking the top k results with their scores..

        final_results_with_scores = combined_results[:top_k] 
        
        results = []

        for rank, (doc_id, score) in enumerate(final_results_with_scores, start = 1):
            row = self.df.loc[doc_id]
            
            cleaned_head = str(row['cleaned_heading']).strip()

            full_clean_text = str(row['cleaned_full_text']).strip()
            
            start_index = full_clean_text.find(cleaned_head) + len(cleaned_head) + 2 
            
            preview_text = full_clean_text[start_index:].strip()
            
            preview_text = preview_text.replace('strong>', '').replace('</strong', '').strip()

            results.append({
                'rank' : rank,
                'doc_id' : int(doc_id),
                'heading' : row['Heading'],
                'date' : row['Date'],
                'news_type' : row['NewsType'],
                'score' : float(score), 
                'preview' : preview_text[:220] + '...' 
            })
            
        return results

# CLI test ...

if __name__ == '__main__':
    engine = SearchEngine()

    while True:
        q = input("\n--> Enter your query (or 'exit') : ").strip()
        if q.lower() == 'exit':
            print('\nHave a nice day!\n')
            break

        hits = engine.search(q, top_k = 5)
        print('\n--> Top Results : ')
        for h in hits:
            print(f"\n--> [{h['rank']}] ({h['news_type']}) {h['heading']} - {h['date']}")
            print('\n   ', h['preview'])
