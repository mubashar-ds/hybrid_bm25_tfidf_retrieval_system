from spellchecker import SpellChecker
import pandas as pd
from pathlib import Path
import os


project_root = Path(__file__).resolve().parents[1]

datasets_dir = project_root / 'datasets'

cleaned_data_path = datasets_dir / r'Articles_Cleaned.csv'

spell = SpellChecker()

try:
    df = pd.read_csv(cleaned_data_path)

    all_tokens = ' '.join(df['full_text_tokens'].dropna().astype(str)).split()
    
    spell.word_frequency.load_words(all_tokens)
    
except Exception as e:
    print(f'Could not load corpus for spell correction : {e}')

def correcting_query_terms(tokens):

    corrected = []

    for t in tokens:
        if len(t) <= 2:  
            corrected.append(t)
            continue

        corrected_term = spell.correction(t)
        
        if corrected_term is not None:
            corrected.append(corrected_term)

        else:
            corrected.append(t)
            
    return corrected
