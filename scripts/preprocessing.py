import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import spacy

import html

project_root = Path(__file__).resolve().parents[1]

datasets_dir = project_root / 'datasets'

raw_data_path = datasets_dir / r'Articles_Raw.csv'
cleaned_data_path = datasets_dir / r'Articles_Cleaned.csv'

# url_pattern = re.compile(r'http\S+|www\.\S+', re.IGNORECASE)

url_pattern = re.compile(r'(?:http[s]?://|www\.)\S+|[a-zA-Z0-9-]+\.(?:com|org|net|pk|co|info)\S*', re.IGNORECASE)
whitespace_patten = re.compile(r'\s+')

# html_tag_pattern = re.compile(r"<.*?>", re.DOTALL)
# html_entity_pattern = re.compile(r'&[a-zA-Z]+;')

html_tag_pattern = re.compile(r"<[^>]+>", re.DOTALL)
html_entity_pattern = re.compile(r"&(#\d+|[a-zA-Z]+);")

def basic_cleanup(text: str) -> str:

    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text)

    text = text.lower()

    text = html_tag_pattern.sub(' ', text)

    text = html_entity_pattern.sub(' ', text)

    text = re.sub(r'(?:<|>|\/strong\s*|strong\s*)', ' ', text, flags=re.IGNORECASE)

    text = whitespace_patten.sub(' ', text).strip()

    text = url_pattern.sub(' ', text)

    text = text.encode('ascii', 'ignore').decode()

    return text

def init_spacy_model():
    
    nlp = spacy.load(
        'en_core_web_sm',
        disable = ['parser', 'ner', 'textcat']
    )
    return nlp

def spacy_clean_texts(texts, nlp):
    
    cleaned = []

    for doc in tqdm(nlp.pipe(texts, batch_size = 50), total = len(texts), desc = 'The Spacy Preprocessing'):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        cleaned.append(' '.join(tokens))

    return cleaned

def preprocessing_dataset():
   
    if not raw_data_path.exists():
        raise FileNotFoundError(f'\n! ----> Raw dataset is not found at : {raw_data_path}')

    print(f'\n--> Loading the raw dataset from : {raw_data_path}')
    
    df = pd.read_csv(raw_data_path, encoding = 'latin-1')

    expected_cols = {'NewsType', 'Heading', 'Article', 'Date'}

    missing = expected_cols.difference(df.columns)

    if missing:
        raise ValueError(f'\n! ----> Dataset is missing expected columns : {missing}')
    
    # here making a single text field..

    print('\n--> Now combining the heading and th article into a single text field...')
    df['full_text'] = (df['Heading'].fillna('') + '. ' + df['Article'].fillna('')).str.strip()

    # droping the rows where the full text is empty after the combination.,,

    before = len(df)
    df = df[df['full_text'].str.len() > 0].copy()

    after = len(df)
    print(f'\n--------> Dropped {before - after} empty rows.')

    # now removing the exact duplicates based on the full text...

    before = len(df)
    df = df.drop_duplicates(subset = ['full_text']).reset_index(drop = True)

    after = len(df)
    print(f'--------> Removed {before - after} exact duplicate documents.')

    # here the baisc cleanup is needed before spacy (such as lowercase, and  removign the URLs etc)...

    print('\n--> Running the basic regex cleanup...')
    df['cleaned_heading'] = df['Heading'].fillna('').apply(basic_cleanup)
    df['cleaned_full_text'] = df['full_text'].apply(basic_cleanup)

    # so now initializing the spacy model..,

    print('\n--> Loading the spacy model...')
    nlp = init_spacy_model()

    # creating a cleaned version of the heading only from the dataset...

    print('\n--> Now running the spacy preprocessing on the headings...')
    df['heading_tokens'] = spacy_clean_texts(df['cleaned_heading'].tolist(), nlp)

    # appply the spacy to cleaned full text..

    print('\n--> Now running the spacy preprocessing on the full_text...')
    df['full_text_tokens'] = spacy_clean_texts(df['cleaned_full_text'].tolist(), nlp)

    cleaned_data_path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(cleaned_data_path, index = False)
    
    print(f'\n--> Saved the cleaned dataset to the path : {cleaned_data_path}')
    print('\n--> FInally the preprocessing is completed!\n')

if __name__ == '__main__':
    preprocessing_dataset()
