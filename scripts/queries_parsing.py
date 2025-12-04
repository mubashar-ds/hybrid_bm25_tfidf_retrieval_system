import re
from .correct_spelling import correcting_query_terms

# basic cleaning like lowercasing and clean spacing..

def basic_clean(q: str) -> str:

    if not isinstance(q, str):
        return ''
    q = q.lower().strip()
    q = re.sub(r'\s+', ' ', q)
    return q

def extract_phrases(q: str):
   
    phrases = re.findall(r'"([^"]+)"', q)
    return phrases

# removign the quoted phrases from main query text..

def remove_phrases(q: str, phrases):

    for p in phrases:
        q = q.replace(f'"{p}"', '')
    return q.strip()

def parse_category(q: str):
   
    match = re.search(r'category:(sports|business)', q)
    if match:
        return match.group(1)
    return None

def parse_date_filter(q: str):
    
    # exactting the year filter...

    year_match = re.search(r'date : (\d{4})', q)
    if year_match:
        return {"type": "year", "value": year_match.group(1)}

    # range filter..

    range_match = re.search(r'date : \[(\d{4}-\d{2})\s+to\s+(\d{4}-\d{2})\]', q)
    if range_match:
        start, end = range_match.groups()
        return {'type': 'range', 'start' : start, 'end' : end}

    return None

# boolean operator parsing...

def parse_boolean_tokens(q: str):
        
    tokens = q.split()

    NOT_terms = []
    OR_terms = []
    AND_terms = []

    simple_terms = []

    mode = 'SIMPLE'  

    for t in tokens:
        if t == 'and':
            mode = 'AND'
            continue
        elif t == 'not':
            mode = 'NOT'
            continue
        elif t == 'or':
            mode = 'OR'
            continue
    
        # assigning the term to active mode..

        if mode == 'AND':
            AND_terms.append(t)
        elif mode == 'NOT':
            NOT_terms.append(t)
        elif mode == 'OR':
            OR_terms.append(t)
        else:
            simple_terms.append(t)
        
        # to reset mode after using once..

        # mode = 'SIMPLE'  

    return {
        'NOT' : NOT_terms,
        'OR' : OR_terms,
        'AND' : AND_terms,
        'SIMPLE' : simple_terms
    }

# main parse function..

def parse_query(query: str):
    
    q = basic_clean(query)

    # extracting the metadata filters before removing phrases...

    category = parse_category(q)
    date_filter = parse_date_filter(q)

    # phrases..

    phrases = extract_phrases(q)
    q_no_phrases = remove_phrases(q, phrases)

    # removing the category/date parts from the token space..

    q_no_special = re.sub(r'category:(sports|business)', ' ', q_no_phrases)
    q_no_special = re.sub(r'date:\d{4}', ' ', q_no_special)
    q_no_special = re.sub(r'date:\[[^\]]+\]', ' ', q_no_special)
    
    q_no_special = basic_clean(q_no_special)

    # single terms with AND / OR / NOT ...
    bool_parts = parse_boolean_tokens(q_no_special)

    # merge simple terms + AND + OR into one list for bm25...

    all_positive = bool_parts['SIMPLE'] + bool_parts['AND'] + bool_parts['OR']

    all_positive_corrected = correcting_query_terms(all_positive)

    parsed = {
        'phrases' : phrases,
        'tokens' : all_positive_corrected,
        'NOT' : bool_parts['NOT'],
        'OR' : bool_parts['OR'],
        'AND' : bool_parts['AND'],
        'category' : category,
        'date_filter' : date_filter
    }
    return parsed


# cli testing..

if __name__ == '__main__':
    
    while True:
        q = input('\n--> Enter query (or exit) : ')
        if q == 'exit':
            break
        print(parse_query(q))
