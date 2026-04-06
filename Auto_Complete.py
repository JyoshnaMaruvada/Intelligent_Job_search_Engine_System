# Auto_Complete.py

import pandas as pd
from collections import Counter
from rapidfuzz import process, fuzz

# =========================
# LOAD DATA
# =========================
# Make sure your dataset has 'clean_title' column
df = pd.read_csv("your_dataset.csv")

# Clean titles
titles = df['clean_title'].dropna().str.lower().tolist()

# Count frequency
title_freq = Counter(titles)

# =========================
# AUTOCOMPLETE FUNCTION
# =========================
def autocomplete(query, top_n=5):
    query = query.lower()

    # Get best matches using fuzzy matching
    matches = process.extract(
        query,
        title_freq.keys(),
        scorer=fuzz.token_sort_ratio,
        limit=top_n
    )

    # Extract only titles
    suggestions = [match[0] for match in matches]

    return suggestions


# =========================
# CORRECT QUERY FUNCTION
# =========================
def correct_query(query):
    query = query.lower()

    best_match = process.extractOne(
        query,
        title_freq.keys(),
        scorer=fuzz.token_sort_ratio
    )

    if best_match:
        return best_match[0]
    return query


# =========================
# TESTING
# =========================
if __name__ == "__main__":

    queries = ["dta anlyts", "soft eng", "data scntist", "machin learn"]

    for q in queries:
        print("\nUser Query:", q)

        corrected = correct_query(q)
        print("Corrected Query:", corrected)

        suggestions = autocomplete(q)
        print("Suggestions:")
        for s in suggestions:
            print("-", s)