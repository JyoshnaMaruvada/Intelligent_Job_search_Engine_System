import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -----------------------------
# LOAD DATA
# -----------------------------
# df = pd.read_csv("/Users/jyoshnamaruvada/Desktop/MyProjects/intelligent_job_search_system/processed_jobs.csv")
df = pd.read_csv("processed_jobs.csv")


df['combined_text'] = df['clean_title'] + " " + df['clean_desc']

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# -----------------------------
# BUILD VOCAB
# -----------------------------
all_text = " ".join(df['combined_text'])
words = all_text.split()
word_freq = Counter(words)

# -----------------------------
# EDIT DISTANCE
# -----------------------------
def edit_distance(w1, w2):
    dp = [[0]*(len(w2)+1) for _ in range(len(w1)+1)]

    for i in range(len(w1)+1):
        for j in range(len(w2)+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif w1[i-1] == w2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i][j-1],
                    dp[i-1][j],
                    dp[i-1][j-1]
                )
    return dp[len(w1)][len(w2)]

# -----------------------------
# SPELL CORRECTION
# -----------------------------
def correct_word(word):
    word = word.lower()

    known_terms = ["dtr", "ai", "ml", "nlp", "etl"]
    if word in known_terms:
        return word

    if word_freq.get(word, 0) > 3:
        return word

    min_dist = float('inf')
    best_word = word

    for vocab_word in word_freq:
        dist = edit_distance(word, vocab_word)

        if dist < min_dist or (dist == min_dist and word_freq[vocab_word] > word_freq.get(best_word, 0)):
            min_dist = dist
            best_word = vocab_word

    return best_word

def correct_query(query):
    return " ".join([correct_word(w) for w in query.split()])

# -----------------------------
# AUTO COMPLETE
# -----------------------------
title_freq = Counter(df['clean_title'])

def autocomplete(prefix):
    matches = [t for t in title_freq if prefix in t]
    matches = sorted(matches, key=lambda x: title_freq[x], reverse=True)
    return matches[:5]

# -----------------------------
# SEARCH
# -----------------------------
def search_jobs(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-5:][::-1]
    return df.iloc[top_indices][['company', 'jobtitle', 'location']]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🔍 Intelligent Job Search System")

query = st.text_input("Enter job search:")

if query:
    corrected = correct_query(query)
    st.write("✅ Corrected Query:", corrected)

    st.write("💡 Suggestions:")
    suggestions = autocomplete(query)
    for s in suggestions:
        st.write("-", s)

    st.write("📊 Top Jobs:")
    results = search_jobs(corrected)
    st.dataframe(results)