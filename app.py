import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ✅ IMPORT FROM YOUR FILE
from Auto_Complete import autocomplete, correct_query

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("processed_jobs.csv")

df['combined_text'] = df['clean_title'] + " " + df['clean_desc']

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# =========================
# SEARCH FUNCTION
# =========================
def search_jobs(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-5:][::-1]
    return df.iloc[top_indices][['company', 'jobtitle', 'location']]

# =========================
# STREAMLIT UI
# =========================
st.title("🔍 Intelligent Job Search System")

query = st.text_input("Enter job search:")

if query:
    # ✅ STEP 1: CORRECT SPELLING
    corrected = correct_query(query)
    st.write("✅ Corrected Query:", corrected)

    # ✅ STEP 2: USE CORRECTED QUERY FOR SUGGESTIONS
    st.write("💡 Suggestions:")
    suggestions = autocomplete(corrected)

    if suggestions:
        for s in suggestions:
            st.write("-", s)
    else:
        st.write("No suggestions found")

    # ✅ STEP 3: SEARCH USING CORRECTED QUERY
    st.write("📊 Top Jobs:")
    results = search_jobs(corrected)
    st.dataframe(results)