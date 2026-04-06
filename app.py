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

# Initialize session state
if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

if query:
    # -----------------------------
    # STEP 1: CORRECT QUERY FIRST
    # -----------------------------
    corrected = correct_query(query)
    st.write("✅ Corrected Query:", corrected)

    # -----------------------------
    # STEP 2: SUGGESTIONS BELOW
    # -----------------------------
    suggestions = autocomplete(corrected)

    if suggestions:
        st.write("💡 Suggestions (click one):")

        for s in suggestions:
            if st.button(s, key=s):
                st.session_state.selected_query = s  # store selection

    else:
        st.write("No suggestions found")

    # -----------------------------
    # STEP 3: FINAL QUERY LOGIC
    # -----------------------------
    final_query = (
        st.session_state.selected_query
        if st.session_state.selected_query
        else corrected
    )

    # -----------------------------
    # STEP 4: RESULTS
    # -----------------------------
    st.write("📊 Top Jobs:")
    results = search_jobs(final_query)
    st.dataframe(results)