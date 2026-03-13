import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.set_page_config(page_title="TF-IDF Explorer", page_icon="🔎", layout="wide")

st.title("🔎 TF-IDF Document Explorer")

st.write("""
Esta aplicación permite analizar un conjunto de documentos usando **TF-IDF** y encontrar cuál es el más relacionado con una pregunta.

Funciones adicionales de esta versión:
- Ranking completo de documentos
- Gráfico de similitud
- Palabras más importantes del corpus
""")

# -----------------------------
# Barra lateral
# -----------------------------

st.sidebar.header("📘 Instrucciones")

st.sidebar.write("""
1. Escribe varios documentos (uno por línea).
2. Escribe una pregunta.
3. La aplicación calculará TF-IDF y encontrará el documento más relevante.
""")

# -----------------------------
# Ejemplo inicial
# -----------------------------

text_input = st.text_area(
    "📄 Documents (one per line, in English):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.\nBirds sing in the morning.",
    height=150
)

question = st.text_input(
    "❓ Ask a question:",
    "Who is playing?"
)

# -----------------------------
# Stemmer inglés
# -----------------------------

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):

    text = text.lower()

    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = [t for t in text.split() if len(t) > 1]

    stems = [stemmer.stem(t) for t in tokens]

    return stems


# -----------------------------
# BOTÓN
# -----------------------------

if st.button("🔍 Analyze Documents"):

    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:

        st.warning("⚠️ Please enter at least one document.")

    else:

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)

        # -----------------------------
        # MATRIZ TF-IDF
        # -----------------------------

        st.subheader("📊 TF-IDF Matrix")

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.dataframe(df_tfidf.round(3), use_container_width=True)

        # -----------------------------
        # VECTOR DE PREGUNTA
        # -----------------------------

        question_vec = vectorizer.transform([question])

        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        # -----------------------------
        # RESPUESTA
        # -----------------------------

        st.subheader("🎯 Best Match")

        st.success(f"Best document: {best_doc}")

        st.info(f"Similarity score: {best_score:.3f}")

        # -----------------------------
        # RANKING DE DOCUMENTOS
        # -----------------------------

        ranking = pd.DataFrame({
            "Document": documents,
            "Similarity": similarities
        }).sort_values("Similarity", ascending=False)

        st.subheader("📊 Document Ranking")

        st.dataframe(ranking, use_container_width=True)

        # -----------------------------
        # GRÁFICO DE SIMILITUD
        # -----------------------------

        st.subheader("📈 Similarity Chart")

        chart_df = pd.DataFrame({
            "Document": [f"Doc {i+1}" for i in range(len(documents))],
            "Similarity": similarities
        })

        st.bar_chart(chart_df.set_index("Document"))

        # -----------------------------
        # PALABRAS MÁS IMPORTANTES
        # -----------------------------

        st.subheader("⭐ Most Important Words (Corpus TF-IDF)")

        tfidf_scores = X.sum(axis=0).A1

        terms = vectorizer.get_feature_names_out()

        tfidf_global = pd.DataFrame({
            "Word": terms,
            "Importance": tfidf_scores
        }).sort_values("Importance", ascending=False)

        st.bar_chart(tfidf_global.head(10).set_index("Word"))

        # -----------------------------
        # STEMS COINCIDENTES
        # -----------------------------

        vocab = vectorizer.get_feature_names_out()

        q_stems = tokenize_and_stem(question)

        matched = [
            s for s in q_stems
            if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0
        ]

        st.subheader("🔤 Matching stems")

        st.write(matched)




