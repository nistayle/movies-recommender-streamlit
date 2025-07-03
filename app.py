import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return df, tfidf

df, tfidf = load_data()

# Hitung TF-IDF matrix & cosine similarity (kita hitung sekali aja)
@st.cache_data
def compute_cosine_sim(df, tfidf):
    tfidf_matrix = tfidf.transform(df['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_cosine_sim(df, tfidf)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def hybrid_rekomendasi(judul, jumlah=5):
    idx = indices.get(judul)
    if idx is None:
        return None

    sinopsis_scores = list(enumerate(cosine_sim[idx]))
    genre_film = df.loc[idx, 'genre']
    genre_scores = [(i, 1.0 if df.loc[i, 'genre'] == genre_film else 0.0) for i in range(len(df))]

    combined_scores = [(i, sinopsis_scores[i][1] + genre_scores[i][1]) for i in range(len(df))]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    combined_scores = combined_scores[1:jumlah+1]
    film_indices = [i[0] for i in combined_scores]

    return df[['title', 'genre', 'rating']].iloc[film_indices]

st.title('Movie Recommendation System')

option = st.selectbox('Pilih Film:', df['title'].tolist())

if st.button('Cari Rekomendasi'):
    rekom = hybrid_rekomendasi(option)
    if rekom is None:
        st.write('Film tidak ditemukan.')
    else:
        st.write('Rekomendasi film:')
        st.dataframe(rekom)