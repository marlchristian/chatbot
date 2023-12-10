import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings

# Suppressing warnings for cleaner output
warnings.simplefilter('ignore')

# Load data
moviesA = pd.read_csv('https://raw.githubusercontent.com/Koersken/recommender/main/imdb_top_1000.csv')
moviesB = pd.read_csv('https://raw.githubusercontent.com/Koersken/recommender/main/Filipino_Movies_Letterbox_Koersken.csv')

# Data preprocessing
moviesA = moviesA.rename(columns={'Series_Title': 'title', 'Overview': 'desc'})
moviesA['desc'] = moviesA['desc'] + moviesA['Genre']
moviesA = moviesA.drop(columns=['Poster_Link', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'])

moviesB = moviesB.drop_duplicates(subset='title')

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(moviesB['desc'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

moviesA = moviesA.reset_index()
titles = moviesA['title']
indices = pd.Series(moviesA.index, index=moviesA['title'])

# Function to get movie recommendations
def get_recommendations(title):
    if title in indices:
        idx_A = indices[title]

        sim_scores = list(enumerate(cosine_sim[:, idx_A]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]

        return moviesB.iloc[movie_indices]['title']
    else:
        return "Movie not found in the database."

# Streamlit app
st.title('Movie Recommender')

# User input for movie title
user_input = st.text_input("Enter a movie title:", '')

if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_input)
    st.write("Recommended Movies:")
    st.write(recommendations)
