import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process
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
titles = moviesA['title'].str.lower()  # Convert titles to lowercase
indices = pd.Series(moviesA.index, index=titles)

# Function to get movie recommendations with descriptions
def get_recommendations_with_desc(title):
    title_lower = title.lower()  # Convert input title to lowercase
    match, score = process.extractOne(title_lower, indices.keys())
    idx_A = indices[match]

    sim_scores = list(enumerate(cosine_sim[:, idx_A]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    recommended_movies = moviesB.iloc[movie_indices][['title', 'desc']]
    return recommended_movies

# Streamlit app
st.title('Movie Recommender: Foreign Movies to Filipino Movies')

# User input for movie title
user_input = st.text_input("Enter a movie title:", '')

if st.button('Get Recommendations'):
    recommended_movies = get_recommendations_with_desc(user_input)
    
    # Display input movie if found
    input_movie = match
    if not input_movie.empty:
        st.write("Input Movie:")
        st.write(f"Title: {input_movie['title'].values[0]}")
        st.write(f"Description: {input_movie['desc'].values[0]}")
        st.write("-" * 50)
    
    st.write("Recommended Movies with Descriptions:")
    for index, row in recommended_movies.iterrows():
        st.write(f"Title: {row['title']}")
        st.write(f"Description: {row['desc']}")
        st.write("-" * 50)
