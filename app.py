from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Load datasets
try:
    movies_df = pd.read_csv('movies.csv', encoding='latin1')
    ratings_df = pd.read_csv('ratings.csv')
    print("Data loaded successfully!")
    print(f"Movies shape: {movies_df.shape}, Ratings shape: {ratings_df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    movies_df = pd.DataFrame()
    ratings_df = pd.DataFrame()

# Preprocessing
if not movies_df.empty:
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
else:
    tfidf = None
    cosine_sim = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if movies_df.empty or ratings_df.empty:
        return jsonify({'error': 'Dataset not loaded properly'})
    
    try:
        user_id = int(request.form['user_id'])
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return jsonify({'error': f'No ratings found for user {user_id}'})
        
        user_movies = movies_df[movies_df['movieId'].isin(user_ratings['movieId'])]
        user_genres = ' '.join(user_movies['genres'].tolist())
        user_profile = tfidf.transform([user_genres])
        user_sim = cosine_similarity(user_profile, tfidf_matrix)
        
        sim_scores = list(enumerate(user_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        user_rated_movie_ids = user_ratings['movieId'].tolist()
        recommendations = []
        
        for idx, score in sim_scores:
            movie_id = movies_df.iloc[idx]['movieId']
            if movie_id not in user_rated_movie_ids:
                recommendations.append(movies_df.iloc[idx])
                if len(recommendations) >= 10:
                    break
        
        recommendations_data = [{
            'title': movie['title'],
            'genres': movie['genres'].replace(' ', ', '),
            'movieId': int(movie['movieId']),
            'posterUrl': f"https://image.tmdb.org/t/p/w500{movie['poster_url']}" if pd.notna(movie['poster_url']) else 'https://via.placeholder.com/300x450?text=No+Poster'
        } for movie in recommendations]
        
        return jsonify({'recommendations': recommendations_data})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Running on port 5001