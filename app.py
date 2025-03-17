import sqlite3
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from bs4 import BeautifulSoup
from rapidfuzz import process
from flask_caching import Cache
import threading
from datetime import datetime

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

def load_data_from_db(db_name, table_name):
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]
        base_columns = ["name", "genres", "rating", "premiered", "image_url", "summary"]
        if "crew_and_cast" in columns:
            base_columns.append("crew_and_cast")
        query = f"SELECT {', '.join(base_columns)} FROM {table_name}"
        return pd.read_sql(query, conn)

db_name = "tv_shows.db"
table_name = "tv_shows_from_2000"
df = load_data_from_db(db_name, table_name)

if df.empty:
    raise ValueError("Dataset is empty!")

df['summary'] = df['summary'].fillna('').apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
df[['genres', 'image_url', 'name']] = df[['genres', 'image_url', 'name']].fillna('')
df['name'] = df['name'].str.lower()

# Convert 'premiered' to datetime and extract year
df['premiered'] = pd.to_datetime(df['premiered'], errors='coerce')
df['premiered_year'] = df['premiered'].dt.year

# Filter out shows with missing image URLs
df_with_images = df[df['image_url'] != '']

# Sort by premiered date in descending order and get the latest 30 shows with images
latest_shows = df_with_images.sort_values(by='premiered', ascending=False).head(44)

# Text features for recommendation (rest of your code remains the same)
text_features = df['summary'] + " " + df['genres'] + " " + df.get('crew_and_cast', '')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(text_features)

svd = TruncatedSVD(n_components=100, random_state=42)
reduced_matrix = svd.fit_transform(tfidf_matrix)

knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(reduced_matrix)

show_index_map = {name: i for i, name in enumerate(df['name'])}
precomputed_recommendations = {}

def precompute_recommendations():
    for show in df['name']:
        precomputed_recommendations[show] = hybrid_recommendation(show)
    print("Precomputed recommendations cached.")

@cache.memoize(300)
def hybrid_recommendation(show_name, top_n=10):
    show_name = show_name.lower().strip()
    match = process.extractOne(show_name, df['name'])
    if not match or match[1] < 50:
        return f"Show '{show_name}' not found."
    
    show_index = show_index_map.get(match[0])
    distances, indices = knn.kneighbors([reduced_matrix[show_index]], n_neighbors=top_n + 1)

    recommendations = df.iloc[indices[0][1:]].copy()
    recommendations['similarity_score'] = 1 - distances[0][1:]
    recommendations['final_score'] = recommendations['similarity_score']

    return recommendations.nlargest(top_n, 'final_score')[['name', 'genres', 'rating', 'premiered', 'image_url', 'summary']].to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations, error_message = [], ""
    searched_show_image = None  # Initialize searched_show_image

    if request.method == "POST":
        show_name = request.form["show_name"]
        recommendations = precomputed_recommendations.get(show_name.lower().strip(), hybrid_recommendation(show_name))
        if isinstance(recommendations, str):
            error_message, recommendations = recommendations, []
        else:
            # Get the image URL of the searched show
            match = process.extractOne(show_name.lower().strip(), df['name'])
            if match:
                searched_show = df[df['name'] == match[0]].iloc[0]
                searched_show_image = searched_show['image_url']

    return render_template("index.html", recommendations=recommendations, error_message=error_message, latest_shows=latest_shows.to_dict(orient="records"), searched_show_image=searched_show_image)

if __name__ == "__main__":
    threading.Thread(target=precompute_recommendations, daemon=True).start()
    app.run(debug=True)