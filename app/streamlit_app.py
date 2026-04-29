from pathlib import Path
import sys 

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.data.split import train_test_split_by_user
from src.data.load_data import load_all_data 

from src.features.content import build_genre_matrix

from src.models.item_collaborative import ItemCollaborativeRecommender
from src.models.cold_start_hybrid import ColdStartHybridRecommender
from src.models.popularity import PopularityRecommender
from src.models.content import ContentRecommender
from src.models.hybrid import HybridRecommender

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Hybrid Movie Recommender",
    layout="wide",
)

st.title("Hybrid Movie Recommendation System")
st.write(
    "A production-style recommendation demo using popularity, content-based, collaborative filtering, hybrid ranking, and cold-start handling."
)

@st.cache_data
def load_data():
    ratings, movies, users = load_all_data()
    train_df, test_df = train_test_split_by_user(ratings, test_size=0.2)
    genre_matrix = build_genre_matrix(movies)

    return ratings, movies, users, train_df, test_df, genre_matrix

@st.cache_resource
def build_models(train_df, movies, genre_matrix):
    popularity = PopularityRecommender(method="bayesian").fit(train_df)
    content = ContentRecommender().fit(movies, genre_matrix)
    collaborative = ItemCollaborativeRecommender(min_rating=4).fit(train_df)

    hybrid = HybridRecommender(
        popularity_model=PopularityRecommender(method="bayesian"),
        content_model=ContentRecommender(),
        collaborative_model=ItemCollaborativeRecommender(min_rating=4),
        weights={
            "popularity": 0.05,
            "content": 0.05,
            "collaborative": 0.90,
        },
    ).fit(train_df, movies, genre_matrix)

    cold_start = ColdStartHybridRecommender(
        hybrid_model=HybridRecommender(
            popularity_model=PopularityRecommender(method="bayesian"),
            content_model=ContentRecommender(),
            collaborative_model=ItemCollaborativeRecommender(min_rating=4),
            weights={
                "popularity": 0.05,
                "content": 0.05,
                "collaborative": 0.90,
            },
        ),
        content_model=ContentRecommender(),
        popularity_model=PopularityRecommender(method="bayesian"),
        min_user_ratings=10
    ).fit(train_df, movies, genre_matrix)

    return {
        "Popularity": popularity,
        "Content-Based": content,
        "Item Collaborative": collaborative,
        "Hybrid": hybrid,
        "Cold-Start Hybrid": cold_start,
    }

ratings, movies, users, train_df, test_df, genre_matrix = load_data()
models = build_models(train_df, movies, genre_matrix)

st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Choose a recommendation model",
    list(models.keys()),
)

recommendation_mode = st.sidebar.radio(
    "Recommendation mode",
    ["Existing user", "New user cold-start"],
)

if recommendation_mode == "Existing user":
    model_name = st.sidebar.selectbox(
        "Choose recommendation model",
        list(models.keys()),
    )
else:
    model_name = "Cold-Start Hybrid"
    st.sidebar.info("Using Cold-Start Hybrid for new-user recommendations.")

if recommendation_mode == "Existing user":
    user_id = st.sidebar.selectbox(
        "Choose user ID",
        sorted(train_df["user_id"].unique()),
    )

    active_ratings = train_df.copy()

else:
    selected_movies = st.sidebar.multiselect(
        "Select movies you like",
        movies["title"].sort_values().tolist(),
    )

    fake_user_id = 999999
    selected_movie_ids = movies[movies["title"].isin(selected_movies)]["movie_id"]

    fake_ratings = pd.DataFrame({
        "user_id": fake_user_id,
        "movie_id": selected_movie_ids,
        "rating": 5,
        "timestamp": pd.Timestamp.now(),
    })

    active_ratings = pd.concat([train_df, fake_ratings], ignore_index=True)
    user_id = fake_user_id

k = st.sidebar.slider("Number of recommendations (k)", min_value=1, max_value=20, value=10)

model = models[model_name]

user_history = (
    active_ratings[active_ratings["user_id"] == user_id]
    .merge(movies, on="movie_id")
    .sort_values("timestamp", ascending=False)
)

if recommendation_mode == "New user cold-start" and len(selected_movies) == 0:
    st.warning("Select at least one movie to generate cold-start recommendations.")
    st.stop()

recs = model.recommend(user_id=user_id, ratings=active_ratings, k=k)
recs = recs.merge(movies, on="movie_id")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Top {k} Recommendations")
    st.dataframe(
        recs[["title", "genres", "score"]],
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.subheader("Recent User History")
    st.dataframe(
        user_history[["title", "genres", "rating"]],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Model Notes")

notes = {
    "Popularity": "Recommends globally popular/highly rated movies. Strong baseline but not personalized.",
    "Content-Based": "Uses movie genres to recommend items similar to the user's highly rated movies.",
    "Item Collaborative": "Uses user-item rating patterns to recommend movies similar users liked.",
    "Hybrid": "Combines popularity, content, and collaborative scores with tuned weights.",
    "Cold-Start Hybrid": "Uses hybrid recommendations for normal users, content fallback for sparse users, and popularity fallback for new users.",
}

st.write(notes[model_name])

st.subheader("Project summary")

st.markdown(
    """
    This recommender system includes:

    - Time-based train/test split by user
    - Popularity baseline
    - Content-based filtering
    - Item-item collaborative filtering
    - Hybrid score blending
    - Cold-start fallback logic
    - Ranking evaluation with Precision@K, Recall@K, and NDCG@K
    """
)