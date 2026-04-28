from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "ml-1m" 

def load_ratings() -> pd.DataFrame:
    ratings = pd.read_csv(
        RAW_DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

    return ratings

def load_movies() -> pd.DataFrame:
    movies = pd.read_csv(
        RAW_DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    movies["genres"] = movies["genres"].str.split("|")

    return movies

def load_users() -> pd.DataFrame:
    users = pd.read_csv(
        RAW_DATA_DIR / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    return users

def load_all_data():
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    return ratings, movies, users