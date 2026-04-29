import pandas as pd

def build_genre_matrix(movies: pd.DataFrame):
    all_genres = sorted({g for genres in movies["genres"] for g in genres})

    genre_to_index = {genre: i for i, genre in enumerate(all_genres)}

    genre_matrix = []

    for genres in movies["genres"]:
        vector = [0] * len(all_genres)

        for g in genres:
            vector[genre_to_index[g]] = 1
        genre_matrix.append(vector)

    genre_df = pd.DataFrame(genre_matrix, columns=all_genres)
    genre_df["movie_id"] = movies["movie_id"].values

    return genre_df