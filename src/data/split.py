import pandas as pd

def train_test_split_by_user(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    ratings = ratings.sort_values(by=["user_id", "timestamp"])

    train_list = []
    test_list = []

    for user_id, group in ratings.groupby("user_id"):
        n_test = max(1, int(len(group) * test_size))

        train = group.iloc[:-n_test]
        test = group.iloc[-n_test:]

        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df