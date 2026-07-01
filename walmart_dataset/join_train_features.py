import pandas as pd
import os
def create_standard_df(train_df, dataset_path=None):
    """
    Creates unified Walmart dataset by merging:
    - train.csv
    - features.csv
    - stores.csv
    """

    if train_df is None:
        train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))

    features = pd.read_csv(os.path.join(dataset_path, "features.csv"))
    stores = pd.read_csv(os.path.join(dataset_path, "stores.csv"))

    df = train_df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    df["Date"] = pd.to_datetime(df["Date"])

    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"] = df["Date"].dt.year

    return df