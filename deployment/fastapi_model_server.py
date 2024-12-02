from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

def data_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning and feature engineering pipeline.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be cleaned and prepared.

    Returns:
        pd.DataFrame: DataFrame with various appropriate transformations applied.
    """
    df_prepared = df.copy()
    spending_columns = ["RoomService", "FoodCourt", "Spa", "VRDeck"]
    df_prepared.loc[
        df_prepared["CryoSleep"] == True, spending_columns
    ] = df_prepared.loc[df_prepared["CryoSleep"] == True, spending_columns].fillna(0)

    df_prepared["PassengerGroup"] = df_prepared["PassengerId"].str[:4]
    df_prepared["HomePlanet"] = (
        df_prepared.groupby("PassengerGroup")["HomePlanet"]
        .apply(lambda x: x.ffill().bfill().infer_objects())
        .reset_index(drop=True)
    )
    df_prepared.drop("PassengerGroup", axis=1, inplace=True)

    df_prepared.loc[df_prepared["Age"] <= 13, spending_columns] = df_prepared.loc[
        df_prepared["Age"] <= 13, spending_columns
    ].fillna(0)

    df_prepared["GroupId"] = df_prepared["PassengerId"].str[:4]
    df_prepared["GroupSize"] = (
        df_prepared.groupby("GroupId")["GroupId"].transform("count").astype("float64")
    )

    df_prepared["LuxurySpending"] = df_prepared[spending_columns].sum(axis=1)
    group_spending = df_prepared.groupby("GroupId")[spending_columns].mean()
    group_spending.columns = [f"Avg{col}" for col in spending_columns]
    df_prepared = df_prepared.merge(group_spending, on="GroupId", how="left")

    deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
    df_prepared["Deck"] = df_prepared["Cabin"].str[:1].map(deck_mapping)
    df_prepared["CabinNumber"] = df_prepared["Cabin"].str[2:-2].apply(pd.to_numeric)
    df_prepared["Side"] = df_prepared["Cabin"].str[-1:]

    df_prepared.drop(
        columns=["GroupId", "PassengerId", "Cabin"], inplace=True
    )

    return df_prepared


with open("trained_model.pkl", "rb") as f:
    model = joblib.load(f)

app = FastAPI()


class InputData(BaseModel):
    PassengerId: str
    Cabin: str
    HomePlanet: str
    CryoSleep: str
    Destination: str
    Age: float
    RoomService: float
    FoodCourt: float
    ShoppingMall: float
    Spa: float
    VRDeck: float


@app.post("/predict/")
def predict(data: InputData):
    input_data = pd.DataFrame([data.dict()])
    prepared_data = data_preparation(input_data)
    prediction = model.predict(prepared_data)
    return {"prediction": int(prediction[0])}