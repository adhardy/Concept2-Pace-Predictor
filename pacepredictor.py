import pandas as pd

class Predictor():

    def __init__(self):
        pass

    def load_csvs(self, input_path:str):
        """Load the input csv into a pandas dataframe (self.df)"""
        self.df_athletes = pd.read_csv(f"{input_path}/athletes_clean.csv", index_col=0)
        self.df_workouts = pd.read_csv(f"{input_path}/workouts_clean.csv", index_col=0)

