import pandas as pd
from sklearn.model_selection import train_test_split

class Predictor():

    def __init__(self, file, dummies=True):
        self.load_csv(file)
        self.dummies = dummies
        if self.dummies:
            self.df_no_dummies = self.df
            self.df = pd.get_dummies(self.df)

    def load_csv(self, file:str):
        """Load the input csv into a pandas dataframe (self.df)"""
        self.df = pd.read_csv(f"{file}", index_col=0)
        self.df_dummies = pd.get_dummies(self.df)

    def split_data(self, df):
        # self.y = self.df_dummies[y_col].to_list()
        # self.X = self.df_dummies.loc[:, self.df_dummies.columns != y_col]
        columns = self.df_dummies.columns

