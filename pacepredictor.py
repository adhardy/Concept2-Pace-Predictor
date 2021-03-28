import pandas as pd

class Predictor():

    def __init__(self, file):
        self.load_csv(file)

    def load_csv(self, file:str):
        """Load the input csv into a pandas dataframe (self.df)"""
        self.df = pd.read_csv(f"{file}", index_col=0)
        self.df_dummies = pd.get_dummies(self.df)

    def split_data(self, y_col)
        self.y = self.df_dummies[y_col].to_list()
        self.X - self,df_dummies
