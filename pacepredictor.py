import pandas as pd
from sklearn.model_selection import train_test_split

class Predictor():

    def __init__(self, file, dummies=True, random_state=None):
        self.load_csv(file)
        self.dummies = dummies
        self.random_state = random_state
        if self.dummies:
            self.df_no_dummies = self.df
            self.df = pd.get_dummies(self.df)
        #replace all space in column names with underscores for patsy
        self.df = self.df.rename(columns=lambda x: x.replace(" ", "_"))

    def load_csv(self, file:str):
        """Load the input csv into a pandas dataframe (self.df)"""
        self.df = pd.read_csv(f"{file}", index_col=0)
        self.df_dummies = pd.get_dummies(self.df)

    def split_data(self):
        columns = self.df.columns
        dtypes = self.df.dtypes.to_dict()
        np_arr = self.df.to_numpy()

        arr_train, arr_test = train_test_split(np_arr, test_size=0.3, random_state=self.random_state)

        self.df_train = pd.DataFrame(arr_train)
        self.df_train.columns = columns
        self.df_train = self.df_train.astype(dtypes)

        arr_test, arr_val = train_test_split(arr_test, test_size=0.5, random_state=self.random_state)

        self.df_test = pd.DataFrame(arr_test)
        self.df_test.columns = columns
        self.df_test = self.df_test.astype(dtypes)

        self.df_val = pd.DataFrame(arr_val)
        self.df_val.columns = columns
        self.df_val = self.df_val.astype(dtypes)

def create_lr_formula(y_col:str, x_cols:list):

    formula = f"{y_col} ~ "
    for x_col in x_cols:
        formula += f"{x_col} + "
    
    return formula[:-3]
