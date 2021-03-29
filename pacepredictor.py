import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import numpy as np

class LinearRegressionIteration():
    def __init__(self, linear_regression):       
        
        self.df_train = linear_regression.df_train.copy() #make sure we have copies as the model will alter the contents
        self.df_test = linear_regression.df_test.copy()
        self.df_val = linear_regression.df_val.copy()

        self.col_y = linear_regression.col_y
        self.cols_x = linear_regression.cols_x

        self.order = linear_regression.order
        self.descrption = linear_regression.descrption
        
        self.formula = self.get_formula()

    def get_formula(self):
        formula = f"{self.col_y} ~ "
        for col_x in self.cols_x:
            formula += f"{col_x} + "
        return formula[:-3]

    def fit(self):
        self.model = smf.ols(formula=self.formula, data=self.df_train).fit()
        print(self.model.summary())

    def predict(self):
        self.col_pred = f"{self.col_y}_pred"
        self.df_test[self.col_pred] = self.model.predict(self.df_test[self.cols_x])
        self.df_test["squared_error"] = squared_error(self.df_test, self.col_y, self.col_pred)
        self.df_test["error"] = np.sqrt(self.df_test["squared_error"])
        self.mse = self.df_test["squared_error"].mean()
        self.mean_error = self.df_test["error"].mean()

    def stats(self):
        print(f"Mean squared error: {round(self.mse,1)}")
        print(f"Mean error: {round(self.mean_error,1)}")

def squared_error(df, col_a, col_b):
    return (df[col_a] - df[col_b])**2

def create_linear_formula(col_y:str, cols_x:list):
    formula = f"{col_y} ~ "
    for col_x in cols_x:
        formula += f"{col_x} + "
    
    return formula[:-3]

class LinearRegression():

    def __init__(self, df_train, df_test, df_val, col_y:str, cols_x:list, order:int, description:str=None, alpha:float=0.05):       
        
        self.df_train = df_train.copy() #make sure we have copies as the model will alter the contents
        self.df_test = df_test.copy()
        self.df_val = df_val.copy()

        self.alpha = alpha

        self.col_y = col_y
        self.cols_x = cols_x

        self.order = order
        self.descrption = description

        self.iterations = []

    def iterate(self):
        #if all p values are below alpha, stop
        self.iterations.append(LinearRegressionIteration(self))

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
        self.models = {}

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

    def add_model(self, model_name:str, col_y:str, cols_x:list, order:int=1, descrption:str=None, alpha:float=0.05):
        self.models[model_name] = LinearRegression(self.df_train, self.df_test, self.df_val, col_y, cols_x, order, descrption, alpha)

# Creating a general plotting function for plotting a scatter plot and line on the same figure
def plot_scatter_and_line(x, scatter_y, line_y, scatter_name, line_name, title, x_title, y_title):

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=scatter_y, name=scatter_name, mode="markers"))
    fig.add_trace(go.Scatter(
        x=x, y=line_y, name=line_name))
    fig.update_layout(title=title, xaxis_title=x_title,
        yaxis_title=y_title)
    
    return fig

def get_cols_x(df, col_y):
    """get a list of all the columns names not matching col_y"""
    return df.loc[:,df.columns != col_y].columns