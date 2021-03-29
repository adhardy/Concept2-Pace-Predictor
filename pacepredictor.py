import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import numpy as np

class LinearRegressionIteration():
    def __init__(self, linear_regression, cols_x, col_y):       
        
        self.df_train = linear_regression.df_train.copy() #make sure we have copies as the model will alter the contents
        self.df_test = linear_regression.df_test.copy()
        self.df_val = linear_regression.df_val.copy()

        self.col_y = col_y
        self.cols_x = cols_x

        self.order = linear_regression.order
        
        self.formula = self.get_formula()

        self.fit()
        self.predict()

    def get_formula(self):
        formula = f"{self.col_y} ~ "
        for col_x in self.cols_x:
            formula += f"{col_x} + "
        return formula[:-3]

    def fit(self):
        self.model = smf.ols(formula=self.formula, data=self.df_train).fit()

    def predict(self):
        self.col_pred = f"{self.col_y}_pred"
        self.df_test[self.col_pred] = self.model.predict(self.df_test[self.cols_x])
        self.df_test["squared_error"] = squared_error(self.df_test, self.col_y, self.col_pred)
        self.df_test["root_squared_error"] = np.sqrt(self.df_test["squared_error"])
        self.mse = self.df_test["squared_error"].mean()
        self.rmse = self.df_test["root_squared_error"].mean()

    def stats(self):
        print(f"Mean squared error: {round(self.mse,1)}")
        print(f"Root mean squared error: {round(self.rmse,1)}")

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
        self.p_max = None

    def iterate(self):
        cols_x = self.cols_x
        col_y = self.col_y

        #get our intial model

        self.iterations.append(LinearRegressionIteration(self, cols_x, col_y))
        p_values = self.iterations[-1].model.pvalues

        #loop until p_max > alpha
        while p_values.max() > self.alpha:
            p_max_idx = p_values.idxmax()

            cols_x.remove(p_max_idx)
            self.iterations.append(LinearRegressionIteration(self, cols_x, col_y))
            p_values = self.iterations[-1].model.pvalues
    
        self.summarise()

    def summarise(self):
        summary_columns = ["mse", "rmse", "cols_x", "coefficients", "p_vals"]
        self.summary = pd.DataFrame(columns=summary_columns)
        for iteration in self.iterations:
            row = [iteration.mse, iteration.rmse, iteration.cols_x, iteration.model.params.to_list(), iteration.model.pvalues.to_list()]
            self.summary.loc[len(self.summary)] = row

class Predictor():

    def __init__(self, file, random_state=None):
        self.load_csv(file)
        self.random_state = random_state
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
    return df.loc[:,df.columns != col_y].columns.to_list()



if __name__ == "__main__":
    file_path = "./"
    file_5_2_10 = "5_2_10.csv"

    pp_5_2_10 = Predictor(f"{file_path}{file_5_2_10}", random_state=42)

    pp_5_2_10.df = pp_5_2_10.df.drop(["adaptive_rowing_category","weight_class", "profile_id"], axis=1)

    pp_5_2_10.split_data()

    col_y = "time_2000"
    cols_x = get_cols_x(pp_5_2_10.df, col_y)

    pp_5_2_10.add_model("Linear", col_y, cols_x, 1, "First order model, using all vars", alpha=0.005)
    pp_5_2_10.models["Linear"].iterate()

    pp_5_2_10.models["Linear"].summarise()
