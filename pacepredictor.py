import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import numpy as np
import pprint
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt
import statsmodels.api as sm

class LinearRegressionIteration():
    def __init__(self, linear_regression, cols_x, col_y, intercept=True):       
        
        self.df_train = linear_regression.df_train.copy() #make sure we have copies as the model will alter the contents
        self.df_val = linear_regression.df_val.copy()

        self.col_y = col_y
        self.cols_x = cols_x.copy()
        self.intercept = intercept
        self.order = linear_regression.order
        
        self.formula = self.get_formula()

        self.fit()
        self.predict()

    def get_formula(self):
        formula = f"{self.col_y} ~ "

        for col_x in self.cols_x:
            formula += f"{col_x} + "
        formula = formula[:-3]

        if self.intercept == False:
            formula = f"{formula} - 1" 

        return formula

    def fit(self):
        self.model = smf.ols(formula=self.formula, data=self.df_train).fit()

    def predict(self):
        self.col_pred = f"{self.col_y}_pred"
        self.df_val[self.col_pred] = self.model.predict(self.df_val[self.cols_x])
        self.df_val["squared_error"] = squared_error(self.df_val, self.col_y, self.col_pred)
        self.df_val["root_squared_error"] = np.sqrt(self.df_val["squared_error"])
        self.mse = self.df_val["squared_error"].mean()
        self.rmse = self.df_val["root_squared_error"].mean()

    def stats(self):
        print(f"Mean squared error: {round(self.mse,1)}")
        print(f"Root mean squared error: {round(self.rmse,1)}")

class AutoLinearRegression():

    def __init__(self, df_train, df_val, col_y:str, cols_x:list, order:int, description:str=None, alpha:float=0.05, intercept:bool=True):       
        
        self.df_train = df_train.copy() #make sure we have copies as the model will alter the contents
        self.df_test = df_test.copy()
        self.df_val = df_val.copy()
        self.order = order
        self.alpha = alpha
        self.col_y = col_y
        self.cols_x = cols_x
        self.descrption = description
        self.iterations = []
        self.p_max = None
        self.get_polynomial_data()
        self.intercept = intercept

    def get_polynomial_data(self):
        cols_x_new = []
        for o in range(2,self.order+1):
            for col_x in self.cols_x:
                col_x_new = f"{col_x}_{o}"
                self.df_train[col_x_new] = self.df_train[col_x]**o
                self.df_test[col_x_new] = self.df_train[col_x]**o
                self.df_val[col_x_new] = self.df_train[col_x]**o
                cols_x_new.append(col_x_new)
        self.cols_x += cols_x_new

    def iterate(self):
        cols_x = self.cols_x
        col_y = self.col_y

        #get our intial model

        self.iterations.append(LinearRegressionIteration(self, cols_x, col_y, intercept=self.intercept))
        p_values = self.iterations[-1].model.pvalues

        #loop until p_max > alpha
        while p_values.max() > self.alpha and len(cols_x) > 0: #stop if remove all the columns 
            p_max_idx = p_values.idxmax()

            if p_max_idx == "Intercept":
                self.intercept = False
            else:
                cols_x.remove(p_max_idx)
            self.iterations.append(LinearRegressionIteration(self, cols_x, col_y, intercept=self.intercept))
            p_values = self.iterations[-1].model.pvalues
    
        self.summarise()

    def summarise(self):
        summary_columns = ["mse", "rmse", "p_max_param", "p_max", "intercept", "parameters","coefficients", "p_vals"]
        self.summary = pd.DataFrame(columns=summary_columns)
        for iteration in self.iterations:
            row = [iteration.mse, iteration.rmse, iteration.model.pvalues.idxmax(), iteration.model.pvalues.max(),iteration.intercept, iteration.cols_x, iteration.model.params.to_list(), iteration.model.pvalues.to_list()]
            self.summary.loc[len(self.summary)] = row

class LinearRegression():
    def __init__(self, df_train, df_val, target:str, parameters:list, order:int, description:str=None, intercept:bool=True, drop=[]):
        self.df_train = df_train.copy()
        self.df_val = df_val.copy()

        self.order = order
        self.target = target
        self.parameters = parameters
        self.description = description
        self.intercept = intercept
        self.drop = drop

        self.get_polynomial_data()
        self.drop_columns()
        self.get_formula()
        self.fit()
        self.predict()

    def drop_columns(self):    
        #drop any non-polynomial columns
        for column in self.drop:
            if column in self.df_train.columns:
                self.df_train.drop(column, axis=1, inplace=True)
                self.df_val.drop(column, axis=1, inplace=True)
                self.parameters.remove(column)
    
    def get_polynomial_data(self):
        parameters_new = []
        for o in range(2,self.order+1):
            for parameter in self.parameters:
                if self.df_train[parameter].dtype in (float, int):
                    parameter_new = f"{parameter}_{o}"
                    #don't create polynomial columns we don't want
                    if parameter_new in self.drop:
                        pass
                    else:
                        self.df_train[parameter_new] = self.df_train[parameter]**o
                        self.df_val[parameter_new] = self.df_val[parameter]**o
                        parameters_new.append(parameter_new)
        self.parameters += parameters_new
    
    def get_formula(self):
        self.formula = f"{self.target} ~ {' + '.join(self.parameters)}"
        if self.intercept == False:
            self.formula += "-1"

    def fit(self):
        self.model = smf.ols(formula=self.formula, data=self.df_train).fit()
        self.df_train["residuals"] = self.model.resid

    def predict(self):
        self.df_val[f"{self.target}_pred"] = self.model.predict(self.df_val)
        self.df_val["squared_error"] = (self.df_val[self.target] - self.df_val[f"{self.target}_pred"])**2
        self.mse = self.df_val["squared_error"].mean()
        
    def anova(self):
        return sm.stats.anova_lm(self.model, typ=2)


    # Creating a general plotting function for plotting a scatter plot and line on the same figure
    def plot_residuals(self):

        line_y = [0] * len(self.df_train[self.target])

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df_train[self.target], y=self.df_train["residuals"], name="Residuals", mode="markers"))
        fig.add_trace(go.Scatter(
            x=self.df_train[self.target], y=line_y, name="y=0"))
        fig.update_layout(title="Plot of Model Residuals", xaxis_title=self.target,
            yaxis_title="Residuals")
        
        return fig

    def plot_QQ(self):
        fig = sm.qqplot(self.model.resid, fit=True, line='45')
        plt.title("QQ Plot of residuals")
        plt.show()

    def p_max(self):
        print(f"Max p: {self.model.pvalues.idxmax()} p={round(self.model.pvalues.max(), 3)}")

class Predictor():

    def __init__(self, file, random_state=None):
        self.load_csv(file)
        self.random_state = random_state
        #replace all space in column names with underscores for patsy
        self.df = self.df.rename(columns=lambda x: x.replace(" ", "_"))
        self.models = {}
        self.auto_models = {}

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

    def plot_scatters(self, target, parameters, cols=2, height=4000, width=2000):

        rows = math.ceil(len(parameters) / cols) 

        subplot_titles = tuple(f"{parameter} vs {target}" for parameter in parameters)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        for i, col_name in enumerate(parameters):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(go.Scattergl(x=self.df[col_name], y=self.df[target], mode="markers"), row=row, col=col)

        fig.update_layout(height=height, width=width, title="Target vs Parameters", showlegend=False)
        #fig.update_yaxes(title_text=target, row = row, col = row)
        return fig

    def plot_distributions(self, cols = 4, height=4000, width=2000):

        rows = math.ceil(len(self.df.columns) / cols) 

        subplot_titles = tuple(parameter for parameter in self.df.columns)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        for i, col_name in enumerate(self.df.columns):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(go.Histogram(x=self.df[col_name]), row=row, col=col)
            #fig.update_xaxes(title_text=col_name, row = row, col = row)
            #fig.update_yaxes(title_text="Count", row = row, col = row)
        fig.update_layout(height=height, width=width, title="Distribution", showlegend=False)
        return fig

    def plot_violins(self, target, parameters,cols=4, height=4000, width=2000):

        rows = math.ceil(len(parameters) / cols) 

        cols = 2
        subplot_titles = tuple(f"{parameter} vs {target}" for parameter in parameters)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        for i, col_name in enumerate(parameters):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(go.Violin(
                x=self.df[col_name], y=self.df[target], name=col_name, box_visible=True
            ), row=row, col=col)
            
            fig.update_xaxes(patch=dict(type='category', categoryorder='mean ascending'), row=row, col=col)
            
        fig.update_layout(height=height, width=width)
        return fig
        
    def add_auto_model(self, model_name:str, col_y:str, cols_x:list, order:int=1, description:str=None, alpha:float=0.05, intercept:bool=True):
        self.auto_models[model_name] = AutoLinearRegression(self.df_train, self.df_val, col_y, cols_x, order, description, alpha, intercept)

    def add_model(self, model_name:str, target:str, parameters:list, order:int=1, description:str=None, intercept:bool=True, drop=[]):
        self.models[model_name] = LinearRegression(self.df_train, self.df_val, target, parameters, order, description, intercept, drop)

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
    model_name = "2nd order 2k"
    pp_5_2_10.add_model(model_name, col_y, cols_x, 2, "2nd order model to predict 2k, using all vars", alpha=0.05)
    pp_5_2_10.models[model_name].iterate()
    print(pp_5_2_10.models[model_name].summary)

def percent_complete(df):
    print(df.notnull().mean()*100)

def squared_error(df, col_a, col_b):
    return (df[col_a] - df[col_b])**2

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