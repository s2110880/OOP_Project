import pandas as pd
import statsmodels.api as sm
import numpy as np

#creating super class for all dataloaders
class DataLoaders():
    #innit method for instance variables
    def __init__(self, X, y):
        self._X = X
        self._y = y
    #method that will be overriden by 3 subclasses
    def data_loading(self, **kwargs):
        raise NotImplementedError
    #method to add constant that will be inherited by subclasses
    def add_constant(self):
        self._X = sm.add_constant(self._X)
    #method to return X

    #@property will allow to call those methods as .X, .Y, .x_transpose (helpful)
    @property
    def X(self): 
        return self._X
    #method to return y
    @property
    def y(self):
        return self._y
    #method to return transposed X
    @property
    def x_transpose(self):
        return np.transpose(self._X)

#class inherites from superclass: DataLoaders
class read_csv(DataLoaders):
    #method overrides from superclass to specifically load csv
    def data_loading(self, **kwargs):
        path = kwargs.get("path")
        X = kwargs.get("X")
        y = kwargs.get("y")
        df = pd.read_csv(path)
        self._X = df[X]
        self._y = df[y]

#class inherites from superclass: DataLoaders
class read_statsmodels(DataLoaders):
    #method overrides from superclass to specifically load statsmodels datasets
    #this method also tries to implement two types of statsmodels loaders (e.g duncan and spector)
    def data_loading(self, **kwargs):
        path_1 = kwargs.get("path_1")
        path_2 = kwargs.get("path_2")
        name = kwargs.get("name")
        X = kwargs.get("X")
        y = kwargs.get("y")
        try:
            df = sm.datasets.get_rdataset(path_1, path_2)
            df = pd.DataFrame(data = df.data)
        except:
            print("Error downloading")
            try:
                df = getattr(sm.datasets, name).load_pandas()
                df = pd.DataFrame(data = df.data)
            except:
                print("Error loading")

        self._X = df[X]
        self._y = df[y]

#class inherites from superclass: DataLoaders
class readCSV_fromWEB(DataLoaders):
    #method overrides from superclass to specifically load csv from web
    def data_loading(self, **kwargs):
        url = kwargs.get("url")
        X = kwargs.get("X")
        y = kwargs.get("y")
        df = pd.read_csv(url)
        self._X = df[X]
        self._y = df[y]

