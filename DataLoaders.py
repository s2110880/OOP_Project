import pandas as pd
import statsmodels.api as sm
import numpy as np

#creating super class for all dataloaders
class DataLoaders():
    def __init__(self, X, y):
        self._X = X
        self._y = y
    def data_loading(self, *args, **kwargs):
        raise NotImplementedError
    def add_constant(self):
        self._X = sm.add_constant(self._X)
    
    @property
    def X(self):
        return self._X
    @property
    def y(self):
        return self._y
    @property
    def x_transpose(self):
        return np.transpose(self._X)

class read_csv(DataLoaders):
    def data_loading(self, path, y, X):
        df = pd.read_csv(path)
        self._X = df[X]
        self._y = df[y]

class read_statsmodels(DataLoaders):
    def data_loading(self, path_1, path_2, y, X):
        df = sm.datasets.get_rdataset(path_1, path_2)
        df = pd.DataFrame(data = df.data)
        self._X = df[X]
        self._y = df[y]


# https://treinamentolivre.com/aluno01/wp-content/uploads/2023/02/Python-for-Everyone-PDFDrive-.pdf