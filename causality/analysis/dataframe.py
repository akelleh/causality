import pandas as pd
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg

class CausalDataFrame(pd.DataFrame):
    def zplot(self, *args, **kwargs):
        if kwargs.get('z', {}):
            if kwargs.get('kind') == 'line':
                treatment = kwargs.get('x')
                outcome = kwargs.get('y')
                variable_types = kwargs.get('z', {})
                variable_types[treatment] = 'c'
                confounders = kwargs.get('z', {}).keys()

                if kwargs.get('model'):
                    model = kwargs.get('model')()
                    model.fit(self[[treatment] + confounders], self[outcome])
                else:
                    model = KernelModelWrapper()
                    model.fit(self[[treatment] + confounders], self[outcome], variable_types=variable_types)

                xs = []
                ys = []
                xmin, xmax = kwargs.get('xlim', (self[treatment].quantile(0.01), self[treatment].quantile(0.99)))
                for xi in np.arange(xmin, xmax, (xmax - xmin) / 100.):
                    df = self.copy()
                    df[treatment] = xi
                    df['$E[Y|X=x,Z]$'] = model.predict(df[[treatment] + confounders])
                    yi = df.mean()['$E[Y|X=x,Z]$']
                    xs.append(xi)
                    ys.append(yi)
                del kwargs['z']
                df = pd.DataFrame({treatment: xs, outcome: ys})
                return df.plot(*args, **kwargs)

        else:
            self.plot(*args, **kwargs)

class KernelModelWrapper(object):
    def __init__(self):
        self.model = None
        self.variable_types = {}
        self.X_shape = None
        self.y_shape = None

    def fit(self, X, y, variable_types={}):
        self.X_shape = X.shape
        self.y_shape = y.shape
        if variable_types:
            variable_type_string = ''.join([variable_types[col] for col in X.columns])
            self.model = KernelReg(y, X, variable_type_string, reg_type='ll')
        else:
            self.model = KernelReg(y, X, 'c' * X.shape[1], reg_type='ll')
        return self

    def predict(self, X):
        if X.shape != self.X_shape:
            raise Exception("Expected shape {}, received {}".format(self.X_shape, X.shape))
        return self.model.fit(X)[0]
