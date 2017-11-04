import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class CausalDataFrame(pd.DataFrame):
    def zplot(self, *args, **kwargs):
        if kwargs.get('z', []):
            if kwargs.get('kind') == 'line':
                treatment = kwargs.get('x')
                outcome = kwargs.get('y')
                confounders = kwargs.get('z', [])

                model = RandomForestRegressor(n_estimators=1000, n_jobs=8)
                model.fit(self[[treatment] + confounders], self[outcome])

                xs = []
                ys = []
                xmin, xmax = kwargs.get('xlim', (-1,1))
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
