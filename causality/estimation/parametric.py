import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM

class DifferenceInDifferences(object):
    def __init__(self, robust=True):
        """
        We will take a dataframe where each row is a user,
        and the columns are:
        (1) Assignment: 1 = test, 0 = control
        (2) Start: The value of the metric you're interested
            in at the start of the experiment.
        (3) End: The value of the metric you're interested in
            at the end of the experiment.
        """
        if robust:
            self.model = RLM
        else:
            self.model = OLS

    def average_treatment_effect(self, X, start='Start', end='End', assignment='Assignment'):
        test = X[X['Assignment']==1][['Start','End']]
        control = X[X['Assignment']==0][['Start','End']]
        del X

        test_initial = test['Start']
        test_final = test['End']
        control_initial = control['Start']
        control_final = control['End']
        del test, control

        df = pd.DataFrame({'y' : test_initial, 
                   'assignment' : [1. for i in test_initial], 
                   't' :[0. for i in test_initial] })
        df = df.append(pd.DataFrame({'y' : test_final, 
                                     'assignment' : [1. for i in test_final], 
                                     't' :[1. for i in test_final] }))

        df = df.append(pd.DataFrame({'y' : control_initial, 
                                     'assignment' : [0. for i in control_initial], 
                                     't' :[0. for i in control_initial] }))

        df = df.append(pd.DataFrame({'y' : control_final, 
                                     'assignment' : [0. for i in control_final], 
                                     't' :[1. for i in control_final] }))
        del test_initial, test_final, control_initial, control_final
        df['did'] = df['t'] * df['assignment'] 
        df['intercept'] = 1.

        model = self.model(df['y'], df[['t', 'assignment','did', 'intercept']])
        result = model.fit()
        conf_int = result.conf_int().ix['did']
        expected = result.params['did']
        return conf_int[0], expected, conf_int[1]
        
    def test_parallel_trend(self, X, start='Start', end='End', assignment='Assignment'):
        """
        This will find the average treatment effect on
        a dataset before the experiment is run, to make
        sure that it is zero.  This tests the assumption
        that the average treatment effect between the test
        and control groups when neither is treated is 0.

        The format for this dataset is the same as that 
        for the real estimation task, except that the start
        time is some time before the experiment is run, and
        the end time is the starting point for the experiment.
        """
        lower, exp, upper = self.average_treatment_effect(X,start=start, end=end, assignment=assignment)
        if lower <= 0 <= upper:
            return True
        return False


 
