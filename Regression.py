import FileReader as f
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels


class RegressionException(Exception):
    pass

class Regression():
    def __init__(self, ideal=None):
        if ideal is None:
            raise RegressionException("No ideal data given")
        
        indices = None
        try:
            indices = pd.DataFrame(ideal.index.to_list(), columns=["i"])
        except ValueError:
            raise RegressionException("Multi index provided")

        if indices.isnull().values.any():
            raise RegressionException("NaN values in index column.")

        try:
            indices.i.apply(float)
        except ValueError as e:
            raise RegressionException("No number values in index column.")

        self._ideal = ideal

    def mse(self, training):
        try:
            return self._ideal.apply(mean_squared_error,y_pred=training)
        except ValueError as e:
            raise RegressionException("{} raised: {}".format(type(e).__name__, e))

    def _max_deviation(self, df_1, df_2):
        diff = (df_1-df_2).abs()
        #print("md df1", df_1.info())
        #print("md df2", df_2.info())
        #print("md diff", diff.info())
        
        if diff.isnull().values.any():
            raise RegressionException("Found NaN values in max_deviation column.")

        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(df_compare)

        return diff.max()*math.sqrt(2)

    def min_value(self, df, diff):
        output = df.apply(min).to_frame("min_value")
        output = output.assign(min_y = df.idxmin())
        output.index.name = "y"
        chosen = None
        try:
            chosen = self._ideal[output.min_y]
            chosen.columns = output.min_y.index
            output["max_deviation"] = self._max_deviation(diff, chosen);
        except KeyError as e:
            raise RegressionException("{} raised: {}".format(type(e).__name__, e))
        return output
    
    def match(self, file, lookup, df, skip_header=True):
        data_list = []
        lines = f.FileReader.csv_line_gen(file)

        # Get next() element to skip the header row before iterating
        if skip_header:
            next(lines)

        for line in lines:
            #print(chosen_func.index)
            match = None
            for index in lookup.index.tolist():
                x = line[0]
                y = float(line[1])
                max_deviation = lookup.max_deviation[index]
                y_caret = df[lookup.min_y[index]][df.index == float(line[0])].values[0]
                y_diff = y - y_caret

                if isinstance(y_diff, np.ndarray):
                    y_diff = y_diff[0]
                if abs(y_diff) <= max_deviation:
                    match = TestData(x=x, y_test=y, delta_y=y_diff, y_ideal=lookup.min_y[index]) \
                    if (match == None or abs(match.delta_y) > abs(y_diff)) else match
                    #min(matches.items(), key=lambda diff: diff[1]))
                    match_list = []
            if match:
                # Add match to data_list
                data_list.append(match.to_dict())
        
        return data_list


class TestData():
    def __init__(self, x, y_test, delta_y, y_ideal):
        self._x = x
        self._y_test = y_test
        self._delta_y = delta_y
        self._y_ideal = y_ideal

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {"X (test func)": self._x, "Y (test func)": self._y_test, \
                "Delta Y (test func)": self._delta_y, "No. of ideal func": self._y_ideal}

    @property
    def delta_y(self):
        return self._delta_y

    @delta_y.setter
    def delta_y(self, delta_y):
        self._delta_y = delta_y

    @delta_y.deleter
    def delta_y(self):
        del self._delta_y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @x.deleter
    def x(self):
        del self._x
