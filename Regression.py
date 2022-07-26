import FileReader as f
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels


class RegressionException(Exception):
    '''
    Custom exception to classify errors in calculations.
    
    '''
    
    pass

class Regression():
    '''
    A class, containing the data wrangling related business logic.
    
    Although, parameter `ideal` is optional, it will raise an exception if not 
    provided. This covers the general use case. However, some methods of the 
    class do not require `ideal` and the caller can use exception handling to 
    cater for this.
    The class has no public attributes to be documented.

    Args:
        ideal: A pandas data frame containing ideal functions.
            Defaults to None.

    Raises:
        RegressionException: If `ideal` is None or not a data frame, has multi
            index, missing indexes or some that are incompatible with float.
            
    '''

    def __init__(self, ideal=None):
        if ideal is None or not isinstance(ideal, pd.DataFrame):
            raise RegressionException("No ideal data given or wrong type " +
                                      "detected. Pandas data frame expected.")
        
        # Data frame ideal should be single indexed.
        indices = None
        try:
            indices = pd.DataFrame(ideal.index.to_list(), columns=["i"])
        except ValueError:
            raise RegressionException("Multi index provided")
        
        # A data frame with missing indexes is not supported.
        if indices.isnull().values.any():
            raise RegressionException("NaN values in index column.")

        # Convert indexes to floating point numbers.
        try:
            indices.i.apply(float)
        except ValueError as e:
            raise RegressionException("No number values in index column.")

        self._ideal = ideal

    def mse(self, training):
        '''Calculates the Mean Squared Error for a pandas Series.
        
        The method uses sklearn functionality and the benchmark for the 
        calculation is `self._ideal`.

        Args:
            training: The pandas Series to be checked.
        
        Returns:
            A Series with `ideal` functions as index and MSE value for
                `training`.
        
        Raises:
            RegressionException: If `training` is not a Series or if a 
                ValueError occured during calculation.
                
        '''
        if not isinstance(training, pd.core.series.Series):
            raise RegressionException("Wrong type for parameter training. " + \
                                      "Pandas data frame expected.")
            
        try:
            return self._ideal.apply(mean_squared_error,y_pred=training)
        except ValueError as e:
            raise RegressionException("{} raised: {}"\
                                      .format(type(e).__name__, e))

    def _max_deviation(self, df_1, df_2):
        '''Calculates the maximum deviation for each value in two pandas data 
        frames, adjusted by a factor.
        
        The order of the arguments does not matter for the result.
        
        Args:
            df_1: The first data frame.
            df_2: The Second data frame.
        
        Returns:
            The positive distance between each value in the data frame as a
                data frame and multiplied by the square root of 2.
        
        Raises:
            RegressionException: If an error occured during calculation or the
                columns of the data frames do not match.
                
        '''
        try:
            diff = (df_1-df_2).abs()
            if diff.isnull().values.any():
                raise RegressionException("Found NaN values in " + 
                                          "max_deviation column.")
            return diff.max()*math.sqrt(2)
        except:
            raise RegressionException("{} raised: {}"\
                                      .format(type(e).__name__, e))

    def min_value(self, df, diff):
        '''Finds the minimum value for all columns of a given data frame.
        
        Args:
            df: The data frame to get minimum value of its columns.
            diff: A data frame to get or calculate statistical data from.

        Returns:
            A dataframe consisting of columns about the minimum and statistical
                data (mean, variance, maximum deviation) using `diff`.
        
        Raises:
            RegressionException: If an index mismatch between `df` and `diff` 
                was detected or one of the parameters is not a data frame.
                
        '''
        if not isinstance(df, pd.DataFrame):
            raise RegressionException("Wrong type for parameter df. " + \
                                      "Pandas data frame expected.")
        if not isinstance(diff, pd.DataFrame):
            raise RegressionException("Wrong type for parameter diff. " + \
                                      "Pandas data frame expected.")
        
        # Create a new data frame and get minimum value for each column.
        output = df.apply(min).to_frame("min_value")
        # Add index of that row to the data frame.
        output = output.assign(min_y = df.idxmin())
        # Add statistical information from `diff`.
        output["mean"] = diff.mean()
        output["variance"] = diff.var()
        output.index.name = "y"

        try:
            # Copy the matched ideal functions to a new data frame.
            chosen = self._ideal[output.min_y]
            # Set the index to the function name/number.
            chosen.columns = output.min_y.index
            # Add another column to the output data frame containing the
            # maximum deviation between `diff` and the chosen ideal functions.
            output["max_deviation"] = self._max_deviation(diff, chosen);
        except KeyError as e:
            raise RegressionException("{} raised: {}"\
                                      .format(type(e).__name__, e))

        return output
    
    def match(self, file, lookup, df, skip_header=True):
        '''Matches a data to a data frame if provided maximum is not exceeded.
        
        Args:
            file (str): Path to the CSV file containing the data to test.
            lookup: Data frame that provides lookup criteria.
            df: Data frame with data to evaluate against.
            skip_header (bool): If the CSV file comes with a header row skip
                this when reading the data. Defaults to True.

        Returns:
            list (TestData): Results of the matching process for each line in 
                the CSV file.
        
        Raises:
            RegressionException: If the matching process raises an Exception.
            
        '''
        data_list = []
        lines = f.FileReader.csv_line_gen(file)

        # Get next() element to skip the header row before iterating.
        if skip_header:
            next(lines)
        
        for line in lines:
            match = None
            # Try to match to a subset (defined as index in `lookup`) of `df`.
            for index in lookup.index.tolist():
                try:
                    x = line[0]
                    y = float(line[1])
                    # By default, the test data is saved with NULL values for
                    # the matching columns.
                    match = TestData(x=x, y_test=y, delta_y=None, \
                                     y_ideal=None) if not match else match
                    max_deviation = lookup.max_deviation[index]
                    # Get the corresponding y value from `df`.
                    y_caret = df[lookup.min_y[index]][df.index == \
                                                      float(line[0])].values[0]
                    y_diff = y - y_caret

                    # If multiple values found, only interested in one.
                    # This happens if an ideal function is the best match for 
                    # more than one training data set.
                    if isinstance(y_diff, np.ndarray):
                        y_diff = y_diff[0]
                    if abs(y_diff) <= max_deviation:
                        match = TestData(x=x, y_test=y, delta_y=y_diff, \
                                         y_ideal=lookup.min_y[index]) \
                        if (match.delta_y == None or abs(match.delta_y) > \
                            abs(y_diff)) else match
                except Exception as e:
                    raise RegressionException("{} raised: {}"\
                                              .format(type(e).__name__, e))

            # Add match to data_list
            data_list.append(match.to_dict())

        return data_list


class TestData():
    '''
    A data class serving for the requirements of holding test data.
    
    Test data consists of an independent `x` and dependent `y` value that has 
    a certain difference (`y_delta`) to a corresponding y value in an ideal 
    function.

    Args:
        x (float): Independent value.
        y_test (float): Dependent value.
        delta_y (float): Difference of `y_test` to a corresponding value in 
            the ideal function.
        y_ideal (str): Name of the ideal function.
        
    '''
    
    def __init__(self, x, y_test, delta_y, y_ideal):
        self._x = x
        self._y_test = y_test
        self._delta_y = delta_y
        self._y_ideal = y_ideal

    def __str__(self):
        '''Overwrites string representation.

        Returns:
            The object's custom, private attributes as dictionary string.
            
        '''
        return str(self.to_dict())

    def to_dict(self):
        '''Creates a dictionary that contains the object's custom, private 
        attributes.

        Returns:
            The object's custom, private attributes as a dictionary.
            
        '''
        return {"X (test func)": self._x, \
                "Y (test func)": self._y_test, \
                "Delta Y (test func)": self._delta_y, \
                "No. of ideal func": self._y_ideal}

    @property
    def delta_y(self):
        '''float: Difference of `y_test` to a corresponding value in the 
        ideal function.
        '''
        return self._delta_y

    @delta_y.setter
    def delta_y(self, delta_y):
        self._delta_y = delta_y

    @delta_y.deleter
    def delta_y(self):
        del self._delta_y

    @property
    def x(self):
        '''float: Independent value.'''
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @x.deleter
    def x(self):
        del self._x
