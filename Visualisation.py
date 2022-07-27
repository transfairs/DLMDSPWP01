from abc import ABC, abstractmethod
import bokeh as bk
from bokeh.layouts import row, gridplot
from bokeh.models import Band, ColumnDataSource
from bokeh.plotting import figure, output_file, show
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
import os
import re

@dataclass
class Data():
    '''
    Holds data to display and information about how to format a chart.

    Args:
        label: Name of the chart.
        data: Data to be displayed
        vtype: Symbol type used in the chart. Defaults to line.
        color: Color of the symbols.
    
    '''

    label: str
    data: pd.DataFrame
    vtype: str = "line"
    color: str = "blue"
    line_width: int = 1

        
class Visualizer(ABC):
    '''
    Blueprint class for creating different kind of charts.
    
    Creates the `output` URL from 'label' if provided.

    Args:
        real_data: Data to be displayed in the chart.
        result_data: Additional information about `real_data`. 
            Defaults to None.
        ideal_data: Ideal functions for providing an overlay. Defaults to None.
        label: Name of the Chart. Defaults to None.
        plot (bokeh.plotting.figure.Figure): An existing plot.
    
    Attributes:
        label: Name of the Chart.
        output: URL of the HTML output file.
        real: Data to be displayed in the chart.
        result: Additional information about `real`. 
        ideal: Ideal functions for providing an overlay.
    
    '''

    def __init__(self, real_data, result_data=None, ideal_data=None, \
                 label=None, plot = None):
        self.label = (label if label else "unnamed")
        self.output = "{}.html".format(Visualizer.to_url(self.label))
        self.remove_file()
        output_file(self.output)
        self._plot = (plot if plot is not None else self.new_plot())
        
        self.real = real_data
        self.result = result_data
        self.ideal = ideal_data
    
    def new_plot(self):
        '''Creates a new (x,y) plot for the instance, titled `label` value.

        Returns:
            bokeh.plotting.figure.Figure: The created plot.

        '''
        self._plot = figure(title="{}".format(self.label), x_axis_label="x", \
                            y_axis_label="y")
        return self._plot

    @staticmethod
    def to_url(str):
        '''Removes special characters from a string.

        Args:
            str: The string to be cleaned.

        Returns:
            The cleaned string, consisting of word characters only.

        '''
        # Replace all whitespaces with an underscore.
        str = re.sub("\s+", "_", str).lower()

        # Remove anything other than numbers letters and underscore.
        return re.sub("[^\w]", "", str)

    def graphs(self, vtype):
        '''Selects glyphs factory method for a related string value.

        This can be used as a switch-case instruction.

        Args:
            vtype (str): One of `line` or `circle`.

        Returns:
            method: Creator for `_plot`. Will return the line glyph renderer by
                default.
            
        '''
        return {
            "line": self._plot.line,
            "circle": self._plot.circle,
        }.get(vtype, self._plot.line)

    def create_plot(self, *data):
        '''Creates a plot for given data.

        Args:
            *data (tuple): The data to be displayed in the plot.
            
        '''
        self.remove_file()

        for d in data:
            self.graphs(d.vtype)(x="x", y=d.label, \
                                 source=ColumnDataSource(data=d.data), \
                                 legend_label=d.label, \
                                 color=d.color, line_color=d.color, \
                                 line_width=d.line_width)
    
    def error_bars(self, real, approx):
        '''Adds vertical, grey lines between two pandas Series to the plot.

        Args:
            real (pandas.core.series.Series): The upper boundary for the 
                vertical lines.
            approx (pandas.core.series.Series): The lower boundary for the 
                vertical lines.
            
        '''
        df = pd.DataFrame(index=[(i,i) for i, y in real.items()])
        df["y"] = list(zip(real, approx))
        df.index.name = "x"
        self._plot.multi_line(xs="x", ys="y", \
                              source=ColumnDataSource(data=df), color="grey")
    
    def add_band(self, data, index, result_data):
        '''Adds a band/interval to the plot.

        Args:
            data (pandas.core.series.Series): The (invisible) center line for 
                the interval.
            index (str): Label for the added dots. Used to get the correct
                data out of result data.
            result_data (pandas.core.frame.DataFrame): Contains information 
                about the size of the band.
            
        '''
        df = pd.DataFrame(data)
        max_deviation = result_data.loc[index].max_deviation
        df["lower"] = data - max_deviation
        df["upper"] = data + max_deviation

        # Plots the yellow filled area around the center line.
        self._plot.varea(x='x', y1='upper', y2='lower', \
                         source=ColumnDataSource(data=df), \
                         fill_alpha=0.2, hatch_color="black", \
                         fill_color='yellow', \
                         legend_label = "Max Deviation Interval")

        # Plots the boundary lines around the intervall.
        band = Band(base='x', lower='lower', upper='upper', \
                    source=ColumnDataSource(data=df), level='underlay', \
                    line_width=1, line_color='black', fill_alpha=0.0)
        
        self._plot.add_layout(band)

    def show(self):
        '''Displays the plot to the caller.'''
        show(self._plot)
        
    def remove_file(self):
        '''Deletes an existing HTML output file with the same name as given 
        in `output`.
            
        '''
        try:
            os.remove(self.output)
        except OSError:
            pass
    
    def prepare_data(self):
        '''Optional functionality to be defined by inheriting classes for 
        data wrangling before the plot gets created.
        
        '''
        pass

    @abstractmethod
    def additional_elements(self, index, min_y):
        '''Adds additional elements to the plot. Mandatory to implement by 
        inheriting classes.

        Args:
            index (str): Supposed be used to identify data from `real` to be 
                used. Also serves as label for this.
            min_y (str): Identifies the ideal function to use from `ideal`.
                Also serves as label for this.
            
        '''
        pass
    
    @property
    def plot(self):
        '''Returns the plot associated to the instance.

        Returns:
             bokeh.plotting.figure.Figure: The plot of the instance.
            
        '''
        return self._plot

    
class TestVisualizer(Visualizer):
    '''
    A visualizer object for the test data task.
    
    '''

    def additional_elements(self, index, min_y):
        '''Adds an error band to the plot to show that the test data is 
        within the specified boundary.
        
        '''
        super().add_band(self.ideal[min_y], index, self.result)

    def prepare_data(self):
        '''Overwrites `real` with a df containing the Test data and matched 
        ideal function and updates `result` based on the updated data set.
            
        '''
        x_list = []
        y_test_list = []
        func_list = []
        
        # Define columns for the to be generated data set.
        for d in self.real:
            x_list.append(float(d["X (test func)"]))
            y_test_list.append(d["Y (test func)"])
            func_list.append(d["No. of ideal func"])

        df = pd.DataFrame(zip(x_list,y_test_list, func_list), \
                          columns=["x", "y_test", "y_func"]).set_index("x")

        df_test_group = None
        result_group = self.result.copy()
        
        # Anonymous function to label test data.
        test_name = lambda y: "Test Data matched to {}".format(y)

        # Anonymous function to merge a data set g to an existing 
        # dataframe df.
        # If df does not exist g will define the initial state.
        merger = lambda g, df: (g if df is None else df\
                                .reset_index().merge(g.reset_index(), \
                                                     how="outer")\
                                .set_index("x"))

        # Re-shaping of `result` to cater for test data.
        result_group["y"] = result_group.min_y.apply(test_name)
        result_group = result_group.set_index("y")
        
        # Test data is mapped to none or one of the chosen ideal functions.
        # Group by ideal function to get all the test data that matches to 
        # this function.
        grouped = df.groupby("y_func")
        result_group_index = []
        for y, group in grouped:
            # Add the ideal function data to the group.
            group["y_ideal"] = self.ideal[y]
            
            # Rename columns to standardise the output and add the ideal 
            # function to the new result data frame.
            group = group.rename(columns={"y_func": "min_y"})
            index_name = test_name(y)
            result_group_index.append(index_name)

            # Extract test data for this ideal function.
            test_group = group.rename({"y_test": \
                                       result_group[result_group.min_y == y]\
                                       .index.to_list().pop()}, \
                                      axis=1).drop("y_ideal", axis=1)

            # Merge to existing test data in the data frame.
            df_test_group = merger(test_group, df_test_group)

        df_test_group = df_test_group.sort_index()
        
        self.result = result_group
        self.real = df_test_group

class TrainingVisualizer(Visualizer):
    '''
    A visualizer object for the training data task.
    
    '''

    def additional_elements(self, index, min_y):
        '''Adds error bars to the plot to show the deviation to the mapped 
        ideal function.
        
        '''
        super().error_bars(self.real[index], self.ideal[min_y])
    
class PlotFactory():
    '''
    Provides a factory for plots.
    
    '''
    @staticmethod
    def create_visualizer(result_data, real_data, ideal_data, label=None, \
                          vtype=None):
        '''Static method that creates an instance of Visualizer.

        Args:
            result_data: Serves as `result` for the Visualizer object.
            real_data: Serves as `real` for the Visualizer object.
            ideal_data: Serves as `ideal` for the Visualizer object.
            label: Serves as `label` for the Visualizer object. Defaults to
                None.
            vtype (str): Determines which type of Visualizer to be created.
                Defaults to None.
        
        Returns:
            Visualizer: Either TestVisualizer or TrainingVisualizer. Defaults 
                to the latter.
            
        '''
        dict_ = {"train": TrainingVisualizer(result_data, real_data, \
                                             ideal_data, label), \
                 "test": TestVisualizer(result_data, real_data, ideal_data, \
                                        label)}
        return dict_.get(vtype, dict_.get("train"))
    
    @staticmethod
    def show(label, real_data, result_data, ideal_data, vtype=None):
        '''Static method that displays a plot.
        
        Call this function to show your data in a plot.

        Args:
            label: Serves as label/title for the plot.
            real_data: The data to be displayed in the plot.
            result_data: Additional information about `real_data`.
            ideal_data: Ideal functions for providing an overlay.
            vtype (str): Determines which type of Visualizer to be created.
                Defaults to None which will create a training data plot.
            
        '''
        v = PlotFactory.create_visualizer(real_data, result_data, \
                                          ideal_data, label, vtype)

        v.prepare_data()

        data = np.array([])

        # Create plot, including all specified overlays as per Visualizer type.
        if v.result is not None:
            for index, row in v.result.iterrows():
                v.create_plot(Data(index, v.real, color="red", \
                                   vtype="circle"), Data(row.min_y, v.ideal, \
                                                         line_width=3))
                try:
                    v.additional_elements(index, row.min_y)
                except ValueError as e:
                    print("Cannot add index {} with value " + \
                          "{}: {}".format(index, row.min_y, e))
                data = np.append(data, v.plot)
                v.new_plot()

        plot = gridplot(list(np.reshape(data,(2,2))))
        
        show(plot)
