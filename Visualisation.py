from abc import ABC, abstractmethod
import bokeh as bk
from bokeh.layouts import row, gridplot
from bokeh.models import Band, ColumnDataSource
from bokeh.plotting import figure, output_file, show
import math
import numpy as np
import pandas as pd
import os
import re

class Data():
    def __init__(self, label, data, vtype="line", color="blue", line_width=1):
        self.label = label
        self.data = data
        self.vtype = vtype
        self.color = color
        self.line_width = line_width

class Visualizer(ABC):
    def __init__(self, real_data, result_data=None, ideal_data=None, label=None, plot = None):
        self.label = (label if label else "unnamed")
        self.output = "{}.html".format(Visualizer.to_url(self.label))
        self.remove_file()
        output_file(self.output)
        self._plot = (plot if plot is not None else self.new_plot())
        
        self.real = real_data
        self.result = result_data
        self.ideal = ideal_data
    
    def new_plot(self):
        self._plot = figure(title="{}".format(self.label), x_axis_label="x", y_axis_label="y")
        return self._plot

    @staticmethod
    def to_url(str):
        # Replace all whitespaces with an underscore.
        str = re.sub("\s+", "_", str).lower()

        # Remove anything other than numbers letters and underscore.
        return re.sub("[^\w]", "", str)

    def graphs(self, vtype):
        return {
            "line": self._plot.line,
            "circle": self._plot.circle,
        }.get(vtype, self._plot.line)

    def create_plot(self, *data):
        self.remove_file()
        for d in data:
            self.graphs(d.vtype)(x="x", y=d.label, source=ColumnDataSource(data=d.data), legend_label=d.label, \
                                      color=d.color, line_color=d.color, line_width=d.line_width)
    
    def error_bars(self, real, approx):
        df = pd.DataFrame(index=[(i,i) for i, y in real.items()])
        df["y"] = list(zip(real, approx))
        df.index.name = "x"
        #print(df)
        self._plot.multi_line(xs="x", ys="y", source=ColumnDataSource(data=df), color="grey")
        
        #show(self._plot)
    
    def add_band(self, data, index, result_data):
        df = pd.DataFrame(data)
        max_deviation = result_data.loc[index].max_deviation*math.sqrt(2)
        df["lower"] = data - max_deviation
        df["upper"] = data + max_deviation

        self._plot.varea(x='x', y1='upper', y2='lower', source=ColumnDataSource(data=df), \
        fill_alpha=0.2, hatch_color="black", fill_color='yellow', legend_label = "Max Deviation Interval")

        band = Band(base='x', lower='lower', upper='upper', source=ColumnDataSource(data=df), level='underlay',
            line_width=1, line_color='black', fill_alpha=0.0)
        
        self._plot.add_layout(band)
        #show(self._plot)

    def show(self):
        show(self._plot)
        
    def remove_file(self):
        try:
            os.remove(self.output)
        except OSError:
            pass
    
    def prepare_data(self):
        pass

    @abstractmethod
    def additional_elements(self, index, min_y):
        pass
    
    @property
    def plot(self):
        return self._plot

    
class TestVisualizer(Visualizer):
    def additional_elements(self, index, min_y):
        super().add_band(self.ideal[min_y], index, self.result)

    def prepare_data(self):
        x_list = []
        y_test_list = []
        func_list = []
        for d in self.real:
            x_list.append(float(d["X (test func)"]))
            y_test_list.append(d["Y (test func)"])
            func_list.append(d["No. of ideal func"])

        df = pd.DataFrame(zip(x_list,y_test_list, func_list), columns=["x", "y_test", "y_func"]).set_index("x")

        df_test_group = None
        #df_ideal_group = None
        result_group = self.result.copy()
        test_name = lambda y: "Test Data matched to {}".format(y)
        merger = lambda g, df: (g if df is None else df.reset_index()\
                         .merge(g.reset_index(), how="outer").set_index("x"))

        result_group["y"] = result_group.min_y.apply(test_name)
        result_group = result_group.set_index("y")
        grouped = df.groupby("y_func")
        result_group_index = []
        for y, group in grouped:
            group["y_ideal"] = self.ideal[y]
            group = group.rename(columns={"y_func": "min_y"})
            index_name = test_name(y)
            result_group_index.append(index_name)


            test_group = group.rename({"y_test": result_group[result_group.min_y == y].index.to_list().pop()}, \
                                      axis=1).drop("y_ideal", axis=1)
            #ideal_group = group.rename({"y_ideal": y}, axis=1).drop("y_test", axis=1)

            df_test_group = merger(test_group, df_test_group)
            #df_ideal_group = merger(ideal_group, df_ideal_group)

        df_test_group = df_test_group.sort_index()
        #df_ideal_group = df_ideal_group.sort_index()

        #print("Result Group")
        #print(result_group)
        #print("Test Group\n")
        #print(df_test_group)
        #print("Ideal Group\n")
        #print(df_ideal_group)
        #print("\nNext")
        
        self.result = result_group
        self.real = df_test_group

class TrainingVisualizer(Visualizer):
    def additional_elements(self, index, min_y):
        super().error_bars(self.real[index], self.ideal[min_y])
    
class PlotFactory():
    @staticmethod
    def create_visualizer(result_data, real_data, ideal_data, label=None, vtype=None):
        dict_ = {"train": TrainingVisualizer(result_data, real_data, ideal_data, label), \
                "test": TestVisualizer(result_data, real_data, ideal_data,label)}
        return dict_.get(vtype, dict_.get("train"))
    
    @staticmethod
    def show(label, real_data, result_data, ideal_data, vtype=None):
        v = PlotFactory.create_visualizer(real_data, result_data, ideal_data, label, vtype)

        v.prepare_data()

        data = np.array([])

        if v.result is not None:
            for index, row in v.result.iterrows():
                v.create_plot(Data(index, v.real, color="red", vtype="circle"), \
                              Data(row.min_y, v.ideal, line_width=3))
                v.additional_elements(index, row.min_y)
                data = np.append(data, v.plot)
                v.new_plot()

        plot = gridplot(list(np.reshape(data,(2,2))))
        
        show(plot)
