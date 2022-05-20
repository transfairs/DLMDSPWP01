# from abc import ABC, abstractmethod
# import bokeh as bk
# from bokeh.layouts import row, gridplot
# from bokeh.models import Band, ColumnDataSource
# from bokeh.plotting import figure, output_file, show
# import csv
# import math
# import matplotlib as plt
# import numpy as np
# import os
import pandas as pd
# from pandas.io import sql
# import pymysql
# import re
# from sklearn.metrics import mean_squared_error
# import scipy
# import sqlalchemy as db
# import statsmodels
import unittest


class UnitTestFileReader(unittest.TestCase):
    def test_read_csv_to_df(self):
        try:
            output = FileReader.read_csv_to_df()
            self.assertIsNone(output)

            output = FileReader.read_csv_to_df(file="data/test.csv")
            self.assertIsNotNone(output)
            self.assertEqual(output.size, 100)
            self.assertIsInstance(output, pd.DataFrame)
            self.assertIsInstance(output["y"], pd.Series)
            self.assertEqual(output.index.name, "x")

            output = FileReader.read_csv_to_df(file="doesNotExist.csv")
            self.assertIsNone(output)
        except:
            self.fail("Exception raised")

    def test_csv_line_gen(self):
        try:
            no_gen = FileReader.csv_line_gen()
            self.assertIsNotNone(no_gen)
            try:
                next(no_gen)
                self.fail("Got entry from empty generator")
            except StopIteration:
                pass

            generator = FileReader.csv_line_gen(file="data/train.csv")
            counter = 0
            for line in generator:
                if counter == 4:
                    self.assertEqual(len(line), 5)
                counter += 1
            self.assertEqual(counter, 401)
        except:
            self.fail("Exception raised")

            
            
            
            
            
            
            
class UnitTestRegression(unittest.TestCase):
    def test_constructor(self):
        try:
            Regression()
            self.fail("No Exception raised")
        except AssertionError as e:
            raise e
        except RegressionException:
            pass
        except Exception as e:
            self.fail("Wrong Exception raised: {}".format(type(e).__name__))
            
        x_list = [x for x in range(1,101,1)]
        y_list = ["y{}".format(x) for x in range(1,101,1)]
        
        df_ideal = pd.DataFrame(20*np.random.random_sample(size=(100,100))-10, columns=[x for x in range(100)])
        df_ideal["y"] = y_list
        df_ideal = df_ideal.set_index("y")

        try:
            Regression(df_ideal)
            self.fail("No RegressionException raised")
        except RegressionException as e:
            #self.assertEqual("NaN values in index column.", e.__str__())
            self.assertEqual("No number values in index column.", e.__str__())

        multi_index = [(float(x/10),20*np.random.random_sample()-10) for x in range(100)]
        df_ideal["x"] = multi_index
        df_ideal = df_ideal.set_index("x")
        
        try:
            Regression(df_ideal)
            self.fail("No RegressionException raised")
        except RegressionException as e:
            #self.assertEqual("NaN values in index column.", e.__str__())
            self.assertEqual("Multi index provided", e.__str__())
        

        df_ideal["x"] = x_list
        df_ideal = df_ideal.set_index("x")
        
        regression = Regression(df_ideal)
        
        self.assertIsInstance(regression, Regression)

    def test_mse(self):
        df = pd.DataFrame(20*np.random.random_sample(size=(100, 3))-10, columns=["x", "y", "y2"]).set_index("x")
        df2 = df.copy().drop("y2", axis=1)
        regression = Regression(df2)
        try:
            regression.mse(df)
            self.fail("No RegressionException raised")
        except RegressionException:
            pass
        except Exception as e:
            self.fail("Wrong Exception raised: {}".format(type(e).__name__))

        self.assertEqual(regression.mse(df2).y, 0)

    def test_min_value(self):
        x_list = [(float(x/10),20*np.random.random_sample()-10) for x in range(-50, 50, 1)]
        y_list = ["y{}".format(x) for x in range(1,101,1)]

        df_ideal = pd.DataFrame(20*np.random.random_sample(size=(100,100))-10, columns=y_list)
        df_ideal["x"] = [float(x/10) for x in range(-50, 50, 1)]
        df_ideal = df_ideal.set_index("x")
        
        df = pd.DataFrame(np.random.random_sample(size=(100,100)), columns=y_list)
        df["y"] = y_list
        df = df.set_index("y")

        diff = pd.DataFrame(x_list, columns=["x", "z"]).set_index("x")

        regression = Regression(df_ideal)
        
        try:
            regression.min_value(df, diff)
            self.fail("No RegressionException raised")
        except RegressionException as e:
            self.assertEqual("Found NaN values in max_deviation column.", e.__str__())
        except Exception as e:
            self.fail("Wrong Exception raised: {}".format(type(e).__name__))
        
        df.index = y_list
        diff = pd.DataFrame(np.random.random_sample(size=(100,100)), columns=y_list)
        diff["x"] = [float(x/10) for x in range(-50, 50, 1)]
        diff = diff.set_index("x")
        
        min_value = regression.min_value(df, diff)
        
        self.assertFalse(min_value.isnull().values.any())

    def test_match(self):
        y_list = ["y{}".format(x) for x in range(1,11,1)]
        index_list = ["y{}".format(np.random.randint(1,10)) for i in range(0,400)]

        df_ideal = pd.DataFrame(20*np.random.random_sample(size=(400,10))-10, columns=y_list)
        df_ideal["x"] = [float(x/10) for x in range(-200, 200, 1)]
        df_ideal = df_ideal.set_index("x")

        df = pd.DataFrame(np.random.random_sample(size=(400,10)), columns=y_list)
        df.index = index_list
        diff = pd.DataFrame(np.random.random_sample(size=(400,10)), columns=y_list)
        diff["x"] = [float(x/10) for x in range(-200, 200, 1)]
        diff = diff.set_index("x")

        regression = Regression(df_ideal)
        
        min_value = regression.min_value(df, diff)
        chosen_ideal = df_ideal[min_value.min_y.tolist()]
        
        match = regression.match("data/test.csv", min_value, chosen_ideal)
        
        self.assertIsNotNone(match)
        self.assertNotEqual(len(match),0)

class UnitTestPersistenceUtil(unittest.TestCase):
    def test_df_column_name(self):
        pu = PersistenceUtil(db.create_engine(f"sqlite:///data/unittest.db"))
        
        tests = {"test": ["abc", "test"], \
                 "t3st": ["abc", "t3st"], \
                 "ty3st": ["abc", "tabcst"], \
                 "ty3st": [r"a\1c", "ta3cst"], \
                 "y1y4y67": ["abc", "abcabcabc7"], \
                 "y1 y4 y67": [r"a\1b\1c\1d\1", "a1b1c1d1 a4b4c4d4 a6b6c6d67"], \
                 "tY1st": ["abc", "tY1st"]}
        
        for k in tests:
            self.assertEqual(pu.df_column_name(k,tests[k][0]), tests[k][1])

    def test_save_df_to_db(self): # df, table, regex_column_name):
        y_list = ["y{}".format(x) for x in range(1,11,1)]
        ut_list = ["u_t{}".format(x) for x in range(1,11,1)]
        pu = PersistenceUtil(db.create_engine(f"sqlite:///data/unittest.db"))
        df = pd.DataFrame(np.random.random_sample(size=(400,10)), columns=y_list)
        df.index.name = "x"
        table_name = "u_test_01"
        
        rows_count = pu.save_df_to_db(df, table_name, r"u_t\1")
        df_db = pu.read_df(table_name).set_index("X")

        self.assertEqual(df_db.columns.to_list(), ut_list)

        df_db.columns = y_list
        df_db.index.name = df_db.index.name.lower()
        
        self.assertEqual(len(df.index), rows_count)
        self.assertEqual(len(df.index), len(df_db.index))
        self.assertEqual(len(df.columns), len(df_db.columns))

        self.assertTrue(df.equals(df_db))
    
    def test_create_and_persist(self): # table, columns):
        data_list = [\
                     {"col_str": "Test value", "col_int": -3}, \
                     {"col_str": "Test value 2", "col_int": 5.0} \
                    ]
        
        table_name = "test_create"
        pu = PersistenceUtil(db.create_engine(f"sqlite:///data/unittest.db"))
        try:
            table = pu.create_table(table_name, {"col_str": db.String(10), "col_int": db.Integer}, \
                                    drop_if_exists=True)
            self.assertEqual(table, table_name)
            pu.persist(table, data_list)

            df = pu.read_df(table_name).set_index("col_str")
            df_dict = pd.DataFrame(data_list).set_index("col_str")
            df_dict["col_int"] = df_dict.col_int.apply(int)
            self.assertTrue(df.equals(df_dict))

        except Exception as e:
            self.fail("Exception raised: {}".format(type(e).__name__))
        
        data_list = [\
                     {"col_str": "Test value 3", "col_int": -3.5}, \
                     {"col_str": "Test value 4", "col_int": "String"} \
                    ]
        try:
            pu.persist(table, data_list)
            df = pu.read_df(table_name).set_index("col_str")
            print(df)
        except Exception as e:
            self.assertIsInstance(e, ValueError)

class UnitTestVisualizer(unittest.TestCase):
    def test_to_url(self):
        tests = {"I lov' #coo$k-ies.": "i_lov_cookies", \
               " ": "_", \
               "": ""}
        
        for k in tests:
            self.assertEqual(Visualizer.to_url(k), tests[k])
    
class UnitTestPlotfactory(unittest.TestCase):
    def test_create_visualizer(self):
        df = pd.DataFrame(np.random.randint(0,20,size=(5, 3)), columns=["x", "y", "z"]).set_index("x")

        test = TestVisualizer(df, df, df)
        train = TrainingVisualizer(df, df, df)
        tests = {"I lov' #coo$k-ies.": train, \
                 "test": test, \
               None: train, \
               "train": train}
        
        for k in tests:
            self.assertIsInstance(PlotFactory.create_visualizer(df, df, df, vtype=k), type(tests[k]))






if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
