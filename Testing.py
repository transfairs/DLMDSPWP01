from FileReader import FileReader
import numpy as np
import pandas as pd
import Regression as r
from SQL import PersistenceUtil
import sqlalchemy as db
import Visualisation as visual
import unittest

'''
This file contains a collection of unit tests

The classes and methods are named after the main functionality that they
test. It does not seem to be best practice to add docstrings to unittests.
Comments are primarily in the failing messages of the tests.

'''

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
            # Parameter `ideal` is mandatory although defined as optional. 
            # Requires exception handling if not set.
            r.Regression()
            self.fail("No Exception raised")
        except AssertionError as e:
            raise e
        except r.RegressionException:
            pass
        except Exception as e:
            self.fail("Wrong Exception raised: {}".format(type(e).__name__))
            
        # Set up scenario where indexes are not numbers.
        x_list = [x for x in range(1,101,1)]
        y_list = ["y{}".format(x) for x in range(1,101,1)]
        
        df_ideal = pd.DataFrame(20*np.random.random_sample(size=(100,100))-10,\
                                columns=[x for x in range(100)])
        df_ideal["y"] = y_list
        df_ideal = df_ideal.set_index("y")

        try:
            r.Regression(df_ideal)
            self.fail("No RegressionException raised")
        except r.RegressionException as e:
            self.assertEqual("No number values in index column.", e.__str__())

        multi_index = [(float(x/10),\
                        20*np.random.random_sample()-10) for x in range(100)]
        df_ideal["x"] = multi_index
        df_ideal = df_ideal.set_index("x")
        
        try:
            r.Regression(df_ideal)
            self.fail("No RegressionException raised")
        except r.RegressionException as e:
            self.assertEqual("Multi index provided", e.__str__())

        df_ideal["x"] = x_list
        df_ideal = df_ideal.set_index("x")
        
        regression = r.Regression(df_ideal)
        
        self.assertIsInstance(regression, r.Regression)

    def test_mse(self):
        # Set up a scenario where `training` is not a Series (df).
        df = pd.DataFrame(20*np.random.random_sample(size=(100, 3))-10, \
                          columns=["x", "y", "y2"]).set_index("x")

        ideal = df.copy().drop("y2", axis=1)

        regression = r.Regression(ideal)
        try:
            regression.mse(df)
            self.fail("No RegressionException raised")
        except r.RegressionException:
            pass
        except Exception as e:
            self.fail("Wrong Exception raised: {}".format(type(e).__name__))

        # Test again with a Series.
        series = ideal.copy().squeeze()
        self.assertEqual(regression.mse(series).y, 0)

    def test_min_value(self):
        # Set up a scenario where columns do not fit. This results in NaN in 
        # max_deviation column.
        x_list = [(float(x/10), \
                   20*np.random.random_sample()-10) for x in range(-50, 50, 1)]
        y_list = ["y{}".format(x) for x in range(1,101,1)]

        df_ideal = pd.DataFrame(20*np.random.random_sample(size=(100,100))-10,\
                                columns=y_list)
        df_ideal["x"] = [float(x/10) for x in range(-50, 50, 1)]
        df_ideal = df_ideal.set_index("x")
        
        df = pd.DataFrame(np.random.random_sample(size=(100,100)), \
                          columns=y_list)
        df["y"] = y_list
        df = df.set_index("y")

        diff = pd.DataFrame(x_list, columns=["x", "z"]).set_index("x")

        regression = r.Regression(df_ideal)
        
        try:
            regression.min_value(df, diff)
            self.fail("No RegressionException raised")
        except r.RegressionException as e:
            self.assertEqual("RegressionException raised: Found NaN " + \
                             "values in max_deviation column.", \
                             e.__str__())
        except Exception as e:
            self.fail("Wrong Exception raised: {}".format(type(e).__name__))
        
        # Double-check with fixed example.
        df.index = y_list

        diff = pd.DataFrame(np.random.random_sample(size=(100,100)), \
                            columns=y_list)
        diff["x"] = [float(x/10) for x in range(-50, 50, 1)]
        diff = diff.set_index("x")
        
        min_value = regression.min_value(df, diff)
        
        self.assertFalse(min_value.isnull().values.any())

    def test_match(self):
        # For any given comparison of test data, the number of matches
        # returned should match the number of rows in the test data
        # file (100) - even when unmatched.
        
        # Create a scenario with random ideal functions.
        y_list = ["y{}".format(x) for x in range(1,11,1)]
        index_list = ["y{}".format(np.random.\
                                   randint(1,10)) for i in range(0,400)]

        df_ideal = pd.DataFrame(20*np.random.random_sample(size=(400,10))-10, \
                                columns=y_list)
        df_ideal["x"] = [float(x/10) for x in range(-200, 200, 1)]
        df_ideal = df_ideal.set_index("x")

        df = pd.DataFrame(np.random.random_sample(size=(400,10)), \
                          columns=y_list)
        df.index = index_list
        diff = pd.DataFrame(np.random.random_sample(size=(400,10)), \
                            columns=y_list)
        diff["x"] = [float(x/10) for x in range(-200, 200, 1)]
        diff = diff.set_index("x")

        regression = r.Regression(df_ideal)
        
        min_value = regression.min_value(df, diff)
        chosen_ideal = df_ideal[min_value.min_y.tolist()]
        
        match = regression.match("data/test.csv", min_value, chosen_ideal)
        
        self.assertIsNotNone(match)
        self.assertEqual(len(match),100)

class UnitTestPersistenceUtil(unittest.TestCase):
    def test_save_df_to_db(self):
        y_list = ["y{}".format(x) for x in range(1,11,1)]
        ut_list = ["u_t{}".format(x) for x in range(1,11,1)]
        pu = PersistenceUtil(db.create_engine(f"sqlite:///data/unittest.db"))
        df = pd.DataFrame(np.random.random_sample(size=(400,10)), \
                          columns=y_list)
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
    
    def test_create_and_persist(self):
        data_list = [\
                     {"col_str": "Test value", "col_int": -3}, \
                     {"col_str": "Test value 2", "col_int": 5.0} \
                    ]
        
        table_name = "test_create"
        pu = PersistenceUtil(db.create_engine(f"sqlite:///data/unittest.db"))
        try:
            table = pu.create_table(table_name, {"col_str": db.String(10), \
                                                 "col_int": db.Integer}, \
                                    drop_if_exists=True)
            self.assertEqual(table, table_name)
            pu.persist(table, data_list)

            df = pu.read_df(table_name).set_index("col_str")
            df_dict = pd.DataFrame(data_list).set_index("col_str")
            df_dict["col_int"] = df_dict.col_int.apply(int)
            self.assertTrue(df.equals(df_dict))

        except Exception as e:
            self.fail("Exception raised: {}".format(type(e).__name__))
        
        # Try to write a string value to an int column.
        data_list = [\
                     {"col_str": "Test value 3", "col_int": -3.5}, \
                     {"col_str": "Test value 4", "col_int": "String"} \
                    ]
        try:
            pu.persist(table, data_list)
            df = pu.read_df(table_name).set_index("col_str")
            print(df)
        except Exception as e:
            self.assertIsInstance(e, TypeError)

class UnitTestVisualizer(unittest.TestCase):
    def test_to_url(self):
        tests = {"I lov' #coo$k-ies.": "i_lov_cookies", \
               " ": "_", \
               "": ""}
        
        for k in tests:
            self.assertEqual(visual.Visualizer.to_url(k), tests[k])
    
class UnitTestPlotfactory(unittest.TestCase):
    def test_create_visualizer(self):
        # Test that  PlotFactory generates the correct Visualizer type.
        df = pd.DataFrame(np.random.randint(0,20,size=(5, 3)), \
                          columns=["x", "y", "z"]).set_index("x")

        test = visual.TestVisualizer(df, df, df)
        train = visual.TrainingVisualizer(df, df, df)
        tests = {"I lov' #coo$k-ies.": train, \
                 "test": test, \
               None: train, \
               "train": train}
        
        for k in tests:
            self.assertIsInstance(visual.PlotFactory.\
                                  create_visualizer(df, df, df, vtype=k), \
                                  type(tests[k]))


if __name__ == '__main__':
    # Will generate some output.
    unittest.main(argv=[''], verbosity=2, exit=False)
