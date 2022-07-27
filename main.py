import FileReader as f
import Regression as r
import SQL as sql
import sqlalchemy as db
import Visualisation as visual
import Testing as test

'''
Documentation will follow Google Style syntax for docstrings.
'''

def main():
    '''
    This is the main program to be run.
    
    '''
    # Read in training and ideal data CSV files.
    df_train = f.FileReader.read_csv_to_df("data/train.csv")
    df_ideal = f.FileReader.read_csv_to_df("data/ideal.csv")
    
    # Define Placeholders.
    chosen_ideal, df_result, regression, test_data = None, None, None, None
    pu = sql.PersistenceUtil()

    # First part of the task:  Map training data to ideal functions.
    try:
        regression = r.Regression(df_ideal)
        df_train_mse = df_train.apply(regression.mse)
        df_result = regression.min_value(df_train_mse, df_train)

        # Persist both data sets in a database and adjust column names.
        pu.save_df_to_db(df_train, "train", r"Y\1 (training func)")
        pu.save_df_to_db(df_ideal, "ideal", r"Y\1 (ideal func)")

        # Save the four mapped ideal functions for later reference.
        chosen_ideal = df_ideal[df_result.min_y.tolist()]
    except Exception as e:
        print("{} raised: {}".format(type(e).__name__, e.__str__()))

    # Second part of the task:  Get test data set, match it to the chosen ideal
    # functions and save the results to a db table.
    try:
        test_data = regression.match("data/test.csv", df_result, chosen_ideal)
    except Exception as e:
        print("{} raised: {}".format(type(e).__name__, e.__str__()))

    test_table = pu.create_table("test", {"X (test func)": db.Float, \
                                          "Y (test func)": db.Float, \
                                          "Delta Y (test func)": db.Float, \
                                          "No. of ideal func": db.String(5)}, \
                                 drop_if_exists=True)
    try:
        pu.persist(test_table, test_data)
    except Exception as e:
        print("Could not persist data, continue anyway: {}"\
              .format(e.__str__()))

    # Print matched data for both tasks.
    try:
        visual.PlotFactory.show("Training Data", df_train, \
                                df_result, chosen_ideal)
        visual.PlotFactory.show("Test Data", test_data, df_result, \
                                chosen_ideal, vtype="test")
    except (TypeError, ValueError) as e:
        print("Could not plot data: {}".format(e.__str__()))

    print("Done")

if __name__ == '__main__':
    main()