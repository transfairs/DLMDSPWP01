import FileReader as f
import Regression as r
import SQL as sql
import sqlalchemy as db
import Visualisation as visual
import Testing as test

'''
Documentation will follow syntax of course book from page 141 ff.
'''

def main():
    '''
    This is the main program to be run.
    '''

    # Read in training and ideal data CSV files.
    df_train = f.FileReader.read_csv_to_df("data/train.csv")
    df_ideal = f.FileReader.read_csv_to_df("data/ideal.csv")

    # First part of the task:  Map training data to ideal functions.
    regression = r.Regression(df_ideal)
    df_train_mse = df_train.apply(regression.mse)
    df_result = regression.min_value(df_train_mse, df_train)

    # Persist both data sets in a database but adjust column names before that.
    pu = sql.PersistenceUtil()
    pu.save_df_to_db(df_train, "train", r"Y\1 (training func)")
    pu.save_df_to_db(df_ideal, "ideal", r"Y\1 (ideal func)")
    
    # Save the four mapped ideal functions in a new data frame.
    chosen_ideal = df_ideal[df_result.min_y.tolist()]

    # Second part of the task:  Get test data set, match it to the chosen ideal
    # functions and save the results to a db table.
    data = regression.match("data/test.csv", df_result, chosen_ideal)
    test_table = pu.create_table("test", {"X (test func)": db.Float, \
                                          "Y (test func)": db.Float, \
                                          "Delta Y (test func)": db.Float, \
                                          "No. of ideal func": db.String(5)}, \
                                 drop_if_exists=True)
    # Example for standard exception handling as requested.
    try:
        pu.persist(test_table, data)
    except Exception as e:
        print("Couldn't persist data, continue anyway: {}".format(e.__str__()))

    # Print matched data for both tasks.
    visual.PlotFactory.show("Training Data", df_train, df_result, chosen_ideal)
    visual.PlotFactory.show("Test Data", data, df_result, \
                            chosen_ideal, vtype="test")

    print("Done")

if __name__ == '__main__':
    main()
