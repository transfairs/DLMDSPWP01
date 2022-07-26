import csv
import pandas as pd


class FileReader():
    '''
    Provides required CSV file read functionalities as static methods.
    
    '''

    @staticmethod
    def read_csv_to_df(file=None):
        '''Converts contents of a CSV file to pandas data frame.
        
        Args:
            file (str): The CSV file to be read.

        Returns:
            The created data frame or None if an error occured.
        
        '''
        df = None

        try:
            df = pd.read_csv(filepath_or_buffer=file)
            
            # All files will have the index column renamed to 'x'
            df.set_index('x', inplace=True)
        except FileNotFoundError as e:
            print("File not found: {}".format(file))
        except:
            print("An unknown error occured with file: {}.".format(file))

        return df
    
    @staticmethod
    def csv_line_gen(file=None):
        '''Creates a generator over the line by line content of a CSV file.
        
        Args:
            file (str): The CSV file to be read.
        
        Yields:
            CSV file content, line by line.
            
        '''
        if file:
            handler = open(file, "r")
            reader = csv.reader(handler)
            data_list = []
            line = None
            
            for i, line in enumerate(reader):
                yield line
                
            handler.close()