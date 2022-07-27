import pandas as pd
from pandas.io import sql
import re
import sqlalchemy as db

class PersistenceUtil():
    '''
    Provides database communication options.
    
    A database engine can be provided, otherwise an SQLite database will 
    be created.

    Args:
        engine: The database engine. Defaults to an SQLite database at 
            location `data/sqlite.db`.

    '''

    def __init__(self, engine=db.create_engine(f"sqlite:///data/sqlite.db")):
        self._engine = engine

    def _convert(self, dtype):
        '''Returns cast operation for a given SQL data type.

        This can be used as a switch-case instruction.

        Args:
            dtype: One of db.sql.sqltypes.FLOAT, db.sql.sqltypes.INTEGER, 
                db.sql.sqltypes.VARCHAR.

        Returns:
            type: Cast operation for matched data type or str() by default.
            
        '''
        return {
            db.sql.sqltypes.FLOAT: float,
            db.sql.sqltypes.INTEGER: int,
            db.sql.sqltypes.VARCHAR: str,
        }.get(dtype, str)
    
    def save_df_to_db(self, df, table, regex_column_name):
        '''Saves a pandas data frame to the database.

        Args:
            df: The data frame to be saved.
            table (str): The database table name. If existing in the database 
                it will be replaced.
            regex_column_name (str): Column name pattern.

        Returns:
            int or None: Number of rows affected by pandas.DataFrame.to_sql.
            
        '''
        df_sql = df.copy()
        # Set the column names, using the provided regex pattern.
        df_sql.columns = df_sql.columns.map(lambda c: self\
                                            ._sql_column(c, \
                                                         regex_column_name, \
                                                         "y(\d+)"))
        df_sql.index.names = ['X']

        return df_sql.to_sql(table, self.engine, if_exists="replace", \
                             index=True)
    
    def create_table(self, table, columns, drop_if_exists=False):
        '''Creates a table in the database.
        
        If the table exists it can be replaced by optional parameter
        `drop_if_exists`.

        Args:
            table (str): The database table name. If existing in the database 
                it will be replaced.
            columns (dict): Column names and types.
            drop_if_exists (bool): Whether to drop an existing table with the 
                same name. Defaults to false.

        Returns:
            str: The table name.
            
        '''
        # Get MetaData object.
        meta_data = db.MetaData()
        
        # Drop existing table.
        if (drop_if_exists):
            connection = self.engine.raw_connection()
            cursor = connection.cursor()
            command = "DROP TABLE IF EXISTS {};".format(table)
            cursor.execute(command)
            connection.commit()
            cursor.close()

        # Set up table creation script.
        tbl_obj = db.Table(table, meta_data, \
                           *(db.Column(column, \
                                       columns[column]) for column in columns))

        meta_data.create_all(self._engine)
        
        return table

    def _sql_column(self, original, replacement, pattern):
        '''Replaces a pattern in a given string with a provided 
        replacement using regex.

        Args:
            original: The string to check for the pattern.
            replacement: The replacement for any occurrence of the pattern.
            pattern: The pattern to search for.

        Returns:
            The modified string.
            
        '''
        return re.sub(pattern,replacement,original)
        
    def persist(self, table_name, data):
        '''Stores data into a database table.

        Args:
            table_name (str): The database table name. Table must exist.
            data: The data to be stored.

        Raises:
            NoSuchTableError: If the table does not exist.
            ValueError: If there is any incompatibility between data and
                table columns.
            
        '''
        # Get MetaData object.
        meta_data = db.MetaData()

        # This line will raise a NoSuchTableError if the table does not exist.
        table = db.Table(table_name, meta_data, autoload=True, \
                         autoload_with=self._engine)
        columns = table.c
        try:
            for c in columns:
                [self._convert(type(c.type))(d[c.name]) if d[c.name] \
                 else d[c.name] for d in data]
        except Exception as e:
            raise ValueError("Cannot convert data to column types in " + \
                             + "database table: {}".format(data))

        sql_query = db.insert(table)
        connection = self._engine.connect()

        # Execute the insert statement.
        result = connection.execute(sql_query, data)

        connection.close()
    
    def read_df(self, table_name):
        '''Returns data from a database table as pandas data frame.

        Args:
            table_name (str): The database table name. Table must exist.

        Returns:
            The table as data frame as per pd.read_sql_table.
        
        Raises:
            ValueError: If the table does not exist.
            
        '''
        meta_data = db.MetaData()
        connection = self.engine.connect()
        
        return pd.read_sql_table(table_name, self.engine)

    @property
    def engine(self):
        '''The database engine used by the instance.'''
        return self._engine

    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @engine.deleter
    def engine(self):
        del self._engine

