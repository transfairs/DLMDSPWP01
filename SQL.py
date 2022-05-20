import pandas as pd
from pandas.io import sql
import re
import sqlalchemy as db

__version__ = "1.0.0"

class PersistenceUtil():
    def __init__(self, engine=db.create_engine(f"sqlite:///data/sqlite.db")):
        self._engine = engine

    def df_column_name(self, y_string, replacement):
        return re.sub("y(\d)",replacement,y_string)

    def _convert(self, dtype):
        return {
            db.sql.sqltypes.FLOAT: float,
            db.sql.sqltypes.INTEGER: int,
            db.sql.sqltypes.VARCHAR: str,
        }.get(dtype, str)
    
    def save_df_to_db(self, df, table, regex_column_name):
        df_sql = df.copy()
        df_sql.columns = df_sql.columns.map(lambda c: self._sql_column(c, regex_column_name, "y(\d+)"))
        df_sql.index.names = ['X']

        return df_sql.to_sql(table, self.engine, if_exists="replace", index=True)
    
    def create_table(self, table, columns, drop_if_exists=False):
        # Get MetaData object
        meta_data = db.MetaData()
        
        if (drop_if_exists):
            connection = self.engine.raw_connection()
            cursor = connection.cursor()
            command = "DROP TABLE IF EXISTS {};".format(table)
            cursor.execute(command)
            connection.commit()
            cursor.close()

        # Set test data creation script table
        test_table = db.Table(table, meta_data, \
                              *(db.Column(column, columns[column]) for column in columns))

        meta_data.create_all(self._engine)
        
        return table

    def _sql_column(self, original, replacement, pattern):
        return re.sub(pattern,replacement,original)
        
    def persist(self, table_name, data):
        # Get MetaData object
        meta_data = db.MetaData()


        table = db.Table(table_name, meta_data, autoload=True, autoload_with=self._engine)
        columns = table.c
        #print("Python Type: ", [c.type.python_type for c in columns])
        try:
            for c in columns:
                [self._convert(type(c.type))(d[c.name]) for d in data]
        except Exception as e:
            raise ValueError("Cannot convert data to column types in database table: {}".format(data))

        sql_query = db.insert(table)
        connection = self._engine.connect()

        # execute the insert statement
        result = connection.execute(sql_query, data)

        connection.close()
    
    def read_df(self, table_name):
        meta_data = db.MetaData()
        connection = self.engine.connect()
        #table = db.Table(table_name, meta_data, autoload=True, autoload_with=self.engine)
        #query = db.select([table])
        #resultset = connection.execute(select_actor).fetchall()
        
        #return resultset
        
        return pd.read_sql_table(table_name, self.engine)

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, delta_y):
        self._engine = engine

    @engine.deleter
    def engine(self):
        del self._engine

