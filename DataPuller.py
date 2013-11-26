#!/usr/bin/python

import MySQLdb as mdb
import pandas as pd
import settings
import dataSplitter as dataSplitter
import logging


def main():    
    dp = DataPuller()
    rawDat = dp.fetchData()
    ds = dataSplitter.dataSplitter()
    valList = ds.validate(rawDat)
    splitDict = ds.split(valList)
    for n in ['train','test']:
        ds.write(n,splitDict)
    
    
class DataPuller:
    def __init__(self):
        isDataPuller = True
    
    def fetchData(self):
        try:
            con = mdb.connect('localhost', 'arenold', settings.mysql_pass, 'arenold')
            con.set_character_set('utf8')
            logging.warning('Established db con')
        except Exception as e:
            logging.warning('Failed to get db con')
            logging.exception(e)
        df = pd.io.sql.read_frame("SELECT * FROM RSS", con, \
                                  index_col=None, coerce_float=True, params=None)
        con.close()
        return df
        


if __name__ == '__main__':
    main()