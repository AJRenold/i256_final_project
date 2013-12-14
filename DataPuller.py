#!/usr/bin/python

import MySQLdb as mdb
import pandas as pd
import dataSplitter as dataSplitter
import settings

def main():    
    """
        Running DataPuller.py generates a new training
        and test data set. This must be run after the 
        database is updated with new rows
    """
    dp = DataPuller()
    rawDat = dp.fetchData()
    valList = dp.validate(rawDat)
    ds = dataSplitter.dataSplitter()
    splitDict = ds.split(valList)
    for n in ['train','test']:
        ds.write(n,splitDict)

class DataPuller:

    def __init__(self):
        isDataPuller = True

    def fetchData(self):
        """
            Load a dataframe from our database
        """
        try:
            con = mdb.connect('localhost', 'arenold', settings.mysql_pass, 'arenold')
            con.set_character_set('utf8')
            print('Established db con')
        except Exception as e:
            print('Failed to get db con')
            print(e)
        df = pd.io.sql.read_frame("SELECT * FROM RSS", con, \
                                  index_col=None, coerce_float=True, params=None)
        con.close()
        return df

    def validate(self,df,runReport=True):
        """
            Inspect the report_content column of the dataframe and
            return a list of indices with valid data
        """
        series = df['reporter_content']
        probList = []
        keepList = []
        for i in range(len(series)):
            ent = series[i]
            # print ent
            if isinstance(ent, float) == True:
                probList.append(df['id'][i])
            elif ent == 'None':
                probList.append(df['id'][i])
            elif 'Thanks for Registering!' in ent:
                probList.append(df['id'][i])
            else:
                keepList.append(i)
        if runReport == True:
            if len(probList) > 0:
                print "\nATTENTION: Problem with 'reporter_content' cell in rows with following id's: "
                for p in probList:
                    print str(p)
                print('*'*40)

        return keepList

if __name__ == '__main__':
    main()
