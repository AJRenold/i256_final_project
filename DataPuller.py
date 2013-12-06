#!/usr/bin/python

import MySQLdb as mdb
import pandas as pd
import settings
import dataSplitter as dataSplitter
import logging


def main():    
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
    
    
    def validate(self,df,runReport=True):
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
        #valid = df.ix[keepList]
        #return(valid.reset_index(drop=True))
        return keepList
        


if __name__ == '__main__':
    main()
