#!/usr/bin/python

import csv
import logging
import MySQLdb as mdb
import os
import settings
import re
from itertools import islice

def main():

    logging.basicConfig(filename='data/turk_data/insert.log',level=logging.WARNING, format='%(asctime)s %(message)s')

    ## Establish DB connection
    try:
        con = mdb.connect('localhost', 'arenold', settings.mysql_pass, 'arenold')
        con.set_character_set('utf8')
        print('Established db con')
    except Exception as e:
        print('Failed to get db con')
        print(e)


    # get each batch file from the completed directory
    data_dir = 'data/turk_data/completed/'
    files = []
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        files.extend(filenames)

    # read each file and insert the batch
    for f in files:
        print 'attemping to insert', f
        with open(data_dir + f, 'rU') as infile: 
            f_csv = csv.reader(infile)
            insertBatchResults(f, f_csv, con)

def insertBatchResults(filename, csv_file, db_con):
    """
        Insert each of the three links and ratings from the csv row
    """
    for row in islice(csv_file,1,None):
        link1, link2, link3, rating1, rating2, rating3 = row[27:33]
        insertLinkRating(filename, link1, rating1, db_con)
        insertLinkRating(filename, link2, rating2, db_con)
        insertLinkRating(filename, link3, rating3, db_con)

def insertLinkRating(filename, link, rating, db_con):
    """
        Look up a link in the database and insert the turk rating
        into an empty turk rating column
    """
    link_id = link[link.find('#')+1:]
    link = link[:link.find('#')]

    with db_con:
        cur = db_con.cursor(mdb.cursors.DictCursor)
        cur.execute("SELECT turk_rating0, turk_rating1, turk_rating2, turk_rating3, turk_rating4 FROM RSS WHERE id='{}';".format(link_id))
        res = cur.fetchone()

        for column, current_rating in res.items():
            if current_rating == '':
                update = "UPDATE RSS SET {0}='{1}' WHERE id={2};".format(column, rating, link_id)
                print update
                cur.execute(update)
                logging.warning(','.join([filename, link_id, rating, column]))
                break

if __name__ == '__main__':
    main()
