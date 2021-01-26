from elasticsearch import helpers
import csv

def csv_load(es, filename):
    fp = 'data\\' + filename + '.csv'
    with open(fp, 'r', encoding="utf8") as outfile:
        reader = csv.DictReader(outfile)
        try:
            response = helpers.bulk(es, reader, index = "movieid", refresh = True)
            print("\nRESPONSE:", response)
        except Exception as e:
            print("\nERROR:", e)