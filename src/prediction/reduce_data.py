import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import datetime
date_parse = lambda t: pd.to_datetime(t, format="%m/%d/%Y %I:%M:%S %p")

def nulls(data):
    # n: Number of nulls
    n = data.isnull().sum()
    data.dropna(inplace = True)
    return n

def get_data(filename, columns, types):
    data = pd.read_csv(
        filename,
        parse_dates = ['Date'],
        date_parser = date_parse,
        usecols = columns,
        dtype = types
    )
    nls = nulls(data)
    data = data[data.Year > 2014]
    return data

def main():
    filename = '/Users/mariaknigge/Dropbox/CSCI/CSCI4502/ProjectData/Chicago_Crime.csv'
    destination = '/Users/mariaknigge/Dropbox/APPM/APPM4580/FinalProject/Chicago_Crime.csv'
    types = {
        'IUCR': 'category', 
        'Primary Type' : 'category', 
        'Description': 'category', 
        'Location Description': 'category',
        'Block': 'category',
        'Beat': 'category',
        'District': 'category',
        'Community Area': 'category',
    }
    columns = ['Date','Year', 'Arrest', 'Domestic', 'Latitude', 'Longitude'] + list(types.keys())
    data = get_data(filename, columns, types)
    data.to_csv(destination, index = False, header = True)

if __name__ == '__main__':
    main()

