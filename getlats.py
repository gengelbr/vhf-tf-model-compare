#!/usr/bin/python
from __future__ import print_function
import googlemaps
import csv
import statistics
import numpy as np
import geopy
from geopy.distance import vincenty
import math
import sys


def intersections(array):
    pass

if __name__ == '__main__':
    print(sys.argv)
    f = open(sys.argv[1],'r')
    incsv = csv.DictReader(f)
    out = open(sys.argv[2],'w+')
    outcsv = csv.writer(out)
    gmaps = googlemaps.Client(key='AIzaSyDaB27bqXlow_jVRPUInx-enuxxTpi8AbY')

    origin = geopy.Point(39.826275,-105.08205)

    header_row = ["Measured BPL Minus FSL",
                  "Measured BPL",
                  "Freespace Loss",
                  "Distance (Meters)",
                  "Std Elevations",
                  "Total Elevation Delta",
                  "Highest Point Delta",
                  "Lowest Point Delta",
                  "Total Path Delta",
                  "Begin Elevations"]

    outcsv.writerow(header_row)

    i = 0
    for row in incsv:
        i+=1
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        dist = [vincenty(origin,(lat,lon)).meters][0]
        freespace_loss = (20*math.log10(dist) + 20*math.log10(3500) - 27.55)

        #extract gmaps elevation data from path.
        res = gmaps.elevation_along_path([(39.826275,-105.08205),(lat,lon)],256)
        elevs = [float(pnt['elevation']) for pnt in res]

        stats = []
        stats += [float(row['VSA_BPL']) - freespace_loss]
        stats += [row['VSA_BPL']]
        stats += [str(freespace_loss)]
        stats += [dist]
        stats += [np.std(elevs)]
        stats += [max(elevs) - min(elevs)]
        stats += [max(elevs) - elevs[0]]
        stats += [min(elevs) - elevs[0]]
        stats += [elevs[0] - elevs[-1]]
        stats += elevs
        #print(stats)
        print("Row ",i)
        outcsv.writerow(stats)
    print(res)
