from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer

c_dbl_1d = ndpointer(dtype=np.float64,ndim=1,flags="C")

s1 = np.loadtxt(open('s1_out_elev.csv','rb'),delimiter=',',dtype=np.float64)
s2 = np.loadtxt(open('s2_out_elev.csv','rb'),delimiter=',',dtype=np.float64)

indata = np.vstack((s1,s2))

itmdll = cdll.LoadLibrary('./ITM.so')
itmdll.ITMDLLVersion.restype = c_double
dbloss = c_double()
strmode = (c_char*100)()
errno = c_int(0)
itmdll.point_to_point.argtypes = [c_dbl_1d, c_double, c_double, \
                                  c_double, c_double, c_double, \
                                  c_double, c_int, c_int, c_double, c_double, \
                                  POINTER(c_double), c_char*100,POINTER(c_int)]
for x in indata:
    elevs = x[9:]
    elevs = np.insert(elevs,0,x[7] / len(x[9:]),axis=0)
    elevs = np.insert(elevs,0,len(x[9:])-1,axis=0)
    res = itmdll.point_to_point(elevs,10.7,3.0,15.1,0.005,301.0,3500.0,5,1,0.5,0.5,dbloss,strmode,errno)
    print(dbloss.value,',',str(strmode.value).replace(",","-"),',',errno.value,',',elevs[1],',',elevs[0])
