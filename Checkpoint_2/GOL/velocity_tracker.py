"""
Simple python program to read in CSV file with two or three col and do a liner fit.
The first two cols are the x/y data and the third (if given) is the error on y.
"""


import math
import matplotlib.pyplot as plt     # Import the plot package as plt
import numpy as np
import csvfile as f                 # Local CSV reader
from scipy.optimize import curve_fit


def linear(x,a,b):
    """
    Define function to fit, (linear)
    """
    return a*x + b

def main():
    
    data = np.genfromtxt("Checkpoint_2/GOL/output-one.dat") # Read in data to array

    y =data[:,1]
  
    x = np.arange(len(y))                              # set x/y to first two cols
    
  

    # Do a fit using curve_fit, note retuns two lists.
    popt,pcov = curve_fit(linear,x,y)
    perr = np.sqrt(np.diag(pcov))        # Errors

    #    Print out the results, popt contains fitted vales and perr the errors
    print("a is : {0:10.5e} +/- {1:10.5e}".format(popt[0],perr[0]))
    print("b is : {0:10.5e} +/- {1:10.5e}".format(popt[1],perr[1]))

    #            Upper plot of linear fit
    plt.subplot(2,1,1)
    plt.errorbar(x,y,fmt='.')     # Plot the data
    #      Plot the line
    plt.plot(x,linear(x,*popt),"r")

    #       Label the graph
    plt.title("Linear Fit a(speed): {0:8.4g} b: {1:8.4g}".format(popt[0],popt[1]))
    plt.xlabel("x value")
    plt.ylabel("y value")
    
    #       Lower plot of residuals
    plt.subplot(2,1,2)
    plt.errorbar(x,linear(x,*popt) - y,fmt = "rx")
    plt.xlabel("x value")
    plt.ylabel("y residual")
    plt.show()

main()
