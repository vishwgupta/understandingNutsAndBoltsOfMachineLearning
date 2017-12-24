#conditional expectation for missing value prediction
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

def gaussian_2d(x, y, xmean, ymean, xsig, ysig,rho):
    return (1.0/(2.0*math.pi*xsig*ysig*(math.sqrt(1.0-rho**2)))) *\
        np.exp(-(0.5*(1-rho**2))*(((x-xmean)/xsig)**2+((y-ymean)/ysig)**2-(2.0*rho*(x-xmean)*(y-ymean)/(xsig*ysig))))


if __name__ == "__main__":
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
    y = np.array([d[0] for d in data if d[0]>0]) #weight
    x = np.array([d[1] for d in data if d[0]>0]) #height
    x_out = np.array([d[1] for d in data if d[0]<0]) #height
    mean_w = np.mean(y) #mean of  w
    mean_h = np.mean(x) #mean of h
    mean = np.hstack((mean_h,mean_w))

    count = len(x)

    Data = np.vstack((x,y))
    covariance = np.cov(Data)

    #find covariances
    sigma_h = math.sqrt(covariance[0][0])
    sigma_w = math.sqrt(covariance[1][1])
    rho = covariance[0][1]/(sigma_h*sigma_w)


    #height = input('Enter outlier height: ')
    height = x_out
    expectation = mean_w + ((rho * (sigma_w/sigma_h))*(height - mean_h))
    print expectation


    #Plot the contours
    x_ = sorted(np.random.normal(mean_h, sigma_h, 2000))
    y_ = sorted(np.random.normal(mean_w, sigma_w, 2000))
    X, Y = np.meshgrid(x_, y_)
    Z = gaussian_2d(X, Y, mean_h, mean_w, sigma_h, sigma_w,rho)
    CS = plt.contour(X,Y,Z)

    #scatter the given and expectations
    plt.scatter(x,y,s=9, facecolors='b', edgecolors='b',label='data')
    plt.scatter(height,expectation,s = 18, facecolors='red',edgecolors='black')

    plt.show()

