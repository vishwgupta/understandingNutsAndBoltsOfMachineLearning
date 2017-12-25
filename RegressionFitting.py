import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from numpy.linalg import inv
import numpy.matlib
dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

# read height, weight and gender information into 1D arrays
y = np.array([d[0] for d in data if d[0]>0])
hs = np.array([d[1] for d in data if d[0]>0])
sigma = 3
d = 5
one = np.ones(hs.shape)
X = np.c_[one]
N = len(X)

#CALCULATING PARAMETER W_mle
for j in range(1,d+1):
	X = np.c_[X,hs**j]

p_X = np.linalg.pinv(X)
w =np.dot(p_X,y)
print "w={}".format(w)
Y_ML = np.polyval(list(reversed(w)),hs)

#calculating w_map
sigma = 1.0/(1.0/len(y) *(sum((Y_ML-y)**2)))
sigma_zero = 3

transpose_X = X.transpose()
product_X = np.dot(transpose_X,X)
sigma_term = sigma/sigma_zero

identity_matrix = np.matlib.identity(d+1)

matrix_sigma =sigma_term *identity_matrix
sum = product_X + matrix_sigma
inverse_sum = inv(sum)
product = np.dot(inverse_sum , transpose_X)
w_map = np.dot(product,y)
w_map = w_map.transpose()


F = np.polyval(list(reversed(w_map)),hs)

print "Wmap={}".format(w_map)

plt.scatter(hs,y,s=9, facecolors='none', edgecolors='g',label='data')


#read the outliers
outliers_hs = np.array([d[1] for d in data if d[0]<0])
outliers_F = np.transpose(np.polyval(list(reversed(w_map)),outliers_hs))
outliers_F_MLE = np.transpose(np.polyval(list(reversed(w)),outliers_hs))
plt.scatter(outliers_hs,outliers_F,s=9, facecolors='red', edgecolors='r',label='Outlier with MAP')
plt.scatter(outliers_hs,outliers_F_MLE,s=9, facecolors='y', edgecolors='y',label='Outlier with MLE')

X_ = np.linspace(159.9, 186.1, 50)

F_ = np.transpose(np.polyval(list(reversed(w_map)),X_))

plt.plot(X_,F_)


F_MLE = np.transpose(np.polyval(list(reversed(w)),X_))
plt.legend( loc='upper left', numpoints = 1 )
plt.plot(X_,F_MLE)

plt.show()



