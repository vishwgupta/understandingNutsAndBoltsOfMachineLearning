import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import math
from operator import itemgetter

#Node of the K-D tree
class Node :
    left = None,
    right =  None,
    split_dim  =0
    split = None

##Function returns split dimension.
def getDimension(X, depth,k,split_dimension):
	##splitting in round robin fashion
    if(split_dimension == "dim"):
        return depth%k
    else: 
		#according to "maxVar"
        maxVar= 0
        d = 0
        for i in range(0,k):
            var = np.std([row[i] for row in X])
            if var>maxVar:
                maxVar = var
                d= i
        return d

#function returns splitting point
def splitData(X, depth,k,splitpoint, d):
    X_sorted = sorted(X,key=itemgetter(d))
    
    i = 0;
    if(splitpoint=="midpoint"):
	    #mid point is assumed to be the average of max and min values in the array
        midpoint = (X_sorted[0][d]+X_sorted[len(X_sorted)-1][d])/2.0 
        for item in [row[d] for row in X_sorted]:
            if(item>=midpoint):
                break
            i+=1
    else:
        i= len(X_sorted)//2

    index = 0;
    
    return X_sorted[i],X_sorted[:i],X_sorted[i+1:]




def createKDTree(data,k,split_dimension="dim",splitpoint="midpoint"):
    maxdepth = [0]
    #function returns Tree
    def kDtree(X,depth,k,split_dimension,splitpoint):
        if len(X) == 0:#if there is no element left, then do nothing
            return None
        if(depth > maxdepth[0]):
            maxdepth[0] = depth
        if len(X) == 1: #If the length is 1 then return the lead node
            node = Node()
            node.split = X[0]
            node.split_dim = -1
            node.left = None
            node.right = None
            return node

        #entries are more than one, then split it further
        d = getDimension(X, depth, k,split_dimension)
        i,X_l,X_r = splitData(X, depth,k,splitpoint, d)
        node = Node()
        node.split = i
        node.split_dim = d
        node.left = kDtree(X_l,depth+1,k,split_dimension,splitpoint)
        node.right = kDtree(X_r,depth+1,k,split_dimension,splitpoint)
        return node

    root = kDtree(data,0,k,split_dimension,splitpoint)
    return root,maxdepth[0]

def searchNode(point, root):
    nearest = [float('inf'),None]#store a tupke containing distance and the node
    def recursiveSearch(root):
        if(root == None):
            return
        #print "point={},split={}".format(point,root.split)
        dist = (point[0]-root.split[0])**2+(point[1]-root.split[1])**2
        bestSqrrdDist,nearestNode = nearest
        if(bestSqrrdDist>dist):
            nearest[:] = dist,root


        if root.split_dim !=-1 :#search recursively on non-leaf node
            target, other = (root.left, root.right) if root.split[root.split_dim]>point[root.split_dim] else (root.right, root.left)
            recursiveSearch(target)
            if(bestSqrrdDist > (root.split[root.split_dim]-point[root.split_dim])**2): #search on the other node if best node is not found
                recursiveSearch(other)

    recursiveSearch(root)
    return nearest


def plotKDTree(root,minArr,maxArr):
    if root is None:
        return
    
    if(root.split_dim == 0):
        ys = np.arange(minArr[1], maxArr[1])
        xs = [root.split[root.split_dim]]*len(ys)
        plt.plot(xs,ys,'-')
        plotKDTree(root.left,minArr,[root.split[root.split_dim],maxArr[1]])
        plotKDTree(root.right,[root.split[root.split_dim],minArr[1]],maxArr)

    elif root.split_dim == 1:
        xs = np.arange(minArr[0], maxArr[0])
        ys = [root.split[root.split_dim]]*len(xs)
        plt.plot(xs,ys,'-')
        plotKDTree(root.left,minArr,[maxArr[0],root.split[root.split_dim]])
        plotKDTree(root.right,[minArr[0],root.split[root.split_dim]],maxArr)
    plt.draw()

if __name__ == "__main__":
    # read data as 2D array of data type 'object'
    data = np.loadtxt('data2-train.dat',dtype=np.object,comments='#',delimiter=None)

    XY = data[:,0:3].astype(np.float)

    # read data as 2D array of data type 'object'
    data_test = np.loadtxt('data2-test.dat',dtype=np.object,comments='#',delimiter=None)

    XY_test = data_test[:,0:3].astype(np.float)

    k = 2

    x = XY[:,0]
    y = XY[:,1]

    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)

    plt.figure(0)
    plt.scatter([xy[0] for xy in XY if xy[2]>0],[xy[1] for xy in XY if xy[2]>0],s=15, marker='o', facecolors='none', edgecolors='b',label='train')
    plt.scatter([xy[0] for xy in XY if xy[2]<0],[xy[1] for xy in XY if xy[2]<0],s=15, marker='o', facecolors='none', edgecolors='r')
    # plt.plot(x, y, 'x')
    plt.ion()
    plt.draw()

    root,maxdepth = createKDTree(XY, k, split_dimension="dim",splitpoint="median")
    plotKDTree(root,[xmin,ymin],[xmax,ymax])

    timetaken = time.clock()
    for i in range(len(XY_test)):
        bestDist, nearestnode = searchNode(XY_test[i],root)
        XY_test[i][2] = nearestnode.split[2] #update the XY_tests class with predicted values (dirty way. need to improve)

    timetaken = time.clock()-timetaken
    plt.scatter([xy[0] for xy in XY_test if xy[2]>0],[xy[1] for xy in XY_test if xy[2]>0],s=15, marker='v', facecolors='none', edgecolors='b',label='test')
    plt.scatter([xy[0] for xy in XY_test if xy[2]<0],[xy[1] for xy in XY_test if xy[2]<0],s=15, marker='v', facecolors='none', edgecolors='r')

    print "Average time taken to traverse KD tree= {}, depth={}".format(timetaken/len(XY_test),maxdepth)
    plt.legend( loc='upper left', numpoints = 1 )

    plt.figure(1)
    plt.scatter([xy[0] for xy in XY if xy[2]>0],[xy[1] for xy in XY if xy[2]>0],s=15, marker='o', facecolors='none', edgecolors='b',label='train')
    plt.scatter([xy[0] for xy in XY if xy[2]<0],[xy[1] for xy in XY if xy[2]<0],s=15, marker='o', facecolors='none', edgecolors='r')
    plt.draw()
    root,maxdepth = createKDTree(XY, k, split_dimension="dim",splitpoint="midpoint")
    plotKDTree(root,[xmin,ymin],[xmax,ymax])

    timetaken = time.clock()
    for i in range(len(XY_test)):
        bestDist, nearestnode = searchNode(XY_test[i],root)
        XY_test[i][2] = nearestnode.split[2] #update the XY_tests class with predicted values (dirty way. need to improve)

    timetaken = time.clock()-timetaken
    plt.scatter([xy[0] for xy in XY_test if xy[2]>0],[xy[1] for xy in XY_test if xy[2]>0],s=15, marker='v', facecolors='none', edgecolors='b',label='test')
    plt.scatter([xy[0] for xy in XY_test if xy[2]<0],[xy[1] for xy in XY_test if xy[2]<0],s=15, marker='v', facecolors='none', edgecolors='r')
    print "Average time taken to traverse KD tree= {}, depth={}".format(timetaken/len(XY_test),maxdepth)
    plt.legend( loc='upper left', numpoints = 1 )

    plt.figure(2)
    plt.scatter([xy[0] for xy in XY if xy[2]>0],[xy[1] for xy in XY if xy[2]>0],s=15, marker='o', facecolors='none', edgecolors='b',label='train')
    plt.scatter([xy[0] for xy in XY if xy[2]<0],[xy[1] for xy in XY if xy[2]<0],s=15, marker='o', facecolors='none', edgecolors='r')
    plt.draw()
    root,maxdepth = createKDTree(XY, k, split_dimension="maxVar",splitpoint="median")
    plotKDTree(root,[xmin,ymin],[xmax,ymax])

    timetaken = time.clock()
    for i in range(len(XY_test)):
        bestDist, nearestnode = searchNode(XY_test[i],root)
        XY_test[i][2] = nearestnode.split[2] #update the XY_tests class with predicted values (dirty way. need to improve)

    timetaken = time.clock()-timetaken
    plt.scatter([xy[0] for xy in XY_test if xy[2]>0],[xy[1] for xy in XY_test if xy[2]>0],s=15, marker='v', facecolors='none', edgecolors='b',label='test')
    plt.scatter([xy[0] for xy in XY_test if xy[2]<0],[xy[1] for xy in XY_test if xy[2]<0],s=15, marker='v', facecolors='none', edgecolors='r')
    print "Average time taken to traverse KD tree= {}, depth={}".format(timetaken/len(XY_test),maxdepth)
    plt.legend( loc='upper left', numpoints = 1 )

    plt.figure(3)
    plt.scatter([xy[0] for xy in XY if xy[2]>0],[xy[1] for xy in XY if xy[2]>0],s=15, marker='o', facecolors='none', edgecolors='b',label='train')
    plt.scatter([xy[0] for xy in XY if xy[2]<0],[xy[1] for xy in XY if xy[2]<0],s=15, marker='o', facecolors='none', edgecolors='r')

    plt.draw()
    root,maxdepth = createKDTree(XY, k, split_dimension="maxVar",splitpoint="midpoint")
    plotKDTree(root,[xmin,ymin],[xmax,ymax])

    timetaken = time.clock()
    for i in range(len(XY_test)):
        bestDist, nearestnode = searchNode(XY_test[i],root)
        XY_test[i][2] = nearestnode.split[2] #update the XY_tests class with predicted values (dirty way. need to improve)

    timetaken = time.clock()-timetaken
    plt.scatter([xy[0] for xy in XY_test if xy[2]>0],[xy[1] for xy in XY_test if xy[2]>0],s=15, marker='v', facecolors='none', edgecolors='b',label='test')
    plt.scatter([xy[0] for xy in XY_test if xy[2]<0],[xy[1] for xy in XY_test if xy[2]<0],s=15, marker='v', facecolors='none', edgecolors='r')

    print "Average time taken to traverse KD tree= {}, depth={}".format(timetaken/len(XY_test),maxdepth)


    plt.ioff()

    plt.legend( loc='upper left', numpoints = 1 )

    plt.show()
