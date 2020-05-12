from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
from Kmeans_serial import generateCluster, plotClusters, computeDistanceToCentroid, computeCentroids





@cuda.jit
def getClosestCentroidNaiveCuda( distances , m , closestCentroid):
    '''
        distances : (m,n) array where 
            m : number of centroids
            n : number of data points
        closestCentroid : 1,n array
    '''
    pos = cuda.grid(1)
    if pos < closestCentroid.size:
        x=1
        minVal=distances[0,pos]
        minInd=0
        while x < m:
            if distances[x,pos]<minVal:
                minVal=distances[x,pos]
                minInd=x
            x +=1            
        closestCentroid[pos] = minInd
    

def computeNearestCentroid( points , centroids ):
    """ computes the closest <centroid> for each point in <points>
        <points> is a 'number of dimensions'*n array
        <centroids> is a list of 1*'number of dimensions' array

        The closest are returned as a 1*n array
        that contains the index of the closest centroid


    doctest strings :

    >>> p = np.array([[ 0,  0, 1, 1],[ 0,  1, 0, 1]])
    >>> c = [ np.array([ -0.5, 0.5]) , np.array([ 1.5, 0.5]) ]
    >>> computeNearestCentroid(p,c)
    array([0, 0, 1, 1])

    """ 

    nbPoints = points.shape[1]
    nbCentroids = len(centroids)

    #1. computing distances 
    distances = np.empty([ nbCentroids , nbPoints ])
    for i in range( nbCentroids ):
        distances[i,] = computeDistanceToCentroid( points , centroids[i] )

    #2. finding the closest centroid for each point
 
    closestCentroid = np.empty((nbPoints,) , dtype=int)
 

    threadsperblock=1
    blockspergrid = (nbPoints + (threadsperblock - 1)) // threadsperblock

    getClosestCentroidNaiveCuda[blockspergrid, threadsperblock]( distances , nbCentroids , closestCentroid )
 
    #closestCentroid = np.apply_along_axis(np.argmin , 0 , distances )
    
    return closestCentroid

def KmeanRound( points , centroids ):
    """
    For each point, compute the nearest centroid.
    Then computes new centroids based on the assignment.
    

    <points> is a 'number of dimensions'*n array
    <centroids> is a list of 1*'number of dimensions' array

    Returns :
        * the new centroids : list of 'number of dimensions'*1 arrays
        * new assignment : as a 1*n array containing indexes from 0 to k

    """

    assignment = computeNearestCentroid( points , centroids )
    newCentroids = computeCentroids( points , assignment , len(centroids) )
    return newCentroids,assignment

def Kmeans( points , k , maxNbRounds=1000 , assignment = None):
    """
    <points> is a 'number of dimensions'*n array
    <k> : number of clusters
    <maxNbRounds> : maximum number of Kmean round to perform

    Returns a cluster assignment : as a 1*n array containing indexes from 0 to k

    """
    nbPoints = points.shape[1]

    #1. initialization 
    if assignment is None:
        #I use the random assignment here.
        assignment = np.random.randint(0,k,nbPoints)

    centroids = computeCentroids( points , assignment , k )
    round = 1

    while( round < maxNbRounds ):
        centroids,newAssignment = KmeanRound( points , centroids )

        nbChanged = np.sum( newAssignment != assignment )
        
        assignment = newAssignment
        
        if nbChanged == 0: # nothing has changed -> we have converged !
            break
        elif nbChanged == nbPoints:
            ## something fishy occurs -> redraw random points to allow convergence
            assignment = np.random.randint(0,k,nbPoints)
            centroids = computeCentroids( points , assignment , k )

        #print("round {}, {:.2%} points changed assignment".format(round,nbChanged/nbPoints))
        round+=1
        #plotClusters( points , assignment , assignment , dimX=0 , dimY=1 )
        #plt.show()


    return assignment


if __name__ == "__main__":
    import doctest
    test = doctest.testmod() 
    if test.failed>0:
        print('doctest failed some tests')
        print(test)
        exit(0)

    import time

    # generating data

    ## 3 clusters of points 
    clusterSizes = [10000,5000,7500,14000,12000 ]
    clusterMeans = [ [ 0 , -2 ] ,
                     [ 3 , 3 ] ,
                     [ -1 , 3 ], 
                     [-5, 0] , 
                     [5,-1] ]
    clusterSDs = [ [0.5,1] ,
                   [1,0.5] ,
                   [0.5,0.5],
                   [2,1] ,
                   [1,1] ]
    C = []
    A = []
    for i in range( len(clusterSizes) ):
        C.append( generateCluster( clusterSizes[i] , clusterMeans[i] , clusterSDs[i] ) )    
        A += [i]*clusterSizes[i]

    Points = np.concatenate( C , axis=1)

    realAssignment = np.array(A)


    # performing Kmean
    k=5
    t0 = time.time()
    kmeanAssignment = Kmeans( Points , k , maxNbRounds=1000 )
    t1 = time.time()
    # reporting.
    print( "finished in",t1-t0,'seconds' )

    ## plotting : color are the Kmean assignment, symbols are the real assignment
    # OK up until 22 real assignments
    #plotClusters( Points , kmeanAssignment , realAssignment , dimX=0 , dimY=1 )
    #plt.savefig('Kmeans_numba.cuda.result.png')

