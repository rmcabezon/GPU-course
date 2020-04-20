
import numpy as np
import matplotlib.pyplot as plt


def generateCluster( n , means , sds ):
	''' 
	draw <n> points 
	from a multidimensional normal distribution 
	of means <means> (list)
	and standard deviation <sds> (list)

	the point are returned as a 'number of dimensions'*<n> array
	'''

	P = np.random.randn( len(means) , n )
	for i in range(len(means)):
		P[i,] = P[i,] * sds[i] + means[i]
	return P


def plotClusters( points , color_assignment , symbol_assignment , dimX=0 , dimY=1 ):
	"""
	plots a scatter plot with the defined color and symbol assignment.
	<points> is a 'number of dimensions'*<n> array
	<dimX> : index of the dimension that will be used as X (default 0)
	<dimY> : index of the dimension that will be used as Y (default 1)

	OK up until 22 symbol assignments
	"""
	symbols = ["o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|",]
	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	for i, c in enumerate(np.unique(symbol_assignment)):
		mask = symbol_assignment == c
		plt.scatter(points[dimX,:][mask],
					points[dimY,:][mask],
					c=[ colors[x] for x in color_assignment[mask]], marker=symbols[c])



def computeDistanceToCentroid( points , centroid ):
	""" computes the squared euclidian distance between a set of <points> and a <centroid>.
		<points> is a 'number of dimensions'*n array
		<centroid> is a 1*'number of dimensions' array

		the distances are returned as a 1*n array


	doctest strings :

	>>> p = np.array([[ 0,  0, 1, 1],[ 0,  1, 0, 1]])
	>>> c = np.array([ 0.5, 0.5])
	>>> computeDistanceToCentroid(p,c)
	array([[0.5, 0.5, 0.5, 0.5]])

	""" 
	distances = np.zeros([1,points.shape[1]])
	for i in range( points.shape[0] ):
		distances += np.power( points[i,] - centroid[i] , 2 )
	return distances

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
	closestCentroid = np.apply_along_axis(np.argmin , 0 , distances )
	
	return closestCentroid

def computeCentroids( points , assignments , k ):
	""" computes the centroids for <points> with a given <assignment>
		<points> is a 'number of dimensions'*n array
		<assignements> is a 1*n array containing indexes from 0 to <k>
		

		Centroids are returned as a list of 'number of dimensions'*1 arrays

	doctest strings :

	>>> p = np.array([[ 0,  0, 1, 1],[ 0,  1, 0, 1]])
	>>> a = np.array([0, 0, 1, 1])
	>>> computeCentroids( p , a , 2 )
	[array([0. , 0.5]), array([1. , 0.5])]
	"""
	nbDim = points.shape[0]
	nbPoint = points.shape[1]

	centroids = [ np.zeros(nbDim) for i in range(k) ]
	clusterSize = [ 0 ] * k

	##summing all values
	for i in range(nbPoint) :
		cluster  = assignments[i]
		centroids[ cluster ] +=  points[:,i]
		clusterSize[ cluster ] += 1
	
	##dividng by cluster size
	for i in range(k):
		centroids[ i ] /= clusterSize[ i ]

	return centroids

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

def Kmeans( points , k , maxNbRounds=1000 ):
	"""
	<points> is a 'number of dimensions'*n array
	<k> : number of clusters
	<maxNbRounds> : maximum number of Kmean round to perform

	Returns a cluster assignment : as a 1*n array containing indexes from 0 to k

	"""
	nbPoints = points.shape[1]

	#1. initialization 
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

		print("round {}, {:.2%} points changed assignment".format(round,nbChanged/nbPoints))
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
	clusterSizes = [2000,1000,2000,2000,1000 ]
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
	plotClusters( Points , kmeanAssignment , realAssignment , dimX=0 , dimY=1 )
	plt.show()


	### sk learn version
	from sklearn.cluster import KMeans as scKmeans
	X = Points.T ## scikit learn expects another shpae for the points
	t0 = time.time()
	kmeans = scKmeans(n_clusters=k, random_state=0).fit(X)
	t1 = time.time()
	print('scikit-learn :',t1-t0,'seconds')


