import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
	Page Rank Algorithm 

"""


# eigenvector for alpha = 0
#e_alpha_0 = np.array([2/11, 3/11, 6/11])

def compute_P(alpha):
	# link matrix
	L = np.array([[ 0, 0.5 ,0.5], [ 0, 0 ,1], [ (1/3), (1/3) ,(1/3)]])

	t1 = np.array([1/3, 1/3, 1/3])
	t2 = np.array([1, 1, 1])
	R = np.outer(t1, t2)

	# probability transition matrix row stocastic
	P = np.multiply(L, 1-alpha) + np.multiply(R, alpha)

	return P


def converge(P, v0, iters):
	vi = v0
	
	for i in range(iters):
		vi = np.dot(P.T, vi)
	return vi



# alpha's affect on eigenvector
# num iterations
iters = 100

#
# columns: a, b, c - rows: alpha = row_index/iters
# 		ex: access alpha = .15 eigenvector - M[15]

M = np.zeros((iters, 3))
for i in range(0,100):
	alphai = i/100
	P = compute_P(alphai)
	v0 = np.array([1/3, 1/3, 1/3]) # uniform start
	eigenvectori = converge(P, v0, iters)
	M[i] = eigenvectori



# display alphas affet on eigenvector in 3d scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(M[:,0], M[:,1], M[:,2], c='r', marker='o')

# change color with alpha 0:Yellow, 1:red
#%% Create Color Map
colormap = plt.get_cmap("YlOrRd")

for i in range(0,iters):
	color = i/iters

	"""
	size = 2
	if i == 15:
		size=size*4
	"""
	
	ax.scatter(M[i,0], M[i,1], M[i,2], c=colormap(color), marker='o') #, s=size)


ax.set_xlabel('page rank a')
ax.set_ylabel('page rank b')
ax.set_zlabel('page rank c')

plt.show()


"""
# alpha's affect on eigenvector
for i in range(0,100):
	alphai = i/100
	P = compute_P(alphai)
	v0 = np.array([1/3, 1/3, 1/3]) # uniform start
	eigenvectori = converge(P, v0, iters)
	print("alpha = %s:" % alphai)
	print(eigenvectori)
	print("\n")
"""

"""
# alpha
alpha = 0.15
P = compute_P(alpha)


# eigen vector of alpha = 0.15
e = np.array([0.19757929, 0.28155074, 0.52086996])

# eigen vector of alpha = 0 calculated from flow equations
max_right_eigenvector_alpha_0 = np.array([2/11, 3/11, 6/11])
print(np.subtract(e,max_right_eigenvector_alpha_0))



# page rank vector / eigen vector
print("\n\nEigenvector alpha = 0:")
max_right_eigenvector_alpha_0 = np.array([2/11, 3/11, 6/11])
print(converge(P, max_right_eigenvector_alpha_0, iters))


#  starting vector page a
print("\n\n<1,0,0>")
a = np.array([1, 0, 0])
print(converge(P, a, iters))



# uniform starting vector
print("\n\n<1/3,1/3,1/3>")
u = np.array([1/3, 1/3, 1/3])
print(converge(P, u, iters))
"""