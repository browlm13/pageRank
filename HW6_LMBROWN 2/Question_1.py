import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
	Page Rank Algorithm 

"""

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


# Question 1

iters = 3
alpha = 0.15
vstart = np.array([0.19757929, 0.28155074, 0.52086996])

P = compute_P(alpha)
vend = converge(P, vstart, iters)

print(vend)


"""
#
#	visulize alpha's affect on eigenvector
#

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

# change color with alpha 0:Yellow, 1:red
#%% Create Color Map
colormap = plt.get_cmap("YlOrRd")

for i in range(0,iters):
	color = i/iters
	
	ax.scatter(M[i,0], M[i,1], M[i,2], c=colormap(color), marker='o') #, s=size)


ax.set_xlabel('page rank a')
ax.set_ylabel('page rank b')
ax.set_zlabel('page rank c')

plt.show()

fig.savefig('pageRank_color_alpha.png')
"""
