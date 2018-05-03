"""
KNN - 

2.	(25/25 points) Assume the following training set:
Food: “cherry pie”
Food: “buffalo wings”
Beverage: “cream soda”
Beverage: “orange soda”
Apply 3-nearest-neighbor (kNN) text categorization to the name “cherry soda”. 
Show all the similarity calculations needed to classify the name, 
and the final categorization. 
Assume simple term-frequency weights (no IDF) with cosine similarity.


				Question 2 Answer
				-----------------
	Answer:

					"Beverage"
"""

from collections import Counter

def distance(word_list1, word_list2):
	# every vector length is sqrt(2)
	common = len(set(word_list1) & set(word_list2))
	return common/2


def knn(x, k, training_set):

	# compute distances
	class_n_distance = []
	for item in training_set:
		distance_i = distance(x, item['words'])
		class_i = item['class']
		class_n_distance.append((distance_i, class_i))

	# get top k classes
	class_n_distance.sort(reverse=True)
	_, top_k_classes = zip(*class_n_distance[:k])

	# find mode class
	cnt = Counter(top_k_classes)
	return cnt.most_common(1)[0][0]

	

# given training set
training_set = [
	{'class':'Food', 'words': ['cherry', 'pie']},
	{'class':'Food', 'words': ['buffalo', 'wings']},
	{'class':'Beverage', 'words': ['cream', 'soda']},
	{'class':'Beverage', 'words': ['orange', 'soda']}
]

# to classify
x = ["cherry", "soda"]

# k
k = 3

# classify
xclass = knn(x,k,training_set)

# display
print(xclass)



