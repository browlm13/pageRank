"""
Question 3
naive bayes
"""
from functools import reduce

class_names = ['TRUCK', 'SUV', 'SEDAN']

probabilities = {
	'TRUCK' : {
			'P' : 0.35, 
			'f1' : 0.2,
			'f2' : 0.01,
			'f3' : 0.1,
			'f4' : 0.001,
			'f5' : 0.005
		},
	'SUV' : {
			'P' : 0.4, 
			'f1' : 0.01,
			'f2' : 0.1,
			'f3' : 0.001,
			'f4' : 0.2,
			'f5' : 0.008
		},
	'SEDAN' : {
			'P' : 0.25, 
			'f1' : 0.2,
			'f2' : 0.05,
			'f3' : 0.005,
			'f4' : 0.005,
			'f5' : 0.01
		}
}


def propto_prob_class_given_features(class_name, features_list):
	""" c_i * f_0 * ... * f_n, for each feature probability in class column for f_i in features list"""

	global probabilities

	cp = [probabilities[class_name]['P']]
	fgcs = [probabilities[class_name][f] for f in features_list]

	return reduce(lambda x, y: x*y, cp + fgcs)


def find_class(features_list):
	global class_names

	c_pcs = []
	for c in class_names:
		pc = propto_prob_class_given_features(c, features_list)
		c_pcs.append((pc, c))

	# find maximum prob
	c_pcs.sort()
	print('\nFor features %s,\nlikley class is %s' % (str(features_list), c_pcs[-1][1]))


find_class(['f1', 'f2', 'f3'])
find_class(['f1', 'f2', 'f4', 'f5'])