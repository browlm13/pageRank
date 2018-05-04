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
		},
	'f1' : 0.41,
	'f2' : 0.16,
	'f3' : 0.106,
	'f4' : 0.206,
	'f5' : 0.023
}



def prob_class_given_features(class_name, features_list):
	global probabilities

	cp = [probabilities[class_name]['P']]
	fps = [probabilities[f] for f in features_list]
	fgcs = [probabilities[class_name][f] for f in features_list]

	numerator =  reduce(lambda x, y: x*y, cp + fgcs)
	denominator = reduce(lambda x, y: x*y, fps)

	return numerator/denominator


def find_class(features_list):
	global class_names

	print("for features %s" % str(features_list))


	c_pcs = []
	for c in class_names:
		pc = prob_class_given_features(c, features_list)
		c_pcs.append((pc, c))
		print("probability of class %s is %s" % (c, pc))

	# find maximum prob
	c_pcs.sort()
	print('likley class for given features is %s' % c_pcs[-1][1])


find_class(['f1', 'f2', 'f3'])
find_class(['f1', 'f2', 'f4', 'f5'])