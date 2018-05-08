import numpy as np
import pandas as pd

"""
							Question 5 Computation
							----------------------

 final row : 	Weather 	Temp		Humidity		Windy			Play
 				Sunny 		Cool		High			False			?


probability of Play being 'Yes',

 			P('Yes' | 'Sunny', 'Cool', 'High', 'False') 

 				= [ p('Yes' | 'Sunny') * p('Yes' | 'Cool') * p('Yes' | 'High')  * p('Yes' | 'False') * p('Yes') ] /
 					[ p('Sunny') * p('Cool') * p('High') * p('False') ]


		Bayes' theorem: 
 						P(A | B) = P(B | A) * P(A) / P(B)

"""

def probability(column, value, dataFrame):
	""" P (value) =  number of rows where column == value / total number of rows """
	n_df = df.loc[df[column] == value]
	return n_df.shape[0] / df.shape[0]


def prior_a_given_b(a_column, a_value, b_column, b_value, df):
	""" 
	P (a_value | b_value) = 
		number of rows where a_column is a_value and b_column is b_value 
			/ number of rows where b_column is b_value
	"""

	# select only rows where a_column == a_value
	a_df = df.loc[df[a_column] == a_value]

	# find percentage of those where b_column == b_value (P(A | B))
	a_and_b_df = a_df.loc[a_df[b_column] == b_value]
	return a_and_b_df.shape[0] / a_df.shape[0]

# read question 5 table
df = pd.read_csv("table_1.csv", header=0, dtype=str)
#df.to_csv("test.csv", sep='&', index=False, line_terminator='\\\\\n') write for latex

T = 'No'

p00 = prior_a_given_b('Play', T, 'Weather', 'Sunny', df)
p01 = prior_a_given_b('Play', T, 'Temp', 'Cool', df)
p02 = prior_a_given_b('Play', T, 'Humidity', 'High', df)
p03 = prior_a_given_b('Play', T, 'Windy', 'FALSE', df)

p10 = probability('Weather', 'Sunny', df)
p11 = probability('Temp', 'Cool', df)
p12 = probability('Humidity', 'High', df)
p13 = probability('Windy', 'FALSE', df)

p_play = (p00 * p01 * p02 * p03 * probability('Play', T, df)) / (p10 * p11 * p12 * p13)

#
# display results
#

# result
s = "P(vPlay|vWeather ... ∩ vWindy) = %s \n" % p_play

#
# display work
#

# numerator
s += "P(vPlay) = %s \n" % probability('Play', T, df)

# denominator
s += "P(vWeather) = %s \n" % p10
s += "P(vTemp) = %s \n" % p11
s += "P(vHumidity) = %s \n" % p12
s += "P(vWindy) = %s \n" % p13

# numerator
s += "P(vWeather|vPlay) = %s \n" % p00
s += "P(vTemp|vPlay) = %s \n" % p01
s += "P(vHumidity|vPlay) = %s \n" % p02
s += "P(vWindy|vPlay) = %s \n" % p03

print(s)

