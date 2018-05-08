"""
	Question 2
"""

# system contains 8 relavent documents
# recalled docs (+ : relavent): R = { -, +, +, +, -, +, -, -, +, -}
# a.	Create the Precision versus Recall Curve for q. 
# b.	What is the MAP value?

import matplotlib.pyplot as plt

num_relevant_docs = 8
recalled_docs = ['-', '+', '+', '+', '-', '+', '-', '-', '+', '-']

recal_indices = range(1, len(recalled_docs) + 1)
precision_scores = [1]
recal_scores = [0] 
for i in range(0, len(recalled_docs)):
	matches_i = 0
	for recalled_d in recalled_docs[:i+1]:
		if recalled_d == '+':
			matches_i += 1
	p_score_i = float(matches_i)/len(recalled_docs[:i+1])
	r_score_i = float(matches_i)/num_relevant_docs
	
	precision_scores.append(p_score_i)
	recal_scores.append(r_score_i)

# save plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(recal_scores, precision_scores, c='r', linewidth=4.0, zorder=0)
plt.scatter(recal_scores, precision_scores, c='b', s=15.0, zorder=10)
plt.ylim([0,1])
plt.xlim([0,1])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
plt.savefig('question_2.png')

# Mean Average Precision
map_score = sum(precision_scores)/len(precision_scores)
print("MAP: %f" % map_score)


