import matplotlib.pyplot as plt

relevant_docs = ["d2", "d8", "d17", "d27", "d32", "d46", "d59", "d73", "d80", "d94"]
recalled_docs = ["d38", "d199", "d7", "d17", "d231", "d111", "d94", "d21", "d222", "d27", "d30", "d46", "d99", "d80", "d3", "d73", "d321", "d58", "d5", "d32"]

recal_indices = range(1, len(recalled_docs) + 1)
precision_scores = [1]
recal_scores = [0] 
for i in range(0, len(recalled_docs)):
	matches_i = 0
	for recalled_d in recalled_docs[:i+1]:
		if recalled_d in relevant_docs:
			matches_i += 1
	p_score_i = float(matches_i/len(recalled_docs[:i+1]))
	r_score_i = float(matches_i/len(relevant_docs))
	
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
plt.savefig('question_5.png')

# Mean Average Precision
map_score = sum(precision_scores)/len(precision_scores)
print("MAP: %f" % map_score)
