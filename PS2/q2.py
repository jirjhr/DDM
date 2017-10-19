import sqlite3 as lite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def q2(con):
	"""Calculate summary statistics for dataset, plot in 2D and in 1D [here only x is plotted]"""

	fig2d = plt.figure(figsize=(20,20)) 
	figh = plt.figure(figsize=(20,20))
	figv = plt.figure(figsize=(20,20))
	for number in range(1, 14): # 1...13
		query = """
		SELECT x,y FROM Set{0:02d}
		""".format(number)
		t = pd.read_sql_query(query, con)
		x, y = t['x'], t['y']
		print 'Set{0:02d}: mean(x) = {1:1.1f}, mean(y) = {2:1.1f},\n\
			var(x) = {3:1.1f}, var(y) = {4:1.1f}'.format(number,
			np.mean(x), np.mean(y), np.var(x), np.var(y))
		
		# Plot each set in 2D
		ax2d = fig2d.add_subplot(4,4, number)
		ax2d.scatter(x, y)
		ax2d.set_xlim(0,100)
		ax2d.set_ylim(-20,120)
		ax2d.set_xlabel('x', fontsize=14)
		ax2d.set_ylabel('y', fontsize=14)
		ax2d.set_title('Set {0:02d}'.format(number), fontsize=16)

		# Plot a histogram/kernel density estimate
		subhist = figh.add_subplot(4,4, number)
		sns.distplot(x, ax=subhist)
		subhist.set_title('Set {0:02d}'.format(number), fontsize=16)

		# Plot a violin plot
		subv = figv.add_subplot(4,4, number)
		sns.violinplot(data=x, bw=.2, cut=1)
		subv.set_ylim(0,100)
		subv.set_ylabel('x', fontsize=14)
		subv.set_title('Set {0:02d}'.format(number), fontsize=16)
	
	fig2d.savefig('all-sets-2d.jpg')
	figh.savefig('all-sets-hist-kd.jpg')
	figv.savefig('all-sets-violin.jpg')



if __name__ == '__main__':
	con = lite.connect('ThirteenDatasets.db')
 
	q2(con)
