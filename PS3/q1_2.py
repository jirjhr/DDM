from astropy.table import Table 
from sklearn.neighbors import KernelDensity
import numpy as np
from matplotlib import pyplot as plt
from astroML.plotting import hist
from sklearn.cross_validation import KFold
from scipy import integrate

def plot_histogram(data, binning_style, label, ax=None, xmin=10, xmax=17):
    """
    Convenience routine for plotting a histogram.
    """

    if ax is not None:
        ax = plt.axes(ax)

    counts, bins, patches = hist(data, bins=binning_style, ax=ax,
                                 color='k', histtype='step', normed=True)
    ax.text(0.95, 0.93, '{0:s}:\n{1:d} bins'.format(label, len(counts)),
            transform=ax.transAxes, ha='right', va='top')

    # Turn off y-axis labeling.
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_xlim(xmin, xmax)

    return ax

def cv1(x, bws, model='gaussian', plot=False, n_folds=10):
    """
    This calculates the leave-one-out cross validation. If you set 
    plot to True, then it will show a big grid of the test and training
    samples with the KDE chosen at each step. You might need to modify the 
    code if you want a nicer layout :)
    """

    # Get the number of bandwidths to check and the number of objects
    N_bw = len(bws)
    N = len(x)
    cv_1 = np.zeros(N_bw)
    
    # If plotting is requested, set up the plot region
    if plot:
        fig, axes = plt.subplots(N_bw, np.ceil(N/n_folds), figsize=(15, 8))
        xplot = np.linspace(-3, 8, 1000)

    # Loop over each band-width and calculate the probability of the 
    # test set for this band-width
    for i, bw in enumerate(bws):
    
        # I will do N-fold CV here. This divides X into N_folds
        kf = KFold(N, n_folds=n_folds)

        # Initiate - lnP will contain the log likelihood of the test sets
        # and i_k is a counter for the folds that is used for plotting and
        # nothing else..
        lnP = 0.0
        i_k = 0
                                 
        # Loop over each fold
        for train, test in kf:
            x_train = x[train, :]
            x_test = x[test, :]
            
            # Create the kernel density model for this bandwidth and fit
            # to the training set.
            kde = KernelDensity(kernel=model, bandwidth=bw).fit(x_train)
                                 
            # score evaluates the log likelihood of a dataset given the fitted KDE.
            log_prob = kde.score(x_test)
            
            if plot:
                # Show the tries
                ax = axes[i][i_k]

                # Note that the test sample is hard to see here.
                hist(x_train, bins=10, ax=ax, color='red')
                hist(x_test, bins=10, ax=ax, color='blue')
                ax.plot(xplot, np.exp(kde.score_samples(xplot[:, np.newaxis])))
                i_k += 1
            

            lnP += log_prob
            
        # Calculate the average likelihood          
        cv_1[i] = lnP/N
        
    return cv_1

def q1a(MBH):
	"""Plot KDE for range of bandwidths"""
	fig = plt.figure(figsize=(20,20))
	for bwidth in range(1,8): # 1...8
		# Plot each KDE
		ax = fig.add_subplot(4,2, bwidth)

		kde = KernelDensity(bwidth, kernel='gaussian').fit(MBH[:][:,None])
		Xgrid = np.linspace(0, 50, 100)[:, np.newaxis] #<same shape as X: N samples x N features >
		log_dens = kde.score_samples(Xgrid)
	
		ax.plot(Xgrid, np.exp(log_dens))
		axh = plot_histogram(MBH, 'knuth', 'BH', ax=plt.gca(), xmin=0, xmax=50)
		axh.set_xlabel('BH number')
		ax.set_title('bandwidth: {0}'.format(bwidth))

	plt.show()

def q1b(MBH):
	"""Use cross-validation to determine optimal KDE bandwidth"""
	bws = np.linspace(1,7,50)
	cv = cv1(MBH, bws)
	plt.plot(bws, np.exp(cv))
	plt.xlabel('bandwidth')
	plt.ylabel('CV likelihood')
	plt.text(3,.005, 'Best BW={0:.4f}'.format(bws[np.argmax(cv)]))
	plt.show()

def naive_prob(mass):
	n_greater = 0.0
	for mi in mass: 
		if mi > 1.8:
			n_greater += 1
	p = n_greater/len(mass)
	print "Naive probability of m > 1.8 solar mass: {0}".format(p)


def q2():
	t = Table().read('pulsar_masses.vot')

	mass = t['Mass']
	sigma = 0.5 * (t['dMlow'] + t['dMhigh']) # Assume equally distributed uncertainties for simplicity
	 
	naive_prob(mass)

	bws = np.linspace(.01,1,50)
	cv = cv1(mass[:,None], bws)
	best_bw = bws[np.argmax(cv)]

	# Assume Gaussian mass distribution
	# (and no systematic errors in neutron star detections in order to use this sample for kde)
	kde = KernelDensity(best_bw, kernel='gaussian').fit(mass[:,None]) 
	Xgrid = np.linspace(0, 3, 100)[:, np.newaxis] 
	log_dens = kde.score_samples(Xgrid)

	prob_func = lambda mi: np.exp(kde.score(mi))
	prob = lambda lower, upper: integrate.quad(prob_func, lower, upper) #prob[1] is int. error
	print "Likelihood of M > 1.8 solar mass: {0:1.3f}".format(prob(1.8, np.inf)[0])

	prob1 = prob(1.36, 2.26)[0]
	prob2 = prob(0.86, 1.36)[0]
	probbin = prob1 * prob2 # Assume mass distributions of stars in binary are independent...
	print "Likelihood of M in [1.36, 2.26]: {0:1.3f}".format(prob1)
	print "Likelihood of M in [0.86, 1.36]: {0:1.3f}".format(prob2)
	print "Likelihood of binary: {0:1.3f}".format(probbin)

	avg_m = np.average(kde.sample(5)) 
	print "\nPrediction for average mass of next 5 detections: {0:1.3f} solar masses".format(avg_m)


if __name__ == '__main__':
	t = Table().read('joint-bh-mass-table.csv')
	MBH = t['MBH']
	q1a(MBH)	
	q1b(MBH[:, None])
	q2()
