##!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys
import cPickle
import emcee
import scipy.optimize as op
import corner

def pickle_from_file(fname):
	"""Restore a variable saved with pickle_to_file"""
	try:
		fh = open(fname, 'r')
		data = cPickle.load(fh)
		fh.close()
	except:
		#raise
		print "Loading pickled data failed!", sys.exc_info()[0]
		data = None
 
	return data

def lnL(theta, x, y, yerr):
	"""Calculate log likelihood; lec2p73"""
	a, b = theta
	model = b * x + a
	inv_sigma2 = 1.0/(yerr**2)
	return -0.5*(np.sum((y-model)**2*inv_sigma2))

def neglnL(theta, x, y, yerr):
	"""Just - lnL: Negative log likelihood; lec2p68"""
	a, b = theta
	model = b * x + a
	inv_sigma2 = 1.0/(yerr**2)

	return 0.5*(np.sum((y-model)**2*inv_sigma2))

def lnprior(theta): 
	"""Prior for line fitting; lec2p74"""
	a, b = theta
	if -5.0 < a < 5.0 and -10.0 < b < 10.0:
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr):
	""" The likelihood to include in the MCMC; lec2p75 """
  
	lp = lnprior(theta) 
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnL(theta, x, y, yerr)

def func_line(x, a, b):
	return a + b*x

def plot_initial_pos(pos, p_init):
	"""Show set of starting positions for each walker"""
  
	plt.figure(figsize=(8,8))
	for posi in pos:
		plt.scatter(posi[0], posi[1])
	
	plt.xlabel('a', fontsize=16)
	plt.ylabel('b', fontsize=16)
	plt.xlim(p_init[0]-0.0005,p_init[0]+0.0005)
	plt.ylim(p_init[1]-0.0005,p_init[1]+0.0005)
	plt.show()

def plot_chain(chain):
	"""Plot chain behaviour - note burn-in; lec2p84"""

	labels = ['a', 'b']
	plt.figure(figsize=(20,6))
	for i_dim in range(2):
		plt.subplot(2,1,i_dim+1)
		plt.ylabel(labels[i_dim])

		for i in range(100):
			plt.plot(chain[i,:,i_dim],color='black', alpha=0.5)
   
	plt.show()

def q3():
	"""Answer q3: Fit straight line using emcee process"""
  
	d = pickle_from_file('points_example1.pkl')
	x, y_obs, sigma = d['x'], d['y'], d['sigma']

	# Model a line for the data, y = a*x + b
	pars, cov = op.curve_fit(func_line, x, y_obs) 

	# Maximum likelihood for starting positions for Bayesian fitting
	result = op.minimize(neglnL, [1.0, 0.0], args=(x, y_obs, sigma))
	a_ml, b_ml = result["x"]
	p_init = np.array([a_ml, b_ml])
	print "Starting point for MCMC: a = {0}, b = {1}".format(a_ml, b_ml)


	# Set up the properties of the problem.
	ndim, nwalkers = 2, 100
	# Set up a number of initial positions.
	pos = [p_init + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

	plot_initial_pos(pos, p_init)

	# Create the sampler.
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y_obs, sigma))
   
	print 'Running MCMC process...'
	sampler.run_mcmc(pos, 500)
	print 'Done!'

	# Look at chain behaviour
	chain = sampler.chain
	plot_chain(chain)

	# Show result
	samples = sampler.chain[:, 50:, :].reshape((-1, 2))

	fig = corner.corner(samples, labels=["$a$", "$b$"], truths=[0.0, 1.3], quantiles=[0.16, 0.84])
	fig.show()
	fig.savefig('corner_res.jpg')
  

if __name__ == '__main__':
	q3()
