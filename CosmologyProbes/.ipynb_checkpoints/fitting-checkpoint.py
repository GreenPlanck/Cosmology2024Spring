import os
import math as m
import statistics as s
from scipy import stats
import scipy.linalg as la
import numpy as np
from scipy.integrate import quad

#defines a simple linear regression to be used later by the york linear regression
def lin_reg(x,y):

    #length of data set
	n = len(x)

    #summing independent variable
	x_sum = sum(x)
    #summing dependent variable
	y_sum = sum(y)

    #mean of independent variable
	x_mean = s.mean(x)

    #mean of dependent variable
	y_mean= s.mean(y)

    #sum of x squared
	x_sqr = []
	for i in range(len(x)):
		x_temp = x[i]**2
		x_sqr.append(x_temp)
	x_sqr_sum = sum(x_sqr)

    #sum of y squared
	y_sqr = []
	for i in range(len(y)):
		y_temp = y[i]**2
		y_sqr.append(y_temp)
	y_sqr_sum = sum(y_sqr)

    #sum of xy product
	xy_prod = []
	for i in range(len(y)):
		xy_temp = y[i]*x[i]
		xy_prod.append(xy_temp)
	xy_prod_sum = sum(xy_prod)

    #numerator and denominator of slope estimate
	S_xx = x_sqr_sum - (x_sum**2/n)
	S_xy = xy_prod_sum - (x_sum*y_sum/n)

    #slope estimate
	B_1 = S_xy/S_xx

    #intercept estimate
	B_0 = y_mean - B_1*x_mean

	return B_0, B_1

# linear regression using error as a weighting scheme, fixing slope and slope error
def weighted_fixed_slope(x,y,m,dm,dx,dy):
    #wx = 1/dx^2
    #wy = 1/dy^2
	w = 1/(dx**2+dy**2)
    
	ybar = sum(w*y)/sum(w)
	xbar = sum(w*x)/sum(w)
    
	B0 = ybar - m*xbar

	r = y - (B0 +m*x)
	wr = np.sqrt(w)*r
	n = len(x)

	SE_wr = np.sqrt(sum(wr**2)/(n-2))
	dB0 = np.sqrt((SE_wr**2/sum(w))+dm*xbar**2)
	B1 = m
	dB1 = dm

	return B0, B1, dB0, dB1

#york linear regression

# York correction to linear regression including error in both x and y
# https://aapt.scitation.org/doi/abs/10.1119/1.1632486

def york_fit(x,y,sigma_x,sigma_y,r,tol,n_max):
#make sure inputs are a numpy array
	#if the error is 0, replace with something very very small to
	# to prevent nan
	sigma_x[sigma_x == 0] = 10**-15
	sigma_y[sigma_y == 0] = 10**-15

#define an array which tracks the changes in slope, B_1
	b_hist = np.ones(n_max)

#1) choose an approximate initial value for the slope, b
	# -> simple linear regression
# B_0 is intercept, B_1 is slope from simple linear regression
	[B_0_simple, B_1_simple] = lin_reg(x,y)
	b_hist[0] = B_1_simple
	B_0 = B_0_simple
	B_1 = B_1_simple

#2) determine the weights omega of each point for both x and y
	# usually 1/sigma where sigma is the error associated with x and y
	# at the i'th point

	omega_x = 1/np.square(np.array(sigma_x))
	omega_y = 1/np.square(np.array(sigma_y))

#3) use these weights with B_1 and the correlation r (if any) to
	# evaluate W_i for each point
	alpha = np.sqrt(omega_x*omega_y)
#6) calculate B_1_new until the difference between B_1_new and B_1 is
	# less than the tolerance provided
	counter = 1
	while counter < n_max:
		W = (omega_x*omega_y)/(omega_x + (B_1**2)*omega_y - 2*B_1*r*alpha)
    
#4) use the observed points and W to calculate x_bar and y_bar from
    # from which U V and beta can be evaluated for each point
    
		x_bar = sum(W*x)/sum(W)
		y_bar = sum(W*y)/sum(W)
    
		U = x - x_bar
		V = y - y_bar
		beta = W*((U/omega_y)+(B_1*V/omega_x)-(B_1*U+V)*(r/alpha))
    
#5) use W U V and beta to calculate a new estimate of B_1

		B_1_new = sum(W*beta*V)/sum(W*beta*U)
		b_hist[counter] = B_1_new

		if(abs(B_1_new-B_1)< tol):
			B_1 = B_1_new
			break

		counter += 1
		B_1 = B_1_new

#7) using the final value of B_1, x_bar, y_bar, calculate B_0
    
	B_0 = y_bar - B_1*x_bar
    
#8) for each point x and y, calculate the adjusted values
	x_adj = x_bar + beta
	y_adj = y_bar + B_1*beta
    
#9) use x_adj and W to calc x_bar_adj and u abd v
	x_bar_adj = sum(W*x_adj)/sum(W)
    #y_bar_adj = sum(W*y_adj)/sum(W)
    
	u = x_adj - x_bar_adj
    #v = y_adj - y_bar_adj
    
#10) use W x_bar and u to calculate sigma_a and sigma_b
    
	var_b = 1/(sum(W*u**2))
	var_a = 1/sum(W)+(x_bar**2)*var_b
    
	sigma_a_new = np.sqrt(var_a)
	sigma_b_new = np.sqrt(var_b)

    
	return B_0, B_1, sigma_a_new, sigma_b_new, b_hist, B_0_simple, B_1_simple

