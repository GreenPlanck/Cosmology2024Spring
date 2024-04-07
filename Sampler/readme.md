# MCMC chanllenge

After you finish your sampler code, test it!

## toy model
We test with a toy model 
$$
y=mx
$$

while our obersvation data is contained by noise.
$$
\hat y=mx+N
$$
The noise follows a normal distribution with a known variance;
$$
N\sim\mathcal N(\mu=0,\sigma=2)
$$
The distribution of observed $\hat y$ is thus known: a guassian distribution around the truth theory $\hat y\sim \mathcal N(y,\sigma)$.

The likelihood is then
$$
\mathcal L=\frac{1}{\sqrt{2\pi}\sigma}\exp[{-\frac{1}{2}}\left(\frac{\hat y-y}{\sigma^2}\right)^2]
$$


It can be easily extended to multiple parameters.  Use the same data but   a new theory
$$
y=mx+b
$$
And finally assume no knowledge about the noise variance!

> **Overall**
>
> Theory1: $y=mx$​ with known covariance 
>
> Theory2: $y=mx$+b with known covariance 
>
> Theory3: $y=mx$+b with unknown covariance 

## Cosmology test

To apply it on cosmology, you need an engine  for cosmo quantity calculating. The case we are working needs Hubble parameter and angular distance at some redshift. It also needs sound horizon at drag time but can be set priorly with CMB knowledge. A lazy choice is to use CAMB/CLASS, some public Cosmology Boltzmann Solver.

### Data Description

The data is in "bao_data.txt" which is originally included in montepython. 

It includes following obersvations
$$
D_v/r_s, D_A/r_s, c/Hr_s, r_s/D_v
$$
$D_v$​ is the dilation scale defined as
$$
D_v(z)=\left[(1+z)^2D_A^2\frac{cz}{H(z)}\right]^{1/3}
$$
$r_s$ is the coming sound horizon scale at drag time. It can be set as
$$
r_s=149.92 \rm{[Mpc]}
$$
and $D_A$ is the angular distance.

The data file contain data and error, thus the likelihood is easily written as
$$
\mathcal L=\prod_i\mathcal L_i
$$


### strategy

To make your code easily extended to more complicated case, you may consider following structure

> **MCMC**: this class needs a parameter tuple $\theta$, which contains all the free parameters you want to sample; it also needs a class "likelihood" with which it can calculate the "like/$\chi^2$" on some $\theta$​
>
> **Likelihood**: This class needs two things: the data read from file; the cosmology engine which can give a observable; it will then compare the theory and observations and return the $\chi^2$
>
> **Theory:** This class needs parameter tuple $\theta$, and calculate the obersvations on given paramerters. This block can then be feed to the **liklihood** block 



In my case, I focus on this $\Lambda$CDM theory
$$
H^2=H_0^2(\Omega_c(1+z)^2+\Omega_b(1+z)^3+1-\Omega_m)
$$
with three free parameters $H_0,\Omega_m,\Omega_b$​. New model can be easily included.

> **Overall**
>
> 1. Sample your theory
> 2. Compare with some public MCMC code, e.g. montepython/emcee

