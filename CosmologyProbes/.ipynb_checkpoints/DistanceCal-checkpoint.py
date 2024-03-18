from fitting import *
import os
import math as m
import statistics as s
from scipy import stats
import scipy.linalg as la
import numpy as np
from scipy.integrate import quad

#######################################################################################
#CLASS DEFINITIONS OF STEPS OF THE DISTANCE LADDER#
#######################################################################################

# This class defines the structure in which Anchors with known distances 
# and contain cepheids are imported

class Anchor:
	"This class defines a structure which holds data for objects that have geometric distance measurements, and cepheid variables."

	def __init__(self,Name='',Dist=0,dDist=0):

		self.Name = Name
		self.Dist = Dist
		self.dDist = dDist
		self.mu = 5*np.log10(self.Dist)-5
		self.dmu = 5*np.log10(np.exp(1))*self.dDist/self.Dist

	#computes the absolute magnitude of cepheids in the anchor galaxy, assuming cepheid data is available
	def Compute_abs_ceph_mag(self,period,dperiod,mh,dmh):
		r = 0
		tol = 10**-15
		n = 20
		[B_0, B_1, sigma_B_0, sigma_B_1, b_save, B_0_simple, B_1_simple] = york_fit(period,mh,dperiod,dmh,r,tol,n)

		self.Mceph = B_0 - self.mu
		self.dMceph = np.sqrt(np.power(sigma_B_0,2)+np.power(self.dmu,2))



# This class defines the structure in which the SHOES cepheid data is 
# imported, and the DM calculated to those cepheids

class SHOES_ceph_data:
	"This class creates a structure which holds SHOES cepheid data for a given host"

	def __init__(self,Host='',ID='',Period=0,V=0,dV=0,I=0,dI=0,NIR=0,dNIR=0,OH=0):

		self.Host = Host
		self.ID = ID
		self.Period = Period
		self.NIR = NIR
		self.dNIR = dNIR
		self.V = V
		self.dV = dV
		self.I = I
		self.dI = dI
		# R value is a correlation coeffecient, forced to be the same 
		#as what is used in previous analysis
		self.R = 0.386
		self.mh = NIR - self.R*(V-I)
		self.dmh = np.sqrt(np.power(dNIR,2)+self.R*np.power(dV,2)+self.R*np.power(dI,2))

	# Using the absolute magnitude of cepheids, the distance modulus to SH0ES cepheids is calculated.
	def proto_Compute_mu(self,mh,dmh,period,dperiod,Mceph,dMceph,slope,dslope):

		[B0, B_1, dB0, dB1] = weighted_fixed_slope(period,mh,slope,dslope,dperiod,dmh)

		self.mu = B0 - Mceph
		self.dmu = np.sqrt(np.power(dB0,2)+np.power(dMceph,2))

# This class defines the structure in which the LMC cepheid data is imported, and the DM calculated to those cepheids

class LMC_ceph_data:
	"This class creates a structure which holds cepheid data in the LMC"

	def __init__(self,Host='',ID='',Period=0,mh=0,dmh=0):
        
		self.Host = Host
		self.ID = ID
		self.Period = Period
		self.mh = mh
		self.dmh = dmh
        
	# Using the absolute magnitude of cepheids, the distance modulus to LMC cepheids is calculated.
	def proto_Compute_mu(self,mh,dmh,period,dperiod,Mceph,dMceph,slope,dslope):
        
		[B0, B_1, dB0, dB1] = weighted_fixed_slope(period,mh,slope,dslope,dperiod,dmh)

		self.mu = B0 - Mceph
		self.dmu = np.sqrt(np.power(dB0,2)+np.power(dMceph,2))



# This class defines the structure in which the SN which are nearby and calibrated are imported

class Local_SN_data:
	"This class creates a structure which holds SN data for a given host."

	def __init__(self,Host='',ID='',m=0,dm=0):

		self.Host = Host
		self.ID = ID
		self.m = m
		self.dm = dm

	def Compute_abs_sn_mag(self,m,dm,mu,dmu):

		x = mu
		y = m
		xerror = dm
		yerror = dmu
		[Msn, B1, dMsn, sigma_B1] = weighted_fixed_slope(x,y,1,0,xerror,yerror)
		self.Msn = Msn
		self.dMsn = dMsn

# This class defines the structure in which hubble flow SN are imported

class Hubble_SN_data:
	"This class creates a structure which holds SN data in the hubble flow."

	def __init__(self,ID='',z_hel = 0,z_cmb=0,dz=0,m=0,dm=0):

		self.ID = ID
		self.m = m
		self.dm = dm
		self.z_hel = z_hel
		self.z_cmb = z_cmb
		self.dz = dz

	def Compute_hubble_mu(self,m,dm,Msn,dMsn):
		#self.mu = hubble_sn.m - Msn
		#self.dmu = np.sqrt(np.power(hubble_sn.dm,2)+np.power(dMsn,2))
		self.mu = m - Msn
		self.dmu = np.sqrt(np.power(dm,2)+np.power(dMsn,2))





