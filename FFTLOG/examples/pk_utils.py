import scipy as sp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline,CubicSpline

class ComplexPowerLawApprox_FFTlog:
    # this class contain the karray and rarray
    # just calculate the pk/xi at corresponding point
    def __init__(self):
        self.Nmax = 1024
        self.bk = -1.1001
        self.kmax = 100.

        self.rtab = np.zeros(self.Nmax)
        self.rmin = 0.001
        self.rmax = 10000.
        self.k0 = 1.e-4
        self.Delta = np.log(self.kmax/self.k0) / (self.Nmax - 1)
        self.Delta_r = np.log(self.rmax/self.rmin) / (self.Nmax - 1)

        self.k_arr= np.array([self.k0*np.exp(self.Delta*i) for i in range(self.Nmax)])
        self.r_arr = np.array([self.rmin*np.exp(self.Delta_r * i) for i in range(self.Nmax)])
        
        self.jsNm = np.arange(-self.Nmax//2,self.Nmax//2+1,1)
        self.etam = self.bk + 2*1j*np.pi*(self.jsNm)/(1.*self.Nmax)/self.Delta
        
        
        #xi2p
        self.bR=-2.001
        self.etamR = self.bR + 2*1j*np.pi*(self.jsNm)/self.Nmax/self.Delta_r
    
    @staticmethod
    def J0(r,nu):
        return -1.*np.sin(np.pi*nu/2.)*r**(-3.-1.*nu)*sp.special.gamma(2+nu)/(2.*np.pi**2.)
    @staticmethod
    def J2(r,nu):
        return -1.*r**(-3.-1.*nu)*(3.+nu)*sp.special.gamma(2.+nu)*np.sin(np.pi*nu/2.)/(nu*2.*np.pi**2.)
    
    def P2xi(self,pkarr,ell=0):
        #input: pk(k*h)*h**3
        Nmax=self.Nmax
        karr=self.k_arr
    
        Pdiscrin0 = np.zeros(Nmax)
        for i in range(Nmax):
            Pdiscrin0[i] = pkarr[i] * np.exp( -1.*(karr[i]/4.)**4. -1.*self.bk*i*self.Delta)

        cm = np.fft.fft(Pdiscrin0)/ Nmax
        cmsym = np.zeros(Nmax+1,dtype=np.complex_)

        for i in range(Nmax+1):
            if (i+2 - Nmax//2) < 1:
                cmsym[i] =  self.k0**(-self.etam[i])*np.conjugate(cm[-i + self.Nmax//2])
            else:
                cmsym[i] = self.k0**(-self.etam[i])* cm[i - self.Nmax//2]

        cmsym[-1] = cmsym[-1] / 2
        cmsym[0] = cmsym[0] / 2

        if ell==0:
            xi = np.real(np.matmul(cmsym,self.J0(self.r_arr.reshape(-1,1),self.etam).T))
        elif ell==2:
            xi = np.real(np.matmul(cmsym,self.J2(self.r_arr.reshape(-1,1),self.etam).T))
        else:
            raise ValueError("invalid ell")
        return xi
    
    
    def wiggle_split(self,wiggled_xi):
        xi_=np.array(wiggled_xi)
        s=self.r_arr
        index1=np.where((s<60))
        index2=np.where((s>200))
        index=np.append(index1,index2)
        _s = s[index]
        _xi = xi_[index]

        spline_r2xi = InterpolatedUnivariateSpline(_s,_s**2*_xi)
        nowiggle_r2xi = spline_r2xi(s)
        nowiggle_xi = nowiggle_r2xi/s**2
        
        return nowiggle_xi
    
    
    # Define inverse transform functions
    @staticmethod
    def J0k(k,nu):
        return -1.*k**(-3.-1.*nu)*sp.special.gamma(2+nu)*np.sin(np.pi*nu/2.)*(4.*np.pi)
    @staticmethod
    def J2k(k,nu):
        return -1.*k**(-3.-1.*nu)*(3.+nu)*sp.special.gamma(2.+nu)*np.sin(np.pi*nu/2.)*4.*np.pi/nu
    
    def xi2P(self,xiarr,ell=0):
        Nmax=self.Nmax
        i_range=np.arange(Nmax)
        Xidiscrin = xiarr*np.exp(-1.*self.bR*i_range*self.Delta_r)
        cmr = np.fft.fft(Xidiscrin)/ Nmax
        cmsymr = np.zeros(self.Nmax+1,dtype=np.complex_)
        
        for i in range(Nmax+1):
            if (i+2 - Nmax/2) < 1:
                cmsymr[i] =  self.rmin**(-self.etamR[i])*np.conjugate(cmr[-i + self.Nmax//2])
            else:
                cmsymr[i] = self.rmin**(-self.etamR[i])* cmr[i - self.Nmax//2]
                
        cmsymr[-1] = cmsymr[-1] / 2
        cmsymr[0] = cmsymr[0] / 2
        
        if ell==0:
            Pk = np.real(np.matmul(cmsymr,self.J0k(self.k_arr.reshape(-1,1),self.etamR).T))
        elif ell==2:
            Pk = np.real(np.matmul(cmsymr,self.J2k(self.k_arr.reshape(-1,1),self.etamR).T))
        else:
            raise ValueError("invalid ell")
        return Pk