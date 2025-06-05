'''
This is the basic wrapper of the fftlog, without considering the requirement Eq19 or imporvement Eq28
The original code: https://github.com/minaskar/hankl/tree/master
The theory: https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf
'''

import numpy as np
from scipy.special import gamma



def preprocess(x,f,ext=0,range=None):
    ext_left=ext_right = ext
    x, f, N_left, N_right = padding(
        x, f, ext_left=ext_left, ext_right=ext_right, n_ext=0
    )
    return x, f, N_left, N_right


def padding(x, f, ext_left=0, ext_right=0, n_ext=0):
    N = x.size
    if N < 2:
        raise ValueError("Size of input arrays needs to be larger than 2")
    N_prime = 2 ** ((N - 1).bit_length() + n_ext)
    if N_prime > N:
        raise ValueError(" 2 ** ((N - 1).bit_length() + n_ext)>N")
    else:
        return x,f,0,0

def _gamma_term(mu, x, cutoff=200.0):
    # \Gamma[(\mu + 1 + x) / 2] / \Gamma[(\mu + 1 - x) / 2] eq16
    
    g_m=np.zeros(x.size,dtype=complex)
    alpha_plus= (mu+1.0+x)/2.0
    alpha_minus= (mu+1.0-x)/2.0
    g_m = gamma(alpha_plus)/gamma(alpha_minus)
    
    return g_m


def _u_m_term(m, mu, q, xy, L, cutoff=200.0):
    # u_{m}(\mu, q) = (x_{0}y_{0})^{-2\pi i m / L} U_{\mu}(q + 2\pi i m/ L) eq18
    
    omega = 1j * 2 * np.pi * m / float(L)
    x = q+omega
    
    U_mu = 2 ** x * _gamma_term(mu, x, cutoff)
        
    u_m = (xy) ** (-omega) * U_mu  # Eq18
    
    u_m[m.size - 1] = np.real(u_m[m.size - 1])  # Eq19????????
    
    return u_m



def FFTLog(x, f_x, q, mu, xy=1.0, lowring=False, ext=0, range=None, return_ext=False, stirling_cutoff=200.0):
    # f(y)= \int_0^\infty F(x) (xy)^{q} J_\mu(xy) y dx
    
    if mu + 1.0 + q == 0.0:
        raise ValueError("The FFTLog Hankel Transform is singular when mu + 1 + q = 0.")

    '''In practie we want to preprocess the data to imporve the precision'''
    x, f_x, N_left, N_right = preprocess(x, f_x, ext=ext, range=range)

    '''divide a uniform grid in log space; assume the middle point of the input x is the center of period '''
    N = f_x.size
    delta_L = (np.log(np.max(x)) - np.log(np.min(x))) / float(N - 1)
    L=np.log(np.max(x)) - np.log(np.min(x))
    log_x0 = np.log(x[N // 2])
    x0 = np.exp(log_x0)

    # do the fft for the input to get cm, see eq13
    c_m = np.fft.rfft(f_x)
    # get the mode index
    m = np.fft.rfftfreq(N, d=1.0) * float(N)  
    
    if lowring:
        raise ValueError("no lowering")
        
    y0 = xy / x0
    log_y0 = np.log(y0)

    '''define the uniform grid for y in log space'''
    indy = np.arange(-N // 2, N // 2)
    indy_shift = np.fft.fftshift(indy)
    logy_arr = delta_L * (-indy) + log_y0
    logy_arr=logy_arr[indy_shift] # transform to numpy.fft's convention

    y = np.exp(logy_arr)#10 ** (s[id] / np.log(10))

    u_m = _u_m_term(m, mu, q, xy, L, stirling_cutoff)
    b = c_m * u_m

    A_m = np.fft.irfft(b)
    
    f_y = A_m[indy_shift]

    f_y = f_y[::-1]
    y = y[::-1]

    if q != 0:
        f_y = f_y * (y) ** (-float(q))

    return y, f_y

