# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:29:23 2019

@author: Luciano Masullo, Lucía López and Lars Richter

Different tools used for MINFLUX exp analysis and simulations

"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import stats
from scipy.special import erf

alfa = 1.3
fwhm = 360 # fwhm doughnut and gaussian
wvlen = 640
#theta = 0
π = np.pi

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Plot of covariance ellipse
    
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width(w), height(h), rotation(theta in degrees):
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """
    
    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * stats.norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = stats.chi2.ppf(q, 2)
    print('r2', r2)
    
    val, vec =  LA.eig(cov)
    order = val.argsort()[::]
    val = val[order]
    vec = vec[order]
#    w, h = 2 * np.sqrt(val[:, None] * r2)  #TODO: check this r2 valuel
    w, h = 2 * np.sqrt(val[:, None])

    theta = np.degrees(np.arctan2(*vec[::, 0]))
    return w, h, theta

def spaceToIndex(space, size_nm, px_nm):

    # size and px have to be in nm
    index = np.zeros(2)
    index[0] = (size_nm/2 - space[1])/px_nm
    index[1] = (space[0]+ size_nm/2)/px_nm 

    return np.array(index, dtype=np.int)

def indexToSpace(index, size_nm, px_nm):

    space = np.zeros(2)
    space[0] = index[1]*px_nm - size_nm/2
    space[1] = size_nm/2 - index[0]*px_nm

    return np.array(space)


def gaussian(r, d=fwhm):
    
    """ 2D gaussian beam intensity """
    
    β = 1
    I = β * np.exp(-4 * np.log(2) * (r**2/d**2))
    
    return I

def doughnut(r, d=1.2*fwhm, P=1):
    
    """ 2D donut """


#    β = (4/π) * 2 * np.log(2) * (P/d**2)  # normalization to 1 (BUSCAR!!!)
    β = 2*np.e
    I = β * 2 * np.log(2) * (r**2/d**2) * np.exp(-4 * np.log(2) * (r**2/d**2))

    return I

def psf(central_zero, size, px, fov_center, d):
    """ 2D extension for a 1D function in (r,θ) """

    center = fov_center
    x0 = central_zero[0]
    y0 = central_zero[1]

    x = np.arange(-size/2 - center[0], size/2 - center[0], px)
    y = np.arange(-size/2 - center[1], size/2 - center[1], px)

    # CAMBIE Y POR -Y PARA NO TENER QUE HACER FLIPUD
    [Mx, My] = np.meshgrid(x, -y)
    # My = np.flipud(My)
    
    if d == 'donut':
        Mro = np.sqrt((Mx-x0)**2 + (My-y0)**2)
        result = doughnut(Mro)
    
    elif d == 'gaussian':
        Mro = np.sqrt((Mx-x0)**2 + (My-y0)**2)
        result = gaussian(Mro)
    
    else:
        result = SW(central_zero, size, wvlen, theta, fov_center)

    return result

def beams(K, L, center, d):
    """ beams center positions """
    # TODO: change all scripts to the ebp_centres function and delete this function
    
    pos_nm = {} # TODO: make pos_nm an array and not a dictionary
                # Warning: changing this will probably crash all of the other 
                # scripts that use this function like this at the moment

    if d is not None:
    
        if center is not None:
            θ = {}
            pos_nm = {}
            pos_nm[0] = np.array([0,0]) 
            # beams centers (for regular polygons incribed in a circle of diameter L)
            Kθ = K - 1
            if (Kθ+1) % 2 == 0:  # if K is even
                for k in np.arange(1,Kθ+1):
                    θ[k] = π * 2*k/Kθ
                    pos_nm[0] = np.array([0,0]) 
                    pos_nm[k]= (L/2)*np.array([np.cos(θ[k]), np.sin(θ[k])])
            else:       # if K is odd
                for k in np.arange(1,Kθ+1):
                    θ[k] = π * (2*k+1)/Kθ 
                    pos_nm[0] = np.array([0,0]) 
                    pos_nm[k]= (L/2)*np.array([np.cos(θ[k]), np.sin(θ[k])])
                    
        if center is None:
                        
            θ = {}
            pos_nm = {}
            if (K+1) % 2 == 0:  # if K is even
                for k in np.arange(K):
                    θ[k] = π * 2*k/K 
                    pos_nm[k]= (L/2)*np.array([np.cos(θ[k]), np.sin(θ[k])])
            else:       # if K is odd
                for k in np.arange(K):
                    θ[k] = π * (2*k+1)/K
                    pos_nm[k]= (L/2)*np.array([np.cos(θ[k]), np.sin(θ[k])])
                    
    return pos_nm   


def ebp_centres(K, L, center, phi=0):
    
    """   
    Calculates doughnut centre positions of 2D MINFLUX EBP.
    [Lars] This is the new version of the old beams() function.

    Input
    ----------
        K: number of distinct doughnuts, usually K=4
        L: EBP diameter
        center: boolean, true/false: EBP with or without centre 
    
    Returns
    -------
        pos_nm: EBP beam centre positions
       
    """
    
    pos_nm = np.zeros((K, 2)) 
    θ = np.zeros(K)
    L = [L, L, L, L]
#    θrandom = [π/10, π/12.3, -π/18, -π/9]
    
    if center:
        # beams centers (for regular polygons incribed in a circle of diameter L)
        pos_nm[0, :] = 0. # central doughnut at (0, 0)
        Kθ = K - 1
        if (Kθ+1) % 2 == 0:  # if K is odd
            for k in np.arange(1, Kθ+1):
                θ[k] = π * 2*k/Kθ + phi 
                pos_nm[k, 0] = (L[k]/2) * np.cos(θ[k])
                pos_nm[k, 1] = (L[k]/2) * np.sin(θ[k])
        else:       # if K is even
            for k in np.arange(1, Kθ+1):
                θ[k] = π * (2*k+1)/Kθ + phi 
                pos_nm[k, 0] = (L[k]/2) * np.cos(θ[k])
                pos_nm[k, 1] = (L[k]/2) * np.sin(θ[k])
    else:
        if (K+1) % 2 == 0:  # if K is odd
            for k in np.arange(K):
                θ[k] = π * 2*k/K + phi 
                pos_nm[k, 0] = (L[k]/2) * np.cos(θ[k])
                pos_nm[k, 1] = (L[k]/2) * np.sin(θ[k])
        else:       # if K is even
            for k in np.arange(K):
                θ[k] = π * (2*k+1)/K + phi 
                pos_nm[k, 0] = (L[k]/2) * np.cos(θ[k])
                pos_nm[k, 1] = (L[k]/2) * np.sin(θ[k])
                    
    return pos_nm  


def sim_exp(key, t_mask, psf, r0, SBR, Ns, Nb, M_p, Tlife, factor, dt, 
            cycle_time=125000, DEBUG=False):
    
    """
    
    This function is the core of all the simulations scripts as it generates
    a complete MINFLUX experiment (either cw or p) and estimates the position 
    of the emitter. It can include blinking dynamics.
    
    - key: str
    
        "p_minflux" or "cw_minflux" experiment
        
    - t_mask: np.array
    
        binary mask of the blinking dynamics
        
    - psf: np.array
    
        Stack of K PSFs of the minflux experiment
    
    - r0: np.array
    
        2D position of the emitter (x, y)
        
    - SBR: int
    
        Signal to background ratio
        
    - Ns: int
    
        Number of signal photons
        
    - Nb: int
    
        Number of background photons
    
    - M_p: int
    
        Number of cycles. Each cycle is a TCSPC cycle, typically 25 ns
        
    - T_life: float
    
        Lifetime of the emitter
        
    - factor: float
        
        This parameter can be confusing, it determines, how many more
        photons than the target one Ns we generate, it should be kept low 
        enough so that the number Nh doesn't cause too many photons in the 
        signal for a TCSPC measurement
        
    - dt: int 
        
        Macrotime resolution of the experiment, TCPSC cycle, typically 25 [ns].
        M_p x dt = total time of the experiment
        
    - cycle_time: int
    
        Duration of the cw-minflux cycle [ns]
        Typically takes values between 50000 - 200000
    
    """

    failed = 0 # this variable goes to 1 only if the simulation fails
    Nh = Ns * factor # Nh: N_higher, 
    K = np.shape(psf)[0] # number of expositions
    
    if key=='simplified':
        
        N = Ns+Nb
        
        λ = np.zeros(K)
        for i in np.arange(K):
            λ[i] = psf[i, r0[0], r0[1]] 
        
        # backgroung given a SBR level
        λb = np.sum(λ)/(K*SBR)
        normλ = (np.sum(λ) + K*λb)
        
        # probability for each beam
        p = np.zeros(K)
        for i in np.arange(K):
            p[i] = (λ[i] + λb)/ normλ
                        
        n_array = np.random.multinomial(N, p)
        
        return n_array
        
        
    if key=='cw_minflux':
    
        M = int(M_p / (cycle_time/dt)) # number of cw-minflux cycles
            
        λ = np.zeros(K)
        for i in np.arange(K):
            λ[i] = psf[i, r0[0], r0[1]] 
        
        normλ = M * np.sum(λ)
        
        # probability for each beam
        p = np.zeros(K)
        for i in np.arange(K):
            p[i] = λ[i] / normλ
            
        ct = int(cycle_time/(dt/K)) # time points in units of (dt/K) = 6.5 ns (for dt = 25 ns) 
        pp = np.ones((K, int(ct/K))) # probability array, has a total number of elements ct
        for i in np.arange(K):
            pp[i, :] = (pp[i, :] * p[i])/(ct/K) # (ct/K) is the normalization factor
            
        pp_sub = pp.flatten() # recreate a cw-minflux array (one cycle) with p-minflux exp time resolution
        pp_tot = np.tile(pp_sub, M)
    
        if DEBUG:
        
            plt.figure('Probability plot (two cycles)')
            tt = (np.arange(len(pp_tot)) * dt/K)/10**6 # aux time array [ms]
            
            plt.plot(tt[:ct], pp_tot[:ct] * M_p) # show prob normalized to one cycle
            plt.xlabel('Time [ms]')
            plt.ylabel('Probability')
            
    elif key=='p_minflux':
        
        #  λ for each beam
        λ = np.zeros(K)
        for i in np.arange(K):
            λ[i] = psf[i, r0[0], r0[1]] 

        normλ = M_p * np.sum(λ)
        
        # probability for each beam
        p = np.zeros(K)
        for i in np.arange(K):
            p[i] = λ[i] / normλ
            
        # array of probabilities
        pp_tot = np.tile(p, M_p)
        
    else:
        
        print('Wrong experiment key, please chose p_minflux or cw_minflux')
        
    # Use the blinking mask
    
    if t_mask is not None:
        
        try:
        
            t_mask = np.repeat(t_mask, K, axis=0)
            pp_tot = pp_tot * t_mask
            pp_tot = pp_tot / np.sum(pp_tot) # renormalization
        
        except(ValueError):
            
            print('Simulation failed because the blinking mask (T_mask) was not long enough')
            print('Try set a larger amount of nblinks to avoid this')
            
            failed = 1
            
            return None, None, failed
    
    else:
        pass
        # print('No blinking mask selected')
        
    photons = np.random.multinomial(Nh, pp_tot)
    
    # section to correct the array 'photons' such that it has exactly N elements (and not close to N)
    
    photons[photons>1] = 1 # corrects for double detection within one cycle/4
    
    # reshape to obtain array with M lin K col
    nk = np.reshape(photons, (M_p, K))
    
    absTbinary = nk.sum(axis=1) # trace with one event every cycle (of typically 25 ns)
    absTbinary[absTbinary>1] = 1 

    # photons for each illumination (sum over all cycles M)
    n = np.zeros(K)
    for i in np.arange(K):
        n[i] = np.sum(nk[:, i])

    # indexes of nonzero elements in nk (events)
    [m, k] = np.nonzero(nk)
    
    # correct to the total number of valid counts (see TCSPC, two photons same cycle)
    Ncorr = np.size(m)

    # micro time, time tag in each cycle
    Tmicro = np.zeros(M_p)
    # micro time of fluorescence events including fluorescence lifetime
    # Tmicro will be the experimental output
    Texp = dt/K
    for i in range(Ncorr):
        m1 = m[i]
        k1 = k[i]
        
        if key=='p_minflux':
    
            Tmicro[m1] = k1*Texp + np.random.exponential(Tlife)
            
        elif key=='cw_minflux':
            
            Tmicro[m1] = np.random.exponential(Tlife)
    
    # Remove extra photons to get exactly Ns counts  
    inadvance = len(Tmicro[Tmicro>0]) - Ns    
    nonzero = np.where(Tmicro>0)[0].flatten()
    
    if DEBUG:
    
        print('len(Tmicro[Tmicro>0])', len(Tmicro[Tmicro>0]))
        print('inadvance', inadvance)

    try:
    
        todelete = np.sort(np.random.choice(nonzero, inadvance, replace=False))
        
    except(ValueError):
        
        print('Simulation failed because there were either not enough photons or too many photons')
        print('Try another set of T_on, T_off or dutycycle to avoid this')
        
        failed = 1
        
        return None, None, failed

    Tmicro[todelete] = 0
    absTbinary[todelete] = 0

    # add uniform temporal bkg (draw from uniform dist)
    Tmicrob = np.random.uniform(0, dt, Nb)
    
    # add bkg counts in the trace only where there are no signal counts
    p_bkg = np.ones(M_p)
    p_bkg[absTbinary>0] = 0
    norm_bkg = np.sum(p_bkg)
    
    absTimeb = np.random.multinomial(Nb, p_bkg/(norm_bkg))
        
    # put together Tmicro, Tmicroextra and Tmicrob
    Tmicrot = np.concatenate((Tmicro, Tmicrob))
    absTimeBinary = absTbinary + absTimeb

    # if Tmicro falls outside add to first pulse
    Tmicrot[Tmicrot>dt] = Tmicrot[Tmicrot>dt]%dt
    
    if DEBUG:
        
        print('Tmicro[Tmicro>0]', len(Tmicro[Tmicro>0]))
        print('Tmicrob[Tmicrob>0]', len(Tmicrob[Tmicrob>0]))

    return  Tmicrot, absTimeBinary, failed


def crb_minflux(K, PSF, SBR, px_nm, size_nm, N, method='1'):
    
    """
    
    Cramer-Rao Bound for a given MINFLUX experiment 
    
    Input
    ----------
    K : int, number of excitation beams
    PSF : (K, size, size) array, experimental or simulated PSF 
    SBR : float, signal to background ratio
    px_nm : pixel of the grid in nm
    size_nm : size of the grid in nm
    N : total number of photons
    method: parameter for the chosen method
    
    There are three methods to calculate it. They should be equivalent but
    provide different outputs.
    
    Method 1: calculates the σ_CRB using the most analytical result 
    (S26, 10.1126/science.aak9913)
    
    Output 1: σ_CRB (size, size) array, mean of CRB eigenval in each position
    
    Method 2: calculates the Σ_CRB from the Fisher information matrix in 
    emitter position space (Fr), from there it calculates Σ_CRB and σ_CRB
    (S11-13, 10.1126/science.aak9913)
    
    Output 2: Fr, Σ_CRB, σ_CRB
    
    Method 3: calculates the Fisher information matrix in reduced probability
    space and calculates J jacobian transformation matrices. From there it
    calculates Fr, Σ_CRB, σ_CRB. Fp, Σ_CRB_p and σ_CRB_p are additional outputs
    (S8-10, 10.1126/science.aak9913)
    
    Output 3: Fr, Σ_CRB, σ_CRB, Fp, Σ_CRB_p, σ_CRB_p
    """
    
    # size of the σ_CRB matrix in px and dimension d=2
    size = int(size_nm/px_nm)
    d = 2
    
    # size of the (x,y) grid
    dx = px_nm
    dy = px_nm
    
    if method=='1':
                
        # define different arrays needed to compute CR
        
        p, λ, dpdx, dpdy, A, B, C, D = (np.zeros((K, size, size)) for i in range(8))
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))

        for i in range(K):
    
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
            
        # λb using the approximation in Balzarotti et al, (S29)
            
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(K*SBR)
        
        # probabilities in each (x,y)
        
        for i in np.arange(K):
            
            # probability arrays
    
            p[i, :, :] = (λ[i, :, :] + λb)/(K*λb + np.sum(λ, axis=0))
            
            # plot of p
            
            locx = (size/4) * np.sqrt(2)/2
            locy = (size/4) * np.sqrt(2)/2
            
            plt.figure(str(i))
            plt.plot(np.arange(-size/2, size/2), p[i, int(size/2 - locx), :], label='p x axis')
            plt.plot(np.arange(-size/2, size/2), p[i, ::-1, int(size/2 - locy)], label='p y axis')
                                    
            # gradient of ps in each (x,y)
            
            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
           
            # terms needed to compute CR bound in aeach (x,y)
            
            A[i, :, :] = (1/p[i, :, :]) * dpdx[i, :, :]**2
            B[i, :, :] = (1/p[i, :, :]) * dpdy[i, :, :]**2
            C[i, :, :] = (1/p[i, :, :]) *(dpdx[i, :, :] * dpdy[i, :, :])
            D[i, :, :] = (1/p[i, :, :]) * (dpdx[i, :, :]**2 + dpdy[i, :, :]**2)
    
        # sigma Cramer-Rao numerator and denominator    
        E = np.sum(D, axis=0) 
        F = (np.sum(A, axis=0) * np.sum(B, axis=0)) - np.sum(C, axis=0)**2
        
        σ_CRB = np.sqrt(1/(d*N))*np.sqrt(E/F)
            
        return σ_CRB
    
    if method=='2':
    
        # initialize different arrays needed to compute σ_CRB, Σ_CRB and Fr
        
        σ_CRB = np.zeros((size, size))
        p, λ, dpdx, dpdy = (np.zeros((K, size, size)) for i in range(4))
        Fr, Σ_CRB = (np.zeros((d, d, size, size)) for i in range(2))
        
        Fr_aux = np.zeros((K, d, d, size, size))
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))

        for i in range(K):
            
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
            
        # λb using the approximation in Balzarotti et al, (S29)
          
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(K*SBR)
            
        for i in range(K):
            
            # probability arrays
        
            p[i, :, :] = (λ[i, :, :] + λb)/(K*λb + np.sum(λ, axis=0))

            # partial derivatives in x and y direction
    
            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
            
        # compute relevant information for every (i, j) position
        # TODO: vectorize this part of the code
            
        for i in range(size):
            for j in range(size):
                
                for k in range(K):
            
                    A = np.array([[dpdx[k, i, j]**2, 
                                   dpdx[k, i, j]*dpdy[k, i, j]],
                                  [dpdx[k, i, j]*dpdy[k, i, j], 
                                   dpdy[k, i, j]**2]])
        
                    Fr_aux[k, :, :, i, j] = (1/p[k, i, j]) * A
                    
                Fr[:, :, i, j] = N * np.sum(Fr_aux[:, :, :, i, j], axis=0)
                                    
                Σ_CRB[:, :, i, j] = np.linalg.inv(Fr[:, :, i, j])
                σ_CRB[i, j] = np.sqrt((1/d) * np.trace(Σ_CRB[:, :, i, j]))
                
        
        return σ_CRB, Σ_CRB, Fr
            
         
    if method=='3':
    
        # initalize σ_CRB and E(logL)
        
        I_f = np.zeros((size, size))
        σ_CRB = np.zeros((size, size))
        σ_CRB2 = np.zeros((size, size))
        σ_CRB_p = np.zeros((size, size))
        
        logL = np.zeros((size, size))
    
        # initialize different arrays needed to compute σ_CRB, Σ_CRB, Fr, etc
    
        p, λ, dpdx, dpdy, logL_aux = (np.zeros((K, size, size)) for i in range(5))
        Fr, Σ_CRB = (np.zeros((d, d, size, size)) for i in range(2))
        Fp, Σ_CRB_p = (np.zeros((K-1, K-1, size, size)) for i in range(2))
    
        J = np.zeros((K-1, d, size, size))
        
        diag_aux = np.zeros(K-1)
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))
                      
        for i in range(K):
            
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
                    
        # λb using the approximation in Balzarotti et al, (S29)
              
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(K*SBR)
            
        for i in range(K):
            
            # probability arrays
            
            p[i, :, :] = (λ[i, :, :] + λb)/(K*λb + np.sum(λ, axis=0))
            
            logL_aux[i, :, :] = N * p[i, :, :] * np.log(p[i, :, :])
            
            # partial derivatives in x and y direction

            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
            
#            locx = (size/4) * np.sqrt(2)/2
#            locy = (size/4) * np.sqrt(2)/2
            
            locx = (size/4)
            locy = (size/4)
#            
            plt.figure(str(i))
            plt.imshow(p[i, :, :])
            
            plt.figure()
            plt.plot(np.arange(-size/2, size/2), p[i, int(size/2), :], label='p x axis')
            plt.plot(np.arange(-size/2, size/2), p[i, ::-1, int(size/2 - locy)], label='p y axis')
            plt.legend()   
            
            
        for i in range(size):
            for j in range(size):
                    
                for k in range(K):
                    
                    if k < K-1:
                        
                        J[k, :, i, j] = np.array([dpdx[k, i, j], dpdy[k, i, j]])
                        
#                    if k < K-2:
                        
                        diag_aux[k] = 1/p[k, i, j]
                        
                    else:
                        
                        pass
                    
                logL[i, j] = np.sum(logL_aux[:, i, j], axis=0)
                        
                p_aux = np.diag(diag_aux)
                
                Fp[:, :, i, j] = N * (p_aux + np.ones((K-1, K-1))*(1/p[K-1, i, j]))
                Fr[:, :, i, j] = J[:, :, i, j].T.dot(Fp[:, :, i, j]).dot(J[:, :, i, j])
                                
                Σ_CRB[:, :, i, j] = np.linalg.inv(Fr[:, :, i, j])
                σ_CRB[i, j] = np.sqrt((1/d) * np.trace(Σ_CRB[:, :, i, j]))
                
                Σ_CRB_p[:, :, i, j] = np.linalg.inv(Fp[:, :, i, j])
                σ_CRB_p[i, j] = np.sqrt((1/(K-1)) * np.trace(Σ_CRB_p[:, :, i, j]))
                
                I_f[i, j] = np.sqrt((1/d) * np.trace(Fr[:, :, i, j]))
                σ_CRB2[i, j] = 1/I_f[i, j] 

                
        print(Fr[:, :, int(size/2), int(size/2 - locx)])
        print(Fr[:, :, int(size/2 + locy), int(size/2)])
        print(Fr[:, :, int(size/2), int(size/2 + locy)])
        print(Fr[:, :, int(size/2 - locy), int(size/2)])
        
                
        
#        print(p[:, int(size/2 - locy), int(size/2 - locx)])
#        print(p[:, int(size/2 + locy), int(size/2 - locx)])
#        print(p[:, int(size/2 - locy), int(size/2 + locx)])
#        print(p[:, int(size/2 + locy), int(size/2 + locx)])
        
        print(p[:, int(size/2), int(size/2)])
        
        print(p[:, int(size/2), int(size/2 - locx)])
        print(p[:, int(size/2 + locy), int(size/2)])
        print(p[:, int(size/2), int(size/2 + locy)])
        print(p[:, int(size/2 - locy), int(size/2)])
        
        print(σ_CRB[int(size/2), int(size/2 - locx)])
        print(σ_CRB[int(size/2 + locy), int(size/2)])
        print(σ_CRB[int(size/2), int(size/2 + locy)])
        print(σ_CRB[int(size/2 - locy), int(size/2)])


        return σ_CRB, Σ_CRB, Fr, σ_CRB_p, Σ_CRB_p, Fp, logL, I_f, σ_CRB2
    
    else:
        
        raise ValueError('Invalid method number, please choose 1, 2 or 3 \
                         according to the desired calculation')
     


def crb_camera(K, px_size_nm, sigma_psf, SBR, N, range_nm, dx_nm):
 
    """
    Cramer-Rao Bound for camera-based localization
    
    Input
    ----------
        px_num:     total number of physical pixels in grid,
                    e.g. px_num=81 if 9x9 grid
        px_size_nm: pixel size of camera in nm
        sigma_psf:  standard deviation of emission PSF in nm
        SBR:        signal to background ratio
        N :         total number of photons
        range_nm:   grid range in emitter position space
        dx_nm:      grid resolution along x-axis in emitter position space
        inf:        boolean, if true, set SBR to infinity, default: false
    
    Returns
    -------
        crb:        Cramer-Rao bound for given set of input parameter
    
    """
    σ_psf = sigma_psf
    dy_nm = dx_nm
    size = int(range_nm/dx_nm)
    
    if np.sqrt(K).is_integer(): 
        k = int(np.sqrt(K)) # k^2 is equivalent to the K expositions in MINFLUX CRB
    else:
        raise ValueError("K should be a perfect square number (i.e. 81, 64, etc)")
    
    p, λ, dpdx, dpdy, A, B, C, D = (np.zeros((size, size, k, k)) for i in range(8))
    
    x = np.arange(-range_nm/2, range_nm/2, dx_nm)
    y = np.arange(-range_nm/2, range_nm/2, dx_nm)
    
    px_range_nm = k * px_size_nm
    px = np.arange(-px_range_nm/2, px_range_nm/2, px_size_nm) + px_size_nm/2
    py = np.arange(-px_range_nm/2, px_range_nm/2, px_size_nm) + px_size_nm/2
    
    # -y for cartesian coordinates
    [Mx, My] = np.meshgrid(x, -y)  # emitter position matrices, dim: size x size (i.e. 100 x 100)
    [Mpx, Mpy] = np.meshgrid(px, -py) # pixel coord matrices dim: k x k (i.e. 9 x 9)
        
    for i in range(k):
        for j in range(k):
            
            # calculates terms
            a = erf((Mpx[i, j] + px_size_nm/2 - Mx)/(np.sqrt(2)*σ_psf))
            b = erf((Mpx[i, j] - px_size_nm/2 - Mx)/(np.sqrt(2)*σ_psf))
            c = erf((Mpy[i, j] + px_size_nm/2 - My)/(np.sqrt(2)*σ_psf))
            d = erf((Mpy[i, j] - px_size_nm/2 - My)/(np.sqrt(2)*σ_psf))
            
            p_0 = (a-b) * (c-d)
            
            # prob array for a given pixel (i, j) and for every position (x, y)
            p[:, :, i, j] = (1/(K + SBR)) + (1/4) * (SBR/(K + SBR)) * p_0
            
            # gradient of ps in each (x,y), careful with (i, j) -> (x, y)
            dpdy[:, :, i, j] = np.gradient(p[:, :, i, j], -dy_nm, axis=0)
            dpdx[:, :, i, j] = np.gradient(p[:, :, i, j], dx_nm, axis=1)
       
            # terms needed to compute CR bound for each (x,y) in emitter space
            A[:, :, i, j] = (1/p[:, :, i, j]) * dpdx[:, :, i, j]**2
            B[:, :, i, j] = (1/p[:, :, i, j]) * dpdy[:, :, i, j]**2
            C[:, :, i, j] = (1/p[:, :, i, j]) *(dpdx[:, :, i, j] * dpdy[:, :, i, j])
            D[:, :, i, j] = (1/p[:, :, i, j]) * (dpdx[:, :, i, j]**2 + dpdy[:, :, i, j]**2)
                
    # sigma Cramer-Rao numerator and denominator    
    E = np.sum(D, axis=(2,3)) 
    F = (np.sum(A, axis=(2,3)) * np.sum(B, axis=(2,3))) - np.sum(C, axis=(2,3))**2
    
    σ_CRB = np.sqrt(1/(2*N))*np.sqrt(E/F)
    
    return σ_CRB


def nMINFLUX(K, τ, relTime, a, b):
    
    """
    Photon collection in a MINFLUX experiment
    (n0, n1, n2, n3)
    
    Inputs
    ----------
    K : number of expositions
    τ : array, times of EBP pulses (1, K)
    relTime : photon arrival times relative to sync (N)
    a : init of temporal window (in ns)
    b : temporal window lenght (in ns)
    
    a,b can be adapted for different lifetimes

    Returns
    -------
    n : (1, K) array acquired photon collection.
    
    """
    
    # total number of detected photons
    N = np.shape(relTime)[0]

    # number of photons in each exposition
    n = np.zeros(K)    
    for i in np.arange(K):
        ti = τ[i] + a
        tf = τ[i] + a + b
        r = relTime[(relTime>ti) & (relTime<tf)]
        n[i] = np.size(r)
     
    return n 


def pos_MINFLUX(n, PSF, SBR, px_nm=1, prior=None, L=None, DEBUG=False):
    
    """    
    MINFLUX position estimator (using MLE)
    
    Inputs
    ----------
    n : acquired photon collection (K)
    PSF : array with EBP (K x size x size)
    SBR : estimated (exp) Signal to Bkgd Ratio

    Returns
    -------
    mle_index : position estimator in index coordinates (MLE)
    likelihood : Likelihood function
    
    Parameters 
    ----------
    px_nm : grid px in nm
        
    """
       
    # number of beams in EBP
    K = np.shape(PSF)[0]
    
    # FOV size
    size = np.shape(PSF)[1] 
    
    normPSF = np.sum(PSF, axis = 0)
    
    # probabilitiy vector 
    p = np.zeros((K, size, size))

    for i in np.arange(K):        
        p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * (1/K)
        
    # log-likelihood function
    l_aux = np.zeros((K, size, size))
    for i in np.arange(K):
        l_aux[i, :, :] = n[i] * np.log(p[i, : , :])
        
    likelihood = np.sum(l_aux, axis = 0)
        
    if prior == 'r<L':
                
        x = np.arange(-size/2, size/2)
        y = np.arange(-size/2, size/2)
        
        Mx, My = np.meshgrid(x, y)
        Mr = np.sqrt(Mx**2 + My**2)
        
        likelihood[Mr>L/2] = -np.inf
        
#    if DEBUG:
#        
#        size_nm = size*px_nm
#        fig, ax = plt.subplots()
#        
#        lfig = ax.imshow(likelihood, interpolation=None, 
#                         extent=[-size_nm/2, size_nm/2, -size_nm/2, size_nm/2])
#        
#        cbar = fig.colorbar(lfig, ax=ax)
#        cbar.ax.set_ylabel('log-likelihood')
#        
#        ax.set_xlabel('x (nm)')
#        ax.set_ylabel('y (nm)')
#        
#        circ = plt.Circle((0,0), radius=L/2, zorder=10, 
#                          facecolor='None', edgecolor='w')
#        ax.add_patch(circ)
                 
    # maximum likelihood estimator for the position    
    mle_index = np.unravel_index(np.argmax(likelihood, axis=None), 
                                 likelihood.shape)
    
    mle_nm = indexToSpace(mle_index, size, px_nm)
    
    if DEBUG:
        
        return mle_index, likelihood
    
    return mle_index


