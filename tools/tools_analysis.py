#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:53:34 2020

@author: Luciano Masullo, Lucía López and Lars Richter
"""

import numpy as np
from skimage import io
from scipy import optimize as opt
from scipy import ndimage as ndi
import os
import glob

π = np.pi

#     define polynomial function to fit PSFs
def poly_func(grid, x0, y0, c00, c01, c02, c03, c04, c10, c11, c12, c13, c14, 
              c20, c21, c22, c23, c24, c30, c31, c32, c33, c34,
              c40, c41, c42, c43, c44):
    
    """    
    Polynomial function to fit PSFs.
    Uses built-in function polyval2d.
    
    Inputs
    ----------
    grid : x,y array
    cij : coefficients of the polynomial function
    Returns
    -------
    q : polynomial evaluated in grid.
    
    """

    x, y = grid
    c = np.array([[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14], 
                  [c20, c21, c22, c23, c24], [c30, c31, c32, c33, c34],
                  [c40, c41, c42, c43, c44]])
    q = np.polynomial.polynomial.polyval2d((x - x0), (y - y0), c)

    return q.ravel()


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


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def getUniqueName(name):
    
    n = 1
    while os.path.exists(name + '.txt'):
        if n > 1:
            name = name.replace('_{}'.format(n - 1), '_{}'.format(n))
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1

    return name

    
def open_psf(filename, folder, subfolder):
    
    """   
    Open exp PSF images and drift data, fit exp data with poly_func
    Input
    ----------
    filename
    folder   
    subfolder 
    
    Returns
    -------
    
    PSF : (K, sizeg, sizeg) array, function from fit evaluated over grid
    x0, y0 : arrays, coordinates of EBP centers
    index : array, coordinates of EBP centers in indexes
    aopt : fit parameters, coefficients of polynomial function
       
    """
    
    # change dir to dir where PSF and drift data are located
    rootdir = r"C:\Users\Lucia\Documents\NanoFísica\MINFLUX\Mejor data TDIs"
    folder =  str(folder)
    subfolder = str(subfolder)
    newpath = os.path.join(rootdir, folder, subfolder)   
    os.chdir(newpath)
    
    # open any file with metadata from PSF images 
    fname = glob.glob('filename*')[0]   
    f = open(fname, "r")
    lines=f.readlines()
    # exp pixel size extracted from metadata
    pxexp = float(lines[8][22])
    print(pxexp)

    # open txt file with xy drift data
    c = str(filename) 
    cfile = str(c) + '_xydata.txt'
    coord = np.loadtxt(cfile, unpack=True)

    
    # open tiff stack with exp PSF images
    psffile = str(filename) + '.tiff'
    im = io.imread(psffile)
    imarray = np.array(im)
    psfexp = imarray.astype(float)
    
    # total number of frames
    frames = np.min(psfexp.shape)
    factor = 5.0
    # number of px in frame
    npx = np.size(psfexp, axis = 1)
    # final size of fitted PSF arrays (1 nm pixels)             
    sizepsf = int(factor*pxexp*npx)
        
    # number of frames per PSF (asumes same number of frames per PSF)
    fxpsf = frames//4

    # initial frame of each PSF
    fi = fxpsf*np.arange(5)  

    
    # interpolation to have 1 nm px and realignment with drift data
    psf = np.zeros((frames, sizepsf, sizepsf))        
    for i in np.arange(frames):
        psfz = ndi.interpolation.zoom(psfexp[i,:,:], factor*pxexp)    
        deltax = coord[1, i] - coord[1, 0]
        deltay = coord[2, i] - coord[2, 0]
        psf[i, :, :] = ndi.interpolation.shift(psfz, [deltax, deltay])

    # sum all interpolated and re-centered images for each PSF
    psfT = np.zeros((frames//fxpsf, sizepsf, sizepsf))
    for i in np.arange(4):
        psfT[i, :, :] = np.sum(psf[fi[i]:fi[i+1], :, :], axis = 0)
        
    # crop borders to avoid artifacts 
    w, h = psfT.shape[1:3]
    border = (w//5, h//5, w//5, h//5) # left, up, right, bottom
    psfTc = psfT[:, border[1]:h-border[1], border[0]:w-border[0]]
    psfTc = psfT[:, border[1]:h-border[1], border[0]:w-border[0]]
          
    # spatial grid
    sizeg = np.size(psfTc, axis = 1)
    sizexy = sizeg/factor
    pxg = 1/factor  # 1 nm px size for the function grid
    
    x = np.arange(0, sizexy, pxg)
    y = sizexy - np.arange(0, sizexy, pxg)
    x, y = np.meshgrid(x, y)
    
    # fit PSFs  with poly_func and find central coordinates (x0, y0)
    PSF = np.zeros((4, sizeg, sizeg))
    x0 = np.zeros(4)
    y0 = np.zeros(4)
    index = np.zeros((4,2))
    aopt = np.zeros((4,27))
#    x0fit = np.zeros(4)
#    y0fit = np.zeros(4)
        
    for i in np.arange(4):
        # initial values for fit parameters x0,y0 and c00
        ind1 = np.unravel_index(np.argmin(psfTc[i, :, :], 
            axis=None), psfTc[i, :, :].shape)
        x0i = x[ind1]
        y0i = y[ind1]
        c00i = np.min(psfTc[i, :, :])        
        p0 = [x0i, y0i, c00i, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        aopt[i,:], cov = opt.curve_fit(poly_func, (x,y), psfTc[i, :, :].ravel(), p0 = p0)    
        q = poly_func((x,y), *aopt[i,:])   
        PSF[i, :, :] = np.reshape(q, (sizeg, sizeg))
        # find min value for each fitted function (EBP centers)
        ind = np.unravel_index(np.argmin(PSF[i, :, :], 
            axis=None), PSF[i, :, :].shape)
        
        x0[i] = x[ind]
        y0[i] = y[ind]        
        index[i,:] = ind
  
    
    return PSF, x0, y0, index, aopt

def open_tcspc(supfolder, folder, subfolder, name):
    
    """   
    Open exp TCSPC data
    Input
    ----------
    
    supfolder
    folder   
    subfolder 
    name
    
    Returns
    -------
    
    absTime : array, absTime tags of collected photons
    relTime : array, relTime tags of collected photon
    τ : times of EBP pulses
    
    """
    
    # change dir to dir where TCSPC data is located
    rootdir = r"C:\Users\Lucia\Documents\NanoFísica\MINFLUX"
    supfolder = str(supfolder)
    folder =  str(folder)
    subfolder = str(subfolder)
    
    
    newpath = os.path.join(rootdir, supfolder, folder, subfolder)   
    os.chdir(newpath)
    
    # open txt file with TCSPC data
    tcspcfile = str(name) + '.txt'
    coord = np.loadtxt(tcspcfile, unpack=True)
    absTime = coord[1, :]
    relTime = coord[0, :]
    
    globRes = 1e-3 # absTime resolution 
    timeRes = 1 # relTime resolution (it's already in ns)
    
    absTime = absTime * globRes
    relTime = relTime * timeRes
    
    # find EBP pulses times
    [y, bin_edges] = np.histogram(relTime, bins=100)
    x = np.ediff1d(bin_edges) + bin_edges[:-1]
    
    T = len(y)//4*np.arange(5)
    
    ind = np.zeros(4)
    
    for i in np.arange(4):        
        ind[i] = np.argmax(y[T[i]:T[i+1]]) + T[i]
            
    
    ind = ind.astype(int)
    τ = x[ind]
    
    return absTime, relTime, τ
    
def trace_seg(absTime):
    
#    NEEDS TO BE FULLY AUTOMATED, TO DO
    
    """    
    Trace segmentation in case of blinking
    Find optimal bin width and threshold
    (Adapted from F.D. Stefani PhD Thesis)
        
    Inputs
    ----------
    absTime : time tags of detected photons (macro time)
    Returns
    -------
    bw : optimal bin width
    T : threshold
    
    Parameters
    ----------
    bwmin, bwmax : boundaries for bw
    Tamx : max threshold
       
    """
        
    # compute inter-photon times (in absTime units)
    ipt = np.ediff1d(absTime)
    size = len(ipt)
    binsipt = int(np.sqrt(size))
    # inter-photon times histogram 
    [y, bin_edges] = np.histogram(ipt, bins=binsipt)
    x = np.ediff1d(bin_edges) + bin_edges[:-1]
    
    ind = np.min(np.where(y<0.01)) 
    x = x[0:ind]
    y = y[0:ind]
    # double exponential fit of the ipt histogram
    def double_exp(t, a1, a2, b1, b2):
        return a1 * np.exp(b1 * t) + a2 * np.exp(b2 * t)
    
    p,cov = opt.curve_fit(double_exp, x, y)
    
#    plt.figure()
#    plt.plot(x, y)
#    plt.plot(x, double_exp(x, *p))
     
    # ON and OFF parameters from fit
    Ion = -np.min(p[2:4])
    Ioff = -np.max(p[2:4])
    Aon = np.max(p[0:2])
    Aoff = np.min(p[0:2])

    # minimum bin width
    bwmin = 2*(np.max(absTime)/len(absTime))  
    ordermin = int(-np.log10(bwmin))
    
    # bw boundaries
    bwmin = 0.001
    bwmax = 0.005
    
    # iterations over bw and over threshold T    
    for i in np.arange(bwmin, bwmax, 0.0001):
  
        bwt = i     
        
        # average ON and OFF times
        Ton = Aon/(Ion*(1-np.exp(-Ion*bwt)))
        Toff = Aoff/(Ioff*(1-np.exp(-Ioff*bwt)))
        # bins on and off
        bon = Ton/bwt
        boff = Toff/bwt
        
#        # maximum threshold for a given bw
#        Tmax = Ion * bwt
        # Pon and Poff functions
        def P(n, I, b):
            
            a = (((I*b)**n)*np.exp(-I*b))
            b = np.math.factorial(n)
     
            r = (a/b)
            
            return r
        # maximum threshold for a given bw
        Tmax = Tmax
        for j in np.arange(0,int(Tmax),1):
        
            T = j 
            Pon = np.zeros(T+1)
            Poff = np.zeros(T+1)
            
            for n in np.arange(0, T+1, 1):
                Pon[n] = P(n, Ion, bwt)
                Poff[n] = P(n, Ioff, bwt)
            
            PonT = np.sum(Pon)
            PoffT = np.sum(Poff)
            # evaluate wrong bins for a given bw and T
            Wb = bon*PonT + boff*(1-PoffT)
#            print(Wb)
#        
            if Wb <0.0000001:
                break
        if Wb <0.000001:
            break 

                   
    # trace segmentation using the optimum bw and T 
    nbinsopt = int((np.max(absTime)/bwt))
    seqbin, time = np.histogram(absTime, bins=nbinsopt)
    seqbinkHz = seqbin*(1/(bwt*1000))
    
    
    mask = seqbin>T
    indexes = np.argwhere(np.diff(mask)).squeeze()
    number = int(len(indexes))

    timeON = time[indexes]
    t = {}
    absTimeON = {}


    for i in np.arange(0, number, 2):
        j = i//2
        t[i] = (timeON[i]<absTime) & (absTime<timeON[i+1])
        absTimeONp = absTime * t[i]
        absTimeON[j] = absTime[np.nonzero(absTimeONp)]

    
    return T, bwt


        
def n_minflux(τ, relTime, a, b):
    
    """
    Photon collection in a MINFLUX experiment
    (n0, n1, n2, n3)
    
    Inputs
    ----------
    τ : array, times of EBP pulses (1, K)
    relTime : photon arrival times relative to sync (N)
    a : init of temporal window (in ns)
    b : the temporal window lenght (in ns)
    
    a,b can be adapted for different lifetimes
    Returns
    -------
    n : (1, K) array acquired photon collection.
    
    """
    
    K = 4
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


def pos_minflux(n, PSF, SBR):
    
    """    
    MINFLUX position estimator (using MLE)
    
    Inputs
    ----------
    n : acquired photon collection (K)
    PSF : array with EBP (K x size x size)
    SBR : estimated (exp) Signal to Bkgd Ratio
    Returns
    -------
    indrec : position estimator in index coordinates (MLE)
    pos_estimator : position estimator (MLE)
    Ltot : Likelihood function
    
    Parameters 
    ----------
    step_nm : grid step in nm
        
    """

    step_nm = 0.2
       
    # number of beams in EBP
    K = np.shape(PSF)[0]
    # FOV size
    size = np.shape(PSF)[1] 
    
    normPSF = np.sum(PSF, axis = 0)
    
    # probabilitiy vector 
    p = np.zeros((K, size, size))

    for i in np.arange(K):        
        p[i,:,:] = (SBR/(SBR + 1)) * PSF[i,:,:]/normPSF + (1/(SBR + 1)) * (1/K)

    # likelihood function
    L = np.zeros((K,size, size))
    for i in np.arange(K):
        L[i, :, :] = n[i] * np.log(p[i, : , :])
        
    Ltot = np.sum(L, axis = 0)

    # maximum likelihood estimator for the position    
    indrec = np.unravel_index(np.argmax(Ltot, axis=None), Ltot.shape)
    pos_estimator = indexToSpace(indrec, size, step_nm)
    
    return indrec, pos_estimator, Ltot


def likelihood(K, PSF, n, λb, pos_nm, step_nm, size_nm):
    
    """
    Computes the full likelihood for a given MINFLUX experiment 
    
    Input
    ----------
    K : int, number of excitation beams
    PSF : (K, size, size) array, experimental or simulated PSF
    n :  (1, K) array , photon collection 
    λb : float, bkgd level
    pos _nm : (K, 2) array, centers of the EBP positions
    step_nm : step of the grid in nm
    size_nm : size of the grid in nm
    Returns
    -------
    Like : (size, size) array, Likelihood function in each position
    
    """
    
    # size of the (x,y) grid
    size = int(size_nm/step_nm)
    
    # different arrays
    mle = np.zeros((size, size, K))
    p_array = np.zeros((size, size, K))
    λ_array = np.zeros((size, size, K))
    λb_array = np.ones((size, size)) * λb

    
    # λs in each (x,y)
    for i in np.arange(K):
        λ_array[:, :, i] = PSF[i, :, :]        
    
    norm_array = (K*λb + np.sum(λ_array, axis=2))
        
    # probabilities in each (x,y)
    for i in np.arange(K):
        p_array[:, :, i] = (λ_array[:, :, i] + λb_array)/norm_array
    
    # Likelihood
    for i in np.arange(K):
        mle[:, :, i] = n[i] * np.log(p_array[:, :, i])
        
    Like = np.sum(mle, axis = 2)
        
    return Like


