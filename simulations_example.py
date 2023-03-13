"""
Created on Thu Apr  2 13:54:33 2020
@author: Luciano
This scripts uses the functions in tools.tools_simulations to simulate a
desired number (samples) of p-MINFLUX experiments, either with or without
blinking dynamics.
At the end it saves, plots and summarizes the results of the simulations.
"""

import os 
# path = os.getcwd() # get current directory 
# wdir = os.path.abspath(os.path.join(path, os.pardir)) # gets parent directory 
# os.chdir(wdir)

import numpy as np
import matplotlib.pyplot as plt
import tools.tools_simulations as tools
from datetime import datetime
import configparser
from PIL import Image
import time

DEBUG = False
saverelTimes = False
plt.close('all')

#%% Number of samples of the simulation

samples = 100

# folder
now = str(datetime.now()).replace('-', '')
now = now.replace(':', ' ')
now = now.replace('.', '_')
root = r'/Users/masullo/Documents/GitHub/p-minflux/Results/' # WARNING: change by your own folder!
folder = root + now + ' p_minflux'

os.mkdir(folder)

print(datetime.now(), 'Successfully created the directory {}'.format(folder))

#%%  simulated p-MINFLUX experiment parameters

#M_p = int(5*10**7)
M_p = int(2*10**5)  # p-minflux equivalent illumination cicles
dt = 25 # p-minflux cycle time [ns]
cycle_time = 'None' # [ns] cw-minflux cycle time, i.e. time it takes for 4 exposures
K = 4   # number of excitation beams
exptime = 'None'
Tlife = 0.001 # lifetime [ns]

M = 'None' # number of cycles
ct = 'None' # time points in units of dt = 25 ns

Ns = 90 # photons detected
Nb = 10

SBR = Ns/Nb # Signal to Background Ratio
L = 100 # ditance between beam centers

fov_center = [0, 0] # field of view center
r0_nm = np.array([5, -5])  # position of the emmiter molecule in nm

tot_time = (M_p * dt)/10**3 # [μs]
print('r0_nm', r0_nm)
print('Experiment length', tot_time, 'μs')

size_nm = 400 # field of view size (nm)
step_nm = 1
size = int(size_nm/step_nm)

# Emitter position in grid coordinates
r0 = tools.spaceToIndex(r0_nm, size_nm, step_nm) 

# EBP centers 
pos_nm = tools.beams(K, L, center=True, d='donut')

#%% Simulate PSFs

PSFs = np.zeros((K, size, size))
extent = [-size_nm/2, size_nm/2, -size_nm/2, size_nm/2]

for i in range(K):
    PSFs[i, :, :] = tools.psf(pos_nm[i], size_nm, step_nm, fov_center, d='donut')
    
    if saverelTimes:
        
        filename = folder + '/sim_results' + 'PSF_' + str(i)
#        np.save(filename + 'PSF_' + str(i), PSFs[i, :, :])
        data = PSFs[i, :, :]
        result = Image.fromarray(data)
        result.save(r'{}.tiff'.format(filename))
    
if DEBUG:
    
    fig, axes = plt.subplots(2,2)
    for i, ax in enumerate(axes.flat):
        ax.set(xlabel='x (nm)', ylabel='y (nm)')
        ax.imshow(PSFs[i, :, :], interpolation=None, extent=extent)
    
    plt.tight_layout()
            
#%% Save simulation parameters
    
filename = folder + '/sim_params'

config = configparser.ConfigParser()

config['p_minflux simulation parameters'] = {

    'Date and time': str(datetime.now()),
    'samples': samples,
    'p-minflux equivalent cycles': M_p,
    'Base time resolution (p-minflux cycle time) [ns]': dt,
    'cw-minflux cycle time [ns]': cycle_time,
    'Number of expositions': K,
    'Time per exposition [ns]': exptime,
    'Fluorescence lifetime of the emitter [ns]': Tlife,
    'Number of cw-minflux cycle': M,
    'N photons (signal)': Ns,
    'N photons (bkg)': Nb,
    'SBR': SBR,
    'L EBP parameter [nm]': L,
    'Position of the emitter [nm]': r0_nm,
    'tot_time [μs]': tot_time}

with open(filename + '.txt', 'w') as configfile:
    config.write(configfile)
        
#%% do MINFLUX analysis in loop
    
#τ = np.array([0, 1/4, 2/4, 3/4]) * dt  # [ns]
τ = np.arange(0, K)/K * dt 
a=0.0 # [ns]
b=dt/K # [ns]    

fail_count = 0
r0_est_nm_array = np.zeros((2, samples))

for i in range(samples):
    
    print('Sample', i)
    
    factor = 1.05

    params = [PSFs, r0, SBR, Ns, Nb, M_p, Tlife, factor, dt, cycle_time]
       
    relTime, absTimeBinary, failed = tools.sim_exp('p_minflux', None, *params)
    
    if saverelTimes:
        filename = folder + '/sim_results'
#            np.save(filename + 'microTime_' + str(i), relTime)
        relTime = relTime[relTime>0]
        np.savetxt(filename + 'microTime_' + str(i), relTime)

    if failed:
        
        fail_count += 1
        r0_est_nm = [np.NaN, np.NaN]
        r0_est_nm_array[:, i] = r0_est_nm
        
    else:
        
        # analyze data with p-MINFLUX algorithm
        
        n_array = tools.nMINFLUX(K, τ, relTime, a, b)
                
        r0_est = tools.pos_MINFLUX(n_array, PSFs, SBR=SBR, prior=None, L=L)
        r0_est_nm = tools.indexToSpace(r0_est, size_nm, step_nm)
                
        Ntot = np.sum(n_array)
        
#        print('n_array', n_array)
#        print('Ntot', Ntot)
#        print('r0_est_nm', r0_est_nm)
        
        r0_est_nm_array[:, i] = r0_est_nm
        
#%% Evaluate and save results        
        
print(r0_est_nm_array)
r0_est_nm_array = r0_est_nm_array[~np.isnan(r0_est_nm_array)] 
r0_est_nm_array = r0_est_nm_array.reshape(2, samples-fail_count)
   
mean = np.mean(r0_est_nm_array, axis=1)
std = np.std(r0_est_nm_array, axis=1)
print('r0_est_nm mean', mean)
print('r0_est_nm std', std)

x_est_array = r0_est_nm_array[0, :]
y_est_array = r0_est_nm_array[1, :]
x = r0_nm[0]
y = r0_nm[1]

err_x_array = np.abs(x_est_array - x)
err_y_array = np.abs(y_est_array - y)
    
#err_array = np.abs(r0_est_nm_array.T - r0_nm)
#err_mean = np.around(np.mean(err_array, axis=0), 2)
#err_std = np.around(np.std(err_array, axis=0), 2)

#print('err mean', err_mean)
#print('err std', err_std)
#
#err_1d = np.around(np.sqrt(err_mean[0]**2 + err_mean[1]**2), 2)

#print('1D error', err_1d)

print('2D error is', np.sqrt((1/2)*np.mean(err_x_array**2+err_y_array**2)))

print('Number of failed simulations', fail_count, 'out of', samples)
print('Failed simulations should be kept below 5%')

plt.figure('Histogram x')

nbins=50
plt.hist(r0_est_nm_array[0, :], bins=nbins)

plt.xlabel('x estimator (nm)')
plt.ylabel('Counts')

plt.figure('Histogram y')

nbins=50
plt.hist(r0_est_nm_array[1, :], bins=nbins)

plt.xlabel('y estimator (nm)')
plt.ylabel('Counts')

##%% Save simulation results
#    
#filename = folder + '/sim_results'
#
#config = configparser.ConfigParser()
#
#config['Simulation results'] = {
#
#    'Date and time': str(datetime.now()),
#    'err_mean': err_mean,
#    'err_std [nm]': err_std,
#    'err_1d [nm]': err_1d}
#
#with open(filename + '.txt', 'w') as configfile:
#    config.write(configfile)
#    
#np.save(filename + '_r0_est_nm_array', r0_est_nm_array)