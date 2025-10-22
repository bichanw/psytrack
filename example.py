
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 140


# import local psytrack as psy
import sys
import os

# Get the absolute path of the folder you want to import
folder_to_import = os.path.abspath('/usr/people/bichanw/SpikeSorting/Codes/psytrack/')

# Add the folder to the system path
sys.path.insert(0, folder_to_import) 

# Now you can import modules from that folder
import psytrack as psy



seed = 31
num_weights = 2
num_trials = 5000
hyper = {'sigma'   : 2**np.array([-4.0,-5.0]),
         'sigInit' : 2**np.array([ 0.0, 0.0]),
         'sigmay'  : .001}

# Simulate
simData = psy.generateSim(K=num_weights, N=num_trials, hyper=hyper,
                          boundary=6.0, iterations=1, seed=seed, savePath=None, gaussian=True)


# Plot
psy.plot_weights(simData['W'].T);
plt.ylim(-3.6,3.6);

# plot simData['all_Y'].shape
plt.figure();
plt.plot(simData['all_Y'], alpha=0.5, markersize=2);
plt.title('Simulated Choices');
plt.xlabel('Trial');
plt.ylabel('Choice');
plt.xlim(0, num_trials);
# plt.ylim(0, 3);
plt.show();
plt.savefig("example_simulated_choices.png", bbox_inches='tight')

# rec = psy.recoverSim(simData)

# psy.plot_weights(rec['wMode'], errorbar=rec["hess_info"]["W_std"])
# plt.plot(simData['W'], c="k", ls="-", alpha=0.5, lw=0.75, zorder=0)
# plt.ylim(-3.6,3.6);
# # save plot
# plt.savefig("example_recovered_weights.png", bbox_inches='tight')
quit()