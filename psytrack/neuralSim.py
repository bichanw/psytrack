import numpy as np
from datetime import datetime
from os import makedirs
from .runSim import generateSim
from datetime import datetime
from .hyperOpt import hyperOpt


def gaussian_kernel(x=None, mu=15, sigma=5):
    """Calculates the 1D Gaussian kernel value for a given point x.

    Args:
        x (float or np.ndarray): The point(s) at which to evaluate the kernel.
            If None, defaults to np.arange(0, 30).
        mu (float): The mean of the Gaussian distribution. Default is 15.
        sigma (float): The standard deviation of the Gaussian distribution. 
            Default is 5.

    Returns:
        float or np.ndarray: The Gaussian kernel value(s), normalized to 
            maximum value of 1.
    """
    if x is None:
        x = np.arange(0, 30)
    
    resp = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    resp = resp / np.max(resp)
    return resp


def gen_evoked_resp(t, kernel_weight = None, tr_len=1000, resp_kernel=None):
    """Generate neural response from a list of event times.

    Args:
        t: list of event times. Each element is an array/list of time indices
            where events occur for that trial.
        kernel_weight: list or array of weights for the response kernel.
            Length should match len(t). If None, defaults to ones.
        tr_len: int or array-like, length of each trial. If scalar, all trials
            have the same length. If array, should have length matching len(t).
        resp_kernel: array-like, response kernel to convolve with events.
            If None, uses default Gaussian kernel.

    Returns:
        neural_resp: array, neural response. If tr_len is scalar, shape is
            (len(t), tr_len). If tr_len is array, shape is (sum(tr_len),).
        resp_kernel: array, the response kernel used (for reference).
    """
    if kernel_weight is None:
        kernel_weight = np.ones(len(t))
    if resp_kernel is None:
        # default Gaussian response kernel
        resp_kernel = gaussian_kernel(x=np.arange(0, 30), mu=15, sigma=5)

    # Equal length trials: generate response for each entry in t
    # rows correspond to entries in t
    if np.isscalar(tr_len):
        neural_resp = np.zeros((len(t), tr_len))
        for idx, event_times in enumerate(t):
            resp = np.zeros(tr_len)
            resp[event_times] = 1
            resp = np.convolve(resp, resp_kernel * kernel_weight[idx], 
                              mode='full')[:tr_len]
            neural_resp[idx] = resp
    # Different trial lengths
    else:
        tr_start = np.cumsum(np.insert(tr_len, 0, 0))
        neural_resp = np.zeros(tr_start[-1])
        for idx, event_times in enumerate(t):
            resp = np.zeros(tr_len[idx])
            resp[event_times] = 1
            resp = np.convolve(resp, resp_kernel * kernel_weight[idx], 
                              mode='full')[:tr_len[idx]]
            neural_resp[tr_start[idx]:tr_start[idx+1]] = resp

    return neural_resp, resp_kernel


def generateNeuralSim(K=2,
                      n_tr=30,
                      tr_len=None,
                      rate=0.01,
                      hyper=None,
                      days=None,
                      boundary=4.0,
                      seed=None,
                      savePath=None,
                      resp_kernel=None,
                      if_run_estimation=False):
    """Generate neural simulation data.
    Args:
        K: int, number of weights to simulate
        n_tr: int, number of trials to simulate
        tr_len: int or array-like, length of each trial. If scalar, all trials
            have the same length. If array, should have length matching len(t).
        rate: float, rate of cue occurrence
        hyper: dict, hyperparameters for the simulation
        days: list or array, list of the trial lengths in days
        boundary: float, weights are reflected from this boundary
            during simulation, is a symmetric +/- boundary
        seed: int, random seed to make random simulations reproducible
        savePath: str, if given creates a folder and saves simulation data
            in a file; else data is returned
        resp_kernel: list of response kernels to convolve with events.
            If None, uses default Gaussian kernel.
        if_run_estimation: bool, if True, runs estimation on the simulation data
    Returns:
        dat: dict, contains the simulation data
        weights: dict, contains the weights for the simulation
    """

    np.random.seed(seed)
    
    # initiate trial length
    if tr_len is None:
        tr_len = np.random.randint(300, 600, size=n_tr)
    else:
        # check if tr_len is a list or scalar
        if np.isscalar(tr_len):
            tr_len = np.array([tr_len] * n_tr)
        else:
            if len(tr_len) != n_tr:
                raise ValueError(f"tr_len must be a scalar or a list of length n_tr, but got {len(tr_len)}")
            tr_len = np.array(tr_len, dtype=int)
    tr_start = np.cumsum(np.insert(tr_len, 0, 0)) # starting index of each trial


    # initiate hyperparameters
    if hyper is None:
        hyper = {'sigma': [2**-1] * (K+1)}
    elif 'sigma' not in hyper:
        hyper['sigma'] = [2**-1] * (K+1)
    elif np.isscalar(hyper['sigma']):
        hyper['sigma'] = [hyper['sigma']] * (K+1)
    # noise magnitude
    if 'sigmay' not in hyper:
        hyper['sigmay'] = 0.1
    # initial value of weights
    if 'sigInit' not in hyper:
        hyper['sigInit'] = [2**4] * (K+1)

    # initiate response kernel
    if resp_kernel is None:
        resp_kernel = [gaussian_kernel(x=np.arange(0, 30), mu=15, sigma=5)] * K



    # Generate trial-by-trial weights for this configuration
    tr_weights = generateSim(K=K + 1, N=n_tr, iterations=1, hyper=hyper, boundary=boundary, seed=seed)
    # remove unused Y from tr_weights
    tr_weights.pop('all_Y')

    # Generate inputs and responses
    X = np.ones((tr_start[-1], K))
    y = np.zeros(tr_start[-1])
    true_X = np.zeros((tr_start[-1], K))
    t_cues = []
    for i in range(K):
        # random cue times for each trial
        t_cues.append([np.where(np.random.rand(cur_tr_len) < rate)[0] for cur_tr_len in tr_len])
        # generate evoked response
        true_X[:, i], _ = gen_evoked_resp(t_cues[i], kernel_weight=tr_weights['W'][:, i+1], tr_len=tr_len, resp_kernel=resp_kernel[i])
        # generate unscaled response for input
        unsclaed_X, _  = gen_evoked_resp(t_cues[i], tr_len=tr_len, resp_kernel=None)
        X[:, i] = unsclaed_X
        y += true_X[:, i]


    # Add bias term
    X = np.hstack((np.ones((tr_start[-1], 1)), X))
    for i in range(tr_len.shape[0]):
        y[tr_start[i]:tr_start[i+1]] = y[tr_start[i]:tr_start[i+1]] + tr_weights['W'][i, 0]

    # Add noise
    y += np.random.randn(tr_start[-1]) * hyper['sigmay']

    # generate data dictionary for psytrack
    dat = {
        'inputs': {'x': X},
        'y': y,
        'tr_start': tr_start,
        'dayLength': np.array([], dtype=int),
    }
    weights = {'x': K + 1}
    hyper_guess = {'sigma': [2**-1] * (K + 1), 'sigmay': hyper['sigmay']}
    optList = ['sigma','sigmay']

    # delete variables to save memory
    del X, y, t_cues

    # save data
    if savePath is not None:
        savePath = f'{savePath}/{datetime.now().strftime('%Y%m%d_%H%M%S')}'
        save_dict = {
            'dat': dat,
            'weights': weights,
            'hyper_guess': hyper_guess,
            'seed': seed,
            'true_W': tr_weights,
        }
        np.savez_compressed(f'{savePath}_neural_simu_data.npz', save_dict=save_dict)
    
    
    # run hyperparameter optimization
    if if_run_estimation:
        hyp, evd, wMode, hess_info = hyperOpt(dat, hyper_guess, weights, optList)

        # save estimation results
        if savePath is not None:
            save_dict = {
                'hyp': hyp,
                'evd': evd,
                'wMode': wMode,
                'hess_info': hess_info,
            }
            np.savez_compressed(f'{savePath}_neural_simu_results.npz', save_dict=save_dict)
    else:
        hyp = None
        evd = None
        wMode = None
        hess_info = None

    return dat, tr_weights, hyp, evd, wMode, hess_info

    


