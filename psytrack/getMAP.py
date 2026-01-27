import numpy as np
from scipy.optimize import minimize
import pickle

from psytrack.helper.memoize import memoize
from psytrack.helper.jacHessCheck import jacHessCheck
from psytrack.helper.helperFunctions import (
    DT_X_D,
    sparse_logdet,
    read_input,
    make_invSigma,
    DTinv_v,
    myblk_diags,
    xtx_neural,
)


def getMAP(dat, hyper, weights, method=None, E0=None, showOpt=0,gaussian=False):
    '''Estimates epsilon parameters with a random walk prior.

    Args:
        dat : dict, all data from a specific subject
        hyper : a dictionary of hyperparameters used to construct the prior
            Must at least include sigma, can also include sigInit, sigDay
        weights : dict, name and count of weights in dat['inputs'] to fit
        method : str, control over type of learning, defaults to standard 
            trial-by-trial fitting; '_days' and '_constant' also supported
        E0 : initial parameter estimate, must be of approprite size N*K, 
            defaults to zeros
        showOpt : {0 : no text, 1 : verbose, 2+ : Hess + deriv check, done
            showOpt-1 times}

    Returns:
        wMode : MAP estimate of the weights, ordered alphabetically as
            specified in `weights`.
        Hess : the Hessian of the log posterior at wMode, used for Laplace appx.
            in evidence max in this case, is a dict of sparse terms needed to
            construct Hess (which is not sparse)
        logEvd : log of the evidence
        llstruct : dictionary containing the components of the log evidence and
            other info
    '''

    # -----
    # Initializations and Sanity Checks
    # -----
    # Check and count trials
    if 'inputs' not in dat or 'y' not in dat or type(
            dat['inputs']) is not dict:
        raise Exception('getMAP_PBups: insufficient input, missing y')
    N = len(dat['y'])

    # Determine model type
    if 'tr_start' in dat:
        model_type = 'neural'
    elif gaussian:
        model_type = 'gaussian'
    else:
        model_type = 'standard'
    
    # Check validity of 'y'
    if model_type == 'standard':
        if np.array_equal(np.unique(dat['y']), [0, 1]):
            dat['y'] += 1
        elif not np.array_equal(np.unique(dat['y']), [1, 2]):
            raise Exception('getMAP_PBups: y must be parametrized as 1 and 2 only.')


    # Check and count weights
    K = 0
    if type(weights) is not dict:
        raise Exception('weights must be a dict')
    for i in weights.keys():
        if type(weights[i]) is not int or weights[i] < 0:
            raise Exception('weight values must be non-negative ints')
        K += weights[i]

    # Check if using constant weights or by-day weights
    if method is None:
        if model_type == 'neural':
            w_N = dat['tr_start'].shape[0] - 1
        else:
            w_N = N
    elif method == '_constant':
        w_N = 1
    elif method == '_days':
        w_N = len(dat['dayLength'])
    else:
        raise Exception('method type ' + method + ' not supported')

    # Initialize weights to particular values (default 0)
    if E0 is not None:
        if type(E0) is not np.ndarray:
            raise Exception('E0 must be an array')

        if E0.shape == (w_N * K,):
            eInit = E0.copy()
        elif E0.shape == (w_N, K):
            eInit = E0.flatten()
        else:
            raise Exception('E0 must be shape (w_N*K,) or (w_N,K), not ' +
                            str(E0.shape))
    else:
        eInit = np.zeros(w_N * K)

    # Do sanity checks on hyperparameters
    if 'sigma' not in hyper and 'sigmas_by_ind' not in hyper:
        raise Exception('WARNING: sigma not specified in hyper dict')
    if 'alpha' in hyper:
        raise Exception('WARNING: alpha is not supported')
    if method == '_constant':
        if 'sigInit' not in hyper or hyper['sigInit'] is None:
            print('WARNING: sigInit being set to sigma for method', method)
    if method == '_days':
        if 'sigDay' not in hyper or hyper['sigDay'] is None:
            print('WARNING: sigDay being set to sigma for method', method)

    # Get index of start of each day
    if ('dayLength' not in dat) and (
        ('sigDay' in hyper and hyper['sigDay'] is not None) or
        (method == '_days')):
        print('WARNING: sigDay has no effect, dayLength not supplied in dat')
        dat['dayLength'] = np.array([], dtype=int)

    # Account for missing trials from running xval (i.e. gaps from test set)
    if 'missing_trials' in dat and dat['missing_trials'] is not None:
        if len(dat['missing_trials']) != N:
            raise Exception('missing_trials must be length N if used')
    else:
        dat['missing_trials'] = None

    # -----
    # MAP estimate
    # -----

    
    # Prepare minimization of loss function, Memoize to preserve Jac+Hess info
    # Create a wrapper that captures model_type for memoization
    def negLogPost_wrapper(E_flat, dat, hyper, weights, method):
        return negLogPost(E_flat, dat, hyper, weights, method, model_type=model_type)
    
    lossfun = memoize(negLogPost_wrapper)
    my_args = (dat, hyper, weights, method)

    if showOpt:
        opts = {'disp': True}
        callback = print
    else:
        opts = {'disp': False}
        callback = None

    # Actual optimization call
    # Uses 'hessp' to pass a function that calculates product of Hessian
    #    with arbitrary vector
    try:
        if showOpt:
            print('Obtaining MAP estimate...')
        
        # raise Exception('test')
        result = minimize(
            lossfun,
            eInit,
            jac=lossfun.jacobian,
            hessp=lossfun.hessian_prod,
            method='trust-ncg', # Newton-CG
            tol=1e-9,
            args=my_args,
                options=opts,
                callback=callback,
            )
    except Exception as e:
        # save all relevant variables to a file
        dump = {
            'model_type': model_type,
            'eInit': eInit,
            'my_args': my_args,
            'opts': opts,
            'error': e,
        }
        with open('/usr/people/bichanw/SpikeSorting/Codes/psytrack/data/getMAP_error.pkl', 'wb') as f:
            pickle.dump(dump, f)
        raise e

    # Recover the results of the optimization
    eMode = result.x
    # dict of sparse components of Hess
    Hess = lossfun.hessian(eMode, *my_args)

    # Print message if optimizer does not converge (usually still pretty good)
    if showOpt and not result.success:
        print('WARNING — MAP estimate: minimize() did not converge\n',
              result.message)
        print('NOTE: this is ususally irrelevant as the optimizer still finds '
              'a good solution. If you are concerned, run a check of the '
              'Hessian by setting showOpt >= 2')

    # Run DerivCheck & HessCheck at eMode (will run ShowOpt-1 distinct times)
    if showOpt >= 2:
        print('** Jacobian and Hessian Check **')
        for check in range(showOpt - 1):
            print('\nCheck', check + 1, ':')
            jacHessCheck(lossfun, eMode, *my_args)
            print('')

    # -----
    # Evidence (Marginal likelihood)
    # -----

    # Prior and likelihood at eMode, also recovering the associated wMode
    if showOpt:
        print('Calculating evd, first prior and likelihood at eMode...')
    # Use dispatch helper for consistency
    pT, lT, wMode = _get_posterior_terms_dispatch(eMode, *my_args, model_type=model_type)

    # Posterior term (with Laplace approx), calculating sparse log determinant
    if showOpt:
        print('Now the posterior with Laplace approx...')
    center = DT_X_D(Hess['ddlogprior'], Hess['K']) + Hess['H']
    logterm_post = (1 / 2) * sparse_logdet(center)

    # Compute Log evd and construct dict of likelihood, prior,
    #   and posterior terms
    logEvd = lT['logli'] + pT['logprior'] - logterm_post
    if showOpt:
        print('Evidence:', logEvd)

    # Package up important terms to return
    llstruct = {'lT': lT, 'pT': pT, 'eMode': eMode}

    return wMode, Hess, logEvd, llstruct

def _get_posterior_terms_dispatch(E_flat, dat, hyper, weights, method=None, model_type='standard'):
    """Helper function to dispatch to the appropriate getPosteriorTerms* function.
    
    Args:
        E_flat, dat, hyper, weights, method: same as getPosteriorTerms()
        model_type: str, one of 'standard', 'gaussian', or 'neural'
    
    Returns:
        priorTerms, liTerms, W: same as getPosteriorTerms()
    """
    if model_type == 'neural':
        return getPosteriorTermsNeural(E_flat, dat, hyper, weights, method)
    elif model_type == 'gaussian':
        return getPosteriorTermsGauss(E_flat, dat, hyper, weights, method)
    elif model_type == 'standard':
        return getPosteriorTerms(E_flat, dat, hyper, weights, method)
    else:
        raise Exception(f'Unknown model_type: {model_type}. Must be "standard", "gaussian", or "neural".')


def negLogPost(*args, model_type=None):
    '''Returns negative log posterior (and its first and second derivative)
    Intermediary function to allow for getPosteriorTerms to be optimized.
    
    This unified function handles standard (logistic), gaussian, and neural models.

    Args:
        *args: (E_flat, dat, hyper, weights, method) - same as getPosteriorTerms()
        model_type: str, optional. One of 'standard', 'gaussian', or 'neural'.
            If None, auto-detects based on dat structure:
            - 'neural' if 'tr_start' in dat
            - 'gaussian' if 'sigmay' in hyper or gaussian flag was used
            - 'standard' otherwise

    Returns:
        negL : negative log-likelihood of the posterior
        dL : 1st derivative of the negative log-likelihood
        ddL : 2nd derivative of the negative log-likelihood,
            kept as a dict of sparse terms!
    '''
    # Auto-detect model type if not provided
    if model_type is None:
        if len(args) >= 2:
            dat = args[1]
            hyper = args[2] if len(args) >= 3 else {}
            # Check for neural model (has tr_start)
            if 'tr_start' in dat:
                model_type = 'neural'
            # Check for gaussian model (has sigmay in hyper)
            elif 'sigmay' in hyper:
                model_type = 'gaussian'
            else:
                model_type = 'standard'
        else:
            model_type = 'standard'
    
    # Get prior and likelihood terms using dispatch function
    priorTerms, liTerms, _ = _get_posterior_terms_dispatch(*args, model_type=model_type)

    # Negative log posterior
    negL = -priorTerms['logprior'] - liTerms['logli']
    dL = -priorTerms['dlogprior'] - liTerms['dlogli']
    ddL = {'ddlogprior': priorTerms['ddlogprior'], **liTerms['ddlogli']}

    return negL, dL, ddL


# Backward compatibility aliases
def negLogPostNeural(*args):
    """Backward compatibility wrapper for neural model."""
    return negLogPost(*args, model_type='neural')


def negLogPostGauss(*args):
    """Backward compatibility wrapper for gaussian model."""
    return negLogPost(*args, model_type='gaussian')


def getPosteriorTerms(E_flat, dat, hyper, weights, method=None):
    '''Given a sequence of parameters formatted as an N*K matrix, calculates
    random-walk log priors & likelihoods and their derivatives

    Args:
        E_flat : array, the N*K epsilon parameters, flattened to a single
        vector
        ** all other args are same as in getMAP **

    Returns:
        priorTerms : dict, the log-prior as well as 1st + 2nd derivatives
        liTerms : dict, the log-likelihood as well as 1st + 2nd derivatives
        W : array, the weights, calculated directly from E_flat
    '''

    if method in ['_days', '_constant']:
        raise Exception(
            'Need efficient calculations for _constant or _days methods')

    # ---
    # Initialization
    # ---

    # If function is called directly instead of through getMAP,
    #       fill in dummy values
    if 'dayLength' not in dat:
        dat['dayLength'] = np.array([], dtype=int)
    if 'missing_trials' not in dat:
        dat['missing_trials'] = None

    # Unpack input into g
    if 'g' not in dat:
        dat['g'] = read_input(dat, weights)
    g = dat['g']
    N, K = g.shape

    # Determine type of analysis (standard, constant, or day weights)
    if method is None:
        w_N = N
        # the first trial index of each new day
        days = np.cumsum(dat['dayLength'], dtype=int)[:-1]
        missing_trials = dat['missing_trials']
    elif method == '_constant':
        w_N = 1
        days = np.array([], dtype=int)
        missing_trials = None
    elif method == '_days':
        w_N = len(dat['dayLength'])
        days = np.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception('method ' + method + ' not supported')

    # Check shape of epsilon, with
    #   w_N (effective # of trials) * K (# of weights) elements
    if E_flat.shape != (w_N * K,):
        print(E_flat.shape, w_N, K, method)
        raise Exception('parameter dimension mismatch (#trials * #weights)')

    # ---
    # Construct random-walk prior, calculate priorTerms
    # ---

    # Construct random walk covariance matrix Sigma^-1, use sparsity for speed
    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)

    # Calculate the log-determinant of prior covariance,
    #   the log-prior, 1st, & 2nd derivatives
    logdet_invSigma = np.sum(np.log(invSigma.diagonal()))
    logprior = (1 / 2) * (logdet_invSigma - E_flat @ invSigma @ E_flat)
    dlogprior = -invSigma @ E_flat
    ddlogprior = -invSigma

    priorTerms = {
        'logprior': logprior,
        'dlogprior': dlogprior,
        'ddlogprior': ddlogprior,
    }

    # ---
    # Construct likelihood, calculate liTerms
    # ---

    # Reconstruct actual weights from E values
    E = np.reshape(E_flat, (K, w_N), order='C')
    W = np.cumsum(E, axis=1)

    # Calculate probability of Right on each trial
    y = dat['y'] - 1
    gw = np.sum(g * W.T, axis=1)
    pR = 1 / (1 + np.exp(-gw))

    # Preliminary calculations for 1st and 2nd derivatives
    dlliList = g * (y - pR)[:, None]

    alpha = (pR**2 - pR)[:, None, None]
    HlliList = alpha * (g[:, :, None] @ g[:, None, :])

    # INSERT CODE HERE TO HANDLE _days OR _constant METHODS

    # Calculate the log-likelihood and 1st & 2nd derivatives
    logli = np.sum(y * gw - np.logaddexp(0, gw))
    dlogli = DTinv_v(dlliList.flatten('F'), K)
    ddlogli = {'H': myblk_diags(HlliList), 'K': K}

    # print("passed here")
    liTerms = {'logli': logli, 'dlogli': dlogli, 'ddlogli': ddlogli}

    return priorTerms, liTerms, W

def getPosteriorTermsNeural(E_flat, dat, hyper, weights, method=None):
    '''Given a sequence of parameters formatted as an N*K matrix, calculates
    random-walk log priors & likelihoods and their derivatives

    Args:
        E_flat : array, the N*K epsilon parameters, flattened to a single
        vector
        ** all other args are same as in getMAP **

    Returns:
        priorTerms : dict, the log-prior as well as 1st + 2nd derivatives
        liTerms : dict, the log-likelihood as well as 1st + 2nd derivatives
        W : array, the weights, calculated directly from E_flat
    '''

    # TO-DO: need to update initialization later
    # the data that we need:
    # - y: N x 1 vector of responses
    # - tr_start: N_trial+1 x 1 vector of trial start indices

    # check all the data are present
    if 'y' not in dat:
        raise Exception('y not in dat')
    if 'tr_start' not in dat:
        raise Exception('tr_start not in dat')
    # do we need this? Or based on python we can change dat within function
    # if 'xtx' not in dat:
    #     raise Exception('calculate xtx outside loop to save time')
    # change to include Hessian
    # actually not the full Hessian, just xtx since Hessian includes sigma_y

    # unpack data
    if 'g' not in dat:
        dat['g'] = read_input(dat, weights)
    g = dat['g']
    N, K = g.shape
    w_N = dat['tr_start'].shape[0] - 1

    # testing phase, no days or missing trials
    days = np.array([], dtype=int)
    missing_trials = None

    # ---
    # Construct random-walk prior, calculate priorTerms
    # ---

    # Construct random walk covariance matrix Sigma^-1, use sparsity for speed
    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)

    # Calculate the log-determinant of prior covariance,
    #   the log-prior, 1st, & 2nd derivatives
    logdet_invSigma = np.sum(np.log(invSigma.diagonal()))
    logprior = (1 / 2) * (logdet_invSigma - E_flat @ invSigma @ E_flat)
    dlogprior = -invSigma @ E_flat
    ddlogprior = -invSigma

    priorTerms = {
        'logprior': logprior,
        'dlogprior': dlogprior,
        'ddlogprior': ddlogprior,
    }

    # ---
    # Construct likelihood, calculate liTerms
    # ---


    # Reconstruct actual weights from E values
    E = np.reshape(E_flat, (K, w_N), order='C')
    W_tr = np.cumsum(E, axis=1)

    # fill in W for each time point
    W = np.zeros((K, N))
    for i in range(w_N):
        W[:, dat['tr_start'][i]:dat['tr_start'][i+1]] = W_tr[:, i][:,None]
    

    # Calculate probability of Right on each trial
    y = dat['y']
    gw = np.sum(g * W.T, axis=1) # T x 1 (namely N samples by 1) vector of w_t' * x_t
    y_res = y - gw

    # Preliminary calculations for 1st and 2nd derivatives
    # is it affine with respect to E?
    # Old code (commented for reference):
    # dlliList = np.zeros((w_N, K))
    # for i in range(w_N):
    #     dlliList[i, :] = np.sum(y_res[dat['tr_start'][i]:dat['tr_start'][i+1]][:,None] * g[dat['tr_start'][i]:dat['tr_start'][i+1], :],axis=0)
    # dlliList = dlliList / (hyper['sigmay']**2)
    
    # Optimized version: pre-compute product once, then accumulate per trial
    y_g_product = y_res[:, None] * g  # Shape: (N, K)
    dlliList = np.zeros((w_N, K))
    for i in range(w_N):
        # start, end = dat['tr_start'][i], dat['tr_start'][i+1]
        dlliList[i, :] = np.sum(y_g_product[dat['tr_start'][i]:dat['tr_start'][i+1], :], axis=0)
    dlliList = dlliList / (hyper['sigmay']**2)
    # g - size w_N x K (what dlliList should be)

    # this is only dependent on hyperparameters
    # not sure yet if should be put outside this function or anywhere else
    # for speed purpose
    # xtx = g[:, :, None] @ g[:, None, :]
    if 'xtx' not in dat:
        dat['xtx'] = xtx_neural(g, dat['tr_start'])
    HlliList = - dat['xtx'] / (hyper['sigmay'] ** 2)
    # for HlliList[t], t = 0, ..., w_N-1, it should be dw^2 for sample t
    # so for X (K x t_n) of trial n, HlliList[t] = - X X^T / (hyper['sigmay'] ** 2)
    # !!! we could even go a step further and calculate ddlogli mostly outside loop 


    # Calculate the log-likelihood and 1st & 2nd derivatives
    logli = (
        - N * np.log(hyper['sigmay'])
        - 0.5 * np.sum(y_res ** 2) / (hyper['sigmay'] ** 2)
    )
    dlogli = DTinv_v(dlliList.flatten('F'), K)
    ddlogli = {'H': myblk_diags(HlliList), 'K': K}

    liTerms = {'logli': logli, 'dlogli': dlogli, 'ddlogli': ddlogli}

    return priorTerms, liTerms, W_tr

def getPosteriorTermsGauss(E_flat, dat, hyper, weights, method=None):
    '''Given a sequence of parameters formatted as an N*K matrix, calculates
    random-walk log priors & likelihoods and their derivatives

    Args:
        E_flat : array, the N*K epsilon parameters, flattened to a single
        vector
        ** all other args are same as in getMAP **

    Returns:
        priorTerms : dict, the log-prior as well as 1st + 2nd derivatives
        liTerms : dict, the log-likelihood as well as 1st + 2nd derivatives
        W : array, the weights, calculated directly from E_flat
    '''

    if method in ['_days', '_constant']:
        raise Exception(
            'Need efficient calculations for _constant or _days methods')

    # ---
    # Initialization
    # ---

    # # save all inputs and exit 
    # import pickle
    # with open('/usr/people/bichanw/SpikeSorting/Codes/psytrack/getPosteriorTermsGauss_inputs.pkl', 'wb') as f:
    #     pickle.dump({'E_flat': E_flat, 'dat': dat, 'hyper': hyper, 'weights': weights, 'method': method}, f)
    # raise Exception("Exiting after saving inputs for debugging")


    # If function is called directly instead of through getMAP,
    #       fill in dummy values
    if 'dayLength' not in dat:
        dat['dayLength'] = np.array([], dtype=int)
    if 'missing_trials' not in dat:
        dat['missing_trials'] = None

    # Unpack input into g
    if 'g' not in dat:
        dat['g'] = read_input(dat, weights)
    g = dat['g']
    N, K = g.shape

    # Determine type of analysis (standard, constant, or day weights)
    if method is None:
        w_N = N
        # the first trial index of each new day
        days = np.cumsum(dat['dayLength'], dtype=int)[:-1]
        missing_trials = dat['missing_trials']
    elif method == '_constant':
        w_N = 1
        days = np.array([], dtype=int)
        missing_trials = None
    elif method == '_days':
        w_N = len(dat['dayLength'])
        days = np.arange(1, w_N, dtype=int)
        missing_trials = None
    else:
        raise Exception('method ' + method + ' not supported')

    # Check shape of epsilon, with
    #   w_N (effective # of trials) * K (# of weights) elements
    if E_flat.shape != (w_N * K,):
        print(E_flat.shape, w_N, K, method)
        raise Exception('parameter dimension mismatch (#trials * #weights)')

    # ---
    # Construct random-walk prior, calculate priorTerms
    # ---

    # Construct random walk covariance matrix Sigma^-1, use sparsity for speed
    invSigma = make_invSigma(hyper, days, missing_trials, w_N, K)

    # Calculate the log-determinant of prior covariance,
    #   the log-prior, 1st, & 2nd derivatives
    logdet_invSigma = np.sum(np.log(invSigma.diagonal()))
    logprior = (1 / 2) * (logdet_invSigma - E_flat @ invSigma @ E_flat)
    dlogprior = -invSigma @ E_flat
    ddlogprior = -invSigma

    priorTerms = {
        'logprior': logprior,
        'dlogprior': dlogprior,
        'ddlogprior': ddlogprior,
    }

    # ---
    # Construct likelihood, calculate liTerms
    # ---


    # Reconstruct actual weights from E values
    E = np.reshape(E_flat, (K, w_N), order='C')
    W = np.cumsum(E, axis=1)

    # Calculate probability of Right on each trial
    y = dat['y']
    gw = np.sum(g * W.T, axis=1) # T x 1 (namely N samples by 1) vector of w_t' * x_t
    # pR = 1 / (1 + np.exp(-gw))

    # Preliminary calculations for 1st and 2nd derivatives
    dlliList = g * (y - gw)[:, None] / (hyper['sigmay']**2)
    # dlliList = g * (y - pR)[:, None]

    # this is only dependent on hyperparameters
    # not sure yet if should be put outside this function or anywhere else
    # for speed purpose
    HlliList = - (g[:, :, None] @ g[:, None, :]) / (hyper['sigmay'] ** 2)

    # INSERT CODE HERE TO HANDLE _days OR _constant METHODS

    # Calculate the log-likelihood and 1st & 2nd derivatives
    # logli = np.sum(y * gw - np.logaddexp(0, gw))
    logli = ( #-0.5 * N * np.log(2 * np.pi)
        - N * np.log(hyper['sigmay'])
        - 0.5 * np.sum((y - gw) ** 2) / (hyper['sigmay'] ** 2)
    )
    dlogli = DTinv_v(dlliList.flatten('F'), K)
    ddlogli = {'H': myblk_diags(HlliList), 'K': K}

    liTerms = {'logli': logli, 'dlogli': dlogli, 'ddlogli': ddlogli}

    return priorTerms, liTerms, W