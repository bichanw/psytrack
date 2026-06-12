import numpy as np
from .helperFunctions import read_input
from ..hyperOpt import hyperOpt
from ..getMAP import getMAP


def plt_grid_results(sigma_grid, log_li_array, search_indices):
    '''
    Plot the grid results of the cross-validation

    Args:
        sigma_grid: dictionary of sigma grids for each index
        log_li_array: array of log-likelihood values
        search_indices: list of indices of the sigma grids to search

    Returns:
        fig: figure handle
        ax: axis handles
    '''
    import matplotlib.pyplot as plt

    N_sigma_tested = np.array([x.shape[0] for x in sigma_grid.values()])
    cv_results_arr = log_li_array.reshape(N_sigma_tested[0],N_sigma_tested[1],N_sigma_tested[2],N_sigma_tested[3])

    # find best hyper-parameters
    [_, best_ind] = np.max(cv_results_arr), np.unravel_index(np.argmax(cv_results_arr), cv_results_arr.shape)
    print(f'best index: {best_ind}', flush=True)
    best_sigma_txt = (
        f"best σ (log2): "
        f"{np.log2(sigma_grid[search_indices[0]][best_ind[0]]).astype(int)}, "
        f"{np.log2(sigma_grid[search_indices[1]][best_ind[1]]).astype(int)}, "
        f"{np.log2(sigma_grid[search_indices[2]][best_ind[2]]).astype(int)}, "
        f"{np.log2(sigma_grid[search_indices[3]][best_ind[3]]).astype(int)}"
    )
    print(best_sigma_txt, flush=True)
    


    # plot results
    fig, ax = plt.subplots(N_sigma_tested[0],N_sigma_tested[1], layout='tight')
    for i in range(N_sigma_tested[0]):
        for j in range(N_sigma_tested[1]):
            ax[i,j].imshow(cv_results_arr[i,j])
    ax[best_ind[0],best_ind[1]].plot(best_ind[3],best_ind[2],'r*')

    # set clim and remove ticks
    clim = ax[best_ind[0],best_ind[1]].get_images()[0].get_clim()
    [ax_.get_images()[0].set_clim(clim) for ax_ in ax.flatten()];
    [ax_.set(xticks=[],yticks=[]) for ax_ in ax.flatten()];

    # label each axis of sigma grid tested
    ax[-1,-1].set(xticks=range(0,N_sigma_tested[3],2), xticklabels=np.log2(sigma_grid[search_indices[3]][::2]).astype('int') ,xlabel=r"$log_2(\sigma_4)$",
                yticks=range(N_sigma_tested[2]), yticklabels=np.log2(sigma_grid[search_indices[2]]).astype('int') ,ylabel=r"$log_2(\sigma_3)$");
    ax[0,0].set(title=rf'$\log_{2}\sigma_2={int(np.log2(sigma_grid[search_indices[1]][0]))}$');
    [ax[0,i].set(title=rf'${int(np.log2(sigma_grid[search_indices[1]][i]))}$') for i in range(1,N_sigma_tested[1])];

    ax[0,0].set(ylabel=rf'$\log_{2}\sigma_1={int(np.log2(sigma_grid[search_indices[0]][0]))}$');
    [ax[i,0].set(ylabel=rf'${int(np.log2(sigma_grid[search_indices[0]][i]))}$') for i in range(1,N_sigma_tested[0])];

    fig.suptitle(best_sigma_txt, fontsize=10, y=.98)  # y>1 lifts it above the subplot area

    return fig, ax

def produce_grid(mid=-1, step=2, num=5):
    # produce grid of sigma values
    # adding more functionality later

    # intermediate values
    start = mid - step * (num - 1) / 2
    end = mid + step * (num - 1) / 2

    # generate grid
    grid = np.logspace(base=2,start=start, stop=end, num=num)
    return grid

def index_data(D, ind, model_type = 'standard'):
    ''' 
    index the data dict by the indices
    '''

    # initialize new data dictionary
    newD = {}

    # remove possible intermediate variables produced during optimization
    if 'g' in D:
        del D['g']
    if 'xtx' in D:
        del D['xtx']

    if model_type == 'neural' or 'tr_start' in D:
        N = D['tr_start'].shape[0] - 1

        # indices of time points for the trials in ind
        ind = np.sort(ind)
        ind_t = np.concatenate([
            np.arange(D['tr_start'][i], D['tr_start'][i + 1], dtype=int)
            for i in ind
        ])
        ind_t = ind_t.astype(int)

        # iterate through all keys in the data dict and index the data
        for key in D.keys():

            # Inputs is handled separately below because we always want to slice
            # it along the trial dimension.
            if key == 'inputs' or key == 'tr_start' or key == 'y':
                newD[key] = {}
                continue

            try:
                if N == D[key].shape[0]:
                    # Common case: trial-aligned arrays are sliced by the fold
                    # train/test indices.
                    newD[key] = D[key][ind]
                else:
                    # Non-trial-aligned arrays (or scalars) are copied as-is.
                    newD[key] = D[key].copy()
            except:
                # If the value doesn't support `.shape`/slicing, keep it
                # unchanged for both train and test dicts.
                newD[key] = D[key]


        # `inputs` is assumed to be time-aligned with the same first dimension
        for i in D['inputs'].keys():
            newD['inputs'][i] = D['inputs'][i][ind_t]
        newD['y'] = D['y'][ind_t]

        # recalculate tr_start
        tr_len = np.diff(D['tr_start'])
        newD['tr_start'] = np.insert(np.cumsum(tr_len[ind]), 0, 0)

    else:

        N = D['y'].shape[0]

        # iterate through all keys in the data dict and index the data
        for key in D.keys():

            # Inputs is handled separately below because we always want to slice
            # it along the trial dimension.
            if key == 'inputs':
                newD[key] = {}
                continue

            try:
                if N == D[key].shape[0]:
                    # Common case: trial-aligned arrays are sliced by the fold
                    # train/test indices.
                    newD[key] = D[key][ind]
                else:
                    # Non-trial-aligned arrays (or scalars) are copied as-is.
                    newD[key] = D[key].copy()
            except:
                # If the value doesn't support `.shape`/slicing, keep it
                # unchanged for both train and test dicts.
                newD[key] = D[key]
        
        # `inputs` is assumed to be trial-aligned with the same first dimension
        # as `D['y']`, so we always slice it by `train` and `test`.
        for i in D['inputs'].keys():
            newD['inputs'][i] = D['inputs'][i][ind]

    
    return newD


def crossValidate_old(D, hyper_guess, weight_dict, optList,
                  F=10, seed=None, verbose=True, fix_hyper=False):
    """Calculates the xval loglikelihood and P(y=0) for each trial.
    
    Args:
        D: standard dataset
        weight_dict: name and count of which weights in D['inputs'] to fit. 
        hyper_guess: hyperparameters guess for hyperOpt()
        optList: hyperparameters in 'hyper' to be optimized
        F: Number of cross-validation folds
        seed: to replicate randomness of xval fold division.
        verbose: prints a progress message at end of each fold.
    
    Returns:
        xval_logli: float, the cross-validated loglikelihood of the model
        xval_pL: array, the x-val P(y=0) for each trial. For Gaussian or neural, the estimate average response
    """

    # Split the dataset into F train/test folds.
    # `split_data()` returns lists of length F, where each element is the
    # (trainD, testD) dict for one held-out fold.
    train_dats, test_dats = split_data(D, F=F, seed=seed)

    # determine model type
    if 'tr_start' in D:
        model_type = 'neural'
    elif np.unique(D['y']).shape[0]>2:
        model_type = 'gaussian'
    else:
        model_type = 'standard'

    # Total (summed) cross-validated log-likelihood across all held-out trials.
    xval_logli = 0
    logli_per_fold = np.zeros(F)
    # Collect per-fold per-trial "weight mode" outputs so we can later
    # reorder them back to the original trial ordering.
    all_gw = []
    w_cv = []
    for f in range(F):
        if verbose:
            print("\rRunning xval fold " + str(f+1) + " of " + str(F), end="")

        # Inner step: fit hyperparameters (and weights) using only the current
        # training fold.
        #
        # `hyperOpt()` returns several quantities; for CV we only need the
        # mode of the fitted weights (`wMode`) to compute held-out likelihood.
        if fix_hyper:
            wMode, _, _, _ = getMAP(train_dats[f], hyper_guess, weight_dict, method=None, E0=None, showOpt=0)
        else:
            _, _, wMode, _ = hyperOpt(train_dats[f], hyper_guess, weight_dict,
                                  optList, hess_calc=None)

        # Outer evaluation: compute held-out log-likelihood and predicted
        # weight-mode contribution (`gw`) for the missing trials in `test_dats[f]`.
        logli, gw, test_W = xval_loglike(test_dats[f], wMode,
                                 train_dats[f]['missing_trials'], weight_dict)

        # `logli` is per-held-out trial; sum to accumulate the global CV score.
        xval_logli += np.sum(logli)
        logli_per_fold[f] = np.sum(logli)

        # `gw` is per-held-out trial for this fold; store it for later
        # concatenation/reordering.
        # check if gw is a list
        if isinstance(gw, list):
            all_gw.extend(gw)
        else:
            all_gw += [gw]
        
        # store test_W
        w_cv.append(test_W)
    w_cv = np.concatenate(w_cv, axis=1)
    test_inds = np.array([i['test_inds'] for i in test_dats]).flatten()
    inds = np.argsort(test_inds)
    if model_type == 'standard':
        # Flatten collected held-out predictions into a single array, then reorder
        # them to match the original trial index order in `D['y']`.
        xval_gw = np.array(all_gw).flatten()
        
        xval_gw = xval_gw[inds]
        w_cv = w_cv[:,inds]

        # For the standard (Bernoulli/logistic) model, the probability of
        # `y=0` is `1 / (1 + exp(gw))`.
        xval_pL = 1 / (1 + np.exp(xval_gw))
    elif model_type == 'neural':
        # return xval_logli, test_dats, all_gw
        test_inds = np.concatenate([test_dats[f]['test_inds'] for f in range(F)])
        # xval_pL = np.array([])
        # for i in range(len(all_gw)):
        #     xval_pL = np.concatenate([xval_pL, all_gw[np.where(test_inds==i)[0][0]]])
        all_gw = [all_gw[i] for i in inds]
        xval_pL = np.concatenate(all_gw, axis=0)
        w_cv = w_cv[:,inds]
    elif model_type == 'gaussian':
        raise Exception('wip')
    
    # save some cross validation info
    cv_info = {
        'test_inds': np.array([i['test_inds'] for i in test_dats]),
        'w_cv': w_cv,
        'logli_per_fold': logli_per_fold,
    }
    
    
    return xval_logli, xval_pL, cv_info
    

    
def crossValidate(D, hyper_guess, weight_dict, optList,
                  F=10, seed=None, verbose=True, fix_hyper=False):
    """Calculates the xval loglikelihood and P(y=0) for each trial.
    
    Args:
        D: standard dataset
        weight_dict: name and count of which weights in D['inputs'] to fit. 
        hyper_guess: hyperparameters guess for hyperOpt(). Can be either:
            - dict: a single initialization (backward compatible)
            - list/tuple of dicts: multiple initializations; for each CV fold we
              fit from each guess on the *training* fold and select the one
              with the best training evidence (logEvd), then evaluate on the
              held-out fold.
        optList: hyperparameters in 'hyper' to be optimized
        F: Number of cross-validation folds
        seed: to replicate randomness of xval fold division.
        verbose: prints a progress message at end of each fold.
        fix_hyper: if True, skip hyperparameter optimization and only fit MAP
            weights for the provided hyperparameters. If multiple guesses are
            provided, selects the one with best training evidence per fold.
    
    Returns:
        xval_logli: float, the cross-validated loglikelihood of the model
        xval_pL: array, the x-val P(y=0) for each trial. For Gaussian or neural, the estimate average response
    """

    # Allow multi-start hyperparameter initialization. Backward compatible
    # behavior for a single dict.
    if isinstance(hyper_guess, (list, tuple)):
        hyper_guesses = list(hyper_guess)
    else:
        hyper_guesses = [hyper_guess]
    if len(hyper_guesses) == 0:
        raise ValueError("hyper_guess must be a dict or a non-empty list/tuple of dicts")

    # Split the dataset into F train/test folds.
    # `split_data()` returns lists of length F, where each element is the
    # (trainD, testD) dict for one held-out fold.
    train_dats, test_dats = split_data(D, F=F, seed=seed)

    # determine model type
    if 'tr_start' in D:
        model_type = 'neural'
    elif np.unique(D['y']).shape[0]>2:
        model_type = 'gaussian'
    else:
        model_type = 'standard'

    # Total (summed) cross-validated log-likelihood across all held-out trials.
    xval_logli = 0
    logli_per_fold = np.zeros(F)
    # Track held-out log-likelihood per fold *and* per hyperparameter guess.
    # Shape: (F, G) where G = len(hyper_guesses). Values are summed over the
    # held-out trials of that fold (consistent with `logli_per_fold`).
    logli_per_fold_per_guess = np.full((F, len(hyper_guesses)), np.nan, dtype=float)
    # Collect per-fold per-trial "weight mode" outputs so we can later
    # reorder them back to the original trial ordering.
    all_gw = []
    w_cv = []
    chosen_guess_idx = np.zeros(F, dtype=int)
    chosen_hyper_per_fold = []
    for f in range(F):
        if verbose:
            print("\rRunning xval fold " + str(f+1) + " of " + str(F), end="")

        # Inner step: fit hyperparameters (and weights) using only the current
        # training fold.
        #
        # `hyperOpt()` returns several quantities; for CV we only need the
        # mode of the fitted weights (`wMode`) to compute held-out likelihood.
        best_logEvd = None
        best_wMode = None
        best_hyper = None
        best_guess_idx = 0
        for g_idx, hg in enumerate(hyper_guesses):
            if fix_hyper:
                wMode_tmp, _, logEvd_tmp, _ = getMAP(
                    train_dats[f], hg, weight_dict, method=None, E0=None, showOpt=0
                )
                best_hyper_tmp = hg
            else:
                best_hyper_tmp, logEvd_tmp, wMode_tmp, _ = hyperOpt(
                    train_dats[f], hg, weight_dict, optList, hess_calc=None
                )

            # Save held-out log-likelihood for this guess on this fold.
            # Note: uses the fold's test set with weights fit on the training fold.
            logli_tmp, _, _ = xval_loglike(
                test_dats[f], wMode_tmp, train_dats[f]['missing_trials'], weight_dict
            )
            logli_per_fold_per_guess[f, g_idx] = float(np.sum(logli_tmp))
            print(f"fold {f} guess {g_idx} ", flush=True)

        #     if (best_logEvd is None) or (logEvd_tmp >= best_logEvd):
        #         best_logEvd = logEvd_tmp
        #         best_wMode = wMode_tmp
        #         best_hyper = best_hyper_tmp
        #         best_guess_idx = g_idx

        # wMode = best_wMode
        # chosen_guess_idx[f] = best_guess_idx
        # chosen_hyper_per_fold.append(best_hyper)

        # Outer evaluation: compute held-out log-likelihood and predicted
        # weight-mode contribution (`gw`) for the missing trials in `test_dats[f]`.
        # logli, gw, test_W = xval_loglike(test_dats[f], wMode,
        #                          train_dats[f]['missing_trials'], weight_dict)

        # # `logli` is per-held-out trial; sum to accumulate the global CV score.
        # xval_logli += np.sum(logli)
        # logli_per_fold[f] = np.sum(logli)

        # `gw` is per-held-out trial for this fold; store it for later
        # concatenation/reordering.
        # check if gw is a list
        # if isinstance(gw, list):
        #     all_gw.extend(gw)
        # else:
        #     all_gw += [gw]
        
        # store test_W
        # w_cv.append(test_W)
    # w_cv = np.concatenate(w_cv, axis=1)
    test_inds = np.array([i['test_inds'] for i in test_dats]).flatten()
    inds = np.argsort(test_inds)
    if model_type == 'standard':
        # Flatten collected held-out predictions into a single array, then reorder
        # them to match the original trial index order in `D['y']`.
        xval_gw = np.array(all_gw).flatten()
        
        xval_gw = xval_gw[inds]
        w_cv = w_cv[:,inds]

        # For the standard (Bernoulli/logistic) model, the probability of
        # `y=0` is `1 / (1 + exp(gw))`.
        xval_pL = 1 / (1 + np.exp(xval_gw))
    elif model_type == 'neural':
        # return xval_logli, test_dats, all_gw
        test_inds = np.concatenate([test_dats[f]['test_inds'] for f in range(F)])
        # xval_pL = np.array([])
        # for i in range(len(all_gw)):
        #     xval_pL = np.concatenate([xval_pL, all_gw[np.where(test_inds==i)[0][0]]])
        # all_gw = [all_gw[i] for i in inds]
        xval_pL = None # np.concatenate(all_gw, axis=0)
        # w_cv = w_cv[:,inds]
    elif model_type == 'gaussian':
        raise Exception('wip')
    
    # save some cross validation info
    cv_info = {
        'test_inds': np.array([i['test_inds'] for i in test_dats]),
        'w_cv': w_cv,
        'logli_per_fold': logli_per_fold,
        'logli_per_fold_per_guess': logli_per_fold_per_guess,
        'chosen_guess_idx': chosen_guess_idx,
        'chosen_hyper_per_fold': chosen_hyper_per_fold,
    }
    
    
    return np.sum(logli_per_fold_per_guess, axis=0), xval_pL, cv_info
    

def split_data(D, F=10, seed=None):
    '''Divides data into F sets of train/test sets.
    
    Splits a dataset into F folds, then save each individual fold
    as a test set with the other F-1 folds as a training set. Returns
    a list of F training datasets and F corresponding testing datasets

    Args:
        D : dict, data to be split into folds for cross-validation
        F : int, number of folds
        seed : int, random seed to reproduce xval fold split

    Returns:
        K_trainD : list, contains each fold's training dataset
        K_testD : list, contains each fold's testing dataset
    '''
    
    # # save required variables (D, F, seed)
    # import pickle
    # to_save = {'D': D, 'F': F, 'seed': seed}
    # with open('/usr/people/bichanw/SpikeSorting/Codes/psytrack/data/split_data.pkl', 'wb') as f:
    #     pickle.dump(to_save, f)
    # raise Exception('Stop here')

    # We want a reproducible fold split given `seed`.
    np.random.seed(seed)

    # Determine model type
    if 'tr_start' in D:
        model_type = 'neural'
    else:
        model_type = 'standard'    
    # elif 'sigmay' in D['hyper']:
    #     model_type = 'gaussian'
    # else:

    # The CV logic in this function treats each row of `D['y']` as a
    # single "trial unit" (so trial indices are 0..N-1).
    if model_type == 'neural' or 'tr_start' in D:
        N = D['tr_start'].shape[0] - 1
    else:
        N = D['y'].shape[0]

    shuffled_array = np.arange(N)
    np.random.shuffle(shuffled_array)
    
    # For this implementation, we require equal fold sizes so that
    # `chunk = N / F` is an integer.
    if N % F:
        raise Exception(
            "The number of trials in the data set N, " + str(N) + ",must be "
            "divisible by the number of folds F," + str(F) + ". Try using the "
            "trim() function to shave the last few trials off of the dataset."
            )

    # Accumulate per-fold train/test dicts.
    K_trainD = []
    K_testD = []
    for k in range(F):

        # Define the k-th train/test split by slicing `shuffled_array`
        # into contiguous chunks (in the shuffled trial ordering).
        N_array = np.arange(N)
        chunk = int(N / F)

        # `test` are trial indices (in the original dataset) held out for
        # this fold. We sort them so that downstream logic sees trials in
        # increasing original time order.
        test = np.sort(shuffled_array[k * chunk : (k + 1) * chunk])
        train = np.delete(N_array, test)

        # `missing_trials` encodes, for each training trial in `train`,
        # how many consecutive held-out test trials occur immediately after it
        # in the original time ordering.
        #
        # This is needed by `xval_loglike()`, which "replays" predictions over
        # the test trials and uses the preceding training weights. The details
        # are implemented via the `train_array` construction below.
        train_array = np.zeros(N)
        test2 = test.copy()
        while len(test2) > 0:
            train_array[test2] += 1
            test2 = np.array([i for i in test2 if i - 1 in test2])

        # `train_array` is currently indexed in a test-centric way; slice it
        # down to the training indices so `missing_trials` aligns with `train`.
        if 0 not in train:
            train_array = train_array[train - 1]
        else:
            # Special-case: if trial 0 is in `train`, there is no "previous trial"
            # to anchor the missing count, so we pad with 0 and shift accordingly.
            train_array = np.hstack(([0], train_array[train[1:] - 1]))

        # If `dayLength` exists, it indicates where "day boundaries" occur.
        # The CV split can accidentally separate trials across days in a way that
        # breaks continuity assumptions in the model; the block below shifts
        # any day-boundary overlap in the test set back into training.
        if 'dayLength' in D and D['dayLength'].shape[0] > 0:
            day_array = np.zeros(N)
            cumDays = np.cumsum(D['dayLength'], dtype=int)[:-1]
            day_array[cumDays] = 1
            overlap = np.array([i for i in test if i in cumDays])
            while len(overlap) > 0:
                day_array[overlap + 1] = 1
                overlap = np.array([i + 1 for i in overlap if i + 1 in test])
            day_array = day_array[train]
            days = np.hstack((np.where(day_array)[0], [len(day_array)]))
            new_dayLength = np.hstack((days[0], np.diff(days)))
        else:
            # If there's no day structure, downstream code expects an empty array.
            new_dayLength = np.array([])

        trainD = index_data(D, train, model_type)
        testD  = index_data(D, test, model_type)

        # Store the computed per-training-trial gap counts needed by
        # `xval_loglike()`, plus the explicit test indices for later reordering.
        trainD.update({'missing_trials': train_array, 'dayLength': new_dayLength})
        testD.update({'test_inds': test})

        # Append this fold's train/test dicts.
        K_trainD += [trainD]
        K_testD += [testD]

    return K_trainD, K_testD


def xval_loglike(testD, wMode, missing_trials, weights):
    '''Calculates xval log-likelihood of held out trials.
    
    Calculates the log-likelihood and gw value of each trial in a
    test set given the wMode recovered from a corresponding training set. 

    Args:
        testD : dict, test data
        wMode : (K, trainN) array, weights recovered from training set
        missing_trials : (N*(F-1)/F,) array, indices corresponding to 
            each trial in the training set, with the value indicating
            how many test trials followed it in the original dataset
        weights : dict, name and count of weights in testD['inputs'] 
            to fit

    Returns:
        logli : array, each test trial's log-likelihood
        all_gw : array, each test trial's gw value
    '''

    ### Form input matrix g from test set
    g = read_input(testD, weights)
    _, trainN = wMode.shape

    if 'tr_start' in testD:
        model_type = 'neural'
    elif 'hyper' in testD and 'sigmay' in testD['hyper']:
        model_type = 'gaussian'
    else:
        model_type = 'standard'
    
    if model_type == 'standard':
        logli = []
        all_gw = []
        test_W = []
        test_count = 0  # trial in the test set
        for t in range(trainN):  # iterate through each training trial
            
            # if training trial followed by one or more test trials
            for _ in range(int(missing_trials[t])):  

                ### Currently use the weights form the nearest prior training
                ### trial, could do interpolation...
                gw = g[test_count] @ wMode[:, t]
                yt = int(testD['y'][test_count]) - 1

                ### Save loglikelihood and gw value of each term in test set
                logli += [yt * gw - np.logaddexp(0, gw)]
                all_gw += [gw]
                test_W += [wMode[:, t]]

                ### Increment tracker of test trial index
                test_count += 1

        # Account for test trials at end
        for _ in range(len(g) - np.sum(missing_trials, dtype=int)):
                
            ### Use last training weights
            gw = g[test_count] @ wMode[:, -1]
            yt = int(testD['y'][test_count]) - 1

            ### Save loglikelihood and gw value of each term in test set
            logli += [yt * gw - np.logaddexp(0, gw)]
            all_gw += [gw]
            test_W += [wMode[:, -1]]

            ### Increment tracker of test trial index
            test_count += 1
        logli = np.array(logli)
        all_gw = np.array(all_gw)
        test_W = np.array(test_W)
    elif model_type == 'neural':
        
        # interpolate weights to test set
        # should take care of trials at the end as well
        test_W = np.zeros((wMode.shape[0], testD['test_inds'].shape[0]))
        for i in range(wMode.shape[0]):
            test_W[i,:] = np.interp(testD['test_inds'], 
                np.setdiff1d(np.arange(testD['test_inds'].shape[0]+missing_trials.shape[0]), testD['test_inds']), 
                wMode[i])
        
        # build full weight matrix by repeating the weights for each trial
        test_Wfull = np.zeros((g.shape[1], g.shape[0]))
        for i in range(testD['tr_start'].shape[0]-1):
            test_Wfull[:, testD['tr_start'][i]:testD['tr_start'][i+1]] = test_W[:, i][:,None]

        # predict response
        y_pred = (g * test_Wfull.T).sum(axis=1)
        err = (testD['y'] - y_pred)**2

        # sum each trial up
        # !!! use mean squared error for now. Not comparable across dataset?
        logli = np.zeros(testD['tr_start'].shape[0]-1)
        for i in range(testD['tr_start'].shape[0]-1):
            logli[i] = -err[testD['tr_start'][i]:testD['tr_start'][i+1]].sum() # using negative error for consistency

        # calculate predicted response
        all_gw = []
        # cut into trials
        for i in range(testD['tr_start'].shape[0]-1):
            all_gw += [y_pred[testD['tr_start'][i]:testD['tr_start'][i+1]]]


    return logli, all_gw, test_W
