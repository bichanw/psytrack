import numpy as np
import matplotlib.pyplot as plt
from psytrack.neuralSim import generateNeuralSim
from psytrack.helper.crossValidation import crossValidate


seed = np.random.seed()
D, weights, hyp, evd, wMode, hess_info = generateNeuralSim(if_run_estimation=False,boundary=1e3, n_tr = 5000, hyper = {'sigma': [.5,.5,.5]})
F = 10
hyper_guess = weights['hyp'].copy()
weight_dict = weights['weights'].copy()


## hyperOpt by cv
test_hyp = 2.0** np.arange(-3,3)
# test_hyp = 2.0** np.arange(-7,6,3)
K = np.sum([weight_dict[i] for i in weight_dict.keys()])

test_list = []
for hyp in test_hyp:
    hyper_guess['sigma'] = [hyp]*K
    test_list.append(hyper_guess.copy())
logli_list, xval_pL, cv_info = crossValidate(D, test_list, weight_dict, None, F = F, seed = seed, fix_hyper=True)


from psytrack.helper.my_util import plot_multiple_lines
plt.close()
fig, ax = plt.subplots(1, 1, layout='constrained')
fig.set_size_inches(4,3)
ax.plot(np.log2(test_hyp), cv_info['logli_per_fold_per_guess'].T,'k--',alpha=.3)
# ax.plot(np.log2(test_hyp), logli_list)
# plot_multiple_lines(data=cv_info['logli_per_fold_per_guess'].T - cv_info['logli_per_fold_per_guess'].mean(axis=0)[:,None], ax = ax, x = np.log2(test_hyp))
plot_multiple_lines(data=cv_info['logli_per_fold_per_guess'].T, ax = ax, x = np.log2(test_hyp))
max_idx = np.argmax(logli_list)
ax.axvline(x=np.log2(weights['hyp']['sigma'][0]), color='k', linestyle='--',label='true')
ax.plot(np.log2(test_hyp[max_idx]), logli_list[max_idx] / cv_info['logli_per_fold_per_guess'].shape[0], 'ro', label='max')
ax.set(xlabel = 'sigma (log2) tested', ylabel = 'log-likelihood')
ax.legend()
fig.savefig('sigma_1.png')




def test_cross_validate_sigma_dead_channel():
    ## test that cross-validation converges to sigma ~ 0 for a signal source
    ## that does not actually exist (i.e. a channel whose true weight is
    ## pinned at 0 the whole session, so it never drives y even though its
    ## input cues still occur)
    K = 2
    true_hyper = {
        'sigma':   [2**-4, 2**-4, 1e-8],  # bias, real channel, non-existent channel
        'sigInit': [2**4,  2**4,  1e-8],
        'sigmay': 0.1,
    }
    D2, weights2, _, _, _, _ = generateNeuralSim(
        K=K, if_run_estimation=False, boundary=1e3, n_tr=1000, hyper=true_hyper)

    weight_dict2 = weights2['weights'].copy()
    hyper_guess2 = weights2['hyp'].copy()

    # grid search only over the sigma of the non-existent channel (last entry),
    # holding bias/real-channel sigma fixed at their true values
    test_hyp_dead = np.array([1e-6, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

    test_list2 = []
    for s in test_hyp_dead:
        hg = hyper_guess2.copy()
        hg['sigma'] = list(true_hyper['sigma'][:-1]) + [s]
        test_list2.append(hg)
    F = 10
    seed = np.random.seed()
    logli_list2, xval_pL2, cv_info2 = crossValidate(
        D2, test_list2, weight_dict2, None, F=F, seed=seed, fix_hyper=True)

    best_sigma = test_hyp_dead[np.argmax(logli_list2)]

    plt.close()
    plt.plot(np.log2(test_hyp_dead), logli_list2)
    max_idx2 = np.argmax(logli_list2)
    plt.axvline(x=np.log2(true_hyper['sigma'][-1]), color='k', linestyle='--', label='true')
    plt.plot(np.log2(test_hyp_dead[max_idx2]), logli_list2[max_idx2], 'ro', label='max')
    plt.xlabel('sigma (log2)');
    plt.ylabel('log-likelihood');
    plt.legend()
    plt.savefig('tmp2.png')

    assert best_sigma == test_hyp_dead.min(), (
        f"Expected cross-validated log-likelihood to be maximized at the smallest "
        f"tested sigma (~0) for the non-existent signal channel, but the best "
        f"sigma found was {best_sigma}"
    )






def test_cross_validate_sigma_grid_per_weight():
    from itertools import product
    from psytrack.helper.crossValidation import produce_grid
    # Simulate small neural dataset with known per-weight sigmas
    true_sigma = [2**-6, 2**-4, 2**-2]
    Dg, wg, _, _, _, _ = generateNeuralSim(
        if_run_estimation=False,
        boundary=1e3,
        n_tr=1000,
        hyper={'sigma': true_sigma}
    )

    F = 10
    seed = 0
    hyper_guess = wg['hyp'].copy()
    weight_dict = wg['weights'].copy()
    K = int(np.sum([weight_dict[k] for k in weight_dict.keys()]))

    # Per-weight search grids (different values for different sigma entries)
    # Keep these small so the test runs in reasonable time.
    sigma_grid_by_index = {
        0: produce_grid(mid=-6, step=1, num=3),  # [2^-7, 2^-6, 2^-5]
        1: produce_grid(mid=-4, step=1, num=3),  # [2^-5, 2^-4, 2^-3]
        2: produce_grid(mid=-2, step=1, num=3),  # [2^-3, 2^-2, 2^-1]
    }

    search_indices = [0, 1, 2]
    sigma_grids_in_order = [np.asarray(sigma_grid_by_index[i], dtype=float) for i in search_indices]

    # Build full sigma-vector grid (cv_data-style)
    test_list = []
    sigma_tuples = []
    sigma_base = list(hyper_guess['sigma'])
    assert len(sigma_base) == K

    for sigma_subvec in product(*sigma_grids_in_order):
        sigma_guess = sigma_base.copy()
        for idx, val in zip(search_indices, sigma_subvec):
            sigma_guess[idx] = float(val)

        hg = hyper_guess.copy()
        hg['sigma'] = sigma_guess
        test_list.append(hg)
        sigma_tuples.append(tuple(sigma_guess))

    logli_list, xval_pL, cv_info = crossValidate(
        Dg, test_list, weight_dict, None, F=F, seed=seed, fix_hyper=True
    )

    # Core checks: grid size and output shape consistency
    expected_n = int(np.prod([g.size for g in sigma_grids_in_order]))
    assert len(test_list) == expected_n
    assert len(set(sigma_tuples)) == expected_n
    assert logli_list.shape == (expected_n,)
    assert cv_info['logli_per_fold_per_guess'].shape == (F, expected_n)

    # Ensure this is truly "different per variable", not only uniform vectors
    assert any(len(set(t[:K])) > 1 for t in sigma_tuples), \
        "Expected at least one tested sigma vector with non-uniform values across weights."

    # Optional sanity: best sigma should come from the tested grid
    best_sigma = test_list[int(np.argmax(logli_list))]['sigma']
    for idx in search_indices:
        assert np.any(np.isclose(best_sigma[idx], sigma_grid_by_index[idx]))