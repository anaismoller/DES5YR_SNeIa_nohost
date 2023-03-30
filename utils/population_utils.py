import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import plot_utils as pu
from utils import data_utils as du
from utils import conf_utils as cu
from utils import science_utils as su
from utils import logging_utils as lu
from scipy.optimize import curve_fit, least_squares


def bin_df_fixed_grid(df, var="zHD", grid=None, group=False):
    # Binning df by var with var_step
    df[f"{var}_bin"] = pd.cut(df[var].copy(), grid)
    round_factor = 3
    centres_arr = (
        df[f"{var}_bin"]
        .apply(lambda x: round(x.mid, round_factor))
        .astype(float)
        .values
    )
    df[f"{var}_bin_centres"] = centres_arr

    if group:
        return df.groupby(f"{var}_bin").agg(["count", "median", "std"])
    else:
        return df


def get_migration_matrix(
    fits_sim, dump_sim, var="c", bins=None, verbose=True, path_plots=None, suffix=""
):
    """
    Compute migration matrix from generated parameters to fitted ones
    """

    # if path_plots:
    #     pu.distribution_plots_measured_vs_sim(
    #         fits_sim, dump_sim, var=var, prefix="simulated", path_plots=path_plots
    #     )

    bin_centres = np.round(bins[:-1] + ((bins[1] - bins[0]) / 2.0), 3)

    binned_simulated = bin_df_fixed_grid(dump_sim, var=f"S2{var}", grid=bins)

    size = len(bins) - 1
    migration_matrix = np.zeros((size, size), dtype=np.float32)
    migration_matrix_trans = np.zeros((size, size), dtype=np.float32)
    for i, k1 in enumerate(bin_centres):
        # simulated (row), fitted (column)
        sel_ids = binned_simulated[binned_simulated[f"S2{var}_bin_centres"] == k1][
            "SNID"
        ].values
        binned_fitted = bin_df_fixed_grid(
            fits_sim[fits_sim["SNID"].isin(sel_ids)].copy(), var=f"{var}", grid=bins
        )
        row = binned_fitted.groupby(f"{var}_bin").count().SNID.values
        norm = len(sel_ids) if len(sel_ids) != 0 else 1
        normalized_row = row / norm
        # beware! fill as columns
        migration_matrix[:, i] = normalized_row

    if path_plots:
        pu.plot_matrix(
            migration_matrix,
            i_var=var,
            j_var=f"S2{var}",
            ticks=bin_centres,
            path_plots=path_plots,
            suffix=suffix,
        )

    return migration_matrix


def select_df_interval(df, var="zHD", my_min=0, my_max=1.2):
    sel_df = df[(df[var] >= my_min) & (df[var] < my_max)].copy()
    return sel_df


def bif_gaussian(x, amp, bifurcation, sigma_L, sigma_H):
    """Bifurcated Gaussian Function
    Args:
        x (float or np.array)
    """
    sigma = np.ones_like(x)
    sigma = np.where(x < bifurcation, sigma_L * sigma, sigma_H * sigma)
    return amp * np.exp(-0.5 * np.square(x - bifurcation) / np.square(sigma))


def errfunc(params, m, bins, obs, debugmode=False, lstq=False, fix_amp=False):
    """
    Error function
    # Fixing amplitude of bif gaussian to 1
    """
    gaus = bif_gaussian(bins, 1, *params) if fix_amp else bif_gaussian(bins, *params)
    tmp = m @ gaus
    tmp_norm = abs(tmp.sum()) if abs(tmp.sum()) > 0 else 1
    tmp = tmp * obs.sum() / tmp_norm
    delta = obs - tmp
    error = np.sqrt(obs)
    error[error == 0.0] = 1.0
    ratio = delta / error
    chi2 = (ratio**2).sum()

    # bounds
    upper = np.round(bins[-1], 1)
    lower = np.round(bins[0], 1)
    if fix_amp:
        bifurcation, sigma_L, sigma_H = params
    else:
        amp, bifurcation, sigma_L, sigma_H = params
    if sigma_L < 0 or sigma_H < 0:
        # positive sigmas
        chi2 = np.inf
    if sigma_L > upper or sigma_H > upper:
        # sigmas
        chi2 = np.inf
    if bifurcation < lower or bifurcation > upper:
        # bifurcation
        chi2 = np.inf

    if lstq:
        # minimisation=chi2 (lstq)
        return chi2
    else:
        # maximisation=-chi2 (emcee)
        return -chi2


def get_intrinsic_emcee(
    migration_matrix, x_data, y_data, path_plots="./", fix_amp=False, suffix=""
):
    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps
    ndim = 3 if fix_amp == True else 4
    nwalkers = 12  # number of MCMC walkers
    nburn = 1000  # "burn-in" period to let chains stabilize
    nsteps = 10000  # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    np.random.seed(0)
    starting_guesses = np.random.random((nwalkers, ndim)) / 10

    # The smapler maximizes the function
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        errfunc,
        args=[migration_matrix, x_data, y_data],
        kwargs={"fix_amp": fix_amp},
    )
    # Clear and run the production chain.
    sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    lu.print_green("emcee sampler finished")

    for var in range(ndim):
        # print('.  plotting line time')
        plt.clf()
        plt.plot(sampler.chain[:, :, var].T, color="k", alpha=0.4)
        plt.savefig(f"{path_plots}/line-time_" + str(var) + suffix + ".png")
    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

    result_mean = [np.percentile(samples[:, i], 50) for i in range(samples.shape[1])]
    result_low = [
        np.percentile(samples[:, i], 50) - np.percentile(samples[:, i], 16)
        for i in range(samples.shape[1])
    ]
    result_high = [
        np.percentile(samples[:, i], 84) - np.percentile(samples[:, i], 50)
        for i in range(samples.shape[1])
    ]

    result = np.vstack([result_mean, result_low, result_high])

    return result


def plot_intrinsic_population(
    bin_centres,
    result,
    observed,
    migration_matrix,
    var="c",
    fix_amp=False,
    path_plots="./",
    suffix="",
):
    fig, ax = plt.subplots(1, 1)

    result_50 = result[0, :]
    result_16 = result[1, :]
    result_84 = result[2, :]
    # lower
    tmp_amp = 1 if fix_amp else result_50[-4] - result_16[-4]
    intrinsic_unc_low = bif_gaussian(
        bin_centres,
        tmp_amp,
        result_16[-3],
        result_16[-2],
        result_16[-1],
    )
    # higher
    tmp_amp = 1 if fix_amp else result_50[-4] + result_84[-4]
    intrinsic_unc_high = bif_gaussian(
        bin_centres,
        tmp_amp,
        result_84[-3],
        result_84[-2],
        result_84[-1],
    )
    # intrinsic
    amp = 1 if fix_amp else result_50[-4]

    intrinsic = bif_gaussian(
        bin_centres, amp, result_50[-3], result_50[-2], result_50[-1]
    )

    # plot reversed
    ax.plot(
        bin_centres,
        (migration_matrix @ intrinsic) / (migration_matrix @ intrinsic).sum(),
        label="intrinsic reversed to observed",
        color="grey",
    )
    # plot obs
    ax.errorbar(
        bin_centres,
        observed / observed.sum(),
        yerr=np.sqrt(observed) / observed.sum(),
        label="observed normalized",
        color="blue",
        fmt="o",
    )

    # plot uncertainties
    ax.fill_between(
        bin_centres,
        intrinsic_unc_low / intrinsic_unc_low.sum(),
        intrinsic_unc_high / intrinsic_unc_high.sum(),
        color="red",
        alpha=0.5,
    )

    # plot intirnsic
    ax.plot(
        bin_centres,
        intrinsic / intrinsic.sum(),
        label=print_result(result),
        color="red",
    )

    ax.set_xlabel(var)
    ax.legend()
    plt.savefig(f"{path_plots}/intrinsic_{var}{suffix}.png")
    plt.close("all")


def print_result(result):
    result_50 = result[0, :]
    result_16 = result[1, :]
    result_84 = result[2, :]
    intrinsic_str = (
        f"b={round(result_50[-3],2)}-{round(result_16[-3],2)}+{round(result_84[-3],2)} s_L={round(result_50[-2],2)}-{round(result_16[-2],2)}+{round(result_84[-2],2)} s_H={round(result_50[-1],2)}-{round(result_16[-1],2)}+{round(result_84[-1],2)}",
    )

    return intrinsic_str


def plot_inspection(
    fits_data, fits_sim_flat, ori_sim_flat, path_out="", suffix="", contamination=True
):
    """
    Suite of inspection plots
    """

    list_df = [fits_data, fits_sim_flat, ori_sim_flat]
    list_labels = ["observed", "simulated_fitted", "simulated_oris"]
    var_list = ["x1", "c"]
    for var in var_list:
        pu.plot_histograms_listdf(
            list_df,
            list_labels,
            varx=var,
            density=True,
            norm_factor=1,
            outname="%s/hist_%s%s.png" % (path_out, var, suffix),
        )
    # sim flat evolution vs. z
    # pu.overplot_salt_distributions(fits_data, fits_sim_flat, ori_sim_flat, path_plots=path_out)
