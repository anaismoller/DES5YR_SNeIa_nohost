import os, sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import cuts as cuts
from utils import plot_utils as pu
from utils import data_utils as du
from utils import conf_utils as cu
from utils import logging_utils as lu
from utils import science_utils as su

plt.switch_backend("agg")

import matplotlib as mpl

mpl.rcParams["font.size"] = 16
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["figure.titlesize"] = "large"
mpl.rcParams["lines.linewidth"] = 3


colors = ["grey"] + pu.ALL_COLORS
# SNIa parameters
Mb = 19.365
alpha = 0.144  # from sim
beta = 3.1


def setup_logging():
    logger = None

    # Create logger using python logging module
    logging_handler_out = logging.StreamHandler(sys.stdout)
    logging_handler_out.setLevel(logging.DEBUG)

    logging_handler_err = logging.StreamHandler(sys.stderr)
    logging_handler_err.setLevel(logging.WARNING)

    logger = logging.getLogger("localLogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging_handler_out)
    logger.addHandler(logging_handler_err)

    # create file handler which logs even debug messages
    fh = logging.FileHandler("logpaper.log", mode="w")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


def plot_scatter_mosaic_retro(
    list_df, list_labels, path_out="tmp.png", print_biases=False
):
    # scatter
    fig = plt.figure(constrained_layout=True, figsize=(17, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=False, sharey=False)

    for i, var in enumerate(["z", "c", "x1"]):
        varx = "zHD" if var == "z" else var
        vary = f"{var}PHOT_retro" if var == "z" else f"{var}_retro"
        lims = (0.1, 1.2) if var == "z" else ((-0.4, 0.4) if var == "c" else (-4, 4))
        for idx_df, df in enumerate(list_df):
            h2d, xedges, yedges, _ = axs[i].hist2d(
                df[varx],
                df[vary],
                bins=20,
                cmap="magma_r",  # cmap="YlOrRd"  # , cmin=0.25
            )
            # chech distribution for a given x bin
            tmp_bins = xedges
            mean_bins = tmp_bins[:-1] + (tmp_bins[1] - tmp_bins[0]) / 2
            df[f"{varx}_bin"] = pd.cut(df.loc[:, (varx)], tmp_bins, labels=mean_bins)
            result_median = (
                df[[f"{varx}_bin", varx]].groupby(f"{varx}_bin").median()[varx].values
            )
            # axs[i].scatter(mean_bins, result_median, marker="x")
            if print_biases:
                print(f"Biases for {varx}")
                print(
                    "bins", [round(mean_bins[i], 3) for i in range(len(mean_bins))],
                )
                print(
                    "bias",
                    [
                        round(mean_bins[i] - result_median[i], 3)
                        for i in range(len(mean_bins))
                    ],
                )
                perc = [
                    round(100 * (mean_bins[i] - result_median[i]) / mean_bins[i], 3)
                    for i in range(len(mean_bins))
                ]
                print(
                    "%", perc,
                )
                print(
                    f"% median {np.median(perc)}; max {np.max(perc)}; min {np.min(perc)}"
                )
        axs[i].plot(
            [df[varx].min(), df[varx].max()],
            [df[vary].min(), df[vary].max()],
            color="black",
            linewidth=1,
            linestyle="--",
            zorder=10,
        )
        xlabel = "true redshift" if var == "z" else f"{var} with true redshift"
        ylabel = "fitted redshift" if var == "z" else f"{var} with fitted redshift"
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_xlim(lims[0], lims[1])
        axs[i].set_ylim(lims[0], lims[1])

    # axs[i].legend(loc="best", prop={"size": 10})
    plt.savefig(path_out)


def plot_freez_correlations(list_df, list_labels=["tmp"], path_plots="./"):
    # scatter plot
    plot_scatter_mosaic_retro(
        list_df,
        list_labels,
        path_out=f"{path_plots}/scatter_retro_vs_ori.png",
        print_biases=True,
    )

    # histograms delta
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.4)
    axs = gs.subplots(sharex=False, sharey=True)

    for i, var in enumerate(["z", "c", "x1"]):
        varx = "zHD" if var == "z" else var
        vary = f"{var}PHOT_retro" if var == "z" else f"{var}_retro"
        lims = (0.1, 1.4) if var == "z" else ((-0.5, 0.5) if var == "c" else (-5, 5))
        for idx_df, df in enumerate(list_df):
            delta = df[vary] - df[varx]
            axs[i].hist(
                delta,
                color=colors[idx_df],
                label=list_labels[idx_df],
                histtype="step",
                linestyle=pu.list_linestyle[idx_df],
            )
            axs[i].plot(
                [delta.mean(), delta.mean()],
                [0, 100000],
                color=colors[idx_df],
                linestyle=pu.list_linestyle[idx_df],
            )
        axs[i].set_xlabel(f"{var} fitted -{var} sim")
        axs[i].set_xlim(lims[0], lims[1])
        axs[i].set_yscale("log")
    axs[i].legend(loc="best", prop={"size": 10})
    plt.savefig(f"{path_plots}/hist_delta_fitted_vs_sim.png")

    # scatter plot delta c - delta z
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    for idx_df, df in enumerate(list_df):
        if len(df) > 1000:
            df = df.sample(n=1000)
        plt.errorbar(
            df["zHD_retro"] - df["zHD"],
            df["c_retro"] - df["c"],
            # yerr=df["cERR_retro"] + df["cERR"],
            # xerr=df["zHD_retro"] + df["zHDERR"],
            fmt="o",
            color=colors[idx_df],
            label=list_labels[idx_df],
        )

    plt.xlabel(f"zHD fitted - zHD sim")
    plt.ylabel(f"c fitted - c sim")
    plt.legend(loc="best", prop={"size": 10})
    plt.savefig(f"{path_plots}/scatter_deltac_deltazfitted.png")

    # scatter plot delta c -  z simm
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    for idx_df, df in enumerate(list_df):
        if len(df) > 1000:
            df = df.sample(n=1000)
        plt.errorbar(
            df["zHD"],
            df["c_retro"] - df["c"],
            # yerr=df["cERR_retro"] + df["cERR"],
            # xerr=df["zHD_retro"] + df["zHDERR"],
            fmt="o",
            color=colors[idx_df],
            label=list_labels[idx_df],
        )

    plt.xlabel(f"zHD sim")
    plt.ylabel(f"c fitted - c sim")
    plt.legend(loc="best", prop={"size": 10})
    plt.savefig(f"{path_plots}/scatter_deltac_zHDsim.png")

    # plot c sim -> c fitted vs delta z
    # need to bin to see clearly tendencies
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=False)

    for idx_df, df in enumerate(list_df):
        df[f"zHD_bin"] = pd.cut(df.loc[:, ("zHD")], pu.bins_dic["zHD"])
        sel = df.groupby([f"zHD_bin"]).mean()
        for idx, row in sel.iterrows():
            axs[0].arrow(
                row["zHD"],
                beta * row["c"],
                row["zHD_retro"] - row["zHD"],
                beta * (row["c_retro"] - row["c"]),
                color="black",  # colors[idx_df],
                head_width=0.02 if "sim Ia" in list_labels[idx_df] else 0.02,
                width=0.002,
            )
    axs[0].set_ylabel(r"$\beta c$ with true redshift to fitted", fontsize=20)
    axs[0].axis("equal")
    for idx_df, df in enumerate(list_df):
        df["zHD_bin"] = pd.cut(df.loc[:, ("zHD")], pu.bins_dic["zHD"])
        sel = df.groupby([f"zHD_bin"]).mean()
        for idx, row in sel.iterrows():
            axs[1].arrow(
                row["zHD"],
                alpha * row["x1"],
                row["zHD_retro"] - row["zHD"],
                alpha * (row["x1_retro"] - row["x1"]),
                color="black",  # colors[idx_df],
                head_width=0.02 if "sim Ia" in list_labels[idx_df] else 0.02,
                width=0.002,
            )
    axs[1].axis("equal")
    axs[1].set_ylim(-0.04,0.04)
    axs[1].set_xlabel(f"true to fitted redshift", fontsize=20)
    axs[1].set_ylabel(r"$\alpha x_1$ with true redshift to fitted", fontsize=20)
    plt.savefig(f"{path_plots}/migration_cx1_zHD.png")


if __name__ == "__main__":

    DES5yr = os.getenv("DES5yr")
    DES = os.getenv("DES")

    parser = argparse.ArgumentParser(description="Code to reproduce results paper")

    parser.add_argument(
        "--path_dump",
        default=f"{DES}/DES5YR/DES5YR_SNeIa_nohost/dump_DES5YR",
        type=str,
        help="Path to output",
    )
    parser.add_argument(
        "--path_sim_headers",
        type=str,
        default=f"{DES}/DES5YR/snndump_1X_NOZ/1_SIM/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/",
        help="Path to simulation headers without host-efficiency SALT2 fits",
    )
    parser.add_argument(
        "--path_sim_fits",
        type=str,
        default=f"{DES}/DES5YR/snndump_1X_NOZ/2_LCFIT/",
        help="Path to simulation fits without host-efficiency SALT2 fits",
    )
    parser.add_argument(
        "--path_sim_class",
        type=str,
        default=f"{DES}/DES5YR/snndump_1X_NOZ/models/",
        help="Path to simulation predictions without host-efficiency SALT2 fits",
    )

    # Init
    args = parser.parse_args()
    path_dump = args.path_dump
    path_sim_headers = args.path_sim_headers

    path_plots = f"{path_dump}/plots_SALTbiases/"
    os.makedirs(path_plots, exist_ok=True)

    # logger
    logger = setup_logging()

    logger.info("")
    logger.info("SIMULATIONS: STATS + EVALUATE SIMULTANEOUS SALT2 FIT")
    lu.print_blue("Loading sims")
    # load predictions of snn
    sim_preds = du.load_merge_all_preds(
        path_class=args.path_sim_class,
        model_name="vanilla_S_*_none*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean",
        norm="cosmo_quantile",
    )
    lu.print_blue("Loading SALT2 fit NOT SIMULTANEOUS for comparison")
    # Load usual SALT2 fit (using available redshift)
    sim_fits = du.load_salt_fits(
        f"{args.path_sim_fits}/JLA_1XDES/output/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/FITOPT000.FITRES.gz"
    )
    sim_fits_JLA = su.apply_JLA_cut(sim_fits)
    sim_Ia_fits = sim_fits[sim_fits.SNTYPE.isin(cu.spec_tags["Ia"])]
    sim_Ia_fits_JLA = su.apply_JLA_cut(sim_Ia_fits)

    # Load z,x1,c SALT2 fit
    lu.print_blue("Loading SALT2 SIMULTANEOUS")
    sim_saltz = du.load_salt_fits(
        f"{args.path_sim_fits}/D_FITZ_1XDES/output/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/FITOPT000.FITRES.gz"
    )
    sim_saltz_JLA = su.apply_JLA_cut(sim_saltz)
    sim_saltz_Ia = sim_saltz[sim_saltz.SNTYPE.isin(cu.spec_tags["Ia"])]
    sim_saltz_Ia_JLA = su.apply_JLA_cut(sim_saltz_Ia)

    tmp_sim_saltz = sim_saltz.add_suffix("_retro")
    tmp_sim_saltz = tmp_sim_saltz.rename(columns={"SNID_retro": "SNID"})
    sim_all_fits = pd.merge(sim_fits, tmp_sim_saltz)

    # define simulated sample
    sim_preds_photoIa_noz = sim_preds["cosmo_quantile"][
        sim_preds["cosmo_quantile"]["average_probability_set_0"] > 0.5
    ]
    # predicted photo Ia
    sim_allfits_photoIa_noz = sim_all_fits[
        sim_all_fits.SNID.isin(sim_preds_photoIa_noz.SNID.values)
    ]
    sim_allfits_Ia = sim_all_fits[sim_all_fits.SNID.isin(sim_saltz_Ia.SNID.values)]

    # Inspect z,x1,c simultaneous fit
    plot_freez_correlations(
        [sim_allfits_Ia],  # , sim_allfits_photoIa_noz],
        list_labels=["sim Ia"],  # , "sim photoIa noz"],  # , "sim core-collapse"],
        path_plots=path_plots,
    )

    logger.info("")
    logger.info("SIMULATIONS: CONTAMINATION")
    # contamination and efficiency vs. true z and fitted z
    list_tuple = [
        (sim_preds, sim_fits),
        (sim_preds, sim_saltz),
        (sim_preds, sim_saltz_JLA),
    ]
    list_labels = [
        "true z",
        "fitted z",
        "fitted z + JLA-like cut",
    ]
    pu.plot_contamination_list(
        list_tuple, path_plots=path_plots, list_labels=list_labels, suffix="noz"
    )

    sim_fits_wpreds = pd.merge(sim_fits, sim_preds["cosmo_quantile"])
    sim_saltz_wpreds = pd.merge(sim_saltz, sim_preds["cosmo_quantile"])
    sim_saltz_wpreds_JLA = su.apply_JLA_cut(sim_saltz_wpreds)
    pu.plot_metrics_list(
        [sim_fits_wpreds, sim_saltz_wpreds, sim_saltz_wpreds_JLA],
        path_plots=path_plots,
        list_labels=["true z", "fitted z", "fitted z + JLA-like cut"],
        metric="efficiency",
    )

    # General stats
    sim_cut_nonIa = sim_preds["cosmo_quantile"]["target"] == 1
    total_contamination = []
    total_contamination_JLA = []
    total_contamination_JLA_saltz = []
    for classset in [0, 1, 2]:
        # photo Ia
        sim_cut_photoIa = (
            sim_preds["cosmo_quantile"][f"average_probability_set_{classset}"] > 0.5
        )
        sim_photoIa = sim_preds["cosmo_quantile"][sim_cut_photoIa]
        sim_photoIa_contamination = sim_preds["cosmo_quantile"][
            sim_cut_photoIa & sim_cut_nonIa
        ]
        total_contamination.append(
            100 * len(sim_photoIa_contamination) / len(sim_photoIa)
        )
        sel_cont = sim_photoIa_contamination[
            sim_photoIa_contamination.SNID.isin(sim_fits_JLA.SNID.values)
        ]
        sel_photo = sim_photoIa[sim_photoIa.SNID.isin(sim_fits_JLA.SNID.values)]
        total_contamination_JLA.append(100 * len(sel_cont) / len(sim_photoIa))
        sel_cont = sim_photoIa_contamination[
            sim_photoIa_contamination.SNID.isin(sim_saltz_JLA.SNID.values)
        ]
        sel_photo = sim_photoIa[sim_photoIa.SNID.isin(sim_saltz_JLA.SNID.values)]
        total_contamination_JLA_saltz.append(100 * len(sel_cont) / len(sim_photoIa))
    print(
        f"Total contamination no cuts {round(np.array(total_contamination).mean(),2)} +- {round(np.array(total_contamination).std(),2)}"
    )
    print(
        f"Total contamination JLA {round(np.array(total_contamination_JLA).mean(),2)} +- {round(np.array(total_contamination_JLA).std(),2)}"
    )
    print(
        f"Total contamination JLA saltz {round(np.array(total_contamination_JLA_saltz).mean(),2)} +- {round(np.array(total_contamination_JLA_saltz).std(),2)}"
    )

    logger.info("")
    logger.info("SIMULATIONS: CLASSIFICATION EFFICIENCY")
    # the simultaneous saltz fit may introduce some large biases

    # 1. comparing simulations: photoIa vs true Ia
    variable = "m0obs_i"
    min_var = sim_allfits_photoIa_noz[variable].quantile(0.01)
    df, minv, maxv = du.data_sim_ratio(
        sim_allfits_photoIa_noz,
        sim_allfits_Ia,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simfixedz_simphotoIa_noJLA",
    )
    df, minv, maxv = du.data_sim_ratio(
        sim_saltz[sim_saltz.SNID.isin(sim_photoIa.SNID.values)],
        sim_saltz_Ia,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simfittedz_simphotoIa_noJLA",
    )
    df, minv, maxv = du.data_sim_ratio(
        su.apply_JLA_cut(sim_saltz[sim_saltz.SNID.isin(sim_photoIa.SNID.values)]),
        sim_saltz_Ia_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simfittedz_simphotoIa_JLA",
    )
