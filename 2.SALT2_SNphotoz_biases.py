import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DES5yr = os.getenv("DES5yr")
DES = os.getenv("DES")

sys.path.insert(-1, f"{DES5yr}/DES5YR_SNeIa_nohost")
from myutils import plot_utils as pu
from myutils import data_utils as du
from myutils import conf_utils as cu
from myutils import logging_utils as lu
from myutils import science_utils as su

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


def plot_freez_correlations(list_df, list_labels=["tmp"], path_plots="./"):
    # scatter plot
    pu.plot_scatter_mosaic_SNphotoz_biases(
        list_df,
        list_labels,
        path_out=f"{path_plots}/scatter_SNphotoz_vs_ori.png",
        print_biases=True,
    )

    # histograms delta
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.4)
    axs = gs.subplots(sharex=False, sharey=True)

    for i, var in enumerate(["z", "c", "x1"]):
        varx = "zHD" if var == "z" else var
        vary = f"{var}PHOT_SNphotoz" if var == "z" else f"{var}_SNphotoz"
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
        axs[i].set_xlabel(f"{var} SNphoto -{var} sim s")
        axs[i].set_xlim(lims[0], lims[1])
        axs[i].set_yscale("log")
    axs[i].legend(loc="best", prop={"size": 10})
    plt.savefig(f"{path_plots}/hist_delta_SNphoto_vs_sim.png")

    # scatter plot delta c - delta z
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    for idx_df, df in enumerate(list_df):
        if len(df) > 1000:
            df = df.sample(n=1000)
        plt.errorbar(
            df["zHD_SNphotoz"] - df["zHD"],
            df["c_SNphotoz"] - df["c"],
            fmt="o",
            color=colors[idx_df],
            label=list_labels[idx_df],
        )

    plt.xlabel(f"zHD SNphoto z - zHD sim z")
    plt.ylabel(f"c SNphoto z - c sim z")
    plt.legend(loc="best", prop={"size": 10})
    plt.savefig(f"{path_plots}/scatter_deltac_deltaz.png")

    # scatter plot delta c -  z simm
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    for idx_df, df in enumerate(list_df):
        if len(df) > 1000:
            df = df.sample(n=1000)
        plt.errorbar(
            df["zHD"],
            df["c_SNphotoz"] - df["c"],
            fmt="o",
            color=colors[idx_df],
            label=list_labels[idx_df],
        )

    plt.xlabel(f"zHD sim")
    plt.ylabel(f"c SNphoto - c sim")
    plt.legend(loc="best", prop={"size": 10})
    plt.savefig(f"{path_plots}/scatter_deltac_zHDsim.png")

    # plot c sim -> c SNphoto vs delta z
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
                row["zHD_SNphotoz"] - row["zHD"],
                beta * (row["c_SNphotoz"] - row["c"]),
                color="black",  # colors[idx_df],
                head_width=0.02 if "sim Ia" in list_labels[idx_df] else 0.02,
                width=0.002,
            )
    axs[0].set_ylabel(r"$\Delta (\beta c))_{true,SNphoto \ z}$", fontsize=26)
    axs[0].axis("equal")
    for idx_df, df in enumerate(list_df):
        df["zHD_bin"] = pd.cut(df.loc[:, ("zHD")], pu.bins_dic["zHD"])
        sel = df.groupby([f"zHD_bin"]).mean()
        for idx, row in sel.iterrows():
            axs[1].arrow(
                row["zHD"],
                alpha * row["x1"],
                row["zHD_SNphotoz"] - row["zHD"],
                alpha * (row["x1_SNphotoz"] - row["x1"]),
                color="black",  # colors[idx_df],
                head_width=0.02 if "sim Ia" in list_labels[idx_df] else 0.02,
                width=0.002,
            )
    axs[1].axis("equal")
    axs[1].set_ylim(-0.04, 0.04)
    axs[1].set_xlabel(r"$z_{true}$", fontsize=26)
    axs[1].set_ylabel(r"$\Delta (\alpha x1)_{true,SNphoto \ z}$", fontsize=26)
    plt.savefig(f"{path_plots}/migration_cx1_zHD.png")


def plot_mosaic_scatter_SNphoto_zspe(df, outname, path_plots):

        lines_ranges = {"z": [0.2, 1.3], "c": [-0.4, 0.4], "x1": [-4, 4]}
        plot_ranges = {"z": [0.1, 1.3], "c": [-0.4, 0.4], "x1": [-4, 4]}

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 3, wspace=0.4, hspace=0.1, height_ratios=[1, 0.4])
        axs = gs.subplots(sharex="col", sharey=False)

        for i, var in enumerate(["z", "c", "x1"]):
            varx = "zHD" if var == "z" else var
            vary = f"{var}PHOT_SNphotoz" if var == "z" else f"{var}_SNphotoz"
            errvary = f"{var}PHOTERR_SNphotoz" if var == "z" else f"{var}ERR_SNphotoz"
            # first row: scatter
            axs[0][i].errorbar(
                df[varx],
                df[vary],
                xerr=df[f"{varx}ERR"],
                yerr=df[errvary],
                fmt="o",
                ecolor=None,
                markersize=1,
                alpha=0.05 if var=='z' else 0.01,
                color="indigo",
            )
            axs[0][i].plot(
                lines_ranges[var],
                lines_ranges[var],
                color="black",
                linewidth=1,
                linestyle="--",
                zorder=100,
            )
            ylabel = (
                r"${%s}_{\mathrm{SNphoto ~ z}}$" % var if var != "z" else "SNphoto z"
            )
            axs[0][i].set_ylabel(ylabel)
            axs[0][i].set_xlim(plot_ranges[var])
            axs[0][i].set_ylim(plot_ranges[var])
            # 2nd row: delta 
            df[f"delta {var}"] = df[varx] - df[vary]
            axs[1][i].scatter(
                df[varx],
                df[f"delta {var}"],
                alpha=0.03,
                color="indigo",
                s=10,
            )
            axs[1][i].plot(
                [lines_ranges[var][0], lines_ranges[var][1]],
                [0, 0],
                color="black",
                linewidth=1,
                linestyle="--",
                zorder=100,
            )

            axs[1][i].set_xlabel(
                r"${%s}_{\mathrm{true ~ z}}$" % var if var != "z" else "true z"
            )
            ylabel = r"$\Delta {%s}$" % var
            axs[1][i].set_ylabel(ylabel)
        plt.savefig(f"{path_plots}/{outname}.png")


def plot_mosaic_scatter_SNphoto_zspe_sim(df, outname, path_plots):

        lines_ranges = {"z": [0.2, 1.3], "c": [-0.4, 0.4], "x1": [-4, 4]}
        plot_ranges = {"z": [0.1, 1.3], "c": [-0.4, 0.4], "x1": [-4, 4]}

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 3, wspace=0.4, hspace=0.1, height_ratios=[1, 0.4])
        axs = gs.subplots(sharex="col", sharey=False)

        for i, var in enumerate(["z", "c", "x1"]):
            varx = "zHD" if var == "z" else var
            vary = f"{var}PHOT_SNphotoz" if var == "z" else f"{var}_SNphotoz"
            errvary = f"{var}PHOTERR_SNphotoz" if var == "z" else f"{var}ERR_SNphotoz"
            # first row: scatter
            axs[0][i].errorbar(
                df[varx],
                df[vary],
                xerr=df[f"{varx}ERR"],
                yerr=df[errvary],
                fmt="o",
                ecolor=None,
                markersize=1,
                alpha=0.05 if var=='z' else 0.01,
                color="indigo",
            )
            axs[0][i].plot(
                lines_ranges[var],
                lines_ranges[var],
                color="black",
                linewidth=1,
                linestyle="--",
                zorder=100,
            )
            ylabel = (
                r"${%s}_{\mathrm{SNphoto ~ z}}$" % var if var != "z" else "SNphoto z"
            )
            axs[0][i].set_ylabel(ylabel)
            axs[0][i].set_xlim(plot_ranges[var])
            axs[0][i].set_ylim(plot_ranges[var])
            # 2nd row: delta 
            df[f"delta {var}"] = df[varx] - df[vary]
            axs[1][i].scatter(
                df[varx],
                df[f"delta {var}"],
                alpha=0.03,
                color="indigo",
                s=10,
            )
            axs[1][i].plot(
                [lines_ranges[var][0], lines_ranges[var][1]],
                [0, 0],
                color="black",
                linewidth=1,
                linestyle="--",
                zorder=100,
            )

            axs[1][i].set_xlabel(
                r"${%s}_{\mathrm{true ~ z}}$" % var if var != "z" else "true z"
            )
            ylabel = r"$\Delta {%s}$" % var
            axs[1][i].set_ylabel(ylabel)
        plt.savefig(f"{path_plots}/{outname}.png")


if __name__ == "__main__":

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

    path_plots = f"{path_dump}/plots_SNphotoz_biases/"
    os.makedirs(path_plots, exist_ok=True)

    # logger
    logger = setup_logging()

    logger.info("")
    logger.info("SIMULATIONS: EVALUATE SNPHOTO Z SALT2")
    lu.print_blue("Loading sims")
    # load predictions of snn
    sim_preds = du.load_merge_all_preds(
        path_class=args.path_sim_class,
        model_name="vanilla_S_*_none*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean",
        norm="cosmo_quantile",
    )
    lu.print_blue("Loading SALT2 fit with sim redshift for comparison")
    # Load usual SALT2 fit (using available redshift)
    sim_fits = du.load_salt_fits(
        f"{args.path_sim_fits}/JLA_1XDES/output/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/FITOPT000.FITRES.gz"
    )
    sim_fits_JLA = su.apply_JLA_cut(sim_fits)
    sim_Ia_fits = sim_fits[sim_fits.SNTYPE.isin(cu.spec_tags["Ia"])]
    sim_Ia_fits_JLA = su.apply_JLA_cut(sim_Ia_fits)

    # Load z,x1,c SALT2 fit
    lu.print_blue("Loading SALT2 fit with SNphoto z")
    sim_saltz = du.load_salt_fits(
        f"{args.path_sim_fits}/D_FITZ_1XDES/output/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/FITOPT000.FITRES.gz"
    )
    sim_saltz_JLA = su.apply_JLA_cut(sim_saltz)
    sim_saltz_Ia = sim_saltz[sim_saltz.SNTYPE.isin(cu.spec_tags["Ia"])]
    sim_saltz_Ia_JLA = su.apply_JLA_cut(sim_saltz_Ia)

    tmp_sim_saltz = sim_saltz.add_suffix("_SNphotoz")
    tmp_sim_saltz = tmp_sim_saltz.rename(columns={"SNID_SNphotoz": "SNID"})
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
    plot_mosaic_scatter_SNphoto_zspe(
        sim_allfits_Ia.sample(n=10000, random_state=1), "scatter_SNphotoz_vs_true", path_plots
    )
    sim_allfits_Ia["delta z"] = sim_allfits_Ia['zHD'] - sim_allfits_Ia["zPHOT_SNphotoz"]
    # sim_allfits_Ia["delta z/(1+z)"] = sim_allfits_Ia["delta z"]/(1+sim_allfits_Ia["zHD"])
    sim_allfits_Ia["delta c"] = sim_allfits_Ia['c'] - sim_allfits_Ia["c_SNphotoz"]
    sim_allfits_Ia["delta x1"] = sim_allfits_Ia['x1'] - sim_allfits_Ia["x1_SNphotoz"]

    for var in ["delta z", 'delta c','delta x1']:
        print("")
        print(var)
        # Calculate quantiles
        q25 = np.quantile(sim_allfits_Ia[var], 0.25)  # 25th percentile (first quartile)
        q50 = np.quantile(sim_allfits_Ia[var], 0.5)   # 50th percentile (median)
        q75 = np.quantile(sim_allfits_Ia[var], 0.75)  # 75th percentile (third quartile)
        print(f"25th, median, 75th: {np.round(q25,3)} {np.round(q50,3)} {np.round(q75,3)}")

        # Calculate quartiles and IQR
        q1, q3 = np.percentile(sim_allfits_Ia[var], [25, 75])
        iqr = q3 - q1
        # Define outlier thresholds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = sim_allfits_Ia[var][(sim_allfits_Ia[var] < lower_bound) | (sim_allfits_Ia[var] > upper_bound)]
        outlier_fraction = len(outliers) / len(sim_allfits_Ia[var])
        print(f"Outlier fraction: {outlier_fraction:.4f}")

    plot_freez_correlations(
        [sim_allfits_Ia],
        list_labels=["sim Ia"],
        path_plots=path_plots,
    )

    # Hubble Residuals
    sim_allfits_Ia = su.distance_modulus(sim_allfits_Ia)
    # mB in SNphotoz is wrong, using the one without it
    sim_allfits_Ia['mB_SNphotoz'] = sim_allfits_Ia['mB']
    sim_allfits_Ia = su.distance_modulus(sim_allfits_Ia, suffix="SNphotoz")
    sim_allfits_Ia_JLA = su.apply_JLA_cut(sim_allfits_Ia)

    tmp = sim_allfits_Ia.copy()
    tmp['x1']= tmp['x1_SNphotoz']
    tmp['x1ERR']= tmp['x1ERR_SNphotoz']
    tmp['c']= tmp['c_SNphotoz']
    tmp['FITPROB']= tmp['FITPROB_SNphotoz']
    tmp['PKMJDERR']= tmp['PKMJDERR_SNphotoz']
    sim_allfits_Ia_JLA_SNphotoz = su.apply_JLA_cut(tmp)


    pu.plot_HD_residuals(sim_allfits_Ia_JLA.sample(n=5000, random_state=1),sim_allfits_Ia_JLA_SNphotoz.sample(n=5000, random_state=1), f"{path_plots}/HR.png")

    # stats
    print('Delta LCDM (delmu) with JLA cuts')
    print('true z',round(sim_allfits_Ia_JLA['delmu'].median(),2), round(sim_allfits_Ia_JLA['delmu'].std(),2))
    print('SNphotoz',round(sim_allfits_Ia_JLA['delmu_SNphotoz'].median(),2), round(sim_allfits_Ia_JLA['delmu_SNphotoz'].std(),2))
    print('Delta LCDM (delmu): z>0.7')
    sel = sim_allfits_Ia_JLA[sim_allfits_Ia_JLA.zHD > 0.7]
    print('true z',round(sel['delmu'].median(),2), round(sel['delmu'].std(),2))
    sel = sim_allfits_Ia_JLA[sim_allfits_Ia_JLA.zHD_SNphotoz > 0.7]
    print('SNphotoz',round(sel['delmu_SNphotoz'].median(),2), round(sel['delmu_SNphotoz'].std(),2))
    print('Delta LCDM (delmu): 0.7<z<1')
    sel = sim_allfits_Ia_JLA[(sim_allfits_Ia_JLA.zHD > 0.7) & (sim_allfits_Ia_JLA.zHD < 1)]
    print('true z',round(sel['delmu'].median(),2), round(sel['delmu'].std(),2))
    sel = sim_allfits_Ia_JLA[(sim_allfits_Ia_JLA.zHD_SNphotoz > 0.7) & (sim_allfits_Ia_JLA.zHD_SNphotoz <1)]
    print('SNphotoz',round(sel['delmu_SNphotoz'].median(),2), round(sel['delmu_SNphotoz'].std(),2))

    # with classification
    tmp = pd.merge(sim_allfits_Ia_JLA, sim_preds["cosmo_quantile"])
    tmp2 = pd.merge(sim_allfits_Ia_JLA_SNphotoz, sim_preds["cosmo_quantile"])
    sel = tmp[tmp[f"average_probability_set_0"] > 0.5]
    sel2 = tmp2[tmp2[f"average_probability_set_0"] > 0.5]
    pu.plot_HD_residuals(sel.sample(n=5000, random_state=1), sel2.sample(n=5000, random_state=1), f"{path_plots}/HR_wSNNgt05.png")

    logger.info("")
    logger.info("SIMULATIONS: CONTAMINATION")
    # contamination and efficiency vs. true z and SNphoto z
    list_tuple = [
        (sim_preds, sim_fits),
        (sim_preds, sim_saltz),
        (sim_preds, sim_saltz_JLA),
    ]
    list_labels = [
        "true z",
        "SNphoto z",
        "SNphoto z + HQ",
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
        list_labels=list_labels,
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
    print()
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
        suffix="simSNphotoz_simphotoIa_noJLA",
    )
    df, minv, maxv = du.data_sim_ratio(
        su.apply_JLA_cut(sim_saltz[sim_saltz.SNID.isin(sim_photoIa.SNID.values)]),
        sim_saltz_Ia_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simSNphotoz_simphotoIa_JLA",
    )
