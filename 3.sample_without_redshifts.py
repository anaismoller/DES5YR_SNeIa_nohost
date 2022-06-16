import logging
import argparse
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt

from utils import cuts as cuts
from utils import plot_utils as pu
from utils import data_utils as du
from utils import conf_utils as cu
from utils import metric_utils as mu
from utils import logging_utils as lu
from utils import science_utils as su
from utils import utils_emcee_poisson as mc

plt.switch_backend("agg")

import matplotlib as mpl

mpl.rcParams["font.size"] = 16
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["figure.titlesize"] = "large"
mpl.rcParams["lines.linewidth"] = 3


colors = ["grey"] + pu.ALL_COLORS


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
        xlabel = "actual redshift" if var == "z" else f"{var} with actual redshift"
        ylabel = "fitted redshift" if var == "z" else f"{var} with fitted redshift"
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_xlim(lims[0], lims[1])
        axs[i].set_ylim(lims[0], lims[1])

    # axs[i].legend(loc="best", prop={"size": 10})
    plt.savefig(path_out)


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
        "--path_data",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits",
        help="Path to data",
    )
    parser.add_argument(
        "--path_data_class",
        type=str,
        default=f"{DES}/DES5YR/data_preds/snndump_26XBOOSTEDDES/models/",
        help="Path to data predictions",
    )
    parser.add_argument(
        "--path_data_fits_redshiftsaltfitted",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_FITZ_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz",
        help="Path to data SALT2 fits",
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
    parser.add_argument(
        "--nofit", action="store_true", help="if no fit to selection function",
    )

    # Init
    args = parser.parse_args()
    path_dump = args.path_dump
    path_sim_headers = args.path_sim_headers
    path_data_class = args.path_data_class

    path_plots = f"{path_dump}/plots_sample/"
    os.makedirs(path_plots, exist_ok=True)

    # logger
    logger = setup_logging()

    # Load sample with redshifts (for comparisson only)
    path_samples = "./samples"
    path_sample = f"{path_samples}/BaselineDESsample_looseselcuts.csv"
    photoIa_wz = pd.read_csv(path_sample)
    print(
        f"Photo Ia sample with redshifts (computed previously with ensemble method) {len(photoIa_wz)}"
    )
    # w JLA
    path_sample = f"{path_samples}/BaselineDESsample_JLAlikecuts.csv"
    photoIa_wz_JLA = pd.read_csv(path_sample)
    print(
        f"Photo Ia sample with redshifts w JLA cuts (computed previously with ensemble method) {len(photoIa_wz_JLA)}"
    )

    logger.info("_______________")
    logger.info("PHOTOMETRIC SNE IA")

    logger.info("")
    logger.info("BASIC SELECTION CUTS")
    df_metadata = du.load_headers(args.path_data)

    # OOD cuts
    # detection cuts
    df_metadata_w_dets = cuts.detections("DETECTIONS", df_metadata, logger)
    # transient stats
    df_metadata_w_multiseason = cuts.transient_status(
        "MULTI-SEASON", df_metadata_w_dets, logger
    )
    df_metadata_w_multiseason.to_csv(f"{path_samples}/presel_cuts.csv")

    logger.info("")
    logger.info("SAMPLING SELECTION CUTS")
    # sampling cut
    # need to load photometry + PEAKMJD estimate
    peak_estimate = du.load_salt_fits(
        f"{args.path_data}/DESALL_forcePhoto_real_snana_fits.SNANA.TEXT"
    )
    df_photometry = du.load_photometry(args.path_data)
    # restrict to those that pass detection+multiseason cuts
    tmp = df_photometry[df_photometry.SNID.isin(df_metadata_w_multiseason.SNID.values)]
    print(
        f"DET+MULTISEASON reduced measurements to {len(tmp)} from {len(df_photometry)}"
    )
    df_photometry = tmp

    # eliminate bad photometric points
    # only valid for powers of two combinations
    def powers_of_two(x):
        powers = []
        i = 1
        while i <= x:
            if i & x:
                powers.append(i)
            i <<= 1
        return powers

    tmp = len(df_photometry.SNID.unique())
    tmp2 = len(df_photometry)
    df_photometry["phot_reject"] = df_photometry["PHOTFLAG"].apply(
        lambda x: False
        if len(set([8, 16, 32, 64, 128, 256, 512]).intersection(set(powers_of_two(x))))
        > 0
        else True
    )
    tmp = df_photometry[df_photometry["phot_reject"] == True]
    print(f"PHOTFLAG reduced measurements to {len(tmp)} from {len(df_photometry)}")
    df_photometry = tmp

    # Trim light-curves to -30 < max< 100
    df_pkpho = pd.merge(
        df_photometry[
            ["SNID", "MJD", "FLT", "PHOTFLAG", "FLUXCAL", "FLUXCALERR", "SKY_SIG"]
        ],
        peak_estimate[["SNID", "PKMJDINI"]],
        on="SNID",
        how="left",
    )
    df_pkpho["window_time_cut"] = True
    mask = df_pkpho["MJD"] != -777.00
    df_pkpho["window_delta_time"] = df_pkpho["MJD"] - df_pkpho["PKMJDINI"]
    # apply window of "SN-like event" -30 < PKMJD < 100
    df_pkpho.loc[mask, "window_time_cut"] = df_pkpho["window_delta_time"].apply(
        lambda x: True
        if (x > 0 and x < 100)
        else (True if (x <= 0 and x > -30) else False)
    )
    tmp = df_pkpho[df_pkpho["window_time_cut"] == True]
    print(f"PHOTWINDOW reduced measurements to {len(tmp)} from {len(df_pkpho)}")
    df_pkpho = tmp

    # apply 1 measurement < 1 day before max
    SNID_measurement_before_max = df_pkpho[df_pkpho["window_delta_time"] <= -1][
        "SNID"
    ].unique()
    # apply 1 measurement > 10 day before max
    SNID_measurement_after_maxplus10 = df_pkpho[df_pkpho["window_delta_time"] > 10][
        "SNID"
    ].unique()
    SNID_sampling_measurements = np.intersect1d(
        SNID_measurement_before_max,
        SNID_measurement_after_maxplus10,
        assume_unique=True,
    )
    print(
        f"SAMPLING MEASUREMENTS reduced SNIDs to {len(SNID_sampling_measurements)} from {len(df_pkpho.SNID.unique())}"
    )
    df_pkpho = df_pkpho[df_pkpho.SNID.isin(SNID_sampling_measurements)]

    df_pkpho["SNR"] = df_pkpho["FLUXCAL"] / df_pkpho["FLUXCALERR"]
    # keep only measurements SNR>5 for this exercise
    tmp = df_pkpho[df_pkpho.SNR > 5]
    df_grouped_SNID_flt_occurences = tmp.groupby("SNID")["FLT"].apply(
        lambda x: len(list(np.unique(x)))
    )
    SNID_w_2flt_SNR5 = df_grouped_SNID_flt_occurences[
        df_grouped_SNID_flt_occurences >= 2
    ].index
    df_metadata_w_sampling = df_metadata_w_multiseason[
        df_metadata_w_multiseason.SNID.isin(SNID_w_2flt_SNR5)
    ]

    logger.info("")
    logger.info("PHOTOMETRIC CLASSIFICATION")
    lu.print_blue("Loading predictions without redshift")
    data_preds = du.load_merge_all_preds(
        path_class=path_data_class,
        model_name="vanilla_S_*_none*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean",
        norm="cosmo_quantile",
    )["cosmo_quantile"]

    df_metadata_preds = pd.merge(
        df_metadata_w_sampling, data_preds, on="SNID", how="left"
    )

    # SELECTING PHOTO IA USING ENSEMBLE METHOD
    photoIa_noz = df_metadata_preds[
        df_metadata_preds["average_probability_set_0"] > 0.5
    ]
    lu.print_green(f"photoIa_noz with selcuts set 0: {len(photoIa_noz)}")
    cuts.spec_subsamples(photoIa_noz, logger)

    # some stats on this sample
    print(
        f"Overlap no z set 0 with photoIa_wz {len(photoIa_noz[photoIa_noz.SNID.isin(photoIa_wz.SNID.values)])} ~ {round(100*len(photoIa_noz[photoIa_noz.SNID.isin(photoIa_wz.SNID.values)])/len(photoIa_wz),2)}"
    )
    print(
        f"Overlap no z set 0 with photoIa_wz_JLA {len(photoIa_noz[photoIa_noz.SNID.isin(photoIa_wz_JLA.SNID.values)])} ~ {round(100*len(photoIa_noz[photoIa_noz.SNID.isin(photoIa_wz_JLA.SNID.values)])/len(photoIa_wz_JLA),2)}"
    )

    # Possible contamination
    # dic_photoIa_sel = {"average_probability_set_0": photoIa_noz}
    # dic_tag_SNIDs = cuts.stats_possible_contaminants(
    #     dic_photoIa_sel, method_list=["average_probability_set"]
    # )

    # who are the missing events?
    lost_photoIa_wz_JLA = photoIa_wz_JLA[
        ~photoIa_wz_JLA.SNID.isin(photoIa_noz.SNID.values)
    ]
    print(
        "Missing photoIa_wz_JLA set 0", len(lost_photoIa_wz_JLA),
    )
    list_df = [photoIa_wz_JLA, lost_photoIa_wz_JLA]
    list_labels = ["photoIa_wz_JLA", "photoIa_wz_JLA not in photoIa_noz"]
    pu.plot_mosaic_histograms_listdf(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix="photoIa_wz_JLA",
    )

    # without redshift in this sample
    logger.info("")
    perc = 100 * len(photoIa_noz[photoIa_noz["REDSHIFT_FINAL"] < 0]) / len(photoIa_noz)
    logger.info(f"Without redshift {round(perc,2)} \\%")

    # check if there is a correlation with host_mag_r for missing z
    fig = plt.figure(constrained_layout=True,)
    photoIa_noz_whostmag = photoIa_noz[photoIa_noz["HOSTGAL_MAG_r"] < 40]
    print(
        f"without host {len(photoIa_noz[photoIa_noz['HOSTGAL_MAG_r'] > 40])} equivalent to {round(100*len(photoIa_noz[photoIa_noz['HOSTGAL_MAG_r'] > 40])/len(photoIa_noz),2)} \\%"
    )
    print(
        f'noz host r mag {round(photoIa_noz_whostmag[photoIa_noz_whostmag["REDSHIFT_FINAL"] < 0]["HOSTGAL_MAG_r"].median(),2)}+-{round(photoIa_noz_whostmag[photoIa_noz_whostmag["REDSHIFT_FINAL"] < 0]["HOSTGAL_MAG_r"].std(),2)}, max = {round(photoIa_noz_whostmag[photoIa_noz_whostmag["REDSHIFT_FINAL"] < 0]["HOSTGAL_MAG_r"].max(),2)}'
    )
    # hosts that could be followed-up with AAT (mag<24)
    to_fup_24 = photoIa_noz_whostmag[photoIa_noz_whostmag["HOSTGAL_MAG_r"] < 24]
    print(
        "# hosts r<24 that could be followed-up", len(to_fup_24),
    )
    to_fup_24_nozspe = photoIa_noz_whostmag[
        (photoIa_noz_whostmag["HOSTGAL_MAG_r"] < 24)
        & (photoIa_noz_whostmag["REDSHIFT_FINAL"] < 0)
    ]
    print(
        "  and without spec redshift in database", len(to_fup_24_nozspe),
    )

    # hist hostgalmag
    # DES 5-year (w cuts) + photo ia no z
    fig = plt.figure(figsize=(12, 8))
    # DES_whostmag = df_metadata_preds[df_metadata_preds["HOSTGAL_MAG_r"] < 40]
    # n, bins, tmp = plt.hist(
    #     DES_whostmag["HOSTGAL_MAG_r"],
    #     histtype="step",
    #     label="DES 5-year with selection cuts",
    #     bins=50,
    #     lw=2,
    # )
    n, bins, tmp = plt.hist(
        photoIa_noz_whostmag["HOSTGAL_MAG_r"],
        histtype="step",
        label="photo Ia",
        bins=50,
        lw=2,
    )
    plt.hist(
        photoIa_noz_whostmag[photoIa_noz_whostmag["REDSHIFT_FINAL"] < 0][
            "HOSTGAL_MAG_r"
        ],
        histtype="step",
        label="photo Ia without DES redshift",
        bins=bins,
        lw=2,
    )
    plt.plot(
        [24, 24],
        [0, max(n)],
        linestyle="--",
        color="grey",
        label="follow-up limit approximation",
    )
    plt.ylabel("# events", fontsize=20)
    plt.xlabel("host r magnitude", fontsize=20)
    # plt.yscale("log")
    plt.legend(loc=2)
    plt.savefig(f"{path_plots}/hist_HOSTGAL_MAG_r_vs_REDSHIFT.png")

    plt.clf()
    fig = plt.figure()
    # photo Ia no z + photo Ia wz
    n, bins, tmp = plt.hist(
        photoIa_noz_whostmag["HOSTGAL_MAG_r"],
        histtype="step",
        label="photo Ia (SNN w z)",
        bins=100,
        lw=2,
    )
    plt.hist(
        photoIa_noz_whostmag["HOSTGAL_MAG_r"],
        histtype="step",
        label="photo Ia (SNN no z, this work)",
        bins=bins,
        lw=2,
    )
    plt.hist(
        photoIa_noz_whostmag[photoIa_noz_whostmag["REDSHIFT_FINAL"] < 0][
            "HOSTGAL_MAG_r"
        ],
        histtype="step",
        label="photo Ia (SNN no z, this work) without redshifts",
        bins=bins,
        lw=2,
    )
    plt.plot(
        [24, 24],
        [0, 170],
        linestyle="--",
        color="grey",
        label="follow-up limit approximation",
    )
    plt.ylabel("# events")
    plt.xlabel("host r magnitude")
    plt.legend(loc=2)
    plt.savefig(f"{path_plots}/hist_HOSTGAL_MAG_r_vs_REDSHIFT_wznoz.png")

    # PSNID
    logger.info("")
    logger.info("PSNID as real-time")
    lu.print_green("Loading PSNID fits")
    psnid = du.read_header_fits(f"{DES}/data/PSNID/des_snfit_zprior0.fits")
    photoIa_noz_psnid = pd.merge(photoIa_noz, psnid, on="SNID")
    photoIa_noz_psnid_realtime_cut = photoIa_noz_psnid[
        photoIa_noz_psnid["PBAYES_IA"] > 1e-12
    ]
    print(f"photoIa_noz that pass PSNID cut {len(photoIa_noz_psnid_realtime_cut)} ")
    to_fup_24_psnid = photoIa_noz_psnid_realtime_cut[
        photoIa_noz_psnid_realtime_cut["HOSTGAL_MAG_r"] < 24
    ]
    print(
        "# hosts r<24 that could be followed-up", len(to_fup_24_psnid),
    )
    to_fup_24_nozspe_psnid = photoIa_noz_psnid_realtime_cut[
        (photoIa_noz_psnid_realtime_cut["HOSTGAL_MAG_r"] < 24)
        & (photoIa_noz_psnid_realtime_cut["REDSHIFT_FINAL"] < 0)
    ]
    print(
        "  and without spec redshift in database", len(to_fup_24_nozspe_psnid),
    )
    for var in ["PBAYES_IA", "FITPROB_IA"]:
        fig = plt.figure()
        plt.hist(photoIa_noz_psnid_realtime_cut[var])
        plt.hist(to_fup_24_psnid[var], label="Hostmag<24")
        plt.xlabel(var)
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"{path_plots}/hist_{var}.png")

    logger.info("")
    logger.info("ESTIMATING z,x1,c,t0 WITH SALT2")
    # Load retro fitted redshifts + salt parameters
    # BEWARE: less precise, using host galaxy photo priors and fixed cosmology!
    lu.print_blue("Loading SALT fitted redshifts + SALT params")
    lu.print_yellow(
        "WARNING: these simultaneous redshifts+x1+c+t0 are biased for cosmology"
    )
    salt_fits_noz = du.load_salt_fits(args.path_data_fits_redshiftsaltfitted)

    # merge preds + salt
    photoIa_noz_saltz = pd.merge(photoIa_noz, salt_fits_noz, on=["SNID", "SNTYPE"])
    print(
        f"# of events with simultaneous SALT fit {len(salt_fits_noz)} from which {len(photoIa_noz_saltz)} are photoIa"
    )
    to_fup_24 = pd.merge(photoIa_noz_saltz, to_fup_24_nozspe_psnid, on="SNID")
    print(
        "fup: noz + hostmag", len(to_fup_24),
    )

    pu.plot_mosaic_histograms_listdf(
        [photoIa_noz_saltz],
        list_labels=["photo Ia (fitted z)"],
        path_plots=path_plots,
        suffix="_withoutcuts",
        list_vars_to_plot=["zHD", "c", "x1"],
        data_color_override=True,
    )

    logger.info("")
    logger.info("SALT FUP BASIC CUTS")
    # color error + t0
    photoIa_noz_saltfup_cut = photoIa_noz_saltz[
        (photoIa_noz_saltz.x1ERR < 1)
        & (photoIa_noz_saltz.PKMJDERR < 2)
        & (photoIa_noz_saltz.FITPROB > 0.001)
    ]
    photoIa_noz_saltfup_cut_nozspe = photoIa_noz_saltfup_cut[
        photoIa_noz_saltfup_cut.REDSHIFT_FINAL < 0
    ]
    print(
        f"# of photoIa_noz with SALT fup cuts (x1ERR,t0,FITPROB, no JLA) {len(photoIa_noz_saltfup_cut)} without zspe {len(photoIa_noz_saltfup_cut_nozspe)}",
    )
    to_fup_24_nozspe_psnid_saltfup = pd.merge(
        photoIa_noz_saltfup_cut, to_fup_24_nozspe_psnid, on="SNID"
    )
    print(
        " + psnid, hostmag", len(to_fup_24_nozspe_psnid_saltfup),
    )

    logger.info("")
    logger.info("JLA CUTS")
    # photoIa_noz_saltz_zrange = photoIa_noz_saltz[
    #     (photoIa_noz_saltz.zHD > 0.2) & (photoIa_noz_saltz.zHD < 1.2)
    # ]
    photoIa_noz_saltz_zrange = photoIa_noz_saltz
    photoIa_noz_saltz_JLA = su.apply_JLA_cut(photoIa_noz_saltz_zrange)

    # for all preds that pass sel cuts
    df_metadata_preds_saltz = pd.merge(
        df_metadata_preds, salt_fits_noz, on=["SNID", "SNTYPE"]
    )
    # z
    # df_metadata_preds_saltz_zrange = df_metadata_preds_saltz[
    #     (df_metadata_preds_saltz.zHD > 0.2) & (df_metadata_preds_saltz.zHD < 1.2)
    # ]
    df_metadata_preds_saltz_zrange = df_metadata_preds_saltz
    # JLA
    df_metadata_preds_JLA = su.apply_JLA_cut(df_metadata_preds_saltz_zrange)

    lu.print_green(
        f"photoIa_noz set 0 with JLA cuts (fitted z): {len(photoIa_noz_saltz_JLA)}"
    )
    cuts.spec_subsamples(photoIa_noz_saltz_JLA, logger)

    # distributions of these new events
    list_df = [
        photoIa_noz_saltz,
        photoIa_noz_saltz[~photoIa_noz_saltz.SNID.isin(photoIa_wz_JLA.SNID.values)],
        photoIa_wz_JLA,
    ]
    list_labels = [
        "photo Ia (fitted z)",
        "photo Ia (fitted z) not in photo Ia z",
        "photo Ia z",
    ]
    pu.plot_mosaic_histograms_listdf(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix="_newevents",
        list_vars_to_plot=["zHD", "c", "x1"],
        data_color_override=True,
    )

    logger.info("")
    logger.info("INSPECT MISSING WZ NOT IN NOZ")
    # stats
    overlap = photoIa_noz_saltz_JLA[
        photoIa_noz_saltz_JLA.SNID.isin(photoIa_wz_JLA.SNID.values)
    ]
    print(
        f"Overlap photoIa no z and with z {len(overlap)} ~{round(100*len(overlap)/len(photoIa_wz_JLA),2)}%"
    )
    lost_photoIa_wz_JLA = photoIa_wz_JLA[
        ~photoIa_wz_JLA.SNID.isin(photoIa_noz_saltz_JLA.SNID.values)
    ]
    print("Lost photoIa with z", len(lost_photoIa_wz_JLA))
    sel = salt_fits_noz[salt_fits_noz.SNID.isin(lost_photoIa_wz_JLA.SNID.values)]
    print("   have simultaneous fit", len(sel))
    # sel = sel[(sel.zHD > 0.2) & (sel.zHD < 1.2)]
    # print("   pass zrange", len(sel))
    sel = su.apply_JLA_cut(sel)
    print("   pass JLA cuts", len(sel))

    # extra check how far the redshift was chosen
    sel = sel.add_suffix("_retro")
    sel = sel.rename(columns={"SNID_retro": "SNID"})
    merged = pd.merge(sel, lost_photoIa_wz_JLA, on="SNID")
    merged["delta_zHD"] = abs(merged["zHD"] - merged["zHD_retro"])
    print(
        "   Good z: (fitted - host-galaxy<0.1)", len(merged[merged["delta_zHD"] < 0.1]),
    )
    merged["delta_c"] = abs(merged["c"] - merged["c_retro"])
    print(
        "   Good c: (fitted - host-galaxy<0.1)", len(merged[merged["delta_c"] < 0.1]),
    )
    merged["delta_x1"] = abs(merged["x1"] - merged["x1_retro"])
    print(
        "   Good x1: (fitted - host-galaxy<0.1)", len(merged[merged["delta_x1"] < 0.1]),
    )

    # histo
    list_df = [
        lost_photoIa_wz_JLA,
        salt_fits_noz[salt_fits_noz.SNID.isin(lost_photoIa_wz_JLA.SNID.values)],
    ]
    list_labels = [
        "lost_photoIa_wz_JLA",
        "lost_photoIa_wz_JLA z fitted",
    ]
    pu.plot_mosaic_histograms_listdf(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix="lostphotoIa_wz_JLA_in_noz_differentsalt",
    )
    # lost simultaneous salt fit effect
    tmp_retro = salt_fits_noz.add_suffix("_retro")
    tmp_retro = tmp_retro.rename(columns={"SNID_retro": "SNID"})
    df_tmp = pd.merge(lost_photoIa_wz_JLA, tmp_retro, on="SNID")
    plot_scatter_mosaic_retro(
        [df_tmp],
        ["lost_photoIa_wz_JLA"],
        path_out=f"{path_plots}/scatter_lost_photoIawz_retro_vs_ori.png",
    )
    # overlap simultaneous salt fit effect
    df_tmp = pd.merge(overlap, tmp_retro, on="SNID")
    plot_scatter_mosaic_retro(
        [df_tmp],
        ["lost_photoIa_wz_JLA"],
        path_out=f"{path_plots}/scatter_overlap_photoIawz_retro_vs_ori.png",
    )

    logger.info("")
    logger.info("SAMPLE CONTAMINATION")
    # hack to use the same contaminant inspection
    dic_photoIa_sel = {"average_probability_set_0": photoIa_noz_saltz_JLA}
    dic_tag_SNIDs = cuts.stats_possible_contaminants(
        dic_photoIa_sel, method_list=["average_probability_set"]
    )
    # eliminate the AGNs
    SNIDs_to_eliminate = [
        k
        for k in dic_tag_SNIDs[0]["OzDES_AGN"]
        if k not in [1286337, 1246527, 1370320, 1643063, 1252970]  # >1'' from AGN
    ]
    lu.print_yellow(
        "Eliminating tagged close-by AGNs (rest of results)", SNIDs_to_eliminate
    )
    photoIa_noz_saltz_JLA = photoIa_noz_saltz_JLA[
        ~photoIa_noz_saltz_JLA.SNID.isin(SNIDs_to_eliminate)
    ]

    logger.info("")
    logger.info("SAMPLE STATS")
    lu.print_green(
        f"PhotoIa no z set 0 with all cuts (-2 AGNs): {len(photoIa_noz_saltz_JLA)}"
    )
    # eliminating for all samples
    df_metadata_preds = df_metadata_preds[
        ~df_metadata_preds.SNID.isin(SNIDs_to_eliminate)
    ]
    df_metadata_preds_JLA = df_metadata_preds_JLA[
        ~df_metadata_preds_JLA.SNID.isin(SNIDs_to_eliminate)
    ]

    # table as in redshift photoIa
    df_photoIa_stats = du.get_sample_stats(df_metadata_preds)
    df_photoIa_stats_JLA = du.get_sample_stats(df_metadata_preds_JLA, suffix="_JLA")
    df_photoIa_stats = pd.merge(
        df_photoIa_stats, df_photoIa_stats_JLA, on=["norm", "method"], how="left"
    )

    # just set 0
    list_suffix = ["", "_JLA"]
    df_photoIa_stats_tmp = {}
    for n, df in enumerate([df_metadata_preds, df_metadata_preds_JLA]):
        df_photoIa_stats_tmp[n] = pd.DataFrame()
        photoIa, specIa, _, _ = cuts.photo_sel_prob(
            df, prob_key="average_probability_set_0"
        )
        photoIa_tofup, _, _, _ = cuts.photo_sel_prob(
            df[df.SNID.isin(to_fup_24_nozspe.SNID.values)],
            prob_key="average_probability_set_0",
        )
        suffix = list_suffix[n]
        dic_tmp = {
            "method": "ensemble set 0",
            f"photoIa{suffix}": photoIa,
            f"specIa{suffix}": specIa,
            f"fup{suffix}": photoIa_tofup,
        }
        df_photoIa_stats_tmp[n] = df_photoIa_stats_tmp[n].append(
            dic_tmp, ignore_index=True
        )

    df_photoIa_stats_set0 = pd.merge(
        df_photoIa_stats_tmp[0], df_photoIa_stats_tmp[1], on="method"
    )
    cols_to_change = [k for k in df_photoIa_stats_set0.keys() if k != "method"]
    df_photoIa_stats_set0[cols_to_change] = df_photoIa_stats_set0[
        cols_to_change
    ].astype(int)

    lu.print_blue("cosmo_quantile")
    latex_table = df_photoIa_stats.to_latex(
        index=False,
        columns=["method", "photoIa", "specIa", "photoIa_JLA", "specIa_JLA"],
    )
    logger.info(latex_table)
    latex_table_set0 = df_photoIa_stats_set0.to_latex(
        index=False,
        columns=[
            "method",
            "photoIa",
            "specIa",
            "photoIa_JLA",
            "specIa_JLA",
            "fup",
            "fup_JLA",
        ],
    )
    logger.info(latex_table_set0)

    # Overlap between these samples
    dic_venn = {
        "photoIa w z": set(photoIa_wz_JLA.SNID.values),
        "photoIa no z": set(photoIa_noz.SNID.values),
        "photoIa no z + JLA": set(photoIa_noz_saltz_JLA.SNID.values),
    }
    pu.plot_venn(dic_venn, path_plots=path_plots, suffix="noz_cuts")

    dic_venn = {
        "photoIa w z (JLA)": set(photoIa_wz_JLA.SNID.values),
        "photoIa no z (JLA)": set(photoIa_noz_saltz_JLA.SNID.values),
        "photoIa no z, without z": set(
            photoIa_noz[photoIa_noz["REDSHIFT_FINAL"] < 0].SNID.values
        ),
    }
    pu.plot_venn(dic_venn, path_plots=path_plots, suffix="noz_wz")

    dic_venn = {
        "photoIa w z (JLA)": set(photoIa_wz_JLA.SNID.values),
        "photoIa no z (JLA)": set(photoIa_noz_saltz_JLA.SNID.values),
    }
    pu.plot_venn2(dic_venn, path_plots=path_plots, suffix="wz_noz_JLA")

    # save sample
    photoIa_noz_saltz_JLA.to_csv(
        f"{path_samples}/photoIa_noz_cosmo_quantile_average_probability_set_0.csv"
    )
    print(
        f"Saved {len(photoIa_noz_saltz_JLA)} average_probability_set_0>0.5 after ALL quality cuts"
    )

    # load predictions of snn
    sim_preds = du.load_merge_all_preds(
        path_class=args.path_sim_class,
        model_name="vanilla_S_*_none*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean",
        norm="cosmo_quantile",
    )
    logger.info("")
    logger.info("SIMULATIONS: EFFICIENCY & CONTAMINATION (REALISTIC = NOT BALANCED)")
    df_txt_stats_noz = pd.DataFrame(
        columns=[
            "norm",
            "dataset",
            "method",
            "notbalanced_accuracy",
            "efficiency",
            "purity",
        ]
    )
    for method, desc in cu.dic_sel_methods.items():
        list_seeds_sets = (
            cu.list_seeds_set[0] if method == "predicted_target_S_" else cu.list_sets
        )
        df_txt_stats_noz = mu.get_multiseed_performance_metrics(
            sim_preds["cosmo_quantile"],
            key_pred_targ_prefix=method,
            list_seeds=list_seeds_sets,
            df_txt=df_txt_stats_noz,
            dic_prefilled_keywords={
                "norm": "cosmo_quantile",
                "dataset": "not balanced",
                "method": desc,
            },
        )
    print(
        df_txt_stats_noz[
            ["method", "notbalanced_accuracy", "efficiency", "purity"]
        ].to_latex(index=False)
    )

    logger.info("")
    logger.info("SIMULATIONS: SIMULTANEOUS SALT2 FIT")
    # Load usual SALT2 fit (using available redshift)
    sim_fits = du.load_salt_fits(
        f"{args.path_sim_fits}/JLA_1XDES/output/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/FITOPT000.FITRES.gz"
    )

    sim_fits_JLA = su.apply_JLA_cut(sim_fits)
    sim_Ia_fits = sim_fits[sim_fits.SNTYPE.isin(cu.spec_tags["Ia"])]
    sim_Ia_fits_JLA = su.apply_JLA_cut(sim_Ia_fits)

    # Load z,x1,c SALT2 fit
    sim_saltz = du.load_salt_fits(
        f"{args.path_sim_fits}/D_FITZ_1XDES/output/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/FITOPT000.FITRES.gz"
    )
    sim_saltz_JLA = su.apply_JLA_cut(sim_saltz)
    sim_saltz_Ia = sim_saltz[sim_saltz.SNTYPE.isin(cu.spec_tags["Ia"])]
    sim_saltz_Ia_JLA = su.apply_JLA_cut(sim_saltz_Ia)

    tmp_sim_saltz = sim_saltz.add_suffix("_retro")
    tmp_sim_saltz = tmp_sim_saltz.rename(columns={"SNID_retro": "SNID"})
    sim_all_fits = pd.merge(sim_fits, tmp_sim_saltz)

    # define simulatred photo sample
    sim_cut_photoIa = sim_preds["cosmo_quantile"][f"average_probability_set_0"] > 0.5
    sim_photoIa = sim_preds["cosmo_quantile"][sim_cut_photoIa]

    logger.info("")
    logger.info("SIMULATIONS: CLASSIFICATION EFFICIENCY")
    # the simultaneous saltz fit may introduce some large biases

    # 1. comparing simulations: photoIa vs true Ia
    # define simulated sample
    sim_preds_photoIa_noz = sim_preds["cosmo_quantile"][
        sim_preds["cosmo_quantile"]["average_probability_set_0"] > 0.5
    ]
    # predicted photo Ia
    sim_allfits_photoIa_noz = sim_all_fits[
        sim_all_fits.SNID.isin(sim_preds_photoIa_noz.SNID.values)
    ]
    sim_allfits_Ia = sim_all_fits[sim_all_fits.SNID.isin(sim_saltz_Ia.SNID.values)]

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

    # 2. compraing data with sims
    variable = "m0obs_i"
    quant = 0.01
    min_var = photoIa_noz_saltz_JLA[variable].quantile(quant)
    lu.print_yellow(
        f"Not using photo Ias with {variable}<{min_var} (equivalent to quantile {quant}, possible if low-z)"
    )
    # fixed z for sim
    df, minv, maxv = du.data_sim_ratio(
        photoIa_noz_saltz_JLA,
        sim_Ia_fits_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simfixedz",
    )
    if args.nofit:
        lu.print_red("Not doing sel function emcee_fitting!!!!")
    else:
        # Emcee fit of dat/sim
        theta_mcmc, min_theta_mcmc, max_theta_mcmc = mc.emcee_fitting(
            df, path_plots, min_var=min_var
        )
        print("Emcee estimate", theta_mcmc, min_theta_mcmc, max_theta_mcmc)

    # saltz for sim
    df, minv, maxv = du.data_sim_ratio(
        photoIa_noz_saltz_JLA,
        sim_saltz_Ia_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simfittedz",
    )
    if args.nofit:
        lu.print_red("Not doing sel function emcee_fitting!!!!")
    else:
        # Emcee fit of dat/sim
        theta_mcmc, min_theta_mcmc, max_theta_mcmc = mc.emcee_fitting(
            df, path_plots, min_var=min_var
        )
        print("Emcee estimate", theta_mcmc, min_theta_mcmc, max_theta_mcmc)

    # sanity ratio z
    variable = "zHD"
    quant = 0.01
    min_var = photoIa_noz_saltz_JLA[variable].quantile(quant)
    tmo, tmominv, mtmoaxv = du.data_sim_ratio(
        photoIa_noz_saltz_JLA,
        sim_Ia_fits_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simfixedz",
    )
    tmo, tmominv, mtmoaxv = du.data_sim_ratio(
        photoIa_noz_saltz_JLA,
        sim_saltz_Ia_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simfittedz",
    )

    lu.print_blue("Loading metadata sims")
    sim_metadata = du.load_headers(args.path_sim_headers)
    metadata_sim_Ia_fits_JLA = pd.merge(sim_Ia_fits_JLA, sim_metadata, on="SNID")
    # sanity ratio z
    variable = "HOSTGAL_MAG_r"
    quant = 0.01
    min_var = photoIa_noz_saltz_JLA[variable].quantile(quant)
    tmo, tmominv, mtmoaxv = du.data_sim_ratio(
        photoIa_noz_saltz_JLA,
        metadata_sim_Ia_fits_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
    )

    logger.info("")
    logger.info("SAMPLE PROPERTIES")
    pu.overplot_salt_distributions_lists(
        [sim_saltz_Ia_JLA, photoIa_noz_saltz_JLA,],
        path_plots=path_plots,
        list_labels=[
            "sim Ia JLA (z from SALT)",
            "photometric Ia noz JLA (z from SALT)",
        ],
        suffix="noz",
        sim_scale_factor=30,
    )

    pu.overplot_salt_distributions_lists(
        [sim_Ia_fits_JLA, sim_saltz_Ia_JLA, photoIa_noz_saltz_JLA],
        path_plots=path_plots,
        list_labels=[
            "sim Ia JLA (sim z)",
            "sim Ia JLA (z from SALT)",
            "photo Ia noz JLA (z from SALT)",
        ],
        suffix="noz_simz",
        sim_scale_factor=30,
    )

    # add m0obs_i
    pu.plot_mosaic_histograms_listdf(
        [sim_Ia_fits_JLA, sim_saltz_Ia_JLA, photoIa_noz_saltz_JLA,],
        list_labels=[
            "sim Ia JLA (z fixed)",
            "sim Ia JLA (z from SALT)",
            "photo Ia noz JLA (z from SALT)",
        ],
        path_plots=path_plots,
        suffix="noz_wm0obsi",
        list_vars_to_plot=["zHD", "c", "x1", "m0obs_i"],
        norm=1 / 30,  # using small sims
    )

    # DON'T USE THIS FITS FOR HD
    # mB is not correctly determined, remember the cosmology -> mu is fixed!
    # merged_sample_fits_with_preds = pd.merge(salt_fits_noz, photoIa_noz, on="SNID")
    # pu.plot_HD_residuals(
    #     merged_sample_fits_with_preds,
    #     f"{path_plots}/HDres.png",
    #     prob_key="average_probability_set_0",
    # )
