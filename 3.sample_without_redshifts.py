import logging
import argparse
import numpy as np
import pandas as pd
import os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from supervenn import supervenn

from myutils import cuts as cuts
from myutils import plot_utils as pu
from myutils import data_utils as du
from myutils import conf_utils as cu
from myutils import metric_utils as mu
from myutils import logging_utils as lu
from myutils import science_utils as su
from myutils import utils_emcee_poisson as mc

mpl.rcParams["font.size"] = 16
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["figure.titlesize"] = "large"
mpl.rcParams["lines.linewidth"] = 3
plt.switch_backend("agg")

colors = ["grey"] + pu.ALL_COLORS


def overlap_photoIa(df_sel, photoIa_wz, photoIa_wz_JLA, mssg=""):
    # some stats on this sample
    print(
        f"Overlap {mssg} with photoIa_wz {len(df_sel[df_sel.SNID.isin(photoIa_wz.SNID.values)])} ~ {round(100*len(df_sel[df_sel.SNID.isin(photoIa_wz.SNID.values)])/len(photoIa_wz),2)}"
    )
    print(
        f"Overlap {mssg} with photoIa_wz_JLA {len(df_sel[df_sel.SNID.isin(photoIa_wz_JLA.SNID.values)])} ~ {round(100*len(df_sel[df_sel.SNID.isin(photoIa_wz_JLA.SNID.values)])/len(photoIa_wz_JLA),2)}"
    )


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


if __name__ == "__main__":

    DES = os.getenv("DES")

    parser = argparse.ArgumentParser(
        description="Select SNe Ia without redshift information using SNN"
    )

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
        "--path_data_fits_SNphotoz",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_FITZ_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz",
        help="Path to data SALT2 fits",
    )
    parser.add_argument(
        "--path_data_fits",
        type=str,
        default=f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_JLA_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz",
        help="Path to data SALT2 fits (using zspe)",
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
        "--nofit",
        action="store_true",
        help="if no fit to selection function",
    )

    # Init
    args = parser.parse_args()
    path_dump = args.path_dump
    path_sim_headers = args.path_sim_headers
    path_data_class = args.path_data_class

    path_plots = f"{path_dump}/plots_sample/"
    os.makedirs(path_plots, exist_ok=True)
    os.makedirs(f"{path_dump}/plots_fup", exist_ok=True)

    # logger
    logger = setup_logging()

    # Load Moller 2022 sample SNe Ia with host-redshifts High-quality (for comparisson only)
    logger.info("")
    lu.print_green("Loading previous samples")
    path_samples = "./samples"
    path_sample = f"{path_samples}/previous_works/Moller2022_DES5yr_SNeIa_whostz.csv"
    photoIa_wz = pd.read_csv(path_sample, comment="#")
    path_sample = (
        f"{path_samples}/previous_works/Moller2022_DES5yr_SNeIa_whostz_JLA.csv"
    )
    photoIa_wz_JLA = pd.read_csv(path_sample, comment="#")
    print(
        f"Photo Ia sample with host-redshifts w JLA/HQ cuts (Moller 20222 ensemble method) {len(photoIa_wz_JLA)}"
    )

    # Load Redshift catalogue
    # Alternative use GRC for tagging PRIMUS hosts info
    sngals = pd.read_csv(f"{path_samples}/previous_works/sngals.csv", comment="#")

    # load PSNID info
    psnid = pd.read_csv(f"{path_samples}/previous_works/psnid.csv", comment="#")

    logger.info("_______________")
    logger.info("PHOTOMETRIC SNE IA")

    logger.info("")
    logger.info("BASIC SELECTION CUTS")
    df_metadata = du.load_headers(args.path_data)

    df_stats = mu.cuts_deep_shallow(
        df_metadata, photoIa_wz_JLA, cut="DES-SN 5-year candidate sample"
    )
    df_stats_fup = mu.fup_hostgals_stats(
        df_metadata, sngals, photoIa_wz_JLA, sample="DES-SN 5-year candidate sample"
    )

    # OOD cuts
    # detection cuts
    df_metadata_w_dets = cuts.detections("DETECTIONS", df_metadata, logger)
    # transient stats
    df_metadata_w_multiseason = cuts.transient_status(
        "MULTI-SEASON", df_metadata_w_dets, logger
    )
    # df_metadata_w_multiseason.to_csv(f"{path_samples}/presel_cuts.csv")
    df_stats = mu.cuts_deep_shallow(
        df_metadata_w_multiseason,
        photoIa_wz_JLA,
        df_stats=df_stats,
        cut="Filtering multi-season",
    )

    logger.info("")
    logger.info("SAMPLING SELECTION CUTS")
    # sampling cut
    # need to load photometry + PEAKMJD estimate
    peak_estimate = du.load_salt_fits(
        f"{args.path_data}/DESALL_forcePhoto_real_snana_fits.SNANA.TEXT"
    )
    df_photometry = du.load_photometry(args.path_data)
    # restrict to those that pass detection+multiseason cuts
    df_photometry = df_photometry[
        df_photometry.SNID.isin(df_metadata_w_multiseason.SNID.values)
    ]

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
    # print(f">> PHOTFLAG reduced measurements to {len(tmp)} from {len(df_photometry)}")
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
    df_pkpho = df_pkpho[df_pkpho["window_time_cut"] == True]

    def group_photo_criteria(df, n_measurements):
        # unique flt occurences
        tmp = df.groupby("SNID")["FLT"].apply(lambda x: len(list(np.unique(x))))
        id1 = tmp[tmp >= n_measurements].index
        # unique night occurences
        df["MJDint"] = df["MJD"].astype(int)
        tmp2 = df.groupby("SNID")["MJDint"].apply(lambda x: len(list(np.unique(x))))
        id2 = tmp2[tmp2 >= n_measurements].index
        # flt + nights requirement
        inters = list(set(id1) & set(id2))
        # return tmp[tmp >= n_measurements].index
        return inters

    # 1 filter at least <0 day before max
    SNID_measurement_before_max = group_photo_criteria(
        df_pkpho[df_pkpho["window_delta_time"] < 0], 1
    )
    SNID_measurement_after_maxplus10 = group_photo_criteria(
        df_pkpho[df_pkpho["window_delta_time"] > 10], 1
    )
    SNID_sampling_measurements_std = np.intersect1d(
        SNID_measurement_before_max,
        SNID_measurement_after_maxplus10,
        assume_unique=True,
    )
    print(f">> + 1 point >max+10")

    SNID_sampling_measurements = SNID_sampling_measurements_std
    # reselect photometry for SNIDs only
    df_pkpho = df_pkpho[df_pkpho.SNID.isin(SNID_sampling_measurements)]

    # SNR>5
    df_pkpho["SNR"] = df_pkpho["FLUXCAL"] / df_pkpho["FLUXCALERR"]
    SNID_w_2flt_SNR5 = group_photo_criteria(df_pkpho[abs(df_pkpho.SNR) > 5], 1)
    print(f">> + 1 points SNR>5 to {len(SNID_w_2flt_SNR5)}")

    df_metadata_w_sampling = df_metadata_w_multiseason[
        df_metadata_w_multiseason.SNID.isin(SNID_w_2flt_SNR5)
    ]
    cuts.spec_subsamples(df_metadata_w_sampling, logger)
    df_metadata_w_sampling.to_csv((f"{path_samples}/df_metadata_w_sampling.csv"))

    df_stats = mu.cuts_deep_shallow(
        df_metadata_w_sampling,
        photoIa_wz_JLA,
        df_stats=df_stats,
        cut="Photometric sampling and SNR",
    )
    df_stats_fup = mu.fup_hostgals_stats(
        df_metadata_w_sampling,
        sngals,
        photoIa_wz_JLA,
        sample="Photometric sampling and SNR",
        df_stats=df_stats_fup,
    )

    # psnid only
    tmp = pd.merge(df_metadata_w_sampling, psnid, on="SNID")
    df_stats_fup = mu.fup_hostgals_stats(
        tmp[tmp.PBAYES_IA > 1e-12],
        sngals,
        photoIa_wz_JLA,
        df_stats=df_stats_fup,
        sample="PSNID on Photometric sampling and SNR",
    )

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

    # stats for different RNN prob cut
    for rnn_score in [0.001, 0.5]:
        tmp = df_metadata_preds[
            df_metadata_preds["average_probability_set_0"] > rnn_score
        ]
        df_stats_fup = mu.fup_hostgals_stats(
            tmp,
            sngals,
            photoIa_wz_JLA,
            sample=f"RNN > {rnn_score}",
            df_stats=df_stats_fup,
            verbose=True if rnn_score == 0.5 else False,
        )
        lu.print_green(f"photoIa_noz with {rnn_score}: {len(tmp)}")
        cuts.spec_subsamples(tmp, logger)
        df_stats = mu.cuts_deep_shallow(
            tmp, photoIa_wz_JLA, df_stats=df_stats, cut=f"RNN>{rnn_score}"
        )
        if rnn_score == 0.5:
            photoIa_noz = tmp
        if rnn_score == 0.001:
            photoIa_noz_001 = tmp
            cols_to_print = [
                k for k in df_stats_fup.keys() if "OzDES" not in k and "Y" not in k
            ]
            print(
                df_stats_fup[df_stats_fup["sample"] == f"RNN > {rnn_score}"][
                    cols_to_print
                ]
            )
            print("M22 not selected")
            print(
                photoIa_wz_JLA[~photoIa_wz_JLA.SNID.isin(tmp.SNID.values)][
                    ["SNID", "zHD", "c", "x1"]
                ]
            )

    # paper stats
    rnn05 = df_stats_fup[df_stats_fup["sample"] == "RNN > 0.5"]
    print(
        f"% of M22 in this sample {np.round(rnn05['photoIa M22'].values*100/df_stats_fup[df_stats_fup['sample']=='DES-SN 5-year candidate sample']['photoIa M22'].values,2)}%"
    )
    print(
        f"% without host-galaxy {100-np.round(rnn05['with host'].values*100/rnn05['total'].values)}% = {(rnn05['with host'].values)}"
    )
    print(f"no zfinal {rnn05['without redshift'].values}")

    print(f"mag<24 {rnn05['<24 mag'].values}")
    print(f"mag<24 and no zfinal {rnn05['<24 mag and no zspe'].values}")
    print(f"OzDES QOP 2 {rnn05['OzDES QOP 2'].values}")

    # for fup studies where I estimate potential low-z, see
    potential_fup_nohostz = photoIa_noz[
        (photoIa_noz.REDSHIFT_FINAL < 0) & (photoIa_noz.HOSTGAL_MAG_r < 30)
    ]

    # save photoIa that pass SNN>0.5
    photoIa_noz.to_csv(f"{path_samples}/photoIanoz.csv")
    # M24 for DES data release
    out_sample_probs = photoIa_noz[["SNID", "all_class0_S_0", "average_probability_set_0"]]
    out_sample_probs = out_sample_probs.rename(
        columns={
            "SNID": "CID",
            "all_class0_S_0": "PROB_SNN_noz_singlemodel",
            "average_probability_set_0": "PROB_SNN_noz_ensemble",
        }
    )
    out_sample_probs.to_csv(f"{path_samples}/DES_noredshift_classification_Moller2024.csv")
    # load salt fits wzspe
    tmpsalt = du.load_salt_fits(args.path_data_fits)
    tmpsalt_zspe = tmpsalt[
        ["SNID", "zHD", "zHDERR", "c", "x1", "m0obs_i", "SNTYPE"]
    ].add_suffix("_zspe")
    tmpsalt_zspe = tmpsalt_zspe.rename(
        columns={"SNID_zspe": "SNID", "SNTYPE_zspe": "SNTYPE"}
    )
    salt_fits_wz = tmpsalt_zspe[
        ["SNID", "zHD_zspe", "zHDERR_zspe", "c_zspe", "x1_zspe", "m0obs_i_zspe"]
    ]
    photoIa_noz = pd.merge(photoIa_noz, salt_fits_wz, on="SNID", how="left")

    overlap_photoIa(photoIa_noz, photoIa_wz, photoIa_wz_JLA, mssg="photoIa_noz")

    # new
    not_in_photoIa_wz = photoIa_noz[~photoIa_noz.SNID.isin(photoIa_wz.SNID.values)]
    # not_in_photoIa_wz.to_csv(f"{path_samples}/photoIanoz_notin_photoIa_wz.csv")
    pu.plot_mosaic_histograms_listdf(
        [not_in_photoIa_wz],
        list_labels=["new"],
        path_plots=path_plots,
        suffix="new",
        list_vars_to_plot=["REDSHIFT_FINAL", "average_probability_set_0"],
        data_color_override=True,
    )

    # who are the missing events?
    lost_photoIa_wz_JLA = photoIa_wz_JLA[
        ~photoIa_wz_JLA.SNID.isin(photoIa_noz.SNID.values)
    ]
    print(
        "Missing photoIa_wz_JLA set 0",
        len(lost_photoIa_wz_JLA),
    )
    list_df = [photoIa_wz_JLA, lost_photoIa_wz_JLA]
    list_labels = ["photoIa_wz_JLA", "photoIa_wz_JLA not in photoIa_noz"]
    pu.plot_mosaic_histograms_listdf(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix="photoIa_wz_JLA",
        data_color_override=True,
    )

    pu.plot_mosaic_scatter(
        lost_photoIa_wz_JLA,
        path_plots=path_plots,
        suffix="lost_photoIa_wz_JLA",
    )
    # lost M22
    lost_M22 = photoIa_wz_JLA[(~photoIa_wz_JLA.SNID.isin(photoIa_noz.SNID.values))]
    pu.plot_mosaic_histograms_listdf(
        [lost_M22],
        list_labels=["lost M22"],
        path_plots=path_plots,
        suffix="lost_M22",
        data_color_override=True,
    )
    # check if the lost are due to peak
    lost_M22.to_csv(f"{path_samples}/lost_M22.csv")
    # lc plotting in extra_plots.ipynb

    # Who are the ones that got noz selected but were not in wz
    photoIa_noz_wz_notsel_photoIawz = photoIa_noz[
        (photoIa_noz["REDSHIFT_FINAL"] > 0)
        & (~photoIa_noz["SNID"].isin(photoIa_wz.SNID.values))
    ]
    print(
        f"photoIa_noz but NOT photoIa_wz and HAVE a redshift {len(photoIa_noz_wz_notsel_photoIawz)}"
    )
    # photoIa_noz_wz_notsel_photoIawz.to_csv(f"{path_samples}/not_Iawz_buthavez.csv")
    pu.plot_mosaic_histograms_listdf(
        [photoIa_noz_wz_notsel_photoIawz],
        list_labels=[""],
        path_plots=path_plots,
        suffix="photoIa_noz_wz_notsel_photoIawz",
        list_vars_to_plot=["REDSHIFT_FINAL", "average_probability_set_0"],
    )
    photoIa_noz_wz_notsel_photoIawz_saltzspe = photoIa_noz_wz_notsel_photoIawz[
        photoIa_noz_wz_notsel_photoIawz.SNID.isin(salt_fits_wz.SNID.values)
    ]
    print(f"new w salt fit {len(photoIa_noz_wz_notsel_photoIawz_saltzspe)}")
    pu.plot_mosaic_histograms_listdf(
        [photoIa_noz_wz_notsel_photoIawz_saltzspe],
        list_labels=[""],
        path_plots=path_plots,
        suffix="new_wsaltfit",
        list_vars_to_plot=[
            "zHD_zspe",
            "c_zspe",
            "x1_zspe",
            "average_probability_set_0",
        ],
        chi_bins=False,
    )

    logger.info("")
    logger.info("SNPHOTO Z and SALT2 parameters")
    # Load Snphoto z redshifts + salt parameters
    # BEWARE: less precise, using host galaxy photo priors and fixed cosmology!
    lu.print_blue("Loading SALT SNphoto z + SALT params")
    lu.print_yellow("WARNING: SNphoto z are approximations")
    salt_fits_noz = du.load_salt_fits(args.path_data_fits_SNphotoz)
    cols_to_keep = [k for k in salt_fits_noz.keys() if k != "FIELD"]
    # merge preds + salt
    photoIa_noz_saltz = pd.merge(
        photoIa_noz, salt_fits_noz[cols_to_keep], on=["SNID", "SNTYPE"]
    )

    # parenthesis, checking potential host fup redshifts
    potential_fup_nohostz_saltz = pd.merge(
        potential_fup_nohostz, salt_fits_noz, on=["SNID", "SNTYPE"]
    )
    print(f"POTENTIAL FUP HOSTS {len(potential_fup_nohostz_saltz)}")
    plt.clf()
    plt.hist(potential_fup_nohostz_saltz["zHD"])
    plt.xlabel("SNphoto z")
    plt.savefig(f"{path_dump}/plots_fup/SNPhotoz.png")
    print(
        f'SNphotoz<0.5 {len(potential_fup_nohostz_saltz[potential_fup_nohostz_saltz["zHD"]<0.5])}'
    )
    print(
        f'SNphotoz<0.3 {len(potential_fup_nohostz_saltz[potential_fup_nohostz_saltz["zHD"]<0.3])}'
    )
    plt.clf()
    plt.hist(potential_fup_nohostz_saltz["HOSTGAL_MAG_r"])
    plt.xlabel("HOSTGAL MAG r")
    plt.savefig(f"{path_dump}/plots_fup/HOSTGAL_MAG_r.png")
    print(
        f"HOSTGAL MAG r<22 {len(potential_fup_nohostz_saltz[potential_fup_nohostz_saltz['HOSTGAL_MAG_r']<22])}"
    )

    #
    print(
        f"# of events with SNphoto z SALT fit {len(salt_fits_noz)} from which {len(photoIa_noz_saltz)} are photoIa"
    )
    df_stats = mu.cuts_deep_shallow(
        photoIa_noz_saltz,
        photoIa_wz_JLA,
        cut="Converging SALT2 and redshift fit",
        df_stats=df_stats,
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

    logger.info("")
    logger.info("JLA CUTS")
    # JLA here includes redshift range
    photoIanoz_saltz_JLA = su.apply_JLA_cut(photoIa_noz_saltz)

    lu.print_green(
        f"photoIa_noz set 0 with JLA cuts (SNphoto z): {len(photoIanoz_saltz_JLA)}"
    )
    cuts.spec_subsamples(photoIanoz_saltz_JLA, logger)
    overlap_photoIa(photoIanoz_saltz_JLA, photoIa_wz, photoIa_wz_JLA, mssg="")

    df_stats = mu.cuts_deep_shallow(
        photoIanoz_saltz_JLA, photoIa_wz_JLA, df_stats=df_stats, cut="HQ"
    )

    logger.info("")
    logger.info("SAMPLE CONTAMINATION")
    # hack to use the same contaminant inspection
    dic_photoIa_sel = {"average_probability_set_0": photoIanoz_saltz_JLA}
    try:
        dic_tag_SNIDs = cuts.stats_possible_contaminants(
            dic_photoIa_sel, method_list=["average_probability_set"]
        )
    except Exception:
        lu.print_red("Lists are private. For reproducibility, contact author")
        raise ValueError
    # eliminate the AGNs
    SNIDs_to_eliminate = [
        k
        for k in dic_tag_SNIDs[0]["OzDES_AGN"]
        if k not in [1286337, 1246527, 1370320, 1643063, 1252970]  # >1'' from AGN
    ]
    lu.print_yellow(
        "Eliminating tagged close-by AGNs (rest of results)", SNIDs_to_eliminate
    )
    photoIanoz_saltz_JLA = photoIanoz_saltz_JLA[
        ~photoIanoz_saltz_JLA.SNID.isin(SNIDs_to_eliminate)
    ]
    photoIanoz_SNphotoz_HQ = photoIanoz_saltz_JLA[
        (photoIanoz_saltz_JLA.zHD > 0.05) & (photoIanoz_saltz_JLA.zHD < 1.2)
    ]

    df_stats = mu.cuts_deep_shallow(
        photoIanoz_SNphotoz_HQ, photoIa_wz_JLA, df_stats=df_stats, cut="z<1.2"
    )

    # save sample
    cols_to_keep = [
        "SNID",
        "IAUC",
        "SNTYPE",
        "REDSHIFT_FINAL",
        "REDSHIFT_FINAL_ERR",
        "HOSTGAL_SPECZ",
        "HOSTGAL_SPECZ_ERR",
        "HOSTGAL_LOGMASS",
        "HOSTGAL_LOGMASS_ERR",
        "HOSTGAL_MAG_r",
        "HOSTGAL_MAGERR_r",
        "zHD",
        "zHDERR",
        "x1",
        "x1ERR",
        "c",
        "cERR",
        "FITPROB",
        'all_class0_S_0',
        'all_class0_S_55',
        'all_class0_S_100',
        'all_class0_S_1000',
        'all_class0_S_30469',
        'average_probability_set_0'
    ]
    # Open the file for writing
    with open(f"{path_samples}/DES_noredshift_Moller2024_w_SALTandhostinfo.csv", "w") as f:
        # Write a comment in the header
        f.write("# Moller et al. 2024 \n")
        f.write("# SNe Ia selected with SuperNNova without host-galaxy redshifts \n")
        f.write(
            "# This is NOT the cosmology sample nor the SALT values used for cosmology In DES+2024 \n"
        )
        f.write("# Contains HQ cuts using SNphoto z fit with SALT2 \n")
        f.write("# zHD c x1 are derived simultaneously  \n")
        f.write("# all_class0_s* probabilities for different models  \n")
        f.write("# ensemble=average_probability_set_0  \n")
        f.write("# REDSHIFT_FINAL<0 means no spectroscopic redshift (either SN/host is available)  \n")
    # Write the DataFrame to CSV, appending to the existing file
    photoIanoz_SNphotoz_HQ[cols_to_keep].to_csv(
        f"{path_samples}/DES_noredshift_Moller2024_w_SALTandhostinfo.csv", mode="a", index=False
    )
    print(f"Saved {len(photoIanoz_SNphotoz_HQ)} photoIanoz saltz HQ")

    df_stats = mu.cuts_deep_shallow(
        photoIanoz_SNphotoz_HQ,
        photoIa_wz_JLA,
        df_stats=df_stats,
        cut="HQ (wo AGNs) + zrange",
    )

    # redshift greater than 1.2
    photoIanoz_saltz_JLA_z_gt_12 = photoIanoz_saltz_JLA[photoIanoz_saltz_JLA.zHD > 1.2]
    photoIanoz_saltz_JLA_z_gt_12_wzspe = photoIanoz_saltz_JLA_z_gt_12[
        photoIanoz_saltz_JLA_z_gt_12.REDSHIFT_FINAL > 0
    ]
    print(f"PhotoIanoz JLA SNphotoz>1.2  {len(photoIanoz_saltz_JLA_z_gt_12)}")
    print(
        f"{len(photoIanoz_saltz_JLA_z_gt_12_wzspe)} with zspe median:{round(photoIanoz_saltz_JLA_z_gt_12_wzspe['REDSHIFT_FINAL'].median(),2)} max:{round(photoIanoz_saltz_JLA_z_gt_12_wzspe['REDSHIFT_FINAL'].max(),2)}"
    )

    lu.print_blue("Stats")
    df_stats[[k for k in df_stats.keys() if k != "cut"]] = df_stats[
        [k for k in df_stats.keys() if k != "cut"]
    ].astype(int)
    print(df_stats.to_latex(index=False))
    print("")

    df_stats_fup = mu.fup_hostgals_stats(
        photoIanoz_SNphotoz_HQ,
        sngals,
        photoIa_wz_JLA,
        df_stats=df_stats_fup,
        sample="HQ",
        verbose=True,
    )

    lu.print_blue("Stats FUP")
    df_stats_fup[[k for k in df_stats_fup.keys() if k != "sample"]] = df_stats_fup[
        [k for k in df_stats_fup.keys() if k != "sample"]
    ].astype(int)
    print(df_stats_fup.to_latex(index=False))
    print("")

    #
    # Plots
    #

    pu.hist_fup_targets(df_stats_fup, path_plots=path_plots)

    # HOST
    # hist hostgalmag
    list_df = [photoIa_noz_001, photoIa_noz_saltz, photoIanoz_SNphotoz_HQ]
    list_labels = ["SNN>0.001", "SNN>0.5", "SNN>0.5 + HQ"]
    pu.hist_HOSTGAL_MAG_r_vs_REDSHIFT(list_df, list_labels, path_plots=path_plots)

    # distributions of new events
    list_df = [
        photoIa_noz_saltz,
        photoIa_noz_saltz[~photoIa_noz_saltz.SNID.isin(photoIa_wz_JLA.SNID.values)],
        photoIa_wz_JLA,
    ]
    list_labels = [
        "Photometric SNe Ia (SNphoto z)",
        "Photometric SNe Ia (SNphoto z) not in M22",
        "M22",
    ]
    pu.plot_mosaic_histograms_listdf(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix="_newJLA",
        list_vars_to_plot=["zHD", "c", "x1"],
        data_color_override=True,
    )

    logger.info("")
    logger.info("NOZ vs. M22")

    # common
    print("COMMON EVENTS")
    overlap = photoIanoz_SNphotoz_HQ[
        photoIanoz_SNphotoz_HQ.SNID.isin(photoIa_wz_JLA.SNID.values)
    ]
    print(
        f"Overlap photoIa no z and with z {len(overlap)} ~{round(100*len(overlap)/len(photoIa_wz_JLA),2)}%"
    )
    for k in ["zHD", "c", "x1"]:
        overlap[f"delta_{k}"] = abs(overlap[k] - overlap[f"{k}_zspe"])
        n_good_overlap = len(overlap[overlap[f"delta_{k}"] < 0.1])
        print(
            f"   Good {k}: (SNphoto - zspe<0.1) {n_good_overlap} {round(n_good_overlap*100/len(overlap),2)}% ",
        )
    # overlap SNphotoz salt fit effect
    # terrible way of coding this
    # but it is done to mimic sims format
    tmp_SNphotoz = salt_fits_noz.add_suffix("_SNphotoz")
    tmp_SNphotoz = tmp_SNphotoz.rename(columns={"SNID_SNphotoz": "SNID"})
    df_tmp = pd.merge(overlap, tmp_SNphotoz, on="SNID")
    pu.plot_scatter_mosaic_SNphotoz_biases(
        [df_tmp],
        ["overlap"],
        path_out=f"{path_plots}/scatter_overlapJLA_SNphotoz_vs_ori.png",
        afterHQ=True,
    )
    pu.plot_delta_vs_var(
        df_tmp,
        "zHD",
        "zPHOT_SNphotoz",
        f"{path_plots}/scatter_z_overlapJLA_SNphotoz_vs_ori.png",
        ylabel="spectroscopic - SNphoto redshift",
        xlabel="spectroscopic redshift",
    )
    # all w zspe
    df_IaHQ_tmp = pd.merge(photoIanoz_SNphotoz_HQ, tmp_SNphotoz, on="SNID")
    df_IaHQ_tmp = df_IaHQ_tmp[df_IaHQ_tmp["zHD"] > 0]
    df_IaHQ_tmp = df_IaHQ_tmp.rename(columns={"zPHOTERR_SNphotoz": "zPHOT_SNphotozERR"})
    df_tmp = df_tmp.rename(columns={"zPHOTERR_SNphotoz": "zPHOT_SNphotozERR"})
    pu.plot_list_delta_vs_var(
        [df_IaHQ_tmp, df_tmp],
        "zHD",
        "zPHOT_SNphotoz",
        f"{path_plots}/scatter_deltaz_HQ.png",
        ylabel="spectroscopic - SNphoto redshift",
        xlabel="spectroscopic redshift",
        labels=["SNN>0.5 + HQ", "SNN>0.5 + HQ in M22"],
    )

    dic_venn = {
        "M22": set(photoIa_wz_JLA.SNID.values),
        "this work SNN>0.5": set(photoIa_noz.SNID.values),
        "this work SNN>0.5 + HQ": set(photoIanoz_SNphotoz_HQ.SNID.values),
    }
    pu.plot_venn(dic_venn, path_plots=path_plots, suffix="all")

    dic_venn = {
        "M22": set(photoIa_wz_JLA.SNID.values),
        "this work SNN>0.5": set(photoIa_noz.SNID.values),
    }
    pu.plot_venn(dic_venn, path_plots=path_plots, suffix="M2205")

    plt.clf()
    plt.figure(figsize=(16, 8))
    sets = [
        set(photoIa_wz_JLA.SNID.values),
        set(photoIa_noz.SNID.values),
        set(photoIanoz_SNphotoz_HQ.SNID.values),
    ]
    labels = ["M22", "SNN>0.5", "SNN>0.5 + HQ"]
    supervenn(sets, labels, side_plots=False)
    plt.ylabel("Datasets")
    plt.xlabel(" ")
    plt.savefig(f"{path_plots}/alternative_to_venn.png")

    # lost
    print("LOST M22")
    lost_photoIa_wz_JLA = photoIa_wz_JLA[
        ~photoIa_wz_JLA.SNID.isin(photoIanoz_SNphotoz_HQ.SNID.values)
    ]
    print("Lost M22 JLA", len(lost_photoIa_wz_JLA))
    sel = salt_fits_noz[salt_fits_noz.SNID.isin(lost_photoIa_wz_JLA.SNID.values)]
    print("   have SNphoto z fit", len(sel))
    sel = su.apply_JLA_cut(sel)
    print("   pass JLA cuts", len(sel))

    # lost SNphoto z salt fit effect
    # again terrible way of doing this
    tmp_SNphotoz = salt_fits_noz.add_suffix("_SNphotoz")
    tmp_SNphotoz = tmp_SNphotoz.rename(columns={"SNID_SNphotoz": "SNID"})
    df_tmp = pd.merge(lost_photoIa_wz_JLA, tmp_SNphotoz, on="SNID")
    pu.plot_scatter_mosaic_SNphotoz_biases(
        [df_tmp],
        ["lost_photoIa_wz_JLA"],
        path_out=f"{path_plots}/scatter_lostJLA_SNphotoz_vs_ori.png",
        afterHQ=True,
    )
    pu.plot_delta_vs_var(
        df_tmp,
        "zHD",
        "zPHOT_SNphotoz",
        f"{path_plots}/scatter_z_lostJLA_SNphotoz_vs_ori.png",
    )
    pu.plot_delta_vs_var(
        df_tmp,
        "c",
        "c_SNphotoz",
        f"{path_plots}/scatter_c_lostJLA_SNphotoz_vs_ori.png",
    )

    for k in ["zHD", "c", "x1"]:
        df_tmp[f"delta_{k}"] = abs(df_tmp[f"{k}_SNphotoz"] - df_tmp[f"{k}"])
        n_good_tmp = len(df_tmp[df_tmp[f"delta_{k}"] < 0.1])
        print(
            f"   Good {k}: (SNphoto - zspe<0.1) {n_good_tmp} {round(n_good_tmp*100/len(df_tmp),2)}%",
        )

    # New events
    print("NEW")
    photoIa_noz_wz_notsel_photoIawz = photoIanoz_SNphotoz_HQ[
        (photoIanoz_SNphotoz_HQ["REDSHIFT_FINAL"] > 0)
        & (~photoIanoz_SNphotoz_HQ["SNID"].isin(photoIa_wz.SNID.values))
    ]
    print(f"new with zspe {len(photoIa_noz_wz_notsel_photoIawz)}")
    # photoIa_noz_wz_notsel_photoIawz.to_csv(f"{path_samples}/new_wzspe.csv")
    pu.plot_mosaic_histograms_listdf(
        [photoIa_noz_wz_notsel_photoIawz],
        list_labels=["newJLA"],
        path_plots=path_plots,
        suffix="new_JLA_wz",
        list_vars_to_plot=["REDSHIFT_FINAL", "average_probability_set_0"],
    )

    logger.info("")
    logger.info("SIMULATIONS: EFFICIENCY & CONTAMINATION (REALISTIC = NOT BALANCED)")

    # load predictions of snn
    sim_preds = du.load_merge_all_preds(
        path_class=args.path_sim_class,
        model_name="vanilla_S_*_none*_cosmo_quantile_lstm_64x4_0.05_1024_True_mean",
        norm="cosmo_quantile",
    )
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
    # SNN >0.001
    df_txt_stats_noz = mu.get_multiseed_performance_metrics(
        sim_preds["cosmo_quantile"],
        key_pred_targ_prefix="predicted_target_average_probability_001_set_",
        list_seeds=list_seeds_sets,
        df_txt=df_txt_stats_noz,
        dic_prefilled_keywords={
            "norm": "cosmo_quantile",
            "dataset": "not balanced",
            "method": "ensemble (prob. av.) >0.001",
        },
    )
    print(
        df_txt_stats_noz[
            ["method", "notbalanced_accuracy", "efficiency", "purity"]
        ].to_latex(index=False)
    )
    # reviewers comment
    # from simulations, establish number of SNe Ia expected by each cut for host fup
    sim_Ia = sim_preds["cosmo_quantile"][sim_preds["cosmo_quantile"].target==0]
    sim_Ia_1realisationDES = sim_Ia.sample(frac=1/30,random_state=0)
    sim_Ia_1realisationDES_selIa = sim_Ia_1realisationDES[sim_Ia_1realisationDES['predicted_target_average_probability_001_set_0']==0]
    sim_nonIa = sim_preds["cosmo_quantile"][sim_preds["cosmo_quantile"].target==1]
    sim_nonIa_1realisationDES = sim_nonIa.sample(frac=1/30,random_state=0)
    sim_nonIa_1realisationDES_selIa = sim_nonIa_1realisationDES[sim_nonIa_1realisationDES['predicted_target_average_probability_001_set_0']==0]
    print(f"One realisation DES has: Ia {len(sim_Ia_1realisationDES)} nonIa {len(sim_nonIa_1realisationDES)}")
    print(f"One realisation DES SNN>0.001 selects: Ia {len(sim_Ia_1realisationDES_selIa)} nonIa {len(sim_nonIa_1realisationDES_selIa)}")

    sim_Ia_1realisationDES_selIa05 = sim_Ia_1realisationDES[sim_Ia_1realisationDES['predicted_target_average_probability_set_0']==0]
    sim_nonIa_1realisationDES_selIa05 = sim_nonIa_1realisationDES[sim_nonIa_1realisationDES['predicted_target_average_probability_set_0']==0]
    print(f"One realisation DES SNN>0.5 selects: Ia {len(sim_Ia_1realisationDES_selIa05)} nonIa {len(sim_nonIa_1realisationDES_selIa05)}")

    logger.info("")
    logger.info("SIMULATIONS: SNphoto z SALT2 FIT")
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

    tmp_sim_saltz = sim_saltz.add_suffix("_SNphotoz")
    tmp_sim_saltz = tmp_sim_saltz.rename(columns={"SNID_SNphotoz": "SNID"})
    sim_all_fits = pd.merge(sim_fits, tmp_sim_saltz)

    # define simulated photo sample
    sim_cut_photoIa = sim_preds["cosmo_quantile"][f"average_probability_set_0"] > 0.5
    sim_photoIa = sim_preds["cosmo_quantile"][sim_cut_photoIa]

    logger.info("")
    logger.info("SIMULATIONS: CLASSIFICATION EFFICIENCY")
    # the SNphoto z saltz fit may introduce some large biases

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

    # 2. compraing data with sims
    variable = "m0obs_i"
    quant = 0.01
    min_var = photoIanoz_SNphotoz_HQ[variable].quantile(quant)
    lu.print_yellow(
        f"Not using photo Ias with {variable}<{min_var} (equivalent to quantile {quant}, possible if low-z)"
    )
    # fixed z for sim
    df, minv, maxv = du.data_sim_ratio(
        photoIanoz_SNphotoz_HQ,
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
        photoIanoz_SNphotoz_HQ,
        sim_saltz_Ia_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="SNphotoz",
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
    min_var = photoIanoz_SNphotoz_HQ[variable].quantile(quant)
    tmo, tmominv, mtmoaxv = du.data_sim_ratio(
        photoIanoz_SNphotoz_HQ,
        sim_Ia_fits_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="simz",
    )
    tmo, tmominv, mtmoaxv = du.data_sim_ratio(
        photoIanoz_SNphotoz_HQ,
        sim_saltz_Ia_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
        suffix="SNphotoz",
    )

    lu.print_blue("Loading metadata sims")
    sim_metadata = du.load_headers(args.path_sim_headers)
    metadata_sim_Ia_fits_JLA = pd.merge(sim_Ia_fits_JLA, sim_metadata, on="SNID")
    # sanity ratio z
    variable = "HOSTGAL_MAG_r"
    quant = 0.01
    min_var = photoIanoz_SNphotoz_HQ[variable].quantile(quant)
    tmo, tmominv, mtmoaxv = du.data_sim_ratio(
        photoIanoz_SNphotoz_HQ,
        metadata_sim_Ia_fits_JLA,
        var=variable,
        path_plots=path_plots,
        min_var=min_var,
    )

    logger.info("")
    logger.info("SAMPLE PROPERTIES")
    pu.overplot_salt_distributions_lists(
        [
            sim_saltz_Ia_JLA,
            photoIanoz_SNphotoz_HQ,
        ],
        path_plots=path_plots,
        list_labels=[
            "sim Ia HQ (SNphoto z)",
            "SNe Ia HQ (SNphoto z)",
        ],
        suffix="noz",
        sim_scale_factor=30,
    )

    pu.overplot_salt_distributions_lists(
        [sim_Ia_fits_JLA, sim_saltz_Ia_JLA, photoIanoz_SNphotoz_HQ],
        path_plots=path_plots,
        list_labels=[
            "sim Ia HQ (true z)",
            "sim Ia HQ (SNphoto z)",
            "SNe Ia HQ (SNphoto z)",
        ],
        suffix="noz_simz",
        sim_scale_factor=30,
    )

    # add m0obs_i
    pu.plot_mosaic_histograms_listdf(
        [
            sim_Ia_fits_JLA,
            sim_saltz_Ia_JLA,
            photoIanoz_SNphotoz_HQ,
        ],
        list_labels=[
            "sim Ia HQ (true z)",
            "sim Ia HQ (SNphoto z)",
            "SNe Ia HQ (SNphoto z)",
        ],
        path_plots=path_plots,
        suffix="noz_wm0obsi",
        list_vars_to_plot=["zHD", "c", "x1", "m0obs_i"],
        norm=1 / 30,  # using small sims
    )

    # deep and shallow
    sim_Ia_fits_JLA = du.tag_deep_shallow(sim_Ia_fits_JLA)
    sim_saltz_Ia_JLA = du.tag_deep_shallow(sim_saltz_Ia_JLA)
    photoIanoz_SNphotoz_HQ = du.tag_deep_shallow(photoIanoz_SNphotoz_HQ)

    pu.overplot_salt_distributions_lists_deep_shallow(
        [sim_Ia_fits_JLA, sim_saltz_Ia_JLA, photoIanoz_SNphotoz_HQ],
        list_labels=[
            "simulations (true z)",
            "simulations (SNphoto z)",
            "DES SNe Ia HQ (SNphoto z)",
        ],
        path_plots=path_plots,
        suffix="deep_and_shallow_fields",
        sim_scale_factor=30,  # small sims
    )

    logger.info("")
    logger.info("Sample comparisons")
    # Want to compare:
    # photoIa_noz with salt SNphoto z
    # photoIa_noz with salt zspe when available else SNphoto z
    # photoIa_wz with salt zspe

    # For a mixed histogram
    cols_to_keep = ["SNID", "zHD", "c", "x1"]
    tmp = photoIanoz_SNphotoz_HQ[cols_to_keep]
    tmp_salt_fits_wz = salt_fits_wz.rename(
        columns={"zHD_zspe": "zHD", "c_zspe": "c", "x1_zspe": "x1"}
    )
    tmp.update(tmp_salt_fits_wz[cols_to_keep])

    pu.plot_mosaic_histograms_listdf(
        [tmp, photoIanoz_SNphotoz_HQ, photoIa_wz_JLA],
        list_labels=[
            "DES SNe Ia HQ (z mixed)",
            "DES SNe Ia HQ (SNphoto z)",
            "M22",
        ],
        path_plots=path_plots,
        suffix="comparisonDES5_mixed",
        list_vars_to_plot=["zHD", "c", "x1"],
        data_color_override=True,
        chi_bins=False,
    )

    # DES5 samples comparissons
    tmpsalt2 = pd.merge(tmpsalt, df_metadata[["SNID", "HOSTGAL_MAG_r"]], how="left")
    spec_ia = tmpsalt2[tmpsalt2.SNTYPE.isin(cu.spec_tags["Ia"])]
    photoIa_wz_JLA = pd.merge(photoIa_wz_JLA, tmpsalt[["SNID", "m0obs_i"]], how="left")

    pu.plot_mosaic_histograms_listdf(
        [photoIanoz_SNphotoz_HQ, photoIa_wz_JLA, spec_ia],
        list_labels=[
            "DES SNe Ia HQ (SNphoto z)",
            "DES SNe Ia M22",
            "DES SNe Ia spectroscopic",
        ],
        path_plots=path_plots,
        suffix="comparisonDES5",
        list_vars_to_plot=["zHD", "c", "x1", "m0obs_i", "HOSTGAL_MAG_r"],
        data_color_override=True,
        chi_bins=False,
        use_samples_color_palette=True,
    )

    # All samples properties
    fig = plt.figure(figsize=(20, 20), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    for i, var in enumerate(["HOSTGAL_MAG_r", "x1", "c"]):
        # binning
        list_df = []
        for df in [photoIanoz_SNphotoz_HQ, photoIa_wz_JLA, spec_ia]:
            sel = df[df["HOSTGAL_MAG_r"] < 30] if var == "HOSTGAL_MAG_r" else df
            sel[f"zHD_bin"] = pd.cut(sel.loc[:, ("zHD")], pu.bins_dic["zHD"])
            list_df.append(sel)
        axs[i] = pu.plot_errorbar_binned(
            list_df,
            [
                "DES SNe Ia HQ (SNphoto z)",
                "DES SNe Ia M22",
                "DES SNe Ia spectroscopic",
            ],
            axs=axs[i],
            binname="zHD_bin",
            varx="zHD",
            vary=var,
            data_color_override=True,
            color_list=[
                pu.SAMPLES_COLORS[k]
                for k in [
                    "DES SNe Ia HQ (SNphoto z)",
                    "DES SNe Ia M22",
                    "DES SNe Ia spectroscopic",
                ]
            ],
        )

        ylabel = "host r magnitude" if var == "HOSTGAL_MAG_r" else var
        axs[i].set_ylabel(ylabel, fontsize=20)
    axs[i].legend(fontsize=16)
    axs[i].set_xlabel("z", fontsize=20)
    plt.savefig(f"{path_plots}/2ddist_all_sample_zHD.png")

    # Only host mag r
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    # binning
    list_df = []
    for df in [photoIanoz_SNphotoz_HQ, photoIa_wz_JLA, spec_ia]:
        sel = df[df["HOSTGAL_MAG_r"] < 30]
        sel[f"zHD_bin"] = pd.cut(sel.loc[:, ("zHD")], pu.bins_dic["zHD"])
        list_df.append(sel)
    fig = pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia HQ (SNphoto z)",
            "DES SNe Ia M22",
            "DES SNe Ia spectroscopic",
        ],
        binname="zHD_bin",
        varx="zHD",
        vary="HOSTGAL_MAG_r",
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS["noz"],
            pu.SAMPLES_COLORS["M22"],
            pu.SAMPLES_COLORS["specIa"],
        ],
        marker_size=15,
    )
    plt.ylabel("host r magnitude", fontsize=25)
    plt.legend(fontsize=20)
    plt.xlabel("z", fontsize=25)
    plt.savefig(f"{path_plots}/2ddist_all_sample_HOSTGAL_MAG_r_zHD.png")
