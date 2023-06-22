import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import conf_utils as cu
from utils import plot_utils as pu
from utils import data_utils as du
from utils import metric_utils as mu
from utils import logging_utils as lu

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

"""_summary_

BEWARE! 
You need to define an environment variable snn where 
https://github.com/supernnova/SuperNNova.git is installed
"""

snn = os.getenv("snn")  # SuperNNova installation
sys.path.append(snn)
from supernnova.validation.validate_onthefly import classify_lcs


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


def sampling_criteria(df, n_filters, n_unique_nights, snr):
    """_summary_

    Args:
        df (pd.DataFrame): data
        n_filters (int): number of filters in each light-curve required
        n_unique_nights (int): number of measurements in each light-curve required

    Returns:
        _type_: _description_
    """
    # unique flt occurences
    tmp = df.groupby("SNID")["FLT"].apply(lambda x: len(list(np.unique(x))))
    id1 = tmp[tmp >= n_filters].index
    # unique night occurences
    df["MJDint"] = df["MJD"].astype(int)
    tmp2 = df.groupby("SNID")["MJDint"].apply(lambda x: len(list(np.unique(x))))
    id2 = tmp2[tmp2 >= n_unique_nights].index
    # snr
    sel = df[df["SNR"] > snr]
    tmp3 = sel.groupby("SNID")["MJDint"].apply(lambda x: len(list(np.unique(x))))
    id3 = tmp3[tmp3 >= n_unique_nights].index
    # flt + nights requirement
    inters = list(set(id1) & set(id2) & set(id3))
    return inters


def reformat_preds(pred_probs, ids=None):
    """Reformat SNN predictions to a DataFrame

    # TO DO: suppport nb_inference != 1
    """
    num_inference_samples = 1

    d_series = {}
    for i in range(pred_probs[0].shape[1]):
        d_series["SNID"] = []
        d_series[f"prob_class{i}"] = []
    for idx, value in enumerate(pred_probs):
        d_series["SNID"] += [ids[idx]] if len(ids) > 0 else idx
        value = value.reshape((num_inference_samples, -1))
        value_dim = value.shape[1]
        for i in range(value_dim):
            d_series[f"prob_class{i}"].append(value[:, i][0])
    preds_df = pd.DataFrame.from_dict(d_series)

    # get predicted class
    preds_df["pred_class"] = np.argmax(pred_probs, axis=-1).reshape(-1)

    return preds_df


def get_lc_stats(df, dfmetadata):
    # get some statistics of the photometry
    df["MJDint"] = df["MJD"].astype(int)
    tmp = df.groupby("SNID")["MJDint"].apply(lambda x: len(list(np.unique(x))))
    df_unights = pd.DataFrame()
    df_unights["SNID"] = tmp.index
    df_unights["unights"] = tmp.values

    tmp2 = df.groupby("SNID")["FLT"].apply(lambda x: len(list(np.unique(x))))
    df_uflt = pd.DataFrame()
    df_uflt["SNID"] = tmp2.index
    df_uflt["uflt"] = tmp2.values

    df_unuf = pd.merge(df_unights, df_uflt)

    tmp3 = df.groupby("SNID")["MJDint"].count()
    df_photo_points = pd.DataFrame()
    df_photo_points["SNID"] = tmp3.index
    df_photo_points["photo_points"] = tmp3.values

    df_out = pd.merge(df_unuf, df_photo_points)

    return pd.merge(dfmetadata, df_out)


def early_class(df_photo_sel, df_metadata, photoIa_wz_JLA, df_stats, path_model):
    """_summary_

    Args:
        df_photo_sel (_type_): _description_
        df_metadata (_type_): _description_
        photoIa_wz_JLA (_type_): _description_
        df_stats (_type_): _description_
        path_model (_type_): _description_

    Returns:
        _type_: _description_
    """
    nflt = 1
    n_nights = 2
    snr = 5

    SNID_measurements_criteria = sampling_criteria(df_photo_sel, nflt, n_nights, snr)
    df_metadata_sampling_trigger = df_metadata[
        df_metadata.SNID.isin(SNID_measurements_criteria)
    ]

    df_metadata_sampling_trigger_u = get_lc_stats(
        df_photo_sel, df_metadata_sampling_trigger
    )

    df_stats = mu.cuts_deep_shallow_eventmag(
        df_metadata_sampling_trigger_u,
        photoIa_wz_JLA,
        df_photo_sel,
        df_stats=df_stats,
        cut=f"-7<t<20 nflt:{nflt} nights:{n_nights} snr:{snr}",
    )

    # PREDICTIONS
    # 1. reformat photometry for SuperNNova
    missing_cols = [
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_SPECZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ_ERR",
    ]
    df_snn = df_photo_sel.copy()
    for k in missing_cols:
        df_snn[k] = np.zeros(len(df_snn))
    df_snn = df_snn.sort_values(by=["MJD"])

    print("Obtain predictions")
    ids_preds, pred_probs = classify_lcs(df_snn, path_model, "cpu")
    preds_df = reformat_preds(pred_probs, ids=df_snn.SNID.unique())
    preds_df = pd.merge(
        preds_df,
        df_metadata_sampling_trigger_u[
            [
                "SNID",
                "IAUC",
                "SNTYPE",
                "unights",
                "uflt",
                "photo_points",
                "REDSHIFT_FINAL",
                "PRIVATE(DES_transient_status)",
                "HOSTGAL_MAG_r",
            ]
        ],
    )

    for thres in [0.1, 0.2, 0.3, 0.4, 0.5]:
        df_stats, _ = mu.cuts_deep_shallow_eventmag(
            preds_df[preds_df.prob_class0 > thres],
            photoIa_wz_JLA,
            df_photo_sel,
            df_stats=df_stats,
            cut=f"SNN>{thres}",
            return_extra_df=True,
        )

    return df_stats


if __name__ == "__main__":

    DES5 = os.getenv("DES5")
    DES = os.getenv("DES")

    parser = argparse.ArgumentParser(description="Early classification")

    parser.add_argument(
        "--path_dump",
        default=f"{DES5}/DES5YR_SNeIa_nohost/dump_DES5YR",
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
        "--path_model",
        type=str,
        default=f"{DES5}/snndump_26XBOOSTEDDES/models/vanilla_S_100_CLF_2_R_none_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean/vanilla_S_100_CLF_2_R_none_photometry_DF_1.0_N_cosmo_lstm_64x4_0.05_1024_True_mean.pt",
        help="Path to model for predictions",
    )

    # Init
    args = parser.parse_args()
    path_dump = args.path_dump
    path_model = args.path_model

    path_plots = f"{path_dump}/plots_early_class/"
    os.makedirs(path_plots, exist_ok=True)

    # load previous samples for comparisson
    path_sample = "./samples/BaselineDESsample_JLAlikecuts.csv"
    photoIa_wz_JLA = pd.read_csv(path_sample)
    print(f"M22 sample {len(photoIa_wz_JLA)}")

    print("Load data")
    df_metadata = du.load_headers(args.path_data)

    # need to load photometry + PEAKMJD estimate
    print("Load photometry")
    df_photometry = du.load_photometry(args.path_data)
    tmp = len(df_photometry.SNID.unique())
    tmp2 = len(df_photometry)
    df_photometry["phot_reject"] = df_photometry["PHOTFLAG"].apply(
        lambda x: False
        if len(set([8, 16, 32, 64, 128, 256, 512]).intersection(set(powers_of_two(x))))
        > 0
        else True
    )
    df_photometry = df_photometry[df_photometry["phot_reject"]]

    df_metadata_u = get_lc_stats(df_photometry, df_metadata)
    # STATS all metadata
    df_stats = mu.cuts_deep_shallow_eventmag(
        df_metadata_u,
        photoIa_wz_JLA,
        df_photometry,
        cut="DES-SN 5-year candidate sample",
    )
    # 'PRIVATE(DES_numepochs_ml)'>2 already applied in this
    # transient_status as Smith 2018
    # only asking >0 instead of single season
    presel = df_metadata_u[df_metadata_u["PRIVATE(DES_transient_status)"] > 0]
    idxs_presel = presel.SNID.values
    df_stats = mu.cuts_deep_shallow_eventmag(
        presel,
        photoIa_wz_JLA,
        df_photometry,
        df_stats=df_stats,
        cut="+ transient status",
    )

    print("PEAKMJD")
    # Using M22 peak
    salt_peak = du.load_salt_fits(
        f"{args.path_data}/DESALL_forcePhoto_real_snana_fits.SNANA.TEXT"
    )
    # Photometry
    df_peakpho = pd.merge(
        df_photometry[["SNID", "MJD", "FLT", "FLUXCAL", "FLUXCALERR", "PHOTFLAG"]],
        salt_peak[["SNID", "PKMJDINI", "SNTYPE"]],
        on="SNID",
        how="left",
    )
    df_peakpho["SNR"] = df_peakpho["FLUXCAL"] / df_peakpho["FLUXCALERR"]
    df_peakpho["window_time_cut"] = True
    mask = df_peakpho["MJD"] != -777.00
    df_peakpho["window_delta_time"] = df_peakpho["MJD"] - df_peakpho["PKMJDINI"]
    df_peakpho.loc[mask, "window_time_cut"] = df_peakpho["window_delta_time"].apply(
        lambda x: True if x < 5 and x > -30 else False
    )
    df_peakpho_sel = df_peakpho[
        (df_peakpho["window_time_cut"]) & (df_peakpho.SNID.isin(idxs_presel))
    ]
    lu.print_blue("Selected before peak (-30<p<5)")

    df_stats_peak = early_class(
        df_peakpho_sel, df_metadata, photoIa_wz_JLA, df_stats, path_model
    )

    print("TRIGGER")
    # Using PHOTFLAG 4096 (bit mask)
    # as PRIVATE(DES_mjd_trigger) in header may not be acurate
    estimate_trig = pd.read_csv(f"{args.path_data}/trigger_MJD.csv", delimiter=" ")
    peak_merged = salt_peak[["SNID", "PKMJDINI", "SNTYPE"]].merge(estimate_trig)
    peak_merged["Peak-trigger"] = peak_merged["PKMJDINI"] - peak_merged["trigger_MJD"]
    toplot_peak_merged = peak_merged[
        (peak_merged.PKMJDINI > 1)
    ]  # to eliminate non estimates
    list_spec_sntypes = [
        cu.spec_tags["Ia"],
        cu.spec_tags["Ia"] + cu.spec_tags["CC"] + cu.spec_tags["SLSN"],
    ]
    list_df_spec = [
        toplot_peak_merged[toplot_peak_merged["SNTYPE"].isin(k)]
        for k in list_spec_sntypes
    ]
    pu.plot_histograms_listdf(
        [toplot_peak_merged] + list_df_spec,
        ["DES-SN"] + ["spec Ia", "spec SN"],
        density=False,
        varx="Peak-trigger",
        outname=f"{path_plots}/peak-trigger.png",
        log_scale=True,
        nbins=30,
    )

    # list of events with peak variation near trigger
    # near_trigger = toplot_peak_merged[abs(toplot_peak_merged["Peak-trigger"]) < 20]
    # idxs_near_trigger = near_trigger["SNID"].values
    # list_spec_sntypes = [cu.spec_tags["Ia"], cu.spec_tags["CC"] + cu.spec_tags["SLSN"]]
    # list_near_trigger_spec = [
    #     near_trigger[near_trigger["SNTYPE"].isin(k)] for k in list_spec_sntypes
    # ]
    # df_stats = mu.cuts_deep_shallow_eventmag(
    #     presel[presel.SNID.isin(idxs_near_trigger)],
    #     photoIa_wz_JLA,
    #     df_photometry,
    #     df_stats=df_stats,
    #     cut="+ near trigger",
    # )

    # how about SNe Ia with t0 estimation?
    salt_JLA = du.load_salt_fits(f"{args.path_data}/FITOPT000.FITRES")
    JLA_merged = salt_JLA[["SNID", "PKMJD", "SNTYPE"]].merge(estimate_trig)
    JLA_merged["t0-trigger"] = JLA_merged["PKMJD"] - JLA_merged["trigger_MJD"]
    list_spec_sntypes = []

    pu.plot_histograms_listdf(
        [JLA_merged[JLA_merged["SNTYPE"].isin(cu.spec_tags["Ia"])]],
        ["spec Ia"],
        density=False,
        varx="t0-trigger",
        outname=f"{path_plots}/t0-trigger.png",
        log_scale=True,
        nbins=30,
    )

    # sadly trigger is not a great indicator
    # where SNe are...
    # other methods can be used once SN has reached maximum
    # but early, this is all we have for the time being
    #
    # EARLY CLASSIFICATION
    #

    # Photometry
    df_trigpho = pd.merge(
        df_photometry[["SNID", "MJD", "FLT", "FLUXCAL", "FLUXCALERR", "PHOTFLAG"]],
        estimate_trig[["SNID", "trigger_MJD"]],
        on="SNID",
        how="left",
    )
    df_trigpho["SNR"] = df_trigpho["FLUXCAL"] / df_trigpho["FLUXCALERR"]
    df_trigpho["window_time_cut"] = True
    mask = df_trigpho["MJD"] != -777.00
    df_trigpho["window_delta_time"] = df_trigpho["MJD"] - df_trigpho["trigger_MJD"]

    # Early classification
    # apply window of "SN-like event"
    # -7 (in case there is forced phot before trigger)
    df_trigpho.loc[mask, "window_time_cut"] = df_trigpho["window_delta_time"].apply(
        lambda x: True if x < 20 and x > -7 else False
    )
    df_trigpho_sel = df_trigpho[
        (df_trigpho["window_time_cut"])
        & (df_trigpho.SNID.isin(idxs_presel))
        # & (df_trigpho.SNID.isin(idxs_near_trigger))
    ]
    lu.print_blue(f"Selected -7<trigger<{20}")

    df_stats_trigger = early_class(
        df_trigpho_sel, df_metadata, photoIa_wz_JLA, df_stats, path_model
    )

    cols_to_print = [
        "cut",
        "total maglim<22.7",
        "specIa maglim<22.7",
        "M22 maglim<22.7",
        "nonIa maglim<22.7",
        "multiseason maglim<22.7",
        "photo_points",
        "photo_points_std",
        "unights",
        "unights_std",
    ]
    lu.print_blue("PEAK")
    print(df_stats_peak[cols_to_print])
    lu.print_blue("trigger")
    print(df_stats_trigger[cols_to_print].to_latex(index=False))

    pu.hist_fup_targets_early(
        df_stats_trigger,
        path_plots=path_plots,
        subsamples_to_plot=[
            "total maglim<22.7",
            "M22 maglim<22.7",
            "specIa maglim<22.7",
            "multiseason maglim<22.7",
            "nonIa maglim<22.7",
        ],
        colors=["maroon", "darkorange", "royalblue", "indigo", "grey"],
    )
    pu.hist_fup_targets_early(
        df_stats_trigger,
        path_plots=path_plots,
        subsamples_to_plot=[
            "total maglim<22.7",
            "M22 maglim<22.7",
            "specIa maglim<22.7",
            "multiseason maglim<22.7",
            "nonIa maglim<22.7",
        ],
        colors=["maroon", "darkorange", "royalblue", "indigo", "grey"],
        suffix="log",
        log_scale=True,
    )
    import ipdb

    ipdb.set_trace()