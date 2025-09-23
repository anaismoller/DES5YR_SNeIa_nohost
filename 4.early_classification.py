import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from myutils import conf_utils as cu
from myutils import plot_utils as pu
from myutils import data_utils as du
from myutils import metric_utils as mu
from myutils import logging_utils as lu

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

"""_summary_

BEWARE! 
You need to define an environment variable snn where 
https://github.com/supernnova/SuperNNova.git is installed
"""
print("not supported with uv/mise (need supernnova installation)")

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
    """Apply sampling criteria to light-curves

    Args:
        df (pd.DataFrame): data
        n_filters (int): number of filters in each light-curve required
        n_unique_nights (int): number of measurements in each light-curve required

    Returns:
        ids: SNIDs passing sampling
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
    """get some statistics for the light-curves

    Args:
        df (pd.DataFrame): selected photometry
        dfmetadata (_type_): selected metadata

    Returns:
        pd.DataFrame: stats
    """
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

    df_detections = df[df["photo_detection"]]
    tmp3 = df_detections.groupby("SNID")["MJDint"].count()
    df_photo_dets = pd.DataFrame()
    df_photo_dets["SNID"] = tmp3.index
    df_photo_dets["detections"] = tmp3.values

    df_out = pd.merge(df_unuf, df_photo_dets)

    return pd.merge(dfmetadata, df_out)


def early_class(
    df_photo_sel, df_metadata_sel, photoIa_wz_JLA, df_stats, path_model, cut_str
):
    """Get classification with photometry

    Args:
        df_photo_sel (pd.DataFrame): selected photometry
        df_metadata_sel (pd.DataFrame): selected metadata
        photoIa_wz_JLA (pd.DataFrame): M22
        df_stats (pd.DataFrame): statistics dataframe
        path_model (_type_): SNN model used for classification
        cut_str (string): description of selection cut

    Returns:
        pd.DataFrame: statistics dataframe
    """

    df_metadata_sampling_trigger_u = get_lc_stats(df_photo_sel, df_metadata_sel)

    df_stats = mu.cuts_deep_shallow_eventmag(
        df_metadata_sampling_trigger_u,
        photoIa_wz_JLA,
        df_photo_sel,
        df_stats=df_stats,
        cut=cut_str,
    )

    # PREDICTIONS
    # 1. reformat photometry for SuperNNova
    missing_cols = [
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_SPECZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ_ERR",
    ]
    df_snn_tmp = df_photo_sel.copy()
    for k in missing_cols:
        df_snn_tmp[k] = np.zeros(len(df_snn_tmp))
    df_snn_tmp = df_snn_tmp.sort_values(by=["SNID", "MJD"])
    df_snn_tmp = df_snn_tmp.replace([np.inf, -np.inf], np.nan, inplace=False)
    df_snn = df_snn_tmp.dropna()

    print("Obtain predictions (~run_on_the_fly)")

    num_elem = len(df_snn.SNID.unique())
    num_chunks = num_elem // 10 + 1
    list_chunks = np.array_split(df_snn.SNID.unique(), num_chunks)
    # Loop over chunks of SNIDs
    list_ids_preds = []
    list_pred_prob = []
    for chunk_idx in list_chunks:
        try:
            ids_preds_tmp, pred_probs_tmp = classify_lcs(
                df_snn[df_snn.SNID.isin(chunk_idx)], path_model, "cpu"
            )
            list_ids_preds.append(ids_preds_tmp)
            list_pred_prob.append(pred_probs_tmp)
        except Exception:
            print("ERROR on classification, data must be corrupted")
            raise ValueError
    ids_preds = [item for sublist in list_ids_preds for item in sublist]
    pred_probs = np.vstack(list_pred_prob)
    print("Reformat")
    preds_df = reformat_preds(pred_probs, ids=ids_preds)

    print("Merge")
    preds_df = pd.merge(
        preds_df,
        df_metadata_sampling_trigger_u[
            [
                "SNID",
                "IAUC",
                "SNTYPE",
                "unights",
                "uflt",
                "detections",
                "REDSHIFT_FINAL",
                "PRIVATE(DES_transient_status)",
                "HOSTGAL_MAG_r",
            ]
        ],
    )

    print("Set thresholds for selection")
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
    path_sample = "./samples/previous_works/Moller2022_DES5yr_SNeIa_whostz_JLA.csv"
    photoIa_wz_JLA = pd.read_csv(path_sample, comment="#")
    print(f"M22 sample {len(photoIa_wz_JLA)}")

    print("Load data")
    df_metadata = du.load_headers(args.path_data)

    # need to load photometry + PEAKMJD estimate
    print("Load photometry")
    df_photometry = du.load_photometry(args.path_data)
    tmp = len(df_photometry.SNID.unique())
    tmp2 = len(df_photometry)
    df_photometry["phot_reject"] = df_photometry["PHOTFLAG"].apply(
        lambda x: (
            False
            if len(
                set([8, 16, 32, 64, 128, 256, 512]).intersection(set(powers_of_two(x)))
            )
            > 0
            else True
        )
    )
    df_photometry = df_photometry[df_photometry["phot_reject"]]
    df_photometry["photo_detection"] = df_photometry["PHOTFLAG"].apply(
        lambda x: (
            True if len(set([4096]).intersection(set(powers_of_two(x)))) > 0 else False
        )
    )

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
        cut="+ transient status>0",
    )

    print("-30<PEAKMJD<1")
    # Using M22 peak
    salt_peak = du.load_salt_fits(
        f"{args.path_data}/DESALL_forcePhoto_real_snana_fits.SNANA.TEXT"
    )
    df_peakpho = pd.merge(
        df_photometry[
            [
                "SNID",
                "MJD",
                "FLT",
                "FLUXCAL",
                "FLUXCALERR",
                "PHOTFLAG",
                "photo_detection",
            ]
        ],
        salt_peak[["SNID", "PKMJDINI", "SNTYPE"]],
        on="SNID",
        how="left",
    )
    df_peakpho["SNR"] = df_peakpho["FLUXCAL"] / df_peakpho["FLUXCALERR"]
    df_peakpho["window_time_cut"] = True
    mask = df_peakpho["MJD"] != -777.00
    df_peakpho["window_delta_time"] = df_peakpho["MJD"] - df_peakpho["PKMJDINI"]
    df_peakpho.loc[mask, "window_time_cut"] = df_peakpho["window_delta_time"].apply(
        lambda x: True if x < 1 and x > -30 else False
    )
    df_peakpho_sel = df_peakpho[
        (df_peakpho["window_time_cut"]) & (df_peakpho.SNID.isin(idxs_presel))
    ]
    # apply sampling
    nflt = 1
    n_nights = 2
    snr = 5
    SNID_measurements_criteria = sampling_criteria(df_peakpho_sel, nflt, n_nights, snr)
    df_metadata_sel = df_metadata[df_metadata.SNID.isin(SNID_measurements_criteria)]
    df_stats_peak = early_class(
        df_peakpho_sel,
        df_metadata_sel,
        photoIa_wz_JLA,
        df_stats,
        path_model,
        "-30<peak<1",
    )

    print("LSST-like TRIGGER")
    # Using PHOTFLAG 4096 (bit mask)
    detections_tmp = df_photometry[df_photometry["photo_detection"]]
    detections_tmp = detections_tmp.sort_values(by=["SNID", "MJD"])
    detections_tmp = detections_tmp[["SNID", "MJD"]]
    LSSTtrigger_group = detections_tmp.groupby("SNID").min()
    LSSTestimate_trig = pd.DataFrame()
    LSSTestimate_trig["SNID"] = LSSTtrigger_group.index
    LSSTestimate_trig["LSSTtrigger_MJD"] = LSSTtrigger_group.MJD.values

    peak_merged = salt_peak[["SNID", "PKMJDINI", "SNTYPE"]].merge(LSSTestimate_trig)
    peak_merged["observed peak - LSST trigger"] = (
        peak_merged["PKMJDINI"] - peak_merged["LSSTtrigger_MJD"]
    )
    toplot_peak_merged = peak_merged[
        (peak_merged.PKMJDINI > 1)
    ]  # to eliminate non estimates
    df_trigpho = pd.merge(
        df_photometry[
            [
                "SNID",
                "MJD",
                "FLT",
                "FLUXCAL",
                "FLUXCALERR",
                "PHOTFLAG",
                "photo_detection",
            ]
        ],
        LSSTestimate_trig[["SNID", "LSSTtrigger_MJD"]],
        on="SNID",
        how="left",
    )
    df_trigpho["SNR"] = df_trigpho["FLUXCAL"] / df_trigpho["FLUXCALERR"]
    df_trigpho["window_time_cut"] = True
    mask = df_trigpho["MJD"] != -777.00
    df_trigpho["window_delta_time"] = df_trigpho["MJD"] - df_trigpho["LSSTtrigger_MJD"]

    lu.print_blue("Selected -7<LSST-trigger<20")
    df_trigpho.loc[mask, "window_time_cut"] = df_trigpho["window_delta_time"].apply(
        lambda x: True if x < 20 and x > -7 else False
    )
    df_trigphoLSST_sel = df_trigpho[
        (df_trigpho["window_time_cut"]) & (df_trigpho.SNID.isin(idxs_presel))
    ]
    # apply sampling
    nflt = 1
    n_nights = 2
    snr = 5
    SNID_measurements_criteria = sampling_criteria(
        df_trigphoLSST_sel, nflt, n_nights, snr
    )
    df_metadata_sel = df_metadata[df_metadata.SNID.isin(SNID_measurements_criteria)]
    df_stats_triggerLSST = early_class(
        df_trigphoLSST_sel,
        df_metadata_sel,
        photoIa_wz_JLA,
        df_stats,
        path_model,
        "-7<LSST-trigger<20",
    )

    pu.hist_fup_targets_early(
        df_stats_triggerLSST,
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
        df_stats_triggerLSST,
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

    #
    # With host galaxy photometric redshifts

    print("DES-like TRIGGER")
    print("2 detections within 30 days")
    # compute delta time between detections
    detections_tmp.reset_index(inplace=True, drop=True)
    detections_tmp["delta_time"] = detections_tmp["MJD"].diff()
    # Fill the first row with 0 to replace NaN
    detections_tmp.delta_time = detections_tmp.delta_time.fillna(0)
    # identify  indivdual lcs to reset delta_time to zero
    IDs = detections_tmp.SNID.values
    idxs = np.where(IDs[:-1] != IDs[1:])[0] + 1
    arr_delta_time = detections_tmp.delta_time.values
    arr_delta_time[idxs] = 0
    detections_tmp["delta_time"] = arr_delta_time

    #  check which SNID have two detections within a month
    # select detections within 30 days
    # and in different nights
    detections_within_30days = detections_tmp[
        (detections_tmp.delta_time < 30) & (detections_tmp.delta_time > 1)
    ]
    # 2 detections within 30 days
    detections_within_30days_grouped = detections_within_30days.groupby("SNID").count()
    SNID_detections_2_within_30days = detections_within_30days_grouped[
        detections_within_30days_grouped.MJD > 2
    ].index
    detections_within_30days = detections_within_30days[
        detections_within_30days.SNID.isin(SNID_detections_2_within_30days)
    ]
    # get second detection date as trigger
    DES_like_trigger = pd.DataFrame()
    DES_like_trigger["SNID"] = (
        detections_within_30days.groupby(["SNID"])["MJD"].nth(1).index
    )
    DES_like_trigger["MJD_1stdet"] = (
        detections_within_30days.groupby(["SNID"])["MJD"].nth(0).values
    )
    DES_like_trigger["MJD_2nddet"] = (
        detections_within_30days.groupby(["SNID"])["MJD"].nth(1).values
    )

    df_DEStrigpho = pd.merge(
        df_photometry[
            [
                "SNID",
                "MJD",
                "FLT",
                "FLUXCAL",
                "FLUXCALERR",
                "PHOTFLAG",
                "photo_detection",
            ]
        ],
        DES_like_trigger,
        on="SNID",
        how="left",
    )
    lu.print_blue("Selected -7<DES-like trigger<20")
    df_DEStrigpho["SNR"] = df_DEStrigpho["FLUXCAL"] / df_DEStrigpho["FLUXCALERR"]
    df_DEStrigpho["window_time_cut"] = True
    mask = df_DEStrigpho["MJD"] != -777.00
    df_DEStrigpho["window_delta_time"] = (
        df_DEStrigpho["MJD"] - df_DEStrigpho["MJD_1stdet"]
    )
    df_DEStrigpho.loc[mask, "window_time_cut"] = df_DEStrigpho[
        "window_delta_time"
    ].apply(lambda x: True if x < 20 and x > -7 else False)
    df_trigphoDES_sel = df_DEStrigpho[
        (df_DEStrigpho["window_time_cut"]) & (df_DEStrigpho.SNID.isin(idxs_presel))
    ]
    nflt = 1
    n_nights = 2
    snr = 5
    SNID_measurements_criteria = sampling_criteria(
        df_trigphoDES_sel, nflt, n_nights, snr
    )
    df_metadata_sel = df_metadata[df_metadata.SNID.isin(SNID_measurements_criteria)]

    df_stats_triggerDES = early_class(
        df_trigphoDES_sel,
        df_metadata_sel,
        photoIa_wz_JLA,
        df_stats,
        path_model,
        "-7<DES-like trigger<20",
    )

    pu.hist_fup_targets_early(
        df_stats_triggerDES,
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
        df_stats_triggerDES,
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

    lu.print_blue("Stats early")
    lu.print_blue("OzDES like")
    cols_to_print = [
        "cut",
        "total maglim<22.7",
        "specIa maglim<22.7",
        "M22 maglim<22.7",
        "nonIa maglim<22.7",
        "multiseason maglim<22.7",
    ]
    lu.print_blue("Peak")
    print(df_stats_peak[cols_to_print].to_latex(index=False))
    lu.print_blue("DES-like trigger")
    print(df_stats_triggerDES[cols_to_print].to_latex(index=False))
    lu.print_blue("LSST-like trigger")
    print(df_stats_triggerLSST[cols_to_print].to_latex(index=False))

    cols_to_print = [
        "cut",
        "detections",
        "detections_std",
        "unights",
        "unights_std",
    ]
    lu.print_blue("Peak")
    print(df_stats_peak[cols_to_print].to_latex(index=False))
    lu.print_blue("DES-like trigger")
    print(df_stats_triggerDES[cols_to_print].to_latex(index=False))
    lu.print_blue("LSST-like trigger")
    print(df_stats_triggerLSST[cols_to_print].to_latex(index=False))

    # Trigger comparisson
    lu.print_blue("Observed peak - LSST-like trigger")
    for deltat in [30, 50]:
        # M22
        tmp = toplot_peak_merged[
            (np.abs(toplot_peak_merged["observed peak - LSST trigger"]) < deltat)
            & (toplot_peak_merged.SNID.isin(photoIa_wz_JLA.SNID.values))
        ]
        perc = (
            len(tmp)
            * 100
            / len(
                toplot_peak_merged[
                    toplot_peak_merged.SNID.isin(photoIa_wz_JLA.SNID.values)
                ]
            )
        )
        print(f"M22 within {deltat} {round(perc,1)} %")
        # spec
        tmp = toplot_peak_merged[
            (np.abs(toplot_peak_merged["observed peak - LSST trigger"]) < deltat)
            & (toplot_peak_merged.SNTYPE.isin(cu.spec_tags["Ia"]))
        ]
        perc = (
            len(tmp)
            * 100
            / len(
                toplot_peak_merged[toplot_peak_merged.SNTYPE.isin(cu.spec_tags["Ia"])]
            )
        )
        print(f"spec Ia within {deltat} {round(perc,1)} %")

    list_spec_sntypes = [
        cu.spec_tags["Ia"],
        cu.spec_tags["nonSN"],
    ]
    list_df_spec = [
        toplot_peak_merged[toplot_peak_merged["SNTYPE"].isin(k)]
        for k in list_spec_sntypes
    ]
    pu.plot_histograms_listdf(
        [toplot_peak_merged[toplot_peak_merged.SNID.isin(photoIa_wz_JLA.SNID.values)]]
        + list_df_spec,
        ["M22", "SN (spectroscopic)", "non SN (spectroscopic)"],
        density=False,
        varx="observed peak - LSST trigger",
        outname=f"{path_plots}/peak-trigger_M22spec.png",
        log_scale=True,
        nbins=30,
    )

    pu.plot_histograms_listdf(
        [toplot_peak_merged] + list_df_spec,
        ["DES-SN"] + ["spec SN", "spec non SN"],
        density=False,
        varx="observed peak - LSST trigger",
        outname=f"{path_plots}/peak-trigger.png",
        log_scale=True,
        nbins=30,
    )
    pu.plot_histograms_listdf(
        list_df_spec,
        ["spec SN", "spec non SN"],
        density=False,
        varx="observed peak - LSST trigger",
        outname=f"{path_plots}/peak-trigger_speconly.png",
        log_scale=True,
        nbins=30,
    )

    pu.plot_scatter_mosaic(
        [toplot_peak_merged] + list_df_spec,
        ["DES-SN"] + ["spec SN", "spec non SN"],
        "PKMJDINI",
        "LSSTtrigger_MJD",
        path_out=f"{path_plots}/scatter_peak_trigger.png",
    )

    fig = plt.figure()
    plt.scatter(
        toplot_peak_merged["PKMJDINI"],
        toplot_peak_merged["LSSTtrigger_MJD"],
    )
    plt.savefig(
        f"{path_plots}/scatter_peak_trigger_onebyone.png",
    )

    print("SNe Ia with SALT2")
    # how about SNe Ia with t0 estimation?
    salt_JLA = du.load_salt_fits(f"{args.path_data}/FITOPT000.FITRES")
    JLA_merged = salt_JLA[["SNID", "PKMJD", "SNTYPE"]].merge(LSSTestimate_trig)
    JLA_merged["t0-trigger"] = JLA_merged["PKMJD"] - JLA_merged["LSSTtrigger_MJD"]
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

    # sadly LSST-like trigger is not a great indicator for SN-like variation

    # Rubin + 4MOST TiDES
    lu.print_blue("Rubin + 4MOST TiDES")
    cols_to_print = [
        "cut",
        "total maglim<23.5",
        "specIa maglim<23.5",
        "M22 maglim<23.5",
        "nonIa maglim<23.5",
        "multiseason maglim<23.5",
    ]
    print("LSST-like trigger")
    print(df_stats_triggerLSST[cols_to_print].to_latex(index=False))
    print("DES-like trigger")
    print(df_stats_triggerDES[cols_to_print].to_latex(index=False))

    import ipdb

    ipdb.set_trace()
