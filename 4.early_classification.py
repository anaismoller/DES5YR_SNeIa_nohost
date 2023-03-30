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
    return pd.merge(dfmetadata, df_unuf)


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
        cut="+ transient status",
    )

    print("TRIGGER")
    # Using PHOTFLAG 4096 (bit mask)
    # as PRIVATE(DES_mjd_trigger) in header may not be acurate
    estimate_trig = pd.read_csv(f"{args.path_data}/trigger_MJD.csv", delimiter=" ")

    # comparisson between estimated peak and trigger time
    salt_peak = du.load_salt_fits(
        f"{args.path_data}/DESALL_forcePhoto_real_snana_fits.SNANA.TEXT"
    )
    peak_merged = salt_peak[["SNID", "PKMJDINI", "SNTYPE"]].merge(estimate_trig)
    peak_merged["PKMJDINI-trigger"] = (
        peak_merged["PKMJDINI"] - peak_merged["trigger_MJD"]
    )
    toplot_peak_merged = peak_merged[
        peak_merged.PKMJDINI > 1
    ]  # to eliminate non estimates

    # spec
    list_sntypes = [
        cu.spec_tags["Ia"],
        cu.spec_tags["Ia"] + cu.spec_tags["CC"] + cu.spec_tags["SLSN"],
        cu.spec_tags["nonSN"],
    ]
    list_df_spec = [
        toplot_peak_merged[toplot_peak_merged["SNTYPE"].isin(k)] for k in list_sntypes
    ]
    pu.plot_histograms_listdf(
        [toplot_peak_merged] + list_df_spec,
        ["DES-SN"] + ["spec Ia", "spec SN", "spec non-SN"],
        density=False,
        varx="PKMJDINI-trigger",
        outname=f"{path_plots}/peak-trigger.png",
        log_scale=True,
        nbins=30,
    )
    # for my statistics I should evaluate only spec Ia where trigger is close to the SNIa
    # sanity check: verify PKMJDINI close to fitted t0 for SNe Ia
    salt_zspe = du.load_salt_fits(
        f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_JLA_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz"
    )
    salt_zspe = salt_zspe.rename(columns={"PKMJDINI": "PKMJDINI_salt"})
    peak_merged = peak_merged.merge(salt_zspe)
    peak_merged["PKMJDINI_salt-trigger"] = (
        peak_merged["PKMJDINI_salt"] - peak_merged["trigger_MJD"]
    )
    fig = plt.figure()
    plt.hist(
        peak_merged[peak_merged.SNTYPE.isin(cu.spec_tags["Ia"])][
            "PKMJDINI_salt-trigger"
        ],
        bins=20,
    )
    plt.yscale("log")
    plt.xlabel("peak_salt-trigger")
    plt.savefig(f"{path_plots}/peak_salt-peak_specIa.png")
    plt.clf()
    # sadly trigger is not a great indicator

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
    for plus_trigger in [7, 14, 30]:
        # +14 which should include maximum
        df_trigpho.loc[mask, "window_time_cut"] = df_trigpho["window_delta_time"].apply(
            lambda x: True if x < plus_trigger and x > -7 else False
        )
        df_trigpho_sel = df_trigpho[
            (df_trigpho["window_time_cut"]) & (df_trigpho.SNID.isin(idxs_presel))
        ]
        lu.print_blue(f"Selected -7<trigger<{plus_trigger}")

        # # select those SNe Ia which trigger ~peak
        # idxs_specIa_good_trigger = peak_merged[
        #     (peak_merged.SNTYPE.isin(cu.spec_tags["Ia"]))
        #     & (peak_merged["PKMJDINI_salt-trigger"] < plus_trigger)
        # ]

        print("SAMPLING TRIGGER")
        nflt = 2
        n_nights = 3
        for snr in [3, 5]:
            SNID_measurements_criteria = sampling_criteria(
                df_trigpho_sel, nflt, n_nights, snr
            )
            df_metadata_sampling_trigger = df_metadata[
                df_metadata.SNID.isin(SNID_measurements_criteria)
            ]

            df_metadata_sampling_trigger_u = get_lc_stats(
                df_trigpho_sel, df_metadata_sampling_trigger
            )

            df_stats = mu.cuts_deep_shallow_eventmag(
                df_metadata_sampling_trigger_u,
                photoIa_wz_JLA,
                df_trigpho_sel,
                df_stats=df_stats,
                cut=f"-7<t<{plus_trigger} nflt:{nflt} nights:{n_nights} snr:{snr}",
            )

            # PREDICTIONS
            # 1. reformat photometry for SuperNNova
            missing_cols = [
                "HOSTGAL_PHOTOZ",
                "HOSTGAL_SPECZ",
                "HOSTGAL_PHOTOZ_ERR",
                "HOSTGAL_SPECZ_ERR",
            ]
            df_snn = df_trigpho_sel.copy()
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
                        "REDSHIFT_FINAL",
                        "PRIVATE(DES_transient_status)",
                        "HOSTGAL_MAG_r",
                    ]
                ],
            )

            df_stats, df_minmag = mu.cuts_deep_shallow_eventmag(
                preds_df[preds_df.prob_class0 > 0.1],
                photoIa_wz_JLA,
                df_trigpho_sel,
                df_stats=df_stats,
                cut=f"-7<t<{plus_trigger} SNN>0.1 snr:{snr}",
                return_extra_df=True,
            )

    print(
        df_stats[
            [
                "cut",
                "total selected",
                "total spec Ia",
                "total photo Ia M22",
                "multiseason",
                "total maglim<22.5",
                "specIa maglim<22.5",
                "M22 maglim<22.5",
                "multiseason maglim<22.5",
            ]
        ]
    )
    # no ses super super y si bajo el SNN>0.1/.2?
    import ipdb

    ipdb.set_trace()
