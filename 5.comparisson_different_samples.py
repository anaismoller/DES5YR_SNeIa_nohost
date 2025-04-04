import numpy as np
import pandas as pd
import os, sys
import matplotlib as mpl
from astropy.table import Table
import matplotlib.pyplot as plt
from supervenn import supervenn
from myutils import data_utils as du
from myutils import plot_utils as pu

DES5yr = os.getenv("DES5yr")
DES = os.getenv("DES")

path_data = f"{DES}/data/DESALL_forcePhoto_real_snana_fits"
path_samples = "./samples"
path_M22 = f"{path_samples}/previous_works/Moller2022_DES5yr_SNeIa_whostz_JLA.csv"
path_M24 = (
    f"{path_samples}/DES_noredshift_Moller2024_w_SALTandhostinfo.csv"  # JLA quality
)
path_hostproperties = f"{DES}/data/hostproperties/DES_All_SNe_specz_photoz_ugrizJHK_mangled_sedfit_pegase_kroupa_20220919.csv"  # Wiseman
path_hostproperties_SNphotoz = f"{DES}/data/hostproperties/mass_SNphotoz/results_deep_SN_hosts_all_fields_ugrizJHK_SALT2REDSHIFT_forsedfit.cat_pegase_chab_mangle.sav.txt"  # Wiseman, Sullivan
path_salt_wzspe = f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_JLA_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz"
path_salt_fittedz = f"{DES}/data/DESALL_forcePhoto_real_snana_fits/D_FITZ_DATA5YR_DENSE_SNR/output/DESALL_forcePhoto_real_snana_fits/FITOPT000.FITRES.gz"

path_dump = f"{DES}/DES5YR/DES5YR_SNeIa_nohost/dump_DES5YR"
path_plots = f"{path_dump}/plots_samples_comparisson/"
os.makedirs(path_plots, exist_ok=True)

lines_ranges = {"zCMB": [0.2, 1.2], "c": [-0.3, 0.3], "x1": [-3, 3], "mass": [8, 12]}


def cuts(df, var):
    sel = (
        df[df["HOSTGAL_MAG_r to plot"] < 30]
        if var == "HOSTGAL_MAG_r to plot"
        else (
            df[df["mass to plot"] > 7]
            if var == "mass to plot"
            else (df[df["HOST_DDLR to plot"] > 0] if var == "HOST_DDLR to plot" else df)
        )
    )
    return sel


def preprocess(spec_Ia, photoIa_wz_JLA, photoIa_nz_JLA, varbin, varbinreduced):
    # binning
    list_df = []
    for df in [
        spec_Ia,
        photoIa_wz_JLA,
        photoIa_nz_JLA,
    ]:
        sel = cuts(df, varbin)
        sel[f"{varbin}_bin"] = pd.cut(
            sel.loc[:, (f"{varbinreduced} to plot")], pu.bins_dic[f"{varbin}"]
        )
        list_df.append(sel)
    return list_df


if __name__ == "__main__":
    """Comparing DES samples"""

    #
    # Load data
    #
    DES_all = du.load_headers(path_data)
    # spec
    spec_Ia = DES_all[DES_all.SNTYPE.isin([1, 101])]
    # M22, this work
    photoIa_wz_JLA = pd.read_csv(path_M22, comment="#")
    photoIa_nz_JLA = pd.read_csv(path_M24, comment="#")

    # Load fits
    # SALT fit using spectroscopic redshift
    salt_wzspe_ori = du.load_salt_fits(path_salt_wzspe)
    tmp = salt_wzspe_ori.add_suffix("_zspe")
    salt_wzspe = tmp.rename(columns={"SNID_zspe": "SNID"})
    # fitted redshift with photoz as prior if exists
    salt_fits_noz_ori = du.load_salt_fits(path_salt_fittedz)
    tmp = salt_fits_noz_ori.add_suffix("_fittedz")
    salt_fits_noz = tmp.rename(columns={"SNID_fittedz": "SNID"})

    # Host-galaxy properties
    # with host zspec
    host_props_zspe = pd.read_csv(
        path_hostproperties, skipinitialspace=True, delimiter=" ", comment="#"
    )
    host_props_zspe = host_props_zspe[
        [
            k
            for k in host_props_zspe.keys()
            if k not in ["RA", "DEC", "CCDNUM", "NDOF", "FIELD"]
        ]
    ]
    # fig = plt.figure(figsize=(10, 5))
    # gs = fig.add_gridspec(1, 3)
    # axs = gs.subplots(sharex=False, sharey=True)
    # for i, var in enumerate(["sfr", "mass", "PHOTOZ"]):
    #     axs[i].hist(host_props_zspe[var])
    #     axs[i].set_xlabel(var)
    # plt.yscale("log")
    # plt.savefig(f"{path_plots}/host_props_zspe_inspection.png")

    # with SNphotz
    host_props_SNphotoz = Table.read(
        path_hostproperties_SNphotoz, format="ascii"
    ).to_pandas()
    host_props_SNphotoz = host_props_SNphotoz.rename(
        columns={
            "col1": "SNID",
            "col2": "specz_SNphotoz",
            "col3": "chi2_SNphotoz",
            "col4": "NDOF_SNphotoz",
            "col5": "zmax_SNphotoz",
            "col6": "mass_SNphotoz",
            "col7": "mass_lowerr_SNphotoz",
            "col8": "mass_upperr_SNphotoz",
            "col9": "massmc_SNphotoz",
            "col10": "massmc_err_SNphotoz",
            "col11": "sfr_SNphotoz",
            "col12": "sfr_lowerr_SNphotoz",
            "col13": "sfr_upperr_SNphotoz",
            "col14": "sfrmc_SNphotoz",
            "col15": "sfrmc_err_SNphotoz",
            "col16": "u-r_SNphotoz",
            "col17": "u-r_err_SNphotoz",
            "col23": "eb-v_SNphotoz",
            "col24": "Umag_SNphotoz",
            "col25": "Bmag_SNphotoz",
            "col26": "Vmag_SNphotoz",
            "col27": "Rmag_SNphotoz",
        }
    )

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 3)
    axs = gs.subplots(sharex=False, sharey=True)
    for i, var in enumerate(["sfr_SNphotoz", "mass_SNphotoz", "specz_SNphotoz"]):
        axs[i].hist(host_props_SNphotoz[var])
        axs[i].set_xlabel(var)
    plt.yscale("log")
    plt.savefig(f"{path_plots}/host_props_SNphotoz_inspection.png")

    #
    # Enrich data
    #
    spec_Ia = pd.merge(spec_Ia, host_props_zspe, how="left")
    spec_Ia = pd.merge(spec_Ia, host_props_SNphotoz, how="left")
    spec_Ia = pd.merge(spec_Ia, salt_wzspe, how="left")
    spec_Ia = pd.merge(spec_Ia, salt_fits_noz, how="left")

    photoIa_wz_JLA = pd.merge(photoIa_wz_JLA, host_props_zspe, how="left")
    photoIa_wz_JLA = pd.merge(photoIa_wz_JLA, host_props_SNphotoz, how="left")
    photoIa_wz_JLA = pd.merge(photoIa_wz_JLA, salt_wzspe, how="left")
    photoIa_wz_JLA = pd.merge(photoIa_wz_JLA, salt_fits_noz, how="left")

    photoIa_nz_JLA = pd.merge(photoIa_nz_JLA, host_props_zspe, how="left")
    photoIa_nz_JLA = pd.merge(
        photoIa_nz_JLA, host_props_SNphotoz, how="left", on="SNID"
    )
    photoIa_nz_JLA = pd.merge(photoIa_nz_JLA, salt_wzspe, how="left")
    photoIa_nz_JLA = pd.merge(photoIa_nz_JLA, salt_fits_noz, how="left")

    #
    # hack to plot the same variables from different sources
    #
    photoIa_nz_JLA["z to plot"] = photoIa_nz_JLA["zHD_fittedz"]
    photoIa_nz_JLA["z to plot err"] = photoIa_nz_JLA["zHDERR_fittedz"]
    photoIa_wz_JLA["z to plot"] = photoIa_wz_JLA["zHD"]
    photoIa_wz_JLA["z to plot err"] = photoIa_wz_JLA["zHDERR"]
    spec_Ia["z to plot"] = spec_Ia["zHD_zspe"]
    spec_Ia["z to plot err"] = spec_Ia["zHDERR_zspe"]
    photoIa_nz_JLA["c to plot"] = photoIa_nz_JLA["c_fittedz"]
    photoIa_wz_JLA["c to plot"] = photoIa_wz_JLA["c"]
    spec_Ia["c to plot"] = spec_Ia["c_zspe"]
    photoIa_nz_JLA["x1 to plot"] = photoIa_nz_JLA["x1_fittedz"]
    photoIa_nz_JLA["x1 to plot err"] = photoIa_nz_JLA["x1ERR_fittedz"]
    photoIa_wz_JLA["x1 to plot"] = photoIa_wz_JLA["x1"]
    photoIa_wz_JLA["x1 to plot err"] = photoIa_wz_JLA["x1ERR"]
    spec_Ia["x1 to plot"] = spec_Ia["x1_zspe"]
    spec_Ia["x1 to plot err"] = spec_Ia["x1ERR_zspe"]
    photoIa_nz_JLA["HOSTGAL_MAG_r to plot"] = photoIa_nz_JLA["HOSTGAL_MAG_r"]
    photoIa_wz_JLA["HOSTGAL_MAG_r to plot"] = photoIa_wz_JLA["HOSTGAL_MAG_r"]
    spec_Ia["HOSTGAL_MAG_r to plot"] = spec_Ia["HOSTGAL_MAG_r"]
    photoIa_nz_JLA["mass to plot"] = photoIa_nz_JLA["mass_SNphotoz"]
    photoIa_wz_JLA["mass to plot"] = photoIa_wz_JLA["mass"]
    spec_Ia["mass to plot"] = spec_Ia["mass"]

    photoIa_nz_JLA["mass lowerr to plot"] = photoIa_nz_JLA["mass_lowerr_SNphotoz"]
    photoIa_wz_JLA["mass lowerr to plot"] = photoIa_wz_JLA["mass_lowerr"]
    spec_Ia["mass lowerr to plot"] = spec_Ia["mass_lowerr"]

    photoIa_nz_JLA["mass upperr to plot"] = photoIa_nz_JLA["mass_upperr_SNphotoz"]
    photoIa_wz_JLA["mass upperr to plot"] = photoIa_wz_JLA["mass_upperr"]
    spec_Ia["mass upperr to plot"] = spec_Ia["mass_upperr"]

    photoIa_nz_JLA["HOST_DDLR to plot"] = photoIa_nz_JLA["HOST_DDLR_fittedz"]
    photoIa_wz_JLA["HOST_DDLR to plot"] = photoIa_wz_JLA["HOST_DDLR_zspe"]
    spec_Ia["HOST_DDLR to plot"] = spec_Ia["HOST_DDLR_zspe"]

    #
    # properties vs zHD
    #
    bins = pu.bins_dic["zHD"]
    mean_bins = bins[:-1] + (bins[1] - bins[0]) / 2
    fig = plt.figure(figsize=(20, 30), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    for i, var in enumerate(
        [
            "x1 to plot",
            "c to plot",
            "mass to plot",
            "HOSTGAL_MAG_r to plot",
        ]
    ):
        list_df = preprocess(
            cuts(spec_Ia, var),
            cuts(photoIa_wz_JLA, var),
            cuts(photoIa_nz_JLA, var),
            "zHD",
            "z",
        )
        # all other
        axs[i] = pu.plot_errorbar_binned(
            list_df,
            [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
                "DES SNe Ia HQ (SNphoto z)",
            ],
            axs=axs[i],
            binname="zHD_bin",
            varx="z to plot",
            vary=var,
            data_color_override=True,
            color_list=[
                pu.SAMPLES_COLORS[k]
                for k in [
                    "DES SNe Ia spectroscopic",
                    "DES SNe Ia M22",
                    "DES SNe Ia HQ (SNphoto z)",
                ]
            ],
            plot_lines=True,
            bins=mean_bins,
        )
        # subsample with zspe (using those fits)
        tmp = photoIa_nz_JLA.copy()
        tmp["mass to plot"] = tmp["mass"]
        tmp["c to plot"] = tmp["c_zspe"]
        tmp["x1 to plot"] = tmp["x1_zspe"]
        tmp["HOSTGAL_MAG_r to plot"] = tmp[~tmp.c_zspe.isna()]["HOSTGAL_MAG_r to plot"]
        tmp = cuts(tmp, var)
        tmp["zHD_bin_zspe"] = pd.cut(tmp.loc[:, ("zHD")], pu.bins_dic["zHD"])
        axs[i].errorbar(
            mean_bins,
            tmp.groupby("zHD_bin_zspe").mean()[var].values,
            yerr=tmp.groupby("zHD_bin_zspe").std()[var].values
            / np.sqrt(tmp.groupby("zHD_bin_zspe")[var].count()).values,
            color=pu.SAMPLES_COLORS["DES SNe Ia HQ (SNphoto z)"],
            zorder=-100,
            linestyle="dotted",
            label="DES SNe Ia HQ (host zspec)",
            linewidth=5,
        )
        ylabel = (
            "host r magnitude"
            if var == "HOSTGAL_MAG_r to plot"
            else (
                r"host stellar mass (log($M_{*}$/$M_{\odot}$))"
                if var == "mass to plot"
                else var.replace("to plot", "")
            )
        )
        axs[i].set_ylabel(ylabel, fontsize=36)
        # We change the fontsize of minor ticks label
        axs[i].tick_params(axis="both", which="major", labelsize=32)
        axs[i].tick_params(axis="both", which="minor", labelsize=32)
    axs[i].legend(fontsize=32)
    axs[i].set_xlabel("z", fontsize=37)
    plt.savefig(f"{path_plots}/2ddist_all_sample_zHD.png")

    # histo mass per bin redshift
    # answering referee's comment on why do we still have high mass SNe Ia
    for zbin in pd.arrays.IntervalArray(spec_Ia["zHD_bin"]):
        fig = plt.figure()
        sel = photoIa_nz_JLA[photoIa_nz_JLA["zHD_bin"] == zbin]
        n, bins, _ = plt.hist(
            sel["mass to plot"], label="M24", histtype="step", linewidth=2
        )
        sel = spec_Ia[spec_Ia["zHD_bin"] == zbin]
        plt.hist(sel["mass to plot"], label="spec Ia", bins=bins, histtype="step")
        sel = photoIa_wz_JLA[photoIa_wz_JLA["zHD_bin"] == zbin]
        plt.hist(sel["mass to plot"], label="M22", bins=bins, histtype="step")
        plt.legend(loc=0)
        plt.xlabel(r"host stellar mass (log($M_{*}$/$M_{\odot}$))")
        plt.savefig(f"{path_plots}/mass_zbin_{zbin}.png")
    #
    # x1 vs mass
    #
    bins = pu.bins_dic["mass"]
    mean_bins = bins[:-1] + (bins[1] - bins[0]) / 2
    fig = plt.figure(figsize=(20, 20), constrained_layout=True)
    var = "x1 to plot"
    list_df = preprocess(spec_Ia, photoIa_wz_JLA, photoIa_nz_JLA, "mass", "mass")
    # all other
    pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia spectroscopic",
            "DES SNe Ia M22",
            "DES SNe Ia HQ (SNphoto z)",
        ],
        binname="mass_bin",
        varx="mass to plot",
        vary=var,
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS[k]
            for k in [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
                "DES SNe Ia HQ (SNphoto z)",
            ]
        ],
        plot_lines=True,
        bins=mean_bins,
    )
    # scatter bellow
    plt.errorbar(
        cuts(photoIa_nz_JLA, "mass to plot")["mass to plot"],
        cuts(photoIa_nz_JLA, "mass to plot")["x1 to plot"],
        yerr=cuts(photoIa_nz_JLA, "mass to plot")["x1 to plot err"],
        xerr=np.array(
            list(
                zip(
                    cuts(photoIa_nz_JLA, "mass to plot")["mass_lowerr_SNphotoz"],
                    cuts(photoIa_nz_JLA, "mass to plot")["mass_upperr_SNphotoz"],
                )
            )
        ).T,
        zorder=-1000,
        color="grey",
        elinewidth=1,
        fmt="o",
        markersize=1,
        alpha=0.3,
    )
    # mass step
    tmp_ms = cuts(photoIa_nz_JLA, "mass to plot")
    tmp_ms_low_x1_zfree = tmp_ms[tmp_ms.mass < 10]["x1 to plot"].median()
    tmp_ms_high_x1_zfree = tmp_ms[tmp_ms.mass > 10]["x1 to plot"].median()
    tmp_ms_low_mass_zfree = tmp_ms[tmp_ms.mass < 10]["mass to plot"].median()
    tmp_ms_high_mass_zfree = tmp_ms[tmp_ms.mass > 10]["mass to plot"].median()
    plt.scatter(
        tmp_ms_low_mass_zfree,
        tmp_ms_low_x1_zfree,
        s=500,
        marker="X",
        c="green",
        zorder=10000,
    )
    plt.scatter(
        tmp_ms_high_mass_zfree,
        tmp_ms_high_x1_zfree,
        s=500,
        marker="X",
        c="green",
        zorder=10000,
    )
    plt.plot(
        [10, 10],
        [-3, 3],
        color="black",
        linewidth=1,
        linestyle="--",
        zorder=100,
    )

    ylabel = var.replace("to plot", "")
    plt.ylim(-2.5, 2.5)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)
    plt.legend(fontsize=15, loc=1)
    plt.xlabel(r"host stellar mass (log($M_{*}$/$M_{\odot}$))", fontsize=20)
    plt.savefig(f"{path_plots}/2ddist_all_sample_mass.png")

    #
    # mosaic x1 vs mass
    #
    var = "x1 to plot"
    fig = plt.figure(figsize=(20, 30), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    list_df = preprocess(spec_Ia, photoIa_wz_JLA, photoIa_nz_JLA, "mass", "mass")
    # with SNphoto-z
    axs[0] = pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia spectroscopic",
            "DES SNe Ia M22",
            "DES SNe Ia HQ (SNphoto z)",
        ],
        binname="mass_bin",
        varx="mass to plot",
        vary=var,
        axs=axs[0],
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS[k]
            for k in [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
                "DES SNe Ia HQ (SNphoto z)",
            ]
        ],
        plot_lines=True,
        bins=mean_bins,
    )
    # scatter bellow
    axs[0].errorbar(
        cuts(photoIa_nz_JLA, "mass to plot")["mass to plot"],
        cuts(photoIa_nz_JLA, "mass to plot")["x1 to plot"],
        yerr=cuts(photoIa_nz_JLA, "mass to plot")["x1 to plot err"],
        xerr=np.array(
            list(
                zip(
                    cuts(photoIa_nz_JLA, "mass to plot")["mass_lowerr_SNphotoz"],
                    cuts(photoIa_nz_JLA, "mass to plot")["mass_upperr_SNphotoz"],
                )
            )
        ).T,
        zorder=-1000,
        color="grey",
        elinewidth=1,
        fmt="o",
        markersize=1,
        alpha=0.3,
    )
    axs[0].legend(fontsize=28, title_fontsize=25, loc=1)
    axs[0].set_ylim(-2.5, 2.5)
    # mass step
    tmp_ms = cuts(photoIa_nz_JLA, "mass to plot")
    tmp_ms_low_x1_zfree = tmp_ms[tmp_ms.mass < 10]["x1 to plot"].median()
    tmp_ms_high_x1_zfree = tmp_ms[tmp_ms.mass > 10]["x1 to plot"].median()
    tmp_ms_low_mass_zfree = tmp_ms[tmp_ms.mass < 10]["mass to plot"].median()
    tmp_ms_high_mass_zfree = tmp_ms[tmp_ms.mass > 10]["mass to plot"].median()
    axs[0].plot(
        [10, 10],
        [-3, 3],
        color="black",
        linewidth=1,
        linestyle="--",
        zorder=100,
    )

    # with zspe if available
    tmp = photoIa_nz_JLA.copy()
    tmp = cuts(tmp, "mass to plot")
    tmp["mass to plot"] = tmp["mass"]
    tmp["c to plot"] = tmp["c_zspe"]
    tmp["x1 to plot"] = tmp["x1_zspe"]
    tmp["mass_bin_zspe"] = pd.cut(tmp.loc[:, ("mass to plot")], pu.bins_dic["mass"])
    list_df = preprocess(spec_Ia, photoIa_wz_JLA, tmp, "mass", "mass")
    list_df = list_df[:-1]
    axs[1] = pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia spectroscopic",
            "DES SNe Ia M22",
            # "DES SNe Ia HQ (host zspec)",
        ],
        binname="mass_bin",
        varx="mass to plot",
        axs=axs[1],
        vary=var,
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS[k]
            for k in [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
                # "DES SNe Ia HQ (SNphoto z)",
            ]
        ],
        plot_lines=True,
        bins=mean_bins,
    )
    # mixedasides
    df = tmp.copy()
    x_toplot = df.groupby("mass_bin").mean()["mass to plot"].values
    axs[1].errorbar(
        x_toplot,
        df.groupby("mass_bin").mean()[var].values,
        yerr=df.groupby("mass_bin").std()[var].values
        / np.sqrt(df.groupby("mass_bin")[var].count()).values,
        fmt="o",
        color=pu.SAMPLES_COLORS["DES SNe Ia HQ (SNphoto z)"],
        linestyle="dotted",
        zorder=100,
        label="DES SNe Ia HQ (host zspec)",
        linewidth=5,
    )

    # scatter bellow
    axs[1].errorbar(
        cuts(tmp, "mass to plot")["mass to plot"],
        cuts(tmp, "mass to plot")["x1 to plot"],
        yerr=cuts(tmp, "mass to plot")["x1 to plot err"],
        xerr=np.array(
            list(
                zip(
                    cuts(tmp, "mass to plot")["mass_lowerr"],
                    cuts(tmp, "mass to plot")["mass_upperr"],
                )
            )
        ).T,
        zorder=-1000,
        color="grey",
        elinewidth=1,
        fmt="o",
        markersize=1,
        alpha=0.3,
    )
    axs[1].legend(fontsize=28, title_fontsize=25, loc=1)
    axs[1].set_ylim(-2.5, 2.5)
    # mass step
    tmp_ms = cuts(tmp, "mass to plot")
    tmp_ms_low_x1_zfree_zspe = tmp_ms[tmp_ms.mass < 10]["x1 to plot"].median()
    tmp_ms_high_x1_zfree_zspe = tmp_ms[tmp_ms.mass > 10]["x1 to plot"].median()
    tmp_ms_low_mass_zfree_zspe = tmp_ms[tmp_ms.mass < 10]["mass to plot"].median()
    tmp_ms_high_mass_zfree_zspe = tmp_ms[tmp_ms.mass > 10]["mass to plot"].median()

    # a mix of the samples using best z
    M24_w_zspe = cuts(tmp[~tmp.SNID.isin(spec_Ia.SNID.values)], "mass to plot")
    M22_missing = cuts(
        photoIa_wz_JLA[~photoIa_wz_JLA.SNID.isin(M24_w_zspe.SNID.values)],
        "mass to plot",
    )
    M24_SNphotoz_missing = cuts(
        photoIa_nz_JLA[~photoIa_nz_JLA.SNID.isin(M24_w_zspe.SNID.values)],
        "mass to plot",
    )
    mixed_sample = pd.concat([M24_w_zspe, M24_SNphotoz_missing])
    print(
        f"Mixed sample {len(M24_w_zspe)} with zspe, {len(M24_SNphotoz_missing)} with SNphotoz"
    )

    list_df = preprocess(spec_Ia, photoIa_wz_JLA, mixed_sample, "mass", "mass")
    list_df = list_df[:-1]
    axs[2] = pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia spectroscopic",
            "DES SNe Ia M22",
        ],
        binname="mass_bin",
        varx="mass to plot",
        axs=axs[2],
        vary=var,
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS[k]
            for k in [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
            ]
        ],
        plot_lines=True,
        bins=mean_bins,
    )
    # mixedasides
    df = mixed_sample.copy()
    x_toplot = df.groupby("mass_bin").mean()["mass to plot"].values
    axs[2].errorbar(
        x_toplot,
        df.groupby("mass_bin").mean()[var].values,
        yerr=df.groupby("mass_bin").std()[var].values
        / np.sqrt(df.groupby("mass_bin")[var].count()).values,
        fmt="o",
        color=pu.SAMPLES_COLORS["DES SNe Ia HQ (SNphoto z)"],
        linestyle="dashed",
        zorder=100,
        label="DES SNe Ia HQ (host zspec or SNphoto z)",
        linewidth=5,
    )

    # # scatter bellow
    axs[2].errorbar(
        mixed_sample[mixed_sample.mass > 7]["mass"],
        mixed_sample[mixed_sample.mass > 7]["x1 to plot"],
        yerr=mixed_sample[mixed_sample.mass > 7]["x1 to plot err"],
        xerr=np.array(
            list(
                zip(
                    mixed_sample[mixed_sample.mass > 7]["mass lowerr to plot"],
                    mixed_sample[mixed_sample.mass > 7]["mass upperr to plot"],
                )
            )
        ).T,
        zorder=-1000,
        color="grey",
        elinewidth=1,
        fmt="o",
        markersize=1,
        alpha=0.3,
    )
    # mass step
    tmp_ms = cuts(mixed_sample, "mass to plot")
    tmp_ms_low_x1_mixed = tmp_ms[tmp_ms.mass < 10]["x1 to plot"].median()
    tmp_ms_high_x1_mixed = tmp_ms[tmp_ms.mass > 10]["x1 to plot"].median()
    tmp_ms_low_mass_mixed = tmp_ms[tmp_ms.mass < 10]["mass to plot"].median()
    tmp_ms_high_mass_mixed = tmp_ms[tmp_ms.mass > 10]["mass to plot"].median()
    axs[2].legend(fontsize=28, title_fontsize=25, loc=1)
    axs[2].set_ylim(-2.5, 2.5)
    ylabel = var.replace("to plot", "")
    axs[0].set_ylabel(ylabel, fontsize=38)
    axs[1].set_ylabel(ylabel, fontsize=38)
    axs[2].set_ylabel(ylabel, fontsize=38)
    # We change the fontsize of minor ticks label
    axs[0].tick_params(axis="both", which="major", labelsize=32)
    axs[0].tick_params(axis="both", which="minor", labelsize=32)
    axs[1].tick_params(axis="both", which="major", labelsize=32)
    axs[1].tick_params(axis="both", which="minor", labelsize=32)
    axs[2].tick_params(axis="both", which="major", labelsize=32)
    axs[2].tick_params(axis="both", which="minor", labelsize=32)
    plt.xlim(7, 12)
    plt.xlabel(r"host stellar mass (log($M_{*}$/$M_{\odot}$))", fontsize=38)

    plt.savefig(f"{path_plots}/2ddist_all_sample_mass_mixedsample.png")

    #
    # mosaic x1 vs mass
    #
    var = "x1 to plot"
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    list_df = preprocess(photoIa_nz_JLA, M24_w_zspe, mixed_sample, "mass", "mass")
    pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia HQ (SNphoto z)",
            "DES SNe Ia HQ (with host zspec)",
            "DES SNe Ia HQ (with host zspec or SNphoto z)",
        ],
        binname="mass_bin",
        varx="mass to plot",
        vary=var,
        data_color_override=True,
        color_list=[pu.SAMPLES_COLORS["DES SNe Ia HQ (SNphoto z)"]]
        + ["green", "aquamarine"],
        plot_lines=True,
        bins=mean_bins,
    )
    plt.plot(
        [10, 10],
        [-3, 3],
        color="black",
        linewidth=1,
        linestyle="--",
        zorder=100,
    )

    ylabel = var.replace("to plot", "")
    plt.xlabel(r"host stellar mass (log($M_{*}$/$M_{\odot}$))")
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim(-2, 2)
    # We change the fontsize of minor ticks label
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)
    plt.legend()
    plt.savefig(f"{path_plots}/2ddist_all_sample_mass_mixedsample_single.png")

    #
    # mosaic redshift vs mass
    #
    var = "z to plot"
    fig = plt.figure(figsize=(20, 20), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    list_df = preprocess(spec_Ia, photoIa_wz_JLA, photoIa_nz_JLA, "mass", "mass")
    # with SNphoto-z
    axs[0] = pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia spectroscopic",
            "DES SNe Ia M22",
            "DES SNe Ia HQ (SNphoto z)",
        ],
        binname="mass_bin",
        varx="mass to plot",
        vary=var,
        axs=axs[0],
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS[k]
            for k in [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
                "DES SNe Ia HQ (SNphoto z)",
            ]
        ],
        plot_lines=True,
        bins=mean_bins,
    )
    # scatter bellow
    axs[0].errorbar(
        cuts(photoIa_nz_JLA, "mass to plot")["mass to plot"],
        cuts(photoIa_nz_JLA, "mass to plot")[var],
        yerr=cuts(photoIa_nz_JLA, "mass to plot")[f"{var} err"],
        zorder=-1000,
        color="grey",
        elinewidth=1,
        fmt="o",
        markersize=1,
        alpha=0.3,
    )
    axs[0].legend(fontsize=15, loc=1, title="DES SNe Ia HQ (SNphoto z)")

    # with zspe if available
    tmp = photoIa_nz_JLA.copy()
    tmp = cuts(tmp, "mass to plot")
    tmp["mass to plot"] = tmp["mass"]
    tmp["c to plot"] = tmp["c_zspe"]
    tmp["x1 to plot"] = tmp["x1_zspe"]
    tmp["z to plot"] = tmp["zHD_zspe"]
    tmp["mass_bin_zspe"] = pd.cut(tmp.loc[:, ("mass to plot")], pu.bins_dic["mass"])
    list_df = preprocess(spec_Ia, photoIa_wz_JLA, tmp, "mass", "mass")
    axs[1] = pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia spectroscopic",
            "DES SNe Ia M22",
            "DES SNe Ia HQ (zspe)",
        ],
        binname="mass_bin",
        varx="mass to plot",
        axs=axs[1],
        vary=var,
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS[k]
            for k in [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
                "DES SNe Ia HQ (SNphoto z)",
            ]
        ],
        plot_lines=True,
        bins=mean_bins,
    )
    # scatter bellow
    axs[1].errorbar(
        cuts(tmp, "mass to plot")["mass to plot"],
        cuts(tmp, "mass to plot")[var],
        yerr=cuts(tmp, "mass to plot")[f"{var} err"],
        zorder=-1000,
        color="grey",
        elinewidth=1,
        fmt="o",
        markersize=1,
        alpha=0.3,
    )
    axs[1].legend(fontsize=15, loc=1, title="DES SNe Ia HQ (with host zspec)")
    # a mix of the samples using best z
    list_df = preprocess(spec_Ia, photoIa_wz_JLA, mixed_sample, "mass", "mass")
    axs[2] = pu.plot_errorbar_binned(
        list_df,
        [
            "DES SNe Ia spectroscopic",
            "DES SNe Ia M22",
            "All samples combined (mixed z)",
        ],
        binname="mass_bin",
        varx="mass to plot",
        axs=axs[2],
        vary=var,
        data_color_override=True,
        color_list=[
            pu.SAMPLES_COLORS[k]
            for k in [
                "DES SNe Ia spectroscopic",
                "DES SNe Ia M22",
                "DES SNe Ia HQ (SNphoto z)",
            ]
        ],
        plot_lines=True,
        bins=mean_bins,
    )
    # # scatter bellow
    axs[2].errorbar(
        mixed_sample[mixed_sample.mass > 7]["mass"],
        mixed_sample[mixed_sample.mass > 7][var],
        yerr=mixed_sample[mixed_sample.mass > 7][f"{var} err"],
        zorder=-1000,
        color="grey",
        elinewidth=1,
        fmt="o",
        markersize=1,
        alpha=0.3,
    )

    ylabel = var.replace("to plot", "")
    # axs = axs[1].set_ylabel(ylabel, fontsize=20)
    # We change the fontsize of minor ticks label
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)
    plt.legend(fontsize=15, loc=1, title="Mixed sample")
    # axs = axs[-1].set_xlabel("mass", fontsize=20)
    plt.savefig(f"{path_plots}/2ddist_all_sample_z_mass_mixedsample.png")

    #
    # comparing mass steps
    #
    fig = plt.figure(figsize=(8, 6))
    # SNphotoz
    plt.scatter(
        tmp_ms_low_mass_zfree,
        tmp_ms_low_x1_zfree,
        s=500,
        marker="X",
        c=pu.SAMPLES_COLORS["DES SNe Ia HQ (SNphoto z)"],
    )
    plt.scatter(
        tmp_ms_high_mass_zfree,
        tmp_ms_high_x1_zfree,
        s=500,
        marker="X",
        c=pu.SAMPLES_COLORS["DES SNe Ia HQ (SNphoto z)"],
        label="DES SNe Ia HQ (SNphoto z)",
    )
    # SNphotoz zspe
    plt.scatter(
        tmp_ms_low_mass_zfree_zspe,
        tmp_ms_low_x1_zfree_zspe,
        s=500,
        marker="X",
        c="green",
    )
    plt.scatter(
        tmp_ms_high_mass_zfree_zspe,
        tmp_ms_high_x1_zfree_zspe,
        s=500,
        marker="X",
        c="green",
        label="DES SNe Ia HQ (subsample zspe)",
    )

    # mixed
    plt.scatter(
        tmp_ms_low_mass_mixed,
        tmp_ms_low_x1_mixed,
        s=500,
        marker="X",
        c="aquamarine",
    )
    plt.scatter(
        tmp_ms_high_mass_mixed,
        tmp_ms_high_x1_mixed,
        s=500,
        marker="X",
        c="aquamarine",
        label="Mixed sample",
    )
    plt.legend()
    plt.ylim(-0.4, 0.4)
    plt.ylabel("x1")
    plt.xlabel(r"host stellar mass (log($M_{*}$/$M_{\odot}$))")
    plt.savefig(f"{path_plots}/mass_step_comparisson.png")

    # extra

    def plot_mosaic_scatter_SNphoto_zspe(df, outname, path_plots, plot_ranges):

        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 3, wspace=0.4, hspace=0.1, height_ratios=[1, 0.4])
        axs = gs.subplots(sharex="col", sharey=False)

        for i, var in enumerate(["zCMB", "c", "x1"]):
            # first row: scatter
            axs[0][i].errorbar(
                df[f"{var}_zspe"],
                df[f"{var}_fittedz"],
                xerr=df[f"{var}ERR_zspe"],
                yerr=df[f"{var}ERR_fittedz"],
                fmt="o",
                alpha=0.1,
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
                r"${%s}_{\mathrm{SNphoto ~ z}}$" % var if var != "zCMB" else "SNphoto z"
            )
            axs[0][i].set_ylabel(ylabel)
            axs[0][i].set_xlim(plot_ranges[var])
            axs[0][i].set_ylim(plot_ranges[var])
            if plot_ranges["zCMB"][-1] > 1.2:
                axs[0][i].plot(
                    lines_ranges[var],
                    [lines_ranges[var][0], lines_ranges[var][0]],
                    color="grey",
                    linewidth=1,
                    linestyle="--",
                    zorder=100,
                )
                axs[0][i].plot(
                    lines_ranges[var],
                    [lines_ranges[var][1], lines_ranges[var][1]],
                    color="grey",
                    linewidth=1,
                    linestyle="--",
                    zorder=100,
                )
                axs[0][i].plot(
                    [lines_ranges[var][0], lines_ranges[var][0]],
                    lines_ranges[var],
                    color="grey",
                    linewidth=1,
                    linestyle="--",
                    zorder=100,
                )
                axs[0][i].plot(
                    [lines_ranges[var][1], lines_ranges[var][1]],
                    lines_ranges[var],
                    color="grey",
                    linewidth=1,
                    linestyle="--",
                    zorder=100,
                )
            # 2nd row: delta
            df["delta {var}"] = df[f"{var}_zspe"] - df[f"{var}_fittedz"]
            axs[1][i].errorbar(
                df[f"{var}_zspe"],
                df[f"{var}_zspe"] - df[f"{var}_fittedz"],
                fmt="o",
                alpha=0.1,
                color="indigo",
            )
            axs[1][i].plot(
                [lines_ranges[var][0], lines_ranges[var][1]],
                [0, 0],
                color="black",
                linewidth=1,
                linestyle="--",
                zorder=100,
            )
            # axs[1][i].set_ylim([-0.5, 0.5] if var != "x1" else [-2.5, 2.5])

            axs[1][i].set_xlabel(
                r"${%s}_{\mathrm{zspe}}$" % var if var != "zCMB" else "host zspec"
            )
            ylabel = r"$\Delta {%s}$" % var if var != "zCMB" else r"$\Delta z$"
            axs[1][i].set_ylabel(ylabel)
        plt.savefig(f"{path_plots}/{outname}.png")

    # M22 in M24
    M22_in_M24 = photoIa_wz_JLA[
        (photoIa_wz_JLA["SNID"].isin(photoIa_nz_JLA.SNID.values))
    ]
    # M22 not in M24
    M22_notin_M24 = photoIa_wz_JLA[
        (~photoIa_wz_JLA["SNID"].isin(photoIa_nz_JLA.SNID.values))
    ]
    M22_notinM24_wSNphotoz = M22_notin_M24[~M22_notin_M24["zCMB_fittedz"].isna()]
    print(
        f"M22 not in M24 {len(M22_notin_M24)} with a SNphoto z {len(M22_notinM24_wSNphotoz)}"
    )
    plot_ranges = {"zCMB": [0.1, 1.4], "c": [-0.4, 0.4], "x1": [-4, 4]}
    plot_mosaic_scatter_SNphoto_zspe(
        M22_notinM24_wSNphotoz, "M22_notinM24_wSNphotoz", path_plots, plot_ranges
    )
    plot_ranges_JLA = {"zCMB": [0.2, 1.2], "c": [-0.3, 0.3], "x1": [-3, 3]}
    plot_mosaic_scatter_SNphoto_zspe(
        M22_in_M24, "M22_in_M24", path_plots, plot_ranges_JLA
    )

    # mass
    fig = plt.figure()
    plt.errorbar(
        M24_w_zspe["mass_SNphotoz"],
        M24_w_zspe["mass"],
        xerr=np.array(
            list(
                zip(
                    M24_w_zspe["mass_lowerr_SNphotoz"],
                    M24_w_zspe["mass_upperr_SNphotoz"],
                )
            )
        ).T,
        yerr=np.array(
            list(
                zip(
                    M24_w_zspe["mass_lowerr"],
                    M24_w_zspe["mass_upperr"],
                )
            )
        ).T,
        fmt="o",
        alpha=0.3,
    )
    plt.xlabel(r"$mass_{SNphoto z}$")
    plt.ylabel(r"$mass_{zspe}$")
    plt.plot(
        [7.5, 12.5],
        [7.5, 12.5],
        color="black",
        linewidth=1,
        linestyle="--",
        zorder=100,
    )
    plt.savefig(f"{path_plots}/M24_wzspe_mass.png")

    # 2d
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, wspace=0.4, hspace=0.1, height_ratios=[1, 0.4])
    axs = gs.subplots(sharex="col", sharey=False)

    sel = M24_w_zspe[M24_w_zspe["mass_SNphotoz"] > 8]
    axs[0].hist2d(
        sel["mass_SNphotoz"],
        sel["mass"],
        bins=70,
        cmap="magma_r",
        density=True,
    )
    axs[0].plot(
        [8, 12],
        [8, 12],
        color="black",
        linewidth=1,
        linestyle="--",
        zorder=100,
    )
    axs[0].set_xlim(8, 12)
    axs[0].set_ylim(8, 12)

    # 2nd row: delta
    axs[1].hist2d(
        sel["mass_SNphotoz"],
        sel["mass"] - sel["mass_SNphotoz"],
        bins=70,
        cmap="magma_r",
        density=True,
    )
    axs[1].plot(
        [lines_ranges["mass"][0], lines_ranges["mass"][1]],
        [0, 0],
        color="black",
        linewidth=1,
        linestyle="--",
        zorder=100,
    )
    axs[1].set_ylim(-0.5, 0.5)

    axs[0].set_ylabel("mass")
    axs[1].set_xlabel(r"$mass_{SNphoto z}$")
    axs[1].set_ylabel("Delta m - m SNphoto z")
    plt.savefig(f"{path_plots}/M24_wzspe_mass_2d.png")

    # DLR
    fig = plt.figure()
    n, bins, _ = plt.hist(
        photoIa_nz_JLA["DLR"], label="M24", histtype="step", linewidth=2, bins=30
    )
    sel = photoIa_nz_JLA[photoIa_nz_JLA.SNID.isin(spec_Ia.SNID.values)]
    plt.hist(sel["DLR"], label="spec Ia", bins=bins, histtype="step")
    plt.hist(photoIa_wz_JLA["DLR"], label="M22", bins=bins, histtype="step")
    plt.legend(loc=0)
    plt.xlabel(r"DLR")
    plt.savefig(f"{path_plots}/DLR.png")
