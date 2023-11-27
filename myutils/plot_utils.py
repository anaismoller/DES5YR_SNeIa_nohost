import numpy as np
import pandas as pd
from venn import venn
from numpy import inf
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pylab as plt
from myutils import conf_utils as cu
from myutils import science_utils as su
from myutils import metric_utils as mu
from myutils import data_utils as du


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

CMAP = plt.cm.YlOrBr
pd.options.mode.chained_assignment = None
np.seterr(divide="ignore", invalid="ignore")

mpl.rcParams["font.size"] = 16
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["figure.titlesize"] = "large"
mpl.rcParams["lines.linewidth"] = 3

ALL_COLORS = [
    "maroon",
    "royalblue",
    "indigo",
    "darkorange",
    "royalblue",
    "indigo",
    "maroon",
    "darkorange",
    "royalblue",
    "indigo",
    "maroon",
    "darkorange",
    "royalblue",
    "indigo",
    "maroon",
    "darkorange",
    "royalblue",
    "indigo",
    "maroon",
    "darkorange",
    "royalblue",
    "indigo",
]
ALL_COLORS_nodata = (
    ["royalblue", "darkorange", "indigo", "maroon"]
    + [k for k in ALL_COLORS if k != "maroon"]
    + ["darkorange"]
    + [k for k in ALL_COLORS if k != "maroon"]
)
override_color = ["grey", "maroon", "black", "royalblue", "indigo"]
BI_COLORS = ["darkorange", "royalblue"]
CONTRAST_COLORS = ["darkorange", "grey", "indigo"]
MARKER_DIC = {"randomforest": "o", "CNN": "D", "vanilla": "s"}
FILL_DIC = {"None": "none", "zpho": "bottom", "zspe": "full"}
MARKER_LIST = ["o", "o", "v", "v", "^", "^", ">", ">", "<", "<", "s", "s", "D", "D"]
CMAP = plt.cm.YlOrBr
LINE_STYLE = ["-", "-", "-", "-", ":", ":", ":", ":", "-.", "-.", "-.", "-."]
FILTER_COLORS = {"z": "maroon", "i": "darkorange", "r": "royalblue", "g": "indigo"}
PATTERNS = ["", "", "", "", "", ".", ".", ".", "."]

cmap = plt.get_cmap("viridis", 6)  # matplotlib color palette name, n colors
colors_for_sample = [cmap(i) for i in range(cmap.N)]

SAMPLES_COLORS = {
    "M22": "darkorange",
    "noz": colors_for_sample[-2],
    "specIa": colors_for_sample[-5],
    "SNN>0.001": colors_for_sample[-4],
    "SNN>0.5": colors_for_sample[-3],
    "SNN>0.5 + HQ": colors_for_sample[-2],
    "DES SNe Ia HQ (fitted z)": colors_for_sample[-2],
    "DES SNe Ia M22": "darkorange",
    "DES SNe Ia spectroscopic": colors_for_sample[-5],
}


list_linestyle = 5 * ["-", "--", ":", "-."]
fmt_list = ["o", "^", "v", "s", "p", "*", "+", "x"]

color_dic = {
    "data": "black",
    "sim": "darkorange",
    "ratio": "indigo",
    "spec": "maroon",
}

bins_dic = {
    "zHD": np.linspace(0.1, 1.1, 12),
    "c": np.linspace(-0.3, 0.3, 12),
    "x1": np.linspace(-3, 3, 12),
    "HOST_LOGMASS": np.linspace(7, 13, 12),
    "m0obs_i": np.linspace(21, 25, 12),
    "zHD_zspe": np.linspace(0.1, 1.1, 12),
    "c_zspe": np.linspace(-0.3, 0.3, 12),
    "x1_zspe": np.linspace(-3, 3, 12),
    "HOST_LOGMASS_zspe": np.linspace(7, 13, 12),
    "m0obs_i_zspe": np.linspace(21, 25, 12),
    "average_probability_set_0": np.linspace(0.0, 1.0, 12),
    "HOSTGAL_MAG_r": np.linspace(17, 29, 12),
    "sfr": np.linspace(-10, 5, 12),
    "mass": np.linspace(5, 12, 10),
    "PHOTOZ": np.linspace(0.1, 1.1, 12),
}
chi_bins_dic = {
    "zHD": np.linspace(0.1, 1.1, 12),
    "c": np.linspace(-0.3, 0.3, 12),
    "x1": np.linspace(-3, 3, 12),
    "HOST_LOGMASS": np.linspace(7, 13, 12),
    "m0obs_i": np.linspace(19, 25, 12),
    "REDSHIFT_FINAL": np.linspace(0.1, 1.1, 12),
    "average_probability_set_0": np.linspace(0.0, 1.0, 12),
    "zHD_zspe": np.linspace(0.1, 1.1, 12),
    "c_zspe": np.linspace(-0.3, 0.3, 12),
    "x1_zspe": np.linspace(-3, 3, 12),
    "HOST_LOGMASS_zspe": np.linspace(7, 13, 12),
    "m0obs_i_zspe": np.linspace(21, 25, 12),
    "HOSTGAL_MAG_r": np.linspace(17, 29, 12),
    "sfr": np.linspace(-10, 5, 12),
    "mass": np.linspace(5, 12, 10),
    "PHOTOZ": np.linspace(0.1, 1.1, 12),
}


def my_chisquare(obs_list, obs_error_list, sim_list):
    """Computing chi square of two histogram values
    Args:
    - obs_list (list): list of values in histogram for observations/data
    - obs_error_list (list): list of error values in histogram for observations/data (poisson stats)
    - sim_list (list):list of values in histogram for predicitons/sim

    Returns:
    - chi square
    """

    # to avoid empty values
    mask = np.where(obs_error_list != 0)
    den = obs_error_list[mask]

    num = obs_list[mask] - sim_list[mask]

    fraction = (num**2) / (den**2)
    chi2 = fraction.sum()
    return chi2


def plot_calibration(mean_bins, std_bins, TPF, nameout):
    """ """
    plt.clf()
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax11 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i, f in enumerate(mean_bins.keys()):
        ax1.errorbar(
            mean_bins[f],
            TPF[f],
            yerr=std_bins[f],
            color=ALL_COLORS[i],
            label=f,
            marker=MARKER_LIST[i],
        )
        ax11.errorbar(
            mean_bins[f],
            TPF[f] - mean_bins[f],
            color=ALL_COLORS[i],
            marker=MARKER_LIST[i],
        )
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel("fraction of positives (fP)", fontsize=20)
    ax1.set_ylim([-0.05, 1.05])
    ax11.set_xlabel(f"mean predicted probability", fontsize=20)
    ax1.legend(loc="best", prop={"size": 20})
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax11.tick_params(labelsize=16)
    ax11.set_ylabel("residual fP", fontsize=20)
    ax11.set_ylim([-0.2, 0.2])
    ax11.plot([0, 1], np.zeros(len([0, 1])), "k:")
    ax11.plot([0, 1], 0.1 * np.ones(len([0, 1])), ":", color="grey")
    ax11.plot([0, 1], -0.1 * np.ones(len([0, 1])), ":", color="grey")
    plt.setp(ax11.get_xticklabels(), visible=True)

    plt.savefig(nameout)
    plt.close(fig)
    plt.clf()
    del fig

    # Hres w. histogram


def HRwhisto(df, spec_Ias, ax_left, ax_right, visible=False, prob_key="all_class0"):

    mubins = np.arange(-2, 2 + 0.1, 0.1)

    ax_left.scatter(
        df["zHD"],
        df["delmu"],
        c=df[prob_key],
        cmap=CMAP,
        vmin=0.5,
        vmax=1,
        s=50,
    )
    ax_left.errorbar(
        df["zHD"],
        df["delmu"],
        yerr=df["delmu_err"],
        color="gray",
        zorder=0,
        fmt="none",
        marker="None",
    )

    ax_left.scatter(
        spec_Ias["zHD"],
        spec_Ias["delmu"],
        c=spec_Ias[prob_key],
        cmap=CMAP,
        vmin=0.5,
        vmax=1,
        s=150,
        marker="*",
        edgecolors="darkorange",
    )

    ax_left.set_ylim(df["delmu"].min() - 0.1, df["delmu"].max() + 0.1)
    # ax_left.set_xlim(0.2, 1.2)
    ax_left.set_ylabel(f"HD residual", fontsize=18)
    ax_left.tick_params(labelsize=14)
    plt.setp(ax_left.get_xticklabels(), visible=visible)
    if visible is True:
        ax_left.set_xlabel("zHD", fontsize=18)
    sel = df
    n_SNe = len(sel)
    ax_right.hist(
        df["delmu"],
        orientation="horizontal",
        histtype="step",
        color="black",
        bins=mubins,
        # density=True,
        label=f"photoIa {n_SNe}",
        lw=2,
    )
    ax_right.hist(
        spec_Ias["delmu"],
        orientation="horizontal",
        histtype="step",
        color="maroon",
        bins=mubins,
        # density=True,
        label=f"specIa {len(spec_Ias)}",
        lw=2,
        linestyle="-.",
    )

    ax_right.legend(loc="lower center", prop={"size": 13})
    plt.setp(ax_right.get_yticklabels(), visible=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)
    ax_right.plot(
        [ax_right.get_xlim()[0], ax_right.get_xlim()[1]],
        np.zeros(len([ax_right.get_xlim()[0], ax_right.get_xlim()[1]])),
        "k:",
    )


def plot_HD(
    preds_with_salt2, path_output, prob_key="all_class0", plot_contamination=True
):
    """
    Args:
        - preds_with_salt2 (DataFrame): predictions with their SALT2 fits
        - path_output (str): path with filename to write
    """
    plt.clf()
    fig = plt.figure(constrained_layout=True)
    df = su.distance_modulus(preds_with_salt2)
    # plt.scatter(df.zCMB, df.mu,label='all')
    if prob_key != "None":
        photo = df[df[prob_key] > 0.5]
    else:
        # provided events is the photo sammple
        photo = df
    plt.scatter(photo["zHD"], photo["mu"], label=f"photo {len(photo)}")
    if plot_contamination:
        photo_cont = df[(df[prob_key] > 0.5) & (df.target == 1)]
        plt.scatter(
            photo_cont["zHD"],
            photo_cont["mu"],
            label=f"contamination {len(photo_cont)} - {round(len(photo_cont)/len(photo),3)}%",
        )
    plt.ylabel("mu")
    plt.xlabel("zHD")
    plt.ylim(37, 46)
    plt.legend()
    plt.savefig(path_output)
    del df, fig
    plt.clf()


def plot_HD_residuals(preds_with_salt2, path_output, prob_key="all_class0"):

    df = su.distance_modulus(preds_with_salt2)
    spec_Ias = df[df.SNTYPE.isin(cu.spec_tags["Ia"])]

    plt.clf()
    cm = CMAP
    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

    # gridspec init
    ax00 = plt.subplot(gs[0, 0:2])  # Hres photoIa
    ax01 = plt.subplot(gs[0, 2], sharey=ax00)  # histo delta mu
    ax10 = plt.subplot(gs[1, 0])  # histo redshift
    ax11 = plt.subplot(gs[1, 1])  # histo x1
    ax12 = plt.subplot(gs[1, 2])  # histo c

    # lines
    ax00.plot([0, 1.2], np.zeros(len([0, 1.2])), "k:")

    HRwhisto(df, spec_Ias, ax00, ax01, visible=True, prob_key=prob_key)

    # z histos
    n, bins_to_use, tmp = ax10.hist(
        df["zHD"],
        histtype="step",
        color="black",
        bins=15,
        lw=3,
    )
    ax10.hist(
        spec_Ias["zHD"],
        histtype="step",
        color="maroon",
        bins=bins_to_use,
        lw=3,
        linestyle="-.",
    )
    ax10.set_xlabel("zHD", fontsize=18)

    # hist stretch
    n, bins_to_use, tmp = ax11.hist(
        df["x1"],
        color="black",
        histtype="step",
        lw=3,
    )
    ax11.hist(
        spec_Ias["x1"],
        color="maroon",
        histtype="step",
        lw=3,
        bins=bins_to_use,
        linestyle="-.",
    )
    ax11.set_xlabel("x1", fontsize=18)
    ax11.yaxis.set_label_position("right")
    ax11.set_xlim(-3, 3)
    ax11.tick_params(labelsize=14)
    # color histo
    n, bins_to_use, tmp = ax12.hist(df["c"], color="black", histtype="step", lw=3)
    ax12.hist(
        spec_Ias["c"],
        color="maroon",
        histtype="step",
        lw=3,
        bins=bins_to_use,
        linestyle="-.",
    )

    ax12.set_xlabel("c", fontsize=18)
    ax12.set_xlim(-0.3, 0.3)
    ax12.tick_params(labelsize=14)
    ax12.yaxis.set_label_position("right")

    gs.tight_layout(fig)
    plt.savefig(path_output)
    plt.close()
    del fig


def plot_venn_percentages(dic_venn, path_plots="./", suffix="", data=False):
    """Venn diagram of samples (max 6 sets)
    Args:
        - dic_venn (dictionary): dict with IDs of selected sample
        - path_plots (str): path to save plots
        - suffix (str): filename/exp name
    """
    prefix = "DES5YR" if data else "sim"

    plt.clf()
    fig = plt.figure(constrained_layout=True)
    venn(dic_venn, legend_loc="best", fmt="{percentage:.1f}%")
    plt.savefig(f"{path_plots}/{prefix}_venn_{suffix}.png")
    plt.clf()
    del fig


def plot_venn(dic_venn, path_plots="./", suffix=""):
    """Venn diagram of samples (max 6 sets)
    Args:
        - dic_venn (dictionary): dict with IDs of selected sample
        - path_plots (str): path to save plots
        - suffix (str): filename/exp name
    """

    plt.clf()
    fig = plt.figure(constrained_layout=True)
    venn(dic_venn, legend_loc="best")
    plt.savefig(f"{path_plots}/venn_{suffix}.png")
    plt.clf()
    del fig


def plot_venn2(dic_venn, path_plots="./", suffix=""):
    """Venn diagram of samples (max 2 sets)
    Args:
        - dic_venn (dictionary): dict with IDs of selected sample
        - path_plots (str): path to save plots
        - suffix (str): filename/exp name
    """
    from matplotlib_venn import venn2

    plt.clf()
    fig = plt.figure(constrained_layout=True)
    venn2(
        subsets=[dic_venn[x] for x in dic_venn.keys()],
        set_labels=[k for k in dic_venn.keys()],
    )
    plt.savefig(f"{path_plots}/venn2_{suffix}.png")
    plt.clf()
    del fig


def plot_labels(df, dic_SNIDs_vi, path_plots="./", suffix="test"):

    plt.clf()
    fig = plt.figure(constrained_layout=True)
    for k in dic_SNIDs_vi.keys():
        plt.hist(
            df[df.SNID.isin(dic_SNIDs_vi[k])].REDSHIFT_FINAL,
            histtype="step",
            label=k,
        )
    plt.xlabel("REDSHIFT_FINAL")
    plt.legend()
    plt.savefig(f"{path_plots}/labels_summary.png")
    plt.clf()
    del fig


def plot_2d(
    mean_dic,
    err_dic,
    var1,
    var2,
    zbin_dic,
    path_plots="./",
    prefix="testd",
    suffix="test",
):
    fig = plt.figure(constrained_layout=True)
    fig = plt.errorbar(
        zbin_dic["z_bins_plot"], mean_dic[var1], yerr=err_dic[var1], fmt="o"
    )
    plt.xlim(0, zbin_dic["max_z"] + zbin_dic["half_z_bin_step"])
    plt.ylabel(var1)
    plt.xlabel(var2)
    plt.legend()
    plt.savefig(f"{path_plots}/{prefix}_photoIa_{var1}_{var2}_{suffix}.png")
    del fig

    # def plots_vs_z(df, path_plots="./", prefix="testd", suffix="test", data=False):
    # Binning data by z, c and x1 distributions
    zkey = "REDSHIFT_FINAL" if data else "SIM_ZCMB"

    mean_dic, err_dic = bin_c_x1_vs_z(df, zkey=zkey)

    # Plot
    plot_2d(
        mean_dic,
        err_dic,
        "c",
        "zHD",
        zbin_dic,
        path_plots=path_plots,
        prefix=prefix,
        suffix=suffix,
    )
    plot_2d(
        mean_dic,
        err_dic,
        "x1",
        "zHD",
        zbin_dic,
        path_plots=path_plots,
        prefix=prefix,
        suffix=suffix,
    )


def plot_errorbar_binned(
    list_df,
    list_labels,
    binname="zbin",
    varx="zHD",
    vary="c",
    axs=None,
    sim_scale_factor=150,
    data_color_override=False,
    ignore_y_label=False,
    color_offset=0,
    color_list=[],
    marker_size=10,
):
    color_list = ALL_COLORS_nodata if len(color_list) < 2 else color_list
    if axs == None:
        plt.clf()
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        gs = fig.add_gridspec(1, 1)
        axs = gs.subplots(sharex=False, sharey=False)
    for i, df in enumerate(list_df):
        if binname == "HOST_LOGMASS_bin" or vary == "HOST_LOGMASS":
            df = df[df["HOST_LOGMASS"] > 0]
        if "sim" in list_labels[i]:
            # hack to see the dispersion if I sampled randomly once DES
            # scaling to 5XDES (150=30 realizations of 5 times DES)
            list_varx = []
            list_vary = []
            for subsamples in range(100):  # alway do a 100 sample
                sel = df.sample(frac=1 / sim_scale_factor)
                val_varx = sel.groupby(binname).mean()[varx].values
                val_vary = sel.groupby(binname).mean()[vary].values
                list_varx.append(val_varx)
                list_vary.append(val_vary)
                axs.plot(
                    val_varx,
                    val_vary,
                    color="darkgrey",
                    linewidth=0.5,
                    zorder=-32,
                    alpha=0.5,
                )
            # plot 68 percentile
            result_low = np.nanpercentile(list_vary, 16, axis=0)
            result_high = np.nanpercentile(list_vary, 84, axis=0)
            axs.fill_between(
                val_varx,
                result_low,
                result_high,
                color="darkgrey",
                alpha=0.8,
                zorder=-20,
            )
        axs.errorbar(
            df.groupby(binname).mean()[varx].values,
            df.groupby(binname).mean()[vary].values,
            yerr=df.groupby(binname).std()[vary].values
            / np.sqrt(df.groupby(binname)[vary].count()).values,
            label=list_labels[i],
            fmt="o"
            if "data" in list_labels[i]
            or "photo" in list_labels[i]
            or "DES" in list_labels[i]
            else "",
            color=color_list[i + color_offset]
            if data_color_override
            else color_dic["data"]
            if "data" in list_labels[i]
            or "photo" in list_labels[i]
            or "DES" in list_labels[i]
            else color_list[i + color_offset],
            zorder=50 if "data" in list_labels[i] else i,  # hack to put data on top
            ms=marker_size,
        )

    if not ignore_y_label:
        axs.set_ylabel(vary, fontsize=20)
    if ignore_y_label:
        axs.set_yticks([])
    if axs == None:
        axs.set_xlabel(varx)
        return fig
    else:
        return axs


def plot_histograms_listdf(
    list_df,
    list_labels,
    varx="zHD",
    density=True,
    norm_factor=1,
    outname="test.png",
    log_scale=False,
    nbins=10,
):
    plt.clf()
    fig = plt.figure(constrained_layout=True)
    for i, df in enumerate(list_df):
        col = ALL_COLORS[i]
        n, bins, _ = plt.hist(
            df[varx],
            histtype="step",
            color=col,
            label=list_labels[i],
            density=density,
            bins=bins_dic[varx] if varx in bins_dic.keys() else nbins,
            weights=norm_factor * np.ones(len(df)),
            ls=list_linestyle[i],
        )
        if i == 0 and varx not in bins_dic.keys():
            bins_dic[varx] = bins
    plt.xlabel(varx)
    if log_scale:
        plt.yscale("log")
    plt.legend(loc="upper left")
    plt.savefig(outname)
    plt.clf()
    del fig
    plt.close("all")


def plot_salt_distributions(
    df,
    target_key="predicted_target_S_0",
    path_plots="./",
    suffix="test",
    data=False,
):

    list_to_plot = ["REDSHIFT_FINAL", "c", "x1"] if data else ["SIM_ZCMB", "c", "x1"]
    for k in list_to_plot:
        plt.clf()
        fig = plt.figure(constrained_layout=True)
        if data:
            prefix = "DES5yr"
            plt.hist(
                df[k],
                histtype="step",
                label="photo Ia",
            )
            plt.hist(
                df[df.SNTYPE.isin(cu.spec_tags["Ia"])][k],
                histtype="step",
                label="photo + spec Ia",
            )
        else:
            prefix = "sim"
            plt.hist(
                df[df.target == 0][k],
                histtype="step",
                label="all sim Ia",
            )
            plt.hist(
                df[(df.target == 0) & (df[target_key] == 0)][k],
                histtype="step",
                label="photo Ia",
            )
        plt.xlabel(k)
        plt.legend()
        plt.yscale("log")
        plt.savefig(f"{path_plots}/{prefix}_photoIa_{k}_{suffix}.png")
        plt.clf()
        del fig


def plot_contamination_list(list_tuple, path_plots="./", suffix="", list_labels=[""]):
    """Plot contamination as a function of SALT2 parameters
    Args:
        - list_tuple (list of tuples):  [(x,saltfitsofx),(...)]
        - path_plots (str): path to save plots
        - suffix (str): identification string
        - list_labels (list): list of strings to tag the tuples
    """

    my_bins_dic = acc_bins_dic = {
        "zHD": np.linspace(0.0, 1.2, 12),
        "c": np.linspace(-0.4, 0.4, 14),
        "x1": np.linspace(-4, 4, 14),
        "m0obs_i": np.linspace(21, 25, 12),
    }
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.01)
    axs = gs.subplots(sharex=False, sharey=True)

    for i, k in enumerate(["m0obs_i", "zHD", "c", "x1"]):
        # bin
        bins = my_bins_dic[k]
        for idx_tuple, tupl in enumerate(list_tuple):
            sim_preds, sim_fits = tupl
            sim_cut_nonIa = sim_preds["cosmo_quantile"]["target"] == 1
            contamination = sim_preds["cosmo_quantile"][sim_cut_nonIa]

            contamination_fits = sim_fits[sim_fits.SNID.isin(contamination.SNID.values)]
            contamination_fits[f"{k}_bin"] = pd.cut(
                contamination_fits.loc[:, (k)], bins
            )
            sim_fits[f"{k}_bin"] = pd.cut(sim_fits.loc[:, (k)], bins)
            tmp_my_bins = bins[:-1] + (bins[1] - bins[0]) / 2

            percentage_contamination = []
            error_content = []
            my_bins = []
            for classset in [0, 1, 2]:
                # photo Ia
                SNID_photoIa_tmp = sim_preds["cosmo_quantile"][
                    sim_preds["cosmo_quantile"][f"average_probability_set_{classset}"]
                    > 0.5
                ].SNID.values
                sim_cut_photo = sim_fits.SNID.isin(SNID_photoIa_tmp)
                conta_cut = contamination_fits.SNID.isin(SNID_photoIa_tmp)

                photo = sim_fits[sim_cut_photo].groupby(f"{k}_bin").count()["SNID"]
                conta = (
                    contamination_fits[conta_cut].groupby(f"{k}_bin").count()["SNID"]
                )
                tmp_perc_cont = 100 * conta.values / photo.values
                tmp_err_cont = 100 * (
                    np.sqrt(conta) / photo - conta / np.power(photo, 2) * np.sqrt(photo)
                )

                percentage_contamination.append(tmp_perc_cont)
                error_content.append(tmp_err_cont)
                my_bins.append(tmp_my_bins)

            axp = axs[0][i] if k in ["m0obs_i", "zHD"] else axs[1][i - 2]
            plot_bins = np.array(my_bins).mean(axis=0)
            # assuming additive error which is not necessarily true!!!!!
            axp.errorbar(
                plot_bins,
                np.array(percentage_contamination).mean(axis=0),
                yerr=np.array(percentage_contamination).std(axis=0)
                + np.array(error_content).mean(axis=0),
                label=list_labels[idx_tuple],
                ls=list_linestyle[idx_tuple],
                color=override_color[idx_tuple],
            )
            if i in [0, 2]:
                axp.set_ylabel("contamination", fontsize=20)
            axp.set_xticks(plot_bins)
            tmp = [k if bool(n % 2) else "" for n, k in enumerate(plot_bins.round(1))]
            axp.set_xticklabels(tmp, fontsize=15)
            xlabel = k if k != "m0obs_i" else r"$i_{peak}$"
            xlabel = xlabel if k != "zHD" else "redshift"
            axp.set_xlabel(xlabel, fontsize=20)
    axp.legend(loc="best", prop={"size": 20})
    if suffix != "":
        plt.savefig(f"{path_plots}/contamination_photoIa_{suffix}.png")
    else:
        plt.savefig(f"{path_plots}/contamination_photoIa.png")
    plt.clf()
    del fig


def plot_metrics_list(
    list_sims, path_plots="./", suffix="", list_labels=[""], metric="accuracy"
):
    """Plot metric as a function of SALT2 parameters
    Args:
        - list_ (list):  list of sim preds with salt fits to use for the evaluation
        - path_plots (str): path to save plots
        - suffix (str): identification string
        - list_labels (list): list of strings to tag the tuples
        - metric (str): which metric to plot
    """

    from sklearn import metrics

    acc_bins_dic = {
        "zHD": np.linspace(0.0, 1.2, 12),
        "c": np.linspace(-0.4, 0.4, 14),
        "x1": np.linspace(-4, 4, 14),
        "m0obs_i": np.linspace(21, 25, 12),
    }

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.01)
    axs = gs.subplots(sharex=False, sharey=True)

    for i, k in enumerate(["m0obs_i", "zHD", "c", "x1"]):
        bins = acc_bins_dic[k]
        mean_bins = bins[:-1] + (bins[1] - bins[0]) / 2

        for idx_sim, sim in enumerate(list_sims):
            list_accuracy = []
            list_notbalaccuracy = []
            list_purity = []
            list_efficiency = []
            # binning
            sim[f"{k}_bin"] = pd.cut(sim.loc[:, (k)], bins, labels=mean_bins)

            for classset in [0, 1, 2]:
                acc_bins = []
                notbalacc_bins = []
                pur_bins = []
                eff_bins = []
                # this may not work with cut bins
                for saltbin in mean_bins:
                    sel = sim[sim[f"{k}_bin"] == saltbin]
                    (
                        accuracy,
                        balancedaccuracy,
                        _,
                        purity,
                        efficiency,
                        _,
                    ) = mu.performance_metrics(
                        sel,
                        key_pred_targ=f"predicted_target_average_probability_set_{classset}",
                        compute_auc=False,
                    )
                    acc_bins.append(balancedaccuracy)
                    notbalacc_bins.append(accuracy)
                    pur_bins.append(purity)
                    eff_bins.append(efficiency)
                list_accuracy.append(acc_bins)
                list_purity.append(pur_bins)
                list_efficiency.append(eff_bins)
                list_notbalaccuracy.append(notbalacc_bins)

            axp = axs[0][i] if k in ["m0obs_i", "zHD"] else axs[1][i - 2]
            if metric == "accuracy":
                to_plot = list_accuracy
            elif metric == "efficiency":
                to_plot = list_efficiency
            tmp = np.array(to_plot).mean(axis=0)
            mask = np.where(tmp != 0)
            mean_to_plot = tmp[mask]
            mean_bins_to_plot = mean_bins[mask]
            tmperr = np.array(to_plot).std(axis=0)
            err_to_plot = tmperr[mask]
            axp.errorbar(
                mean_bins_to_plot,
                mean_to_plot,
                yerr=err_to_plot,
                label=f"{list_labels[idx_sim]}",
                ls=list_linestyle[idx_sim],
                color=override_color[idx_sim],
            )
            if i in [0, 2]:
                axp.set_ylabel(metric, fontsize=20)
            axp.set_xticks(mean_bins)
            tmp = [k if bool(n % 2) else "" for n, k in enumerate(mean_bins.round(1))]
            axp.set_xticklabels(tmp, fontsize=16)

            xlabel = k if k != "m0obs_i" else r"$i_{peak}$"
            if k == "zHD":
                xlabel = "redshift"
            axp.set_xlabel(xlabel, fontsize=24)
    axp.legend(loc="best", prop={"size": 20})
    if suffix != "":
        plt.savefig(f"{path_plots}/{metric}_photoIa_{suffix}.png")
    else:
        plt.savefig(f"{path_plots}/{metric}_photoIa.png")
    plt.clf()
    del fig


def overplot_salt_distributions(
    sample_photo_Ia_fits,
    sim_Ia_fits,
    sim_photoIa_fits,
    path_plots="./",
    suffix="",
    list_labels=[
        "photometric Ia",
        "sim JLA SNN photoIa",
        "sim photoIa",
    ],
):
    from numpy import inf

    list_to_plot = ["zHD", "c", "x1"]

    sim_JLA_Ia_fits = su.apply_JLA_cut(sim_Ia_fits)

    sim_JLA_photoIa_fits = su.apply_JLA_cut(sim_photoIa_fits)

    for density_option in [False]:  # [True, False]:
        density_str = "normed" if density_option == True else "notnormed"
        norm_factor = 1 / 150 if density_option == False else 1  # 5XDES x 30 randseeds
        plt.clf()
        fig = plt.figure(figsize=(20, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, hspace=0, wspace=0.05)
        axs = gs.subplots(sharex=False, sharey=True)
        for i, k in enumerate(list_to_plot):
            axs[i].hist(
                sim_JLA_Ia_fits[k],
                histtype="step",
                color=color_dic["sim"],
                label="sim Ia JLA cuts",
                density=density_option,
                bins=bins_dic[k],
                weights=norm_factor * np.ones(len(sim_JLA_Ia_fits)),
                linewidth=10,
            )
            axs[i].set_xlabel(k)

            # hack for data
            sample_photo_Ia_fits["tmp_bin"] = pd.cut(
                sample_photo_Ia_fits.loc[:, (k)], bins_dic[k]
            )
            err = np.sqrt(sample_photo_Ia_fits.groupby("tmp_bin").count()[k].values)
            # ano errors in normed histos but it is not great
            if density_str == "normed":
                err = 0.0
            err[np.abs(err) == inf] = 0
            hist_vals, bin_edges = np.histogram(
                sample_photo_Ia_fits[k], density=density_option, bins=bins_dic[k]
            )
            axs[i].errorbar(
                bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
                hist_vals,
                yerr=err,
                label="data",
                fmt="o",
                color=color_dic["data"],
            )
            axs[i].set_xlabel(k)
        axs[i].legend()
        plt.savefig(f"{path_plots}/hists_sample_sim_Ia.png")
        plt.clf()
        del fig

    # c,x1 vs z, host_logmass
    for xbin in ["zHD", "HOST_LOGMASS", "m0obs_i"]:
        bins = bins_dic[xbin][:-1]
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        sim_JLA_Ia_fits[f"{xbin}_bin"] = pd.cut(sim_JLA_Ia_fits.loc[:, (xbin)], bins)
        sim_JLA_photoIa_fits[f"{xbin}_bin"] = pd.cut(
            sim_JLA_photoIa_fits.loc[:, (xbin)], bins
        )
        sample_photo_Ia_fits[f"{xbin}_bin"] = pd.cut(
            sample_photo_Ia_fits.loc[:, (xbin)], bins
        )

        # Testing bench
        list_df = [
            sample_photo_Ia_fits,
            sim_JLA_Ia_fits,
            sim_JLA_photoIa_fits,
        ]

        to_plot = ["c", "x1", "HOST_LOGMASS"] if xbin == "zHD" else ["c", "x1", "zHD"]

        for k in to_plot:
            fig = plot_errorbar_binned(
                list_df, list_labels, binname=f"{xbin}_bin", varx=xbin, vary=k
            )
            plt.legend()
            plt.xlabel(xbin)
            if suffix != "":
                plt.savefig(
                    f"{path_plots}/2ddist_sample_sim_Ia_{k}_{xbin}_{suffix}.png"
                )
            else:
                plt.savefig(f"{path_plots}/2ddist_sample_sim_Ia_{k}_{xbin}.png")
            del fig


def plot_mosaic_histograms_listdf(
    list_df,
    list_labels=["tmp"],
    path_plots="./",
    suffix="",
    list_vars_to_plot=["zHD", "c", "x1"],
    norm=1 / 150,
    data_color_override=False,
    chi_bins=True,
    log_scale=False,
    use_samples_color_palette=False,
):

    bins_to_plot = chi_bins_dic if chi_bins == True else bins_dic

    plt.clf()
    ysize = 4 if len(list_vars_to_plot) > 4 else 5
    fig = plt.figure(figsize=(18, ysize), constrained_layout=True)
    gs = fig.add_gridspec(1, len(list_vars_to_plot), hspace=0, wspace=0.05)
    axs = gs.subplots(sharex=False, sharey=True)

    for i, k in enumerate(list_vars_to_plot):
        sim_vals_list = []
        sim_label_list = []
        for df_idx, df in enumerate(list_df):
            if "sim" in list_labels[df_idx]:
                norm = norm
                sim_vals, _, _ = axs[i].hist(
                    df[k],
                    histtype="step",
                    label=list_labels[df_idx],
                    density=False,
                    bins=bins_to_plot[k],
                    weights=norm * np.ones(len(df)),
                    linewidth=5,
                    color=SAMPLES_COLORS[list_labels[df_idx]]
                    if use_samples_color_palette
                    else ALL_COLORS_nodata[df_idx],
                    linestyle=list_linestyle[df_idx],
                )
                sim_vals_list.append(sim_vals)
                tmp = "f" if "fixed" in list_labels[df_idx] else "S"
                sim_label_list.append(tmp)
            else:
                df["tmp_bin"] = pd.cut(df.loc[:, (k)], bins_to_plot[k])
                err = np.sqrt(df.groupby("tmp_bin").count()[k].values)

                err[np.abs(err) == inf] = 0
                data_hist_vals, bin_edges = np.histogram(
                    df[k], density=False, bins=bins_to_plot[k]
                )
                # du.print_stats(df[k], context=f"{k} {list_labels[df_idx]}")
                if data_color_override == True:
                    axs[i].errorbar(
                        bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
                        data_hist_vals,
                        yerr=err,
                        fmt="none",
                        ecolor=SAMPLES_COLORS[list_labels[df_idx]]
                        if use_samples_color_palette
                        else ALL_COLORS_nodata[df_idx],
                    )
                    axs[i].hist(
                        df[k],
                        histtype="step",
                        label=list_labels[df_idx],
                        density=False,
                        bins=bins_to_plot[k],
                        linewidth=5,
                        color=SAMPLES_COLORS[list_labels[df_idx]]
                        if use_samples_color_palette
                        else ALL_COLORS_nodata[df_idx],
                        linestyle=list_linestyle[df_idx],
                    )
                else:
                    axs[i].errorbar(
                        bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
                        data_hist_vals,
                        yerr=err,
                        label=list_labels[df_idx],
                        fmt="o",
                        color=color_dic["data"]
                        if data_color_override == False
                        else (
                            SAMPLES_COLORS[list_labels[df_idx]]
                            if use_samples_color_palette
                            else ALL_COLORS[df_idx]
                        ),
                        markersize=8,
                        mfc=color_dic["data"]
                        if data_color_override == False
                        else (
                            SAMPLES_COLORS[list_labels[df_idx]]
                            if use_samples_color_palette
                            else ALL_COLORS[df_idx]
                        ),
                    )

        counter = 0
        for sim_vals in sim_vals_list:

            chi = my_chisquare(data_hist_vals, err, sim_vals)
            mask = np.where(sim_vals != 0)
            nbins = len(mask[0])
            if len(sim_label_list) > 1:
                tmptag = sim_label_list[counter]
                tmp = rf"$\chi^2_{tmptag}=$"
                text_chi2 = f"{tmp} {round(chi,1)}/{nbins}"
            else:
                tmp = r"$\chi^2/bins=$"
                text_chi2 = f"{tmp} {round(chi,1)}/{nbins}"
            axs[i].annotate(
                text_chi2,
                xy=(0.05, 0.90 - 0.15 * counter),
                size=16,
                xycoords="axes fraction",
            )
            counter += 1
        xlabel = k if k != "m0obs_i" else r"$i_{peak}$"
        if k == "zHD":
            xlabel = "z"
        if k == "HOSTGAL_MAG_r":
            xlabel = "host r magnitude"
        axs[i].set_xlabel(xlabel, fontsize=20)
        if log_scale:
            axs[i].set_yscale("log")
    axs[0].set_ylabel("# events")
    legfont = 14 if len(list_vars_to_plot) > 4 else 16
    axs[-1].legend(
        bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.0, fontsize=legfont
    )
    plt.savefig(f"{path_plots}/hists_sample_sim_Ia{suffix}.png")
    plt.clf()
    del fig


def overplot_salt_distributions_lists(
    list_df,
    list_labels=["tmp"],
    path_plots="./",
    suffix="",
    sim_scale_factor=150,
    data_color_override=False,
):

    plot_mosaic_histograms_listdf(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix=suffix,
        norm=1 / sim_scale_factor,
        list_vars_to_plot=["zHD", "c", "x1", "m0obs_i"],
        data_color_override=data_color_override,
    )

    # # c,x1 vs z, host_logmass
    for xbin in ["zHD", "HOST_LOGMASS", "m0obs_i"]:
        bins = bins_dic[xbin]

        for df_idx, df in enumerate(list_df):
            df[f"{xbin}_bin"] = pd.cut(df.loc[:, (xbin)], bins)

        # to_plot = ["c", "x1", "HOST_LOGMASS"] if xbin == "zHD" else ["c", "x1", "zHD"]
        # for k in to_plot:
        #     fig = plot_errorbar_binned(
        #         list_df,
        #         list_labels,
        #         binname=f"{xbin}_bin",
        #         varx=xbin,
        #         vary=k,
        #         sim_scale_factor=sim_scale_factor,
        #     )
        #     plt.legend()
        #     plt.xlabel(xbin)
        #     if suffix != "":
        #         plt.savefig(
        #             f"{path_plots}/2ddist_sample_sim_Ia_{k}_{xbin}_{suffix}.png"
        #         )
        #     else:
        #         plt.savefig(f"{path_plots}/2ddist_sample_sim_Ia_{k}_{xbin}.png")
        #     del fig

    # zHD binned c,x1 together (same as above only formatting)
    xbin = "zHD"
    to_plot = ["c", "x1"]
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    for i, k in enumerate(to_plot):
        plot_errorbar_binned(
            list_df,
            list_labels,
            binname=f"{xbin}_bin",
            varx=xbin,
            vary=k,
            axs=axs[i],
            sim_scale_factor=sim_scale_factor,
            data_color_override=data_color_override,
        )
    axs[i].set_xlabel(xbin, fontsize=20)
    axs[i].legend(loc="best")
    if suffix != "":
        plt.savefig(f"{path_plots}/2ddist_sample_sim_Ia_cx1_{xbin}_{suffix}.png")
    else:
        plt.savefig(f"{path_plots}/2ddist_sample_sim_Ia_cx1_{xbin}.png")
    del fig


def overplot_salt_distributions_allmodels(fits, preds, path_plots="./", namestr="sim"):
    from numpy import inf

    list_to_plot = ["zHD", "c", "x1"]

    # do JLA cuts for sim
    JLA_fits = su.apply_JLA_cut(fits)

    # init
    # c,x1 vs z
    minv = JLA_fits["zHD"].quantile(0.01)
    maxv = JLA_fits["zHD"].quantile(0.99)
    bins = np.linspace(minv, maxv, 12)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    fits["zbin"] = pd.cut(np.array(fits.loc[:, ("zHD")]), bins)
    JLA_fits["zbin"] = pd.cut(np.array(JLA_fits.loc[:, ("zHD")]), bins)

    # all models classifications
    list_JLA_fits_photo = []
    list_class_model = []
    for class_model in [k for k in preds.keys() if "predicted_target" in k]:
        photo = preds[preds[class_model] == 0]["SNID"].values
        list_JLA_fits_photo.append(JLA_fits[JLA_fits.SNID.isin(photo)])
        list_class_model.append(class_model)

    list_df = list_JLA_fits_photo
    list_labels = list_class_model
    for k in list_to_plot:
        plot_histograms_listdf(list_df, list_labels, varx=k)
        plt.gca().set_ylim(bottom=0)
        plt.savefig(f"{path_plots}/hist_{namestr}_allmodels_photoIa_{k}.png")
        plt.clf()

    list_df = [fits, JLA_fits] + list_JLA_fits_photo
    list_labels = (
        ["all", "+JLA cut"]
        + ["" for k in range(len(list_JLA_fits_photo) - 1)]
        + ["JLA+SNN"]
    )
    for k in ["c", "x1"]:
        fig = plot_errorbar_binned(
            list_df, list_labels, binname="zbin", varx="zHD", vary=k
        )
        plt.legend()
        plt.savefig(f"{path_plots}/2ddist_{namestr}_allmodels_photoIa_{k}_z.png")


def plot_matrix(matrix, i_var="", j_var="", ticks=[], path_plots="./", suffix=""):

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    # Major ticks
    ax.set_xticks(np.arange(0, len(ticks), 1))
    ax.set_yticks(np.arange(0, len(ticks), 1))

    ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticklabels(ticks)
    ax.set_xlabel(j_var)
    ax.set_ylabel(i_var)
    plt.savefig(f"{path_plots}/migration_{i_var}_vs_{j_var}{suffix}.png")
    plt.clf()
    plt.close("all")
    del fig


def overplot_salt_distributions_listdf(
    list_df, list_labels, sim_Ia_fits, path_plots="./", suffix=""
):
    from numpy import inf

    list_to_plot = ["zHD", "c", "x1"]

    sim_JLA_Ia_fits = su.apply_JLA_cut(sim_Ia_fits)

    # init
    # c,x1 vs z
    minv = min([x.quantile(0.01) for x in [df["zHD"] for df in list_df]])
    maxv = max([x.quantile(0.99) for x in [df["zHD"] for df in list_df]])
    bins = np.linspace(minv, maxv, 12)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    sim_JLA_Ia_fits["zbin"] = pd.cut(np.array(sim_JLA_Ia_fits.loc[:, ("zHD")]), bins)
    for df in list_df:
        df["zbin"] = pd.cut(np.array(df.loc[:, ("zHD")]), bins)

    list_df_plot = [sim_JLA_Ia_fits] + list_df
    list_labels_plot = ["sim JLA Ia"] + list_labels
    for k in ["c", "x1"]:
        fig = plot_errorbar_binned(
            list_df_plot,
            list_labels_plot,
            binname="zbin",
            varx="zHD",
            vary=k,
            list_colors=ALL_COLORS[: len(list_labels_plot)],
        )
        plt.legend()
        if suffix != "":
            plt.savefig(f"{path_plots}/2ddist_sample_photoIa_{k}_z_{suffix}.png")
        else:
            plt.savefig(f"{path_plots}/2ddist_sample_photoIa_{k}_z.png")


def plot_cx1z_histos(
    list_df, list_labels, nameout="./test.png", my_bins_dic=None, verbose=False
):

    if not my_bins_dic:
        my_bins_dic = bins_dic

    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, hspace=0, wspace=0.05)
    axs = gs.subplots(sharex=False, sharey=True)
    for i, k in enumerate(["c", "x1", "zHD"]):
        for j, df in enumerate(list_df):
            axs[i].hist(
                df[k],
                histtype="step",
                label=list_labels[j],
                bins=my_bins_dic[k],
                linestyle=list_linestyle[j],
            )
            if verbose:
                du.print_stats(df[k], context=f"{k}")
        axs[i].set_xlabel(k)
        plt.legend()
    plt.savefig(nameout)


def DESVRO_plot_mosaic_histograms_listdf(
    sim_Ia_fits_JLA_wz,
    sim_Ia_fits_JLA_noz,
    sim_Ia_fits_JLA_saltz,
    photoIa_wz,
    photoIa_noz,
    path_plots=".",
    list_vars_to_plot=["zHD", "c", "x1", "m0obs_i"],
):

    plt.clf()
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, len(list_vars_to_plot), hspace=0, wspace=0.05)
    axs = gs.subplots(sharex=False, sharey=True)
    for i, k in enumerate(list_vars_to_plot):
        counter = 0
        # data for norm
        # hist_vals, bin_edges = np.histogram(
        #         photoIa_wz[k], density=False, bins=bins_dic[k]
        #     )
        # dat_wz = np.max(hist_vals)

        # hist_vals, bin_edges = np.histogram(
        #         photoIa_noz[k], density=False, bins=bins_dic[k]
        #     )
        # dat_noz = np.max(hist_vals)

        # sim plot
        list_df_wz = [sim_Ia_fits_JLA_wz]
        for df_idx, df in enumerate(list_df_wz):
            hist_vals_wz, bin_edges = np.histogram(
                df[k], density=False, bins=chi_bins_dic[k]
            )
            # norm = (
            #     dat_wz / np.max(hist_vals_wz) if dat_wz != 0 else 1
            # )
            norm = 1 / 150
            axs[i].hist(
                df[k],
                histtype="step",
                label="sim Ia JLA (host z)",
                density=False,
                bins=chi_bins_dic[k],
                weights=norm * np.ones(len(df)),
                linewidth=5,
                color=ALL_COLORS_nodata[df_idx],
                linestyle=list_linestyle[df_idx],
            )
        #
        list_df_noz = [sim_Ia_fits_JLA_noz, sim_Ia_fits_JLA_saltz]
        list_labels = ["sim Ia JLA (no z, sim z)", "sim Ia JLA (no z, SALT z)"]
        for df_idx, df in enumerate(list_df_noz):
            # sim_vals, bin_edges = np.histogram(df[k], density=False, bins=chi_bins_dic[k])
            # norm = (
            #     dat_noz / np.max(sim_vals) if dat_noz != 0 else 1
            # )
            norm = 1 / 30
            sim_vals, _, _ = axs[i].hist(
                df[k],
                histtype="step",
                label=list_labels[df_idx],
                density=False,
                bins=chi_bins_dic[k],
                weights=norm * np.ones(len(df)),
                linewidth=5,
                color=ALL_COLORS_nodata[df_idx],
                linestyle=list_linestyle[df_idx + 2],
            )

        # data
        df = photoIa_wz
        df["tmp_bin"] = pd.cut(df.loc[:, (k)], chi_bins_dic[k])
        err = np.sqrt(df.groupby("tmp_bin").count()[k].values)
        err[np.abs(err) == inf] = 0
        data_hist_vals, bin_edges = np.histogram(
            df[k], density=False, bins=chi_bins_dic[k]
        )
        axs[i].errorbar(
            bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
            data_hist_vals,
            yerr=err,
            label="photoIa JLA (host z)",
            fmt=fmt_list[0],
            color=color_dic["data"],
            markersize=8,
            mfc="black",
        )
        #
        df = photoIa_noz
        df["tmp_bin"] = pd.cut(df.loc[:, (k)], chi_bins_dic[k])
        err = np.sqrt(df.groupby("tmp_bin").count()[k].values)
        err[np.abs(err) == inf] = 0
        data_hist_vals, bin_edges = np.histogram(
            df[k], density=False, bins=chi_bins_dic[k]
        )
        axs[i].errorbar(
            bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
            data_hist_vals,
            yerr=err,
            label="photoIa JLA (SALT z)",
            fmt=fmt_list[1],
            color=color_dic["data"],
            markersize=8,
            mfc="white",
        )

        chi = my_chisquare(data_hist_vals, err, sim_vals)
        mask = np.where(sim_vals != 0)
        nbins = len(mask[0])
        tmp = r"$\chi^2 = $"
        text_chi2 = f"{tmp} {round(chi,1)}/{nbins}"
        axs[i].annotate(
            text_chi2,
            xy=(0.6, 0.9),
            size=14,
            xycoords="axes fraction",
        )

        axs[i].set_xlabel(k)

    axs[-1].legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.0, fontsize=16)
    plt.savefig(f"{path_plots}/hists_sample_sim_Ia_DESall.png")
    plt.clf()
    del fig


def plot_mosaic_histograms_listdf_deep_shallow(
    list_df,
    list_labels=["tmp"],
    path_plots="./",
    suffix="",
    list_vars_to_plot=["zHD", "c", "x1"],
    norm=1 / 150,
    data_color_override=False,
    chi_bins=True,
    log_scale=False,
    color_list=None,
):

    bins_to_plot = chi_bins_dic if chi_bins == True else bins_dic

    plt.clf()
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, len(list_vars_to_plot), hspace=0, wspace=0.01)
    axs = gs.subplots(sharex=False, sharey=True)

    for i, k in enumerate(list_vars_to_plot):
        sim_vals_list_deep = []
        sim_label_list_deep = []
        sim_vals_list_shallow = []
        sim_label_list_shallow = []
        counter_sim = 0
        for df_idx, df in enumerate(list_df):
            if (
                "sim" in list_labels[df_idx]
                or "simulations (fixed true z)" in list_labels[df_idx]
            ):
                # deep
                deep = df[df.deep == True]
                norm = norm
                sim_vals_deep, _, _ = axs[1][i].hist(
                    deep[k],
                    histtype="step",
                    label=f"{list_labels[df_idx]}",
                    density=False,
                    bins=bins_to_plot[k],
                    weights=norm * np.ones(len(deep)),
                    linewidth=5,
                    color="lightgrey" if counter_sim == 0 else "maroon",
                    linestyle=list_linestyle[df_idx],
                )
                sim_vals_list_deep.append(sim_vals_deep)
                tmp = "f" if "fixed" in list_labels[df_idx] else "S"
                sim_label_list_deep.append(tmp)
                # shallow
                shallow = df[df.shallow == True]
                norm = norm
                sim_vals_shallow, _, _ = axs[0][i].hist(
                    shallow[k],
                    histtype="step",
                    label=f"{list_labels[df_idx]}",
                    density=False,
                    bins=bins_to_plot[k],
                    weights=norm * np.ones(len(shallow)),
                    linewidth=5,
                    color="lightgrey" if counter_sim == 0 else "red",
                    linestyle=list_linestyle[df_idx],
                )
                sim_vals_list_shallow.append(sim_vals_shallow)
                tmp = "f" if "fixed" in list_labels[df_idx] else "S"
                sim_label_list_shallow.append(tmp)
                counter_sim += 1
            else:
                # deep
                deep = df[df.deep == True]
                deep["tmp_bin"] = pd.cut(deep.loc[:, (k)], bins_to_plot[k])
                err = np.sqrt(deep.groupby("tmp_bin").count()[k].values)

                err[np.abs(err) == inf] = 0
                data_hist_vals_deep, bin_edges = np.histogram(
                    deep[k], density=False, bins=bins_to_plot[k]
                )
                du.print_stats(deep[k], context=f"{k} {list_labels[df_idx]}")
                if data_color_override == True:
                    axs[0][i].errorbar(
                        bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
                        data_hist_vals_deep,
                        yerr=err,
                        fmt="none",
                        ecolor=ALL_COLORS[df_idx],
                    )
                    axs[0][i].hist(
                        df[k],
                        histtype="step",
                        label=list_labels[f"{df_idx}"],
                        density=False,
                        bins=bins_to_plot[k],
                        linewidth=5,
                        color=ALL_COLORS[df_idx],
                        linestyle=list_linestyle[df_idx],
                    )
                # shallow
                shallow = df[df.shallow == True]
                shallow["tmp_bin"] = pd.cut(shallow.loc[:, (k)], bins_to_plot[k])
                err = np.sqrt(shallow.groupby("tmp_bin").count()[k].values)

                err[np.abs(err) == inf] = 0
                data_hist_vals_shallow, bin_edges = np.histogram(
                    shallow[k], density=False, bins=bins_to_plot[k]
                )
                du.print_stats(shallow[k], context=f"{k} {list_labels[df_idx]}")

                if data_color_override == True:
                    axs[0][i].errorbar(
                        bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
                        data_hist_vals_shallow,
                        yerr=err,
                        fmt="none",
                        ecolor=ALL_COLORS[df_idx],
                    )
                    axs[0][i].hist(
                        df[k],
                        histtype="step",
                        label=f"{list_labels[df_idx]}",
                        density=False,
                        bins=bins_to_plot[k],
                        linewidth=5,
                        color=ALL_COLORS[df_idx],
                        linestyle=list_linestyle[df_idx],
                    )
                else:
                    # deep
                    axs[1][i].errorbar(
                        bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
                        data_hist_vals_deep,
                        yerr=err,
                        label=f"{list_labels[df_idx]}",
                        fmt="o",
                        color=color_dic["data"]
                        if data_color_override == False
                        else ALL_COLORS[df_idx],
                        markersize=8,
                        mfc=color_dic["data"]
                        if data_color_override == False
                        else ALL_COLORS[df_idx],
                    )
                    # shallow
                    axs[0][i].errorbar(
                        bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2.0,
                        data_hist_vals_shallow,
                        yerr=err,
                        label=f"{list_labels[df_idx]}",
                        fmt="o",
                        color=color_dic["data"],
                        markersize=8,
                        mfc=color_dic["data"]
                        if data_color_override == False
                        else ALL_COLORS[df_idx],
                    )

        counter = 0
        for sim_vals in sim_vals_list_deep:
            chi = my_chisquare(data_hist_vals_deep, err, sim_vals)
            mask = np.where(sim_vals != 0)
            nbins = len(mask[0])
            if len(sim_label_list_deep) > 1:
                tmptag = sim_label_list_deep[counter]
                tmp = rf"$\chi^2_{tmptag}=$"
                text_chi2 = f"{tmp} {round(chi,1)}/{nbins}"
            else:
                tmp = r"$\chi^2/bins=$"
                text_chi2 = f"{tmp} {round(chi,1)}/{nbins}"
            axs[1][i].annotate(
                text_chi2,
                xy=(0.03, 0.90 - 0.15 * counter),
                size=16,
                xycoords="axes fraction",
            )
            counter += 1
        xlabel = k if k != "m0obs_i" else r"$i_{peak}$"
        if k == "zHD":
            xlabel = "z"
        axs[1][i].set_xlabel(xlabel, fontsize=20)
        if log_scale:
            axs[1][i].set_yscale("log")

        counter = 0
        for sim_vals in sim_vals_list_shallow:
            chi = my_chisquare(data_hist_vals_shallow, err, sim_vals)
            mask = np.where(sim_vals != 0)
            nbins = len(mask[0])
            if len(sim_label_list_shallow) > 1:
                try:
                    tmptag = sim_label_list_shallow[counter]
                except Exception:
                    import ipdb

                    ipdb.set_trace()
                tmp = rf"$\chi^2_{tmptag}=$"
                text_chi2 = f"{tmp} {round(chi,1)}/{nbins}"
            else:
                tmp = r"$\chi^2/bins=$"
                text_chi2 = f"{tmp} {round(chi,1)}/{nbins}"
            axs[0][i].annotate(
                text_chi2,
                xy=(0.03, 0.9 - 0.15 * counter),
                size=16,
                xycoords="axes fraction",
            )
            counter += 1
        xlabel = k if k != "m0obs_i" else r"$i_{peak}$"
        if k == "zHD":
            xlabel = "z"
        axs[0][i].set_xlabel(xlabel, fontsize=20)
        if log_scale:
            axs[0][i].set_yscale("log")
    axs[1][0].set_ylabel("# events")
    axs[0][0].set_ylabel("# events")
    axs[1][-1].legend(
        bbox_to_anchor=(1.05, 0.5),
        loc=2,
        borderaxespad=0.0,
        fontsize=16,
        title="Deep fields",
    )
    axs[0][-1].legend(
        bbox_to_anchor=(1.05, 0.5),
        loc=2,
        borderaxespad=0.0,
        fontsize=16,
        title="Shallow fields",
    )
    plt.savefig(f"{path_plots}/hists_sample_sim_Ia{suffix}.png")
    plt.clf()
    del fig


def overplot_salt_distributions_lists_deep_shallow(
    list_df,
    list_labels=["tmp"],
    path_plots="./",
    suffix="",
    sim_scale_factor=150,
    data_color_override=False,
):
    plot_mosaic_histograms_listdf_deep_shallow(
        list_df,
        list_labels=list_labels,
        path_plots=path_plots,
        suffix=suffix,
        norm=1 / sim_scale_factor,
        list_vars_to_plot=["zHD", "c", "x1", "m0obs_i"],
        data_color_override=data_color_override,
        color_list=["red", "maroon"],
    )

    # zHD binned c,x1 together (same as above only formatting)
    xbin = "zHD"
    to_plot = ["c", "x1"]
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    list_df_shallow = [f[f.shallow == True] for f in list_df]
    list_df_deep = [f[f.deep == True] for f in list_df]
    for i, k in enumerate(to_plot):
        # shallow
        plot_errorbar_binned(
            list_df_shallow,
            [f"{l}" for l in list_labels],
            binname=f"{xbin}_bin",
            varx=xbin,
            vary=k,
            axs=axs[i][0],
            sim_scale_factor=sim_scale_factor,
            data_color_override=data_color_override,
            color_list=["darkgrey", "red", "black"],
        )
        # deep
        plot_errorbar_binned(
            list_df_deep,
            [f"{l}" for l in list_labels],
            binname=f"{xbin}_bin",
            varx=xbin,
            vary=k,
            axs=axs[i][1],
            sim_scale_factor=sim_scale_factor,
            data_color_override=data_color_override,
            ignore_y_label=True,
            color_list=["darkgrey", "maroon", "black"],
        )
    axs[0][0].legend(
        loc="best",
        title="Shallow fields",
    )
    axs[0][1].legend(
        loc="best",
        title="Deep fields",
    )
    if xbin == "zHD":
        xlabel = "z"
    else:
        xlabel = xbin
    axs[1][0].set_xlabel(xlabel, fontsize=20)
    axs[1][1].set_xlabel(xlabel, fontsize=20)
    # lims
    axs[0][0].set_ylim(-0.17, 0.17)
    axs[1][0].set_ylim(-2, 2)
    axs[0][1].set_ylim(-0.17, 0.17)
    axs[1][1].set_ylim(-2, 2)
    if suffix != "":
        plt.savefig(
            f"{path_plots}/2ddist_sample_sim_Ia_cx1_{xbin}_{suffix}_deep_shallow.png"
        )
    else:
        plt.savefig(f"{path_plots}/2ddist_sample_sim_Ia_cx1_{xbin}_deep_shallow.png")
    del fig


def plot_scatter_mosaic_retro(
    list_df,
    list_labels,
    path_out="tmp.png",
    print_biases=False,
    fitted_suffix="_retro",
    zspe_suffix="",
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
                    "bins",
                    [round(mean_bins[i], 3) for i in range(len(mean_bins))],
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
                    "%",
                    perc,
                )
                print(
                    f"% median {np.median(perc)}; max {np.max(perc)}; min {np.min(perc)}"
                )
        axs[i].plot(
            [df[varx].min(), df[varx].max()],
            [df[varx].min(), df[varx].max()],
            color="black",
            linewidth=1,
            linestyle="--",
            zorder=10,
        )
        xlabel = "zspe" if var == "z" else f"{var} with zspe"
        ylabel = "fitted z" if var == "z" else f"{var} with fitted z"
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_xlim(lims[0], lims[1])
        axs[i].set_ylim(lims[0], lims[1])

    # axs[i].legend(loc="best", prop={"size": 10})
    plt.savefig(path_out)


def plot_scatter_mosaic(
    list_df,
    labels,
    varx,
    vary,
    path_out="tmp.png",
):
    # scatter
    fig = plt.figure(constrained_layout=True, figsize=(17, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.05)
    axs = gs.subplots(sharex=False, sharey=True)

    for idx_df, df in enumerate(list_df):
        h2d, xedges, yedges, _ = axs[idx_df].hist2d(
            df[varx],
            df[vary],
            bins=30,
            cmap="viridis",
            vmin=1,
        )
        # check distribution for a given x bin
        tmp_bins = xedges
        mean_bins = tmp_bins[:-1] + (tmp_bins[1] - tmp_bins[0]) / 2
        df[f"{varx}_bin"] = pd.cut(df.loc[:, (varx)], tmp_bins, labels=mean_bins)
        result_median = (
            df[[f"{varx}_bin", varx]].groupby(f"{varx}_bin").median()[varx].values
        )
        axs[idx_df].plot(
            [56520, 58167],
            [56520, 58167],
            color="black",
            linewidth=1,
            linestyle="--",
            zorder=10,
        )
        axs[idx_df].plot(
            [56520, 58167],
            [56550, 58197],
            color="black",
            linewidth=1,
            linestyle="--",
            zorder=10,
        )
        xlabel = "Observed peak"
        axs[idx_df].set_xlabel(xlabel)
        axs[idx_df].legend(title=labels[idx_df])

    ylabel = "Trigger"
    axs[0].set_ylabel(ylabel)
    plt.savefig(path_out)


def plot_delta_vs_var(df, varx, vary2, fout, ylabel=None, xlabel=None):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(df[varx], np.zeros(len(df)), linestyle="dashed", color="grey")
    plt.errorbar(
        df[varx],
        df[varx] - df[vary2],
        xerr=df[f"{varx}ERR"],
        fmt="o",
    )
    plt.xlabel(xlabel if xlabel != None else varx)
    plt.ylabel(ylabel if ylabel != None else f"{varx}-{vary2}")
    # plt.title(
    #     f"{np.median(df[varx] - df[vary2]):.3f} \pm {np.std(df[varx] - df[vary2]):.3f} & max: {np.max(df[varx] - df[vary2]):.3f}"
    # )
    plt.savefig(fout)
    del fig


def plot_list_delta_vs_var(
    df_list, varx, vary2, fout, ylabel=None, xlabel=None, labels=None
):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(
        df_list[0][varx], np.zeros(len(df_list[0])), linestyle="dashed", color="grey"
    )
    if labels == None:
        tmp_label = [f"{k}" for k in np.zeros(len(df_list))]
    else:
        tmp_label = labels
    for n, df in enumerate(df_list):
        plt.errorbar(
            df[varx],
            df[varx] - df[vary2],
            # xerr=df[f"{varx}ERR"],
            # yerr=np.sqrt(df[f"{varx}ERR"] ** 2 + df[f"{vary2}ERR"] ** 2),
            color="grey" if n < 1 else ALL_COLORS[n],
            fmt="o",
            label=tmp_label[n],
        )
    plt.xlabel(xlabel if xlabel != None else varx)
    plt.ylabel(ylabel if ylabel != None else f"{varx}-{vary2}")
    # plt.title(
    #     f"{np.median(df[varx] - df[vary2]):.3f} \pm {np.std(df[varx] - df[vary2]):.3f} & max: {np.max(df[varx] - df[vary2]):.3f}"
    # )
    if labels != None:
        plt.legend()
    plt.xlim(0, 1.3)
    plt.ylim(-0.1, 0.1)
    plt.savefig(fout)
    del fig


def plot_probas_set_vs_seed(
    to_plot,
    nameout="./tmp.png",
    xvar="average_probability_set_0",
    set_to_plot=0,
    prefix1="all",
    prefix2="PEAKMJD-2",
):
    plt.clf()
    figure = plt.figure()

    def med(x):
        return x.stack().median()

    def low(x):
        return x.stack().quantile(0.16)

    def high(x):
        return x.stack().quantile(0.84)

    tmp_bins = {"proba": np.linspace(0, 1, 30)}
    mean_bins = (
        tmp_bins["proba"][:-1] + (tmp_bins["proba"][1] - tmp_bins["proba"][0]) / 2
    )
    # bin in av prob set 0
    to_plot["bins"] = pd.cut(
        to_plot.loc[:, (xvar)],
        tmp_bins["proba"],
        labels=mean_bins,
    )

    # all
    cols = [f"{prefix1}_class0_S_{s}" for s in cu.list_seeds_set[set_to_plot]]
    plt.errorbar(
        mean_bins,
        to_plot[cols + ["bins"]].groupby("bins").apply(med),
        color="maroon",
    )
    plt.fill_between(
        mean_bins,
        to_plot[cols + ["bins"]].groupby("bins").apply(low),
        to_plot[cols + ["bins"]].groupby("bins").apply(high),
        color="maroon",
        alpha=0.5,
        zorder=-10,
        label=prefix1,
    )
    # PEAKMJD
    # Beware these may have missing pred values
    # bin in av prob set 0
    to_plot["bins"] = pd.cut(
        to_plot.loc[:, (f"{prefix2}_{xvar}")],
        tmp_bins["proba"],
        labels=mean_bins,
    )
    cols = [f"{prefix2}_class0_S_{s}" for s in cu.list_seeds_set[set_to_plot]]
    tmp = to_plot[cols + ["bins"]].dropna()
    plt.errorbar(
        mean_bins,
        tmp[cols + ["bins"]].groupby("bins").apply(med),
        color="blue",
        linestyle="--",
        label=prefix2,
    )
    plt.fill_between(
        mean_bins,
        tmp[cols + ["bins"]].groupby("bins").apply(low),
        tmp[cols + ["bins"]].groupby("bins").apply(high),
        color="blue",
        alpha=0.5,
        zorder=-20,
    )
    plt.legend()
    plt.xlabel(xvar)
    plt.ylabel(r"$P_{Ia}$")
    plt.savefig(nameout)
    plt.clf()


def plot_hists_prob(df, pathout="./", xvar="PEAKMJD-2_average_probability_set_0"):

    tmp_bins = {"proba": np.linspace(0, 1, 30)}
    mean_bins = (
        tmp_bins["proba"][:-1] + (tmp_bins["proba"][1] - tmp_bins["proba"][0]) / 2
    )
    # bin in av prob set 0
    df["bins"] = pd.cut(
        df.loc[:, (xvar)],
        tmp_bins["proba"],
        labels=mean_bins,
    )

    for b in df["bins"].unique():
        sel = df[df["bins"] == b]
        fig = plt.figure()
        for p in [f"PEAKMJD-2_class0_S_{s}" for s in cu.list_seeds_set[0]]:
            plt.hist(sel[p], histtype="step")
        plt.plot((np.mean(sel[p]) + np.std(sel[p])) * (np.ones(2)), [0, 1000])
        plt.yscale("log")
        plt.xlim(0, 1)
        plt.savefig(f"{pathout}/hist_prob_{np.round(b,2)}.png")


def plot_mosaic_scatter(
    df,
    path_plots="./",
    suffix="",
    list_vars_to_plot=["zHD", "c", "x1"],
):

    plt.clf()
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, len(list_vars_to_plot), hspace=0, wspace=0.05)
    axs = gs.subplots(sharex=False, sharey=True)

    for i, k in enumerate(list_vars_to_plot):
        axs[i].scatter(
            df[k],
            df["average_probability_set_0"],
        )
        axs[i].set_xlabel(k, fontsize=20)
    axs[0].set_ylabel("average_probability_set_0")
    axs[-1].legend(bbox_to_anchor=(1.05, 0.5), loc=2, borderaxespad=0.0, fontsize=16)
    plt.savefig(f"{path_plots}/scatter_{suffix}.png")
    plt.clf()
    del fig


def hist_fup_targets(df_stats_fup, path_plots="./"):
    rows_to_plots = [
        "DES-SN 5-year candidate sample",
        "Photometric sampling and SNR",
        "RNN > 0.001",
        "RNN > 0.5",
    ]
    df_sel = df_stats_fup[df_stats_fup["sample"].isin(rows_to_plots)]
    df_sel["sample_simplified"] = [
        "DES-SN",
        "sampling",
        "SNN>0.001",
        "SNN>0.5",
    ]
    fig = plt.figure(figsize=(12, 8))
    labels = ["without host", "with host", "host<24 mag", "photo Ia M22"]
    for n, k in enumerate(["total", "with host", "<24 mag", "photoIa M22"]):
        plt.bar(
            df_sel["sample_simplified"],
            df_sel[k],
            label=labels[n],
            zorder=n,
            color=ALL_COLORS[n],
        )
    # PSNID
    df_sel = df_stats_fup[
        df_stats_fup["sample"].isin(["PSNID on Photometric sampling and SNR"])
    ]
    for n, k in enumerate(["total", "with host", "<24 mag", "photoIa M22"]):
        plt.bar(
            ["sampl. + PSNID"],
            df_sel[k],
            zorder=n,
            color=ALL_COLORS[n],
            hatch="*",
            edgecolor="white",
        )
    plt.axhline(
        y=7000, linewidth=1, color="k", linestyle="--", label="OzDES follow-up targets"
    )
    plt.legend()
    # plt.yscale("log")
    # plt.xticks(rotation=20)
    plt.ylabel("Number of candidates")
    plt.savefig(f"{path_plots}/hist_fup_targets.png")


def hist_fup_targets_early(
    df_stats_fup,
    path_plots="./",
    subsamples_to_plot=[
        "total maglim<22.7",
        "M22 maglim<22.7",
        "multiseason maglim<22.7",
    ],
    suffix="",
    colors=[],
    log_scale=False,
):
    rows_to_plots = [
        # "DES-SN 5-year candidate sample",
        # "-7<t<20 nflt:1 nights:2 snr:5",
        "SNN>0.1",
        "SNN>0.2",
        "SNN>0.3",
        "SNN>0.4",
        "SNN>0.5",
    ]
    df_sel = df_stats_fup[df_stats_fup["cut"].isin(rows_to_plots)]
    df_sel["sample_simplified"] = [
        # "DES-SN",
        # "sampling",
        "SNN>0.1",
        "SNN>0.2",
        "SNN>0.3",
        "SNN>0.4",
        "SNN>0.5",
    ]
    fig = plt.figure(figsize=(12, 8))
    for n, k in enumerate(subsamples_to_plot):
        plt.bar(
            df_sel["sample_simplified"],
            df_sel[k],
            label=k.replace(" maglim<22.7", ""),
            zorder=n,
            color=colors[n] if len(colors) > 0 else ALL_COLORS[n],
        )

    # plt.axhline(y=7000, linewidth=1, color="k", linestyle="--", label="OzDES lim mag")
    plt.legend()
    if log_scale:
        plt.yscale("log")
    # plt.xticks(rotation=20)
    plt.ylabel("Number of candidates")
    plt.savefig(f"{path_plots}/hist_fup_targets_early{suffix}.png")


def hist_HOSTGAL_MAG_r_vs_REDSHIFT(list_df, list_labels, path_plots="./"):

    list_n = []
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, df in enumerate(list_df):
        whost = df[df["HOSTGAL_MAG_r"] < 40]
        color_to_plot = SAMPLES_COLORS[list_labels[i]]
        print(color_to_plot)
        n, bins, tmp = plt.hist(
            whost["HOSTGAL_MAG_r"],
            histtype="step",
            label=list_labels[i],
            bins=50,
            lw=2,
            color=color_to_plot,
        )
    for i, df in enumerate(list_df):
        ax.hist(
            whost[whost["REDSHIFT_FINAL"] < 0]["HOSTGAL_MAG_r"],
            histtype="step",
            label=f"{list_labels[i]} without host z",
            bins=bins,
            lw=2,
            color=color_to_plot,
            linestyle="dotted",
        )
        list_n.append(max(n))
    ax.axvline(x=24, linestyle="-.", label="follow-up limit", color="black")
    plt.ylabel("# events", fontsize=20)
    plt.xlabel("host r magnitude", fontsize=20)
    plt.legend(loc=2)
    plt.savefig(f"{path_plots}/hist_HOSTGAL_MAG_r_vs_REDSHIFT.png")
