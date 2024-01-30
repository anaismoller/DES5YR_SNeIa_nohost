import glob
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
        "--path_sim_fits",
        type=str,
        default=f"{DES}/DES5YR/snndump_1X_NOZ/2_LCFIT/",
        help="Path to simulation fits without host-efficiency SALT2 fits",
    )

    parser.add_argument(
        "--path_sim_headers",
        type=str,
        default=f"{DES}/DES5YR/snndump_1X_NOZ/1_SIM/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES",
        help="Path to simulation headers without host-efficiency SALT2 fits",
    )

    # Init
    args = parser.parse_args()
    path_dump = args.path_dump
    path_sim_headers = args.path_sim_headers

    path_plots = f"{path_dump}/plots_sample/"
    os.makedirs(path_plots, exist_ok=True)

    # Load simulations
    lu.print_blue("Loading metadata sims")
    dic_sim_df = {}
    for seed_tmp in np.arange(1, 31, 1):
        seed = seed_tmp if seed_tmp > 9 else f"0{seed_tmp}"
        list_files = glob.glob(f"{args.path_sim_headers}/*{seed}_HEAD.FITS*")
        df_list = []
        for fil in list_files:
            df_list.append(du.read_header_fits(fil, drop_separators=True))
        df = pd.concat(df_list)
        df["IAUC"] = df["IAUC"].str.decode("utf-8")
        dic_sim_df[seed] = df

    # Load fits
    # single file for all seeds
    sim_fits = du.load_salt_fits(
        f"{args.path_sim_fits}/JLA_1XDES/output/PIP_AM_DES5YR_SIMS_TEST_NOZ_SMALL_1XDES/FITOPT000.FITRES.gz"
    )

    # for each individual simulation & fit
    # gather # SN & #SNIa
    nSN_list = []
    nSNIa_list = []
    nSNnonIa_list = []
    nSN_wfit_list = []
    nSNIa_wfit_list = []
    nSN_wfitJLA_list = []
    nSNIa_wfitJLA_list = []
    nSNnonIa_wfitJLA_list = []
    for k, v in dic_sim_df.items():
        # sim
        nSN_list.append(len(v))
        nSNIa_list.append(len(v[v["SNTYPE"].isin(cu.spec_tags["Ia"])]))
        nSNnonIa_list.append(len(v[~v["SNTYPE"].isin(cu.spec_tags["Ia"])]))
        # fits
        sel_sim_fits = sim_fits[sim_fits.SNID.isin(v.SNID.values)]
        nSN_wfit_list.append(len(sel_sim_fits))
        nSNIa_wfit_list.append(
            len(sel_sim_fits[sel_sim_fits["SNTYPE"].isin(cu.spec_tags["Ia"])])
        )
        # JLA
        sel_sim_fits_JLA = su.apply_JLA_cut(sel_sim_fits)
        nSN_wfitJLA_list.append(len(sel_sim_fits_JLA))
        nSNIa_wfitJLA_list.append(
            len(sel_sim_fits_JLA[sel_sim_fits_JLA["SNTYPE"].isin(cu.spec_tags["Ia"])])
        )
        nSNnonIa_wfitJLA_list.append(
            len(sel_sim_fits_JLA[~sel_sim_fits_JLA["SNTYPE"].isin(cu.spec_tags["Ia"])])
        )
    print(f"# seeds {len(nSN_list)}")
    print(
        f"#SN expected {int(np.mean(np.array(nSN_list)))} +- {int(np.std(np.array(nSN_list)))}"
    )
    print(
        f"#SNIa expected {int(np.mean(np.array(nSNIa_list)))} +- {int(np.std(np.array(nSNIa_list)))}"
    )
    print(
        f"#SNIa cosmology-quality expected {int(np.mean(np.array(nSNIa_wfitJLA_list)))} +- {int(np.std(np.array(nSNIa_wfitJLA_list)))}"
    )
    print(
        f"#SNe non Ia expected {int(np.mean(np.array(nSNnonIa_list)))} +- {int(np.std(np.array(nSNnonIa_list)))}"
    )
    print(
        f"#SNe non Ia passing cosmology-quality expected {int(np.mean(np.array(nSNnonIa_wfitJLA_list)))} +- {np.std(np.array(nSNnonIa_wfitJLA_list))}"
    )
