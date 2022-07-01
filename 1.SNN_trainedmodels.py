import glob
import argparse
import pandas as pd
from utils import conf_utils as cu
from utils import data_utils as du
from utils import metric_utils as mu
from utils import logging_utils as lu

pd.options.mode.chained_assignment = None  # default='warn'

"""
Evaluating performance of SNN trained models with diffferent configurations

"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SNN trained models performance")

    parser.add_argument(
        "--path_models26X",
        default="./../snndump_26XBOOSTEDDES",
        type=str,
        help="Path to 26X output",
    )
    parser.add_argument(
        "--path_models1X",
        type=str,
        default=f"./../snndump_1X_NOZ/models/",
        help="Path to models not balanced testset",
    )
    args = parser.parse_args()

    lu.print_green("SUPERNNOVA PERFORMACE WITH SIMULATIONS NO REDSHIFT")

    lu.print_blue("Load", "PREDS 26XB no redshift")
    df_dic_noz = {}
    df_txt_stats_noz = pd.DataFrame(
        columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
    )
    for norm in ["cosmo", "cosmo_quantile", "global"]:
        # exception for global norm
        methods = (
            cu.dic_sel_methods.items()
            if norm != "global"
            else {"predicted_target_S_": "single model"}.items()
        )
        seeds = cu.all_seeds if norm != "global" else cu.list_seeds_set[0]

        df_dic_noz[norm] = du.get_preds_seeds_merge(
            seeds,
            f"{args.path_models26X}/models",
            norm=norm,
            model_suffix=f"_CLF_2_R_none_photometry_DF_1.0_N_{norm}_lstm_64x4_0.05_1024_True_mean",
        )

        list_pred_targets = [
            k for k in df_dic_noz[norm].keys() if "predicted_target" in k
        ]

        # Ensemble methods
        if norm != "global":
            df_dic_noz, list_sets = du.add_ensemble_methods(df_dic_noz, norm)

        for method, desc in methods:
            list_seeds_sets = (
                cu.list_seeds_set[0] if method == "predicted_target_S_" else list_sets
            )

            df_txt_stats_noz = mu.get_multiseed_performance_metrics(
                df_dic_noz[norm],
                key_pred_targ_prefix=method,
                list_seeds=list_seeds_sets,
                df_txt=df_txt_stats_noz,
                dic_prefilled_keywords={
                    "norm": norm,
                    "dataset": "balanced",
                    "method": desc,
                },
            )

    # print(df_txt_stats_noz.to_string(index=False))
    mu.reformatting_tolatex(df_txt_stats_noz, norm_list=["cosmo_quantile", "global"])

    lu.print_blue("Load", "PREDS 1X no redshift (not balanced)")
    df_dic_noz_1X = {}
    df_txt_stats_noz_1X = pd.DataFrame(
        columns=["norm", "dataset", "method", "accuracy", "efficiency", "purity"]
    )
    for norm in ["cosmo_quantile"]:
        # exception for global norm
        methods = cu.dic_sel_methods.items()
        seeds = cu.all_seeds

        df_dic_noz_1X[norm] = du.get_preds_seeds_merge(
            seeds,
            f"{args.path_models1X}",
            norm=norm,
            model_suffix=f"_CLF_2_R_none_photometry_DF_1.0_N_{norm}_lstm_64x4_0.05_1024_True_mean",
        )

        list_pred_targets = [
            k for k in df_dic_noz_1X[norm].keys() if "predicted_target" in k
        ]

        # Ensemble methods
        df_dic_noz_1X, list_sets = du.add_ensemble_methods(df_dic_noz_1X, norm)

        for method, desc in methods:
            list_seeds_sets = (
                cu.list_seeds_set[0] if method == "predicted_target_S_" else list_sets
            )
            df_txt_stats_noz_1X = mu.get_multiseed_performance_metrics(
                df_dic_noz_1X[norm],
                key_pred_targ_prefix=method,
                list_seeds=list_seeds_sets,
                df_txt=df_txt_stats_noz_1X,
                dic_prefilled_keywords={
                    "norm": norm,
                    "dataset": "not balanced",
                    "method": desc,
                },
            )
            df_txt_stats_noz_1X = mu.get_multiseed_performance_metrics(
                df_dic_noz_1X[norm],
                key_pred_targ_prefix=f"PEAKMJD_{method}",
                list_seeds=list_seeds_sets,
                df_txt=df_txt_stats_noz_1X,
                dic_prefilled_keywords={
                    "norm": norm,
                    "dataset": "not balanced",
                    "method": f"PEAKMJD_{desc}",
                },
            )
            df_txt_stats_noz_1X = mu.get_multiseed_performance_metrics(
                df_dic_noz_1X[norm],
                key_pred_targ_prefix=f"PEAKMJD-2_{method}",
                list_seeds=list_seeds_sets,
                df_txt=df_txt_stats_noz_1X,
                dic_prefilled_keywords={
                    "norm": norm,
                    "dataset": "not balanced",
                    "method": f"PEAKMJD-2_{desc}",
                },
            )

    # print(df_txt_stats_noz_1X.to_string(index=False))
    mu.reformatting_tolatex(
        df_txt_stats_noz_1X, norm_list=["cosmo_quantile"], dataset="not balanced"
    )

