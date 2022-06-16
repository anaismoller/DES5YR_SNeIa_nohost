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
    args = parser.parse_args()

    #
    # NO REDSHIFT information
    #
    lu.print_green("SUPERNNOVA PERFORMACE WITH SIMULATIONS NO REDSHIFT")
    lu.print_blue("Load", "PREDS 26XB no redshift")
    tmp_pred_dic = {}
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
    mu.reformatting_tolatex(
        df_txt_stats_noz, norm_list=["cosmo", "cosmo_quantile", "global"]
    )

