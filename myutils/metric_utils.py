import numpy as np
import pandas as pd
from sklearn import metrics
from . import logging_utils as lu
from myutils import conf_utils as cu

deep_fields = ["X3", "C3"]
shallow_fields = ["X1", "X2", "C1", "C2", "E1", "E2", "S1", "S2"]


def performance_metrics(
    df,
    sample_target=0,
    key_pred_targ="predicted_target",
    compute_auc=True,
    print_metrics=False,
):
    """Get performance metrics
    AUC: only valid for binomial classification, input proba of highest label class.

    Args:
        df (pandas.DataFrame) (str): with columns [target, predicted_target, class1]
        (optional) sample_target (str): for SNIa sample default is target 0
        key_pred_targ (str): column name of predicted target to be used
        print_metrics (Boolean): if metrics to be printed

    Returns:
        accuracy, auc, purity, efficiency,truepositivefraction
    """
    n_targets = len(np.unique(df["target"]))

    # Accuracy & AUC
    df = df[~df[key_pred_targ].isna()]
    balancedaccuracy = metrics.balanced_accuracy_score(
        df["target"].astype(int), df[key_pred_targ].astype(int)
    )
    balancedaccuracy = balancedaccuracy * 100

    accuracy = metrics.accuracy_score(
        df["target"].astype(int), df[key_pred_targ].astype(int)
    )
    accuracy = accuracy * 100
    if n_targets == 2 and compute_auc:  # valid for biclass only
        auc = round(metrics.roc_auc_score(df["target"], df["all_class1"]), 4)
    else:
        auc = 0.0

    SNe_Ia = df[df["target"] == sample_target]
    SNe_CC = df[df["target"] != sample_target]
    TP = len(SNe_Ia[SNe_Ia[key_pred_targ] == sample_target])
    FP = len(SNe_CC[SNe_CC[key_pred_targ] == sample_target])

    P = len(SNe_Ia)
    N = len(SNe_CC)

    truepositivefraction = P / (P + N) if (P + N) > 0 else 0
    purity = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    efficiency = 100.0 * TP / P if P > 0 else 0

    if print_metrics:
        print(
            f"Accuracy: {balancedaccuracy} | Purity: {purity} | Efficiency {efficiency}"
        )

    return accuracy, balancedaccuracy, auc, purity, efficiency, truepositivefraction


def txt_meanstd_w_significant_figures(arr):
    # default 2
    decimals = 2
    if round(np.array(arr).std(), decimals) == 0.0:
        decimals = 3
        if round(np.array(arr).std(), 3) == 0.0:
            decimals = 4
    txt = f"${round(np.array(arr).mean(),decimals)} pm {round(np.array(arr).std(),decimals)}$"

    return txt


def get_multiseed_performance_metrics(
    df,
    key_pred_targ_prefix="predicted_target_S_",
    list_seeds=[0],
    df_txt=None,
    dic_prefilled_keywords=None,
):
    """ """
    # in case a seed is missing
    list_seeds_updated = [
        k for k in list_seeds if f"{key_pred_targ_prefix}{k}" in df.keys()
    ]
    if not set(list_seeds) == set(list_seeds_updated):
        lu.print_red(
            "Missing seed", [k for k in list_seeds if k not in list_seeds_updated]
        )
        lu.print_red("processing with missing seed", dic_prefilled_keywords)
        list_seeds = list_seeds_updated

    balacc_dic = []
    acc_dic = []
    eff_dic = []
    pur_dic = []
    for seed in list_seeds:
        (
            accuracy,
            balancedaccuracy,
            auc,
            purity,
            efficiency,
            truepositivefraction,
        ) = performance_metrics(
            df,
            key_pred_targ=f"{key_pred_targ_prefix}{seed}",
            compute_auc=False,
        )
        balacc_dic.append(balancedaccuracy)
        acc_dic.append(accuracy)
        eff_dic.append(efficiency)
        pur_dic.append(purity)

    # increase decimals in case missing significant figures in accuracy
    txt_acc = txt_meanstd_w_significant_figures(np.array(acc_dic))
    txt_balacc = txt_meanstd_w_significant_figures(np.array(balacc_dic))
    txt_eff = txt_meanstd_w_significant_figures(np.array(eff_dic))
    txt_pur = txt_meanstd_w_significant_figures(np.array(pur_dic))

    if isinstance(df_txt, pd.DataFrame):
        dic_prefilled_keywords["notbalanced_accuracy"] = txt_acc
        dic_prefilled_keywords["accuracy"] = txt_balacc
        dic_prefilled_keywords["efficiency"] = txt_eff
        dic_prefilled_keywords["purity"] = txt_pur
        df_txt = df_txt.append(dic_prefilled_keywords, ignore_index=True)
        return df_txt
    else:
        print(
            f"Balanced Accuracy: {txt_balacc} Notbalanced accuracy: {txt_acc} Efficiency: {txt_eff} Purity: {txt_pur} "
        )


def reformatting_tolatex(df, norm_list=["cosmo", "cosmo_quantile"], dataset="balanced"):
    tmp = {}
    for norm in norm_list:
        tmp = df[(df.dataset == dataset) & (df.norm == norm)][
            ["method", "accuracy", "efficiency", "purity"]
        ]
        print("\multicolumn", {tmp.shape[-1]}, "{c}{", norm, "}\\\\")
        print("\hline")
        print(tmp.to_latex(index=False))


def cuts_deep_shallow(df_sel, photoIa_wz_JLA, df_stats=pd.DataFrame(), cut=""):

    if len(df_stats) == 0:
        df_stats = pd.DataFrame(
            columns=[
                "cut",
                "shallow selected",
                "shallow spec Ia",
                "deep selected",
                "deep spec Ia",
                "total selected",
                "total spec Ia",
            ]
        )

    # determine shallow or deep
    df_shallow = df_sel[df_sel["FIELD"].isin(shallow_fields)]
    df_deep = df_sel[df_sel["FIELD"].isin(deep_fields)]

    dict_t = {}
    dict_t["cut"] = cut
    dict_t["shallow selected"] = len(df_shallow.SNID.unique())
    dict_t["shallow spec Ia"] = len(
        df_shallow[df_shallow.SNTYPE.isin(cu.spec_tags["Ia"])].SNID.unique()
    )
    dict_t["deep selected"] = len(df_deep.SNID.unique())
    dict_t["deep spec Ia"] = len(
        df_deep[df_deep.SNTYPE.isin(cu.spec_tags["Ia"])].SNID.unique()
    )
    dict_t["total selected"] = len(df_sel.SNID.unique())
    dict_t["total spec Ia"] = len(
        df_sel[df_sel.SNTYPE.isin(cu.spec_tags["Ia"])].SNID.unique()
    )
    dict_t["total photo Ia M22"] = len(
        df_sel[df_sel.SNID.isin(photoIa_wz_JLA.SNID.values)].SNID.unique()
    )
    df_stats = df_stats.append(dict_t, ignore_index=True)

    return df_stats


def fup_hostgals_stats(
    df,
    sngals,
    photoIa_wz_JLA,
    df_stats=pd.DataFrame(),
    sample="sample",
    verbose=False,
):
    """
    returns:
        df_stats (pd.DataFrame):
    """

    dict_t = {}
    dict_t["sample"] = sample

    # total selected
    # same as df_stats without follow-up
    dict_t["total"] = len(df)

    # with host
    df_whostmag = df[df["HOSTGAL_MAG_r"] < 40]
    dict_t["with host"] = len(df_whostmag)

    # without redshift
    dict_t["without redshift"] = len(df[df["REDSHIFT_FINAL"] < 0])

    # hosts that could be followed-up with AAT (mag<24)
    to_fup_24 = df_whostmag[df_whostmag["HOSTGAL_MAG_r"] < 24]
    dict_t["<24 mag"] = len(to_fup_24)

    #
    to_fup_24_nozspe = df_whostmag[
        (df_whostmag["HOSTGAL_MAG_r"] < 24) & (df_whostmag["REDSHIFT_FINAL"] < 0)
    ]
    dict_t["<24 mag and no zspe"] = len(to_fup_24_nozspe)

    # OzDES flags
    tmp = pd.merge(to_fup_24_nozspe, sngals, on="SNID")
    aat = tmp[tmp.SPECZ_CATALOG == "DES_AAOmega"]
    aat["SPECZ_FLAG"] = aat["SPECZ_FLAG"].astype(float)
    if verbose:
        lu.print_yellow("TO FUP OzDES FLAGS")
        print(aat.groupby("SPECZ_FLAG").count()["SNID"])
        print(f"QOP 6 (stars!) {aat[aat.SPECZ_FLAG==6]['IAUC'].values}")

    for ind in np.arange(1.0, 6.0, 1.0):
        if ind in aat["SPECZ_FLAG"].unique():
            dict_t[f"OzDES QOP {int(ind)}"] = int(
                aat.groupby("SPECZ_FLAG").count()["SNID"][ind]
            )
        else:
            dict_t[f"OzDES QOP {int(ind)}"] = 0

    # stats per year
    for y, year_str in enumerate(["DES13", "DES14", "DES15", "DES16", "DES17"]):
        per_year = aat[aat["IAUC"].str.contains(year_str)]
        n_year = len(per_year)
        dict_t[f"Y{y+1}"] = n_year
        if verbose:
            print(f"{year_str} {n_year} = {round(n_year*100/len(aat),2)}%")

    dict_t["specIa"] = len(df[df.SNTYPE.isin(cu.spec_tags["Ia"])])
    dict_t["photoIa M22"] = len(df[df.SNID.isin(photoIa_wz_JLA.SNID.values)])

    df_stats = df_stats.append(dict_t, ignore_index=True)

    cols_int = [k for k in df_stats.keys() if k != "sample"]
    df_stats[cols_int] = df_stats[cols_int].astype(int)

    return df_stats


def cuts_deep_shallow_eventmag(
    df_sel,
    photoIa_wz_JLA,
    df_photo,
    df_stats=pd.DataFrame(),
    cut="",
    return_extra_df=False,
):
    """Stats for selection w. deep and shallow fields

    Args:
        df_sel (pd.DataFrame): _description_
        photoIa_wz_JLA (pd.DataFrame): _description_
        df_photo (pd.DataFrame): _description_
        df_stats (pd.DataFrame, optional): _description_. Defaults to pd.DataFrame().
        cut (str, optional): _description_. Defaults to "".
        return_extra_df (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # multiseason
    single_seasons = [1.0, 2.0, 4.0, 8.0, 16.0]

    if len(df_stats) == 0:
        df_stats = pd.DataFrame(
            columns=[
                "cut",
                "shallow selected",
                "shallow spec Ia",
                "deep selected",
                "deep spec Ia",
                "total selected",
                "total spec Ia",
            ]
        )

    # determine shallow or deep
    df_shallow = df_sel[df_sel.IAUC.str.contains("|".join(shallow_fields))]
    df_deep = df_sel[df_sel.IAUC.str.contains("|".join(deep_fields))]

    # magnitude cuts
    df_photo["mag"] = 27.5 - 2.5 * np.log10(df_photo["FLUXCAL"].values)
    tmp = df_photo.groupby(by="SNID")["mag"].min()
    df_tmp = pd.DataFrame()
    df_tmp["SNID"] = tmp.index
    df_tmp["minmag"] = tmp.values

    # Sample 1.a Smith+2020  mag i|r < 21.5
    df_tmp_sel = df_tmp[(df_tmp.minmag < 21.5)]
    maglimSNID = df_tmp_sel.SNID.values
    mask_maglimSNID = df_sel.SNID.isin(maglimSNID)

    # Sample 1.b Smith+2020  mag i|r < 22.7 (OzDES)
    df_tmp_sel = df_tmp[(df_tmp.minmag < 22.7)]
    maglimOzDES = df_tmp_sel.SNID.values
    mask_maglimOzDES = df_sel.SNID.isin(maglimOzDES)

    # Sample 2. Smith+2020
    df_tmp_sel = df_sel[(df_sel.HOSTGAL_MAG_r > 24)]
    maglimFainthosts = df_tmp_sel.SNID.values
    mask_maglimFainthosts = df_sel.SNID.isin(maglimFainthosts)

    # TiDES maglim sample
    df_tmp_sel2 = df_tmp[df_tmp.minmag < 22.5]
    maglimSNIDTiDES = df_tmp_sel2.SNID.values
    mask_maglimSNIDTiDES = df_sel.SNID.isin(maglimSNIDTiDES)
    # extending 1 mag higher as AAT we did for fup to cover the complete
    df_tmp_sel3 = df_tmp[df_tmp.minmag < 23.5]
    maglimSNIDTiDES1 = df_tmp_sel3.SNID.values
    mask_maglimTiDES1 = df_sel.SNID.isin(maglimSNIDTiDES1)

    # masks
    mask_specIa = df_sel.SNTYPE.isin(cu.spec_tags["Ia"])
    mask_M22 = df_sel.SNID.isin(photoIa_wz_JLA.SNID.values)
    mask_multiseason = ~df_sel["PRIVATE(DES_transient_status)"].isin(single_seasons)

    dict_t = {}
    dict_t["cut"] = cut
    dict_t["shallow selected"] = len(df_shallow)
    dict_t["shallow spec Ia"] = len(
        df_shallow[df_shallow.SNTYPE.isin(cu.spec_tags["Ia"])]
    )
    dict_t["deep selected"] = len(df_deep)
    dict_t["deep spec Ia"] = len(df_deep[df_deep.SNTYPE.isin(cu.spec_tags["Ia"])])

    # total
    dict_t["total selected"] = len(df_sel)
    dict_t["total spec Ia"] = len(df_sel[mask_specIa])
    dict_t["total spec nonnormIa"] = len(
        df_sel[df_sel.SNTYPE.isin(cu.spec_tags["nonnormIa"])]
    )
    dict_t["total spec nonIa"] = len(df_sel[df_sel.SNTYPE.isin(cu.spec_tags["nonIa"])])
    dict_t["total spec nonSN"] = len(df_sel[df_sel.SNTYPE.isin(cu.spec_tags["nonSN"])])
    dict_t["total photo Ia M22"] = len(df_sel[mask_M22])

    for name_mask, this_mask in zip(
        ["21.5", "22.7", "24", "22.5", "23.5"],
        [
            mask_maglimSNID,
            mask_maglimOzDES,
            mask_maglimFainthosts,
            mask_maglimSNIDTiDES,
            mask_maglimTiDES1,
        ],
    ):
        dict_t[f"total maglim<{name_mask}"] = len(df_sel[this_mask])
        dict_t[f"specIa maglim<{name_mask}"] = len(df_sel[(this_mask) & (mask_specIa)])
        dict_t[f"nonIa maglim<{name_mask}"] = len(
            df_sel[(this_mask) & (df_sel.SNTYPE.isin(cu.spec_tags["nonIa"]))]
        )
        dict_t[f"M22 maglim<{name_mask}"] = len(df_sel[(mask_M22) & (this_mask)])
        dict_t[f"multiseason maglim<{name_mask}"] = len(
            df_sel[(mask_multiseason) & (this_mask)]
        )

    # general
    dict_t["multiseason"] = len(df_sel[mask_multiseason])
    dict_t["unights"] = df_sel["unights"].median()
    dict_t["unights_std"] = df_sel["unights"].std()
    dict_t["uflt"] = df_sel["uflt"].median()
    dict_t["detections"] = df_sel["detections"].median()
    dict_t["detections_std"] = df_sel["detections"].std()

    df_stats = pd.concat(
        [df_stats, pd.DataFrame.from_records([dict_t])], ignore_index=True
    )

    cols_to_int = [k for k in df_stats.keys() if k != "cut"]
    df_stats[cols_to_int] = df_stats[cols_to_int].astype(int)

    if return_extra_df:
        return df_stats, df_tmp
    else:
        return df_stats
