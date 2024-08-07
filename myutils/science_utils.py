import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.295)


def dist_mu(redshift):
    mu = cosmo.distmod(redshift)

    return mu.value


def distance_modulus(df, suffix=None):
    """Add distance modulus
    Args:
        df (DataFrame): with SALT2 fitted features
        suffix (str): variable suffix
    Returns:
        df (DataFrame): with distance modulus computed
    """

    # SNIa parameters
    Mb = 19.365
    alpha = 0.144  # from sim
    beta = 3.1
    # Add distance modulus to this Data Frame
    if suffix:
        df[f"mu_{suffix}"] = (
            np.array(df[f"mB_{suffix}"]) + Mb + np.array(alpha * df[f"x1_{suffix}"]) - np.array(beta * df[f"c_{suffix}"])
        )
        df[f"delmu_{suffix}"] = df[f"mu_{suffix}"].values - dist_mu(df[f"zHD_{suffix}"].values.astype(float))
        # assuming theoretical mu nor alpha, beta, abs mag have errors
        df[f"delmu_err_{suffix}"] = (
            np.array(df[f"mBERR_{suffix}"])
            + np.array(alpha * df[f"x1ERR_{suffix}"])
            - np.array(beta * df[f"cERR_{suffix}"])
        )
    else:
        df["mu"] = (
            np.array(df["mB"]) + Mb + np.array(alpha * df["x1"]) - np.array(beta * df["c"])
        )
        df["delmu"] = df["mu"].values - dist_mu(df["zHD"].values.astype(float))
        # assuming theoretical mu nor alpha, beta, abs mag have errors
        df["delmu_err"] = (
            np.array(df["mBERR"])
            + np.array(alpha * df["x1ERR"])
            - np.array(beta * df["cERR"])
        )
    return df


def apply_JLA_cut(df):
    """
    can includeg redshift cut (different from with host zspe)
    """
    cut_salt_JLA = (
        (df.x1 > -3)
        & (df.x1 < 3)
        & (df.c > -0.3)
        & (df.c < 0.3)
        & (df.FITPROB > 0.001)
        & (df.x1ERR < 1)
        & (df.PKMJDERR < 2)
        # & (df.zHD > 0.2)
        # & (df.zHD < 1.2)
    )
    df_JLA = df[cut_salt_JLA]

    return df_JLA
