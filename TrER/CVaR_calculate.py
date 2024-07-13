import numpy as np, pandas as pd
from utils import wtdquantile, rearrange_cvar, goldsectmax
from IF_calculate import IF

import pandas as pd


def cvar_if(
    data,
    p,
    q,
    mu1_col="mu1",
    mu0_col="mu0",
    A_col="A",
    ipw_col="ipw",
    Y_col="Y",
    tau="tau",
):
    # Intermediate calculations
    difference = data[mu1_col] - data[mu0_col]
    weighted_difference = (
        (2 * data[A_col] - 1)
        * data[ipw_col]
        * (
            data[Y_col]
            - data[A_col] * data[mu1_col]
            - (1 - data[A_col]) * data[mu0_col]
        )
    )
    condition = data[tau] <= q

    # Calculation
    result = q + (difference + weighted_difference - q) * (condition) / p

    return result


def cvar_if_plugin(data, p, q, tau="tau"):
    # Intermediate calculations
    tau_values = data[tau]
    condition = tau_values <= q

    # Calculation
    result = q + (tau_values - q) * condition / p

    return result


def cvar_if_tauate(
    data, p, q, mu1="mu1", mu0="mu0", A="A", ipw="ipw", Y="Y", tau="tau"
):
    # Extract columns from DataFrame
    mu1_values = data[mu1]
    mu0_values = data[mu0]
    A_values = data[A]
    ipw_values = data[ipw]
    Y_values = data[Y]
    tau_values = data[tau]

    # Intermediate calculations
    difference = mu1_values - mu0_values
    weighted_difference = (
        (2 * A_values - 1)
        * ipw_values
        * (Y_values - A_values * mu1_values - (1 - A_values) * mu0_values)
    )
    condition = tau_values <= q

    # Calculation
    result = (
        q
        + (difference + weighted_difference) * (condition / p - 1)
        - q * (condition / p)
    )

    return result


def cvar_if_bbouns_ate1(
    data,
    p,
    q,
    mu1="mu1",
    mu0="mu0",
    A="A",
    ipw="ipw",
    Y="Y",
    tau="tau",
    varsum01="varsum01",
    rho="rho",
    sdprod01="sdprod01",
):
    # Extract columns from DataFrame
    mu1_values = data[mu1]
    mu0_values = data[mu0]
    A_values = data[A]
    ipw_values = data[ipw]
    Y_values = data[Y]
    tau_values = data[tau]
    varsum01_values = data[varsum01]
    rho_values = data[rho]
    sdprod01_values = data[sdprod01]

    # Intermediate calculations

    weighted_difference = (
        (2 * A_values - 1)
        * ipw_values
        * (Y_values - A_values * mu1_values - (1 - A_values) * mu0_values)
    )

    sqrt_term = np.sqrt(
        (tau_values - q) ** 2 + varsum01_values - 2 * rho_values * sdprod01_values
    )

    # Calculation
    term1 = -weighted_difference
    term2 = (tau_values - q - sqrt_term) / (2 * p)
    term3 = (1 - (tau_values - q) / sqrt_term) * weighted_difference / (2 * p)

    result = term1 + q + term2 + term3

    return result


def cvar_bbound_mate(
    data, p, q, b, mu1="mu1", mu0="mu0", A="A", ipw="ipw", Y="Y", tau="tau"
):
    # Extract columns from DataFrame
    mu1_values = data[mu1]
    mu0_values = data[mu0]
    A_values = data[A]
    ipw_values = data[ipw]
    Y_values = data[Y]
    tau_values = data[tau]

    # Intermediate calculations
    weighted_difference = (
        (2 * A_values - 1)
        * ipw_values
        * (Y_values - A_values * mu1_values - (1 - A_values) * mu0_values)
    )
    mu = mu1_values - mu0_values
    condition = tau_values - b <= q

    # Calculation
    result = (
        -(mu + weighted_difference)
        + q
        + (mu + weighted_difference - q - b) * condition / (2 * p)
        + (mu + weighted_difference - q + b) * condition / (2 * p)
    )

    return result


def cvar_calculate(data, p, tau="tau", sw="sw", method_if=cvar_if, b=None):
    # Extract columns from DataFrame
    tau_ref = data[tau]
    sw_ref = data[sw]

    # Calculate q using weighted quantile
    q = wtdquantile(tau_ref, sw_ref, p)

    # Calculate IF using the specified method_if function
    if b is not None:
        IF = cvar_bbound_mate(data, p, q, b, tau=tau)
    else:
        IF = method_if(data, p, q, tau=tau)

    # Calculate cvar and CVaR
    cvar = IF * sw_ref
    CVaR = np.nanmean(cvar)
    CVaR_se = np.nanstd(cvar) / np.sqrt(data.shape[0])

    # Return results as a DataFrame (assuming you want to return results in a structured format)
    result = pd.DataFrame({"CVaR": [CVaR], "CVaR_se": [CVaR_se], "p": [p]})

    if b is not None:
        result["b"] = [b]

    return result


def IF_bbound_mate(
    q,
    p,
    b,
    data,
    mu1_col="mu1",
    mu0_col="mu0",
    A_col="A",
    ipw_col="ipw",
    Y_col="Y",
    tau_col="tau",
):
    mu1 = data[mu1_col]
    mu0 = data[mu0_col]
    A = data[A_col]
    ipw = data[ipw_col]
    Y = data[Y_col]
    tau = data[tau_col]

    weighted_difference = (2 * A - 1) * ipw * (Y - A * mu1 - (1 - A) * mu0)
    mu = mu1 - mu0
    condition = tau - b <= q

    result = (
        -(mu + weighted_difference)
        + q
        + (mu + weighted_difference - q - b) * condition / (2 * p)
        + (mu + weighted_difference - q + b) * condition / (2 * p)
    )

    return result


def cvar_bbound_mate(data, ps, bs, tau_col="tau", sw_col="sw", sort_cvar=True):
    tau = data[tau_col]
    sw = data[sw_col]

    sw1 = np.concatenate((sw, sw))
    results = pd.DataFrame()

    for b in bs:
        tau1 = np.concatenate((tau + b, tau - b))
        for p in ps:
            q = wtdquantile(tau1, sw1, p)
            IF = IF_bbound_mate(q, p, b, data)
            cvar = IF * sw
            CVaR = np.nanmean(cvar)
            CVaR_se = np.nanstd(cvar) / np.sqrt(len(sw))
            result = pd.DataFrame({"CVaR": [CVaR], "CVaR_se": [CVaR_se], "p": [p]})

            if b is not None:
                result["b"] = [b]

            results = pd.concat((results, result))
    # results = results.round(2)
    if sort_cvar:
        n_results = results.copy()
        n_results["CVaR"] = n_results.groupby("b")["CVaR"].transform(rearrange_cvar)
        return n_results
    return results


def IF_cvar_bbouns_ate(
    data: pd.DataFrame,
    p: float,
    q: float,
    rho: float,
    mu1_col="mu1",
    mu0_col="mu0",
    A_col="A",
    ipw_col="ipw",
    Y_col="Y",
    tau_col="tau",
    varsum01_col="varsum01",
    sdprod01_col="sdprod01",
):
    # Extraer columnas del DataFrame
    mu1 = data[mu1_col]
    mu0 = data[mu0_col]
    A = data[A_col]
    ipw = data[ipw_col]
    Y = data[Y_col]
    tau = data[tau_col]
    varsum01 = data[varsum01_col]
    sdprod01 = data[sdprod01_col]

    # Cálculos intermedios

    weighted_difference = (2 * A - 1) * ipw * (Y - A * mu1 - (1 - A) * mu0)
    sqrt_term = np.sqrt((tau - q) ** 2 + varsum01 - 2 * rho * sdprod01)

    # Cálculo final
    term1 = -weighted_difference
    term2 = (tau - q - sqrt_term) / (2 * p)
    term3 = (1 - (tau - q) / sqrt_term) * weighted_difference / (2 * p)

    result = term1 + q + term2 + term3
    # if
    return result


class CVaR:
    def __init__(
        self,
        ps: np.array,
        tau_col: str = "tau",
        mu1_col: str = "mu1",
        mu0_col: str = "mu0",
        A_col: str = "A",
        ipw_col: str = "ipw",
        Y_col: str = "Y",
        sw_col: str = "sw",
        bbound_varsum01: str = "varsum01",
        bbound_rho: str = "rho",
        bbound_sdprod01: str = "sdprod01",
        data: pd.DataFrame = None,
        **kwargs
    ) -> None:
        cols = [tau_col, mu1_col, mu0_col, A_col, ipw_col, Y_col, sw_col]
        cols_bound = [bbound_varsum01, bbound_rho, bbound_sdprod01]

        if data is not None:
            data = data[cols].to_numpy()
            # data_bbound = data[cols_bound].to_numpy()
            tau, mu1, mu0, A, ipw, Y, sw = [data[:, i] for i in range(len(cols))]
            # varsum01, rho, sdprod01 = [
            #     data_bbound[:, i] for i in range(len(cols_bound))
            # ]
            ref_IF = IF(tau, mu1, mu0, A, ipw, Y)
            qs = [wtdquantile(Y, sw, p) for p in ps]
        else:
            ref_IF = IF(**kwargs)
            sw = kwargs["sw"]
            qs = [wtdquantile(kwargs["Y"], kwargs["sw"], p) for p in ps]

        print(qs)

        self.ref_IF = ref_IF
        self.sw = sw
        self.qs = qs

        IF_base = [ref_IF.base(p, q) for p, q in zip(ps, qs)]
        IF_plugin = [ref_IF.plugin(p, q) for p, q in zip(ps, qs)]
        IF_tau_ate = [ref_IF.tau_ate(p, q) for p, q in zip(ps, qs)]

        self.CVaR_base = [
            self.summary_cvar(IF_base_i, p) for IF_base_i, p in zip(IF_base, ps)
        ]
        # self.CVaR_plugin = self.summary_cvar(IF_plugin)
        # self.CVaR_tau_ate = self.summary_cvar(IF_tau_ate)

    def summary_cvar(self, IF_result: np.array, p):
        cvar = IF_result * self.sw
        cvar_mean = np.nanmean(cvar)
        cvar_std = np.nanstd(cvar) / np.sqrt(len(IF_result))
        return cvar_mean, cvar_std, p
