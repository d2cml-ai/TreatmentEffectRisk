import numpy as np, pandas as pd
from utils import wtdquantile
from IF_calculate import IF


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
            data_bbound = data[cols_bound].to_numpy()
            tau, mu1, mu0, A, ipw, Y, sw = [data[:, i] for i in range(len(cols))]
            varsum01, rho, sdprod01 = [
                data_bbound[:, i] for i in range(len(cols_bound))
            ]
            ref_IF = IF(tau, mu1, mu0, A, ipw, Y)
            qs = [wtdquantile(Y, sw, p) for p in ps]
        else:
            ref_IF = IF(**kwargs)
            sw = kwargs["sw"]
            qs = [wtdquantile(kwargs["Y"], kwargs["sw"], p) for p in ps]

        self.ref_IF = ref_IF
        self.sw = sw
        self.qs = qs

        IF_base = [ref_IF.base(p, q) for p, q in zip(ps, qs)]
        IF_plugin = [ref_IF.plugin(p, q) for p, q in zip(ps, qs)]
        IF_tau_ate = [ref_IF.tau_ate(p, q) for p, q in zip(ps, qs)]

    def summary_cvar(self, IF_result: np.array):
        cvar = IF_result * self.sw
        cvar_mean = np.nanmean(cvar)
        cvar_std = np.nanstd(cvar) / np.sqrt(len(IF_result))
        return cvar_mean, cvar_std
