"""
Converter to convert parameters in a sampler to physical parameters
needed for the likelihood calculation.
"""
import numpy as np
from scipy import constants as ct
from scipy import stats
import warnings
from astropy import units

from .constants import *


def modpi(phase):
    # from sylvan
    return phase - np.floor(phase / np.pi) * np.pi


def mod2pi(phase):
    # from sylvan
    return phase - np.floor(phase / (2 * np.pi)) * 2 * np.pi


# Compute Solar System Barycenter time tSSB from retarded time at the center of the LISA constellation tL */
# NOTE: depends on the sky position given in SSB parameters */
def tSSBfromLframe(tL, lambdaSSB, betaSSB, t0):
    ConstPhi0 = ConstOmega * t0
    OrbitR = 1.4959787066e11  # AU_SI
    C_SI = 299792458.0
    phase = ConstOmega * tL + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return (
        tL
        + RoC * np.cos(betaSSB) * np.cos(phase)
        - 1.0 / 2 * ConstOmega * pow(RoC * np.cos(betaSSB), 2) * np.sin(2.0 * phase)
    )


# Compute retarded time at the center of the LISA constellation tL from Solar System Barycenter time tSSB */
def tLfromSSBframe(tSSB, lambdaSSB, betaSSB, t0):
    ConstPhi0 = ConstOmega * t0
    OrbitR = 1.4959787066e11  # AU_SI
    C_SI = 299792458.0
    phase = ConstOmega * tSSB + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return tSSB - RoC * np.cos(betaSSB) * np.cos(phase)


source_recycle_guide = {
    "mbh": {"phiRef": 2 * np.pi, "lam": 2 * np.pi, "psi": np.pi},
    "emri": {
        "phi_S": 2 * np.pi,
        "phi_K": 2 * np.pi,
        "phiRef": 2 * np.pi,
        "gamma": 2 * np.pi,
        "psi": 2 * np.pi,
        "alpha": 2 * np.pi,
    },
}


class Converter:
    def __init__(
        self,
        source_type,
        key_order,
        dist_unit="Gpc",
        t0=None,
        transform_frame=None,
        **kwargs
    ):

        variables_to_recycle = source_recycle_guide[source_type].keys()
        recycle_guide = source_recycle_guide[source_type]

        if t0 is not None:
            self.t0 = t0 * YRSID_SI

        self.dist_unit = getattr(units, dist_unit)
        self.dist_conversion = self.dist_unit.to(units.m)

        self.recycles = []
        self.conversions = []
        self.inds_list = []
        self.keys = []
        for i, key_in in enumerate(key_order):

            quant = key_in.split("_")[-1]
            setattr(self, "ind_" + quant, i)

            self.inds_list.append(getattr(self, "ind_" + quant))
            self.keys.append(quant)
            if key_in.split("_")[0] in ["ln", "cos", "sin"]:
                self.conversions.append(
                    [getattr(self, key_in.split("_")[0] + "_convert"), np.array([i])]
                )

            if quant == "distance":
                self.conversions.append([self.convert_distance, np.array([i])])

            if quant in variables_to_recycle:
                wrap_value = recycle_guide[quant]
                if wrap_value == 2 * np.pi:
                    self.recycles.append([self.wrap_2pi, np.array([i])])

                elif wrap_value == np.pi:
                    self.recycles.append([self.wrap_pi, np.array([i])])

                else:
                    raise ValueError("wrap_value must be preprogrammed available")

        # for similar mass binaries
        if "mT" in self.keys and "q" in self.keys:
            self.conversions.append([self.mT_mr, np.array([self.ind_mT, self.ind_q])])

        if transform_frame is not None:
            if t0 is None:
                raise ValueError(
                    "If converting from LISA to SSB frame, need to provide t0."
                )

            if transform_frame == "tLtoSSB":
                self.conversions.append(
                    [
                        self.LISA_to_SSB,
                        np.array(
                            [self.ind_tRef, self.ind_lam, self.ind_beta, self.ind_psi]
                        ),
                    ]
                )

            elif transform_frame == "tSSBtoL":
                self.conversions.append(
                    [
                        self.SSB_to_LISA,
                        np.array(
                            [self.ind_tRef, self.ind_lam, self.ind_beta, self.ind_psi]
                        ),
                    ]
                )

            else:
                raise ValueError(
                    "If transforming frame, must be 'tLtoSSB' or 'tSSBtoL'."
                )

    def ln_convert(self, x):
        return np.exp(x)

    def cos_convert(self, x):
        return np.arccos(x)

    def sin_convert(self, x):
        return np.arcsin(x)

    def mT_mr(self, mT, q):
        m1 = mT / (1 + q)
        m2 = mT * q / (1 + q)

        return m1, m2

    """
    def chi_s_chi_a(self, x):
        chi_s = x[self.ind_chi_s]
        chi_a = x[self.ind_chi_a]
        a1 = chi_s + chi_a
        a2 = chi_s - chi_a
        x[self.ind_chi_s] = a1
        x[self.ind_chi_a] = a2
        return x

    def chi_s_chi_a_m_weight(self, x):
        chi_s = x[self.ind_chi_s_m_weight]
        chi_a = x[self.ind_chi_a_m_weight]
        m1 = x[self.ind_ln_mT]
        m2 = x[self.ind_mr]
        a1 = (chi_s + chi_a) * (m1 + m2) / (2 * m1)
        a2 = (chi_s - chi_a) * (m1 + m2) / (2 * m2)
        x[self.ind_chi_s_m_weight] = a1
        x[self.ind_chi_a_m_weight] = a2
        return x


    def ln_mC_mu(self, x):
        mC = np.exp(x[self.ind_ln_mC])
        mu = x[self.ind_mu]

        m1 = mT/(1+mr)
        m2 = mT*mr/(1+mr)

        x[self.ind_ln_mT] = m1
        x[self.ind_mr] = m2
        return x
    """

    def convert_distance(self, x):
        return x * self.dist_conversion

    def LISA_to_SSB(self, tL, lambdaL, betaL, psiL):
        """
            # from Sylvan
            int ConvertLframeParamsToSSBframe(
              double* tSSB,
              double* lambdaSSB,
              double* betaSSB,
              double* psiSSB,
              const tL,
              const lambdaL,
              const betaL,
              const psiL,
              const LISAconstellation *variant)
            {
        """
        ConstPhi0 = ConstOmega * (self.t0)
        coszeta = np.cos(np.pi / 3.0)
        sinzeta = np.sin(np.pi / 3.0)
        coslambdaL = np.cos(lambdaL)
        sinlambdaL = np.sin(lambdaL)
        cosbetaL = np.cos(betaL)
        sinbetaL = np.sin(betaL)
        cospsiL = np.cos(psiL)
        sinpsiL = np.sin(psiL)
        lambdaSSB_approx = 0.0
        betaSSB_approx = 0.0
        # Initially, approximate alpha using tL instead of tSSB - then iterate */
        tSSB_approx = tL
        for k in range(3):
            alpha = ConstOmega * tSSB_approx + ConstPhi0
            cosalpha = np.cos(alpha)
            sinalpha = np.sin(alpha)
            lambdaSSB_approx = np.arctan2(
                cosalpha * cosalpha * cosbetaL * sinlambdaL
                - sinalpha * sinbetaL * sinzeta
                + cosbetaL * coszeta * sinalpha * sinalpha * sinlambdaL
                - cosalpha * cosbetaL * coslambdaL * sinalpha
                + cosalpha * cosbetaL * coszeta * coslambdaL * sinalpha,
                cosbetaL * coslambdaL * sinalpha * sinalpha
                - cosalpha * sinbetaL * sinzeta
                + cosalpha * cosalpha * cosbetaL * coszeta * coslambdaL
                - cosalpha * cosbetaL * sinalpha * sinlambdaL
                + cosalpha * cosbetaL * coszeta * sinalpha * sinlambdaL,
            )
            betaSSB_approx = np.arcsin(
                coszeta * sinbetaL
                + cosalpha * cosbetaL * coslambdaL * sinzeta
                + cosbetaL * sinalpha * sinzeta * sinlambdaL
            )
            tSSB_approx = tSSBfromLframe(tL, lambdaSSB_approx, betaSSB_approx, self.t0)

        lambdaSSB_approx = lambdaSSB_approx % (2 * np.pi)
        #  /* Polarization */
        psiSSB = modpi(
            psiL
            + np.arctan2(
                cosalpha * sinzeta * sinlambdaL - coslambdaL * sinalpha * sinzeta,
                cosbetaL * coszeta
                - cosalpha * coslambdaL * sinbetaL * sinzeta
                - sinalpha * sinbetaL * sinzeta * sinlambdaL,
            )
        )

        return (tSSB_approx, lambdaSSB_approx, betaSSB_approx, psiSSB)

    # Convert SSB-frame params to L-frame params  from sylvain marsat / john baker
    # NOTE: no transformation of the phase -- approximant-dependence with e.g. EOBNRv2HMROM setting phiRef at fRef, and freedom in definition
    def SSB_to_LISA(self, tSSb, lambdaSSB, betaSSB, psiSSB):

        ConstPhi0 = ConstOmega * (self.t0)
        alpha = 0.0
        cosalpha = 0
        sinalpha = 0.0
        coslambda = 0
        sinlambda = 0.0
        cosbeta = 0.0
        sinbeta = 0.0
        cospsi = 0.0
        sinpsi = 0.0
        coszeta = np.cos(np.pi / 3.0)
        sinzeta = np.sin(np.pi / 3.0)
        coslambda = np.cos(lambdaSSB)
        sinlambda = np.sin(lambdaSSB)
        cosbeta = np.cos(betaSSB)
        sinbeta = np.sin(betaSSB)
        cospsi = np.cos(psiSSB)
        sinpsi = np.sin(psiSSB)
        alpha = ConstOmega * tSSB + ConstPhi0
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        tL = tLfromSSBframe(tSSB, lambdaSSB, betaSSB, self.t0)
        lambdaL = np.arctan2(
            cosalpha * cosalpha * cosbeta * sinlambda
            + sinalpha * sinbeta * sinzeta
            + cosbeta * coszeta * sinalpha * sinalpha * sinlambda
            - cosalpha * cosbeta * coslambda * sinalpha
            + cosalpha * cosbeta * coszeta * coslambda * sinalpha,
            cosalpha * sinbeta * sinzeta
            + cosbeta * coslambda * sinalpha * sinalpha
            + cosalpha * cosalpha * cosbeta * coszeta * coslambda
            - cosalpha * cosbeta * sinalpha * sinlambda
            + cosalpha * cosbeta * coszeta * sinalpha * sinlambda,
        )
        betaL = np.arcsin(
            coszeta * sinbeta
            - cosalpha * cosbeta * coslambda * sinzeta
            - cosbeta * sinalpha * sinzeta * sinlambda
        )
        psiL = modpi(
            psiSSB
            + np.arctan2(
                coslambda * sinalpha * sinzeta - cosalpha * sinzeta * sinlambda,
                cosbeta * coszeta
                + cosalpha * coslambda * sinbeta * sinzeta
                + sinalpha * sinbeta * sinzeta * sinlambda,
            )
        )

        return (tL, lambdaL, betaL, psiL)

    def convert(self, x):
        for func, inds in self.conversions:
            x[inds] = np.asarray(func(*x[inds]))
        return x

    def wrap_2pi(self, x):
        return x % (2 * np.pi)

    def wrap_pi(self, x):
        return x % (np.pi)

        """if x[self.ind_beta] < -np.pi/2 or x[self.ind_beta] > np.pi/2:
            # assumes beta = 0 at ecliptic plane [-pi/2, pi/2]
            x_trans = np.cos(x[self.ind_beta])*np.cos(x[self.ind_lam])
            y_trans = np.cos(x[self.ind_beta])*np.sin(x[self.ind_lam])
            z_trans = np.sin(x[self.ind_beta])

            x[self.ind_lam] = np.arctan2(y_trans, x_trans)
            x[self.ind_beta] = np.arcsin(z_trans/np.sqrt(x_trans**2 + y_trans**2 + z_trans**2))  # check this with eccliptic coordinates
        """

    def recycle(self, x):
        for func, inds in self.recycles:
            x[inds] = np.asarray(func(*x[inds]))
        return x
