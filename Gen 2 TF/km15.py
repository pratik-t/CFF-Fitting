import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import pi, sqrt, pow

 # KM15 parameters
nval = 1.35
pval = 1.
nsea = 1.5
rsea = 1.
psea = 2.
bsea = 4.6
Mval = 0.789
rval = 0.918
bval = 0.4
C0 = 2.768
Msub = 1.204
Mtval = 3.993
rtval = 0.881
btval = 0.4
ntval = 0.6
Msea = sqrt(0.482)
rpi = 2.646
Mpi = 4.

def ModKM15_CFFs(QQ, xB, t, k=0.0):
    alpha_val = 0.43 + 0.85 * t
    alpha_sea = 1.13 + 0.15 * t
    Ct = C0 / pow(1. - t / Msub / Msub, 2.)
    xi = xB / (2. - xB)

    def fHval(x):
        return (nval * rval) / (1. + x) * pow((2. * x) / (1. + x), -alpha_val) * \
               pow((1. - x) / (1. + x), bval) * \
                1./ pow(1. - ((1. - x) / (1. + x)) * (t / Mval / Mval), pval)

    def fHsea(x):
        return (nsea * rsea) / (1. + x) * pow((2. * x) / (1. + x), -alpha_sea) * \
               pow((1. - x) / (1. + x), bsea) * \
                1./ pow(1. - ((1. - x) / (1. + x)) * (t / Msea / Msea), psea)

    def fHtval(x):
        return (ntval * rtval) / (1. + x) * pow((2. * x) / (1. + x), -alpha_val) * \
               pow((1. - x) / (1. + x), btval) * \
                1./ (1. - ((1. - x) / (1. + x)) * (t / Mtval / Mtval))

    def fImH(x):
        return pi * ((2. * (4. / 9.) + 1. / 9.) * fHval(x) + 2. / 9. * fHsea(x))

    def fImHt(x):
        return pi * (2. * (4. / 9.) + 1. / 9.) * fHtval(x)

    def fPV_ReH(x):
        return -2. * x / (x + xi) * fImH(x)

    def fPV_ReHt(x):
        return -2. * xi / (x + xi) * fImHt(x)

    # Principal value integrals
    DRReH  = quad(fPV_ReH, 1e-10, 1.0, weight='cauchy', wvar=xi, limit=200)[0]
    DRReHt  = quad(fPV_ReHt, 1e-10, 1.0, weight='cauchy', wvar=xi, limit=200)[0]
    # Evaluate the CFFs
    ImH = fImH(xi)
    ReH = 1. / pi * DRReH - Ct
    ReE = Ct
    ImHtilde = fImHt(xi)
    ReHtilde = 1. / pi * DRReHt
    ReEtilde = rpi / xi * 2.164 / ((0.0196 - t) * pow(1. - t / Mpi / Mpi, 2.))
    ImE = 0.
    ImEt = 0.

    return [round(float(x), 5) for x in [ReH, ReHtilde, ReE, ReEtilde, ImH, ImHtilde, ImE, ImEt]]

