"""LH <-> Rct calibration from Table 1 of Shrivastav et al., Electrochimica Acta 560 (2026) 148659.
Rct DEcreases as LH increases. Keep these numbers in sync with synth_lh.py."""
import numpy as np

CONC = np.array([1, 5, 10, 20, 30, 50, 75, 100, 150, 200], float)          # mIU/mL
RCT = np.array([77.20, 76.87, 75.17, 72.43, 70.10, 66.02,
                64.71, 62.74, 61.27, 59.68])                                # kOhm (lab system)
PORTABLE_OFFSET_KOHM = -2.1   # portable prototype read ~2 kOhm lower than the lab system
LOQ = 8.0                     # mIU/mL, limit of quantification stated in the paper's abstract
SURGE_THRESHOLD = 25.0        # mIU/mL, surge threshold used in the paper


def lh_to_rct(lh):
    return np.interp(np.clip(lh, CONC[0], CONC[-1]), CONC, RCT)


def rct_to_lh(raw_impedance_kohm, offset=PORTABLE_OFFSET_KOHM):
    """Raw device reading (kOhm) -> estimated LH (mIU/mL). Saturates at 1 and 200."""
    return float(np.interp(np.asarray(raw_impedance_kohm) - offset, RCT[::-1], CONC[::-1]))
