
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to find eSSE module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eSSE'))

from eSSE.SSEMarkov import MarkovSSE


def analyze_msm(wp, disc_trajs, states_with_labels, lagtime=600, length_step=20):
    """MSM analysis helper function."""
    msm_info = {}
    msm_info[wp] = {}
    
    for center in list(states_with_labels[wp].keys()):
        msm_info[wp][center] = []
        for disc_traj in disc_trajs[center]:
            msm_info[wp][center].append(MarkovSSE(disc_traj).post_analysis(lagtime=lagtime, length_step=length_step))
    
    return msm_info

def analyze_path_entropy(wp, msm_info, states_with_labels, PA, partial, counts_ratios=[0.14, 0.15, 0.16]):
    """Path entropy analysis for given MSM results and parameters."""
    ps_all = {}
    for center in states_with_labels[wp].keys():
        ps = []
        for counts_ratio in counts_ratios:
            sp = PA.path_entropy(results=msm_info, 
                                wp=wp,
                                center=center,
                                intermidiates=4,
                                flux_type="tpt_net", 
                                counts_ratio=counts_ratio,
                                partial=partial)
            ps.append(sp)
        ps_all[center] = np.array(ps)
    final_ps = np.array([ps_all[center].mean() for center in ps_all.keys()])
    std_ps = np.array([ps_all[center].std() for center in ps_all.keys()])
    for center in ps_all.keys():
        print(f"Path entropy S_p (center {center}) = {ps_all[center].mean():.2f} ± {ps_all[center].std():.2f} J/(mol*K)")
    return final_ps, std_ps
