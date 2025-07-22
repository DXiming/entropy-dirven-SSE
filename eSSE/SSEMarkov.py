import sys
import re
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import ListedColormap
import deeptime.markov as markov
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM
from deeptime.plots.chapman_kolmogorov import plot_ck_test
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales
from .DiscreteTraj import DiscreteTraj

warnings.filterwarnings("ignore")
class MarkovSSE():
    """Markov state model construction and post-analysis"""
    def __init__(self, disc_traj,):
        super(MarkovSSE, self).__init__()
        self.disc_traj = disc_traj
    def msmCheck(self, ):
        """Check if diffusion exists between different states"""
        uniq_states = np.unique(self.disc_traj)
        if len(uniq_states) <=1:
            print("Frozen state space. Seems a bad candidate.")
            return None
        else:
            pass
        return len(uniq_states)
    def implied_timescale(self, times):
        """Find implied timescales"""
        try:
            os.remove(u"warnings.log")
        except OSError:
            pass
        logging.basicConfig(filename="warnings.log",force =True, level=logging.WARNING)
        logging.captureWarnings(True)
        
        models_its = []
        lagtimes = times
        n_states = self.msmCheck()
        states =  np.array(self.disc_traj)
        
        for lagtime in lagtimes:
            counts = TransitionCountEstimator(lagtime=lagtime, 
                                              count_mode='effective',
                                              n_states=n_states
                                             ).fit_fetch(self.disc_traj)
            models_its.append(BayesianMSM(n_samples=50).fit_fetch(counts))
        
        its_data = implied_timescales(models_its)
        n_lines = 8
        cmap = mpl.colormaps['Blues_r']

        colors = cmap(np.linspace(0, 1, n_lines))

        fig, ax = plt.subplots(1, 1, figsize=(2.5,2))
        plot_implied_timescales(its_data,  ax=ax, colors=colors)
        ax.set_yscale('log')
        ax.set_xlabel('Lag time (steps)')
        ax.set_ylabel('Timescale (steps)')
        return  fig 
    def ck_test(self, lagtime):
        """Check Chapman-Kolmogorov test"""
        models = []
        for lag in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            counts_estimator = TransitionCountEstimator(lagtime=lag, count_mode='effective')
            models.append(MaximumLikelihoodMSM().fit_fetch(counts_estimator.fit_fetch(self.disc_traj).submodel_largest()))
        
        counts_estimator = TransitionCountEstimator(lagtime=lagtime, count_mode='effective')
        test_model = MaximumLikelihoodMSM().fit_fetch(counts_estimator.fit_fetch(self.disc_traj).submodel_largest())
        
        n_metastable_sets = self.msmCheck()
        ck_test = test_model.ck_test(models, n_metastable_sets=n_metastable_sets)
        grid = plot_ck_test(ck_test, legend=True)

        return grid
    def post_analysis(self, lagtime, length_step):
        """Get all info from MSM"""

        mlestimator = markov.msm.MaximumLikelihoodMSM(
            reversible=True,
            stationary_distribution_constraint=None
        )
        
        counts_estimator = markov.TransitionCountEstimator(
            lagtime=lagtime, count_mode="sliding"
        )
        counts = counts_estimator.fit(self.disc_traj).fetch_model()
        
        mlmsm = mlestimator.fit(counts).fetch_model()
        
        try:
            os.remove(u"warnings.log")
        except OSError:
            pass
        logging.basicConfig(filename="warnings.log",force =True, level=logging.WARNING)
        logging.captureWarnings(True)
        
        states_num = np.unique(self.disc_traj)
        n_states = self.msmCheck()
        if n_states != None:
            mlestimator = markov.msm.MaximumLikelihoodMSM(
            reversible=True,
            stationary_distribution_constraint=None,
                
            )
            
            counts_estimator = markov.TransitionCountEstimator(
                lagtime=lagtime, count_mode="sliding-effective",
                n_states=n_states
            )
            counts = counts_estimator.fit(self.disc_traj).fetch_model()
            mlmsm = mlestimator.fit(counts).fetch_model()
            
            dic_tpt = {}
            ref_dict = {}
            flux_sum = -np.ones((n_states,n_states))
            
            uniq_state = mlmsm.count_model.state_symbols
            
            num_states = len(uniq_state)
            map_dict = {key: value for key, value in zip(np.arange(0, num_states), uniq_state)}
            
            states_index = map_dict.keys()
            for state1 in states_index:
                for state2 in states_index:
                    dic_tpt[f"{state1}_{state2}"] = {}
                    dic_tpt[f"{state1}_{state2}"]["tpt_gross"] = []
                    dic_tpt[f"{state1}_{state2}"]["tpt_net"] = []
                    dic_tpt[f"{state1}_{state2}"]["sets"] = []
                    dic_tpt[f"{state1}_{state2}"]["R_states"] = []
            
            for state1 in range(num_states):
                for state2 in range(num_states):
                    if state1 != state2:
                        flux = mlmsm.reactive_flux([state1], [state2])
                        sets, tpt = flux.coarse_grain([ [i] for i in np.arange(0, num_states)])
            
                        sets_index = np.array([list(i)[0] for i in sets])
            
                        flux_sum[state1, state2] = flux.mfpt * length_step
                        
                        tpt_gross = -np.zeros((num_states,num_states))
                        tpt_net   = -np.zeros((num_states,num_states))
                        for set1 in sets_index:
                            for set2 in sets_index:
                                tpt_gross[set1, set2] = tpt.gross_flux[set1, set2] 
                                tpt_net[set1, set2]   = tpt.net_flux[set1, set2] 
                        
                        dic_tpt[f"{state1}_{state2}"]["tpt_gross"].append(tpt_gross)
                        dic_tpt[f"{state1}_{state2}"]["tpt_net"].append(tpt_net)
                        dic_tpt[f"{state1}_{state2}"]["sets"].append(sets)
                        dic_tpt[f"{state1}_{state2}"]["R_states"].append([map_dict[state1], map_dict[state2]])
            dic_tpt["mfpt"]= flux_sum
            dic_tpt["ref"]= map_dict
            return dic_tpt
        else:
            return None


class PostAna():
    """Post-analysis using MSM model"""
    def __init__(self, traj, vors, coords_with_labels, states_with_labels):
        super(PostAna, self).__init__()
        self.traj = traj
        self.vors = vors
        self.topology = traj.topology
        self.vcs = np.array([ atom.index for atom in self.topology.atoms if (re.search(r'VC*', atom.name) != None) ])
        self.coords_with_labels = coords_with_labels
        self.states_with_labels = states_with_labels
        self.lithium_index = [ atom.index for atom in self.topology.atoms if (re.search(r'Li', atom.name) != None) ]

    def mfpt(self, results, wp, center, plot=False, partial=True,annotate=False):
        """
        Mean first passage time analysis
        """
        try:
            os.remove(u"warnings.log")
        except OSError:
            pass
        logging.basicConfig(filename="warnings.log",force =True, level=logging.WARNING)
        logging.captureWarnings(True)
        if partial == True:
            df_mfpt = []
            mfpts = []
            cluster_tot_sites = len(self.states_with_labels[wp][center])
            states_init = self.states_with_labels[wp][center].min() -1
            
            for num in range(len(results[wp][center])):
                if results[wp][center][num] is not None:
                    mfpt_aligned  = -np.ones((cluster_tot_sites+1,cluster_tot_sites+1)) #* 1e8
                    
                    results[wp][center][num]["mfpt_aligned"] = []
                    mfpt_partial = results[wp][center][num]["mfpt"]
                    ref_position = results[wp][center][num]["ref"]
                    
                    for ref, real in ref_position.items():
                        if real !=0:
                            real = real - states_init
                        for ref_, real_ in ref_position.items():
                            if real_ !=0:
                                real_ = real_ - states_init
                            mfpt_aligned[real, real_] = mfpt_partial[ref,ref_]
                    mfpts.append(mfpt_aligned)
            mfpts =np.array(mfpts)
            mfpt_mean = np.ones((cluster_tot_sites+1,cluster_tot_sites+1))
            for i in np.arange(0,cluster_tot_sites+1):
                for j in np.arange(0, cluster_tot_sites+1):
                    if i == j:
                        mfpt= np.zeros(len(results[wp][center]))
                    else:
                        mfpt = mfpts[:,i,j][mfpts[:,i,j] >0]
                        mfpt_m = mfpt.mean()/1000
                        if mfpt_m >= 1e7/1000:
                            mfpt_m = np.nan
                        mfpt_mean[i,j] = mfpt_m
            df_mfpt = pd.DataFrame(mfpt_mean)
        else:
            df_mfpt = []
            mfpts = []
            tot_sites = np.sum([ len(i) for i in self.coords_with_labels[wp].values()])
            for num in range(len(results[wp][center])):
                if results[wp][center][num] is not None:
                    mfpt_all  = -np.ones((tot_sites+1,tot_sites+1)) #* 1e8
                    
                    mfpt_partial = results[wp][center][num]["mfpt"]
                    ref_position = results[wp][center][num]["ref"]
                    for i_, i in ref_position.items():
                        for j_, j in ref_position.items():
                            mfpt_all[i, j] = mfpt_partial[i_,j_]
                    mfpts.append(mfpt_all)
            mfpts =np.array(mfpts)
            mfpt_mean = np.ones((tot_sites+1,tot_sites+1))
            for i in np.arange(0,tot_sites+1):
                for j in np.arange(0, tot_sites+1):
                    if i == j:
                        mfpt= np.zeros(len(results[wp][center]))
                    else:
                        mfpt = mfpts[:,i,j][mfpts[:,i,j] >0]
                    
                    mfpt_m = mfpt.mean()/1000
                    if mfpt_m >= 1e7/1000:
                        mfpt_m = np.nan
                    mfpt_mean[i,j] = mfpt_m
            df_mfpt = pd.DataFrame(mfpt_mean)    
        if plot == True:
            figsize = (4,3) if partial == True else (12,10)
            annot_df = df_mfpt.applymap(lambda f: f'{f:.1f}' if f < 999.9 else f'{f:.0f}')
            fig, ax = plt.subplots(squeeze=False,
                                   sharey=True,
                                   sharex=True, 
                                   figsize=figsize
                                   )
            sns.heatmap(
                np.where(df_mfpt.isna(), 0, np.nan),
                ax=ax[0, 0],
                cbar=False,
                fmt="",
                annot_kws={"size": 10, "va": "center_baseline", "color": "white"},
                cmap=ListedColormap(['royalblue']),
                linewidth=0)
            g= sns.heatmap(
                df_mfpt,
                ax=ax[0, 0],
                annot=None if annotate==False else annot_df ,
                fmt="",
                annot_kws={"size": 8, "va": "center_baseline"},
                cmap="PiYG",
                linewidth=0.5,
                linecolor="black",
                vmin=0,
                vmax=1000,
                linewidths=1,
                cbar_kws={'label': 'MFPT (ps)', 'pad': 0.01},
                xticklabels=True,
                yticklabels=True)
            ax[0,0].invert_yaxis()
            from matplotlib.patches import Rectangle
            r_initiates = self.states_with_labels[wp][center][0]
            r_size = len(self.states_with_labels[wp][center])
            if partial == False:
                g.add_patch(Rectangle((r_initiates, r_initiates), r_size, r_size, edgecolor='cyan', fill=False, lw=3))
                tot_sites = np.array([i.shape[0] for i in self.states_with_labels[wp].values()]).sum()
            else:
                tot_sites = len(self.states_with_labels[wp][center])
            states = [f"LS$_{{{n}}}$" for n in range(tot_sites +1) ]
            ax[0,0].set_xticks(np.arange(0, tot_sites + 1)+.5, labels=states,)
            ax[0,0].set_yticks(np.arange(0, tot_sites + 1)+.5, labels=states,)
            return fig
        return df_mfpt
        
    def site_sp(self, results, wp, center, start, end, intermidiates,  flux_type, partial=True,plot=False,counts_ratio=0.15, ):
        """Calculate path entropy using DFS"""
        try:
            os.remove(u"warnings.log")
        except OSError:
            pass
        logging.basicConfig(filename="warnings.log",force =True, level=logging.WARNING)
        logging.captureWarnings(True)
        from sklearn.preprocessing import normalize
        import math
        if partial == True:
            target_states =[start, end]
            jumps = []
            cluster_tot_sites = len(self.states_with_labels[wp][center])
            states_init = self.states_with_labels[wp][center].min() -1
            
            for num in range(len(results[wp][center])):
                if results[wp][center][num] is not None:
                    possible_states = list(results[wp][center][num]['ref'].values())
                    check_jp = np.isin(target_states, possible_states).all()
                    if check_jp != False:
                        new_end = [key for key, value in results[wp][center][num]['ref'].items()
                                           if value == end]
                        new_start = [key for key, value in results[wp][center][num]['ref'].items()
                                           if value == start]
                        jump_type = f"{new_start[0]}_{new_end[0]}"

                        flux_aligned  = np.zeros((cluster_tot_sites+1,cluster_tot_sites+1))
                        for pos in range(cluster_tot_sites + 1):
                            flux_aligned[pos,pos]  = -1
                        flux_partial = results[wp][center][num][jump_type]['tpt_net'][0]
                        ref_position = results[wp][center][num]["ref"]
                        for ref, real in ref_position.items():
                            if real !=0:
                                real = real - states_init
                            for ref_, real_ in ref_position.items():
                                if real_ !=0:
                                    real_ = real_ - states_init
                                if real != real_:
                                    flux_aligned[real, real_] = flux_partial[ref,ref_]
                                else:
                                    flux_aligned[real, real_] = -1
                        jumps.append(flux_aligned)
            jumps =np.array(jumps)
            effective_ratio = len(jumps)/len(results[wp][center])
            jumps =np.array(jumps)
            jumps[jumps <0] = 0
            net_flow = jumps.mean(axis=0)
            all_states = np.concatenate((self.states_with_labels[wp][center] -states_init, [0]))
            graph = {node: list(all_states) for node in all_states}
            start = 0 if start ==0 else start - states_init
            end = 0 if end ==0 else end - states_init
        else:
            jumps = []
            cluster_tot_sites = np.sum([ len(i) for i in self.coords_with_labels[wp].values()])
            target_states =[start, end]
            
            for num in range(len(results[wp][center])):
                
                if results[wp][center][num] is not None :
                    possible_states = list(results[wp][center][num]['ref'].values())
                    check_jp = np.isin(target_states, possible_states).all()
                    if check_jp != False:
                        new_end = [key for key, value in results[wp][center][num]['ref'].items()
                                           if value == end]
                        new_start = [key for key, value in results[wp][center][num]['ref'].items()
                                           if value == start]
                        jump_type = f"{new_start[0]}_{new_end[0]}"
                        flux_aligned  = np.zeros((cluster_tot_sites+1,cluster_tot_sites+1))
                        for pos in range(cluster_tot_sites + 1):
                            flux_aligned[pos,pos]  = -1
                        flux_partial = results[wp][center][num][jump_type]['tpt_net'][0]
                        ref_position = results[wp][center][num]["ref"]
                        for i_, i in ref_position.items():
                            for j_, j in ref_position.items():
                                if i != j:
                                    flux_aligned[i, j] = flux_partial[i_,j_]
                                else:
                                    flux_aligned[i, j] = -1
                        jumps.append(flux_aligned)
            jumps =np.array(jumps)
            effective_ratio = len(jumps)/len(results[wp][center])
            jumps[jumps <0] = 0
            net_flow = jumps.mean(axis=0)
            graph = {node: list(range(0, cluster_tot_sites + 1)) for node in range(0, cluster_tot_sites + 1)}

        def dfs(graph, start, end, path=[], paths=[]):
            if start == end:
                paths.append(path)
            elif len(path) < intermidiates:  # Limit path length to 4 nodes or less
                for node in graph[start]:
                    if node != start:  # Exclude self-nodes (loops)
                        dfs(graph, node, end, path + [(start, node)], paths)
            return paths

        # Sp calc
        all_paths = dfs(graph, start, end)
        S = []
        k_b = 8.314462618  # # J/(mol*K)
        if effective_ratio >= counts_ratio:
            norm_flow = normalize(net_flow, norm="l1", axis=1)
            if all_paths:
                path_arrays = [np.array(path) for path in all_paths]
                flows_list = []
                for path_array in path_arrays:
                    if len(path_array) > 0:
                        flows = norm_flow[path_array[:, 0], path_array[:, 1]]
                        flows_list.append(flows)
                if flows_list:
                    products = np.array([np.prod(flows) for flows in flows_list])
                    s_paths = -k_b * products * np.log(products)
                    # Filter out NaN values
                    valid_s = s_paths[~np.isnan(s_paths)]
                    S.extend(valid_s)
        S = np.array(S)

        ##### Sankey plot the flux 
        if plot == True:
            net_flux = net_flow
            num_states = net_flux.shape[0]
            colormap = plt.cm.get_cmap("tab20c")
            values = np.linspace(0, 1, num_states)
            alpha=0.1
            argb_colors = [colors.to_rgba(colormap(value), alpha=alpha) for value in values]
            nodes_color = [ tuple([ int(i*255) if i !=alpha else i  for i in argb_colors[n]]) 
                           for n in np.arange(0,num_states)] 
            nodes_color = [f"rgb{i}" for i in nodes_color]
            
            links = [ {"source":state1, 
                       "target":state2,
                       "value": net_flux[state1, state2],
                       "color":nodes_color[state1]}
                      for state1 in np.arange(0,num_states)
                      for state2 in np.arange(0,num_states)
                     ]
            nodes = [f"LS<sub>{n}</sub>" for n in np.arange(0,num_states)]
            max_value = max(link["value"] for link in links)
            min_value = min(link["value"] for link in links)
            for link in links:
                link["normalized_value"] =  link["value"] 
            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=20,
                    thickness=10,
                    line=dict(color="black", width=0.8),
                    label=nodes,
                    color = nodes_color,
                    align="justify",
                ),
                link=dict(
                    source=[link["source"] for link in links],
                    target=[link["target"] for link in links],
                    value= [link["value"] for link in links],
                    color= [link["color"] for link in links],
                )
            ))
            
            config = {
              'toImageButtonOptions': {
                'format': 'png',
                'filename': 'custom_image',
                'height': 520,
                'width': 1080,
                'scale':12,
                'color':'rgba(255,0,0,0.1)'
              }
            }
            fig.update_layout(
                xaxis=dict(
                    ticktext=[f"LS$_{n}$" for n in range(num_states)],
                    tickmode='array'
                ),
                font=dict(
                    family="Arial",
                    size=40, 
                    color="RebeccaPurple",
                    weight="bold"
                )
            )
            fig.show(config=config)
            return fig
    
        return S.sum()
        
    def path_entropy(self, results, wp, center,intermidiates, flux_type,counts_ratio, partial=False  ):
        """Calculate path entropy of all paths"""
        if partial==True:
            num_nodes = len(self.coords_with_labels[wp][center])
            all_states = np.concatenate((self.states_with_labels[wp][center], [0]))
            start_grid, end_grid = np.meshgrid(all_states, all_states, indexing='ij')
            mask = start_grid != end_grid
            start_pairs = start_grid[mask]
            end_pairs = end_grid[mask]
            S = np.array([self.site_sp(results, wp, center, start, end, intermidiates, "tpt_net", partial, False, counts_ratio) 
                         for start, end in zip(start_pairs, end_pairs)])
            
        elif partial == False:
            cluster_tot_sites = np.sum([ len(i) for i in self.coords_with_labels[wp].values()])
            sites = np.arange(0, cluster_tot_sites+1)
            current_sites = self.states_with_labels[wp][center]
            other_sites = [item for item in sites if item not in current_sites]
            start_grid, end_grid = np.meshgrid(sites, sites, indexing='ij')
            mask = start_grid != end_grid
            start_pairs = start_grid[mask]
            end_pairs = end_grid[mask]
            
            S = np.array([self.site_sp(results, wp, center, start, end, intermidiates, "tpt_net", partial, False, counts_ratio) 
                         for start, end in zip(start_pairs, end_pairs)])
            
        S_p=np.sum(S)
        return S_p
    
    def escape_entropy(self, results, wp, center,intermidiates, flux_type, counts_ratio, partial=True ):
        """Calculate escape entropy of all paths"""
        if partial==True:
            current_sites = self.states_with_labels[wp][center]
            valid_starts = [site for site in current_sites if site != 0]
            S = np.array([self.site_sp(results, wp, center, start, 0, intermidiates, "tpt_net", partial, False, counts_ratio) 
                         for start in valid_starts])
        elif partial == False:
            cluster_tot_sites = np.sum([ len(i) for i in self.coords_with_labels[wp].values()])
            sites = np.arange(0, cluster_tot_sites+1)
            current_sites = self.states_with_labels[wp][center]
            other_sites = [item for item in sites if item not in current_sites]
            start_grid, end_grid = np.meshgrid(current_sites, other_sites, indexing='ij')
            mask = start_grid != end_grid
            start_pairs = start_grid[mask]
            end_pairs = end_grid[mask]
            S = np.array([self.site_sp(results, wp, center, start, end, intermidiates, "tpt_net", partial, False, counts_ratio) 
                         for start, end in zip(start_pairs, end_pairs)])
        S_p=np.sum(S)
        print(f"Escape entropy S_p = {S_p:.2f} J/(mol*K)")
        return S_p
        