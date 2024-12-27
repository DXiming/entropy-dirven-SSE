import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import deeptime.markov as markov
from deeptime.clustering import KMeans
from deeptime.data import ellipsoids
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM
from deeptime.plots.chapman_kolmogorov import plot_ck_test
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales

def TrackMFPT(trajectory, lagtime, sse_type):
    """
    This function trakcing all the MFPT from one state to another
    Also the TPT net_flux will also be calculated 
    Input:
        trajectory:
            type: numpy array
            shape: (frames,)
        lagtime:
            type: int (usually we could just set 600 for stride 10 md trajs) 
            shape: 1 

    Output:
        dictionary containing all the TPT info
        keys:
           1.flux_sum: for each states in top,plane,bot,intra we have the coorsponding MFPT, the self MFPT is set as -1
               
           2.tpt: Similar as flux_sum, but we cluster the plane states to [1, 2, 3, 4]
                
           3.[state1_state2]: list containting all the flux for the same type jumping
    """
    import os
    import logging
    try:
        os.remove(u"warnings.log")
    except OSError:
        pass
    logging.basicConfig(filename="warnings.log",force =True, level=logging.ERROR)
    logging.captureWarnings(True)
    mlestimator = markov.msm.MaximumLikelihoodMSM(
        reversible=True,
        stationary_distribution_constraint=None
    )
    
    counts_estimator = markov.TransitionCountEstimator(
        lagtime=lagtime, count_mode="sliding"
    )
    counts = counts_estimator.fit(trajectory).fetch_model()
    
    mlmsm = mlestimator.fit(counts).fetch_model()
    if sse_type != "lspscl":
        states_count = 7
    else:
        states_count = 13
    mlestimator = markov.msm.MaximumLikelihoodMSM(
    reversible=True,
    stationary_distribution_constraint=None,
        
    )
    
    counts_estimator = markov.TransitionCountEstimator(
        lagtime=lagtime, count_mode="sliding-effective",
        n_states=states_count
    )
    counts = counts_estimator.fit(trajectory).fetch_model()
    mlmsm = mlestimator.fit(counts).fetch_model()
    
    dic_tpt = {}
    flux_sum = -np.ones((states_count,states_count))
    
    for state1 in np.arange(0, states_count):
        for state2 in np.arange(0, states_count):
            dic_tpt[f"{state1}_{state2}"] = {}
            dic_tpt[f"{state1}_{state2}"]["tpt_gross"] = []
            dic_tpt[f"{state1}_{state2}"]["tpt_net"] = []
            dic_tpt[f"{state1}_{state2}"]["sets"] = []
    dic_tpt["tpb"] = {}
    dic_tpt["tpb"]["sets"] = []
    dic_tpt["tpb"]["tpt"] = []
    uniq_state = mlmsm.count_model.state_symbols # these are states lables count for MSM
    num_states = len(uniq_state)
    
    states_num = np.arange(1, states_count+1) 
    for state1 in np.arange(0, num_states):
        for state2 in np.arange(0, num_states):
            if state1 != state2:
                flux = mlmsm.reactive_flux([state1], [state2])
                sets, tpt = flux.coarse_grain([ [i] for i in np.arange(0, num_states)])
                sets_index = np.array([list(i)[0] for i in sets])
                map = {key: value for key, value in zip(np.arange(0, num_states), uniq_state-1 if sse_type!="lspscl" else uniq_state)}
                flux_sum[map[state1], map[state2]] = flux.mfpt *20
                # get and store the gross and net flux infos
                tpt_gross = -np.zeros((states_count,states_count))
                tpt_net   = -np.zeros((states_count,states_count))
                for set1 in sets_index:
                    for set2 in sets_index:
                        tpt_gross[map[set1], map[set2]] = tpt.gross_flux[set1, set2] 
                        tpt_net[map[set1], map[set2]]   = tpt.net_flux[set1, set2] 
                dic_tpt[f"{map[state1]}_{map[state2]}"]["tpt_gross"].append(tpt_gross)
                dic_tpt[f"{map[state1]}_{map[state2]}"]["tpt_net"].append(tpt_net)
                dic_tpt[f"{map[state1]}_{map[state2]}"]["sets"].append(sets)
    dic_tpt["mfpt"]= flux_sum
    
    if num_states == 7:
        top_states = [0]
        bot_states = [5]
        intra_states = [6]
        plane_states =[1, 2, 3, 4]
        for state1 in [top_states, plane_states, bot_states, intra_states]:
            for state2 in [top_states, plane_states, bot_states, intra_states]:
                if state1[0] != state2[0]:
                    flux = mlmsm.reactive_flux(state1, state2)
                    sets_tpb, tpt_tpb = flux.coarse_grain([top_states, plane_states, bot_states, intra_states])
                    dic_tpt["tpb"]["sets"].append(sets_tpb)
                    dic_tpt["tpb"]["tpt"].append(tpt_tpb)
    return dic_tpt

def MSMana(traj,sse_type,temp ):
    """
    Get the MFPT results from MSMs.
    return the dictionary of the results
    """
    import warnings
    warnings.filterwarnings("ignore")
    results = {}
    results[temp]= {}
    print("Running now!")
    if sse_type == "lpscl_iii":
        for i in np.arange(144, 288):
            states = np.load(f"./data/{sse_type}/{temp}K/discrete_trajs/states_{i}.npy", allow_pickle='TRUE').item()
            results[temp][i]=[]
            for  atom in np.arange(0,6):
                trajectory = np.array(states[atom])
                results[temp][i].append(TrackMFPT(trajectory, 600, sse_type))
        for i in np.arange(0, 144):
            states = np.load(f"./data/{sse_type}/{temp}K/discrete_trajs/states_{i}.npy", allow_pickle='TRUE').item()
            results[temp][i]=[]
            for  atom in np.arange(0,5):
                trajectory = np.array(states[atom])
                results[temp][i].append(TrackMFPT(trajectory, 600, sse_type))
        print("Finished analysis!")
        df_mfpt = {}
        for temp in [temp]:
            df_mfpt[temp] = []
            mfpts_1 = np.array([ results[temp][i][n]["mfpt"] for i in np.arange(0,144) for n in np.arange(0,5)] )
            mfpts_2 = np.array([ results[temp][i][n]["mfpt"] for i in np.arange(144,288) for n in np.arange(0,6)] )
            mfpts = np.concatenate((mfpts_1,mfpts_2))
            mfpt_mean = np.ones((7,7))
            for i in np.arange(0,7):
                for j in np.arange(0, 7):
                    mfpt = mfpts[:,i,j][mfpts[:,i,j] >0]
                    mfpt = mfpt[mfpt <= 1e7] # > 1e7 (10 ns we think the jumping not happen)
                    
                    mfpt_m = mfpt.mean()/1000
                    mfpt_mean[i,j] = mfpt_m
            df_mfpt[temp] = pd.DataFrame(mfpt_mean)
            
            df_mfpt[temp] = df_mfpt[temp].fillna(value=0)
    elif sse_type == "lpscl_ii":
        for i in np.arange(0, 288):
            states = np.load(f"./data/{sse_type}/{temp}K/discrete_trajs/states_{i}.npy", allow_pickle='TRUE').item()
            results[temp][i]=[]
            for  atom in np.arange(0,6):
                trajectory = np.array(states[atom])
                results[temp][i].append(TrackMFPT(trajectory, 600, sse_type))
        print("Finished analysis!")
        df_mfpt = {}
        for temp in [temp]:
            df_mfpt[temp] = []
            mfpts = np.array([ results[temp][i][n]["mfpt"] for i in np.arange(0,288) for n in np.arange(0,6)] )
            mfpt_mean = np.ones((7,7))
            for i in np.arange(0,7):
                for j in np.arange(0, 7):
                    mfpt = mfpts[:,i,j][mfpts[:,i,j] >0]
                    mfpt = mfpt[mfpt <= 1e7] # > 1e7 (10 ns we think the jumping not happen)
                    mfpt_m = mfpt.mean()/1000
                    mfpt_mean[i,j] = mfpt_m
            df_mfpt[temp] = pd.DataFrame(mfpt_mean)
            df_mfpt[temp] = df_mfpt[temp].fillna(value=0)
    elif sse_type == "lspscl":
        for i in np.arange(0, 72):
            states = np.load(f"./data/{sse_type}/{temp}K/discrete_trajs/{i}_states.npy", allow_pickle='TRUE').item()
            results[temp][i]=[]
            for  atom in np.arange(1,21):
                trajectory = np.array(states[atom])
                results[temp][i].append(TrackMFPT(trajectory, 600, sse_type))
        print("Finished analysis!")
        df_mfpt = {}
        for temp in [temp]:
            df_mfpt[temp] = []
            mfpts = np.array([results[temp][i][n]['mfpt'] 
                              for i in np.arange(0,72) 
                              for n in np.arange(0,20) ])
            mfpt_mean = np.ones((13,13))
            for i in np.arange(0,13):
                for j in np.arange(0, 13):
                    if i != j:
                        mfpt = mfpts[:,i,j][mfpts[:,i,j] >= 0]
                        mfpt = mfpt[mfpt <= 1e7] # > 1e7 (10 ns we think the jumping not happen)
                        if len(mfpt) >=10:
                            mfpt_m = mfpt.mean()/1000
                            mfpt_mean[i,j] = mfpt_m
                        else:
                            mfpt_mean[i,j] = np.nan
                    elif i ==j:
                        mfpt_mean[i,j] = 0
            df_mfpt[temp] = pd.DataFrame(mfpt_mean)
            df_mfpt[temp] = df_mfpt[temp].fillna(value=np.nan)
    return results, df_mfpt[temp]

def HeatMFPT(df_mfpt,sse_type, temp):
    """
    This function plot the heatmap of the dataframe of the MFPT get from MSM models
    Input:
        df: pandas dataframe from of MFPT at targt temperature 
            type: pandas Dataframe
            shape: (7,7)
        temp: temp of the MSM model
            type: int, scalar
            shape:1
    Output:
        png file: save in the place "./general/markov/vac/{temp}K/figures/mfpts.png"
    """
    temp = temp
    if temp == 300:
        center = 150
        vmin   = 0
        vmax   =  300
    elif temp == 600:
        center = 1
        vmin = 0
        vmax = 2
    elif temp == 900:
        center = 1
        vmin   = 0
        vmax   =  2
    elif temp == 1200:
        center = 0.5
        vmin   = 0
        vmax   =  1
    if sse_type != "lspscl":
        #sns.set_theme(palette="tab10")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(df_mfpt,
                    annot=True,
                    fmt=".1f", 
                    center= center,
                    cmap="PiYG", #PiYG
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=.5,
                    ax=ax,
                    cbar_kws={'label': 'MFPT (ps)'},
                    annot_kws={"fontsize":10}
                   )
        ax.invert_yaxis()
        states = ["Top", "Plane1", "Plane2", "Plane3", "Plane4", "Bottom", "Outside"]
        states = ["LS1", "LS2", "LS3", "LS4", "LS5", "LS6", "LS7"]
        ax.set_xticks(np.array([0, 1,2,3,4,5,6])+.5, labels=states,)
        ax.set_yticks(np.array([0, 1,2,3,4,5,6])+.5, labels=states,)
    elif sse_type == "lspscl":
        from matplotlib.colors import ListedColormap
        df = df_mfpt
        annot_df = df.applymap(lambda f: f'{f:.1f}')
        fig, ax = plt.subplots(squeeze=False,sharey=True,sharex=True, figsize=(9,6))
        sns.heatmap(
            np.where(df.isna(), 0, np.nan),
            ax=ax[0, 0],
            cbar=False,
            annot=np.full_like(df, "NA", dtype=object),
            fmt="",
            annot_kws={"size": 10, "va": "center_baseline", "color": "white"},
            cmap=ListedColormap(['royalblue']),
            linewidth=0)
        g= sns.heatmap(
            df,
            ax=ax[0, 0],
            #cbar=True,
            annot=annot_df,
            fmt="",
            annot_kws={"size": 10, "va": "center_baseline"},
            cmap="PiYG",
            linewidth=0.5,
            linecolor="black",
            vmin=vmin,
            vmax=vmax,
            linewidths=1,
            cbar_kws={'label': 'MFPT (ps)', },
            xticklabels=True,
            yticklabels=True)
        ax[0,0].invert_yaxis()
        states = ["LS1", "LS2", "LS3", "LS4", "LS5", "LS6", "LS7", "LS8", "LS9", "LS10", "LS11","LS12", "LS13"]
        ax[0,0].set_xticks(np.array([0, 1,2,3,4,5,6,7,8,9,10,11,12])+.5, labels=states,)
        ax[0,0].set_yticks(np.array([0, 1,2,3,4,5,6,7,8,9,10,11,12])+.5, labels=states,)

def Flow(results, temp, jump_type, ):
    """
    This function generates the sankey plots of the Lithim jumping flows 
    Input:
        results: lithium flow through TPT theory
        temp: temperature of the system
            type: int, scalar
            shape: 1
        jump_type: all the jumps from one type site to another
            type: string 
            shape: no limit 
                example: "top_plane"
    Output:
        plotly sankey diagram
    """
    temp = temp
    import plotly.io as pio
    import plotly.graph_objects as go
    # 1.  the number that we do the TPT analysis
    # TPT number: 4 plane to top, 6 plane to bot, 7 plane to intra, 8 bottom to top
    jump_dic = {"top_plane":0, "top_bot":1, "top_intra":2, 
                "plane_top":3, "plane_bot":4, "plane_intra":5,
                "bot_top":6,   "bot_plane":7, "bot_intra":8,
                "intra_top":9, "intra_plane":10, "intra_bot":11
               }
    tpt_number = jump_dic[jump_type]
    cages = 288
    flux_mapping = \
    [[{0}, {5}, {6}, {1, 2, 3, 4}],
     [{0}, {1, 2, 3, 4}, {6}, {5}],
     [{0}, {1, 2, 3, 4}, {5}, {6}],
     [{1, 2, 3, 4}, {5}, {6}, {0}],
     [{1, 2, 3, 4}, {0}, {6}, {5}],
     [{1, 2, 3, 4}, {0}, {5}, {6}],
     [{5}, {1, 2, 3, 4}, {6}, {0}],
     [{5}, {0}, {6}, {1, 2, 3, 4}],
     [{5}, {0}, {1, 2, 3, 4}, {6}],
     [{6}, {1, 2, 3, 4}, {5}, {0}],
     [{6}, {0}, {5}, {1, 2, 3, 4}],
     [{6}, {0}, {1, 2, 3, 4}, {5}]]
    
    all_flux_1 = [results[temp][cage][atom]["tpb"]['tpt'][tpt_number].net_flux 
                for cage in np.arange(0,144)
                for atom in np.arange(0,5)
                if results[temp][cage][atom]["tpb"]['tpt'] != [] ] 
    all_flux_1 = np.array([arr for arr in all_flux_1 if len(arr) == 4])
    ### cage with 6 atoms Li
    all_flux_2 = [results[temp][cage][atom]["tpb"]['tpt'][tpt_number].net_flux 
                for cage in np.arange(144,288)
                for atom in np.arange(0,6)
                if results[temp][cage][atom]["tpb"]['tpt'] != [] ]
    all_flux_2 = np.array([arr for arr in all_flux_2 if len(arr) == 4])
    all_flux = np.concatenate((all_flux_1, all_flux_2))
    net_flux = all_flux.mean(axis=0)
    from sklearn.preprocessing import normalize
    norm_flow = normalize(net_flux, axis=1)
    norm_flow
    
    flow_patterns = {"path_1": np.array([[0,3]]),
                     "path_2": np.array([[0,1], [1,3]]),
                     "path_3": np.array([[0,1], [1,2], [2,3]]),
                     "path_4": np.array([[0,1], [1,2], [2,1], [1,3]]),
                     "path_5": np.array([[0,2], [2,3]]),
                     "path_6": np.array([[0,2], [2,1], [1,3]]),
                     "path_7": np.array([[0,2], [2,1], [1,2], [2,3]]),
                    }
    S = []
    for path,sites in flow_patterns.items():
        flows = []
        for i in sites:
            
            flows.append(norm_flow[i[0], i [1]])
        
        flows = np.array(flows)
        #print(path, ":", flows)
        f_ = np.prod(np.array(flows))
        S.append(- f_ * np.log(f_))
    S = np.array(S)
    
    nodes = []
    sets_mapping = {"{0}":"Top", "{1, 2, 3, 4}":"Planar", "{6}": "External", "{5}":"Bottom" }
    for i in flux_mapping[tpt_number]:
        nodes.append(sets_mapping[f"{i}"])
    nodes_color = ['rgba(252,65,94,0.7)','rgba(255,162,0,0.7)','rgba(55,178,255,0.7)','rgba(150, 252, 167,1)']
    links = [
        {"source": 0, "target": 1, "value": net_flux[0,1], "color":'rgba(252,65,94,0.4)'},
        {"source": 0, "target": 2, "value": net_flux[0,2], "color":'rgba(252,65,94,0.4)'},
        {"source": 0, "target": 3, "value": net_flux[0,3], "color":'rgba(252,65,94,0.4)'},
        {"source": 1, "target": 0, "value": net_flux[1,0], "color":'rgba(255,162,0,0.4)'},
        {"source": 1, "target": 2, "value": net_flux[1,2], "color":'rgba(255,162,0,0.4)'},
        {"source": 1, "target": 3, "value": net_flux[2,3], "color":'rgba(255,162,0,0.4)'},
        {"source": 2, "target": 0, "value": net_flux[2,0], "color":'rgba(55,178,255,0.4)'},
        {"source": 2, "target": 1, "value": net_flux[2,1], "color":'rgba(55,178,255,0.4)'},
        {"source": 2, "target": 3, "value": net_flux[2,3], "color":'rgba(55,178,255,0.4)'},
        {"source": 3, "target": 0, "value": net_flux[3,0], "color":'rgba(150, 252, 167,1)'},
        {"source": 3, "target": 1, "value": net_flux[3,1], "color":'rgba(150, 252, 167,1)'},
        {"source": 3, "target": 2, "value": net_flux[3,2], "color":'rgba(150, 252, 167,1)'},
    ]
    max_value = max(link["value"] for link in links)
    min_value = min(link["value"] for link in links)
    for link in links:
        link["normalized_value"] = (link["value"] - min_value) / (max_value - min_value)
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=0,
            thickness=10,
            line=dict(color="black", width=0.8),
            label=nodes,
            color = nodes_color
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links],
            color= [link["color"] for link in links]
        )
    ))
    
    config = {
      'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 520,
        'width': 1080,
        'scale':12
      }
    }

    fig.update_layout(
        font=dict(
            family="Arial",
            size=40,  # Set the font size here
        )
    )
    fig.show(config=config)
    print("Path entropy:", S.sum())

def sFlow(sse_type, results, temp, jump_type, flux_type="tpt_net"):
    """
    Similar function for drawing the sankey diagram, but only for site to site
    Input:
        results: lithium flux through TPT theory
        temp: int
        jump_type: "site1_site2"
        flux_type: tpt_net 
    Output:
        plotly figures
    """
    from matplotlib import colors
    import plotly.io as pio
    import plotly.graph_objects as go
    if sse_type !="lspscl":
        if sse_type == "lpscl_iii":
            jump = [ results[temp][i][n][jump_type][flux_type][0]
             for i in np.arange(144,288) 
             for n in np.arange(0,6) 
             if len(results[temp][i][n][jump_type][flux_type])!=0 ] 
        elif sse_type == "lpscl_ii":
            jump = [ results[temp][i][n][jump_type][flux_type][0]
             for i in np.arange(144,288) 
             for n in np.arange(0,6) 
             if len(results[temp][i][n][jump_type][flux_type])!=0 ] 
        jump = np.array(jump)
        net_flux = jump.mean(axis=0)
        colormap = plt.cm.get_cmap("rainbow")
        values = np.linspace(0, 2, 12)
        # Convert values to ARGB format
        alpha=0.1
        argb_colors = [colors.to_rgba(colormap(value), alpha=alpha) for value in values]
        
        nodes_color = [ tuple([ int(i*255) if i !=alpha else i  for i in argb_colors[n]]) for n in np.arange(0,12)] 
        nodes_color = [f"rgb{i}" for i in nodes_color]
        import plotly.io as pio
        import plotly.graph_objects as go
        links = [ {"source":state1, "target":state2, "value": net_flux[state1, state2], "color":nodes_color[state1]}
                  for state1 in np.arange(0,6)
                  for state2 in np.arange(0,6)
                 ]
        nodes = ["S-1", "S-2", "S-3", "S-4", "S-5", "S-6", "S-7"]
        max_value = max(link["value"] for link in links)
        min_value = min(link["value"] for link in links)
        for link in links:
            link["normalized_value"] = (link["value"] - min_value) / (max_value - min_value)
    elif sse_type == "lspscl":
        jump = [ results[temp][i][n][jump_type][flux_type][0]
         for i in np.arange(0,72) 
         for n in np.arange(0,20) 
         if len(results[temp][i][n][jump_type][flux_type])!=0 ] 
        jump = np.array(jump)
        net_flux = jump.mean(axis=0)
        colormap = plt.cm.get_cmap("tab20c")
        values = np.linspace(0, 1, 13)
        alpha=0.1
        argb_colors = [colors.to_rgba(colormap(value), alpha=alpha) for value in values]
        nodes_color = [ tuple([ int(i*255) if i !=alpha else i  for i in argb_colors[n]]) for n in np.arange(0,13)] 
        nodes_color = [f"rgb{i}" for i in nodes_color]
        
        links = [ {"source":state1, "target":state2, "value": net_flux[state1, state2], "color":nodes_color[state1]}
                  for state1 in np.arange(0,13)
                  for state2 in np.arange(0,13)
                 ]
        nodes = ["LS1", "LS2", "LS3", "LS4", "LS5", "LS6", "LS7", "LS8", "LS9", "LS10", "LS11", "LS12", "LS13"]
        max_value = max(link["value"] for link in links)
        min_value = min(link["value"] for link in links)
        for link in links:
            link["normalized_value"] =  link["value"] #(link["value"] - min_value) / (max_value - min_value)
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=0,
            thickness=10,
            line=dict(color="black", width=0.8),
            label=nodes,
            color = nodes_color,
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            #value= [link["normalized_value"] for link in links],
            value= [link["value"] for link in links],
            color= [link["color"] for link in links],
        )
    ))
    config = {
      'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 520,
        'width': 1080,
        'scale':12,# Multiply title/legend/axis/canvas sizes by this factor
        'color':'rgba(255,0,0,0.1)'
      }
    }
    fig.update_layout(
        font=dict(
            family="Arial",
            size=40,  # Set the font size here
            #color="RebeccaPurple"
        )
    )
    fig.show(config=config)