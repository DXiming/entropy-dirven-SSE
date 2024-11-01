import numpy as np
import re
import mdtraj as md
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def Switch_fn(r_ij, d_0, r_0):
    """
    This function is for calculation of the switch funcion
    The default is the rational
    d_0: scalar
    r_0: scalar
    """
    n=6 
    m=2*n
    s_r = (1-((r_ij - d_0)/r_0)**n)/ (1-((r_ij - d_0)/r_0)**m)
    return s_r

def Tetra(traj, temp, md_type, sse_type, center, ):
    """
    This function is to calculate the tetrahedral degree of the atom that within the coordination shell.
    traj: 
        MDtraj objects that including the positions of the atoms
    md_type:
        "nn" or "pi"
    sse_type:
        ["f43m", "amm2", "vac", "si"]
    center: 
        ["P", "Si"]
    swith_fn:
        The sigma swith function for calculation
        will be set default
    """
    # 1. first get the indexes of the atoms
    topology = traj.topology
    lithium_index    = np.array([ atom.index for atom in topology.atoms if (re.search(r'Li', atom.name) != None) ])
    phosphorus_index = np.array([ atom.index for atom in topology.atoms if (re.search(r'P', atom.name) != None) ])
    sulfur_index     = np.array([ atom.index for atom in topology.atoms if (re.search(r'^S$', atom.name) != None) ])
    cholride_index   = np.array([ atom.index for atom in topology.atoms if (re.search(r'Cl', atom.name) != None) ])
    silicon_index    = np.array([ atom.index for atom in topology.atoms if (re.search(r'Si', atom.name) != None) ])
    pbc = traj.unitcell_lengths[0]
    # 2. we assume that the PS4 units are not broken during the simulation, so the initial shell will keep
    cutoff=0.3
    if center == "Si":
        haystack_indices = []
        haystack_indices.extend(sulfur_index)
        haystack_indices.extend(cholride_index)
        neighbor_lists = [ md.compute_neighbors(traj[0],cutoff, query_indices=np.array([silicon_index[i]]), haystack_indices=haystack_indices,)[0]
                          for i in np.arange(0, len(silicon_index))]
        num_center = len(silicon_index)
        neighbor_lists = np.array(neighbor_lists)
        
        neighbor_paired_lists = [ [silicon_index[i], neighbor_lists[i][d]] for i in  np.arange(0, len(silicon_index)) for d in np.arange(0,4)]
        neighbor_paired_lists = np.array(neighbor_paired_lists)
        
        paired_lists = np.array(neighbor_paired_lists).reshape(len(silicon_index), 4, 2)
        paired_lists = np.array(paired_lists)
    
    elif center == "P":
        neighbor_lists = [ md.compute_neighbors(traj[0],cutoff, query_indices=np.array([phosphorus_index[i]]), haystack_indices=sulfur_index,)[0]
                          for i in np.arange(0, len(phosphorus_index))]
        num_center = len(phosphorus_index)

        neighbor_lists = np.array(neighbor_lists)
        
        neighbor_paired_lists = [ [phosphorus_index[i], neighbor_lists[i][d]] for i in  np.arange(0, len(phosphorus_index)) for d in np.arange(0,4)]
        neighbor_paired_lists = np.array(neighbor_paired_lists)
        
        paired_lists = np.array(neighbor_paired_lists).reshape(len(phosphorus_index), 4, 2)
        paired_lists = np.array(paired_lists)
    frames = traj.n_frames
    dist_map = paired_lists.reshape(paired_lists.shape[0] * paired_lists.shape[1], 2)
    distances = md.compute_distances(traj, np.array(dist_map)).reshape(frames, num_center, 4)
    comps = {}
    comps_xyz = md.compute_displacements(traj, np.array(dist_map)).reshape(frames, num_center, 4, 3)
    comps[0]  = abs(comps_xyz[:,:,:, 0])
    comps[1]  = abs(comps_xyz[:,:,:, 1])
    comps[2]  = abs(comps_xyz[:,:,:, 2])
    
    s=[]
    sigs = []
    for j in np.arange(0, 4):
        dist3 = np.array(distances[:,:, j])**3
        factor = \
        (comps[0][:, :, j]  + comps[1][:, :, j] + comps[2][:, :, j])**3/ dist3 + \
        (comps[0][:, :, j]  - comps[1][:, :, j] - comps[2][:, :, j])**3/ dist3 + \
        (-comps[0][:, :, j] + comps[1][:, :, j] - comps[2][:, :, j])**3/ dist3 + \
        (-comps[0][:, :, j] - comps[1][:, :, j] + comps[2][:, :, j])**3/ dist3 
    
        sigma = Switch_fn(distances[:, :, j], 1.2, 0.3)
        s.append(factor*sigma)
        sigs.append(sigma)
    
    s = np.array(s) 
    sigs = np.array(sigs)
    new_s = s.swapaxes(0,1).swapaxes(2,1)
    new_sigs = sigs.swapaxes(0,1).swapaxes(2,1)
    sum_sigs = np.sum(new_sigs, axis=2)
    s_final = np.sum(new_s, axis=2) / sum_sigs
    
    return s_final

def S_c(traj, sse_type, temp ):
    """
    This function calculates the config entorpy
    """
    entropy = [] 
    edge_max = 5.5
    split_max = [5e-2, 5.25e-2, 5.5e-2]
    S_sc= []
    for split in split_max:
        if sse_type != "lspscl":
            results = Tetra(traj, temp=temp, md_type="nn", sse_type=sse_type, center="P",)
            data = results.flatten()
            bin_edges = np.arange(0, edge_max, split)
            hist, _ = np.histogram(data, bins=bin_edges)
        else:
            results_1 = Tetra(traj, temp=temp, md_type="nn", sse_type=sse_type, center="P",)
            results_2 = Tetra(traj, temp=temp, md_type="nn", sse_type=sse_type, center="Si",)
            data1 = results_1.flatten()
            data2 = results_2.flatten()
            data  = np.concatenate((data1, data2))
            bin_edges = np.arange(0, edge_max, split)
            hist, _ = np.histogram(data, bins=bin_edges)
        W = np.count_nonzero(hist)
        #k_b = 1.380649 * 10**(-23)      # J/K
        k_b = 8.314462618               # J/(mol*K)
        S = k_b * np.log(W)
        S_sc.append(S)
        entropy.append(S)
    print(f"Config. entorpy S_c = {np.mean(S_sc)} J/mol/K (std {np.std(S_sc)})")

def Site_S_p(sse_type, results, temp, jump_type, flux_type, start, end):
    """
    Function using the net_flux get the path entropy basd on the DFS
    Input:
    NOTE: results[temp] dictionary is a must
        temp: int
        jump_type: "site1_site2"
        flux_type: tpt_net or tpt_gross
        start: DFS start state
        end: DFS end state
    Output:
        Site path entropy
    """
    from sklearn.preprocessing import normalize
    import math
    temp = temp 
    jump_type= jump_type
    flux_type = flux_type
    if sse_type == "lpscl_iii":
        jump_5sites = [ results[temp][i][n][jump_type][flux_type][0]
         for i in np.arange(0,144) 
         for n in np.arange(0,5) 
         if len(results[temp][i][n][jump_type][flux_type])!=0 ] 
        jump_1 = np.array(jump_5sites)
        net_flux_1 = jump_1.mean(axis=0)
        norm_flow_1 = normalize(net_flux_1, axis=1)
    
        jump_6sites = [ results[temp][i][n][jump_type][flux_type][0]
         for i in np.arange(144,288) 
         for n in np.arange(0,6) 
         if len(results[temp][i][n][jump_type][flux_type])!=0 ] 
        jump_2 = np.array(jump_6sites)
        net_flux_2 = jump_2.mean(axis=0)
        norm_flow_2 = normalize(net_flux_2, axis=1)
        # norm_flow
        num_nodes = 6
        graph = {node: list(range(0, num_nodes + 1)) for node in range(0, num_nodes + 1)}
        def dfs(graph, start, end, path=[], paths=[]):
            if start == end:
                paths.append(path)
            elif len(path) < 4:  # Limit path length to 4 nodes
                for node in graph[start]:
                    if node not in path and node != start:  # Exclude self-nodes (loops)
                        dfs(graph, node, end, path + [(start, node)], paths)
            return paths
        all_paths = dfs(graph, start, end)
        S = []
        ############# 5 sites 
        for path in all_paths:
            flows = []
            for i in path:
                
                i = np.array(i)
                flows.append(norm_flow_1[i[0], i [1]])
            
            flows = np.array(flows)
            f_ = np.prod(np.array(flows))
            s_one_path = - f_ * np.log(f_)
            if math.isnan(s_one_path) == False:
                S.append(s_one_path)
        ################## 6 sites
        for path in all_paths:
            flows = []
            for i in path:
                
                i = np.array(i)
                flows.append(norm_flow_2[i[0], i [1]])
            
            flows = np.array(flows)
            f_ = np.prod(np.array(flows))
            s_one_path = - f_ * np.log(f_)
            if math.isnan(s_one_path) == False:
                S.append(s_one_path)
        S = np.array(S)
    elif sse_type == "lpscl_ii":
        jump = [ results[temp][i][n][jump_type][flux_type][0]
         for i in np.arange(0,288) 
         for n in np.arange(0,6) 
         if len(results[temp][i][n][jump_type][flux_type])!=0 ] 
        jump = np.array(jump)
        net_flux = jump.mean(axis=0)
        norm_flow = normalize(net_flux, axis=1)
        num_nodes = 6
        graph = {node: list(range(0, num_nodes + 1)) for node in range(0, num_nodes + 1)}
        def dfs(graph, start, end, path=[], paths=[]):
            if start == end:
                paths.append(path)
            elif len(path) < 4:  # Limit path length to 4 nodes
                for node in graph[start]:
                    if node not in path and node != start:  # Exclude self-nodes (loops)
                        dfs(graph, node, end, path + [(start, node)], paths)
            return paths
        all_paths = dfs(graph, start, end)
        S = []
        for path in all_paths:
            flows = []
            for i in path:
                i = np.array(i)
                flows.append(norm_flow[i[0], i [1]])
            flows = np.array(flows)
            f_ = np.prod(np.array(flows))
            s_one_path = - f_ * np.log(f_)
            if math.isnan(s_one_path) == False:
                S.append(s_one_path)
        S = np.array(S)
    elif sse_type == "lspscl":
        jump = [ results[temp][i][n][jump_type][flux_type][0]
         for i in np.arange(0,72) 
         for n in np.arange(0,20) 
         if len(results[temp][i][n][jump_type][flux_type])!=0 ] 
        jump = np.array(jump)
        net_flux = jump.mean(axis=0)
        if  len(net_flux.shape) ==0:
            S = 0
        else:
            norm_flow = normalize(net_flux, axis=1)
            num_nodes = 12
            graph = {node: list(range(0, num_nodes + 1)) for node in range(0, num_nodes + 1)}
            def dfs(graph, start, end, path=[], paths=[]):
                if start == end:
                    paths.append(path)
                elif len(path) < 4:  # Limit path length to 4 nodes
                    for node in graph[start]:
                        if node not in path and node != start:  # Exclude self-nodes (loops)
                            dfs(graph, node, end, path + [(start, node)], paths)
                return paths
            all_paths = dfs(graph, start, end)
            S = []
            for path in all_paths:
                flows = []
                for i in path:
                    i = np.array(i)
                    flows.append(norm_flow[i[0], i [1]])
                flows = np.array(flows)
                f_ = np.prod(np.array(flows))
                s_one_path = - f_ * np.log(f_)
                if math.isnan(s_one_path) == False:
                    S.append(s_one_path)
        S = np.array(S)
    return S.sum()

def S_p(sse_type, results, temp, ):
    """
    Calculate the path entorpy of all the paths
    """
    S = []
    if sse_type != "lspscl":
        for start in np.arange(0, 7):
            for end in np.arange(0, 7):
                if start != end:
                    s= Site_S_p(sse_type, results, temp, f"{start}_{end}", "tpt_net", start, end)
                    S.append(s)
    else:
        for start in np.arange(0, 13):
            for end in np.arange(0, 13):
                if start != end:
                    s= Site_S_p(sse_type, results, temp, f"{start}_{end}", "tpt_net", start, end)
                    S.append(s)
    print(f"Path entropy S_p = {np.sum(S)}")

