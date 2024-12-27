import numpy as np
import pandas as pd
import mdtraj as md
import pymatgen   
import re

def SortCutoffs(traj, list,lithium_selected):
    """
    This function is for reordering the neighbors based on the distance between two speices
    """
    original_distances =  [md.compute_distances(traj[0], [[list[i][d],lithium_selected[i]] for d in np.arange(0, len(list[i]))]) 
                       for i in np.arange(len(list))]
    indices = [np.argsort(arr) for arr in original_distances]
    sorted_list = [ list[i][indices[i]][0] for i in np.arange(0, len(list)) ]
    sorted_distances  = [md.compute_distances(traj[0], [[sorted_list[i][d],lithium_selected[i]] for d in np.arange(0, len(list[i]))]) 
                       for i in np.arange(len(list))]
    return sorted_list

def CalSphere(lithium_selected, traj, cutoff):
    """
    This function calculates the surrounded Sulfur atoms 
    for each lithium atom in this lithium_index list
    Input:
        lithium_index: 
            type: 2d list 
            shape: (n)
        traj:
            type: mdtraj object
        cutoff:
            type: int
            shape: 1
    Ouput:
        numpy arrays

    """
    topology = traj.topology
    lithium_index = [ atom.index for atom in topology.atoms if (re.search(r'Li', atom.name) != None) ]
    phosphorus_index = [ atom.index for atom in topology.atoms if (re.search(r'P', atom.name) != None) ]
    sulfur_index = [ atom.index for atom in topology.atoms if (re.search(r'S', atom.name) != None) ]
    cholride_index = [ atom.index for atom in topology.atoms if (re.search(r'Cl', atom.name) != None) ]
    silicon_index = [ atom.index for atom in topology.atoms if (re.search(r'Si', atom.name) != None) ]
    cage_si_center = np.load("./data/lspscl/cage_in_si_center.npy")
    cage_p_center = np.load("./data/lspscl/cage_in_p_center.npy")
    cage_si_bridge = np.load("./data/lspscl/cage_in_si_bridge.npy")
    haystack_indices_si_center = []
    haystack_indices_si_center.extend(cage_si_center)
    
    haystack_indices_p_center = []
    haystack_indices_p_center.extend(cage_p_center)
    
    haystack_indices_si_bridge = []
    haystack_indices_si_bridge.extend(cage_si_bridge) 
    cage_list_si_center = np.array([ md.compute_neighbors(traj[0],cutoff=cutoff, query_indices=np.array([lithium_selected[i]]), haystack_indices=haystack_indices_si_center,)[0]
                      for i in np.arange(0, len(lithium_selected))], dtype="object")
    cage_list_p_center  = np.array([ md.compute_neighbors(traj[0],cutoff=cutoff, query_indices=np.array([lithium_selected[i]]), haystack_indices=haystack_indices_p_center,)[0]
                      for i in np.arange(0, len(lithium_selected))], dtype="object")
    cage_list_si_bridge = np.array([ md.compute_neighbors(traj[0],cutoff=cutoff, query_indices=np.array([lithium_selected[i]]), haystack_indices=haystack_indices_si_bridge,)[0]
                      for i in np.arange(0, len(lithium_selected))], dtype="object")

    cage_list_si_center = SortCutoffs(traj, cage_list_si_center, lithium_selected)
    cage_list_p_center  = SortCutoffs(traj, cage_list_p_center,  lithium_selected)
    cage_list_si_bridge = SortCutoffs(traj, cage_list_si_bridge, lithium_selected)
    least_si_center     = np.array([ len(arr) for arr in cage_list_si_center]).min()
    least_p_center      = np.array([ len(arr) for arr in cage_list_p_center]).min()
    least_si_bridge     = np.array([ len(arr) for arr in cage_list_si_bridge]).min()
    cage_list_si_center = np.array([ arr[:least_si_center] for arr in cage_list_si_center])
    cage_list_p_center  = np.array([ arr[:least_p_center] for arr in cage_list_p_center])
    
    si_paired_lists = np.array([
        [[lithium_selected[d], cage_list_si_center[d][ref]]
         for ref in np.arange(0, least_si_center)]
        for d in np.arange(0,len(lithium_selected))
    ])
    
    p_paired_lists = np.array([
        [[lithium_selected[d], cage_list_p_center[d][ref]]
        for ref in np.arange(0, least_p_center)] 
        for d in np.arange(0,len(lithium_selected)) 
    ])
    cage_list_si_bridge = np.array([ arr[:least_si_bridge] for arr in cage_list_si_bridge])
    top_pairs = np.load("./data/lspscl/top_pairs.npy")
    bot_pairs = np.load("./data/lspscl/bot_pairs.npy")

    filtered_bot = [bot_pairs[np.isin(bot_pairs, cage_list_si_bridge[i]).astype("int64").sum(axis=1) >= 1] for i in np.arange(0,21)] 
    filtered_top = [top_pairs[np.isin(top_pairs, cage_list_si_bridge[i]).astype("int64").sum(axis=1) >= 1] for i in np.arange(0,21)] 
    least_filter_bp = int(least_si_bridge/2)
    filtered_bot = np.array([ arr[:least_filter_bp] for arr in filtered_bot])
    filtered_top = np.array([ arr[:least_filter_bp] for arr in filtered_top])
    top_paired_lists = np.array([
        [[lithium_selected[d], top[0]],
         [lithium_selected[d], top[1]]] 
        for d in np.arange(0,len(lithium_selected)) for top in filtered_top[d]]).reshape((len(lithium_selected), least_filter_bp, 2, 2))
    bot_paired_lists = np.array([
        [[lithium_selected[d], bot[0]],
         [lithium_selected[d], bot[1]]] 
        for d in np.arange(0,len(lithium_selected)) for bot in filtered_bot[d]]).reshape((len(lithium_selected), least_filter_bp, 2, 2))
        
    return {"si":si_paired_lists, "p": p_paired_lists, "top":top_paired_lists, "bot": bot_paired_lists}


def rotate_x_axis(array_3d, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    rotated_array = np.dot(array_3d, rotation_matrix.T)
    return rotated_array

def CoordComp(traj, list,  num_center):
    """
    Get the x,y,z componenets of the paired list, also the distance info
    """
    frames = traj.n_frames
     
    if list.ndim ==3 :

        # calculate the distances in the maped arrays
        num_center= 21
        frames = traj.n_frames
        dist_map = list.reshape(list.shape[0] * list.shape[1], 2)
        distances = md.compute_distances(traj, np.array(dist_map)).reshape(frames, num_center, len(list[0]))
        # calculates the components
    ##************* modified version, using md.displacement since it already incorprates the PBC  and the efficiency is higher
        comps_xyz = md.compute_displacements(traj, np.array(dist_map)).reshape(frames, num_center, len(list[0]), 3)
        v_li_rot = rotate_x_axis(comps_xyz, 45)
        x_dis_pbc  = v_li_rot[:,:,:, 0]
        y_dis_pbc  = v_li_rot[:,:,:, 1]
        z_dis_pbc  = v_li_rot[:,:,:, 2]
    ##****************************
 
    elif list.ndim  == 4:
        #calculate the distances
        num_center= 21
        frames = traj.n_frames
        dist_map = list.reshape(list.shape[0] * list.shape[1] * list.shape[2], 2)
        distances = md.compute_distances(traj, np.array(dist_map)).reshape(frames, num_center, len(list[0]), 2)
        #calcuialte the components
    ##************* modified version, using md.displacement since it already incorprates the PBC  and the efficiency is higher
        comps_xyz = md.compute_displacements(traj, np.array(dist_map)).reshape(frames, num_center, len(list[0]),2, 3)
        v_li_rot = rotate_x_axis(comps_xyz, 45)
        x_dis_pbc  = v_li_rot[:,:,:,:,0]
        y_dis_pbc  = v_li_rot[:,:,:,:,1]
        z_dis_pbc  = v_li_rot[:,:,:,:,2]
    ##****************************
    return {"x_dis":x_dis_pbc, "y_dis":y_dis_pbc, "z_dis":z_dis_pbc, "dist": distances}

def GetCageInfo(traj, si_paired_lists, p_paired_lists, top_paired_lists, bot_paired_lists,temp,cage):
    import os 
    """
    This function fetch the cages info and store it to files.
    For next step discretization of the Lithium
    Input:
        The cage paired lists
    Output:
        The cages info containing distances and x,y,z components
    """
    from pathlib import Path
    Path(f"./data/lspscl/{temp}K/data").mkdir(parents=True, exist_ok=True)   
    
    save_prefix = f"./data/lspscl/{temp}K/data"
    file = f"{save_prefix}/{cage}_cage_1.npy"
    if os.path.isfile(file) == 0 :
        cage_1 = CoordComp(traj, si_paired_lists,  21)
        cage_2 = CoordComp(traj, p_paired_lists,  21)
        bridge_top = CoordComp(traj, top_paired_lists,  21)
        bridge_bot = CoordComp(traj, bot_paired_lists,  21)

    
        np.save(f"{save_prefix}/{cage}_cage_1.npy", cage_1,)
        np.save(f"{save_prefix}/{cage}_cage_2.npy", cage_2,)
        np.save(f"{save_prefix}/{cage}_bridge_top.npy", bridge_top,)
        np.save(f"{save_prefix}/{cage}_bridge_bot.npy", bridge_bot,)
    else:
        cage_1 = np.load(f'{save_prefix}/{cage}_cage_1.npy',allow_pickle='TRUE').item()
        cage_2 = np.load(f'{save_prefix}/{cage}_cage_2.npy',allow_pickle='TRUE').item()
        bridge_top = np.load(f'{save_prefix}/{cage}_bridge_top.npy',allow_pickle='TRUE').item()
        bridge_bot = np.load(f'{save_prefix}/{cage}_bridge_bot.npy',allow_pickle='TRUE').item()
    return {"cage_1":cage_1, "cage_2":cage_2, "bridge_top":bridge_top, "bridge_bot": bridge_bot}


def DiscreteMD(traj, dis_dic,cage_number, temp):
    """
    This function is for getting the discreted MD trajs
    Input:
        dist_dic: Calculted from `GetCageInfo` (CoordComp)
            type: dictionary
            shape:  {"cage_1":cage_1, "cage_2":cage2, "bridge_top":bridge_top, "bridge_bot": bridge_bot}
    Output:
        states_dic
            type: dictionary
            shape: {state1 to state12}
    """
    states = {}
    cage_1 = dis_dic["cage_1"]
    cage_2 = dis_dic["cage_2"]
    bridge_top = dis_dic["bridge_top"]
    bridge_bot = dis_dic["bridge_bot"]
    for d in np.arange(1, 21):
        lithium_index = d
        states[d] = []
        for frame in np.arange(0, traj.n_frames):
            cage_1_bonds = cage_1['dist'][frame][lithium_index]
            cage_2_bonds = cage_2['dist'][frame][lithium_index]
            bridge_top_bonds      = bridge_top['dist'][frame][lithium_index]
            bridge_top_bonds_mean = bridge_top['dist'][frame][lithium_index].mean(axis=1)
            bridge_bot_bonds      = bridge_bot['dist'][frame][lithium_index]
            bridge_bot_bonds_mean = bridge_bot['dist'][frame][lithium_index].mean(axis=1)
            
            bridge_top_bonds_mean_min_index = np.argmin(bridge_top_bonds_mean)
            bridge_bot_bonds_mean_min_index = np.argmin(bridge_bot_bonds_mean)
            
            cage_1_bmin = np.min(cage_1_bonds)
            cage_1_bmin_index = np.argmin(cage_1_bonds)
            cage_2_bmin = np.min(cage_2_bonds)
            cage_2_bmin_index = np.argmin(cage_2_bonds)
            top_b_all_min = np.min(bridge_top_bonds_mean)
            bot_b_all_min = np.min(bridge_bot_bonds_mean)
            
            # In cage_1, or cage_2 sites
            if (cage_1_bmin <= 0.40) | (cage_2_bmin <= 0.40):
                if cage_1_bmin < cage_2_bmin:
                #if (cage_1_bmin <= 0.5) & (cage_1_bmin < cage_2_bmin):
                    if cage_2_bonds[cage_1_bmin_index] <=0.35  :
                        z_p = cage_1['z_dis'][frame][lithium_index][index]
                        if z_p > 0 :
                            states[d].append(1)
                        elif z_p <= 0:
                            states[d].append(0)
                    else:
                        index = np.argmin(cage_1_bonds)
                        x_p = cage_1['x_dis'][frame][lithium_index][index]
                        y_p = cage_1['y_dis'][frame][lithium_index][index]  
                        z_p = cage_1['z_dis'][frame][lithium_index][index]
                        if (y_p >0) & (x_p > 0 ) :
                            states[d].append(2)
                        elif (y_p <= 0) & (x_p > 0 ):
                            states[d].append(3)
                        elif (y_p >0)  & (x_p <= 0 ):
                            states[d].append(4)
                        elif (y_p <= 0) & (x_p <= 0 ):
                            states[d].append(5)
                        else:
                            states[d].append(12)
                elif cage_1_bmin >= cage_2_bmin  :
                    if cage_1_bonds[cage_2_bmin_index] <=0.35  :
                        z_p = cage_2['z_dis'][frame][lithium_index][index]
                        if z_p > 0 :
                            states[d].append(0)
                        elif z_p < 0:
                            states[d].append(1)                            
                    else:
                        index = np.argmin(cage_2_bonds)
                        x_p = cage_2['x_dis'][frame][lithium_index][index]
                        y_p = cage_2['y_dis'][frame][lithium_index][index]
                        z_p = cage_2['z_dis'][frame][lithium_index][index]
                        if (y_p >0) & (x_p < 0): 
                            states[d].append(6)
                        elif (y_p <= 0) & (x_p <= 0) :
                            states[d].append(7)
                        elif (y_p >0) &  (x_p > 0 ) :               
                            states[d].append(8)
                        elif (y_p <= 0) & (x_p > 0 ) :               
                            states[d].append(9)
                        else:
                            states[d].append(12)
            else:
                if bridge_top_bonds_mean.min() < bridge_bot_bonds_mean.min() : 
                    if (((bridge_top_bonds<=0.35).sum(axis=1) == 2) == True).any():
                        if (bridge_top_bonds[bridge_top_bonds_mean_min_index,:] >=0.25).sum() == 2: 
                            states[d].append(10)
                        else:
                            states[d].append(12)
                    else:
                        states[d].append(12)
                elif bridge_bot_bonds_mean.min() <= bridge_top_bonds_mean.min():
                    if (((bridge_bot_bonds<=0.35).sum(axis=1) == 2) == True).any():
                        if (bridge_bot_bonds[bridge_bot_bonds_mean_min_index,:] >= 0.25).sum() == 2: 
                    #elif bot_b_all_min <= 0.35:
                            states[d].append(11)
                        else:
                            states[d].append(12)
                    else:
                        states[d].append(12)
                else:
                    states[d].append(12) # outside the states intra states
    from pathlib import Path
    Path(f"./data/lspscl/{temp}K/discrete_trajs").mkdir(parents=True, exist_ok=True)   
    
    save_prefix = f"./data/lspscl/{temp}K/discrete_trajs"
    np.save(f"{save_prefix}/{cage_number}_states.npy", states)
    #return states


def MarkovTraj(_traj, temp, cage_number):
    li_cages_merged = np.load("./data/lspscl/li_cages_merged.npy")
    temps = [ temp,]
    for temp in temps:
        traj = {}
        traj[temp] = _traj 
        coord_dic = CalSphere(li_cages_merged[cage_number], traj[temp], cutoff=3.0)
        
        cage_1 = CoordComp(traj[temp], coord_dic["si"],  21)
        cage_2 = CoordComp(traj[temp], coord_dic["p"],  21)
        bridge_top = CoordComp(traj[temp], coord_dic["top"] ,  21)
        bridge_bot = CoordComp(traj[temp], coord_dic["bot"],  21)
        dis_dic =  {"cage_1":cage_1, "cage_2":cage_2, "bridge_top":bridge_top, "bridge_bot": bridge_bot}
        states = DiscreteMD(traj[temp], dis_dic,cage_number, temp)