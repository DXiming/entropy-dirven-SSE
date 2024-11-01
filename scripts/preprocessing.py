import mdtraj as md
import pymatgen 
import numpy as np
import re

def load_traj(sse_type, md_type, temp, stride):
    """
    Load traj for processing
    """
    traj_prefix= f"./data/trajs/{sse_type}/{temp}K_stride_{stride}.h5"
    traj = md.load(f"./data/trajs/{sse_type}/{temp}K_stride_{stride}.h5")
    if sse_type == "lpscl_ii":
        length = np.array([4.285070, 4.285070, 4.040000])
        angles = np.array([90., 90., 90.])
    elif sse_type == "lpscl_i":
        length = np.array([4.710400, 4.146000, 4.282800])
        angles = np.array([90., 90., 90.])
    elif sse_type == "lspscl":
        length = np.array([3.684000, 3.685050, 5.006010 ])
        angles = np.array([89.92290, 90.07780, 89.47760])
    elif sse_type == "lpscl_iii":
        length = np.array([4.361220, 4.361220, 4.111800])
        angles = np.array([90., 90., 90.])
    unitcell_lengths = np.vstack([length]*traj.n_frames)
    unitcell_angles = np.vstack([angles]*traj.n_frames)
    
    topology = traj.topology
    
    traj.unitcell_lengths = unitcell_lengths
    traj.unitcell_angles  = unitcell_angles

    return traj

def rotate_x_axis(array_3d, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    rotated_array = np.dot(array_3d, rotation_matrix.T)
    return rotated_array
    
def OrderLi(traj, cage_list, cage_center, ):
    """
    This function is for ordering the Lithium cage list with it's initial potions 
    Input:
        d_x: Li - S distances projected to x axis
            type: Numpy array
            shape: (1, number of cage_center, number of lithium in this cage)
        d_y: Li - S distances projected to y axis
        d_z: Li - S distances projected to z axis
        cage_center: indexes of the cages' center
            type: Numpy array
            shape: (number of centers,)
    Output:
        lithium_order_array: the list that with right positions of lithium
            type: Numpy nd array
            shape: (number of cage center, lithium in cages, 2)
    """
    lithium_number = cage_list.shape[1]
    x_comp = np.array(traj[0].xyz[:,cage_list[:], 0])
    y_comp = np.array(traj[0].xyz[:,cage_list[:], 1])
    z_comp = np.array(traj[0].xyz[:,cage_list[:], 2])
    dist_map = cage_list.reshape(cage_list.shape[0] * cage_list.shape[1], 2)
    comps_xyz = md.compute_displacements(traj[300][0], np.array(dist_map)).reshape(1,cage_list.shape[0], cage_list.shape[1],3)
    d_x_pbc  = comps_xyz[:,:,:, 0]
    d_y_pbc  = comps_xyz[:,:,:, 1]
    d_z_pbc  = comps_xyz[:,:,:, 2]

    z_top = 0.15 # the max of the top Li in octahedral
    z_bot = -0.15 # the max of the bot Li in octahedral
    lithium_order_array = np.zeros((len(cage_center), cage_list.shape[1]), dtype=np.int64)
    for pair in np.arange(0, len(cage_center)):
        for position in np.arange(0, cage_list.shape[1]):
            dz = d_z_pbc[0, pair,position] 
            dx = d_x_pbc[0, pair,position]
            dy = d_y_pbc[0, pair,position]
            #print(dz)
            if dz >= z_top:
                lithium_order_array[pair, position] = 5  
            elif dz < z_bot:
                lithium_order_array[pair, position] = 0
            else:
                # clock-wise states seperation
                if ( dx > 0 ) and (dy >0):
                    lithium_order_array[pair, position] = 4
                elif (dx> 0 ) and (dy <=0):
                    lithium_order_array[pair, position] = 3
                elif (dx<= 0 ) and (dy <= 0):
                    lithium_order_array[pair, position] = 2
                elif (dx<= 0 ) and (dy > 0): 
                    lithium_order_array[pair, position] = 1
    return lithium_order_array


def CalSphere(lithium_selected, traj, sse_type):
    """
    This function calculates the surrounded Sulfur atoms 
    for each lithium atom in this lithium_index list
    Input:
        lithium_index: 
            type: 2d list 
            shape: (n)
        traj:
            type: mdtraj object
        sse_type:
            type: string 'lpscl'
    """
    topology = traj.topology
    lithium_index = [ atom.index for atom in topology.atoms if (re.search(r'Li', atom.name) != None) ]
    phosphorus_index = [ atom.index for atom in topology.atoms if (re.search(r'P', atom.name) != None) ]
    sulfur_index = [ atom.index for atom in topology.atoms if (re.search(r'S', atom.name) != None) ]
    cholride_index = [ atom.index for atom in topology.atoms if (re.search(r'Cl', atom.name) != None) ]
    silicon_index = [ atom.index for atom in topology.atoms if (re.search(r'Si', atom.name) != None) ]
    cutoff_small = 0.3 # in unit of nm
    cutoff_large = 0.6 # in unit of nm
    haystack_indices = []
    if sse_type != "lpscl_i":
        if sse_type == "lpscl_iii":
            target = np.load(f"./data/{sse_type}/cl_s_oct_center.npy")
        elif sse_type == "lpscl_ii":
            target = np.load(f"./data/{sse_type}/s_octahedral_center.npy")
        haystack_indices.extend(target)
    
        cage_list_large = np.array([ md.compute_neighbors(traj[0],cutoff=2.0, 
                                                          query_indices=np.array([lithium_selected[i]]),
                                                          haystack_indices=haystack_indices,)[0] 
                                    for i in np.arange(0, len(lithium_selected))])
        cage_paired_lists = [ [lithium_selected[i], cage_list_large[i][d]] for i in  np.arange(0, len(lithium_selected)) for d in np.arange(0,len(cage_list_large[0]))]
        cage_paired_lists = np.array(cage_paired_lists).reshape(len(lithium_selected),len(cage_list_large[0]),2)
        np.save(f"./data/{sse_type}/cage_paired_lists.npy", cage_paired_lists)
    else:
        target =  np.load(f"./data/{sse_type}/cholride_cages_center.npy")
        haystack_indices.extend(target)
        cage_list_large = [ md.compute_neighbors(traj[0],cutoff=0.6,
                                                      query_indices=np.array([lithium_selected[i]]), 
                                                      haystack_indices=haystack_indices,)[0] 
                                for i in np.arange(0, len(lithium_selected))]
        cage_list_large = [row[:4] for row in cage_list_large]
        cage_paired_lists = [ [lithium_selected[i], cage_list_large[i][d]] for i in  np.arange(0, len(lithium_selected)) for d in np.arange(0,len(cage_list_large[0]))]
        cage_paired_lists = np.array(cage_paired_lists).reshape(len(lithium_selected),len(cage_list_large[0]),2)
        np.save(f"./data/{sse_type}/cage_paired_lists.npy", cage_paired_lists)
    return cage_paired_lists

def CoordComp(traj, cage_paired_lists):
    num_center = cage_paired_lists.shape[0]
    frames = traj.n_frames
    dist_map = cage_paired_lists.reshape(cage_paired_lists.shape[0] * cage_paired_lists.shape[1], 2)
    distances = md.compute_distances(traj, np.array(dist_map)).reshape(frames, num_center, len(cage_paired_lists[0]))
    # new version that handles the pbc with inside features of mdtraj
    comps_xyz = md.compute_displacements(traj, np.array(dist_map)).reshape(frames, num_center, len(cage_paired_lists[0]), 3)
    x_dis  = comps_xyz[:,:,:, 0]
    y_dis  = comps_xyz[:,:,:, 1]
    z_dis  = comps_xyz[:,:,:, 2]
    return {"x_dis":x_dis, "y_dis":y_dis, "z_dis":z_dis, "dist": distances}

def DiscreteMD(dis_dic,cage_number, temp, sse_type):
    """
    This function is for getting the discreted MD trajs
    Input:
        dist_dic: 
            type: dictionary
            shape:  {"x_dis":x_dis, "y_dis":y_dis, "z_dis":z_dis, "dist": distances}
    Output:
        states_dic
            type: dictionary
            shape: {state1 to state6}
    """
    distances = dis_dic["dist"]
    x_dis_pbc = dis_dic["x_dis"]
    y_dis_pbc = dis_dic["y_dis"]
    z_dis_pbc = dis_dic["z_dis"]
    states = {}
    d_0 = 0.30 # the max of Li-S bond 
    z_top = 0.15 # the max of the top Li in octahedral
    z_bot = -0.15 # the max of the bot Li in octahedral
    for d in np.arange(0,distances.shape[1]):
        states[d] = []
        for i in np.arange(0, distances.shape[0]):
            min_li_s = distances[i][d].min()
        
            if min_li_s > d_0 :
                #print("intracage")
                states[d].append(7)
            else:
                proj_index = np.argmin(np.abs(distances[i][d]))
                z_proj = z_dis_pbc[i][d][proj_index]
                if z_proj > z_top:
                    #print("Top state")
                    states[d].append(1)
                elif z_proj < z_bot:
                    #print("Bottom state")
                    states[d].append(6)
                else:
                    #print("Plane state")
                    
                    ## More accurate discription of the possible states
                    x_proj = x_dis_pbc[i][d][proj_index]
                    y_proj = y_dis_pbc[i][d][proj_index]
                    # clock-wise states seperation
                    if (x_proj > 0 ) and (y_proj >0):
                        states[d].append(2)
                    elif (x_proj > 0 ) and (y_proj <=0):
                        states[d].append(3)
                    elif (x_proj <= 0 ) and (y_proj <= 0):
                        states[d].append(4)
                    elif (x_proj <= 0 ) and (y_proj > 0):
                        states[d].append(5)
    from pathlib import Path
    Path(f"./data/{sse_type}/{temp}K/discrete_trajs").mkdir(parents=True, exist_ok=True)                
    np.save(f"./data/{sse_type}/{temp}K/discrete_trajs/states_{cage_number}.npy", states)
    return states

def SoftShell(traj, ):
    """
    This function returns the soft shell of lpscl_iii
    args:
        traj
    returns:
        cage_{5,6} and cage_center
    """
    topology = traj.topology
    lithium_index = [ atom.index for atom in topology.atoms if (re.search(r'Li', atom.name) != None) ]
    phosphorus_index = [ atom.index for atom in topology.atoms if (re.search(r'P', atom.name) != None) ]
    sulfur_index = [ atom.index for atom in topology.atoms if (re.search(r'S', atom.name) != None) ]
    cholride_index = [ atom.index for atom in topology.atoms if (re.search(r'Cl', atom.name) != None) ]
    silicon_index = [ atom.index for atom in topology.atoms if (re.search(r'Si', atom.name) != None) ]
    # find S that do not place in octahedral site center (P coordinated)
    # if the S or cl are coordinated with more that four lithium atoms we think they are in the oct-center
    cl_s_full_list = np.concatenate((sulfur_index,cholride_index) )
    coordinates = [len(md.compute_neighbors(traj[0],cutoff=0.25, 
                                            query_indices=[cl_s_full_list[i]], 
                                            haystack_indices=lithium_index,)[0]) 
                   for i in np.arange(0, len(cl_s_full_list)) ]
    mask = np.array(coordinates) >= 4
    
    cl_s_oct_center = cl_s_full_list[mask]
    np.save("./data/lpscl_iii/cl_s_oct_center.npy", cl_s_oct_center)
    print(f"we finally get the oct centers index with shape: {cl_s_oct_center.shape}")
    lithium_in_cage = [ md.compute_neighbors(traj[0],cutoff=0.25, 
                                             query_indices=np.array([cl_s_oct_center[i]]),
                                             haystack_indices=lithium_index,)[0] 
                                    for i in np.arange(0, len(cl_s_oct_center))]
    
    
    lithium_cage_lists = [ [cl_s_oct_center[i], lithium_in_cage[i][d]] 
                          for i in  np.arange(0, len(cl_s_oct_center)) 
                          for d in np.arange(0,len(lithium_in_cage[i]))]
    #lithium_cage_lists = np.array(lithium_cage_lists).reshape(len(s_octahedral_center),len(lithium_in_cage[0]),2)
    
    # Dictionary to store grouped sublists

    grouped_lithium_cages = {}
    
    # Group sublists based on the first element
    for sublist in lithium_cage_lists:
        key = sublist[0]
        if key not in grouped_lithium_cages:
            grouped_lithium_cages[key] = []
        grouped_lithium_cages[key].append(sublist)
    
    # Convert the dictionary values to a list of lists
    cage_lists = list(grouped_lithium_cages.values())
    
    cage_5 = []
    cage_6 = []
    for sublist in cage_lists:
        if len(sublist) ==6:
            cage_6.append(sublist)
        else:
            cage_5.append(sublist)
   
    cage_5 = np.array(cage_5)
    cage_6 = np.array(cage_6)
    cage_5_center = np.unique(cage_5[:,:,0].flatten())
    cage_5_cl_ratio = np.sum(np.isin(cage_5_center, cholride_index).astype(int))/len(cage_5_center)
    cage_6_center = np.unique(cage_6[:,:,0].flatten())
    cage_6_cl_ratio = np.sum(np.isin(cage_6_center, cholride_index).astype(int))/len(cage_6_center)
    
    lithium_order_array_5 = OrderLi(traj, cage_5, cage_5_center, )
    lithium_order_array_5 = np.argsort(lithium_order_array_5, axis= 1)
    lithium_order_array_6 = OrderLi(traj, cage_5, cage_5_center, )

    lithiums_6 = cage_6[:,:,1]
    ordered_cage_6_center = np.array([ lithiums_6[i][lithium_order_array_6[i]] 
                                      for i in np.arange(0,144) ])
    lithiums_5 = cage_5[:,:,1]
    ordered_cage_5_center = np.array([ lithiums_5[i][lithium_order_array_5[i]] 
                                      for i in np.arange(0,144) ])
    
    #print(f"We finally get the ordered cages of all Lithium")
    return lithiums_5, lithiums_6
def MarkovTraj(traj, temp, sse_type):
    """
    Get the discrete Markov traj
    """
    if sse_type == "lpscl_iii":
        lithium_5, lithium_6 = SoftShell(traj)
        cl_s_oct_center = np.load(f"./data/{sse_type}/cl_s_oct_center.npy")
        cage_shape = lithium_5.shape[0]
        for cage_number in np.arange(0, 2):
    
            cage_paired_lists  = CalSphere(lithium_5[cage_number], traj, sse_type)
            dis_dic = CoordComp(traj,cage_paired_lists )
            states = DiscreteMD(dis_dic,cage_number, temp, sse_type)
        for cage_number in np.arange(0, 2):
            cage_paired_lists  = CalSphere(lithium_6[cage_number], traj, sse_type)
            dis_dic = CoordComp(traj,cage_paired_lists )
            states = DiscreteMD(dis_dic,cage_number+144, temp, sse_type)
        print("All done.")
    elif sse_type == "lpscl_ii":
        li_ordered_cages =  np.load(f"./data/{sse_type}/li_ordered_cages.npy")
        cage_shape = li_ordered_cages.shape[0]
        s_octahedral_center =  np.load(f"./data/{sse_type}/s_octahedral_center.npy")
        for cage_number in np.arange(0, cage_shape):
            cage_paired_lists  = CalSphere(li_ordered_cages[cage_number], traj, sse_type)
            dis_dic = CoordComp(traj, cage_paired_lists)
            states = DiscreteMD(dis_dic,cage_number, temp, sse_type)
        print("All done.")
    elif sse_type == "lpscl_i":
        li_ordered_cages = np.load("./data/lpscl_i/li_ordered_cages.npy")
        cage_shape = li_ordered_cages.shape[0]
        for cage_number in np.arange(0, cage_shape):
            cage_paired_lists = CalSphere(li_ordered_cages[cage_number], traj, "lpscl_i")
            num_center=5
            frames = traj.n_frames
            dist_map = cage_paired_lists.reshape(
                cage_paired_lists.shape[0] * cage_paired_lists.shape[1], 2)
            distances = md.compute_distances(traj,np.array(dist_map)).reshape(frames, num_center, len(cage_paired_lists[0]))
            s_x = np.array(traj.xyz[:,cage_paired_lists[:], 0])
            s_y = np.array(traj.xyz[:,cage_paired_lists[:], 1])
            s_z = np.array(traj.xyz[:,cage_paired_lists[:], 2])
            x_dis  = s_x[:,:,:,1] - s_x[:, :, :,  0]
            y_dis  = s_y[:,:,:,1] - s_y[:, :, :,  0]
            z_dis  = s_z[:,:,:,1] - s_z[:, :, :,  0]
            v_li = np.array([[x_dis[i][d][m], y_dis[i][d][m], z_dis[i][d][m]] 
                             for i in np.arange(0, x_dis.shape[0]) 
                             for d in np.arange(0, x_dis.shape[1])
                             for m in np.arange(0, x_dis.shape[2])]
                           )
            # calculate the projections
            x_norm = np.array([1,0,0])
            y_norm = np.array([0,1,0])
            z_norm = np.array([0,0,1])
            n_x = x_norm / np.linalg.norm(x_norm)
            n_y = y_norm / np.linalg.norm(y_norm)
            n_z = z_norm / np.linalg.norm(z_norm)
            v_li_rot = rotate_x_axis(v_li, 60)
            v_p_x =  np.array([np.dot(v_li_rot[i], x_norm)  * n_x for i in np.arange(0, len(v_li))]).reshape((frames, 5, 4, 3))
            v_p_y =  np.array([np.dot(v_li_rot[i], y_norm)  * n_y for i in np.arange(0, len(v_li))]).reshape((frames, 5, 4, 3))
            v_p_z =  np.array([np.dot(v_li_rot[i], z_norm)  * n_z for i in np.arange(0, len(v_li))]).reshape((frames, 5, 4, 3))
            v_li_rot  = v_li_rot.reshape((frames, 5, 4, 3))
            d_0 = 0.45
            back_top = 0.10
            back_bot = -0.10
            states = {}
            for d in np.arange(0,distances.shape[1]):
                states[d] = []
                for i in np.arange(0, distances.shape[0]):
                    min_li_s = distances[i][d].min()
                    if min_li_s > d_0 :
                        states[d].append(5)
                    else:
                        proj_index = np.argmin(np.abs(distances[i][d]))
                        dx = v_li_rot[i, d, proj_index, 0]
                        dy = v_li_rot[i, d, proj_index, 1]
                        dz = v_li_rot[i, d, proj_index, 2]
                        if (dz > back_bot) and (dz < back_top) :
                            states[d].append(4) # the larges will be in the first 
                        elif (dy >= 0 )  and (dz < 0) :
                            states[d].append(0)
                        elif (dy > 0 )  and ( dz > 0) :
                            states[d].append(1)        
                        elif (dx <= 0 )  and ( dz >= 0) :
                            states[d].append(2)
                        elif (dx < 0 )  and ( dy <= 0) :
                            states[d].append(3)            
            from pathlib import Path
            Path(f"./data/{sse_type}/{temp}K/discrete_trajs").mkdir(parents=True, exist_ok=True)                
            np.save(f"./data/{sse_type}/{temp}K/discrete_trajs/states_{cage_number}.npy", states)
        print("All done.")