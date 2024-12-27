import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def PlotDtraj(states, n):
    """
    This function plots the discreted trajs for only one atom
    """

    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.scatter(np.arange(0,  len(states[n]))/100 *2, np.array(states[n]),
               c= states[n],
               s= 0.1, vmin=1.5, vmax=7,
               cmap = "rainbow")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Lithium states")
    #ax.set_yticklabels(["", "LS1","LS2","LS3","LS4","LS5", "LS6", "LS7"])
    plt.show()

def rotate_x_axis(array_3d, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    rotated_array = np.dot(array_3d, rotation_matrix.T)
    return rotated_array

def PlotProj(traj,sse_type, cage_number,):
    """
    Plot the projected traj
    """
    if sse_type != "lpscl_i" and sse_type != "lspscl":
        if sse_type == "lpscl_iii":
            if cage_number >= 144:
        
                cages = np.load(f"./data/{sse_type}/300K/ordered_cage_6_center.npy")
                states = np.load(f"./data/{sse_type}/300K/discrete_trajs/states_{cage_number}.npy",
                            allow_pickle='TRUE').item()
                cage_number= cage_number-144
            else:
                cages =  np.load(f"./data/{sse_type}/300K/ordered_cage_5_center.npy")
                states = np.load(f"./data/{sse_type}/300K/discrete_trajs/states_{cage_number}.npy",
                            allow_pickle='TRUE').item()
        elif sse_type == "lpscl_ii":
            cages = np.load(f"./data/{sse_type}/li_ordered_cages.npy")
            states = np.load(f"./data/{sse_type}/300K/discrete_trajs/states_{cage_number}.npy",
                                    allow_pickle='TRUE').item()
            
        fig, ax = plt.subplots(1,1)
        scatter={}
        all_x= traj.xyz[:,cages[cage_number],:][:,:,0],
        all_y= traj.xyz[:,cages[cage_number],:][:,:,1]
        cs = np.moveaxis(np.array([ arr for key, arr in states.items()]), 0, -1)
        all_fig=ax.scatter(all_x,
                 all_y,
                 c=cs,
                 s= 0.1, vmin=1.5, vmax=7,
                 cmap="rainbow",
                 alpha=.3
                 )    
        
        legend_labels = ['Top', 'Planar-1','Planar-2','Planar-3','Planar-4', 'Bottom', 'External']
        legend = ax.legend(*all_fig.legend_elements(), 
                                title='Lithium states',
                                loc='lower center',
                                bbox_to_anchor=(0.5, -0.4),
                                fancybox=True,
                                shadow=True,
                                ncol=4,
                               )
        legend_labels = ['LS1 (Top)', 
                         'LS2 (Planar-1)',
                         'LS3 (Planar-2)',
                         'LS4 (Planar-3)',
                         'LS5 (Planar-4)',
                         'LS6 (Bottom)', 
                         'LS7 (External)']
    elif sse_type == "lpscl_i":
        fig, ax = plt.subplots(1,1)
        scatter={}
        cages = np.load("./data/lpscl_i/li_ordered_cages.npy")
        states = np.load(f"./data/{sse_type}/300K/discrete_trajs/states_{cage_number}.npy",
                            allow_pickle='TRUE').item()
        li_xyz_rot = rotate_x_axis(traj.xyz,60)
        all_x= li_xyz_rot[:,cages[cage_number],:][:,:,0],
        all_y= li_xyz_rot[:,cages[cage_number],:][:,:,2]
        cs = np.moveaxis(np.array([ arr for key, arr in states.items()]), 0, -1)
        all_fig=ax.scatter(all_x,
                 all_y,
                 c=cs,
                 s= 0.1, vmin=1.5, vmax=7,
                 cmap="rainbow",
                 alpha=.3
                 )
        legend = ax.legend(*all_fig.legend_elements(), 
                                title='Lithium states',
                                loc='lower center',
                                bbox_to_anchor=(0.5, -0.4),
                                fancybox=True,
                                shadow=True,
                                ncol=4,
                               )
        legend_labels = ['LS1', 
                         'LS2',
                         'LS3',
                         'LS4',
                         'LS5',]
    elif sse_type == "lspscl":
        fig, ax = plt.subplots(1,1)
        scatter={}
        cages = np.load(f"./data/{sse_type}/li_cages_merged.npy")
        states = np.load(f"./data/{sse_type}/300K/discrete_trajs/{cage_number}_states.npy",
                            allow_pickle='TRUE').item()
        li_xyz = rotate_x_axis(traj.xyz, 45)
        all_x= li_xyz[:,cages[cage_number][1:],:][:,:,0],
        all_y= li_xyz[:,cages[cage_number][1:],:][:,:,2]
        cs = np.moveaxis(np.array([ arr for key, arr in states.items()]), 0, -1)
        all_fig=ax.scatter(all_x,
                 all_y,
                 c=cs,
                 s= 0.1, vmin=0, vmax=12,
                 cmap="rainbow",
                 alpha=.9
                 )
        legend = ax.legend(*all_fig.legend_elements(), 
                                title='Lithium states',
                                loc='lower center',
                                bbox_to_anchor=(0.5, -0.4),
                                fancybox=True,
                                shadow=True,
                                ncol=4,
                               )
        legend_labels = ['LS1','LS2','LS3','LS4','LS5','LS6','LS7','LS8','LS9','LS10','LS11','LS12','LS13' ]
    for d in np.arange(0, len(legend_labels)):
        legend.get_texts()[d].set_text(legend_labels[d])
    ax.set_xlabel("Dimension-1", labelpad=10)
    ax.set_ylabel("Dimension-2", labelpad=10)
    plt.show()


