def AngleDis(traj, sse, temp, element ):
    """
    This function calculate the theta and phi for all P-S amd P-Si units
    Input:
        traj: mdtraj objec
        paired_list: former saved paired list of the tetrahedron units
            type: numpy 3d array
            shape: (number of tetrahedron, number of pairs (4), pair (2) )
        sse: 
            type: string
            shape: None
        temp:
            type: int
        element: P or Si center
    Output:
        theta, phi in pandas Dataframe
    """
    prefix = f"./general/rotation/tetrahedral/{sse}/{temp}K_nn_{sse}_{element}_paired_lists.npy"
    paired_lists=np.load(prefix)
    print(f"Load paried_lists with shape: {paired_lists.shape}")
    theta = []
    phi = []
    for sulfur_index in np.arange(0, paired_lists.shape[0]):
        for index in [0, 1, 2, 3]:
            sulfur =  np.array(traj.xyz[:,paired_lists[sulfur_index][index][1], :]) 
            p =  np.array(traj.xyz[:,paired_lists[sulfur_index][1][0], :]) 
            vector = sulfur - p
        
            # get the projected vector
            projected = vector.copy()
            projected[:,2]=0
            # the param 
            
            vert_vec = np.array([0, 0 ,1])
            x_vec  = np.array([1, 0, 0])
            # using this np.arccos only get the absolute values in [0, pi]
            theta_ = np.degrees(np.arccos(np.dot(vector, vert_vec) / (np.linalg.norm(vector, axis=1) * np.linalg.norm(vert_vec))))
            phi_ = np.degrees(np.arccos(np.dot(projected, x_vec) / (np.linalg.norm(projected, axis=1) * np.linalg.norm(vert_vec))))
            
            #mask = np.dot(vector, vert_vec) < 0
            #mask = np.where(mask, 1, -1)
            #print(np.dot(vector, vert_vec).shape)
            #theta_ = mask * theta_
            
            theta.append(theta_)
            
            mask_p = np.cross(projected, x_vec)[:,-1] < 0
            #print(mask_p.shape)
            mask_p = np.where(mask_p, 1, -1)
            #print(mask.min(), mask_p.min())
            phi_ = mask_p * phi_
            phi.extend(phi_)
            # Inside, we have extended this to have values in [-pi, pi]
    theta = np.array(theta)
    phi = np.array(phi)
    print("original shape: ", theta.shape, phi.shape)
    
    theta = theta.flatten()
    phi = phi.flatten()
    print("final shape: ",theta.shape, phi.shape)
    angles = pd.DataFrame(theta, phi )
    angles = angles.reset_index()
    angles.columns=[r"$\theta$",r"$\phi$"]
    return angles

def PlotDF(angles, sse, temp, element, bins ):
    """
    This function plots the free energy and the distribution of the theta and phi
    Input:
        angles: DF
    Output:
        figures and save figures 
    """
    ## calculate the free energy
    theta = angles[r"$\theta$"].to_numpy()
    phi=angles[r"$\phi$"].to_numpy()
    theta_hist, theta_bin_edges = np.histogram(theta, bins=bins,range=(0,180),weights=np.ones(len(theta)), density=True)
    phi_hist, phi_bin_edges = np.histogram(phi, bins=bins,range=(-180,180),weights=np.ones(len(phi)), density=True)
    
    # just integration to get the free energy 
    kb =  2.494339/300
    kbt =  kb * temp # kj/mol at 300 K
    d_edge = 180/bins
    # the free energy calculated from the distribution 
    f_theta = -kbt * np.log(theta_hist * d_edge)
    f_phi = -kbt * np.log(phi_hist * d_edge)
    #f_theta[np.isinf(f_theta)] = 100
    #f_phi[np.isinf(f_phi)] = 100


    fig, ax = plt.subplots(2,1, figsize=(4,4))
    
    ax[0].plot(np.arange(0, 180, 180/bins), f_theta - f_theta.min(), label=r"$\theta$",color="orange")
    ax[1].plot(np.arange(0, 180, 180/bins), f_phi - f_phi.min(), label=r"$\phi$", color="blue")
    ax[0].set_ylabel("Free energy (kJ/mol)", )
    ax[1].set_ylabel("Free energy (kJ/mol)")
    ax[1].set_xlabel("Rotation angle (°)")
    ax[0].legend(loc="upper right", )
    ax[1].legend(loc="upper right", )
    ax[0].set_xlim(0,180)
    ax[1].set_xlim(0,180)
    y0_ticks = np.linspace(0,f_theta[~np.isinf(f_theta)].max(), 4)
    y1_ticks = np.linspace(0,f_phi[~np.isinf(f_phi)].max(), 4)
    #print(y0_ticks, y1_ticks, f_phi)
    ax[0].set_yticks([int(i) for i in y0_ticks])
    ax[1].set_yticks([int(i) for i in y1_ticks])
    ax[0].set_xticks(np.arange(0, 181, 60))
    ax[1].set_xticks(np.arange(0, 181, 60))
    
    plt.savefig(f"./general/rotation/tetrahedral/figures/free_{sse}_{temp}_{element}.png", 
                dpi=600, bbox_inches="tight", transparent=True)
    
    #####  add one more figure
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    #sns.set(rc={'figure.figsize':(2.5, 1.5)})
    
    
    sample = angles.sample(20000)
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    g = sns.jointplot( x=sample[r"$\theta$"], y=sample[r"$\phi$"], 
                        kind='kde', 
                        fill=True, 
                        height=2, 
                        cbar=True, 
                        cbar_kws = {"format": formatter, },
                       )
    
    ########adjust the cbar to the right
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    # reposition the joint ax so it has the same width as the marginal x ax
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    # reposition the colorbar using new x positions and y positions of the joint ax
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    # Customize the colorbar
    #cax = fig.fig.add_axes([1, 0.20, 0.02, 0.4])  # Adjust the position and size of the colorbar
    #cb = plt.colorbar(cax=cax)
    
    
    g.ax_joint.set_xticks([0, 60, 120, 180])
    g.ax_joint.set_yticks([0, 60, 120, 180])
    
    g.set_axis_labels("Cell length $\it{a}$ (Å)", "Cell length $\it{b}$ (Å)",  )
    g.ax_marg_x.set_xlim(0, 180)
    g.ax_marg_y.set_ylim(0, 180)
    # Optionally, set a label for the colorbar
    #cb.set_label("Distribution count")
    g.set_axis_labels(r"$\theta$ (°)", r"$\phi$ (°)",fontsize=10,)
    
    plt.savefig(f"./general/rotation/tetrahedral/figures/dis_{sse}_{temp}_{element}.png", 
                dpi=600, bbox_inches="tight", transparent=True)