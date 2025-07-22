import mdtraj as md 
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pymatgen.core import Structure

def load_traj(xyzfile, virtual_atoms, topology):
    """Load trajectory with cluster centers as virtual atoms"""
    cluster_index = []
    cluster_centers = []
    for wp, centers in virtual_atoms.items():
        centers = np.array(centers)
        index = np.arange(0, centers.shape[0]).repeat(centers.shape[1] )
        centers = centers.reshape(centers.shape[0]*centers.shape[1], 3)
        cluster_index.append(index)
        cluster_centers.append(centers)
    
    traj = md.load(xyzfile, top=topology)

    def add_virtual_atoms_to_topology(topology, cluster_centers_all,cluster_index_all, chain_id=0):
        """Add virtual atoms representing cluster centers to topology"""
        if chain_id >= topology.n_chains:
            raise ValueError(f"Chain ID {chain_id} exceeds the number of chains in the topology.")
    
        for cluster_centers, cluster_index in zip(cluster_centers_all, cluster_index_all):
            virtual_residue_name = 'VIR'
            virtual_residue_indices = []
            virtual_atom_names = [f'VC{i}' for i in cluster_index]
            virtual_atom_types = ['VIR'] * len(cluster_centers)
            
            for center in cluster_centers:
                residue_index = topology.add_residue(virtual_residue_name, list(topology.chains)[0])
                virtual_residue_indices.append(residue_index)
        
            virtual_atom_indices = []
            for i, (res_idx, center) in enumerate(zip(virtual_residue_indices, cluster_centers)):
                atom_index = topology.add_atom(virtual_atom_names[i], virtual_atom_types[i], res_idx)
                virtual_atom_indices.append(atom_index)
    
        return topology, np.array(virtual_atom_indices)
    
    updated_topology, virtual_atom_indices = add_virtual_atoms_to_topology(traj.topology.copy(), cluster_centers, cluster_index)
    all_xyzs = []
    all_xyzs.append(traj.xyz)
    for centers in cluster_centers:
        all_xyzs.append(np.repeat(centers[ np.newaxis, :, :,], traj.n_frames, axis=0) / 10 )
    
    time = traj.time 
    ul = traj.unitcell_lengths
    ua = traj.unitcell_angles
    
    xyz_new = np.concatenate(all_xyzs, axis=1)
    traj_new = md.Trajectory(xyz=xyz_new, 
                             topology=updated_topology, 
                             time=time, 
                             unitcell_lengths=ul,
                             unitcell_angles=ua)
    return traj_new, virtual_atom_indices

from scipy.spatial import  Delaunay
import itertools

class DiscreteTraj( ):
    """Class for getting discrete trajectories"""
    def __init__(self, traj,vors, coords_with_labels  ):
        super(DiscreteTraj, self).__init__()
        self.traj = traj
        self.vors = vors
        self.topology = traj.topology
        self.vcs = np.array([ atom.index for atom in self.topology.atoms if (re.search(r'VC[0-9]*', atom.name) != None) ])
        self.coords_with_labels = coords_with_labels
        self.lithium_index = [ atom.index for atom in self.topology.atoms if (re.search(r'Li', atom.name) != None) ]
        
        self.ref_lists =[ md.compute_neighbors(self.traj[0],cutoff=4.0, query_indices=np.array([self.lithium_index[i]]), haystack_indices=self.vcs,)[0] 
                                    for i in np.arange(0, len(self.lithium_index))]
    def get_state_labels(self,):
        """Return states with labels"""
        states_with_labels = {}
        for wp, center_coords in self.coords_with_labels.items():
            states_with_labels[wp] = {}
            num_sites = 0
            for center in center_coords.keys():
                len_of_states= len(center_coords[center])
                states_with_labels[wp][center] = 1 + np.arange(0, len_of_states) + num_sites
                num_sites += len_of_states
        return states_with_labels

        
    def get_all_cluster_index(self,tolerance=0.1,num_processes=32,lattice=None, plot = False):
        """Return dict {cluster: arr of atoms' index}"""
        global find_closest
        def find_closest(array1, array2, cell_length, tolerance, num_processes):
            """Parallel function for checking differences"""
            from multiprocessing import Pool
            global periodic_distance
            global custom_cdist
            global find_closest_matches
            
            def periodic_distance(point1, point2, cell_length):
                return np.linalg.norm((point1 - point2 + cell_length / 2) % cell_length - cell_length / 2)
            
            def custom_cdist(chunk1, chunk2, cell_length):
                dist_matrix = np.zeros((chunk1.shape[0], chunk2.shape[0]))
                for i in range(chunk1.shape[0]):
                    for j in range(chunk2.shape[0]):
                        dist_matrix[i, j] = periodic_distance(chunk1[i], chunk2[j], cell_length)
                return dist_matrix
            
            def find_closest_matches(chunk_with_idx):
                chunk, start_idx = chunk_with_idx
                distances_chunk = custom_cdist(chunk, array2, cell_length)
                closest_indices_chunk = np.argmin(distances_chunk, axis=1)
                filtered_indices_chunk = np.where(np.min(distances_chunk, axis=1) <= tolerance)[0]
                return [(i + start_idx, closest_indices_chunk[i]) for i in filtered_indices_chunk]
            
            chunk_size = len(array1) // num_processes + (len(array1) % num_processes > 0)
            chunks_with_indices = [(array1[i:i+chunk_size], i ) for i in range(0, len(array1), chunk_size)]
            
            with Pool(num_processes) as p:
                results = p.map(find_closest_matches, chunks_with_indices)
            
            results = [item for sublist in results for item in sublist]
            
            return np.array(results)
        
        def wrap_coords(cartesian_coords, lattice_matrix):
            """Check similarity if cubic"""
            struct = Structure(lattice_matrix,
                            ["Li"] * len(cartesian_coords), 
                            cartesian_coords, coords_are_cartesian=True)
            return struct.frac_coords *  np.sqrt( (lattice_matrix ** 2).sum(axis=1))
        
        li_in_traj = self.traj[0].xyz[:,self.lithium_index,:]
        #print(li_in_traj[0].shape)
        li_mapping_dic = {}
        for wp in self.vors.keys():
            li_mapping_dic[wp] = {}
            for center, coords in self.vors[wp].items():
                sp = coords['points'].shape[0]
                cp = coords['points'].shape[1]
                coords = coords['points'].reshape(sp * cp, 3 )
                
                results = find_closest(coords, 
                                   li_in_traj[0] *10 ,
                                   np.array(self.traj.unitcell_lengths[0] * 10),
                                   tolerance, 
                                   num_processes)
                li_mapping_dic[wp][center] = results[:, 1].reshape(sp,cp,)
                
        if plot==True:
            fig = go.Figure()
            for wp in li_mapping_dic.keys():
                for number, inds in li_mapping_dic[wp].items():
                    sp_count = inds.shape[0]
                    cl_count = inds.shape[1]
                    points = self.traj[0].xyz[:,inds, :][0].reshape(sp_count * cl_count, 3)
                    
                    color = number
                    point_trace = go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode='markers',
                        marker=dict(size=5,opacity=0.6, color=color, colorscale='viridis'),
                        name=f'Atoms in \'{wp}\' cluster {number}'
                    )
                    fig.add_trace(point_trace)
                        
            fig.update_layout(
                autosize=False,
                height=800,
                width=800,
            )
            fig.show()
        return li_mapping_dic
                
    def get_cluster_indexes(self,atom_index,li_mapping_dic, all_atoms=True):
        """Find corresponding cluster for specified lithium"""
        for wp, center_info in li_mapping_dic.items():
            for center, values in center_info.items():
                for cluster in values:
                    if atom_index in cluster:
                        return wp,center, cluster
        
    def get_disc_traj(self, lithium_selected_idx, li_mapping_dic, timestep=2, stride=10, partial=True, plot=False):
        """Get final discrete trajectories"""
        pairs = np.array([ [self.lithium_index[lithium_selected_idx],self.ref_lists[lithium_selected_idx][d] ] 
                          for d in range(len(self.ref_lists[lithium_selected_idx]))])
        
        distances_to_cluster = md.compute_distances(self.traj, pairs)
        
        min_dis = np.copy(distances_to_cluster.min(axis=1))
        min_dis_arg = self.vcs[distances_to_cluster.argmin(axis=1)]
        
        vcs_id = [ atom.name for atom in self.topology.atoms if (re.search(r'VC[0-9]*', atom.name) != None) ]
        vcs_dic = {}
        for vcid, index in zip(vcs_id, self.vcs):
            vcs_dic[index] = int(vcid[2])
        
        center_index = [vcs_dic[index] for index in min_dis_arg]
        def count_unique_sequences(arr):
            """Find unique cluster center"""
            change_indices = np.where(np.diff(arr) != 0)[0] + 1
            
            all_indices = np.insert(change_indices, 0, 0)
            all_indices = np.append(all_indices, len(arr))
            
            counts = np.diff(all_indices)
            unique_idx = np.insert(0, 1, change_indices)
            cumsum_counts = np.cumsum(counts)
            
            starts = np.insert(cumsum_counts, 0, 0)[:-1] 
            ends = cumsum_counts
            
            intervals = np.vstack((starts, ends)).T
            
            return intervals, arr[unique_idx]
        
        
        traj_counts, cluster_centers_ids = count_unique_sequences(min_dis_arg)
        
        cluster_center_pos = self.traj.xyz[0,cluster_centers_ids, :] *10
        pos_ref_disc = []
        vor_polys = []
        for pos in cluster_center_pos:
            for wp, vors_info in self.vors.items():
                for center_num, vor in vors_info.items():
                    for idx, npos in enumerate(vor['center_pos']):
                        identical = abs(np.round(npos-pos, decimals=2)).sum()
                        if identical == 0:
                            target_pos  = idx
                            pos_ref_disc.append([wp, center_num,target_pos ])
                            vor_polys.append(vor['voronoi'][target_pos])

        lixyz = self.traj.xyz[:,self.lithium_index[lithium_selected_idx], :] *10
        disc_traj = []
        states_with_labels = self.get_state_labels()
        for ref, seq in zip(pos_ref_disc, traj_counts,):
            wp = ref[0]
            center_num = ref[1]
            target_pos = ref[2]
            states_for_center = states_with_labels[wp][center_num]
            polys = self.vors[wp][center_num]['voronoi'][target_pos]
            disc_vors = {}
            for i in range(len(polys)):
                checks = Delaunay(polys[i]['vertices']).find_simplex(lixyz[seq[0]:seq[1]]) >= 0
                disc_vors[i] = np.where(checks==0, checks, states_for_center[i], )
            all_states_in_one = np.sum(list(disc_vors.values()), axis=0)

            d_all= np.array(list(disc_vors.values()))
            diff = d_all > 0
            check_part  = diff.sum(axis=0) > 1
            true_indices = np.where(check_part)[0]
            for true_indice in true_indices:
                states_possible = d_all[:,true_indices] 
                states_possible = states_possible[states_possible>0]
                former_states = d_all[:,true_indices-1].sum()
                state = states_possible[(states_possible - former_states).argmin()]
                all_states_in_one[true_indice] = state
            disc_traj.append(all_states_in_one)
        
        disc_traj = list(itertools.chain.from_iterable(disc_traj))
       
        if partial == True:
            wp, center, cluster_index = self.get_cluster_indexes(lithium_selected_idx, li_mapping_dic)
            disc_traj = [num if num in states_with_labels[wp][center] else 0 for num in disc_traj]
        
        if plot==True:
            fig, ax = plt.subplots(1,1, figsize=(3,2))
            scatter= ax.scatter(
                       np.arange(0, len(lixyz)) * timestep*stride/1000, 
                       disc_traj,
                       c= disc_traj,
                       s= 0.1, vmin=0, vmax=24,
                       cmap = "PiYG")
            plt.colorbar(scatter, label="Lithium states")
            ax.set_xlabel("Time (ps)")
            ax.set_ylabel("Lithium states")
            unique_states = np.unique(disc_traj)
            labels = [f'LS$_{{{int(n)}}}$'  for n in unique_states]
            ax.set_yticks(unique_states)
            ax.set_yticklabels(labels)
            
            return np.array(disc_traj),fig
        return np.array(disc_traj)