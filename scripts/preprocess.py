import numpy as np
from pymatgen.core import Structure
from pyxtal import pyxtal   
from pyxtal.symmetry import Group
import re

def wrap_coords(cartesian_coords, lattice_matrix):
    """Wrap coordinates to unit cell using pymatgen Structure."""
    struct = Structure(lattice_matrix,
                    ["Li"] * len(cartesian_coords), 
                    cartesian_coords, coords_are_cartesian=True)
    return struct.frac_coords * np.sqrt((lattice_matrix ** 2).sum(axis=1))

def wrap_supercell(sse_type, scaling_mat, super_coords, super_centers):
    """Wrap supercell coordinates to unit cell.
    
    Note: This wrapper is only necessary if MD simulation is not running through ASE package.
    ASE automatically wraps the supercell.
    """
    pm = Structure.from_file(f'./data/structures/{sse_type}.cif') 

    pymat_super = pm.make_supercell(scaling_mat)
    wy = list(super_centers.keys())
    num_cluster = super_coords[wy[0]][0].shape[0]
    num_sites = super_coords[wy[0]][0].shape[1]
    lattice_matrix = pymat_super.lattice.matrix
    
    for center, coords in super_coords[wy[0]].items():
        super_coords['24g'][center] = wrap_coords(coords.reshape(num_cluster*num_sites, 3),
         lattice_matrix).reshape(num_cluster,num_sites,3)

    for num in range(len(super_centers['24g'])):
        coords = super_centers['24g'][num]
        super_centers['24g'][num] = wrap_coords(coords, lattice_matrix).reshape(num_cluster,3)    

    return super_centers, super_coords