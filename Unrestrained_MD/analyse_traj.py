import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import dist
import pickle
import sys
from tqdm import tqdm
import os

def getDistance(idx1, idx2, u):
    """
    Get the distance between two atoms in a universe.

    Parameters
    ----------
    idx1 : int
        Index of the first atom
    idx2 : int
        Index of the second atom
    u : MDAnalysis.Universe
        The MDA universe containing the atoms and
        trajectory.

    Returns
    -------
    distance : float
        The distance between the two atoms in Angstroms.
    """
    distance = dist(
        mda.AtomGroup([u.atoms[idx1]]),
        mda.AtomGroup([u.atoms[idx2]]),
        box=u.dimensions,
    )[2][0]
    return distance

def closest_residue_to_point(atoms, point):
    """Find the closest residue in a selection of atoms to a given point"""
    residues = atoms.residues
    distances = np.array([np.linalg.norm(res.atoms.center_of_mass() - point) for res in residues])

    # Find the index of the smallest distance
    closest_residue_index = np.argmin(distances)

    # Return the closest residue
    return residues[closest_residue_index], distances[closest_residue_index]

def obtain_CA_idx(u, res_idx):
    """Function to obtain the index of the alpha carbon for a given residue index"""
    
    selection_str = f"protein and resid {res_idx} and name CA"
    
    selected_CA = u.select_atoms(selection_str)

    if len(selected_CA.indices) == 0:
        print('CA not found for the specified residue...')
    
    elif len(selected_CA.indices) > 1:
        print('Multiple CAs found, uh oh...')

    else:  
        return selected_CA.indices[0]
    
def obtain_angle(run_number, pos1, pos2, pos3):

    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')

    return mda.lib.distances.calc_angles(pos1, pos2, pos3)

def obtain_dihedral(run_number, pos1, pos2, pos3, pos4):
    
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')

    return mda.lib.distances.calc_dihedrals(pos1, pos2, pos3, pos4)

def obtain_RMSD(run_number, res_range=[0,606], eq=False):
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')
    protein = u.select_atoms("protein")

    if eq==True:
        u_ref = u = mda.Universe('equilibrated_structures/complex_eq.prmtop', 'equilibrated_structures/complex_eq.inpcrd')
        ref = u_ref.select_atoms("protein")
    
    else:
        ref = protein

    R_u =rms.RMSD(protein, ref, select=f'backbone and resid {res_range[0]}-{res_range[1]}')
    R_u.run()

    rmsd_u = R_u.rmsd.T #take transpose
    time = rmsd_u[1]/1000
    rmsd= rmsd_u[2]

    return time, rmsd

def save_RMSD(run_number, res_range=[0,606]):
    """
    Save the RMSD of a given run in a .csv file
    """
    time, RMSD = obtain_RMSD(run_number, res_range)

    df = pd.DataFrame()
    df['Time (ns)'] = time
    df['RMSD (Angstrom)'] = RMSD

    filename = 'RMSD.csv'

    df.to_csv(f"results/run{run_number}/{filename}")

    return df

def obtain_RMSF(run_number, res_range=[0,606]):
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')
    
    start, end = int(res_range[0]), int(res_range[1])

    alignment_selection = f'protein and name CA and resid {start}-{end}'
    c_alphas = u.select_atoms(alignment_selection)
    if len(c_alphas) == 0:
        raise ValueError(f"No atoms selected with selection: '{alignment_selection}'")

    # build average structure 
    avg = align.AverageStructure(u, select=alignment_selection, ref_frame=0)
    avg.run()
    ref = avg.results.universe

    # align trajectory in memory 
    align.AlignTraj(u, ref, select=alignment_selection, in_memory=True).run()

    # compute RMSF
    R = rms.RMSF(c_alphas)
    R.run()

    return c_alphas.resids, R.results.rmsf

def save_RMSF(run_number, res_range=[0,606]):
    """
    Save the RMSD of a given run in a .csv file
    """
    residx, RMSF = obtain_RMSF(run_number, res_range)

    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{run_number}/RMSF.csv")

    return df

def run_analysis(systems, k_values):

    for system in systems:
        for k_DDB1 in k_values:
            for n_run in [1,2,3]:
                print(f"\nGenerating RMSD 1for {system}, run {n_run}, with k={k_DDB1} kcal/mol AA^-2\n")
                save_RMSD(system, k_DDB1, n_run, glob=True)
                save_RMSD(system, k_DDB1, n_run, glob=False)
                print(f"\nGenerating RMSF for {system}, run {n_run}, with k={k_DDB1} kcal/mol AA^-2\n")
                save_RMSF(system, k_DDB1, n_run)

def obtain_Boresch_dof(run_number, dof):

    rec_group =  [539, 545, 562, 578, 1227, 1243, 1260, 2019, 3384, 3416, 3433, 3443, 3460, 3475, 3800, 3816, 3838, 3844, 3861, 3868, 3889, 3905, 4083, 4102, 4116, 4135, 4154, 4401, 4417, 4427, 4444, 4454, 4476, 4495, 4505, 4515, 4526, 4543, 4562, 4929]
    lig_group =  [5064, 5071, 5078, 5100, 5387, 5406, 5420, 5439, 5453, 5467, 5474, 5489, 5504, 5881, 5903, 5922, 5941, 5958, 5965, 5972, 5988, 5995, 6022, 6028, 6045, 6064, 6088, 6112, 6133, 6292, 6309, 6321, 7241, 7248, 7272]

    res_b = 90
    res_c = 172 
    res_B = 424 
    res_C = 508 

    group_a = u.atoms[rec_group]
    group_b = u.atoms[[obtain_CA_idx(u, res_b)]]
    group_c = u.atoms[[obtain_CA_idx(u, res_c)]]
    group_A = u.atoms[lig_group]
    group_B = u.atoms[[obtain_CA_idx(u, res_B)]]
    group_C = u.atoms[[obtain_CA_idx(u, res_C)]]

    pos_a = group_a.center_of_mass()
    pos_b = group_b.center_of_mass()
    pos_c = group_c.center_of_mass()
    pos_A = group_A.center_of_mass()
    pos_B = group_B.center_of_mass()
    pos_C = group_C.center_of_mass()

    dof_indices = {
        'thetaA' : [pos_b, pos_a, pos_A],
        'thetaB' : [pos_a, pos_A, pos_B],
        'phiA' : [pos_c, pos_b, pos_a, pos_A],
        'phiB': [pos_b, pos_a, pos_A, pos_B],
        'phiC': [pos_a, pos_A, pos_B, pos_C]
    }

    indices = dof_indices[dof]

    if len(indices) == 3:
        return obtain_angle(run_number, indices[0], indices[1], indices[2])

    else:
        return obtain_dihedral(run_number, indices[0], indices[1], indices[2], indices[3])

for n_run in [1,2,3]:
    # # complex
    # print(f"\nGenerating RMSD for run {n_run}")
    # save_RMSD(n_run)

    # Complex wrt equilibrated structure
    time, RMSD = obtain_RMSD(n_run, eq=True)
    df = pd.DataFrame()
    df['Time (ns)'] = time
    df['RMSD (Angstrom)'] = RMSD
    filename = 'RMSD_eq_ref.csv'
    df.to_csv(f"results/run{n_run}/{filename}")

    # # CRBN
    # time, RMSD = obtain_RMSD(n_run, [0,312])
    # df = pd.DataFrame()
    # df['Time (ns)'] = time
    # df['RMSD (Angstrom)'] = RMSD
    # filename = 'RMSD_CRBN.csv'
    # df.to_csv(f"results/run{n_run}/{filename}")

    # # CK1a
    # time, RMSD = obtain_RMSD(n_run, [314, 606])
    # df = pd.DataFrame()
    # df['Time (ns)'] = time
    # df['RMSD (Angstrom)'] = RMSD
    # filename = 'RMSD_CK1a.csv'
    # df.to_csv(f"results/run{n_run}/{filename}")


    # # complex
    # print(f"\nGenerating RMSF for  run {n_run}")
    # save_RMSF(n_run)   

    # CRBN
    # residx, RMSF = obtain_RMSF(n_run, [0,312])
    # df = pd.DataFrame()
    # df['Residue index'] = residx
    # df['RMSF (Angstrom)'] = RMSF
    # df.to_csv(f"results/run{n_run}/RMSF_CRBN.csv")

    # # CK1a
    # residx, RMSF = obtain_RMSF(n_run, [314,606])
    # df = pd.DataFrame()
    # df['Residue index'] = residx
    # df['RMSF (Angstrom)'] = RMSF
    # df.to_csv(f"results/run{n_run}/RMSF_CK1a.csv") 

for run_number in [1,2,3]:

    for dof in ['thetaA', 'thetaB', 'phiA', 'phiB', 'phiC']:

        if os.path.exists(f'results/run{run_number}/{dof}.pkl'):
            continue
        else:
            print(f"Performing Boresch analysis for {dof} run {run_number}")

            u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')

            vals = []

            for ts in tqdm(u.trajectory, total=u.trajectory.n_frames, desc='Frames analysed'):
                vals.append(obtain_Boresch_dof(run_number, dof))

            frames = np.arange(1, len(vals) + 1)

            dof_data = {
                'Frames': frames,
                'Time (ns)': np.round(0.01 * frames, 6),
                'DOF values': vals
            }

            # Save interface data to pickle
            file = f'results/run{run_number}/{dof}.pkl'
            with open(file, 'wb') as f:
                pickle.dump(dof_data, f)