import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
from sys import stdout
import numpy as np
import pandas as pd
import os

run_number = int(sys.argv[1])
savedir = f"results/run{run_number}"

if not os.path.exists(savedir): # Make save directory if it doesn't yet exist
    os.makedirs(savedir)

"""System setup"""

dt = 4*unit.femtoseconds 

# Load param and coord files
prmtop = app.AmberPrmtopFile(f'structures/complex.prmtop')
inpcrd = app.AmberInpcrdFile(f'structures/complex.inpcrd')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, hydrogenMass=1.5*unit.amu, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(0.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

# Add reporters to output data
simulation.reporters.append(app.StateDataReporter(f'{savedir}/system.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.reporters.append(app.DCDReporter(f'{savedir}/traj.dcd', 2500))

# Minimise energy 
simulation.minimizeEnergy()

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)

simulation.step(1000)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

"""RMSD restraints in DDB1-binding region"""

reference_positions = inpcrd.positions

DCAF16_interface_residx = np.append(np.arange(0,55), np.arange(123, 172))
DCAF16_DDB1_binding_residx = np.arange(70,116)

DDB1_binding_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in DCAF16_DDB1_binding_residx and atom.name in ('CA', 'C', 'N')
]

DDB1_rmsd_force = mm.CustomCVForce('0.5*k_DDB1*rmsd^2')
DDB1_rmsd_force.addGlobalParameter('k_DDB1', 30 * unit.kilocalories_per_mole / unit.angstrom**2)
DDB1_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, DDB1_binding_atoms))
system.addForce(DDB1_rmsd_force)
simulation.context.reinitialize(preserveState=True)

"""100 ns simulation"""

print('-------------------------')
print('Starting NVT simulation!')
simulation.step(2500000)