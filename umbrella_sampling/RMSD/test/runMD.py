import openmm as mm
import openmm.app as app
import openmm.unit as unit
from sys import stdout
import numpy as np

# MD parameters
timestep = 4 # fs
sampling_steps = int(10//(timestep*1e-6)) # Run simulation for 10 ns
record_steps = 125 # record CV every every 125 steps

"""System setup"""

dt = timestep*unit.femtoseconds 

# Load param and coord files
prmtop = app.AmberPrmtopFile('complex_eq.prmtop')
inpcrd = app.AmberInpcrdFile('complex_eq.inpcrd')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, hydrogenMass=1.5*unit.amu, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(0.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

# Add reporters to output data
simulation.reporters.append(app.StateDataReporter('system.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
simulation.reporters.append(app.StateDataReporter(stdout, 2500, step=True, time=True, potentialEnergy=True, temperature=True, speed=True))
simulation.reporters.append(app.DCDReporter('traj.dcd', 1000))

# Minimise energy 
simulation.minimizeEnergy()

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)

simulation.step(1000)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

"""Apply RMSD restraining potential"""

reference_positions = inpcrd.positions

receptor_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in range(0, 312) and atom.name in ['CA', 'C', 'N']
]

# Test to ensure correct atom selection
for atom in simulation.topology.atoms():
    if atom.index==receptor_atoms[0] and atom.residue.name!='ASP':
        raise ValueError(f'Incorrect residue selection for CRBN - residue D1 is missing')
    if atom.index==receptor_atoms[-1] and atom.residue.name!='THR':
        raise ValueError(f'Incorrect residue selection for CRBN - residue T312 is missing')

receptor_rmsd_force = mm.CustomCVForce('0.5*k*(rmsd-rmsd_0)^2')
receptor_rmsd_force.addGlobalParameter('k', 30 * unit.kilocalories_per_mole / unit.angstrom**2)
receptor_rmsd_force.addGlobalParameter('rmsd_0', 0.1 unit.angstrom**2)
receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
system.addForce(receptor_rmsd_force)

simulation.context.reinitialize(preserveState=True)

"""NVT simulation"""

# Run the simulation and record the value of the RMSD
rmsd_values=[]

for i in range(sampling_steps//record_steps):

    simulation.step(record_steps)

    current_rmsd = receptor_rmsd_force.getCollectiveVariableValues(simulation.context)

    rmsd_values.append([i, current_rmsd[0]])

# Save the RMSD values
np.savetxt('rmsd.txt', np.array(rmsd_values))