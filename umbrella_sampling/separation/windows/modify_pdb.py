from pymol import cmd
import sys
import os
import numpy as np
import shutil
import subprocess

r0_values = [0.9, 0.95, 1.0, 1.05, 1.1 , 1.15, 1.2 , 1.25, 1.3 , 1.35, 1.4 , 1.45, 1.5 , 1.55, 1.6 , 1.65, 1.7 , 1.75, 1.8 , 1.85, 1.9 , 1.95, 2.0  , 2.05, 2.1 , 2.2 , 2.3 , 2.4 , 2.5 , 2.6 , 2.7 , 2.8 , 2.9 , 3.0 , 3.1 , 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0 , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0 , 5.1, 5.2, 5.3, 5.4, 5.5]

for r_0 in r0_values:

    r_0 = np.round(r_0, 4)
    filepath = f'{r_0}'

    print('\n----------------------------------------')
    print(f"Generating input for r0 = {r_0} nm\n")

    """Remove all hydrogen atoms, water molecules and chlorine ions from the system"""

    # Load the PDB file
    cmd.reinitialize()
    cmd.load(f"{filepath}/{r_0}.pdb", "complex")

    # Remove waters, hydrogens, and chlorine ions
    cmd.remove("resn HOH")
    cmd.remove("hydro")
    cmd.remove("resn CL")
    cmd.remove("resn NA")

    # Save the modified structure temporarily
    cmd.save(f"{filepath}/temp.pdb")

    """Add TER where needed"""

    with open(f"{filepath}/temp.pdb", "r") as infile, open(f"{filepath}/{r_0}_modified.pdb", "w") as outfile:
        last_residue = None
        for line in infile:
            # Skip lines starting with 'CONECT' or 'CONNECT'
            if line.startswith("CONECT") or line.startswith("CONNECT"):
                continue

            # Replace HETATOM with ATOM
            if "HETATM" in line and " ZN " in line:
                line = line.replace("HETATM", "ATOM  ", 1)

            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_num = int(line[22:26].strip())
                # Insert a TER line where needed
                if last_residue in [137, 313] and res_num in [138, 314]:
                    outfile.write("TER\n")
                last_residue = res_num
            # Write the line to the output file
            outfile.write(line)

    # Clean up
    os.remove(f"{filepath}/temp.pdb")

    """Generate tleap.in file"""

    # Copy over frcmod and prep files
    shutil.copy('ZAFF.frcmod', f'{filepath}/ZAFF.frcmod')
    shutil.copy('ZAFF.prep', f'{filepath}/ZAFF.prep')

    # Write tleap lines
    input_filename = f"{r_0}_modified.pdb"

    line0 = 'source leaprc.protein.ff19SB'
    line1 = 'source leaprc.water.tip3p'
    line2 = 'addAtomTypes { { "ZN" "Zn" "sp3" } { "S3" "S" "sp3" } } #Add atom types for the ZAFF metal center with Center ID 1'
    line3 = 'loadamberparams frcmod.ions1lm_126_tip3p'
    line4 = 'loadamberprep ZAFF.prep #Load ZAFF prep file'
    line5 = 'loadamberparams ZAFF.frcmod #Load ZAFF frcmod file\n'

    line6 = f'mol = loadpdb {r_0}_modified.pdb # Load the PDB file\n'

    line7 = 'bond mol.313.ZN mol.210.SG #Bond zinc ion with SG atom of residue CYM'
    line8 = 'bond mol.313.ZN mol.213.SG #Bond zinc ion with SG atom of residue CYM'
    line9 = 'bond mol.313.ZN mol.278.SG #Bond zinc ion with SG atom of residue CYM'
    line10 = 'bond mol.313.ZN mol.281.SG #Bond zinc ion with SG atom of residue CYM'

    line11 = 'solvateOct mol TIP3PBOX 15.0\n'

    line12 = 'addions mol CL 0\n'

    line13 = 'savepdb mol system_solvated.pdb #Save the pdb file'
    line14 = f'saveamberparm mol system.prmtop system.inpcrd #Save the topology and coordinate files'
    line15 = 'quit #Quit tleap'

    lines = [
        line0,
        line1,
        line2,
        line3,
        line4,
        line5,
        line6,
        line7,
        line8,
        line9,
        line10,
        line11,
        line12,
        line13,
        line14,
        line15
    ]

    with open(f'{filepath}/tleap.in', 'w') as file:
        for line in lines:
            file.write(line + '\n')

    # Specify the command and directory
    command = 'tleap -f tleap.in'

    try:
        # Run the command within the specified directory
        subprocess.run(command, check=True, shell=True, cwd=filepath)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")