# AiPROTAC

<img src="https://raw.githubusercontent.com/LiZhang30/AiPROTAC/blob/main/images/cover.png?raw=true" alt="Cover" width="60%" />

## Brief Introduction

AiPROTAC for PROTAC-targeted Degradation Prediction and Androgen Receptor Degrader Design

## Features

- Feature 1
- Feature 2

## Installation

- Main Requirements (CUDA version 11.7):
<br> 1. Create a virtual environment 
<br> conda create -n gnn_gpu python=3.7
<br> 2. Set up PyTorch environment: Linux CUDA 11.7
<br> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
<br> 3. Set up DGL environment: CUDA 11.7, Package Conda(Stable), Linux, Python version 3.7
<br> conda install -c dglteam dgl-cuda11.7
<br> 4. Set up RDKit environment: 2019 version
<br> conda install -c conda-forge/label/cf202003 rdkit

## Datasets

Commands to Obtain Pocket Data Using PyMOL:
<br> 1. fetch (PDB ID)  # Fetch protein from the PDB database.
remove solvent  # Remove solvents.
remove sele  # Remove the selected extra ligands.
button operation  # Extract the target ligand and rename it as obj01.
zoom obj01, animate=1  # Zoom in on the selected part.
bg_color white  # Set the background color to white.
set cartoon_transparency, 0.8  # Adjust the transparency of the cartoon (or other surfaces).
sele obj01 around 5  # Select atoms within 5 Å of the ligand.
zoom sele, animate=1  # Zoom in on the selected residue.
select byres sele  # Select the residues associated with the ligand.
button operation  # Manually extract the selected pocket residues, rename it as obj02, and display as sticks.
button operation  # Manually color the ligand.
zoom obj02, animate=1  # Zoom in on the pocket.
button operation  # Manually visualize the pocket.
Save the pdb file of the protein pocket  # Save the pocket’s PDB file.

## Usage

使用方法说明...

## License

This project is licensed under the [Apache License 2.0](LICENSE).

You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

- You must provide a copy of this license in any distribution of the Software.
- You must provide a prominent notice stating that you have changed the files, if applicable.
- You may not use the trademarks, service marks, or other identifiers of the copyright holders, except as required by the license.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

For more details, see the full license in the [LICENSE](LICENSE) file.
