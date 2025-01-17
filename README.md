# AiPROTAC

##![Cover](https://raw.githubusercontent.com/LiZhang30/AiPROTAC/main/images/cover.png)
<img src="https://raw.githubusercontent.com/LiZhang30/AiPROTAC/main/images/cover.png" alt="Cover" width="60%" />


## Brief Introduction

AiPROTAC for PROTAC-targeted Degradation Prediction and Androgen Receptor Degrader Design

## Features

- A virtual screening tool for PROTAC-targeted degradation prediction.
- Applied this tool to design AR degraders, leading to the identification of the lead compound GT19.
- Integrated both supervised and unsupervised learning frameworks, offering a reference for drug discovery tasks with limited labeled data but high predictive demand.
- Curated the PROTAC-ZL dataset manually.
- Preprocessed data from PROTAC-DB and PROTAC-ZL, ready for developing new PROTAC-targeted degradation prediction tools.

## Installation

- Main Requirements:
<br> 1. Create a virtual environment 
<br> conda create -n gnn_gpu python=3.7
<br> 2. Set up PyTorch environment: Linux CUDA 11.7
<br> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
<br> 3. Set up DGL environment: CUDA 11.7, Package Conda (Stable), Linux, Python version 3.7
<br> conda install -c dglteam dgl-cuda11.7
<br> 4. Set up RDKit environment: 2019 version
<br> conda install -c conda-forge/label/cf202003 rdkit

- Notes: For more dependencies, please refer to [environment.yml](environment.yml)

## Datasets

- PROTAC-DB 2.0 and PROTAC-ZL datasets
<br> 1. The data, which is ready for model training and testing, is stored in the "data" folder of this project.
<br> 2. We provide all raw data, including FASTA sequences, SMILES, and protein structures. Users can easily construct molecular graphs using the processing code we offer. The data processing scripts are available in DataHelper.py, DrugGraph.py, and TargetGraph.py within this project.

- Commands to Obtain Protein Pockets Using PyMOL:
<br> 1. fetch (PDB ID)  # Fetch protein from the PDB database.
<br> 2. remove solvent  # Remove solvents.
<br> 3. remove sele  # Remove the selected extra ligands.
<br> 4. button operation  # Extract the target ligand and rename it as obj01.
<br> 5. zoom obj01, animate=1  # Zoom in on the selected part.
<br> 6. bg_color white  # Set the background color to white.
<br> 7. set cartoon_transparency, 0.8  # Adjust the transparency of the cartoon (or other surfaces).
<br> 8. sele obj01 around 5  # Select atoms within 5 Å of the ligand.
<br> 9. zoom sele, animate=1  # Zoom in on the selected residue.
<br> 10. select byres sele  # Select the residues associated with the ligand.
<br> 11. button operation  # Manually extract the selected pocket residues, rename it as obj02, and display as sticks.
<br> 12. button operation  # Manually color the ligand.
<br> 13. zoom obj02, animate=1  # Zoom in on the pocket.
<br> 14. button operation  # Manually visualize the pocket.
<br> 15. Save the pdb file of the protein pocket  # Save the pocket’s PDB file.

## Usage

- For training and testing AiPROTAC and its baselines, please refer to the scripts in this project.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

- You must provide a copy of this license in any distribution of the Software.
- You must provide a prominent notice stating that you have changed the files, if applicable.
- You may not use the trademarks, service marks, or other identifiers of the copyright holders, except as required by the license.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

For more details, see the full license in the [LICENSE](LICENSE) file.
