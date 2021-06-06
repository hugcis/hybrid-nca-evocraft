# Open-ended creation of hybrid creatures with Neural Cellular Automata

This is our submissiosn to the [Minecraft Open-endedness challenge
2021](https://evocraft.life/) for the Gecco conference.

## Description

Our algorithm is based on Neural Cellular Automata (NCA), a CA-based neural
network model inspired by morphogenesis. We train a NCA to grow different
patterns from various seeds (or "genomes"). Once the training is done, we load
the model in Minecraft and have players modify the genomes. They can be
mutated or merged to create an endless stream of novel growing patterns. The
resulting patterns depend both on the genome and the growth rules learned by the
NCA, which can be unpredictable and surprising. 

The repository contains link to pre-trained models in 2D and 3D as well as Colab
notebooks links to train you own NCA. 

## Usage

Make sure you have the right Python packages installed. The project uses Poetry,
so you only need to do `poetry install` if you have it.

Otherwise `pip install -r requirements.txt` should work too. 

The models are defined in `evo_ca/models.py`.

`python run_2d.py`

## Model weights
### Pre-trained models
The pre-trained weights are in a zip archive that you can download
[here](https://drive.google.com/file/d/1zLyXiFTJEi7wCDK7NHZOR7kg0fT_GE_w/view?usp=sharing).
Unzip the archive in the repo root.

### Train your own

#### 2D Model
[Colab link](https://colab.research.google.com/drive/1WEYtcDMm3HNfHHlso_B9SkDU0NivwXbv?usp=sharing)

#### 3D Model
