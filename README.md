# Open-ended creation of hybrid creatures with Neural Cellular Automata

This is our submissiosn to the [Minecraft Open-endedness challenge
2021](https://evocraft.life/) for the Gecco conference.

## Description

Our algorithm is based on Neural Cellular Automata (NCA), a CA-based neural
network model inspired by morphogenesis. We train a NCA to grow different
patterns from various seeds (or "genomes"). Once the training is done, we load
the model in Minecraft and have players modify the seeds. The seeds can be
mutated or merged to create an endless stream of novel growing patterns. The
resulting patterns depend both on the seed and the growth rules learned by the
NCA, which can be unpredictable and surprising.

## Usage

Make sure you have the right Python packages installed. The project uses Poetry,
so you only need to do `poetry install` if you have it.

Otherwise `pip install -r requirements.txt` should work too. 

The models are defined in `evo_ca/models.py`.

## Model weights
### Pre-trained models
### Train your own
