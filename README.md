# Hollow Knight AI

An attempt at trying to create an AI-based player that attempts to defeat Hollow Knight bosses making use of Behavioural Cloning

## The Mods

The `mods` directly contains the required mods to gather data and allow our model to take control of the player.

Use the Hollow Knight Mods extension in Visual Studio to get started.

More about modding can be read here: https://prashantmohta.github.io/ModdingDocs/

### DataCollector

This mod is used to record player data from the game.

Make sure to set the `saveDirectory` as this will be where all the data will be stored.

Data Collection can be toggled by pressing the 'O' key.

### GameAgent

This mod acts as a pipe. It runs the python process (agent) and sends the data to this program.

Make sure to set the `pythonScriptPath` and `modelPath`.

Also add the path to your python interpreter in `FileName`.

This mod can be toggled with the 'P' key.

## Machine Learning

### Training

The code used to train our model can be found in `train.py`. Make sure to set the `DATA_DIR` in this.

### Agent

The file `agent.py` is responsible for taking the game state from the GameAgent mod and the trained model to take an action

