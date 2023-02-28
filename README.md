# Damped Driven Harmonic Oscillator

### What is this?

Code to perform regression via *stochastic gradient descent* on data collected in the lab in order to find the **natural frequency**, **driving amplitude**, and **damping coefficient** for each trial.

### Where's the important stuff?

For the weights I found, look in the `results.csv` file.

To play around with the hyperparameters and run the fit yourself, look in `script.py` and run it.

The training history gets dumped in the `data/history` folder and comparisons to the original data at each point are generated and stored in the `data/comparisons` folder (for easy plotting).

### What's all that other junk?

`data.py` and `train.py` contain all the code for actually making the regression work.

Everything in the `testing.py` file and the `data/mapping` folder are just from me screwing around with the data, but you are welcome to poke around in it if you're bored enough.
