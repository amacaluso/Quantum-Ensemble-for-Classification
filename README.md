# Quantum-Ensemble-for-Classification

This repository contains the code to reproduce the results in the paper *Quantum Ensemble for Classification*. The code for the implementation of the quantum circuits uses the [IBM Qiskit](https://qiskit.org/) environment.
The three notebook cover all the technical details omitted in the paper.

## Description

The code is organised as follows:
- *Multiple Experiments for Quantum Ensemble (Simulator).ipynb* uses the quantum algorithm to produce the ensemble of two swap tests. It generates 20 small dataset and compare the results of quantum ensemble with the same algorithm computed classically. 
- *Quantum Swap Test.ipynb* explains in detail the swap test by performing also simulation considering a small dataset.
- *Quantum Ensemble of Swap Test.ipynb* explains in detail the ensemble of two swap tests by performing also simulation considering a small dataset. Also, all steps of the computation omitted in the paper are reported.



The script *Utils.py* contains the import of the needed packages and all the custom routines for the circuit generation.

The script *Visualization.py* contains the custom routines for plot the results as reported in the paper.

The script *run_all.py* implements the experiments of 20 random generated dataset in quantum simulator and real device.

