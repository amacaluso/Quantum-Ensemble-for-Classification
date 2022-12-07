# Quantum-Ensemble-for-Classification

This repository contains the code to reproduce the results in the paper 
[Quantum Ensemble for Classification](https://arxiv.org/abs/2007.01028). 
all the quantum implementations use the [IBM qiskit](https://qiskit.org/) package.

## Description

We propose a new quantum algorithm to perform ensemble classification. 
The underpinning idea is to generate several quantum trajectories in superposition 
to obtain *B* different transformations of the training set in only *log(B)* operations. 
This implies an exponential speedup in the size of the ensemble with respect to 
classical methods. Furthermore, when considering the overall time cost of the 
algorithm the training of a single weak classifier impacts additively rather than 
multiplicatively, as it usually happens.

To demonstrate this, we define simple routine of classification named 
*Cosine Classifier* and implement an ensemble with it. 
In particular, we conduct experiments on simulated data to show that 
*(i)* one execution of the quantum cosine classifier allows retrieving 
ensemble prediction, and *(ii)* the ensemble outperforms any of the single 
classifiers.

## Usage

The code is organised in two parts:
 
- The IPython notebooks cover point *(i)* implementing all the experiments in Section 4.1 and 4.2. Also, they cover all
the technical details about the quantum cosine classifier and the generation of 
multiple independent trajectories in superposition,

 
- The python scripts *quantum_cosine_classifier.py* and *quantum_ensemble*, instead, cover point *(ii)* and implement the experiments of Section 4.3
where it is shown that the quantum ensemble always outperforms the single classifier and  has lower variance

The script *Utils.py* is used to import the needed packages and all of the custom 
routines to generate and plot data.

The script *modeling.py* contains the custom routines to define and execute the quantum algorithms.

The script *Post_processing.py* plots the results of quantum implementation.

## Issues

For any issues or questions related to the code, open a new git issue or send a mail to antonio.macaluso@dfki.de
