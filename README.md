# Quantum-Ensemble-for-Classification

This repository contains the code to reproduce the results in the paper *Quantum Ensemble for Classification*. For quantum implementation the project uses the [IBM Qiskit](https://qiskit.org/) package.

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
the IPython notebooks cover all the details omitted in Section 4.2 
and implement all experiments that aim to show that quantum ensemble
 performs the average of multiple quantum trajectories in superposition. 
 The code in the python scripts contain the code to show that the ensemble outperforms the single classifier. 
