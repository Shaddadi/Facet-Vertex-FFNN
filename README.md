# Computation of Reachable Sets for FFNNd

## Introduction

This repository is for the *Reachability Analysis of Deep ReLU Neural Networks using Facet-Vertex Incidence*

## Checked Environment

Software

```txt
Ubunu 18.04
Python 3.6
```

Hardware

```txt
CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz 10 cores
Memory: 128GB
```

## Installation

All dependencies are included

## Implementation

```bash
cd examples/ACASXu
python3 main.py <index of the property(int)> <first index of the network(int)> <second index of the network(int)> 
```

For instance, Property 1 on the Network2_3 can be tested with

```bash
python3 main.py 1 2 3
```

## Reproduce the verification of Property 3&4 on all 45 networks (~20 seconds)

```bash
cd examples/ACASXu
sudo chmod +x run_property34.sh
./run_property34.sh
```
