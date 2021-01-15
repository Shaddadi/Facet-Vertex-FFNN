# Computation of Reachable Sets for FFNN

## Introduction

This repository is for the *Reachability Analysis of Deep ReLU Neural Networks using Facet-Vertex Incidence*. 

**Caution:**
A 64GB RAM is recommended to generate all the results.

## Checked Environment

Software

```txt
Ubunu 18.04
Python 3.6
Matlab 2019b for plotting Figure 4 in the paper
```

Hardware

```txt
CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz 10 cores
Memory: 128GB
```

## Installation

All dependencies are included

## Implementation
To reproduce all the results(~1 hour), run
```bash
chmod +x run_all.sh
./run_all.sh
```
It will generate the result of our method in Table 1, Table 2, Figure 3 and Figure 4 in the paper. Note that you may need to use sudo in front of the command.

***Caution*** The test on ACAS Xu Property 7 and Netwrok1_9 is disabled because its computation time is over 30 minutes. Please run following command if you are intesested in this instance
```bash
cd examples/ACASXu
python3 main.py --property 7 --n1 1 --n2 9
```

## ACASXu
```bash
cd examples/ACASXu
python3 main.py --property <index of the property> --n1 <first index of the network> --n2 <second index of the network> --compute_unsafety <action>
```
For instance, Property 2 on the Network1_2 can be tested with

```bash
 python3 main.py --property 2 --n1 1 --n2 2
```
The unsafe input domain of Property 2 on the Network1_2 can be computed with
```bash
 python3 main.py --property 2 --n1 1 --n2 2 --compute_unsafety
```


## Microbenchmarks
```bash
cd examples/Microbenchmarks
python3 main.py --n1 <first index of the network> --n2 <second index of the network> 
```
For instance, Network2_3 can be tested with

```bash
 python3 main.py --n1 2 --n2 3
 ```
