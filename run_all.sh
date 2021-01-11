#!/bin/bash

pip3 install filelock

cd examples/ACASXu
./property1-4.sh
./property5-10.sh
./property2_N12.sh
python3 load_results.py

cp ./Figure3.png ../../
cp ./Figure4.png ../../
cp ./Table1.txt ../../
cd ../../

cd examples/Microbenchmarks
./microbenchmarks.sh
python3 load_results.py
cp ./Table2.txt ../../
cd ../../
