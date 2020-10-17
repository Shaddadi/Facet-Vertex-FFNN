#!/bin/bash

TIMEOUT=10m

timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py 3 2 1
