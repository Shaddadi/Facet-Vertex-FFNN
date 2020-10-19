import sys
P = range(1,5)
N2 = range(1,10)
N1 = range(1,6)
filename = 'script.sh'
f = open(filename, 'w')
lines = '#!/bin/bash\n\nTIMEOUT=10m\n\n'
for p in P:
    for n1 in N1:
        for n2 in N2:
            line_temp = 'timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py ' + str(p) + ' ' + str(n1) + ' ' + str(n2)+ '\n'
            lines += line_temp

f.write(lines)
f.close()
