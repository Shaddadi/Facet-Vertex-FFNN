import sys
filename = 'script.sh'
f = open(filename, 'w')
lines = '#!/bin/bash\n\nTIMEOUT=10m\n\n'

for n in range(1,101):
    line_temp = 'timeout --foreground --signal=SIGQUIT $TIMEOUT python3 main.py ' + str(n)+ '\n'
    lines += line_temp

f.write(lines)
f.close()
