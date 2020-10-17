import numpy as np

times = []
for p in range(1,5):
    for n1 in range(1,6):
        for n2 in range(1,10):
            filename = 'output_info_'+str(p) +'_' + str(n1) + '_' + str(n2) +'.txt'
            file1 = open(filename, "r")
            lines = file1.readlines()
            time1 = float(lines[0][14:20])
            times.append(time1)

times = np.array(times)
print(np.max(times))
