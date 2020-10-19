import numpy as np
import pickle
import matplotlib.pyplot as plot

def get_data_eran(eran):
    file1 = open(eran,'r')
    lines = file1.readlines()
    val = float(lines[-2][1:5])
    rel = lines[-3][18]
    if rel == 'V':
        rell = 'UNSAT'
        return [val, rell]
    else:
        return []

def get_data_nnv(nnv):
    file1 = open(nnv,'r')
    lines = file1.readlines()
    val = float(lines[1][6:12])
    rel = int(lines[0][6])
    if rel!=2:
        return [val, rel]
    else:
        return []

# logs_facetv: facet-vertex
def get_data_facetv(facetv):
    file1 = open(facetv, "r")
    lines = file1.readlines()
    file1.close()
    val = float(lines[0][14:20])
    rel = lines[1][8:-1]
    if rel == 'safe':
        rell = 'UNSAT'
    else:
        rell = 'SAT'

    return [float(val), rell]

def get_data_bakcav(bakcav):
    try:
        f = open(bakcav, 'r')
        contents = f.readlines()
        f.close()
        try:
            for al in contents:
                if al[:9] == 'Runtime: ':
                    val = ''
                    for n, c in enumerate(al[9:]):
                        if c==' ':
                            cy = al[n+10:n+13]
                            break
                        else:
                            val += c

                    if cy == 'min':
                        val = float(val)*60
                    else:
                        val = float(val)

            if contents[-1][:7] == 'Output':
                rel = 'SAT'
            else:
                rel = 'UNSAT'
        except:
            val = 120.0
            rel = 'TIMEOUT'
        if val == 0.0:
            xx = 1
        if float(val) >120:
            val = 120.0
            rel = 'TIMEOUT'
        return [float(val), rel]

    except:
        # print("ERROR: bakcav parse failure")
        raise Exception("ERROR: bakcav parse failure")



def get_data_mara(mara):
    try:
        f = open(mara, 'r')
        contents = f.readlines()
        f.close()
        result=''
        try:
            idx = len(contents) - 1 - contents[::-1].index('\t--- Time Statistics ---\n')
            i0= 21
            value = ''
            for n in range(i0, len(contents[idx + 1])):
                if contents[idx + 1][n] == 'm':
                    break
                else:
                    value = value+contents[idx + 1][n]

            value = int(value)/1000
        except:
            value = 0.0

        if 'sat\n' in contents:
            result = 'SAT'
        elif 'unsat\n' in contents:
            result = 'UNSAT'
        else:
            value = 120.0
            result = 'TIMEOUT'

        return [float(value), result]
    except:
        print("ERROR: marabou parse failure")



def get_data_reluval(reluval):
    try:
        f = open(reluval, 'r')
        contents = f.readlines()
        f.close()
        result =''

        value = ''
        for e in contents[::-1]:
            if e[:5]=='time:':
                value = e[6:-1]
                break

        if 'adv found:\n' in contents:
            result = 'SAT'
        elif 'No adv!\n' in contents:
            result = 'UNSAT'
        else:
            value = 120.0
            result = 'TIMEOUT'

        return [float(value), result]
    except:
        print("ERROR: reluval result parse failure")
        return '0', 'ERR'



if __name__ == "__main__":
    time_facetv = []
    time_mara = []
    time_reluval = []
    time_reluval_2h = []
    time_bakcav = []
    time_eran = []
    time_nnv = []
    for p in range(1,101):

        facetv = 'logs_facetv_netx4/result' + str(p)
        nnv = 'logs_nnv_netx4/results_net_net256x4.mat_image' + str(p) + '.txt'
        # mara = 'logs_mara/results_' + 'p' + str(p) + '_' + 'n'+str(n1) + str(n2) + '.txt'
        # reluval = 'logs_reluval/results_' + 'p' + str(p) + '_' + 'n' + str(n1) + str(n2) + '.txt'
        # reluval_2h = 'logs_reluval_2h/results_' + 'p' + str(p) + '_' + 'n' + str(n1) + str(n2) + '.txt'
        eran = 'logs_eran_netx4/results' + 'p' + str(p) + '.txt'
        bakcav = 'logs_bakcav_netx4/output_info_' + str(p) + '.txt'

        time_facetv.append(get_data_facetv(facetv)[0])
        if get_data_nnv(nnv):
            time_nnv.append(get_data_nnv(nnv)[0])

        # time_mara.append(get_data_mara(mara)[0])
        # time_reluval.append(get_data_reluval(reluval)[0])
        # time_reluval_2h.append(get_data_reluval(reluval_2h)[0])
        time_bakcav.append(get_data_bakcav(bakcav)[0])
        # if get_data_eran(eran):
        #     time_eran.append(get_data_eran(eran)[0])

    time_facetv = np.array(time_facetv)
    time_nnv = np.array(time_nnv)
    # time_mara = np.array(time_mara)
    # time_reluval = np.array(time_reluval)
    # time_reluval_2h = np.array(time_reluval_2h)
    time_bakcav = np.array(time_bakcav)
    # time_eran = np.array(time_eran)
    # x = np.arange(1,181)

    # plot.semilogy(x, np.sort(time_facetv), x, np.sort(time_mara), x, np.sort(time_mara), x, np.sort(time_bakcav))
    # plot.legend(['Our method','Marabou', 'ReluVal', 'nnenum'])
    # plot.xlabel('Number of instances verified')
    # plot.ylabel('Time(sec)')
    # plot
    #
    # plot.show()

    with open('times.pkl', 'wb') as f:
        pickle.dump([time_facetv, time_nnv, time_bakcav], f)