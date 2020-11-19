# Define up to two variables for plotting here.

import numpy as np

def get_var_funcs():

    var_funcs = []

    def var_func1(f):

        k = 128
        slc = np.s_[0,:,:]

        rho = np.array(f['rho'])[slc]
        p = np.array(f['p'])[slc]
        # ppd = np.array(f['ppd'])[slc]
        # ppl = np.array(f['ppl'])[slc]
        # pe = np.array(f['pe'])[slc]

        vx = np.array(f['vx'])[slc]
        vy = np.array(f['vy'])[slc]
        vz = np.array(f['vz'])[slc]
        bxc = np.array(f['bxc'])[slc]
        byc = np.array(f['byc'])[slc]
        bzc = np.array(f['bzc'])[slc]

        # ppl = 3*p - 2*ppd
        # v2 = vx**2+vy**2+vz**2
        # b2 = Bxc**2+Byc**2+Bzc**2

        return byc
        # return (ppd-ppl)/b2
        # return np.log10(b2)
        # return np.log10(v2)
        # return vy

    var_funcs.append(var_func1)

    # def var_func2(f):
    #
    #     return array(f['rho'])
    #
    # var_funcs.append(var_func2)

    return var_funcs
