from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def plt_latex(dpi=120):
    """Set up latex for plt and change default figure dpi."""
    import matplotlib.colors
    # Latex for plt
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # default figure size
    matplotlib.rcParams['figure.dpi'] = dpi

def h5_read_dict(outfile, path='/'):
    """Read an HDF5 file with given `path` (or default /) and return dictionary of numpy arrays."""
    import h5py
    hf = h5py.File(outfile, 'r')
    cc = {}
    for k in hf[path].keys():
        cc[k] = np.array( hf[path][k] )
    hf.close()
    return cc

def h5_write_dict(outfile, dict, group):
    """Write an HDF5 file `outfile` with given `dict` in `group`."""
    import h5py
    f = h5py.File(outfile, 'w')
    grp = f.create_group(group)
    for k, v in dict.items():
        grp[k] = v
    f.close()

def h5_read_files(basename=None, suffix='#', path='/', filelist=None):
    """Read a set of HDF5 files given by a `basename` and `suffix`.
    E.g., to read `/path/to/file#*`, `basename='/path/to/file'` and `suffix='#'`.
    Alternatively, read all files in `filelist`.
    Set of files are assumed to have the same keys in HDF5 `path`.
    Returns combined dictionary of numpy arrays.
    """
    import h5py
    import glob
    assert bool(basename)^bool(filelist), "Must define basename or filelist (not both)"
    if filelist is None:
        filelist=glob.glob(basename+suffix+'*')

    lists_all = None
    for f in filelist:
        dictf = h5_read_dict(f, path)
        if lists_all is None:
            lists_all = {k:[] for k in dictf.keys()}
        for k in lists_all.keys():
            lists_all[k].append(dictf[k])
    return { k:np.concatenate(lists_all[k]) for k in lists_all.keys() }

def gio_read_dict(f, cols):
    """Read cols of GenericIO file f and return dictionary of numpy arrays."""
    import genericio as gio
    return { k:gio.gio_read(f, k)[0] for k in cols }

def read_binary_sh(step, binarydir, flist=None, verbose=True, columns=[('fof_halo_tag',np.int64), ('fof_halo_count',np.int64), ('subhalo_tag',np.int64), ('subhalo_count',np.int64), ('subhalo_mass',np.float32)]):
    """Read a binary format subhalo catalog and return a dictionary.
    Define either both step and binarydir, or flist."""
    import os
    binary_keys = [k for k,_ in columns]
    dt = np.dtype(columns)

    if flist is None:
        flist = [ os.path.join(binarydir, f) for f in os.listdir(binarydir) if ('binary' in f) and (str(step) in f) ]
    
    binarydict_lists = {k:[] for k in binary_keys}
    for f in flist:
        if verbose:
            print(f)
        x = np.fromfile(f, dtype=dt)
        for k in binary_keys:
            binarydict_lists[k].append(x[k])
    binarydict = { k:np.concatenate(binarydict_lists[k]) for k in binary_keys  }
    return binarydict

def pickle_save_dict(f, d):
    """Pickles dictionary d in outfile f."""
    import pickle
    pickle.dump( d, open( f, 'wb' ) )

def loadpickle(f, python2topython3=False):
    """Returns pickled object in f.
    python2topython3: Reads in file originally pickled with Python2 into Python3 https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
    """
    import pickle
    if python2topython3:
        return pickle.load( open(f, 'rb'), encoding='latin1' )
    else:
        return pickle.load( open(f, 'rb') )

def hist(vals, bins=500, normed=False, plotFlag=True, label='', alpha=1, range=None, normScalar=1, normCnts=False, normBinsize=False, normLogCnts=False, plotOptions='.', ax=None, retEbars=False):
    """
    DOCUMENTATION OUT OF DATE. TODO: fix documentation
    Returns histogram of vals in (default 500) bins.

    Parameters
    ----------
    vals : np 1d array
        Data points
    normed : bool
        If true, normalizes by counts (if `normCnts`), bin size (if `normBinsize`), scalar (if `normScalar` given), log10(cnts) (if `normLogCnts`)
    plotFlag : bool
        If true, plots histogram with `label` and `alpha`.
    range : tuple
        If given, returns hist of vals within range; if not, uses all vals.

    Returns
    -------
    xarr : np 1d array
        Array of len(cnts) of bin midpoints
    cnts : np 1d array
        Bin counts (normalized if `normed`)
    """
    cnts, bedg = np.histogram(vals, bins=bins, range=range)
    xarr = bedg[:-1] + (bedg[1:] - bedg[:-1])/2.

    #Ebars calculation
    if retEbars:
        assert (not normCnts and normLogCnts)
        ### error propogation method for log10(cnts/C) ###
        ebars_log = 1./np.log(10) * np.sqrt(cnts)/cnts
        ### log(x+-dx) method for log10(cnts/C) ###
        # eb_lower = np.log10(np.true_divide(cnts, cnts-np.sqrt(cnts)))
        # eb_upper = -np.log10(np.true_divide(cnts, cnts+np.sqrt(cnts)))
        # ebars_log = (eb_lower, eb_upper)

        ebars = np.true_divide(np.sqrt(cnts), (bedg[1:] - bedg[:-1])*normScalar*1.0)
    # Normalization
    if normed:
        if normCnts:
            cnts =  np.true_divide(cnts, np.sum(cnts))
        if normBinsize:
            cnts = np.true_divide(cnts, (bedg[1:] - bedg[:-1]))
        if normScalar != 1:
            cnts = np.true_divide(cnts, normScalar)
        if normLogCnts:
             cnts = np.log10(cnts)

    if plotFlag:
        if ax:
            #ax.plot(xarr, cnts, plotOptions, ms=2,label=label, alpha=alpha)
            ax.plot(xarr, cnts, alpha=alpha, label=label)
        else:
            #plt.plot(xarr, cnts, plotOptions, ms=2,label=label, alpha=alpha)
            plt.plot(xarr, cnts, alpha=alpha, label=label)
    if retEbars:
        return (xarr, cnts, ebars, ebars_log)
    else:
        return (xarr, cnts)

def periodic_bcs(x, x_ref, boxsize):
    """Returns `x` (np array or scalar) adjusted to have shortest distance to `x_ref` (np array (only permitted if `x` is array) or scalar) according to PBCs of length boxsize."""
    return x + boxsize*((x-x_ref)<-(boxsize/2)) + -boxsize*((x-x_ref)>(boxsize/2))

def many_to_one_OLD(x1, x0, verbose=False, assert_x0_unique=True, assert_x1_in_x0=True):
    """Old version. Performs a many-to-one matching from `x1` to `x0`, and returns array of indices of `x0` such that x0[idx] is x1.
    Every element of `x1` must be in `x0`, and `x0` must be unique.
    """
    if assert_x0_unique:
        assert len(np.unique(x0))==len(x0), 'Elements of x0 are not unique.'
        if verbose:
            print('Assert 1: Elements of x0 are unique.')
    if assert_x1_in_x0:
        assert np.all(np.isin(x1,x0)), "Elements of x1 exist that are not in x0."
        if verbose:
            print('Assert 2: All elements of x1 are found in x0.')

    # Match x1 elements with x0.
    _, _, idx4 = np.intersect1d(x1, x0, return_indices=True, assume_unique=False)
    if verbose:
        print('Intersect x1 and x0.')

    # Unique x1 elements with inverse indices.
    _, idx_inv = np.unique(x1, return_inverse=True)
    if verbose:
        print('Unique x1')

    return idx4[idx_inv]

def many_to_one(x1arr, x0arr, verbose=False, assert_x0_unique=True, assert_x1_in_x0=True):
    """Performs a many-to-one matching from `x1arr` to `x0arr`, and returns array of indices of `x0arr` such that x0arr[idx] is x1arr.
    Every element of `x1arr` must be in `x0arr`, and `x0arr` must be unique.
    Based on np.intersect1d.
    """
    x1, ind1, inv1 = np.unique(x1arr, return_index=True, return_inverse=True)
    x0, ind2 = np.unique(x0arr, return_index=True)
    if assert_x0_unique:
        assert len(x0)==len(x0arr), 'Elements of x0arr are not unique.'
        if verbose:
            print('Assert 1: Elements of x0arr are unique.')
    if assert_x1_in_x0:
        assert np.all(np.isin(x1arr,x0arr)), "Elements of x1arr exist that are not in x0arr."
        if verbose:
            print('Assert 2: All elements of x1arr are found in x0arr.')
    aux = np.concatenate((x1, x0))
    aux_sort_indices = np.argsort(aux, kind='mergesort')
    aux = aux[aux_sort_indices]
    mask = aux[1:] == aux[:-1]
    x0_indices = ind2[ aux_sort_indices[1:][mask] - x1.size ]
    return x0_indices[inv1]

def reldif(x1,x2):
    '''Returns relative difference between arrays `x1` and `x2`.'''
    return np.abs(x1-x2)/((x1+x2)/2)

def nratioerr(n, nerr, nfid, nfiderr):
    '''Returns error of f=n/nfid given arrays `n` and `nfid` with errors `nerr` and `nfiderr`, respectively.'''
    return np.sqrt( (nerr/nfid)**2 + (n*nfiderr/nfid**2)**2 )

def real_fof_tags(fht):
    '''Takes an array of fof_halo_tags `fht` and returns an array of the same length of the "real" fof halo tags.
    For non-fragment halos, the real tag is just the f.h.t.
    For fragment halos, the real tag is the base f.h.t. of the fof group the fragment belongs to.
    '''
    return (fht<0)*np.bitwise_and(fht*-1, 0xffffffffffff) + (fht>=0)*fht

def intersect1d_GPU(ar1, ar2, assume_unique=False, return_indices=False):
    '''Based on np.intersect1d. Special fn for LJ cc gen'''
    import cupy as cp
    assert (not assume_unique) and return_indices

    ar1 = cp.asarray(ar1)                           # ar1 to GPU
    ar1, ind1 = cp.unique(ar1, return_index=True)   # ar1, ind1 on GPU
    ar1 = cp.asnumpy(ar1)                           # ar1 to Host
    ind1 = cp.asnumpy(ind1)                         # ind1 to Host

    ar2 = cp.asarray(ar2)                           # ar2 to GPU
    ar2, ind2 = cp.unique(ar2, return_index=True)   # ar2, ind2 on GPU
    ar2 = cp.asnumpy(ar2)                           # ar2 to Host
    ind2 = cp.asnumpy(ind2)                         # ind2 to Host

    aux = np.concatenate((ar1, ar2))                        # aux on Host
    aux_sort_indices = np.argsort(aux, kind='mergesort')    # aux_sort_indices on Host
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]                          # mask on Host
    int1d = aux[:-1][mask]                              # int1d on Host

    ar1_indices = aux_sort_indices[:-1][mask]           # ar1_indices on Host
    ar2_indices = aux_sort_indices[1:][mask] - ar1.size # ar2_indices on Host

    ar1_indices = ind1[ar1_indices]
    ar2_indices = ind2[ar2_indices]

    return int1d, ar1_indices, ar2_indices # return on Host

def many_to_one_GPU(ar1, ar2):
    '''Based on np.intersect1d. Special fn for LJ cc gen'''
    import cupy as cp
    ar1 = cp.asarray(ar1)                                                       # ar1 to GPU
    ar1, ind1, inv1 = cp.unique(ar1, return_index=True, return_inverse=True)    # ar1, ind1, inv1 on GPU
    ar1 = cp.asnumpy(ar1)                                                       # ar1 to Host
    ind1 = cp.asnumpy(ind1)                                                     # ind1 to Host
    inv1 = cp.asnumpy(inv1)                                                     # inv1 to Host

    ar2 = cp.asarray(ar2)                                                       # ar2 to GPU
    ar2, ind2 = cp.unique(ar2, return_index=True)                               # ar2, ind2 on GPU
    ar2 = cp.asnumpy(ar2)                                                       # ar2 to Host
    ind2 = cp.asnumpy(ind2)                                                     # ind2 to Host

    aux = np.concatenate((ar1, ar2))                                            # aux on Host
    aux_sort_indices = np.argsort(aux, kind='mergesort')                        # aux_sort_indices on Host
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]                                                  # mask on Host

    ar2_indices = ind2[ aux_sort_indices[1:][mask] - ar1.size ]                 # ar2_indices on Host

    return ar2_indices[inv1]                                                    # return on Host

### COSMOLOGY ###
def Omega_b(wb, h):
    return wb/h**2

def Omega_M(OMEGA_DM, wb, h):
    return OMEGA_DM + Omega_b(wb, h)

def Omega_L(OMEGA_DM, wb, h):
    return 1 - Omega_M(OMEGA_DM, wb, h)

def mparticle(OMEGA_DM, wb, h, Vi, Ni):
    """Returns particle mass in Msun/h.
    [Vi]: Mpc/h
    Vi and Ni are defined as V=Vi**3, N=Ni**3.
    """
    from astropy.constants import G
    from astropy import units as u
    
    OMEGA_M = Omega_M(OMEGA_DM, wb, h)
    # rhocrit = (3*(100 * u.km/u.s/u.Mpc)**2/(8*np.pi*G)).to(u.Msun/u.Mpc**3) #h^2 Msun/Mpc^3
    rhocrit = 2.77536627e11 #h^2 Msun/Mpc^3
    return (Vi/Ni)**3 * rhocrit.value * OMEGA_M

# Define `itk.SIMPARAMS` as dict of cosmology simulation parameters. See `simulationParams.yaml`.
import yaml, os                                                                                                                                   
with open( os.path.join(os.path.dirname(__file__), 'simulationParams.yaml'), 'r' ) as f: 
    SIMPARAMS = yaml.safe_load(f)
for k, v in SIMPARAMS.items():
    v['OMEGA_B'] = Omega_b(v['wb'], v['h'])
    v['OMEGA_M'] = Omega_M(v['OMEGA_DM'], v['wb'], v['h'])
    v['OMEGA_L'] = Omega_L(v['OMEGA_DM'], v['wb'], v['h'])
    v['PARTICLEMASS'] = mparticle(v['OMEGA_DM'], v['wb'], v['h'], v['Vi'], v['Ni'])