from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# Set up latex for plt and change default figure dpi.
def plt_latex(dpi=120):
    import matplotlib.colors
    # Latex for plt
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    # default figure size
    matplotlib.rcParams['figure.dpi'] = dpi

# Read an HDF5 file with given path (or default /) and return dictionary of numpy arrays.
def read_dict_from_disk(outfile, path='/'):
    import h5py
    hf = h5py.File(outfile, 'r')
    cc = {}
    for k in hf[path].keys():
        cc[k] = np.array( hf[path][k] )
    hf.close()
    return cc

# Read cols of GenericIO file f and return dictionary of numpy arrays.
def gio_read_dict(f, cols):
    import genericio as gio
    return { k:gio.gio_read(f, k)[0] for k in cols }

# Pickles dictionary d in outfile f.
def pickle_save_dict(f, d):
    import pickle
    pickle.dump( d, open( f, 'wb' ) )

# Returns pickled object in f.
def loadpickle(f):
    import pickle
    return pickle.load( open( f, 'rb' ) )

def hist(vals, bins=500, normed=False, plotFlag=True, label='', alpha=1, range=None, normScalar=1, normCnts=False, normBinsize=False):
    """
    Returns histogram of vals in (default 500) bins.
    
    Parameters
    ----------
    vals : np 1d array 
        Data points
    normed : bool
        If true, normalizes by counts (if normCnts), bin size (if normBinsize), scalar (if normScalar given)
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
    
    # Normalization
    if normed:
        if normCnts:
            cnts =  np.true_divide(cnts, np.sum(cnts))
        if normBinsize:
            cnts = np.true_divide(cnts, (bedg[1:] - bedg[:-1]))
        if normScalar != 1:
            cnts = np.true_divide(cnts, normScalar)

    if plotFlag:
        plt.plot(xarr, cnts, label=label, alpha=alpha)
    return (xarr, cnts)