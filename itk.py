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

def gio_read_dict(f, cols):
    """Read cols of GenericIO file f and return dictionary of numpy arrays."""
    import genericio as gio
    return { k:gio.gio_read(f, k)[0] for k in cols }

def pickle_save_dict(f, d):
    """Pickles dictionary d in outfile f."""
    import pickle
    pickle.dump( d, open( f, 'wb' ) )

def loadpickle(f):
    """Returns pickled object in f."""
    import pickle
    return pickle.load( open( f, 'rb' ) )

def hist(vals, bins=500, normed=False, plotFlag=True, label='', alpha=1, range=None, normScalar=1, normCnts=False, normBinsize=False, normLogCnts=False, plotOptions='.'):
    """
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
        plt.plot(xarr, cnts, plotOptions, ms=2,label=label, alpha=alpha)
    return (xarr, cnts)

def periodic_bcs(x, x_ref, boxsize):
    """Returns `x` (np array or scalar) adjusted to have shortest distance to `x_ref` (np array (only permitted if `x` is array) or scalar) according to PBCs of length boxsize."""
    return x + boxsize*((x-x_ref)<-(boxsize/2)) + -boxsize*((x-x_ref)>(boxsize/2))
