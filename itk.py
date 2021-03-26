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

def dict_deepcopy(dict1):
    '''Given a dict of np arrays, returns a deep copy of the dict.'''
    return { k : dict1[k].copy() for k in dict1.keys() }

def combine_dicts(dict1, dict2):
    '''Combines two dicts of 1D numpy arrays, where the keys in the returned dict are the common keys of dict1 and dict2.'''
    return { k : np.concatenate((dict1[k], dict2[k])) for k in set(dict1.keys()).intersection(dict2.keys()) }

def grep_dict(dict1, col, vals):
    '''Given a dict of 1D numpy arrays (all of same size), returns a new dict of only the rows that have a value of `col` that's in list `vals`.'''
    grep_dict_mask = np.isin(dict1[col], vals)
    return { k : dict1[k][grep_dict_mask].copy() for k in dict1.keys() }

def replace_elems(arr1, vals1, vals2):
    '''Given 1d np array `arr1`, returns a copy of `arr1` with all occurances of vals1[i] in `arr1` replaced with vals2[i].'''
    res = arr1.copy()
    for v1, v2 in zip(vals1, vals2):
        res = np.where(arr1==v1, v2, res)
    return res

def h5_read_dict(outfile, path='/', keys=None):
    """Read an HDF5 file with given `path` (or default /) and return dictionary of numpy arrays."""
    import h5py
    hf = h5py.File(outfile, 'r')
    if keys is None:
        keys = hf[path].keys()
    
    cc = {}
    for k in keys:
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

def gio_combine(flist, cols):
    """Reads cols of the sets of GenericIO files defined in `flist` and combines them into a dict."""
    res_list = {k:[] for k in cols}
    for f in flist:
        print(f)
        fdict = gio_read_dict(f, cols)
        for k in cols:
            res_list[k].append(fdict[k])
    res = { k:np.concatenate(res_list[k]) for k in cols }
    return res

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

def dist(x,y,z,x0,y0,z0, boxsize=None):
    """Find Euclidean distance between two points `(x,y,z)` and `(x0, y0, z0)`. 
    `(x,y,z)` should be 1D np arrays or scalars.
    `(x0,y0,z0)` should be 1D np arrays (only if `(x,y,z)` are 1D np arrays of same length) or scalars.
    If `boxsize` is defined, returns the shortest distance between the points according to PBCs of length `boxsize` for each dimension.
    """
    if boxsize:
        x = periodic_bcs(x, x0, boxsize)
        y = periodic_bcs(y, y0, boxsize)
        z = periodic_bcs(z, z0, boxsize)
    return np.sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )

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

def duplicate_rows(dict1, sort_key, printSame=False):
    '''Given `dict1` of 1d np arrays (all of the same size), examines rows which have duplicate values for the column `sort_key`.'''
    dict_sorted_idx = np.argsort(dict1[sort_key])
    dict_sorted = {k:dict1[k][dict_sorted_idx].copy() for k in dict1.keys() if dict1[k] is not None}
    
    vals_un, idx_un, cnts_un = np.unique(dict_sorted[sort_key], return_index=True, return_counts=True)
    
    idx_repeated_arr = np.flatnonzero(cnts_un>1)
    print(f'There are {len(idx_repeated_arr)} {sort_key} that have at least 1 duplicate.')
    print(f'Any {sort_key} shows up at most {cnts_un.max()} times.\n')
    for idx_repeated in idx_repeated_arr:
        first_idx = idx_un[idx_repeated]
        cnts_el = cnts_un[idx_repeated]
        print(f'\n{sort_key} {vals_un[idx_repeated]} is repeated {cnts_el} times.')
        for k in dict_sorted.keys():
            first_el = dict_sorted[k][first_idx]
            el_array = dict_sorted[k][first_idx:first_idx+cnts_el]
            if (el_array==first_el).all():
                if printSame:
                    print(f'{k} column matches for this {sort_key}: {first_el}.')
            else:
                print(f'{k} column DOES NOT match for this {sort_key}: {el_array}.')

def inrange(a, ra):
    a1, a2 = ra
    return (a1 <= a)&(a <= a2)

def binaryarray_outline(binaryarr, X, Y):
    '''Given a 2D np binary array `binaryarr`, and two 1D np arrays `X` and `Y` that label the columns and rows of `binaryarr`,
    respectively, returns a list of lines segments that form an outline of the contiguous regions of `binaryarr[y,x]==True`.
    
    The outline assumes that `binaryarr` will be plotted with `plt.pcolormesh(X, Y, binaryarr, shading='nearest')`.
    The output list of line segments has the form `line_segments_pts=[line1, line2, ...]` where `linen=((x0,y0),(x1,y1))`.
    
    The outline can be plotted using `ax.add_collection(matplotlib.collections.LineCollection(line_segments_pts))`.
    '''
    line_segments_pts = []
    for i, y in enumerate(Y):
        for j, x in enumerate(X[:-1]):
            if binaryarr[i,j]^binaryarr[i,j+1]:
                x0 = (x+X[j+1])/2
                dy_d = (y - Y[i-1])/2 if i>0 else (Y[i+1] - y)/2
                dy_u = (Y[i+1] - y)/2 if i<(len(Y)-1) else (y - Y[i-1])/2
                line_segments_pts.append( ((x0,y-dy_d),(x0,y+dy_u)) )

    for j, x in enumerate(X):
        for i, y in enumerate(Y[:-1]):
            if binaryarr[i,j]^binaryarr[i+1,j]:
                y0 = (y+Y[i+1])/2
                dx_l = (x - X[j-1])/2 if j>0 else (X[j+1] - x)/2
                dx_r = (X[j+1] - x)/2 if j<(len(X)-1) else (x - X[j-1])/2
                line_segments_pts.append( ((x-dx_l,y0),(x+dx_r,y0)) )
    return line_segments_pts

def is_unique_array(arr):
    '''Given 1D np array `arr`, returns True iff all elements of `arr` are unique (i.e. `arr` has no duplicate elements).'''
    return len(np.unique(arr)) == len(arr)

def indices_to_mask(arr, idx):
    '''Converts array of indices `idx` into boolean mask `mask` 
    such that `np.array_equal(np.flatnonzero(mask), np.unique(idx))` and `len(arr)==len(mask)`.
    '''
    mask = np.zeros_like(arr, dtype=bool)
    mask[idx] = True
    return mask

def sod_spatial_match(sod1, sod2, M1, M2, boxsize):
    '''Given two sod halo catalogs, finds unmatched halos (i.e. `fof_halo_tag`s that occur in only one of the two catalogs). 
    Attempts a spatial matching (using `fof_halo_center`) between the unmatched halos that have `fof_halo_mass` in [`M1`, `M2`].
    Returns two np arrays of `fof_halo_tag`s of unmatched halos in the mass bin in `sod1` and `sod2`, respectively, 
    where `fht1[i]` has been spatially matched to `fht2[i]`.
    '''
    assert is_unique_array(sod2['fof_halo_tag']), 'sod2 has duplicate FOF halo tags.'
    assert is_unique_array(sod1['fof_halo_tag']), 'sod1 has duplicate FOF halo tags.'

    assert len(sod2['fof_halo_tag'])==len(sod1['fof_halo_tag']), 'sod2 and sod1 have a different number of halos.'

    _, idx2, idx1 = np.intersect1d(sod2['fof_halo_tag'], sod1['fof_halo_tag'], assume_unique=True, return_indices=True)
    print(f"There are {len(idx2):,} matching FOF halo tags out of the {len(sod2['fof_halo_tag']):,} halos in each sod catalog ({len(idx2)/len(sod2['fof_halo_tag'])*100}%).")

    # Mask for `sodi` of unmatched halos that have fof_halo_mass in [M1, M2]
    maskmissing2 = (~indices_to_mask(sod2['fof_halo_tag'], idx2))&(inrange(sod2['fof_halo_mass'], (M1, M2)))
    maskmissing1 = (~indices_to_mask(sod1['fof_halo_tag'], idx1))&(inrange(sod1['fof_halo_mass'], (M1, M2)))

    assert np.sum(maskmissing2)==np.sum(maskmissing1), 'Unequal number of unmatched halos in sod2 and sod1 within the mass bin.'

    # For each unmatched sod1 halo in the mass bin, find closest unmatched sod2 halo in the mass bin.
    fht1 = sod1['fof_halo_tag'][maskmissing1]
    idx2_match = []
    for i1 in np.flatnonzero(maskmissing1):
        dr = dist(sod2['fof_halo_center_x'][maskmissing2],
                  sod2['fof_halo_center_y'][maskmissing2],
                  sod2['fof_halo_center_z'][maskmissing2],
                  sod1['fof_halo_center_x'][i1],
                  sod1['fof_halo_center_y'][i1],
                  sod1['fof_halo_center_z'][i1],
                  boxsize
                 )
        idx2_match.append( np.flatnonzero(maskmissing2)[np.argmin(dr)] )

    idx2_match = np.array(idx2_match)
    assert is_unique_array(idx2_match), '1:1 match between sod1 and sod2 unmatched halos in mass bin not found.'
    assert np.array_equal(sod1['fof_halo_mass'][maskmissing1], sod2['fof_halo_mass'][idx2_match]), 'fof halo mass of all spatial matches does not match.'
    
    fht2 = sod2['fof_halo_tag'][idx2_match]
    return fht1, fht2

def intersect1d_parallel(comm, rank, root, arr_root, arr_local, dtype_arr, data_local, dtype_data, assume_unique=True):
    '''
    Performs one-to-one element matching between an array on one rank and other arrays on all ranks.
    Given a 1D numpy array `arr_root` on rank `root`, broadcasts `arr_root` to all ranks.
    Performs np.intersect1d between `arr_root` and `arr_local` on each rank.
    On each rank, `data_local` is an array of shape `arr_local` with data associated with each `arr_local` element.
    Matches from all ranks between `arr_root` and the corresponding data are gathered on `root`.
    
    Returns on `root`:
        `recvbuf_idx1`: indices of `arr_root`
        `recvbuf_data`: corresponding data values found from all ranks
    
    Notes:
        Argument `arr_root` should be None on all ranks except `root`.
        `arr_root` and `arr_local` should have elements of dtype `dtype_arr`, and `data_local` should have elements of dtype `dtype_data`.
        `arr_root` and each `arr_local` are assumed to be unique.
    
    Reference: https://stackoverflow.com/a/38008452
    '''
    if rank == root:
        len_arr_root = len(arr_root)
    else:
        len_arr_root = None
    len_arr_root = comm.bcast(len_arr_root, root=root)
    if rank == root:
        arr_root_cpy = arr_root.copy()
    else:
        arr_root_cpy = np.empty(len_arr_root, dtype=dtype_arr)
    comm.Bcast(arr_root_cpy, root=root)

    _, idx1, idx2 = np.intersect1d( arr_root_cpy, arr_local, return_indices=True, assume_unique=assume_unique )
    sendcounts = np.array( comm.gather(len(idx1), root) )
    if rank == root:
        sendcounts_total = np.sum(sendcounts)
        recvbuf_idx1 = np.empty(sendcounts_total, dtype=np.int64)
        recvbuf_data = np.empty(sendcounts_total, dtype=dtype_data)
    else:
        recvbuf_idx1 = None
        recvbuf_data = None
    comm.Gatherv(sendbuf=idx1, recvbuf=(recvbuf_idx1, sendcounts), root=root)
    comm.Gatherv(sendbuf=data_local[idx2], recvbuf=(recvbuf_data, sendcounts), root=root)

    return recvbuf_idx1, recvbuf_data

def many_to_one_parallel(comm, rank, root, arr_root, arr_local, dtype_arr, data_local, dtype_data):
    '''
    Performs many-to-one element matching between an array on rank `root`, and unique arrays on all ranks.

    Parameters:
        `arr_root`: array on rank `root` that contains (possibly repeated) elements that are all found in `arr_local` over all ranks
        `arr_local`: unique array on `rank` (elements must be unique over all ranks)
        `dtype_arr`: dtype of elements of `arr_root` and `arr_local`
        `data_local`: array on `rank` with data corresponding to each element of `arr_local`
        `dtype_data`: dtype of elements of `data_local`

    Returns on `root`:
        `Data`: array of shape `arr_root` whose elements are from `data_local` (of all ranks) and correspond to `arr_root`
    '''
    if rank == root:
        vals, inv = np.unique(arr_root, return_inverse=True)
    else:
        vals, inv = None, None
    idx3, data = intersect1d_parallel(comm, rank, root, vals, arr_local, dtype_arr, data_local, dtype_data)
    if rank == root:
        assert len(idx3)==len(vals), f'Unique match not found for all elements of arr_root for rank {root}'
        data_un = np.empty_like(data)
        data_un[idx3] = data
        Data = data_un[inv]
    else:
        Data = None
    return Data

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
    return (Vi/Ni)**3 * rhocrit * OMEGA_M

# Define `itk.SIMPARAMS` as dict of cosmology simulation parameters. See `simulationParams.yaml`.
import yaml, os                                                                                                                                   
with open( os.path.join(os.path.dirname(__file__), 'simulationParams.yaml'), 'r' ) as f: 
    SIMPARAMS = yaml.safe_load(f)
for k, v in SIMPARAMS.items():
    v['OMEGA_B'] = Omega_b(v['wb'], v['h'])
    v['OMEGA_M'] = Omega_M(v['OMEGA_DM'], v['wb'], v['h'])
    v['OMEGA_L'] = Omega_L(v['OMEGA_DM'], v['wb'], v['h'])
    v['PARTICLEMASS'] = mparticle(v['OMEGA_DM'], v['wb'], v['h'], v['Vi'], v['Ni'])