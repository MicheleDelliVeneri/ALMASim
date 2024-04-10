import numpy as np
import math
import astropy.units as U
from Hdecompose.atomic_frac import atomic_frac
from illustris_python.snapshot import getSnapOffsets, loadSubset
from martini.sources.sph_source import SPHSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import (AdaptiveKernel, CubicSplineKernel,
                                 GaussianKernel, find_fwhm,  WendlandC2Kernel)

# -------------------------- Modified functions from illustris-tng -------------------------- #

def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)
        
    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['dmlowres']:
        return 2 # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5
    
    raise Exception("Unknown particle type name.")

def gcPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a group catalog HDF5 file (modify as needed). """
    gcPath = basePath + '/groups_%03d/' % snapNum
    filePath1 = gcPath + 'groups_%03d.%d.hdf5' % (snapNum, chunkNum)
    filePath2 = gcPath + 'fof_subhalo_tab_%03d.%d.hdf5' % (snapNum, chunkNum)

    if isfile(expanduser(filePath1)):
        return filePath1
    return filePath2

def offsetPath(basePath, snapNum):
    """ Return absolute path to a separate offset file (modify as needed). """
    offsetPath = basePath + '/../postprocessing/offsets/offsets_%03d.hdf5' % snapNum

    return offsetPath

def loadObjects(basePath, snapNum, gName, nName, fields):
    """ Load either halo or subhalo information from the group catalog. """
    result = {}

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())

        if 'N'+nName+'_Total' not in header and nName == 'subgroups':
            nName = 'subhalos' # alternate convention

        result['count'] = f['Header'].attrs['N' + nName + '_Total']

        if not result['count']:
            print('warning: zero groups, empty return (snap=' + str(snapNum) + ').')
            return result

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for field in fields:
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = result['count']

            # allocate within return dict
            result[field] = np.zeros(shape, dtype=f[gName][field].dtype)

    # loop over chunks
    wOffset = 0

    for i in range(header['NumFiles']):
        f = h5py.File(gcPath(basePath, snapNum, i), 'r')

        if not f['Header'].attrs['N'+nName+'_ThisFile']:
            continue  # empty file chunk

        # loop over each requested field
        for field in fields:
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")

            # shape and type
            shape = f[gName][field].shape

            # read data local to the current file
            if len(shape) == 1:
                result[field][wOffset:wOffset+shape[0]] = f[gName][field][0:shape[0]]
            else:
                result[field][wOffset:wOffset+shape[0], :] = f[gName][field][0:shape[0], :]

        wOffset += shape[0]
        f.close()

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result

def loadSubhalos(basePath, snapNum, fields=None):
    """ Load all subhalo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """

    return loadObjects(basePath, snapNum, "Subhalo", "subgroups", fields)

def loadHalos(basePath, snapNum, fields=None):
    """ Load all halo information from the entire group catalog for one snapshot
       (optionally restrict to a subset given by fields). """

    return loadObjects(basePath, snapNum, "Group", "groups", fields)

def loadHeader(basePath, snapNum):
    """ Load the group catalog header. """
    with h5py.File(gcPath(basePath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())

    return header

def load(basePath, snapNum):
    """ Load complete group catalog all at once. """
    r = {}
    r['subhalos'] = loadSubhalos(basePath, snapNum)
    r['halos']    = loadHalos(basePath, snapNum)
    r['header']   = loadHeader(basePath, snapNum)
    return r

def loadSingle(basePath, snapNum, haloID=-1, subhaloID=-1):
    """ Return complete group catalog information for one halo or subhalo. """
    if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID >= 0 else "Group"
    searchID = subhaloID if subhaloID >= 0 else haloID

    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            offsets = f['FileOffsets/'+gName][()]
    else:
        # use header of group catalog
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            offsets = f['Header'].attrs['FileOffsets_'+gName]

    offsets = searchID - offsets
    fileNum = np.max(np.where(offsets >= 0))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        for haloProp in f[gName].keys():
            result[haloProp] = f[gName][haloProp][groupOffset]

    return result

def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath1 = snapPath + 'snap_' + str(snapNum).zfill(3) + '.' + str(chunkNum) + '.hdf5'
    filePath2 = filePath1.replace('/snap_', '/snapshot_')

    if isfile(filePath1):
        return filePath1
    return filePath2

def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    if 'NumPart_Total_HighWord' not in header:
        return header['NumPart_Total'] # new uint64 convention

    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

    return nPart

def loadSubset(basePath, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False, outPath=None):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            if os.path.isfile(snapPath(basePath, snapNum, i)):
                print('Found')
                f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            else:
                print('Not Found')
                api_key = '8f578b92e700fae3266931f4d785f82c'
                url = f'http://www.tng-project.org/api/TNG100-1/files/snapshot-{str(snapNum)}'
                subdir = os.path.join('output', 'snapdir_0{}'.format(str(i)))
                cmd = f'wget -q --progress=bar  --content-disposition --header="API-Key:{api_key}" {url}.{i}.hdf5'
                print(f'Downloading {message} {i} ...')
                if outPath is not None:
                    os.chdir(outPath)
                subprocess.check_call(cmd, shell=True)
                print('Done.')
                f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        if not os.path.isfile(snapPath(basePath, snapNum, fileNum)):
            print(f'Particles are found in Snapshot {fileNum} which is not present on disk')
            # move directory to the correct directory data !!!
            api_key = '8f578b92e700fae3266931f4d785f82c'
            url = f'http://www.tng-project.org/api/TNG100-1/files/snapshot-{str(snapNum)}'
            subdir = os.path.join('output', 'snapdir_0{}'.format(str(fileNum)))
            savePath = os.path.join(basePath, 'snapdir_0{}'.format(str(snapNum)))
            cmd = f'wget -P {savePath} -q --progress=bar  --content-disposition --header="API-Key:{api_key}" {url}.{fileNum}.hdf5'
            if outPath is not None:
                os.chdir(outPath)
            print(f'Downloading Snapshot {fileNum} in {savePath}...')
            subprocess.check_call(cmd, shell=True)
            print('Done.')
        print('Checking File {}...'.format(fileNum))
        f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypdeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first
        print('Loading File {}...'.format(fileNum))
        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result

def getSnapOffsets(basePath, snapNum, id, type):
    """ Compute offsets within snapshot for a particular group/subgroup. """
    r = {}
    print(f'Checking offset in Snapshot {snapNum} for grouphalo {id}')
    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/'+type][()]
            r['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])  # consistency
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]
            r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']

    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        r['lenType'] = f[type][type+'LenType'][groupOffset, :]

    # old or new format: load the offset (by type) of  this group/subgroup within the snapshot
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            r['offsetType'] = f[type+'/SnapByType'][id, :]

            # add TNG-Cluster specific offsets if present
            if 'OriginalZooms' in f:
                for key in f['OriginalZooms']:
                    r[key] = f['OriginalZooms'][key][()] 
    else:
        with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
            r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset, :]

    return r

def loadSubhalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific subhalo
        (optionally restricted to a subset fields). """
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Subhalo")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)

def loadHalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)

def loadOriginalZoom(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type corresponding to an
        original (entire) zoom simulation. TNG-Cluster specific.
        (optionally restricted to a subset fields). """
    # load fuzz length, compute offset, call loadSubset                                                                     
    subset = getSnapOffsets(basePath, snapNum, id, "Group")

    # identify original halo ID and corresponding index
    halo = loadSingle(basePath, snapNum, haloID=id)
    assert 'GroupOrigHaloID' in halo, 'Error: loadOriginalZoom() only for the TNG-Cluster simulation.'
    orig_index = np.where(subset['HaloIDs'] == halo['GroupOrigHaloID'])[0][0]

    # (1) load all FoF particles/cells
    subset['lenType'] = subset['GroupsTotalLengthByType'][orig_index, :]
    subset['offsetType'] = subset['GroupsSnapOffsetByType'][orig_index, :]

    data1 = loadSubset(basePath, snapNum, partType, fields, subset=subset)

    # (2) load all non-FoF particles/cells
    subset['lenType'] = subset['OuterFuzzTotalLengthByType'][orig_index, :]
    subset['offsetType'] = subset['OuterFuzzSnapOffsetByType'][orig_index, :]

    data2 = loadSubset(basePath, snapNum, partType, fields, subset=subset)

    # combine and return
    if isinstance(data1, np.ndarray):
        return np.concatenate((data1,data2), axis=0)
    
    data = {'count':data1['count']+data2['count']}
    for key in data1.keys():
        if key == 'count': continue
        data[key] = np.concatenate((data1[key],data2[key]), axis=0)
    return data

def get_particles_num(basePath, outputPath, snapNum, subhaloID):
    basePath = os.path.join(basePath, "TNG100-1", "output", )
    print('Looking for Subhalo %d in snapshot %d' % (subhaloID, snapNum))
    partType = 'gas'
    subset = getSnapOffsets(basePath, snapNum, subhaloID, "Subhalo")
    subhalo = loadSubset(basePath, snapNum, partType, subset=subset)
    os.chdir(basePath)
    gas = il.snapshot.loadSubhalo(basePath, snapNum, subhaloID, partType)
    if 'Coordinates' in gas.keys():
        gas_num = len(gas['Coordinates'])
    else:
        gas_num = 0
    return gas_num

class myTNGSource(SPHSource):
    def __init__(
        self,
        snapNum,
        subID,
        basePath=None,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    ):
        X_H = 0.76
        full_fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "CenterOfMass",
            "GFM_Metals",
        )
        mdi_full = [None, None, None, None, None, None, 0]
        mini_fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "Coordinates",
        )
        data_header = loadHeader(basePath, snapNum)
        data_sub = loadSingle(basePath, snapNum, subhaloID=subID)
        haloID = data_sub["SubhaloGrNr"]
        subset_g = getSnapOffsets(basePath, snapNum, haloID, "Group")
        try:
            data_g = loadSubset(
                    basePath,
                    snapNum,
                    "gas",
                    fields=full_fields_g,
                    subset=subset_g,
                    mdi=mdi_full,
                )

            minisnap = False
        except Exception as exc:
            print(exc.args)
            if ("Particle type" in exc.args[0]) and ("does not have field" in exc.args[0]):
                data_g.update(
                        loadSubset(
                            basePath,
                            snapNum,
                            "gas",
                            fields=("CenterOfMass",),
                            subset=subset_g,
                            sq=False,
                        )
                    )
                minisnap = True
                X_H_g = X_H
            else:
                raise
        X_H_g = (
                X_H if minisnap else data_g["GFM_Metals"])  # only loaded column 0: Hydrogen
        a = data_header["Time"]
        z = data_header["Redshift"]
        h = data_header["HubbleParam"]
        xe_g = data_g["ElectronAbundance"]
        rho_g = data_g["Density"] * 1e10 / h * U.Msun * np.power(a / h * U.kpc, -3)
        u_g = data_g["InternalEnergy"]  # unit conversion handled in T_g
        mu_g = 4 * C.m_p.to(U.g).value / (1 + 3 * X_H_g + 4 * X_H_g * xe_g)
        gamma = 5.0 / 3.0  # see http://www.tng-project.org/data/docs/faq/#gen4
        T_g = (gamma - 1) * u_g / C.k_B.to(U.erg / U.K).value * 1e10 * mu_g * U.K
        m_g = data_g["Masses"] * 1e10 / h * U.Msun
        # cast to float64 to avoid underflow error
        nH_g = U.Quantity(rho_g * X_H_g / mu_g, dtype=np.float64) / C.m_p
        # In TNG_corrections I set f_neutral = 1 for particles with density
        # > .1cm^-3. Might be possible to do a bit better here, but HI & H2
        # tables for TNG will be available soon anyway.
        fatomic_g = atomic_frac(
            z, nH_g, T_g, rho_g, X_H_g, onlyA1=True, TNG_corrections=True
            )
        mHI_g = m_g * X_H_g * fatomic_g
        try:
            xyz_g = data_g["CenterOfMass"] * a / h * U.kpc
        except KeyError:
            xyz_g = data_g["Coordinates"] * a / h * U.kpc
        vxyz_g = data_g["Velocities"] * np.sqrt(a) * U.km / U.s
        V_cell = (
            data_g["Masses"] / data_g["Density"] * np.power(a / h * U.kpc, 3)
            )  # Voronoi cell volume
        r_cell = np.power(3.0 * V_cell / 4.0 / np.pi, 1.0 / 3.0).to(U.kpc)
        # hsm_g has in mind a cubic spline that =0 at r=h, I think
        hsm_g = 2.5 * r_cell * find_fwhm(CubicSplineKernel().kernel)
        xyz_centre = data_sub["SubhaloPos"] * a / h * U.kpc
        xyz_g -= xyz_centre
        vxyz_centre = data_sub["SubhaloVel"] * np.sqrt(a) * U.km / U.s
        vxyz_g -= vxyz_centre
        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )
        return

class DataCube(object):
    """
    Handles creation and management of the data cube itself.

    Basic usage simply involves initializing with the parameters listed below.
    More advanced usage might arise if designing custom classes for other sub-
    modules, especially beams. To initialize a DataCube from a saved state, see
    DataCube.load_state.

    Parameters
    ----------
    n_px_x : int, optional
        Pixel count along the x (RA) axis. Even integers strongly preferred.
        (Default: 256.)

    n_px_y : int, optional
        Pixel count along the y (Dec) axis. Even integers strongly preferred.
        (Default: 256.)

    n_channels : int, optional
        Number of channels along the spectral axis. (Default: 64.)

    px_size : Quantity, with dimensions of angle, optional
        Angular scale of one pixel. (Default: 15 arcsec.)

    channel_width : Quantity, with dimensions of velocity or frequency, optional
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency. (Default: 4 km/s.)

    velocity_centre : Quantity, with dimensions of velocity or frequency, optional
        Velocity (or frequency) of the centre along the spectral axis.
        (Default: 0 km/s.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension of the cube centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination of the cube centroid. (Default: 0 deg.)

    stokes_axis : bool, optional
        Whether the datacube should be initialized with a Stokes' axis. (Default: False.)

    See Also
    --------
    load_state
    """

    def __init__(
        self,
        n_px_x=256,
        n_px_y=256,
        n_channels=64,
        px_size=15.0 * U.arcsec,
        channel_width=4.0 * U.km * U.s**-1,
        velocity_centre=0.0 * U.km * U.s**-1,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        stokes_axis=False,
    ):
        self.HIfreq = 1.420405751e9 * U.Hz
        self.stokes_axis = stokes_axis
        datacube_unit = U.Jy * U.pix**-2
        self._array = np.zeros((n_px_x, n_px_y, n_channels)) * datacube_unit
        if self.stokes_axis:
            self._array = self._array[..., np.newaxis]
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size = px_size
        self.arcsec2_to_pix = (
            U.Jy * U.pix**-2,
            U.Jy * U.arcsec**-2,
            lambda x: x / self.px_size**2,
            lambda x: x * self.px_size**2,
        )
        self.velocity_centre = velocity_centre.to(
            U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.channel_width = np.abs(
            (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(self.HIfreq)
                )
                + 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(self.HIfreq)
                )
                - 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.ra = ra
        self.dec = dec
        self.padx = 0
        self.pady = 0
        self._freq_channel_mode = False
        self._init_wcs()
        self._channel_mids()
        self._channel_edges()

        return

    def _init_wcs(self):
        self.wcs = wcs.WCS(naxis=3)
        self.wcs.wcs.crpix = [
            self.n_px_x / 2.0 + 0.5,
            self.n_px_y / 2.0 + 0.5,
            self.n_channels / 2.0 + 0.5,
        ]
        self.units = [U.deg, U.deg, U.m / U.s]
        self.wcs.wcs.cunit = [unit.to_string("fits") for unit in self.units]
        self.wcs.wcs.cdelt = [
            -self.px_size.to_value(self.units[0]),
            self.px_size.to_value(self.units[1]),
            self.channel_width.to_value(
                self.units[2], equivalencies=U.doppler_radio(self.HIfreq)
            ),
        ]
        self.wcs.wcs.crval = [
            self.ra.to_value(self.units[0]),
            self.dec.to_value(self.units[1]),
            self.velocity_centre.to_value(
                self.units[2], equivalencies=U.doppler_radio(self.HIfreq)
            ),
        ]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
        self.wcs.wcs.specsys = "GALACTOC"
        if self.stokes_axis:
            self.wcs = wcs.utils.add_stokes_axis_to_wcs(self.wcs, self.wcs.wcs.naxis)
        return

    def _channel_mids(self):
        """
        Calculate the centres of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels),
            np.zeros(self.n_channels),
            np.arange(self.n_channels) - 0.5,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels),)
        self.channel_mids = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def _channel_edges(self):
        """
        Calculate the edges of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels + 1),
            np.zeros(self.n_channels + 1),
            np.arange(self.n_channels + 1) - 1,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels + 1),)
        self.channel_edges = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def spatial_slices(self):
        """
        Return an iterator over the spatial 'slices' of the cube.

        Returns
        -------
        out : iterator
            Iterator over the spatial 'slices' of the cube.
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].transpose((2, 0, 1)))

    def spectra(self):
        """
        Return an iterator over the spectra (one in each spatial pixel).

        Returns
        -------
        out : iterator
            Iterator over the spectra (one in each spatial pixel).
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].reshape(self.n_px_x * self.n_px_y, self.n_channels))

    def freq_channels(self):
        """
        Convert spectral axis to frequency units.
        """
        if self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = -np.abs(
            (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.Hz, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.wcs.wcs.ctype[2] = "FREQ"
        self.units[2] = U.Hz
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = True
        self._channel_mids()
        self._channel_edges()
        return

    def velocity_channels(self):
        """
        Convert spectral axis to velocity units.
        """
        if not self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = np.abs(
            (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.wcs.wcs.ctype[2] = "VRAD"
        self.units[2] = U.m * U.s**-1
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = False
        self._channel_mids()
        self._channel_edges()
        return

    def add_pad(self, pad):
        """
        Resize the cube to add a padding region in the spatial direction.

        Accurate convolution with a beam requires a cube padded according to
        the size of the beam kernel (its representation sampled on a grid with
        the same spacing). The beam class is required to handle defining the
        size of pad required.

        Parameters
        ----------
        pad : 2-tuple (or other sequence)
            Number of pixels to add in the x (RA) and y (Dec) directions.

        See Also
        ----------
        drop_pad
        """

        if self.padx > 0 or self.pady > 0:
            raise RuntimeError("Tried to add padding to already padded datacube array.")
        tmp = self._array
        shape = (self.n_px_x + pad[0] * 2, self.n_px_y + pad[1] * 2, self.n_channels)
        if self.stokes_axis:
            shape = shape + (1,)
        self._array = np.zeros(shape)
        self._array = self._array * tmp.unit
        xregion = np.s_[pad[0] : -pad[0]] if pad[0] > 0 else np.s_[:]
        yregion = np.s_[pad[1] : -pad[1]] if pad[1] > 0 else np.s_[:]
        self._array[xregion, yregion, ...] = tmp
        extend_crpix = [pad[0], pad[1], 0]
        if self.stokes_axis:
            extend_crpix.append(0)
        self.wcs.wcs.crpix += np.array(extend_crpix)
        self.padx, self.pady = pad
        return

    def drop_pad(self):
        """
        Remove the padding added using add_pad.

        After convolution, the pad region contains meaningless information and
        can be discarded.

        See Also
        --------
        add_pad
        """

        if (self.padx == 0) and (self.pady == 0):
            return
        self._array = self._array[self.padx : -self.padx, self.pady : -self.pady, ...]
        retract_crpix = [self.padx, self.pady, 0]
        if self.stokes_axis:
            retract_crpix.append(0)
        self.wcs.wcs.crpix -= np.array(retract_crpix)
        self.padx, self.pady = 0, 0
        return

    def copy(self):
        """
        Produce a copy of the DataCube.

        May be especially useful to create multiple datacubes with differing
        intermediate steps.

        Returns
        -------
        out : DataCube
            Copy of the DataCube object.
        """
        in_freq_channel_mode = self._freq_channel_mode
        if in_freq_channel_mode:
            self.velocity_channels()
        copy = DataCube(
            self.n_px_x,
            self.n_px_y,
            self.n_channels,
            self.px_size,
            self.channel_width,
            self.velocity_centre,
            self.ra,
            self.dec,
        )
        copy.padx, copy.pady = self.padx, self.pady
        copy.wcs = self.wcs
        copy._freq_channel_mode = self._freq_channel_mode
        copy.channel_edges = self.channel_edges
        copy.channel_mids = self.channel_mids
        copy._array = self._array.copy()
        return copy

    def save_state(self, filename, overwrite=False):
        """
        Write a file from which the current DataCube state can be
        re-initialized (see DataCube.load_state). Note that h5py must be
        installed for use. NOT for outputting mock observations, for this
        see Martini.write_fits and Martini.write_hdf5.

        Parameters
        ----------
        filename : str
            File to write.

        overwrite : bool
            Whether to allow overwriting existing files (default: False).

        See Also
        --------
        load_state
        """
        import h5py

        mode = "w" if overwrite else "w-"
        with h5py.File(filename, mode=mode) as f:
            array_unit = self._array.unit
            f["_array"] = self._array.to_value(array_unit)
            f["_array"].attrs["datacube_unit"] = str(array_unit)
            f["_array"].attrs["n_px_x"] = self.n_px_x
            f["_array"].attrs["n_px_y"] = self.n_px_y
            f["_array"].attrs["n_channels"] = self.n_channels
            px_size_unit = self.px_size.unit
            f["_array"].attrs["px_size"] = self.px_size.to_value(px_size_unit)
            f["_array"].attrs["px_size_unit"] = str(px_size_unit)
            channel_width_unit = self.channel_width.unit
            f["_array"].attrs["channel_width"] = self.channel_width.to_value(
                channel_width_unit
            )
            f["_array"].attrs["channel_width_unit"] = str(channel_width_unit)
            velocity_centre_unit = self.velocity_centre.unit
            f["_array"].attrs["velocity_centre"] = self.velocity_centre.to_value(
                velocity_centre_unit
            )
            f["_array"].attrs["velocity_centre_unit"] = str(velocity_centre_unit)
            ra_unit = self.ra.unit
            f["_array"].attrs["ra"] = self.ra.to_value(ra_unit)
            f["_array"].attrs["ra_unit"] = str(ra_unit)
            dec_unit = self.dec.unit
            f["_array"].attrs["dec"] = self.dec.to_value(dec_unit)
            f["_array"].attrs["dec_unit"] = str(self.dec.unit)
            f["_array"].attrs["padx"] = self.padx
            f["_array"].attrs["pady"] = self.pady
            f["_array"].attrs["_freq_channel_mode"] = int(self._freq_channel_mode)
            f["_array"].attrs["stokes_axis"] = self.stokes_axis
        return

    @classmethod
    def load_state(cls, filename):
        """
        Initialize a DataCube from a state saved using DataCube.save_state.
        Note that h5py must be installed for use. Note that ONLY the DataCube
        state is restored, other modules and their configurations are not
        affected.

        Parameters
        ----------
        filename : str
            File to open.

        Returns
        -------
        out : martini.DataCube
            A suitably initialized DataCube object.

        See Also
        --------
        save_state
        """
        import h5py

        with h5py.File(filename, mode="r") as f:
            n_px_x = f["_array"].attrs["n_px_x"]
            n_px_y = f["_array"].attrs["n_px_y"]
            n_channels = f["_array"].attrs["n_channels"]
            px_size = f["_array"].attrs["px_size"] * U.Unit(
                f["_array"].attrs["px_size_unit"]
            )
            channel_width = f["_array"].attrs["channel_width"] * U.Unit(
                f["_array"].attrs["channel_width_unit"]
            )
            velocity_centre = f["_array"].attrs["velocity_centre"] * U.Unit(
                f["_array"].attrs["velocity_centre_unit"]
            )
            ra = f["_array"].attrs["ra"] * U.Unit(f["_array"].attrs["ra_unit"])
            dec = f["_array"].attrs["dec"] * U.Unit(f["_array"].attrs["dec_unit"])
            stokes_axis = bool(f["_array"].attrs["stokes_axis"])
            D = cls(
                n_px_x=n_px_x,
                n_px_y=n_px_y,
                n_channels=n_channels,
                px_size=px_size,
                channel_width=channel_width,
                velocity_centre=velocity_centre,
                ra=ra,
                dec=dec,
                stokes_axis=stokes_axis,
            )
            D._init_wcs()
            D.add_pad((f["_array"].attrs["padx"], f["_array"].attrs["pady"]))
            if bool(f["_array"].attrs["_freq_channel_mode"]):
                D.freq_channels()
            D._array = f["_array"] * U.Unit(f["_array"].attrs["datacube_unit"])
        return D

    def __repr__(self):
        """
        Print the contents of the data cube array itself.

        Returns
        -------
        out : str
            Text representation of the DataCube._array contents.
        """
        return self._array.__repr__()

class Martini:
    """
    Creates synthetic HI data cubes from simulation data.

    Usual use of martini involves first creating instances of classes from each
    of the required and optional sub-modules, then creating a Martini with
    these instances as arguments. The object can then be used to create
    synthetic observations, usually by calling `insert_source_in_cube`,
    (optionally) `add_noise`, (optionally) `convolve_beam` and `write_fits` in
    order.

    Parameters
    ----------
    source : an instance of a class derived from martini.source._BaseSource
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). Sources leveraging the simobj package for reading
        simulation data (github.com/kyleaoman/simobj) and a few test sources
        (e.g. single particle) are provided, creation of customized sources,
        for instance to leverage other interfaces to simulation data, is
        straightforward. See sub-module documentation.

    datacube : martini.DataCube instance
        A description of the datacube to create, including pixels, channels,
        sky position. See sub-module documentation.

    beam : an instance of a class derived from beams._BaseBeam, optional
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See sub-module documentation.

    noise : an instance of a class derived from noise._BaseNoise, optional
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        sub-module documentation.

    sph_kernel : an instance of a class derived from sph_kernels._BaseSPHKernel
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        SPH kernel submodule documentation for guidance.

    spectral_model : an instance of a class derived from \
    spectral_models._BaseSpectrum
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See sub-module documentation.

    quiet : bool
        If True, suppress output to stdout. (Default: False)

    See Also
    --------
    martini.sources
    martini.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models

    Examples
    --------
    More detailed examples can be found in the examples directory in the github
    distribution of the package.

    The following example illustrates basic use of martini, using a (very!)
    crude model of a gas disk. This example can be run by doing
    'from martini import demo; demo()'::

        # ------make a toy galaxy----------
        N = 500
        phi = np.random.rand(N) * 2 * np.pi
        r = []
        for L in np.random.rand(N):

            def f(r):
                return L - 0.5 * (2 - np.exp(-r) * (np.power(r, 2) + 2 * r + 2))

            r.append(fsolve(f, 1.0)[0])
        r = np.array(r)
        # exponential disk
        r *= 3 / np.sort(r)[N // 2]
        z = -np.log(np.random.rand(N))
        # exponential scale height
        z *= 0.5 / np.sort(z)[N // 2] * np.sign(np.random.rand(N) - 0.5)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz_g = np.vstack((x, y, z)) * U.kpc
        # linear rotation curve
        vphi = 100 * r / 6.0
        vx = -vphi * np.sin(phi)
        vy = vphi * np.cos(phi)
        # small pure random z velocities
        vz = (np.random.rand(N) * 2.0 - 1.0) * 5
        vxyz_g = np.vstack((vx, vy, vz)) * U.km * U.s**-1
        T_g = np.ones(N) * 8e3 * U.K
        mHI_g = np.ones(N) / N * 5.0e9 * U.Msun
        # ~mean interparticle spacing smoothing
        hsm_g = np.ones(N) * 4 / np.sqrt(N) * U.kpc
        # ---------------------------------

        source = SPHSource(
            distance=3.0 * U.Mpc,
            rotation={"L_coords": (60.0 * U.deg, 0.0 * U.deg)},
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            h=0.7,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )

        datacube = DataCube(
            n_px_x=128,
            n_px_y=128,
            n_channels=32,
            px_size=10.0 * U.arcsec,
            channel_width=10.0 * U.km * U.s**-1,
            velocity_centre=source.vsys,
        )

        beam = GaussianBeam(
            bmaj=30.0 * U.arcsec, bmin=30.0 * U.arcsec, bpa=0.0 * U.deg, truncate=4.0
        )

        noise = GaussianNoise(rms=3.0e-5 * U.Jy * U.beam**-1)

        spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

        sph_kernel = CubicSplineKernel()

        M = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
        )

        M.insert_source_in_cube()
        M.add_noise()
        M.convolve_beam()
        M.write_beam_fits(beamfile, channels="velocity")
        M.write_fits(cubefile, channels="velocity")
        print(f"Wrote demo fits output to {cubefile}, and beam image to {beamfile}.")
        try:
            M.write_hdf5(hdf5file, channels="velocity")
        except ModuleNotFoundError:
            print("h5py package not present, skipping hdf5 output demo.")
        else:
            print(f"Wrote demo hdf5 output to {hdf5file}.")
    """

    def __init__(
        self,
        source=None,
        datacube=None,
        beam=None,
        noise=None,
        sph_kernel=None,
        spectral_model=None,
        quiet=False,
        find_distance=False,
    ):
        self.quiet = quiet
        self.find_distance = find_distance
        if source is not None:
            self.source = source
        else:
            raise ValueError("A source instance is required.")
        if datacube is not None:
            self.datacube = datacube
        else:
            raise ValueError("A datacube instance is required.")
        self.beam = beam
        self.noise = noise
        if sph_kernel is not None:
            self.sph_kernel = sph_kernel
        else:
            raise ValueError("An SPH kernel instance is required.")
        if spectral_model is not None:
            self.spectral_model = spectral_model
        else:
            raise ValueError("A spectral model instance is required.")

        if self.beam is not None:
            self.beam.init_kernel(self.datacube)
            self.datacube.add_pad(self.beam.needs_pad())

        self.source._init_skycoords()
        self.source._init_pixcoords(self.datacube)  # after datacube is padded

        self.sph_kernel._init_sm_lengths(source=self.source, datacube=self.datacube)
        self.sph_kernel._init_sm_ranges()
        if self.find_distance == False:
            self._prune_particles()  # prunes both source, and kernel if applicable
            self.spectral_model.init_spectra(self.source, self.datacube)
            self.inserted_mass = 0

        return

    def convolve_beam(self):
        """
        Convolve the beam and DataCube.
        """

        if self.beam is None:
            warn("Skipping beam convolution, no beam object provided to " "Martini.")
            return

        unit = self.datacube._array.unit
        for spatial_slice in self.datacube.spatial_slices():
            # use a view [...] to force in-place modification
            spatial_slice[...] = (
                fftconvolve(spatial_slice, self.beam.kernel, mode="same") * unit
            )
        self.datacube.drop_pad()
        self.datacube._array = self.datacube._array.to(
            U.Jy * U.beam**-1,
            equivalencies=U.beam_angular_area(self.beam.area),
        )
        if not self.quiet:
            print(
                "Beam convolved.",
                "  Data cube RMS after beam convolution:"
                f" {np.std(self.datacube._array):.2e}",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def add_noise(self):
        """
        Insert noise into the DataCube.
        """

        if self.noise is None:
            warn("Skipping noise, no noise object provided to Martini.")
            return

        # this unit conversion means noise can be added before or after source insertion:
        noise_cube = (
            self.noise.generate(self.datacube, self.beam)
            .to(
                U.Jy * U.arcsec**-2,
                equivalencies=U.beam_angular_area(self.beam.area),
            )
            .to(self.datacube._array.unit, equivalencies=[self.datacube.arcsec2_to_pix])
        )
        self.datacube._array = self.datacube._array + noise_cube
        if not self.quiet:
            print(
                "Noise added.",
                f"  Noise cube RMS: {np.std(noise_cube):.2e} (before beam convolution).",
                "  Data cube RMS after noise addition (before beam convolution): "
                f"{np.std(self.datacube._array):.2e}",
                sep="\n",
            )
        return

    def _prune_particles(self):
        """
        Determines which particles cannot contribute to the DataCube and
        removes them to speed up calculation. Assumes the kernel is 0 at
        distances greater than the kernel size (which may differ from the
        SPH smoothing length).
        """

        if not self.quiet:
            print(
                f"Source module contained {self.source.npart} particles with total HI"
                f" mass of {self.source.mHI_g.sum():.2e}."
            )
        spectrum_half_width = (
            self.spectral_model.half_width(self.source) / self.datacube.channel_width
        )
        reject_conditions = (
            (
                self.source.pixcoords[:2] + self.sph_kernel.sm_ranges[np.newaxis]
                < 0 * U.pix
            ).any(axis=0),
            self.source.pixcoords[0] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            self.source.pixcoords[1] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            self.source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            self.source.pixcoords[2] - 4 * spectrum_half_width * U.pix
            > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(self.source.pixcoords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        self.source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        self.sph_kernel._apply_mask(np.logical_not(reject_mask))
        if not self.quiet:
            print(
                f"Pruned particles that will not contribute to data cube, "
                f"{self.source.npart} particles remaining with total HI mass of "
                f"{self.source.mHI_g.sum():.2e}."
            )
        return
    
    def _compute_particles_num(self):
        new_source = self.source
        new_sph_kernel = self.sph_kernel
        initial_npart = self.source.npart
        spectrum_half_width = (
            self.spectral_model.half_width(new_source) / self.datacube.channel_width
        )
        reject_conditions = (
            (
                new_source.pixcoords[:2] + new_sph_kernel.sm_ranges[np.newaxis]
                < 0 * U.pix
            ).any(axis=0),
            new_source.pixcoords[0] - new_sph_kernel.sm_ranges
            > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            new_source.pixcoords[1] - new_sph_kernel.sm_ranges
            > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            new_source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            new_source.pixcoords[2] - 4 * spectrum_half_width * U.pix
            > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(new_source.pixcoords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        new_source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        new_sph_kernel._apply_mask(np.logical_not(reject_mask))
        final_npart = new_source.npart
        del new_source
        del new_sph_kernel
        return final_npart / initial_npart * 100

    def _evaluate_pixel_spectrum(self, ranks_and_ij_pxs, progressbar=True):
        """
        Add up contributions of particles to the spectrum in a pixel.

        This is the core loop of MARTINI. It is embarrassingly parallel. To support
        parallel excecution we accept storing up to a copy of the entire (future) datacube
        in one-pixel pieces. This avoids the need for concurrent access to the datacube
        by parallel processes, which would in the simplest case duplicate a copy of the
        datacube array per parallel process! In realistic use cases the memory overhead
        from a the equivalent of a second datacube array should be minimal - memory-
        limited applications should be limited by the memory consumed by particle data,
        which is not duplicated in parallel execution.

        The arguments that differ between parallel ranks must be bundled into one for
        compatibility with `multiprocess`.

        Parameters
        ----------
        rank_and_ij_pxs : tuple
            A 2-tuple containing an integer (cpu "rank" in the case of parallel execution)
            and a list of 2-tuples specifying the indices (i, j) of pixels in the grid.

        Returns
        -------
        out : list
            A list containing 2-tuples. Each 2-tuple contains and "insertion slice" that
            is an index into the datacube._array instance held by this martini instance
            where the pixel spectrum is to be placed, and a 1D array containing the
            spectrum, whose length must match the length of the spectral axis of the
            datacube.
        """
        result = list()
        rank, ij_pxs = ranks_and_ij_pxs
        if progressbar:
            ij_pxs = tqdm(ij_pxs, position=rank)
        for ij_px in ij_pxs:
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            mask = (
                np.abs(ij - self.source.pixcoords[:2]) <= self.sph_kernel.sm_ranges
            ).all(axis=0)
            weights = self.sph_kernel._px_weight(
                self.source.pixcoords[:2, mask] - ij, mask=mask
            )
            insertion_slice = (
                np.s_[ij_px[0], ij_px[1], :, 0]
                if self.datacube.stokes_axis
                else np.s_[ij_px[0], ij_px[1], :]
            )
            result.append(
                (
                    insertion_slice,
                    (self.spectral_model.spectra[mask] * weights[..., np.newaxis]).sum(
                        axis=-2
                    ),
                )
            )
        return result

    def _insert_pixel(self, insertion_slice, insertion_data):
        """
        Insert the spectrum for a single pixel into the datacube array.

        Parameters
        ----------
        insertion_slice : integer, tuple or slice
            Index into the datacube's _array specifying the insertion location.
        insertion data : array-like
            1D array containing the spectrum at the location specified by insertion_slice.
        """
        self.datacube._array[insertion_slice] = insertion_data
        return

    def insert_source_in_cube(self, skip_validation=False, progressbar=None, ncpu=1):
        """
        Populates the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the DataCube is approximated for
            increased speed. For some combinations of pixel size, distance
            and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            RuntimeError if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter True. (Default: False.)

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If martini was initialised with
            `quiet` set to `True`, progress bars are switched off unless explicitly
            turned on. (Default: None.)

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the `multiprocess` module (n.b. not the same as
            `multiprocessing`). (Default: 1)

        """

        assert self.spectral_model.spectra is not None

        if progressbar is None:
            progressbar = not self.quiet

        self.sph_kernel._confirm_validation(noraise=skip_validation, quiet=self.quiet)

        ij_pxs = list(
            product(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
            )
        )

        if ncpu == 1:
            for insertion_slice, insertion_data in self._evaluate_pixel_spectrum(
                (0, ij_pxs), progressbar=progressbar
            ):
                self._insert_pixel(insertion_slice, insertion_data)
        else:
            # not multiprocessing, need serialization from dill not pickle
            from multiprocess import Pool

            with Pool(processes=ncpu) as pool:
                for result in pool.imap_unordered(
                    lambda x: self._evaluate_pixel_spectrum(x, progressbar=progressbar),
                    [(icpu, ij_pxs[icpu::ncpu]) for icpu in range(ncpu)],
                ):
                    for insertion_slice, insertion_data in result:
                        self._insert_pixel(insertion_slice, insertion_data)

        self.datacube._array = self.datacube._array.to(
            U.Jy / U.arcsec**2, equivalencies=[self.datacube.arcsec2_to_pix]
        )
        pad_mask = (
            np.s_[
                self.datacube.padx : -self.datacube.padx,
                self.datacube.pady : -self.datacube.pady,
                ...,
            ]
            if self.datacube.padx > 0 and self.datacube.pady > 0
            else np.s_[...]
        )
        inserted_flux = (
            self.datacube._array[pad_mask].sum() * self.datacube.px_size**2
        )
        inserted_mass = (
            2.36e5
            * U.Msun
            * self.source.distance.to_value(U.Mpc) ** 2
            * inserted_flux.to_value(U.Jy)
            * self.datacube.channel_width.to_value(U.km / U.s)
        )
        self.inserted_mass = inserted_mass
        if not self.quiet:
            print(
                "Source inserted.",
                f"  Flux in cube: {inserted_flux:.2e}",
                f"  Mass in cube (assuming distance {self.source.distance:.2f}):"
                f" {inserted_mass:.2e}",
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def write_fits(
        self,
        filename,
        channels="frequency",
        overwrite=True,
    ):
        """
        Output the DataCube to a FITS-format file.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)
        """

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()
        wcs_header.rename_keyword("WCSAXES", "NAXIS")

        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", self.datacube.n_px_x))
        header.append(("NAXIS2", self.datacube.n_px_y))
        header.append(("NAXIS3", self.datacube.n_channels))
        if self.datacube.stokes_axis:
            header.append(("NAXIS4", 1))
        header.append(("EXTEND", "T"))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRPIX1", wcs_header["CRPIX1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRPIX2", wcs_header["CRPIX2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CDELT3", wcs_header["CDELT3"]))
        header.append(("CRPIX3", wcs_header["CRPIX3"]))
        header.append(("CRVAL3", wcs_header["CRVAL3"]))
        header.append(("CTYPE3", wcs_header["CTYPE3"]))
        header.append(("CUNIT3", wcs_header["CUNIT3"]))
        if self.datacube.stokes_axis:
            header.append(("CDELT4", wcs_header["CDELT4"]))
            header.append(("CRPIX4", wcs_header["CRPIX4"]))
            header.append(("CRVAL4", wcs_header["CRVAL4"]))
            header.append(("CTYPE4", wcs_header["CTYPE4"]))
            header.append(("CUNIT4", "PAR"))
        header.append(("EPOCH", 2000))
        header.append(("INSTRUME", "MARTINI", martini_version))
        # header.append(('BLANK', -32768)) #only for integer data
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        datacube_array_units = self.datacube._array.unit
        header.append(
            ("DATAMAX", np.max(self.datacube._array.to_value(datacube_array_units)))
        )
        header.append(
            ("DATAMIN", np.min(self.datacube._array.to_value(datacube_array_units)))
        )
        header.append(("ORIGIN", "astropy v" + astropy_version))
        # long names break fits format, don't let the user set this
        header.append(("OBJECT", "MOCK"))
        if self.beam is not None:
            header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("OBSERVER", "K. Oman"))
        # header.append(('NITERS', ???))
        # header.append(('RMS', ???))
        # header.append(('LWIDTH', ???))
        # header.append(('LSTEP', ???))
        header.append(("BUNIT", datacube_array_units.to_string("fits")))
        # header.append(('PCDEC', ???))
        # header.append(('LSTART', ???))
        header.append(("DATE-OBS", Time.now().to_value("fits")))
        # header.append(('LTYPE', ???))
        # header.append(('PCRA', ???))
        # header.append(('CELLSCAL', ???))
        if self.beam is not None:
            header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
            header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header, data=self.datacube._array.to_value(datacube_array_units).T
        )
        hdu.writeto(filename, overwrite=overwrite)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_beam_fits(self, filename, channels="frequency", overwrite=True):
        """
        Output the beam to a FITS-format file.

        The beam is written to file, with pixel sizes, coordinate system, etc.
        similar to those used for the DataCube.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        Raises
        ------
        ValueError
            If Martini was initialized without a beam.
        """

        if self.beam is None:
            raise ValueError(
                "Martini.write_beam_fits: Called with beam set " "to 'None'."
            )
        assert self.beam.kernel is not None
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_beam_fits: Unknown 'channels' "
                "value (use 'frequency' or 'velocity'."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()

        beam_kernel_units = self.beam.kernel.unit
        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        # header.append(('NAXIS', self.beam.kernel.ndim))
        header.append(("NAXIS", 3))
        header.append(("NAXIS1", self.beam.kernel.shape[0]))
        header.append(("NAXIS2", self.beam.kernel.shape[1]))
        header.append(("NAXIS3", 1))
        header.append(("EXTEND", "T"))
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        # this is 1/arcsec^2, is this right?
        header.append(("BUNIT", beam_kernel_units.to_string("fits")))
        header.append(("CRPIX1", self.beam.kernel.shape[0] // 2 + 1))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CRPIX2", self.beam.kernel.shape[1] // 2 + 1))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CRPIX3", 1))
        header.append(("CDELT3", wcs_header["CDELT3"]))
        header.append(("CRVAL3", wcs_header["CRVAL3"]))
        header.append(("CTYPE3", wcs_header["CTYPE3"]))
        header.append(("CUNIT3", wcs_header["CUNIT3"]))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))
        header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
        header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("BTYPE", "beam    "))
        header.append(("EPOCH", 2000))
        header.append(("OBSERVER", "K. Oman"))
        # long names break fits format
        header.append(("OBJECT", "MOCKBEAM"))
        header.append(("INSTRUME", "MARTINI", martini_version))
        header.append(("DATAMAX", np.max(self.beam.kernel.to_value(beam_kernel_units))))
        header.append(("DATAMIN", np.min(self.beam.kernel.to_value(beam_kernel_units))))
        header.append(("ORIGIN", "astropy v" + astropy_version))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header,
            data=self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis].T,
        )
        hdu.writeto(filename, overwrite=True)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_hdf5(
        self,
        filename,
        channels="frequency",
        overwrite=True,
        memmap=False,
        compact=False,
    ):
        """
        Output the DataCube and Beam to a HDF5-format file. Requires the h5py
        package.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.hdf5' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        memmap: bool, optional
            If True, create a file-like object in memory and return it instead
            of writing file to disk. (Default: False.)

        compact: bool, optional
            If True, omit pixel coordinate arrays to save disk space. In this
            case pixel coordinates can still be reconstructed from FITS-style
            keywords stored in the FluxCube attributes. (Default: False.)
        """

        import h5py

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            pass
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".hdf5" else filename + ".hdf5"

        wcs_header = self.datacube.wcs.to_header()

        mode = "w" if overwrite else "x"
        driver = "core" if memmap else None
        h5_kwargs = {"backing_store": False} if memmap else dict()
        f = h5py.File(filename, mode, driver=driver, **h5_kwargs)
        datacube_array_units = self.datacube._array.unit
        s = np.s_[..., 0] if self.datacube.stokes_axis else np.s_[...]
        f["FluxCube"] = self.datacube._array.to_value(datacube_array_units)[s]
        c = f["FluxCube"]
        origin = 0  # index from 0 like numpy, not from 1
        if not compact:
            xgrid, ygrid, vgrid = np.meshgrid(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
                np.arange(self.datacube._array.shape[2]),
            )
            cgrid = (
                np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                        np.zeros(vgrid.shape).flatten(),
                    )
                ).T
                if self.datacube.stokes_axis
                else np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                    )
                ).T
            )
            wgrid = self.datacube.wcs.all_pix2world(cgrid, origin)
            ragrid = wgrid[:, 0].reshape(self.datacube._array.shape)[s]
            decgrid = wgrid[:, 1].reshape(self.datacube._array.shape)[s]
            chgrid = wgrid[:, 2].reshape(self.datacube._array.shape)[s]
            f["RA"] = ragrid
            f["RA"].attrs["Unit"] = wcs_header["CUNIT1"]
            f["Dec"] = decgrid
            f["Dec"].attrs["Unit"] = wcs_header["CUNIT2"]
            f["channel_mids"] = chgrid
            f["channel_mids"].attrs["Unit"] = wcs_header["CUNIT3"]
        c.attrs["AxisOrder"] = "(RA,Dec,Channels)"
        c.attrs["FluxCubeUnit"] = str(self.datacube._array.unit)
        c.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
        c.attrs["RA0_in_px"] = wcs_header["CRPIX1"] - 1
        c.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
        c.attrs["RAUnit"] = wcs_header["CUNIT1"]
        c.attrs["RAProjType"] = wcs_header["CTYPE1"]
        c.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
        c.attrs["Dec0_in_px"] = wcs_header["CRPIX2"] - 1
        c.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
        c.attrs["DecUnit"] = wcs_header["CUNIT2"]
        c.attrs["DecProjType"] = wcs_header["CTYPE2"]
        c.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
        c.attrs["V0_in_px"] = wcs_header["CRPIX3"] - 1
        c.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
        c.attrs["VUnit"] = wcs_header["CUNIT3"]
        c.attrs["VProjType"] = wcs_header["CTYPE3"]
        if self.beam is not None:
            c.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            c.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            c.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
        c.attrs["DateCreated"] = str(Time.now())
        #c.attrs["MartiniVersion"] = martini_version
        #c.attrs["AstropyVersion"] = astropy_version
        if self.beam is not None:
            if self.beam.kernel is None:
                raise ValueError(
                    "Martini.write_hdf5: Called with beam present but beam kernel"
                    " uninitialized."
                )
            beam_kernel_units = self.beam.kernel.unit
            f["Beam"] = self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis]
            b = f["Beam"]
            b.attrs["BeamUnit"] = self.beam.kernel.unit.to_string("fits")
            b.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
            b.attrs["RA0_in_px"] = self.beam.kernel.shape[0] // 2
            b.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
            b.attrs["RAUnit"] = wcs_header["CUNIT1"]
            b.attrs["RAProjType"] = wcs_header["CTYPE1"]
            b.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
            b.attrs["Dec0_in_px"] = self.beam.kernel.shape[1] // 2
            b.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
            b.attrs["DecUnit"] = wcs_header["CUNIT2"]
            b.attrs["DecProjType"] = wcs_header["CTYPE2"]
            b.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
            b.attrs["V0_in_px"] = 0
            b.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
            b.attrs["VUnit"] = wcs_header["CUNIT3"]
            b.attrs["VProjType"] = wcs_header["CTYPE3"]
            b.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            b.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            b.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
            b.attrs["DateCreated"] = str(Time.now())
            #b.attrs["MartiniVersion"] = martini_version
            #b.attrs["AstropyVersion"] = astropy_version

        if channels == "frequency":
            self.datacube.velocity_channels()
        if memmap:
            return f
        else:
            f.close()
            return

    def reset(self):
        """
        Re-initializes the DataCube with zero-values.
        """
        init_kwargs = dict(
            n_px_x=self.datacube.n_px_x,
            n_px_y=self.datacube.n_px_y,
            n_channels=self.datacube.n_channels,
            px_size=self.datacube.px_size,
            channel_width=self.datacube.channel_width,
            velocity_centre=self.datacube.velocity_centre,
            ra=self.datacube.ra,
            dec=self.datacube.dec,
            stokes_axis=self.datacube.stokes_axis,
        )
        self.datacube = DataCube(**init_kwargs)
        if self.beam is not None:
            self.datacube.add_pad(self.beam.needs_pad())
        return

def gaussian(x, amp, cen, fwhm):
    """
    Generates a 1D Gaussian given the following input parameters:
    x: position
    amp: amplitude
    fwhm: fwhm
    """
    return amp*np.exp(-(x-cen)**2/(2*(fwhm/2.35482)**2))

def threedgaussian(amplitude, spind, chan, center_x, center_y, width_x, width_y, angle, idxs):
    """
    Generates a 3D Gaussian given the following input parameters:
    amplitude: amplitude
    spind: spectral index
    chan: channel
    center_x: x position
    center_y: y position
    width_x: width in x
    width_y: width in y
    angle: angle of rotation
    idxs: indices of the datacube

    """
    angle = math.pi/180. * angle
    rcen_x = center_x * np.cos(angle) - center_y * np.sin(angle)
    rcen_y = center_x * np.sin(angle) + center_y * np.cos(angle)
    xp = idxs[0] * np.cos(angle) - idxs[1] * np.sin(angle)
    yp = idxs[0] * np.sin(angle) + idxs[1] * np.cos(angle)
    v1 = 230e9 - (64 * 10e6)
    v2 = v1+10e6*chan
    g = (10**(np.log10(amplitude) + (spind) * np.log10(v1/v2))) * \
        np.exp(-(((rcen_x-xp)/width_x)**2+((rcen_y-yp)/width_y)**2)/2.)
    return g

def insert_pointlike(datacube, amplitude, pos_x, pos_y, pos_z, fwhm_z, n_px, n_chan):
    """
    Inserts a point source into the datacube at the specified position and amplitude.
    datacube: datacube object
    amplitude: amplitude of the point source
    pos_x: x position
    pos_y: y position
    pos_z: z position
    fwhm_z: fwhm in z
    n_px: number of pixels in the cube
    n_chan: number of channels in the cube
    """
    z_idxs = np.arange(0, n_chan)
    g = gaussian(z_idxs, 1, pos_z, fwhm_z)
    ts = np.zeros((n_px, n_px, n_chan))
    ts[int(pos_x), int(pos_y), :] = amplitude
    for z in range(datacube._array.shape[2]):
        slice_ = g[z] * ts[:, :, z]
        datacube._array[:, :, z] += slice_ * U.Jy * U.pix**-2
    return datacube

def insert_gaussian(datacube, amplitude, pos_x, pos_y, pos_z, fwhm_x, fwhm_y, fwhm_z, angle, n_px, n_chan):
    """
    Inserts a 3D Gaussian into the datacube at the specified position and amplitude.
    datacube: datacube object
    amplitude: amplitude of the source
    pos_x: x position
    pos_y: y position
    pos_z: z position
    fwhm_x: fwhm in x
    fwhm_y: fwhm in y
    fwhm_z: fwhm in z
    angle: angle of rotation
    n_px: number of pixels in the cube
    n_chan: number of channels in the cube
    """
    idxs = np.indices((n_px, n_px))
    z_idxs = np.arange(0, n_chan)
    g = gaussian(z_idxs, 1, pos_z, fwhm_z)
    ts = np.zeros((n_px, n_px, n_chan))
    for z in range(datacube._array.shape[2]):
        slice_ = g[z] * threedgaussian(amplitude, 0, z, pos_x, pos_y, fwhm_x, fwhm_y, angle, idxs)
        datacube._array[:, :, z] += slice_ * U.Jy * U.pix**-2
    return datacube

def insert_extended(snap_number, subhalo_id, n_px, n_channels, spatial_resolution, 
                    frequency_resolution, ra, dec, x_rot, y_rot, tngpath, distance, ncpu):
     

    return 