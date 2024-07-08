import h5py
from random import choices
import os
import numpy as np
import pandas as pd
from os.path import isfile, expanduser
import subprocess
import six


# -------------------- TNG Auxiliary Functions ----------------------------- #


def compute_redshift(rest_frequency, observed_frequency):
    """
    Computes the redshift of a source given the rest frequency and the observed frequency.

    Args:
        rest_frequency (astropy Unit): Rest frequency of the source in GHz.
        observed_frequency (astropy Unit): Observed frequency of the source in GHz.

    Returns:
        float: Redshift of the source.

    Raises:
        ValueError: If either input argument is non-positive.
    """
    # Input validation
    if rest_frequency <= 0 or observed_frequency <= 0:
        raise ValueError("Rest and observed frequencies must be positive values.")
    if rest_frequency < observed_frequency:
        raise ValueError("Observed frequency must be lower than the rest frequency.")

    # Compute redshift
    redshift = (rest_frequency.value - observed_frequency.value) / rest_frequency.value
    return redshift


def redshift_to_snapshot(redshift):
    snap_db = {
        0: 20.05,
        1: 14.99,
        2: 11.98,
        3: 10.98,
        4: 10.00,
        5: 9.390,
        6: 9.000,
        7: 8.450,
        8: 8.010,
        9: 7.600,
        10: 7.24,
        11: 7.01,
        12: 6.49,
        13: 6.01,
        14: 5.85,
        15: 5.53,
        16: 5.23,
        17: 5.00,
        18: 4.66,
        19: 4.43,
        20: 4.18,
        21: 4.01,
        22: 3.71,
        23: 3.49,
        24: 3.28,
        25: 3.01,
        26: 2.90,
        27: 2.73,
        28: 2.58,
        29: 2.44,
        30: 2.32,
        31: 2.21,
        32: 2.10,
        33: 2.00,
        34: 1.90,
        35: 1.82,
        36: 1.74,
        37: 1.67,
        38: 1.60,
        39: 1.53,
        40: 1.50,
        41: 1.41,
        42: 1.36,
        43: 1.30,
        44: 1.25,
        45: 1.21,
        46: 1.15,
        47: 1.11,
        48: 1.07,
        49: 1.04,
        50: 1.00,
        51: 0.95,
        52: 0.92,
        53: 0.89,
        54: 0.85,
        55: 0.82,
        56: 0.79,
        57: 0.76,
        58: 0.73,
        59: 0.70,
        60: 0.68,
        61: 0.64,
        62: 0.62,
        63: 0.60,
        64: 0.58,
        65: 0.55,
        66: 0.52,
        67: 0.50,
        68: 0.48,
        69: 0.46,
        70: 0.44,
        71: 0.42,
        72: 0.40,
        73: 0.38,
        74: 0.36,
        75: 0.35,
        76: 0.33,
        77: 0.31,
        78: 0.30,
        79: 0.27,
        80: 0.26,
        81: 0.24,
        82: 0.23,
        83: 0.21,
        84: 0.20,
        85: 0.18,
        86: 0.17,
        87: 0.15,
        88: 0.14,
        89: 0.13,
        90: 0.11,
        91: 0.10,
        92: 0.08,
        93: 0.07,
        94: 0.06,
        95: 0.05,
        96: 0.03,
        97: 0.02,
        98: 0.01,
        99: 0,
    }
    snaps, redshifts = list(snap_db.keys())[::-1], list(snap_db.values())[::-1]
    for i in range(len(redshifts) - 1):
        if redshift >= redshifts[i] and redshift < redshifts[i + 1]:
            return snaps[i]


def get_data_from_hdf(file, snapshot):
    data = list()
    column_names = list()
    r = h5py.File(file, "r")
    for key in r.keys():
        if key == f"Snapshot_{snapshot}":
            group = r[key]
            for key2 in group.keys():
                column_names.append(key2)
                data.append(group[key2])
    values = np.array(data)
    r.close()
    db = pd.DataFrame(values.T, columns=column_names)
    return db


def get_subhaloids_from_db(n, main_path, snapshot):
    pd.options.mode.chained_assignment = None
    file = os.path.join(main_path, "metadata", "morphologies_deeplearn.hdf5")
    db = get_data_from_hdf(file, snapshot)
    catalogue = db[["SubhaloID", "P_Late", "P_S0", "P_Sab"]]
    catalogue = catalogue.sort_values(by=["P_Late"], ascending=False)
    ellipticals = catalogue[
        (catalogue["P_Late"] > 0.6)
        & (catalogue["P_S0"] < 0.5)
        & (catalogue["P_Sab"] < 0.5)
    ]
    lenticulars = catalogue[
        (catalogue["P_S0"] > 0.6)
        & (catalogue["P_Late"] < 0.5)
        & (catalogue["P_Sab"] < 0.5)
    ]
    spirals = catalogue[
        (catalogue["P_Sab"] > 0.6)
        & (catalogue["P_Late"] < 0.5)
        & (catalogue["P_S0"] < 0.5)
    ]

    ellipticals["sum"] = ellipticals["P_S0"] + ellipticals["P_Sab"]
    lenticulars["sum"] = lenticulars["P_Late"] + lenticulars["P_Sab"]
    spirals["sum"] = spirals["P_Late"] + spirals["P_S0"]

    ellipticals = ellipticals.sort_values(by=["sum"], ascending=True)
    lenticulars = lenticulars.sort_values(by=["sum"], ascending=True)
    spirals = spirals.sort_values(by=["sum"], ascending=True)
    ellipticals_ids = ellipticals["SubhaloID"].values
    lenticulars_ids = lenticulars["SubhaloID"].values
    spirals_ids = spirals["SubhaloID"].values
    sample_n = n // 3

    n_0 = choices(ellipticals_ids, k=sample_n)
    n_1 = choices(spirals_ids, k=sample_n)
    n_2 = choices(lenticulars_ids, k=n - 2 * sample_n)
    ids = np.concatenate((n_0, n_1, n_2)).astype(int)
    if len(ids) == 1:
        return ids[0]
    return ids


def partTypeNum(partType):
    """Mapping between common names and numeric particle types."""
    if str(partType).isdigit():
        return int(partType)

    if str(partType).lower() in ["gas", "cells"]:
        return 0
    if str(partType).lower() in ["dm", "darkmatter"]:
        return 1
    if str(partType).lower() in ["dmlowres"]:
        return 2  # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ["tracer", "tracers", "tracermc", "trmc"]:
        return 3
    if str(partType).lower() in ["star", "stars", "stellar"]:
        return 4  # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ["wind"]:
        return 4  # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ["bh", "bhs", "blackhole", "blackholes"]:
        return 5

    raise Exception("Unknown particle type name.")


def gcPath(basePath, snapNum, chunkNum=0):
    """Return absolute path to a group catalog HDF5 file (modify as needed)."""
    gcPath = basePath + "/groups_%03d/" % snapNum
    filePath1 = gcPath + "groups_%03d.%d.hdf5" % (snapNum, chunkNum)
    filePath2 = gcPath + "fof_subhalo_tab_%03d.%d.hdf5" % (snapNum, chunkNum)

    if isfile(expanduser(filePath1)):
        return filePath1
    return filePath2


def offsetPath(basePath, snapNum):
    """Return absolute path to a separate offset file (modify as needed)."""
    offsetPath = basePath + "/../postprocessing/offsets/offsets_%03d.hdf5" % snapNum

    return offsetPath


def loadObjects(basePath, snapNum, gName, nName, fields):
    """Load either halo or subhalo information from the group catalog."""
    result = {}

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(gcPath(basePath, snapNum), "r") as f:

        header = dict(f["Header"].attrs.items())

        if "N" + nName + "_Total" not in header and nName == "subgroups":
            nName = "subhalos"  # alternate convention

        result["count"] = f["Header"].attrs["N" + nName + "_Total"]

        if not result["count"]:
            print("warning: zero groups, empty return (snap=" + str(snapNum) + ").")
            return result

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for field in fields:
            # verify existence
            if field not in f[gName].keys():
                raise Exception(
                    "Group catalog does not have requested field [" + field + "]!"
                )

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = result["count"]

            # allocate within return dict
            result[field] = np.zeros(shape, dtype=f[gName][field].dtype)

    # loop over chunks
    wOffset = 0

    for i in range(header["NumFiles"]):
        f = h5py.File(gcPath(basePath, snapNum, i), "r")

        if not f["Header"].attrs["N" + nName + "_ThisFile"]:
            continue  # empty file chunk

        # loop over each requested field
        for field in fields:
            if field not in f[gName].keys():
                raise Exception(
                    "Group catalog does not have requested field [" + field + "]!"
                )

            # shape and type
            shape = f[gName][field].shape

            # read data local to the current file
            if len(shape) == 1:
                result[field][wOffset : wOffset + shape[0]] = f[gName][field][
                    0 : shape[0]
                ]
            else:
                result[field][wOffset : wOffset + shape[0], :] = f[gName][field][
                    0 : shape[0], :
                ]

        wOffset += shape[0]
        f.close()

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result


def loadSubhalos(basePath, snapNum, fields=None):
    """Load all subhalo information from the entire group catalog for one snapshot
    (optionally restrict to a subset given by fields)."""

    return loadObjects(basePath, snapNum, "Subhalo", "subgroups", fields)


def loadHalos(basePath, snapNum, fields=None):
    """Load all halo information from the entire group catalog for one snapshot
    (optionally restrict to a subset given by fields)."""

    return loadObjects(basePath, snapNum, "Group", "groups", fields)


def loadHeader(basePath, snapNum):
    """Load the group catalog header."""
    with h5py.File(gcPath(basePath, snapNum), "r") as f:
        header = dict(f["Header"].attrs.items())

    return header


def load(basePath, snapNum):
    """Load complete group catalog all at once."""
    r = {}
    r["subhalos"] = loadSubhalos(basePath, snapNum)
    r["halos"] = loadHalos(basePath, snapNum)
    r["header"] = loadHeader(basePath, snapNum)
    return r


def loadSingle(basePath, snapNum, haloID=-1, subhaloID=-1):
    """Return complete group catalog information for one halo or subhalo."""
    if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID >= 0 else "Group"
    searchID = subhaloID if subhaloID >= 0 else haloID

    # old or new format
    if "fof_subhalo" in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), "r") as f:
            offsets = f["FileOffsets/" + gName][()]
    else:
        # use header of group catalog
        with h5py.File(gcPath(basePath, snapNum), "r") as f:
            offsets = f["Header"].attrs["FileOffsets_" + gName]

    offsets = searchID - offsets
    fileNum = np.max(np.where(offsets >= 0))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
        for haloProp in f[gName].keys():
            result[haloProp] = f[gName][haloProp][groupOffset]

    return result


def snapPath(basePath, snapNum, chunkNum=0):
    """Return absolute path to a snapshot HDF5 file (modify as needed)."""
    snapPath = basePath + "/snapdir_" + str(snapNum).zfill(3) + "/"
    filePath1 = (
        snapPath + "snap_" + str(snapNum).zfill(3) + "." + str(chunkNum) + ".hdf5"
    )
    filePath2 = filePath1.replace("/snap_", "/snapshot_")

    if isfile(filePath1):
        return filePath1
    return filePath2


def snapPath2(basePath, snapNum, chunkNum=0):
    snapPath = basePath + "/snapdir_" + str(snapNum).zfill(3) + "/"
    filePath1 = (
        snapPath + "snap_" + str(snapNum).zfill(3) + "." + str(chunkNum) + ".hdf5"
    )
    return filePath1


def getNumPart(header):
    """Calculate number of particles of all types given a snapshot header."""
    if "NumPart_Total_HighWord" not in header:
        return header["NumPart_Total"]  # new uint64 convention

    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header["NumPart_Total"][j] | (
            header["NumPart_Total_HighWord"][j] << 32
        )

    return nPart


def loadSubset(
    basePath,
    snapNum,
    partType,
    fields=None,
    subset=None,
    mdi=None,
    sq=True,
    float32=False,
    outPath=None,
    api_key=None,
):
    """Load a subset of fields for all particles/cells of a given partType.
    If offset and length specified, load only that subset of the partType.
    If mdi is specified, must be a list of integers of the same length as fields,
    giving for each field the multi-dimensional index (on the second dimension) to load.
      For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
      of y-Coordinates only, together with Masses.
    If sq is True, return a numpy array instead of a dict if len(fields)==1.
    If float32 is True, load any float64 datatype arrays directly as float32.
    """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    if not os.path.exists(os.path.join(basePath, "snapdir_0{}".format(str(snapNum)))):
        os.makedirs(os.path.join(basePath, "snapdir_0{}".format(str(snapNum))))
    if not isfile(snapPath(basePath, snapNum)):
        print("Downloading Snapshot {}...".format(snapNum))
        url = f"http://www.tng-project.org/api/TNG100-1/files/snapshot-{str(snapNum)}"
        wget_options = "-q --progress=bar --content-disposition"
        api_key_header = f'--header="API-Key:{api_key}"'
        filename = f"{url}.{0}.hdf5"
        output_file = f"-O {snapPath2(basePath, snapNum)}"
        cmd = f"{wget_options} {api_key_header} {filename} {output_file}"
        subprocess.check_call(cmd, shell=True)
    with h5py.File(snapPath(basePath, snapNum), "r") as f:

        header = dict(f["Header"].attrs.items())
        nPart = getNumPart(header)
        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = (
                subset["offsetType"][ptNum] - subset["snapOffsets"][ptNum, :]
            )

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset["lenType"][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result["count"] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            if os.path.isfile(snapPath(basePath, snapNum, i)):
                print("Found")
                f = h5py.File(snapPath(basePath, snapNum, i), "r")
            else:
                print("Not Found")
                beg_url = "http://www.tng-project.org/api/TNG100-1/files/snapshot"
                url = f"{beg_url}-{str(snapNum)}"
                # subdir = os.path.join("output", "snapdir_0{}".format(str(i)))
                cmd = (
                    f"wget -q --progress=bar --content-disposition "
                    f'--header="API-Key:{api_key}" '  # Separate f-string for clarity
                    f"{url}.{i}.hdf5"
                )
                print(f"Downloading snapshot {i} ...")
                if outPath is not None:
                    os.chdir(outPath)
                subprocess.check_call(cmd, shell=True)
                print("Done.")
                f = h5py.File(snapPath(basePath, snapNum, i), "r")
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception(
                    "Particle type ["
                    + str(ptNum)
                    + "] does not have field ["
                    + field
                    + "]"
                )

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception(
                        "Read error: mdi requested on non-2D field [" + field + "]"
                    )
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32:
                dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        if not os.path.isfile(snapPath(basePath, snapNum, fileNum)):
            print(
                f"Particles are found in Snapshot {fileNum} which is not present on disk"
            )
            # move directory to the correct directory data !!!
            url = (
                f"http://www.tng-project.org/api/TNG100-1/files/snapshot-{str(snapNum)}"
            )
            # subdir = os.path.join("output", "snapdir_0{}".format(str(fileNum)))
            savePath = os.path.join(basePath, "snapdir_0{}".format(str(snapNum)))
            cmd = (
                f"wget -P {savePath} -q --progress=bar --content-disposition "
                f'--header="API-Key:{api_key}" '
                f"{url}.{fileNum}.hdf5"
            )
            if outPath is not None:
                os.chdir(outPath)
            print(f"Downloading Snapshot {fileNum} in {savePath}...")
            subprocess.check_call(cmd, shell=True)
            print("Done.")
        print("Checking File {}...".format(fileNum))
        f = h5py.File(snapPath(basePath, snapNum, fileNum), "r")

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f["Header"].attrs["NumPart_ThisFile"][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset : wOffset + numToReadLocal] = f[gName][field][
                    fileOff : fileOff + numToReadLocal
                ]
            else:
                result[field][wOffset : wOffset + numToReadLocal] = f[gName][field][
                    fileOff : fileOff + numToReadLocal, mdi[i]
                ]

        wOffset += numToReadLocal
        numToRead -= numToReadLocal
        fileNum += 1
        fileOff = 0  # start at beginning of all file chunks other than the first
        print("Loading File {}...".format(fileNum))
        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception(
            "Read ["
            + str(wOffset)
            + "] particles, but was expecting ["
            + str(origNumToRead)
            + "]"
        )

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result


def download_groupcat(basePath, snapNum, fileNum, api_key):
    print("Group Catalogue not found, downloading it")
    url = "http://www.tng-project.org/api/TNG100-1/files/groupcat-{}.{}.hdf5".format(
        snapNum, fileNum
    )
    cmd = (
        f"wget -nd -nc -nv -e robots=off -l 1 -A hdf5 "
        f'--content-disposition --header="API-Key:{api_key}" '  # Split here
        f"{url} "
        f"-O {gcPath(basePath, snapNum, fileNum)}"
    )
    subprocess.check_call(cmd, shell=True)
    print("Done.")


def getSnapOffsets(basePath, snapNum, id, type, api_key):
    """Compute offsets within snapshot for a particular group/subgroup."""
    r = {}
    print("Checking offset")
    # old or new format
    if "fof_subhalo" in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        if not isfile(offsetPath(basePath, snapNum)):
            print("Downloading offset file ")
            url = (
                "https://www.tng-project.org/api/TNG100-1/files/offsets.{}.hdf5".format(
                    snapNum
                )
            )
            cmd = (
                f"wget -q --progress=bar  --content-disposition "
                f'--header="API-Key:{api_key}" '
                f"{url} "
                f"-O {offsetPath(basePath, snapNum)}"
            )
            subprocess.check_call(cmd, shell=True)
            print("Done.")
        if not isfile(gcPath(basePath, snapNum, 0)):
            download_groupcat(basePath, snapNum, 0, api_key)
        with h5py.File(offsetPath(basePath, snapNum), "r") as f:
            groupFileOffsets = f["FileOffsets/" + type][()]
            r["snapOffsets"] = np.transpose(
                f["FileOffsets/SnapByType"][()]
            )  # consistency
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), "r") as f:
            groupFileOffsets = f["Header"].attrs["FileOffsets_" + type]
            r["snapOffsets"] = f["Header"].attrs["FileOffsets_Snap"]

    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog
    group_path = basePath + "/groups_%03d/" % snapNum
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    if not isfile(gcPath(basePath, snapNum, 0)):
        download_groupcat(basePath, snapNum, 0, api_key)
    if isfile(gcPath(basePath, snapNum, fileNum)):
        with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
            r["lenType"] = f[type][type + "LenType"][groupOffset, :]
    else:
        download_groupcat(basePath, snapNum, fileNum, api_key)
        with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
            r["lenType"] = f[type][type + "LenType"][groupOffset, :]
    # old or new format: load the offset (by type) of  this group/subgroup
    # within the snapshot
    if "fof_subhalo" in gcPath(basePath, snapNum):
        with h5py.File(offsetPath(basePath, snapNum), "r") as f:
            r["offsetType"] = f[type + "/SnapByType"][id, :]

            # add TNG-Cluster specific offsets if present
            if "OriginalZooms" in f:
                for key in f["OriginalZooms"]:
                    r[key] = f["OriginalZooms"][key][()]
    else:
        with h5py.File(gcPath(basePath, snapNum, fileNum), "r") as f:
            r["offsetType"] = f["Offsets"][type + "_SnapByType"][groupOffset, :]

    return r


def get_particles_num(basePath, outputPath, snapNum, subhaloID, tng_api_key):
    basePath = os.path.join(
        basePath,
        "TNG100-1",
        "output",
    )
    print("Looking for Subhalo %d in snapshot %d" % (subhaloID, snapNum))
    partType = "gas"
    # subset = getSnapOffsets(basePath, snapNum, subhaloID, "Subhalo", tng_api_key)
    # subhalo = loadSubset(
    #    basePath, snapNum, partType, subset=subset, api_key=tng_api_key
    # )
    os.chdir(basePath)
    gas = loadSubhalo(basePath, snapNum, subhaloID, partType)
    if "Coordinates" in gas.keys():
        gas_num = len(gas["Coordinates"])
    else:
        gas_num = 0
    return gas_num


def loadSubhalo(basePath, snapNum, id, partType, tng_api_key, fields=None):
    """Load all particles/cells of one type for a specific subhalo
    (optionally restricted to a subset fields)."""
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Subhalo")
    return loadSubset(
        basePath, snapNum, partType, fields, subset=subset, api_key=tng_api_key
    )


def loadHalo(basePath, snapNum, id, partType, fields=None, api_key=None):
    """Load all particles/cells of one type for a specific halo
    (optionally restricted to a subset fields)."""
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Group")
    return loadSubset(
        basePath, snapNum, partType, fields, subset=subset, api_key=api_key
    )


# ---------------- Luminosity Functions ---------------------------------- #


def read_line_emission_csv(path_line_emission_csv, sep=";"):
    """
    Read the csv file in which are stored the line emission's rest frequency.

    Parameter:
    path_line_emission_csv (str): Path to file.csv within there are the line
                                  emission's rest frequency.

    Return:
    pd.DataFrame : Dataframe with line names and rest frequencies.
    """
    db_line = pd.read_csv(path_line_emission_csv, sep=sep)
    return db_line


def get_line_info(main_path, idxs=None):
    path_line_emission_csv = os.path.join(
        main_path, "brightnes", "calibrated_lines.csv"
    )
    db_line = read_line_emission_csv(path_line_emission_csv, sep=",").sort_values(
        by="Line"
    )
    rest_frequencies = db_line["freq(GHz)"].values
    line_names = db_line["Line"].values
    if idxs is not None:
        return rest_frequencies[idxs], line_names[idxs]
    else:
        return rest_frequencies, line_names


def compute_rest_frequency_from_redshift(master_path, source_freq, redshift):
    db_line = read_line_emission_csv(
        os.path.join(master_path, "brightnes", "calibrated_lines.csv"), sep=","
    )
    db_line["freq(GHz)"] = db_line["freq(GHz)"].astype(float)
    source_freqs = db_line["freq(GHz)"].values / (1 + redshift)
    freq_names = db_line["Line"].values
    closest_freq = min(source_freqs, key=lambda x: abs(x - source_freq))
    line_name = freq_names[np.where(source_freqs == closest_freq)][0]
    rest_frequency = db_line[db_line["Line"] == line_name]["freq(GHz)"].values[0]
    return rest_frequency


def write_sim_parameters(
    path,
    ra,
    dec,
    ang_res,
    vel_res,
    int_time,
    band,
    band_range,
    central_freq,
    redshift,
    line_fluxes,
    line_names,
    line_frequencies,
    continum,
    fov,
    beam_size,
    cell_size,
    n_pix,
    n_channels,
    snapshot,
    subhalo,
    lum_infrared,
    fwhm_z,
    source_type,
    fwhm_x=None,
    fwhm_y=None,
    angle=None,
):
    with open(path, "w") as f:
        f.write("Simulation Parameters:\n")
        f.write("RA: {}\n".format(ra))
        f.write("DEC: {}\n".format(dec))
        f.write("Band: {}\n".format(band))
        f.write("Bandwidth {}\n".format(band_range))
        f.write("Band Central Frequency: {}\n".format(central_freq))
        f.write("Pixel size: {}\n".format(cell_size))
        f.write("Beam Size: {}\n".format(beam_size))
        f.write("Fov: {}\n".format(fov))
        f.write("Angular Resolution: {}\n".format(ang_res))
        f.write("Velocity Resolution: {}\n".format(vel_res))
        f.write("Redshift: {}\n".format(redshift))
        f.write("Integration Time: {}\n".format(int_time))
        f.write("Cube Size: {} x {} x {} pixels\n".format(n_pix, n_pix, n_channels))
        f.write("Mean Continum Flux: {}\n".format(np.mean(continum)))
        f.write("Infrared Luminosity: {}\n".format(lum_infrared))
        if source_type == "gaussian":
            f.write("FWHM_x (pixels): {}\n".format(fwhm_x))
            f.write("FWHM_y (pixels): {}\n".format(fwhm_y))
        if (source_type == "gaussian") or (source_type == "extended"):
            f.write("Projection Angle: {}\n".format(angle))
        for i in range(len(line_fluxes)):
            f.write(
                f"Line: {line_names[i]} - Frequency: {line_frequencies[i]} GHz "
                f"- Flux: {line_fluxes[i]} Jy  - Width (Channels): {fwhm_z[i]}\n"
            )
        if snapshot is not None:
            f.write("TNG Snapshot ID: {}\n".format(snapshot))
            f.write("TNG Subhalo ID: {}\n".format(subhalo))
        f.close()


# def get_image_from_ssd(ra, dec, fov):
#    DEF_ACCESS_URL = "https://datalab.noirlab.edu/sia/sdss_dr9"
#    svc_sdss_dr9 = sia.SIAService(DEF_ACCESS_URL)
#    ac.whoAmI()
#    imgTable = svc_sdss_dr9.search(
#        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
#    ).to_table()
