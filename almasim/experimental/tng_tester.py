import h5py
import subprocess

api_key = "8f578b92e700fae3266931f4d785f82c"
url = "http://www.tng-project.org/api/TNG100-1/files/snapshot-99.0.hdf5"
cmd = (
    f"wget --progress=bar --content-disposition "
    f'--header="API-Key:{api_key}" '
    f"{url} "
    f"-O snapshot-99.0.hdf5"
)
print(cmd)
subprocess.run(cmd, shell=True)
