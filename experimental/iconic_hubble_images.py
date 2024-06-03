#! /bin/env python3
import os
import requests
import zipfile

def download_hubble_images(hubble_image_path):
    """Download 10GB of iconic Hubble images to hubble_image_path/top100.
       These are large in size which allows random cropping and scaling for data-augmentation. 
    """
    baseurl = 'http://www.spacetelescope.org/static/images/zip/top100/top100-original.zip'
    
    if not os.path.exists(hubble_image_path):
        os.makedirs(hubble_image_path)
        print(f'Hubble images not found on disk, downloading 10GB to {hubble_image_path} ...')
   
        zipfilename = os.path.basename(baseurl)
        response = requests.Session().get(baseurl, stream=True)
        with open(zipfilename, 'wb') as f:     
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(zipfilename) as zf:
            zf.extractall(hubble_image_path)
        os.remove(zipfilename)    
    else:
        print(f'Hubble images already exist on disk.')
   
       
       
if __name__ == '__main__':
    hubble_image_path = './hubble_images'
    download_hubble_images(hubble_image_path)
        

   
   
   
