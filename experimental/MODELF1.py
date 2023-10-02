"""
PROGRAM TO INSERT GAUSSIAN MODELS INTO A FIELD:

SUGGEST JUST RUNNING THIS VERSION TO SEE IF OKAY.

THEN, CHANGE INPUT GAUSIAN MODEL IN STEP 2

1.  doinput: DEFAULT STEPS AND OUTPUT DIRECTORY, PP:
      NOW SET TO GO TO CLEAN IMAGE TO CHECK MODELS
2.  FILL IN MODEL PARAMETERS
3.  FILL IN OTHER PARAMETERS

    MODELF1.py  is the script
    Need one configuration file:
       alma.cycle8.10.cfg
       alma.cycle8.11.cfg
    OUTPUT IN DIRECTORY PP:
       ms with and without noise
       image_stat.txt
       PP/Simage*    with clean images     
"""

import os
import math
import scipy
import sys
import random
import commands
import numpy as np
import datetime
import scipy as sp
from scipy import optimize
import cmath
import string
#
#  1.  CONTROL OF STEPS TO RUN AND OUTPUT DIRECTORY
doinput = ['sim','plotms','image','display']
#doinput = ['sim']                      # just make ms of model data
PP = 'M1'                               # output directory
#
# subroutine to convert ra,dec into HMS,DMS format
def posconv(pos0):                                                
    pra = pos0[0]                                                 
    pdec = pos0[1]                                                
    Hpra = np.int(pra)                                            
    Mpra = np.int((pra - Hpra) * 60.0)                            
    Spra = (pra - Hpra - Mpra / 60.0) * 3600.0
    pdsign = '+'
    if pdec < 0:
        pdsign = '-'
        pdec = -pdec
    Hpdec = np.int(pdec)                                            
    Mpdec = np.int((pdec - Hpdec) * 60.0)                            
    Spdec = (pdec - Hpdec - Mpdec / 60.0) * 3600.0                   
    prast = 'J2000 '+str(Hpra)+'h'+str(Mpra)+'m'+str(Spra)+'s '+pdsign+str(Hpdec)+'d'+str(Mpdec)+'m'+str(Spdec)+'s'
    return prast                                                  
#
# 2. FILL MODEL PARAMETERS 
#
# Choose phase center: (ra(hr), dec(deg)
pos0 = [10.5083333,-8.50833333]
#  Put in input gaussian model
#       flux    bmaj    bmin     bpa    xoff    yoff
Cmp = [[0.040,  0.090,  0.010,  -55.0,  0.00,  -0.02],
       [0.030,  0.001,  0.001,    0.0,  0.40,  -0.40],
       [0.100,  0.150,  0.050,  -50.0, -0.40,  -0.20],
       [0.100,  0.250,  0.150,  -45.0, -0.70,   0.50],
       [0.030,  0.080,  0.001,  -47.0,  0.37,  -0.35],
       [0.100,  0.070,  0.070,    0.0,  0.50,   1.00],
       [0.050,  0.110,  0.020,   45,0, -0.50,  -0.50],
       [0.300,  1.000,  0.200,  -45.0,  0.00,   0.00]]
ngauss = len(Cmp)
#
#  3. FILL IN OTHER PARAMETERS 
# Directory for output
dfreq = '97.0GHz'     #frequency
zimsize = 1024        #image size
fcompwidth = '1GHz'   #total bandwidth
fintegr = '60s'       #sample time
fmapsize = ['20arcsec','20arcsec']   #mapsize
ftotal = '600s'                      #total integration time
fconf = 'alma.cycle8.11.cfg'    #ALMA array conf 11
#fconf = 'alma.cycle8.10.cfg'    #ALMA array conf 10
####################
#
#  print out positions
dcos = np.cos(pos0[1]*pi/180.0)  
dir0 = posconv(pos0)
#                                             
print 'phase center ',pos0,dir0
print 'Gaussian components'
print 'C#  Flux   Bmaj    Bmin   Bpa   Xoff   Yoff'
for i in range(0,ngauss):
    print '%2d %6.3f %6.3f %6.3f %6.2f %6.3f %6.3f'%(i,Cmp[i][0],Cmp[i][1],Cmp[i][2],Cmp[i][3],Cmp[i][4],Cmp[i][5])
#
##################################################
# Make Gauss-pointE.cl file'
os.system('rm -rf Gauss_pointE.cl')
cl.done()
for i in range(0,ngauss):
    FLUX = Cmp[i][0]
    MJS = str(Cmp[i][1])+'arcsec'
    MNS = str(Cmp[i][2])+'arcsec'
    PAS = str(Cmp[i][3])+'deg'
    ddelta = pos0[1] +Cmp[i][5]/3600.0
    dra = pos0[0] + Cmp[i][4]/3600.0/15.0*dcos
    dirc = posconv([dra,ddelta])
#    print i, dirc
#    print '%1d %35s %6.3f  %13s %13s %11s'%(i,dirc,fcomp[i],MJS,MNS,PAS)
    cl.addcomponent(dir=dirc, flux=FLUX, fluxunit='Jy', freq=dfreq, shape="Gaussian",majoraxis=MJS,minoraxis=MNS,positionangle=PAS)
 #
cl.rename('Gauss_pointE.cl')
print 'stored model in directory Gauss_pointE.cl'
cl.close()
#
#
if 'sim' in doinput:
    os.system('rm -rf '+PP)
    print 'make ms'
    default("simobserve")
    project = PP
    complist = 'Gauss_pointE.cl'
    compwidth = fcompwidth
    setpointings = True
    integration = fintegr
    direction = dir0 
    mapsize = fmapsize
    maptype = 'ALMA'
    antennalist= fconf
    totaltime = ftotal
    thermalnoise = 'tsys-atm'
    user_pwv = 0.5
    t_ground = 269.0
    seed = 1111
    graphics = 'both'
    verbose = True
    simobserve()
    #
#
if 'plotms' in doinput:
    print 'plot ms'
    default(plotms)
    vis = PP+'/'+PP+'.'+fconf[0:-3]+'noisy.ms'
    xaxis = 'uvdist'
    yaxis = 'amp'
    #plotfile = 'uv.png'
    plotms()
#
if 'image' in doinput:
    print 'make image'
    Im = PP+'/'+PP+'/Simage'
    os.system('rm -rf '+Im+'*')
    default("tclean") 
    vis = PP+'/'+PP+'.'+fconf[0:-3]+'noisy.ms' 
    imagename = Im
    imsize = [zimsize,zimsize]
    imdirection = dir0
    cell = '0.005arcsec'
    niter = 35000
    cycleniter = 2000
    cyclefactor = 0.2
    minpsffraction = 0.2
    maxpsffration = 0.5
    mask = 'circle[[570pix,570pix],352pix]'
    multiscale = [0,15,50,200]
    interactive = False
    weighting = 'briggs'
    robust=0
    tclean()

if 'display' in doinput:
    g = open(PP+'/image_stat.txt','w')
    Imi = PP+'/'+PP+'/Simage.image'
    Imr = PP+'/'+PP+'/Simage.residual'
    fmax = imstat(Imi)['max'][0]
    fmin = imstat(Imi)['min'][0]
    tflux = imstat(Imi)['flux'][0]
    rms = imstat(Imr)['rms'][0]
    lcntr = 3.5 * rms
    #lcntr = 6.0e-5
    a = imhead(Imi)['restoringbeam']
    bmaj = a['major']['value']
    bmin = a['minor']['value']
    bpa = a['positionangle']['value']
    # stat outside
    a1 = imstat(Imr,
           box='100,100,250,250')
    outmax = 1000.0*a1['max'][0]
    outmin = 1000.0*a1['min'][0]
    outrms = 1000.0*a1['rms'][0]
    #lcntr = 0.00035
    print 'max:%8.5f min:%8.5f flux:%8.5f rms:%8.5f (Jy) lcntr:%9.6f  bmaj:%6.3f bmin:%6.3f bpa:%6.1f'%(fmax,fmin,tflux,rms,lcntr,bmaj,bmin,bpa)
    print 'outmax:%7.2f outmin:%7.2f  outrms:%7.2f (mJy)'%(outmax,outmin,outrms)
    g.write('max:%8.5f min:%8.5f flux:%8.5f rms:%8.5f (Jy)  bmaj:%6.3f bmin:%6.3f bpa:%6.1f\n'%(fmax,fmin,tflux,rms,bmaj,bmin,bpa))
    g.write('outmax:%7.2f outmin:%7.2f  outrms:%7.2f (mJy)\n'%(outmax,outmin,outrms))
    g.close()            
    imview(raster = {'file': Imi, 'range':[-fmax/10.00,fmax], 'colormap': 'RGB 2'}, 
           contour = {'file':Imi,
                        'levels':[-1,1,2,4,8,16,32,64,128,256,512,1024],
                        'unit':float(lcntr)},
                        zoom = 2)
#  rms = 4.5E-5 Jy  How does this compare: Band 3, 1 GHz, 600 sec
#  sens calculator = 60e-6 = 6e-5.  Close enough!!
#
