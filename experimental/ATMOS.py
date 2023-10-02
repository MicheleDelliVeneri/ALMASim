"""
PROGRAM TO INSERT ATMOSPHERIC NOISE INTO A U-V DATA SET

0.  Put in necessary input data
1.  subroutine PRMS to determine atmospheric phase variations
      atmosphere scale factor included
2.  Default steps to run
3.  Put in required input data in main directory
    ORIG.ms    original uv data set
    fconf = ALMA configuration array in standard format
    Other parameters:
    frefant = referance antenna
    im_name = output image name
4.  HGetting ALMA parameters from fconv
5.  Determine antenna-based phased variations = put in gaincal file atmos.cal
     and apply to original ms.
6.  Image and display.  

"""
#
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
#
#    0. Input data:
scaleF = 0.5                    # multiplication factor for tropospheric phases
Orig_data = 'ORIG.ms'            # INPUT ms is ORIG.ms
frefant = '13'                   # reference antenna
fconf = 'alma.cycle8.11.cfg'     # Configuration file
im_name = 'ATM_0.5'              # Image name
'im_name.txt'                    # output file with some image statistics.

#    1. SUBROUTINE TO ADD PHASE FLUCTUATIONS
#
#    prms = PRMS(antbl,scaled):
#       antbl is antenna separation from reference antenna
#       prms is the rms phase for that antenna separation
#
def PRMS(antbl,scaleF):
    #  The structure function here gives 30 deg phase rms at 10000m = 10km
    #
    Lrms = 1.0/52.83 * antbl**0.8     # phase rms ~0.8 power to 10 km
    Hrms = 3.0 * antbl**0.25          # phase rms `0.25 power beyond 10 km
    if antbl < 10000.0:
        prms = scaleF*Lrms
    if antbl > 10000.0:
        prms = scaleF*Hrms
    return prms
#
#   2.  Default steps to run
do_input = ['GT','image','display']
#
#   3. Original uv data
#       USE MODELF1.py to make a uvdata set from a set of Gaussian models.
#       and copy it (or any uvdata set) into 'ORIG.ms' in current directory. 
#

os.system('cp -r ORIG.ms UVdata.ms') 
print 'original_data ORIG.ms'
print 'modified_data UVdata.ms'
#
#  A few other parameters
#
#    4. GET ANTENNA PARAMETERS NEEDED FROM fconf (configuration file)
# 
f = open(fconf)
lines = f.readlines()
nlines = len(lines)
f.close()
#
zant = []
zx = []
zy = []
zz = []
zpad = []
for i in range(3,nlines):
    stuff = lines[i].split()
    zx.append(float(stuff[0]))
    zy.append(float(stuff[1]))
    zz.append(float(stuff[2]))
    zpad.append(stuff[4][0:4])
nant = len(zx)
zxref = []
zyref = []
zzref = []
zztot = []
nref = int(frefant)
for i in range(0,nant):
    zxref.append(zx[i]-zx[nref])
    zyref.append(zy[i]-zy[nref])
    zzref.append(zz[i]-zz[nref])
for i in range(0,nant):
    # The distance antenna is from reference
    # Used for getting antenna structure function
    zztot.append(np.sqrt(zxref[i]**2+zyref[i]**2+zzref[i]**2))
################################################
#
#   5.  NOW DETERMINE THE PHASE VARIATION FOR EACH DATA POINT
#
# make gaintable 'atmos.cal' to be filled with atmospheric phase
if 'GT' in do_input:
    tb.close()
    os.system('rm -rf atmos.cal')
    gaincal(vis = 'UVdata.ms',
            caltable = 'atmos.cal',
            refant = frefant,
            minsnr = 0.001,
            solint = '40s',    # should be time-scale of troposphere change.
            calmode = 'p')
    #
    # Obtain certain 'atmos.cal' columns
    #   ans fill in antenna-based antmospheric phases with
    #
    tb.open('atmos.cal',nomodify=False)
    yant = tb.getcol('ANTENNA1')
    ytime = tb.getcol('TIME')
    ycparam = tb.getcol('CPARAM')
    Nycparam = ycparam
    nant = len(yant)
    print '     ant   time  antsep   rmsphase'
    for i in range(0,nant):  
        antbl = zztot[yant[i]]
        # get rms phase for each antenna
        prms = PRMS(antbl,scaleF)
        # determine random GAUSSIAN phase error from rms phase
        perror = random.gauss(0,prms)
        # PRINT OUT ANT PHASE ERRORS
        print '%3d %3d %7.1f %7.1f %7.1f %7.1f '%(i, yant[i],ytime[i]-ytime[0],antbl,prms,perror)
        # put random phase in gaintable column CPARAM
        rperror = np.cos(perror*pi/180.0)
        iperror = np.sin(perror*pi/180.0)
        Nycparam[0][0][i] = 1.0*np.complex(rperror,iperror)  #X POL
        Nycparam[1][0][i] = 1.0*np.complex(rperror,iperror)  #Y POL  ASSUMED SAME
    tb.putcol('CPARAM',Nycparam)
    tb.flush()                  
    tb.close()
    #
    # plot random atmospheric phases
    #plotcal(caltable = 'atmos.cal',
    #        xaxis = 'time',
    #        yaxis = 'phase',
    #        iteration = 'antenna',
    #        plotrange = [0,0,-180,180],
    #        subplot = 331)

    #apply the atmospheric phases to uvdata
    applycal(vis = 'UVdata.ms',
             gaintable = 'atmos.cal')
###################################################
######$ END OF ATMOSPHERE PHASE ADDITION
###################################################
#
if 'image' in do_input:
    #
    zimsize = 1024
    os.system('rm -rf '+im_name+'*')
    default("tclean") 
    vis = 'UVdata.ms'
    imagename = im_name
    imsize = [zimsize,zimsize]
    #imdirection = dir0
    cell = '0.005arcsec'
    niter = 25000
    cycleniter = 2000
    cyclefactor = 0.2
    minpsffraction = 0.2
    maxpsffration = 0.5
    mask = 'circle[[570pix,570pix],352pix]'
    multiscale = [0,15,50,100]
    interactive = False
    weighting = 'briggs'
    robust=0
    tclean()

if 'display' in do_input:
    g = open(im_name+'.txt','w')
    Imi = im_name+'.image'
    Imr = im_name+'.residual'
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
    lcntr = 0.00035
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

