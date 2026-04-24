# ALMA Data Reduction Script
# $Id: scriptForPI.py,v 2.5 2023/10/03 16:56:39 dpetry Exp $

"""
ALMA scriptForPI.py - control script to restore the calibrated ALMA data for one MOUS from the raw data (ASDMs)

Usage: 1) Unpack delivery package for given MOUS
       2) find the scriptForPI in the subdirectory "script"
       3) Obtain all QA0_PASS ASDMs belonging to the MOUS and place them into the directory "raw"
          (which is at the same level as directory "script")
       4) determine correct CASA version from QA2 report
       5) launch this CASA version in directory "script" 
          (with option --pipeline if you see a *piperestorescript.py in directory "script")
       6) set option variables if needed (see below), e.g. DOSPLIT=True, and/or DOCONTSUB=True
       7) execfile('scriptForPI.py')

Note that the actual name of scriptForPI.py may have a prefix, e.g., member.<MOUS UID>.scriptForPI.py .

Example: DOSPLIT=True; DOCONTSUB=True
         execfile('member.uid___A002_Xe1f219_X7ee8.scriptForPI.py') 

Options:

  DOSPLIT - if global variable DOSPLIT is set to True (default is unset, meaning False),
        then after the end of the pipeline calibration restore, the calibrated data (not the
        contsub data) is split out into new MSs with the name <EB UID>.ms.split.cal .
        For manually calibrated data and for pipeline-calibrated data which was manually imaged, 
        the split is performed in any case.

  DOCONTSUB - if global variable DOCONTSUB is set to True (default is unset, meaning False),
       then for pipeline-imaged data, the first imaging stages are going to be run in order to
       re-create the continuum-subtracted MSs for line imaging.
       For non-pipeline imaged data, this has no effect.
       NOTE: from Cycle 10 onwards, DOCONTSUB will automatically be run if the PL used self-calibration.

  SPACESAVING - global variable SPACESAVING (default is unset, meaning all intermediate MSs are kept in place)
       can be set to an integer between -1 and 3. It controls the disk space consumption during the restore of
       manually calibrated data.
         Valid values: 0 = no saving (default, same as unset)
                              1 = delete *.ms.split
                              2 = delete *.ms and *.ms.split
                            >=3 = delete *.ms, *.ms.split, and if possible *.ms.split.cal
                             -1 = do not check disk space

  USEMS - global variable USEMS (default is unset, meaning there is no pre-existing imported MS)
       is only relevant for manually calibrated data. Setting USEMS to True indicates that you have
       a pre-existing directory "calibrated" where the import of the ASDMs to MS format has already
       taken place. You must have created the directory "calibrated" and put the imported raw MSs
       "uid*.ms" into individual working directories named "uid*.calibration" inside "calibrated/".
       This saves the time for the import. - Mostly relevant for specialized test setups.


Pipeline calibration by pipeline rerun:

  For pipeline-calibrated data, if there is no script named *casa_piperestorescript.py,
  then the scriptForPI looks for a script named *cal*casa_pipescript.py .
  If it exists, it will be run thus triggering a complete calibration pipeline re-run.
  See the pipeline documentation for more details.

"""

import numpy as np
import os
import sys
import glob
from xml.etree import cElementTree as ET

# Utilities

def genscriptnameforsorting(myscript):

    a = myscript.split("_")
    mylast = a[len(a)-1].split('.')[0].split('X')
    a[len(a)-1] = "X"+mylast[1].zfill(5)
    
    mylast = a[len(a)-2].split('X')
    a[len(a)-2] = "X"+mylast[1].zfill(5)
    
    scriptnameforsorting = a[0]
    for i in range(1,len(a)):
        scriptnameforsorting += "_"+a[i]

    return scriptnameforsorting


applyonly = True

os.environ["LANG"] = "en_US.UTF-8"

print('*** ALMA scriptForPI ***')
casalog.origin('casa')
casalog.post('*** ALMA scriptForPI ***', 'INFO', 'scriptForPI')
casalog.post('$Id: scriptForPI.py,v 2.5 2023/10/03 16:56:39 dpetry Exp $', 'INFO', 'scriptForPI')

savingslevel=0
if "SPACESAVING" in globals():
    print('SPACESAVING =', SPACESAVING)
    if (type(SPACESAVING)!=int or SPACESAVING<-1):
        sys.exit('ERROR: SPACESAVING value \"'+str(SPACESAVING)+'\" not permitted, must be int>=-1.\n'
                 + 'Valid values: 0 = no saving,\n'
                 + '              1 = delete *.ms.split,\n'
                 + '              2 = delete *.ms and *.ms.split,\n'
                 + '            >=3 = delete *.ms, *.ms.split, and if possible *.ms.split.cal'
                 + '             -1 = do not check disk space')
        
    savingslevel = SPACESAVING

dosplit = False # In case only pipeline imaging was done, the science SPWs will not be split out. Override using DOSPLIT.
if "DOSPLIT" in globals():
    print('DOSPLIT =', DOSPLIT)
    if (type(DOSPLIT)!=bool):
        sys.exit('ERROR: DOSPLIT value \"'+str(DOSPLIT)+'\" not permitted, must be bool (True or False)')
    dosplit = DOSPLIT

docontsub = False # if true, produce PL contsub products 
if "DOCONTSUB" in globals():
    print('DOCONTSUB =', DOCONTSUB)
    if (type(DOCONTSUB)!=bool):
        sys.exit('ERROR: DOCONTSUB value \"'+str(DOCONTSUB)+'\" not permitted, must be bool (True or False)')
    docontsub = DOCONTSUB

scriptdir = os.getcwd()

if (os.path.basename(scriptdir) != 'script'):
    sys.exit('ERROR: Please start this script in directory \"script\".')


scriptnames = []
unsortedscriptnames = glob.glob('*uid*.ms.scriptForCalibration.py')
# EB UIDs contain numbers which are not zero-padded. Need special treatment to sort in chronological order.
scriptnamesforsorting = []
for myname in unsortedscriptnames:
    scriptnamesforsorting.append(genscriptnameforsorting(myname))

sortindex = np.argsort(scriptnamesforsorting)
for i in sortindex:
    scriptnames.append(unsortedscriptnames[i])


isAPP = False
if len(scriptnames)>0:
    tmppipe = os.popen('grep "ALMA Phasing Project" '+scriptnames[0]+' | wc -l')
    if int((tmppipe.readline()).rstrip('\n')) > 0:
        print("This is an ALMA Phasing Project dataset.")
        isAPP = True
    tmppipe.close()

sdscriptnames = sorted(glob.glob('*uid*.ms.scriptForSDCalibration.py'))

pscriptnames = glob.glob('*casa_piperestorescript.py')
if len(pscriptnames)>1:
    print('Found more than one piperestorescript:')
    print(pscriptnames)
    sys.exit('ERROR: non-unique piperestorescript')

p2scriptnames = glob.glob('*cal*casa_pipescript.py')
if len(p2scriptnames)>1:
    print('Found more than one calibration pipescript:')
    print(p2scriptnames)
    sys.exit('ERROR: non-unique calibration pipescript')

polcalscriptnames = glob.glob('*scriptForPolCalibration.py')
if len(polcalscriptnames)>1:
    print('Found more than one scriptForPolCalibration:')
    print(polcalscriptnames)
    sys.exit('ERROR: non-unique scriptForPolCalibration')

fluxcalscriptnames = glob.glob('*scriptForFluxCalibration.py')
if len(fluxcalscriptnames)>1:
    print('Found more than one scriptForFluxCalibration:')
    print(fluxcalscriptnames)
    sys.exit('ERROR: non-unique scriptForFluxCalibration')

imprepscriptnames = glob.glob('*scriptForImagingPrep.py')
if len(imprepscriptnames)>1:
    print('Found more than one scriptForImagingPrep:')
    print(imprepscriptnames)
    sys.exit('ERROR: non-unique scriptForImagingPrep')


isRenorm = False
if len(fluxcalscriptnames)>0:
    tmppipe = os.popen('grep "ACreNorm" '+fluxcalscriptnames[0]+' | wc -l')
    if int((tmppipe.readline()).rstrip('\n')) > 0:
        print("NOTE: This is a dataset with a correction for the Tsys/Autocorrelation renormalization issue.")
        print("      See the description in "+fluxcalscriptnames[0]+" and references therein.")
        isRenorm = True
    tmppipe.close()
elif len(polcalscriptnames)>0:
    tmppipe = os.popen('grep "Renormalization" '+polcalscriptnames[0]+' | wc -l')
    if int((tmppipe.readline()).rstrip('\n')) > 0:
        print("NOTE: This is a dataset with a correction for the Tsys/Autocorrelation renormalization issue.")
        print("      See the description in "+polcalscriptnames[0]+" and references therein.")
        isRenorm = True
    tmppipe.close()

if isRenorm:
    if savingslevel>1:
        savingslevel = 1
        print("      Limiting SPACESAVING to 1 since *.ms will be needed for the additional processing.")

has_uvcontfit = False
if len(p2scriptnames)>0:
    tmppipe = os.popen('grep "hif_uvcontfit" '+p2scriptnames[0]+' | wc -l')
    if int((tmppipe.readline()).rstrip('\n')) > 0:
        print("This ALMA dataset was processed with a pipeline using hif_uvcontfit.")
        has_uvcontfit = True
    tmppipe.close()


pprnames = glob.glob('*PPR*.xml') # old naming scheme
if len(pprnames)>1:
    print('Found more than one PPR:')
    print(pprnames)
    sys.exit('ERROR: non-unique PPR')

if len(pprnames)==0:
    pprnames = glob.glob('*cal*pprequest.xml')
    if len(pprnames)>1:
        print('Found more than one calibration PPR:')
        print(pprnames)
        sys.exit('ERROR: non-unique calibration PPR')

if not os.path.exists('../calibration'):
    sys.exit('ERROR: Cannot find directory \"calibration\"')

manifestnames = glob.glob('*pipeline_manifest.xml')
if len(manifestnames)>0:
    print('Found pipeline manifest(s):')
    print(manifestnames)
    for mname in manifestnames:
        themanifest = ET.parse(mname).find('ous/aux_products_file')
        os.system('cp '+mname+' ../calibration')
        if not themanifest==None:
            manifest_itemsdict = dict(themanifest.items())
            if 'name' in manifest_itemsdict.keys():
                os.chdir('../calibration')
                supportingfilenames = [manifest_itemsdict['name']]
                if os.path.exists(supportingfilenames[0]):
                    print("Unpacking "+supportingfilenames[0]+" ...")
                    os.system('tar xzf '+supportingfilenames[0])
                else:
                    print("WARNING: could not find the file "+supportingfilenames[0]
                          +" in directory \"calibration\" although the pipeline manifest mentions it.")
                os.chdir('../script')

else:
    if len(pprnames)>0:
        print("No pipeline manifest found.")
    os.chdir('../calibration')
    supportingfilenames = glob.glob('*supporting.tgz')
    if len(supportingfilenames)>0:
        print('Found tarball(s) containing supporting files:')
        print(supportingfilenames)
        for sname in supportingfilenames:
            os.system('tar xzf '+sname)
    else:
        supportingfilenames = glob.glob('*auxproducts.tgz')
        if len(supportingfilenames)>0:
            print('Found tarball(s) containing auxiliary files:')
            print(supportingfilenames)
            for sname in supportingfilenames:
                os.system('tar xzf '+sname)

os.chdir(scriptdir)

pipererun = False
if ((len(scriptnames) + len(pscriptnames) + len(sdscriptnames))  == 0):
    if len(p2scriptnames)>0:
        print('Pipeline calibration by pipeline rerun:')
        pipererun = True
    else:
        sys.exit('ERROR: No calibration script found.')

istppipe = False
jyperknames = glob.glob('../calibration/*jyperk*.csv')
if len(jyperknames)==1:
    istppipe = True
elif len(jyperknames)>1:
    print('Found more than jyperk.csv file:')
    print(jyperknames)
    sys.exit('ERROR: non-unique jyperk.csv file')

pprasdms = []
usedimpipe = False
if (len(pprnames)>0):
    for line in open(pprnames[0]):
        if "<AsdmDiskName>" in line:
            pprasdms.append(line[line.index('uid'):line.index('</')])

    tmppipe = os.popen("grep hif_makeimages "+pprnames[0]+" | wc -l")
    nummkim = int((tmppipe.readline()).rstrip('\n'))
    tmppipe.close()
    if ( nummkim > 1 ):
        print("Science pipeline imaging use was foreseen in PPR.")
        tmppipe = os.popen("ls ../product/*_sci.spw*.fits 2>/dev/null | wc -l ")
        numscienceim = int((tmppipe.readline()).rstrip('\n'))
        tmppipe.close()
        tmppipe = os.popen("cat *scriptForImaging.py | grep -v \"#\" | wc -l ")
        numuncomlines = int((tmppipe.readline()).rstrip('\n'))
        tmppipe.close()
        if (numscienceim>0 and numuncomlines==0):
            print("Science images and only a dummy scriptForImaging were found. Will assume only pipeline imaging took place.")
            usedimpipe = True

try:
    if os.path.islink('../raw'):
        print("Note: your raw directory is a link.")
    os.chdir('../raw/')
except:
    sys.exit('ERROR: directory \"raw\" not present.\n'
             '       Please download your raw data and unpack it to create and fill directory \"raw\".') 

# check available disk space
tmppipe = os.popen("df -P -m $PWD | awk '/[0-9]%/{print($(NF-2))}'")
avspace = int((tmppipe.readline()).rstrip('\n'))
tmppipe.close()
tmppipe = os.popen("du -smc $PWD | grep total | tail -n 1 | cut -f1")
packspace = int((tmppipe.readline()).rstrip('\n'))
tmppipe.close()

spacefactor = 0.

fcalpresent = False
if len(fluxcalscriptnames)>0:
    fcalpresent = True
    spacefactor = 1.

impreppresent = False
if len(imprepscriptnames)>0:
    impreppresent = True
    spacefactor = 1.

polcalpresent = False
if  len(polcalscriptnames)>0:
    polcalpresent = True
    spacefactor = 1.

spaceneed = packspace*(11.+spacefactor*3.)

if (savingslevel==1):
    print('Will delete intermediate MSs named *.ms.split to save disk space.')
    spaceneed = packspace*(7.+spacefactor*3.)   
elif (savingslevel==2):
    print('Will delete intermediate MSs named *.ms and *.ms.split to save disk space.')
    spaceneed = packspace*(3.+spacefactor*3.)   
elif (savingslevel>=3):
    print('Will delete all intermediate MSs to save disk space.')
    spaceneed = packspace*(3.+spacefactor*3.)   
elif (savingslevel==-1):
    print('Will not check available disk space.')
    spaceneed = 0
    

print('Found ',avspace,' MB of available free disk space.')
print('Expect to need up to ',spaceneed,' MB of free disk space.')
if(spaceneed>avspace):
    print('ERROR: not enough free disk space. Need at least '+str(spaceneed)+' MB.')
    print('If you think this is not correct and want to try anyway, please set SPACESAVING to -1.')
    sys.exit('ERROR: probably not enough free disk space.')


asdmnames = glob.glob('uid*.asdm.sdm')

if len(asdmnames) == 0:
    sys.exit('ERROR: No ASDM found in directory \"raw\".')

print('Found the following ASDMs:', asdmnames)

for i in range(len(asdmnames)):
    asdmnames[i] = asdmnames[i].replace('.asdm.sdm', '')


scriptasdms = []
for i in range(len(scriptnames)):
    tmps = scriptnames[i].replace('.ms.scriptForCalibration.py', '')
    tpos = tmps.rfind('uid__')
    scriptasdms.append(tmps[tpos:])    

sdscriptasdms = []
for i in range(len(sdscriptnames)):
    tmps = sdscriptnames[i].replace('.ms.scriptForSDCalibration.py', '')
    tpos = tmps.rfind('uid__')
    sdscriptasdms.append(tmps[tpos:])

allasdms = []
allasdms.extend(scriptasdms)
allasdms.extend(sdscriptasdms)
allasdms.extend(pprasdms)

missing = []

if sorted(asdmnames) != sorted(allasdms):
    print("WARNING: Inconsistency between ASDMs and calibration scripts")
    print("         Calibration info available for: ", sorted(allasdms))
    print("         ASDMs available in directory raw: ", sorted(asdmnames))
    for myname in allasdms:
        if not (myname in asdmnames):
            missing.append(myname)
    if len(missing)==0:
        print("       The ASDMs without calibration info are probably \"QA semipass\" data which were")
        print("       not used to create the science products and are not needed to achieve the science goal.")
        print("       Only the ASDMs for which there is calibration information will be calibrated.")
    else:
        print("ERROR: the following ASDMs have calibration information but are absent from directory \"raw\":")
        print(missing)
        print("Will try to proceed with the rest ...")
        for myname in missing:
            if myname in scriptasdms:
                scriptasdms.remove(myname)
            if myname in sdscriptasdms:
                sdscriptasdms.remove(myname)
            if myname in pprasdms:
                pprasdms.remove(myname)
            if myname in allasdms:
                allasdms.remove(myname)
        if(len(allasdms)==0):
            sys.exit('ERROR: Nothing to process.')

os.chdir(scriptdir)

ephnames = glob.glob('../calibration/*.eph')

if len(ephnames)>0:
    print("Note: this dataset uses external ephemerides.")
    print("      You can find them in directory \"calibration\".")

if os.path.exists('../calibrated') and not ("USEMS" in globals()):
    sys.exit('WARNING: will stop here since directory '+os.path.abspath('../calibrated')
             +' already exists.\nPlease delete it first and then try again.')
    
if not ("USEMS" in globals()):
    print('Creating destination directory for calibrated data.')
    os.mkdir('../calibrated')
else:
    print('You have set USEMS. Will use your pre-imported MSs rather than importing them from the ASDMs.')
    for asdmname in scriptasdms + sdscriptasdms:
        if not os.path.exists('../calibrated/'+asdmname+'.calibration/'+asdmname+'.ms'):
            print('When USEMS is set, you must have created the directory \"calibrated\" and')
            print('put the imported raw MSs \"uid*.ms\" in individual working directories')
            print('named \"uid*.calibration\" inside \"calibrated\".')
            sys.exit('ERROR: cannot find calibrated/'+asdmname+'.calibration/'+asdmname+'.ms')
    
os.chdir('../calibrated')


for asdmname in scriptasdms + sdscriptasdms:

    print('>>> Processing ASDM '+asdmname)
    
    if not ("USEMS" in globals()):
        os.mkdir(asdmname+'.calibration')

    os.chdir(asdmname+'.calibration')

    if not os.path.exists('../../raw/'+asdmname+'.asdm.sdm'):
        sys.exit('ERROR: cannot find raw/'+asdmname+'.asdm.sdm')

    os.system('ln -sf ../../raw/'+asdmname+'.asdm.sdm '+asdmname)

    if isAPP:
        app_cal_tables = glob.glob('../../calibration/*.tgz')
        for cal_table in app_cal_tables:
            if not "plots" in cal_table:
                os.system('tar -xzf  '+cal_table)


    for ephname in ephnames: 
        os.system('ln -sf ../'+ephname)

    thecalscript = []
    if asdmname in scriptasdms:
        thecalscript = glob.glob('../../script/*'+asdmname+'.ms.scriptForCalibration.py')
    if asdmname in sdscriptasdms:
        thecalscript = glob.glob('../../script/*'+asdmname+'.ms.scriptForSDCalibration.py')
    if thecalscript == []:
        casalog.post('ERROR: no calibration script found for ASDM '+asdmname, 'WARN', 'scriptForPI')
    else:
        if len(thecalscript)>1:
            casalog.post('More than one calibration script found for ASDM '+asdmname+'\nWill use the first one.', 'WARN', 'scriptForPI')

        casalog.post('*** Running '+thecalscript[0]+' in calibrated/'+asdmname+'.calibration ***', 'INFO', 'scriptForPI')
        print('Running '+thecalscript[0])
        execfile(thecalscript[0], globals())

    if not os.path.exists(asdmname+'.ms.split.cal'):
        print('ERROR: '+asdmname+'.ms.split.cal was not created.')
    else:
        print(asdmname+'.ms.split.cal was produced successfully, moving it to \"calibrated\" directory.')
        os.system('mv '+asdmname+'.ms.split.cal ..')
        if isAPP:
            os.system('mv '+asdmname+'.polcalibrated.APP.ms ..')

        if (savingslevel>=2):
            print('Deleting intermediate MS ', asdmname+'.ms')
            os.system('rm -rf '+asdmname+'.ms')
        if (savingslevel>=1):
            print('Deleting intermediate MS ', asdmname+'.ms.split')
            os.system('rm -rf '+asdmname+'.ms.split')

    os.chdir('..')

if (len(pprasdms)>0):

    if dir().count('h_init')==0:
        sys.exit("ERROR: Pipeline not available. Make sure you start CASA with option --pipeline to activate the pipeline.") 

    if pipererun:
        print('Processing the ASDMs ', pprasdms, ' in pipeline rerun.')
    else:
        print('Processing the ASDMs ', pprasdms, ' using pipeline restore.')

        os.mkdir('rawdata')
        os.chdir('rawdata')
        for asdmname in pprasdms:
            if not os.path.exists('../../raw/'+asdmname+'.asdm.sdm'):
                sys.exit('ERROR: cannot find raw/'+asdmname+'.asdm.sdm')

            os.system('ln -sf ../../raw/'+asdmname+'.asdm.sdm '+asdmname)

        os.chdir('..')
        
        os.system('ln -sf ../calibration products')

    os.mkdir('working')
    os.chdir('working')

    if pipererun:
        for asdmname in pprasdms:
            if not os.path.exists('../../raw/'+asdmname+'.asdm.sdm'):
                sys.exit('ERROR: cannot find raw/'+asdmname+'.asdm.sdm')

            os.system('ln -sf ../../raw/'+asdmname+'.asdm.sdm '+asdmname)

        if istppipe:
            os.system('cp ../../calibration/*jyperk.csv ./jyperk.csv')
            os.system('cp ../../calibration/*jyperk_query.csv ./jyperk_query.csv')

        os.system('cp ../../calibration/*flagtemplate.txt .')

        os.system('cp ../../calibration/*pipeline_manifest.xml .')

        fluxfiles = glob.glob("../../calibration/*flux.csv")
        if len(fluxfiles)>0:
            if len(fluxfiles)==1:
                os.system('cp ../../calibration/*flux.csv ./flux.csv')
            else:
                print(fluxfiles)
                sys.exit('ERROR: found more than one *flux.csv file in directory "calibration"')

        antennaposfiles = glob.glob("../../calibration/*antennapos.csv")
        if len(antennaposfiles)>0:
            if len(antennaposfiles)==1:
                os.system('cp ../../calibration/*antennapos.csv ./antennapos.csv')
            else:
                print(antennaposfiles)
                sys.exit('ERROR: found more than one *antennapos.csv file in directory "calibration"')

        contfiles = glob.glob("../../calibration/*cont.dat")
        if len(contfiles)>0:
            if len(contfiles)==1:
                os.system('cp ../../calibration/*cont.dat ./cont.dat')
            else:
                print(contfiles)
                sys.exit('ERROR: found more than one *cont.dat file in directory "calibration"')

        print("Directory \"working\" set up for pipeline re-run:")
        os.system('ls -l')
        
        execfile('../../script/'+p2scriptnames[0], globals())

    else:
        print("now running ", pscriptnames[0])
        execfile('../../script/'+pscriptnames[0], globals())

        selfcalfiles = glob.glob("../../calibration/*selfcal.json")
        doselfcal=False
        if len(selfcalfiles)==1:
            print('Found '+selfcalfiles[0]+' . Will create contsub and selfcal products ...')
            os.system('cp ../../calibration/*selfcal.json .')
            docontsub=True
            doselfcal=True
        elif len(selfcalfiles)>1:
            print(selfcalfiles)
            sys.exit('ERROR: found more than one *selfcal.json file in directory "calibration"')

        if docontsub: # run imaging PL stages to produce contsub imaging input MSs
            if istppipe:
                print('\n *** NOTE: You have set DOCONTSUB to True but this is TP data and uvcontsub cannot be performed.')
            else:
                contfiles = glob.glob("../../calibration/*cont.dat")
                if len(contfiles)>0:
                    if len(contfiles)==1:
                        os.system('cp ../../calibration/*cont.dat ./cont.dat')
                    else:
                        print(contfiles)
                        sys.exit('ERROR: found more than one *cont.dat file in directory "calibration"')

                    if has_uvcontfit:
                        print('\n*** Creating contsub products using hif_uvcontfit and existing cont.dat file from directory "calibration" ...')    
                        hif_mstransform(pipelinemode="automatic")
                        hifa_flagtargets(pipelinemode="automatic")
                        hif_uvcontfit(pipelinemode="automatic")
                        hif_uvcontsub(pipelinemode="automatic")
                    else:
                        print('\n*** Creating contsub products using existing cont.dat file from directory "calibration" ...')    
                        hif_mstransform()
                        hifa_flagtargets()
                        hif_uvcontsub()
                        if doselfcal:
                            print('\n*** Applying existing selfcal solutions based on auxproduct '+selfcalfiles[0]+' ...')
                            hif_selfcal()

                    print('Generated pipeline products with continuum-subtraction will be linked into directory "calibrated":')
                    for mysuffix in ['target.ms', 'targets.ms', 'line.ms', 'cont.ms']:
                        mymss = glob.glob('*'+mysuffix)
                        for myms in mymss:
                            print('   '+myms)
                            os.system('cd ..; ln -sf working/'+myms)
                
                else:
                    print('\n *** NOTE: no PL file cont.dat found. No continuum subtraction performed.')

    mysciencespws = {}
    for asdmname in pprasdms:
        if not os.path.exists(asdmname+'.ms'):
            print('ERROR: '+asdmname+'.ms was not created.')
        elif pipererun and istppipe:
            tpmsnames = glob.glob(asdmname+'*.ms*_bl')
            if len(tpmsnames)==0:
                print('ERROR: '+asdmname+'*.ms*_bl was not created.')
            else:
                os.system('mv '+asdmname+'*.ms*_bl ..')
                
            if (savingslevel>=2):
                print('Deleting intermediate MS ', asdmname+'.ms')
                os.system('rm -rf '+asdmname+'.ms')
        else:
            msmd.open(asdmname+'.ms')
            targetspws = msmd.spwsforintent('OBSERVE_TARGET*')
            sciencespws = ''
            outputspws = ''
            i = 0
            for myspw in targetspws:
                if msmd.nchan(myspw)>4:
                            sciencespws += str(myspw)+','
                            outputspws += str(i)+','
                            i += 1
            sciencespws = sciencespws.rstrip(',')
            outputspws = outputspws.rstrip(',')
            msmd.close()

            if usedimpipe and not dosplit:
                print('Imaging pipeline was used. Will not create '+asdmname+'.ms.split.cal')
                print('Linking MS '+asdmname+'.ms into directory "calibrated"')
                os.system('cd ..; ln -sf working/'+asdmname+'.ms')
                if os.path.exists(asdmname+'.ms.flagversions'):
                    os.system('cd ..; ln -sf working/'+asdmname+'.ms.flagversions')
            else:
                mysciencespws[asdmname] = sciencespws
    
    needs_reindex=False
    myasdmnames = list(mysciencespws.keys())
    for asdmname in myasdmnames:
        if mysciencespws[asdmname] != mysciencespws[myasdmnames[0]]:
            needs_reindex=True
            print('Note: The science SPW IDs are not the same for all EBs. SPW IDs will be reindexed to start at 0!')
            break

    for asdmname in myasdmnames:
        print('Splitting out science SPWs for '+asdmname+': '+mysciencespws[asdmname])
        mstransform(vis=asdmname+'.ms', outputvis=asdmname+'.ms.split.cal', spw = mysciencespws[asdmname], reindex=needs_reindex)
        if not os.path.exists(asdmname+'.ms.split.cal'):
            print('ERROR: '+asdmname+'.ms.split.cal was not created.')
        else:
            os.system('mv '+asdmname+'.ms.split.cal ..')
            if (savingslevel>=2):
                print('Deleting intermediate MS ', asdmname+'.ms')
                os.system('rm -rf '+asdmname+'.ms')
        

    os.chdir('..')

if polcalpresent:
    print('Executing scriptForPolCalibration.py ...')
    execfile('../script/'+polcalscriptnames[0], globals())

if fcalpresent:
    print('Executing scriptForFluxCalibration.py ...')
    execfile('../script/'+fluxcalscriptnames[0], globals())

if impreppresent:
    print('Executing scriptForImagingPrep.py ...')
    execfile('../script/'+imprepscriptnames[0], globals())

if (savingslevel>=3) and os.path.exists('calibrated.ms'):
    for asdmname in allasdms:
        print('Deleting intermediate MS ', asdmname+'.ms.split.cal')
        os.system('rm -rf '+asdmname+'.ms.split.cal')

print('Done. Please find results in directory \"calibrated\".')
casalog.origin('casa')
casalog.post('ALMA scriptForPI completed.', 'INFO', 'scriptForPI')
