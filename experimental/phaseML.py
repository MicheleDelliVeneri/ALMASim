import os
import sys
import glob
import numpy as np


bandpass_table = glob.glob('*.bandpass')

if len(bandpass_table) != 1: sys.exit('ERROR')
# gets the bandpass table
bandpass_table = bandpass_table[0]

# uses the bandpass table to get the uid
uid = bandpass_table.split('.')[0]
# uses the uid to get the msname
msname = uid + '.ms'
# opens the bandpass table and gets the field_id, spw_ids, and refant
tb.open(bandpass_table)
field_id = tb.getcol('FIELD_ID')
spw_ids = tb.getcol('SPECTRAL_WINDOW_ID')
refant = tb.getcol('ANTENNA2')
tb.close()

field_id = np.unique(field_id)
if len(field_id) != 1: sys.exit('ERROR')
field_id = field_id[0]
# gets the unique spectral window ids
spw_ids = np.unique(spw_ids)

refant = np.unique(refant)
if len(refant) != 1: sys.exit('ERROR')
refant = refant[0]

msname1 = uid + '.ms.split2'

# splits the ms into a new ms with only the specified field and spws
# found in the bandpass table. Sets reindex to False, and keepflags to True
# to make sure that the data is not reindexed and the flags are kept
mstransform(vis = msname,
    outputvis = msname1,
    datacolumn = 'data',
    field = str(field_id),
    spw = ','.join([str(i) for i in spw_ids]),
    reindex = False,
    keepflags = True)

# sets the names of the antpos table, wvr table, and the modified wvr table
antpos_table = msname + '.antpos'
wvr_table = msname + '.wvr'
wvr_table_mod = msname + '.wvr.mod'


wvr_factors = [0, 0.8, 0.9, 1, 1.1, 1.2]
# It initializes an empty dictionary called phase_rms, 
# which will store the root mean square (rms) values of
# phase solutions for each wvr factor, spectral window, 
# antenna and polarization.
phase_rms = {}
for ij in range(len(wvr_factors)):

    rmtables(wvr_table_mod)
    # copies the wvr table to a new table
    os.system('cp -r ' + wvr_table + ' ' + wvr_table_mod)
    # opens the modified wvr table
    tb.open(wvr_table_mod, nomodify = False)
    # It reads the complex values of cparam column from 
    # the table, which are phase corrections in radians. 
    # It then multiplies them by the current wvr factor 
    # and converts them back to complex values using 
    # numpy.angle and numpy.exp functions. It then writes 
    # them back to cparam column in the table.
    data1 = np.angle(tb.getcol('CPARAM')) * wvr_factors[ij]
    data2 = np.exp(1j * data1)
    # sets the modified wvr table to have the new complex gains
    tb.putcol('CPARAM', value = data2)
    tb.close()


    phasecal_table = msname1 + '.phase_int'
    rmtables(phasecal_table)

    # It uses gaincal tool from CASA to calibrate phase 
    # solutions for msname1 using gaintype G (gain), 
    # calmode p (phase), solint int (integration time), 
    # refant (reference antenna), and gaintable 
    # (a list of tables that contain calibration solutions).
    # The gaintable includes modified wvr table, antpos 
    # table and bandpass table
    gaincal(vis = msname1,
        caltable = phasecal_table,
        solint = 'int',
        refant = str(refant),
        gaintype = 'G',
        calmode = 'p',
        gaintable = [wvr_table_mod, antpos_table, bandpass_table])


    # opens the phasecal table generated with gaincal
    tb.open(phasecal_table)
    # gets the spectral window ids and antenna ids
    spw_ids = np.unique(tb.getcol('SPECTRAL_WINDOW_ID'))
    antenna_ids = np.unique(tb.getcol('ANTENNA1'))
    # initializes a data dictionary for the current wvr factor  
    phase_rms[ij] = {}

    for i in spw_ids:
        if i not in phase_rms[ij].keys(): phase_rms[ij][i] = {}
        for j in antenna_ids:
            if j not in phase_rms[ij][i].keys(): phase_rms[ij][i][j] = {}
            tb1 = tb.query('SPECTRAL_WINDOW_ID == '+str(i)+' AND ANTENNA1 == '+str(j))
            data1 = np.angle(tb1.getcol('CPARAM'))
            for k in range(2):
                data2 = np.unwrap(data1[k][0])
                phase_rms[ij][i][j][k] = np.std(data2)

    tb1.close()
    tb.close()





for ij in range(1, len(wvr_factors)):

    phase_rms_ratio = []

    for i in phase_rms[ij].keys():
        for j in phase_rms[ij][i].keys():
            for k in range(2):
                phase_rms_ratio.append(phase_rms[ij][i][j][k] / phase_rms[0][i][j][k])

    print(wvr_factors[ij], np.nanmean(phase_rms_ratio), np.nanstd(phase_rms_ratio))
