import os
import sys
import glob
import numpy as np


bandpass_table = glob.glob('*.bandpass')

if len(bandpass_table) != 1: sys.exit('ERROR')
bandpass_table = bandpass_table[0]

uid = bandpass_table.split('.')[0]
msname = uid + '.ms'

tb.open(bandpass_table)
field_id = tb.getcol('FIELD_ID')
spw_ids = tb.getcol('SPECTRAL_WINDOW_ID')
refant = tb.getcol('ANTENNA2')
tb.close()

field_id = np.unique(field_id)
if len(field_id) != 1: sys.exit('ERROR')
field_id = field_id[0]

spw_ids = np.unique(spw_ids)

refant = np.unique(refant)
if len(refant) != 1: sys.exit('ERROR')
refant = refant[0]

msname1 = uid + '.ms.split2'


mstransform(vis = msname,
    outputvis = msname1,
    datacolumn = 'data',
    field = str(field_id),
    spw = ','.join([str(i) for i in spw_ids]),
    reindex = False,
    keepflags = True)


antpos_table = msname + '.antpos'

wvr_table = msname + '.wvr'
wvr_table_mod = msname + '.wvr.mod'


wvr_factors = [0, 0.8, 0.9, 1, 1.1, 1.2]

phase_rms = {}



for ij in range(len(wvr_factors)):


    rmtables(wvr_table_mod)
    os.system('cp -r ' + wvr_table + ' ' + wvr_table_mod)

    tb.open(wvr_table_mod, nomodify = False)

    data1 = np.angle(tb.getcol('CPARAM')) * wvr_factors[ij]
    data2 = np.exp(1j * data1)
    tb.putcol('CPARAM', value = data2)

    tb.close()


    phasecal_table = msname1 + '.phase_int'
    rmtables(phasecal_table)


    gaincal(vis = msname1,
        caltable = phasecal_table,
        solint = 'int',
        refant = str(refant),
        gaintype = 'G',
        calmode = 'p',
        gaintable = [wvr_table_mod, antpos_table, bandpass_table])



    tb.open(phasecal_table)

    spw_ids = np.unique(tb.getcol('SPECTRAL_WINDOW_ID'))
    antenna_ids = np.unique(tb.getcol('ANTENNA1'))

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
