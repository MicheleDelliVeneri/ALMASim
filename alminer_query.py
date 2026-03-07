#MC Toribio 2025/10/02
#Query list of observations to the ASA

import alminer
import pandas as pd
from astropy.io import ascii

cycles = [2012 +i for i in range(4,11)] # Cycle 4 was 2016; cycle 0 was 2012

#cycle_list = ['2022.1' , '2023.1']
cycle_list = [str(s)+'.' for s in cycles] # In this way we also include .A (DDT) and  .2 (ACA supplemental call); for main call, use .1

#band_list = ['3']
#band_list_str = ''.join('B' + band for band in band_list)
arrays = ['_TP', '_7M', '_TM']

df_list = []

for cycle in cycle_list:
    print("************************")
    print("Query for cycle:", cycle)
    print("************************\n")
    proposal_id = [cycle]
    #df = alminer.keysearch({'proposal_id': proposal_id, 'band_list' : band_list})
    df = alminer.keysearch({'proposal_id': proposal_id})
   
    if df is None:
        continue
 
    # Keep only regular proposals
    df = df[df['proposal_id'].str.endswith('.S')]

    if df.empty:
        continue
     
    #Keep only QA2 PASS datasets
    df = df.query('qa2_passed == "T"')
    #filename = 'ASA_query_' + band_list_str + '_'  + cycle + '.S_QA2-PASS'
    filename = 'ASA_query_'  + cycle + '.S_QA2-PASS'
    df.to_csv( filename + '.csv')

    for arr in arrays:
        df_arr = df[df['schedblock_name'].str.contains(arr)]
        if arr == '_TM':
            arr = '_12M' 
        df_arr.to_csv(filename  +arr + '.csv') 

    if df.empty:
        continue
    df_list.append(df)

dftot = pd.concat(df_list)
filename = 'ASA_query_cycles-'  + str(min(cycles)) + '-'+ str(max(cycles)) + '.S_QA2-PASS'
dftot.to_csv( filename + '.csv')

