###############################
#########################

#! /bin/bash

LIST=("
https://almascience.eso.org/dataPortal/2024.1.00216.S_uid___A002_X11e59f4_X8c02.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00216.S_uid___A002_X11e59f4_X92f1.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00315.S_uid___A002_X11e3e46_X12b6.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00315.S_uid___A002_X11e3e46_X1743.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00315.S_uid___A002_X11e3e46_X5230.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00315.S_uid___A002_X11e3e46_X983.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00315.S_uid___A002_X11e3e46_Xcbb.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00315.S_uid___A002_X11e3e46_Xf00b.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00360.S_uid___A002_X11e3e46_X4c2.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00360.S_uid___A002_X11eaef9_Xba24.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e3e46_X1079b.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e3e46_X57a2.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e3e46_X76df.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e3e46_Xa615.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e3e46_Xea45.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e59f4_X1078e.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e59f4_X1125a.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e59f4_X12dcd.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e59f4_X800.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11e7adc_Xa83.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11eaef9_X122cd.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11eaef9_X1632.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11eaef9_X19e2.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11eaef9_X1e6b.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11eaef9_Xbd36.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11eaef9_Xe2f8.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11edb41_X345.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11edb41_X998.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11f6e10_X9bf.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11f70c4_X133f6.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X11f70c4_X137a4.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X12021d4_X1ea5.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X12021d4_X5e0.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X120b4e9_X10c0d.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00408.S_uid___A002_X120b4e9_X11a00.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11e3e46_X90a9.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11e3e46_X9518.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11e3e46_X9b6c.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11e3e46_Xa4ee.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11e3e46_Xadab.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11e59f4_X1260a.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11e59f4_X35c.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11eaef9_X1330.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00657.S_uid___A002_X11eaef9_X1783.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00808.S_uid___A002_X11e3e46_X748.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00808.S_uid___A002_X11e3e46_Xc458.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00808.S_uid___A002_X11e3e46_Xd38c.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00808.S_uid___A002_X11e59f4_X1009e.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.00808.S_uid___A002_X11e59f4_X39e8.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01015.S_uid___A002_X11e59f4_Xa1f9.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01015.S_uid___A002_X11edb41_X23aa2.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01015.S_uid___A002_X11edb41_X24631.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01015.S_uid___A002_X120e7e4_X7411.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01104.S_uid___A002_X11e3e46_X723b.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X11e59f4_X1c5.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X11e7adc_X20a.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X11edb41_X139e2.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X11edb41_X2223f.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X11edb41_Xa6fa.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X11f1961_X3d38.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X12235df_X8bbd.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01212.L_uid___A002_X122494b_X1c3c3.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01252.S_uid___A002_X11e3e46_X5e88.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01252.S_uid___A002_X11e3e46_X6626.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01304.S_uid___A002_X12b5566_Xaac8.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01304.S_uid___A002_X12b966a_X4e56.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01304.S_uid___A002_X12ba787_X49e1.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X11e3e46_X99bc.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X11e59f4_X12d.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X11e59f4_Xb5df.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X11e59f4_Xb749.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X11e59f4_Xc570.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X11e59f4_Xf1b.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X121e064_Xa81d.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01335.S_uid___A002_X12235df_X20b9.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01482.S_uid___A002_X11e7adc_X301a.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01482.S_uid___A002_X120b4e9_X3c37.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01553.S_uid___A002_X11e7adc_Xbe4c.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01586.S_uid___A002_X11e3e46_Xd7b3.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01586.S_uid___A002_X11e3e46_Xddf1.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01586.S_uid___A002_X11e3e46_Xe372.asdm.sdm.tar
https://almascience.eso.org/dataPortal/2024.1.01586.S_uid___A002_X11e3e46_Xe939.asdm.sdm.tar
")

run_wget() {

    start=$(date +%s)

    echo Sleeping ...
    sleep `echo "scale=4; 5.0 * $RANDOM/32000" | bc `
    echo Downloading $1 ...
    wget -cq $1
    filename=`basename $1`
    echo $filename
    myfilesize=$(stat --format=%s $filename)
    end=$(date +%s)
    seconds=$(($end-$start+1))
    bytespersecond=`echo $myfilesize/$seconds/1000/1000 | bc`
    echo "Finished $1 time $seconds s, speed $bytespersecond Mytes/s"

}

mkdir -p asdms
cd asdms
export -f run_wget
printf '%s\n' "${LIST[@]}" | xargs -L1 -P10 -I {} bash -c 'run_wget "$@"' _ {}
