#!/bin/bash

python3 test_Lub_Class_CC.py >  test_Lub_Class_result.out

DIFF=$(cmp --quiet test_Lub_Class_correct.out test_Lub_Class_result.out)
if cmp --quiet test_Lub_Class_correct.out test_Lub_Class_result.out # check for exit code 0
then
    echo "TEST PASSED for ResistPairSup_py"
    rm test_Lub_Class_result.out
else
    echo "TEST FAILED for ResistPairSup_py: diff isn't correct. Check test_Lub_Class_result.out and compare to test_Lub_Class_correct.out"
fi

echo "Running TetraTest. This will take a minute."
cd ./Tetra_Test
python3 ./Tetra_Test.py --input-file=./inputfile_tetra.dat

cd ./Tet_Data

matlab -nodisplay -nosplash -nodesktop -r "run('./mob_plot.m');exit;" | tail -n +20

mv ./lub_tetra_test.png ../../lub_tetra_test.png

echo "TetraTest finished. Check lub_tetra_test.png for results."
