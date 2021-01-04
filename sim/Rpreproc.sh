#!/bin/bash
fslpython -c "from sim import generateData; generateData.Rpreproc('$1', $2, [100,100,100], $3, $4,1000)"