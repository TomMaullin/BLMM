#!/bin/bash
fslpython -c "from sim import generateData; generateData.generate_data(1000, [100,100,100], '$1', $2)"