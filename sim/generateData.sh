#!/bin/bash
fslpython -c "from sim import generateData; generateData.generate_data(500, [100,100,100], '$1', $2)"