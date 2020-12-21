#!/bin/bash
fslpython -c "from sim import generateData; generateData.generate_data(100, np.array([100,100,100]), '$1', $2)"