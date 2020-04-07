#!/bin/bash
fslpython -c "from src import blmm_setup; blmm_setup.main('$1')"
