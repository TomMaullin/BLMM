#!/bin/bash
fslpython -c "from src import blmm_compare; blmm_compare.main('$1','$2','$3')"
