#!/bin/bash
fslpython -c "from blmm.lib import blmm_compare; blmm_compare.compare('$1','$2','$3')"
