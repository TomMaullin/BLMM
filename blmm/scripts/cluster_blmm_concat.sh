#!/bin/bash
fslpython -c "from blmm.lib import blmm_concat; blmm_concat.concat('$1')"