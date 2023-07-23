#!/bin/bash
fslpython -c "from blmm.lib import blmm_setup; blmm_setup.setup('$1')"
