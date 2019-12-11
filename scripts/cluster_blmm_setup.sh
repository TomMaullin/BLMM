#!/bin/bash
fslpython -c "from lib import blmm_setup; blmm_setup.main('$1')"
