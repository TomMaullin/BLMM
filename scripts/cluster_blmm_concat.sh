#!/bin/bash
fslpython -c "from lib import blmm_concat; blmm_concat.main('$1')"
