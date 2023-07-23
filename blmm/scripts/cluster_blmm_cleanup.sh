#!/bin/bash
fslpython -c "from blmm.lib import blmm_cleanup; blmm_cleanup.cleanup('$1')"
