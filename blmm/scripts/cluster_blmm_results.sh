#!/bin/bash
fslpython -c "from blmm.lib import blmm_results; blmm_results.results('$1', '$2')"
