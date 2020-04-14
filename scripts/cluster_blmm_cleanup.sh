#!/bin/bash
fslpython -c "from src import blmm_cleanup; blmm_concat.main('$1')"
