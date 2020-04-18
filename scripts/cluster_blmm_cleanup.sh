#!/bin/bash
fslpython -c "from src import blmm_cleanup; blmm_cleanup.main('$1')"
