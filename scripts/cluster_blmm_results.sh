#!/bin/bash
fslpython -c "from src import blmm_results; blmm_results.main('$1', '$2')"
