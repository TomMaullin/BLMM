#!/bin/bash
fslpython -c "from src import blmm_concat; blmm_concat.main('$1', '$2')"
