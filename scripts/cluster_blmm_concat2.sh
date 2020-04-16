#!/bin/bash
fslpython -c "from src import blmm_concat2; blmm_concat2.main('$1', '$2')"