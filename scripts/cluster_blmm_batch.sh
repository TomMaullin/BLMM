#!/bin/bash
fslpython -c "from src import blmm_batch; blmm_batch.main($1,'$2')"