#!/bin/bash
fslpython -c "from blmm.lib import blmm_batch; blmm_batch.batch($1,'$2')"