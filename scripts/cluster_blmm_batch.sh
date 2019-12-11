#!/bin/bash
fslpython -c "from lib import blmm_batch; blmm_batch.main($1,'$2')"