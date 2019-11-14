#!/bin/bash
fslpython -c "from lib import blm_batch; blm_batch.main($1,'$2')"