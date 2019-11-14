#!/bin/bash
fslpython -c "from lib import blm_concat; blm_concat.main('$1')"
