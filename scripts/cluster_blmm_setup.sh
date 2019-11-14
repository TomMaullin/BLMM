#!/bin/bash
fslpython -c "from lib import blm_setup; blm_setup.main('$1')"
