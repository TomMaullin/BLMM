#!/bin/bash
fslpython -c "from sim import cleanup; cleanup.cleanup('$1', $2)"