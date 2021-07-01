#!/bin/bash
fslpython -c "from lmer import cleanup; cleanup.cleanup('$1')"