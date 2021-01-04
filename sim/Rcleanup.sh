#!/bin/bash
fslpython -c "from sim import cleanup; cleanup.Rcleanup('$1', $2, $3, $4)"