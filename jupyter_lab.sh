#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100
