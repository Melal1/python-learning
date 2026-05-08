#!/bin/bash
# run this to update pyrightconfig.json with current nix paths
PATHS=$(.venv/bin/python -c "import sys; import json; print(json.dumps([p for p in sys.path if 'site-packages' in p]))")
echo "{
  \"stubPath\": \"./typings\",
  \"venvPath\": \".\",
  \"venv\": \".venv\",
  \"reportMissingTypeStubs\": \"none\",
  \"typeCheckingMode\": \"basic\",
  \"useLibraryCodeForTypes\": true,
  \"extraPaths\": $PATHS
}" > pyrightconfig.json
echo "pyrightconfig.json updated."
