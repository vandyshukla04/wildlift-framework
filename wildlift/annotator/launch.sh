#!/bin/bash

# Launch WildLIFT Annotation Tool (wildlift-a)
# Usage: bash wildlift/annotator/launch.sh --auto_bboxes <bbox_dir> --images <image_dir> --output <output_dir>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=================================================="
echo "WildLIFT Annotation Tool (wildlift-a)"
echo "=================================================="
echo ""

# Default port
PORT=8080

# Pass all arguments through to the bbox editor
python -m wildlift.annotator.bbox_editor "$@" --port $PORT

echo ""
echo "=================================================="
echo "Annotation session complete!"
echo "=================================================="
