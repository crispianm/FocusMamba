#!/usr/bin/env bash
# Download scripts for common depth estimation datasets.
# Usage: bash tools/download_datasets.sh [dataset_name]
#
# Supported: ddad, nyuv2, scannet, davis

set -euo pipefail

DATASET="${1:-all}"
DATA_DIR="${2:-./datasets}"

mkdir -p "$DATA_DIR"

download_ddad() {
    echo "=== Downloading DDAD ==="
    echo "DDAD requires access from https://github.com/TRI-ML/DDAD"
    echo "Please follow their download instructions and place in $DATA_DIR/ddad/"
}

download_nyuv2() {
    echo "=== Downloading NYUv2 ==="
    echo "Download from: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"
    echo "Place in $DATA_DIR/nyuv2/"
}

download_scannet() {
    echo "=== Downloading ScanNet ==="
    echo "ScanNet requires institutional agreement: http://www.scan-net.org/"
    echo "Place in $DATA_DIR/scannet/"
}

download_davis() {
    echo "=== Downloading DAVIS ==="
    if command -v wget &> /dev/null; then
        wget -P "$DATA_DIR" https://data.vision.ee.ethz.ch/jpont/davis/DAVIS-2017-trainval-480p.zip
        unzip "$DATA_DIR/DAVIS-2017-trainval-480p.zip" -d "$DATA_DIR/davis"
    else
        echo "wget not found. Download manually from:"
        echo "https://data.vision.ee.ethz.ch/jpont/davis/DAVIS-2017-trainval-480p.zip"
    fi
}

case "$DATASET" in
    ddad)    download_ddad ;;
    nyuv2)   download_nyuv2 ;;
    scannet) download_scannet ;;
    davis)   download_davis ;;
    all)
        download_ddad
        download_nyuv2
        download_scannet
        download_davis
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Supported: ddad, nyuv2, scannet, davis, all"
        exit 1
        ;;
esac

echo "Done."
