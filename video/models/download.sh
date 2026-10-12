#!/usr/bin/env bash
# Fetches the MobileNet-SSD weights (MIT license, chuanqi305/MobileNet-SSD) and OpenCV's sample video.
set -euo pipefail
cd "$(dirname "$0")"
B=https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master
curl -L -o MobileNetSSD_deploy.prototxt $B/deploy.prototxt
curl -L -o MobileNetSSD_deploy.caffemodel $B/mobilenet_iter_73000.caffemodel
curl -L -o ../vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
