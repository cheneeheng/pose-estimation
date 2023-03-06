#!/bin/sh
echo "IN: $1"
echo "OUT: $2"
echo "FPS: $3"
ffmpeg -framerate $3 -pattern_type glob -i "$1" -c:v libx264 -pix_fmt yuv420p $2

