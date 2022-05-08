#!/bin/bash 

# This script is for delivery to Anker project on Apr. 11th

INPUT_SCENE='/home/aidrive/tb5zhh/3d_scene_understand/SUField/debug/scene0518_00.label.ply'
OUTPUT_DIR='debug'

python sufield/spec_cluster.py $INPUT_SCENE $OUTPUT_DIR
