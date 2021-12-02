#!/bin/sh

PRETRAINED=$1

python eval-action-recg.py \
  configs/benchmark/ucf/32at16-fold1-r50-4gpu-drop09.yaml configs/main/avid/kinetics/Cross-N1024.yaml \
  --from-moco --ssp-pretrained-weights ${PRETRAINED} \
  --arch resnet50 \
  --name 'cycle-r50-r2v2full-ep200-32at16-4gpu-drop09' \
