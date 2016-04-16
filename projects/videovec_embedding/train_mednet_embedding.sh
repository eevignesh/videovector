#!/usr/bin/env sh

LOG_DIR="/scr/r6/vigneshr/vigneshcaffe/projects/videovec_release/mednet_training_log_dir/"
mkdir $LOG_DIR
rm $LOG_DIR/*
GLOG_log_dir=$LOG_DIR ./.build_debug/tools/caffe.bin train \
  --solver=projects/videovec_release/mednet_embedding_train_solver.prototxt --gpu=3 \
  --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
