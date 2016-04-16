#!/bin/bash

# arg-1: the pre-trained model produced by our method. In this example, we use the model
#        trained on a TRECVID Devkit videos.
# arg-2: imagenet pre-trained model binary proto file
# arg-3: the prototxt file for our network
# arg-4: output blob name
# arg-5: output directory. We save features both in lmdb format as well as in a text file.
#        In the example below, the results would be saved in the file ./sample_data/sample_features/text_output.txt
# arg-6: Number of batches. The total number of images will be this number times the
#        size of mini-batch specified in the network prototxt file.
# arg-7: GPU/CPU device-id


# --- Full noskipgram with hard-negs, mutli-resoution ----
.build_debug/tools/extract_features.bin \
  ./models/mednet/mednet_embedding_final.caffemodel \
  ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  ./projects/videovec_embedding/videovec_extraction.prototxt ip2 \
  ./sample_data/sample_features/ 1 GPU 2
