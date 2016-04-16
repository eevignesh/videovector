# Installation

Installation is similar to caffe. Please contact authors of
the paper "Learning temporal embeddings for complex video analysis", ICCV2016
for issues specific to this code.


For general caffe installation instructions:
See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.

Check the issue tracker in case you need help:
https://github.com/BVLC/caffe/issues

Download the project and model files, and unzip them in the caffe root directory (vigneshcaffe_release)
-------------------------------------------------------------------------------------------------------
wget http://vision.stanford.edu/vigneshr_data/ICCV15_models/models.zip
unzip model.zip

To test feature extraction (extract our embedding) on sample data, download sample images and run feature extraction
--------------------------------------------------------------------------------------------------------------------
```shell
wget http://vision.stanford.edu/vigneshr_data/ICCV15_models/sample_data.zip
unzip sample_data.zip
./projects/videovec_embedding/feature_extraction_pretrained_mednet.sh
```

------------------------------------------------------------------------
To train the model, download the data files and unzip them in the
caffe root directory. Also, you need to make appropriate changes
to the path of the data files in the training prototxt file.
------------------------------------------------------------------------
wget http://vision.stanford.edu/vigneshr_data/ICCV15_models/lmdb_files.zip
unzip lmdb_files.zip
