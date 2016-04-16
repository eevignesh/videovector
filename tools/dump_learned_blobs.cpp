// This program takes in a trained network and an input blob, and then dumps
// all the intermediate blobs produced by the net to individual binary
// files stored in protobuffer binary formats.
// Usage:
//    dump_network input_net_param trained_net_param
//        input_blob output_prefix 0/1
// if input_net_param is 'none', we will directly load the network from
// trained_net_param. If the last argv is 1, we will do a forward-backward pass
// before dumping everyting, and also dump the who network.

#include <string>
#include <vector>

#include "fcntl.h"
#include "google/protobuf/text_format.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/vignesh_util.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_phase(Caffe::TEST);

  boost::shared_ptr<Net<float> > caffe_net;
  if (strcmp(argv[1], "none") == 0) {
    // We directly load the net param from trained file
    caffe_net.reset(new Net<float>(argv[2]));
  } else {
    caffe_net.reset(new Net<float>(argv[1]));
  }
  caffe_net->CopyTrainedLayersFrom(argv[2]);

  string output_prefix(argv[3]);

  const vector<string>& layer_names = caffe_net->layer_names();
  const vector<boost::shared_ptr<Layer<float> > >& layers = caffe_net->layers();
  
  for (int layerid = 0; layerid < caffe_net->layers().size(); ++layerid) {
    LOG(ERROR) << "Dumping " << layer_names[layerid];
    const vector<boost::shared_ptr<Blob<float> > >& blobs = layers[layerid]->blobs();
    for (int blobid = 0; blobid < blobs.size(); ++blobid) {
      // Serialize blob
      LOG(ERROR) << "Dumping blob: " << blobid;
      BlobProto output_blob_proto;
      blobs[blobid]->ToProto(&output_blob_proto);
      string output_name = output_prefix + layer_names[layerid] + "_blob_" +
        stringprintf("%d.prototxt", blobid);
      WriteProtoToBinaryFile(output_blob_proto,
          output_name);
    }
   
  }

  return 0;
}
