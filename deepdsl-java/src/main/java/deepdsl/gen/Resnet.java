package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;


public class Resnet {
// comment the line below for memory efficient mode
 static{ JCudaTensor.enableMemoryCache();}
// decay_1
static float decay_1 = 0.999995f;
// lrn_rate_1
static float lrn_rate_1 = -0.01f;
// momentum
static float momentum = 0.9f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/resnet";
// platform
static LmdbUtils.OS platform = LmdbUtils.OS.WINDOWS;
// test_data_path
static String test_data_path = "dataset/imagenet/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 10;
// test_size
static int test_size = 10000;
// train_data_path
static String train_data_path = "dataset/imagenet/ilsvrc12_train_lmdb";
// train_itr
static int train_itr = 1000;
// train_size
static int train_size = 1000000;

// (BatchNorm(1_bn),List(List(64, 64, 112, 112), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x27 = new JCudnnBatchNorm(network_dir + "/1_bn", new int[]{64,64,112,112});
// (BatchNorm(2a1_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x54 = new JCudnnBatchNorm(network_dir + "/2a1_bn", new int[]{64,256,55,55});
// (BatchNorm(2a2_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x61 = new JCudnnBatchNorm(network_dir + "/2a2_a_bn", new int[]{64,64,55,55});
// (BatchNorm(2a2_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x78 = new JCudnnBatchNorm(network_dir + "/2a2_b_bn", new int[]{64,64,55,55});
// (BatchNorm(2a2_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x93 = new JCudnnBatchNorm(network_dir + "/2a2_c_bn", new int[]{64,256,55,55});
// (BatchNorm(2b_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x117 = new JCudnnBatchNorm(network_dir + "/2b_a_bn", new int[]{64,64,55,55});
// (BatchNorm(2b_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x132 = new JCudnnBatchNorm(network_dir + "/2b_b_bn", new int[]{64,64,55,55});
// (BatchNorm(2b_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x147 = new JCudnnBatchNorm(network_dir + "/2b_c_bn", new int[]{64,256,55,55});
// (BatchNorm(2c_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x167 = new JCudnnBatchNorm(network_dir + "/2c_a_bn", new int[]{64,64,55,55});
// (BatchNorm(2c_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x182 = new JCudnnBatchNorm(network_dir + "/2c_b_bn", new int[]{64,64,55,55});
// (BatchNorm(2c_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x197 = new JCudnnBatchNorm(network_dir + "/2c_c_bn", new int[]{64,256,55,55});
// (BatchNorm(3a1_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x232 = new JCudnnBatchNorm(network_dir + "/3a1_bn", new int[]{64,512,28,28});
// (BatchNorm(3a2_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x225 = new JCudnnBatchNorm(network_dir + "/3a2_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3a2_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x249 = new JCudnnBatchNorm(network_dir + "/3a2_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3a2_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x265 = new JCudnnBatchNorm(network_dir + "/3a2_c_bn", new int[]{64,512,28,28});
// (BatchNorm(3b_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x289 = new JCudnnBatchNorm(network_dir + "/3b_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3b_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x304 = new JCudnnBatchNorm(network_dir + "/3b_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3b_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x319 = new JCudnnBatchNorm(network_dir + "/3b_c_bn", new int[]{64,512,28,28});
// (BatchNorm(3c_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x339 = new JCudnnBatchNorm(network_dir + "/3c_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3c_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x354 = new JCudnnBatchNorm(network_dir + "/3c_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3c_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x369 = new JCudnnBatchNorm(network_dir + "/3c_c_bn", new int[]{64,512,28,28});
// (BatchNorm(3d_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x389 = new JCudnnBatchNorm(network_dir + "/3d_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3d_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x404 = new JCudnnBatchNorm(network_dir + "/3d_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3d_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x419 = new JCudnnBatchNorm(network_dir + "/3d_c_bn", new int[]{64,512,28,28});
// (BatchNorm(4a1_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x447 = new JCudnnBatchNorm(network_dir + "/4a1_bn", new int[]{64,1024,14,14});
// (BatchNorm(4a2_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x454 = new JCudnnBatchNorm(network_dir + "/4a2_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4a2_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x471 = new JCudnnBatchNorm(network_dir + "/4a2_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4a2_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x487 = new JCudnnBatchNorm(network_dir + "/4a2_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4b_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x511 = new JCudnnBatchNorm(network_dir + "/4b_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4b_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x526 = new JCudnnBatchNorm(network_dir + "/4b_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4b_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x541 = new JCudnnBatchNorm(network_dir + "/4b_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4c_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x561 = new JCudnnBatchNorm(network_dir + "/4c_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4c_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x576 = new JCudnnBatchNorm(network_dir + "/4c_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4c_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x591 = new JCudnnBatchNorm(network_dir + "/4c_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4d_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x611 = new JCudnnBatchNorm(network_dir + "/4d_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4d_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x626 = new JCudnnBatchNorm(network_dir + "/4d_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4d_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x641 = new JCudnnBatchNorm(network_dir + "/4d_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4e_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x661 = new JCudnnBatchNorm(network_dir + "/4e_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4e_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x676 = new JCudnnBatchNorm(network_dir + "/4e_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4e_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x691 = new JCudnnBatchNorm(network_dir + "/4e_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4f_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x711 = new JCudnnBatchNorm(network_dir + "/4f_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4f_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x726 = new JCudnnBatchNorm(network_dir + "/4f_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4f_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x741 = new JCudnnBatchNorm(network_dir + "/4f_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(5a1_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x769 = new JCudnnBatchNorm(network_dir + "/5a1_bn", new int[]{64,2048,7,7});
// (BatchNorm(5a2_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x776 = new JCudnnBatchNorm(network_dir + "/5a2_a_bn", new int[]{64,512,7,7});
// (BatchNorm(5a2_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x793 = new JCudnnBatchNorm(network_dir + "/5a2_b_bn", new int[]{64,512,7,7});
// (BatchNorm(5a2_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x809 = new JCudnnBatchNorm(network_dir + "/5a2_c_bn", new int[]{64,2048,7,7});
// (BatchNorm(5b_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x833 = new JCudnnBatchNorm(network_dir + "/5b_a_bn", new int[]{64,512,7,7});
// (BatchNorm(5b_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x848 = new JCudnnBatchNorm(network_dir + "/5b_b_bn", new int[]{64,512,7,7});
// (BatchNorm(5b_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x863 = new JCudnnBatchNorm(network_dir + "/5b_c_bn", new int[]{64,2048,7,7});
// (BatchNorm(5c_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x883 = new JCudnnBatchNorm(network_dir + "/5c_a_bn", new int[]{64,512,7,7});
// (BatchNorm(5c_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x898 = new JCudnnBatchNorm(network_dir + "/5c_b_bn", new int[]{64,512,7,7});
// (BatchNorm(5c_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x913 = new JCudnnBatchNorm(network_dir + "/5c_c_bn", new int[]{64,2048,7,7});
// (Convolv(1,0),List(List(64, 1024, 14, 14), List(256, 1024, 1, 1), List(256)))
static JCudnnConvolution x504 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(List(64, 128, 28, 28), List(512, 128, 1, 1), List(512)))
static JCudnnConvolution x258 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(List(64, 2048, 7, 7), List(512, 2048, 1, 1), List(512)))
static JCudnnConvolution x826 = new JCudnnConvolution(new int[]{64,2048,7,7},new int[]{512,2048,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(List(64, 256, 14, 14), List(1024, 256, 1, 1), List(1024)))
static JCudnnConvolution x480 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(List(64, 256, 55, 55), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x110 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(64, 512, 28, 28), List(128, 512, 1, 1), List(128)))
static JCudnnConvolution x282 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{128,512,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(List(64, 512, 7, 7), List(2048, 512, 1, 1), List(2048)))
static JCudnnConvolution x802 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{2048,512,1,1},new int[]{2048}, 1, 0);
// (Convolv(1,0),List(List(64, 64, 55, 55), List(256, 64, 1, 1), List(256)))
static JCudnnConvolution x40 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(List(64, 64, 55, 55), List(64, 64, 1, 1), List(64)))
static JCudnnConvolution x47 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,1,1},new int[]{64}, 1, 0);
// (Convolv(1,1),List(List(64, 128, 28, 28), List(128, 128, 3, 3), List(128)))
static JCudnnConvolution x242 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(64, 256, 14, 14), List(256, 256, 3, 3), List(256)))
static JCudnnConvolution x464 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(List(64, 512, 7, 7), List(512, 512, 3, 3), List(512)))
static JCudnnConvolution x786 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(List(64, 64, 55, 55), List(64, 64, 3, 3), List(64)))
static JCudnnConvolution x71 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,3,3},new int[]{64}, 1, 1);
// (Convolv(2,0),List(List(64, 1024, 14, 14), List(2048, 1024, 1, 1), List(2048)))
static JCudnnConvolution x755 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{2048,1024,1,1},new int[]{2048}, 2, 0);
// (Convolv(2,0),List(List(64, 1024, 14, 14), List(512, 1024, 1, 1), List(512)))
static JCudnnConvolution x762 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{512,1024,1,1},new int[]{512}, 2, 0);
// (Convolv(2,0),List(List(64, 256, 55, 55), List(128, 256, 1, 1), List(128)))
static JCudnnConvolution x211 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{128,256,1,1},new int[]{128}, 2, 0);
// (Convolv(2,0),List(List(64, 256, 55, 55), List(512, 256, 1, 1), List(512)))
static JCudnnConvolution x218 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{512,256,1,1},new int[]{512}, 2, 0);
// (Convolv(2,0),List(List(64, 512, 28, 28), List(1024, 512, 1, 1), List(1024)))
static JCudnnConvolution x440 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{1024,512,1,1},new int[]{1024}, 2, 0);
// (Convolv(2,0),List(List(64, 512, 28, 28), List(256, 512, 1, 1), List(256)))
static JCudnnConvolution x433 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{256,512,1,1},new int[]{256}, 2, 0);
// (Convolv(2,3),List(List(64, 3, 224, 224), List(64, 3, 7, 7), List(64)))
static JCudnnConvolution x20 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
// (Lmdb(1000000,10000,Win32,1000),false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{64, 3, 224, 224}, platform, 1000, true);
// (Lmdb(1000000,10000,Win32,1000),true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{64, 3, 224, 224}, platform, 1000, false);
// (LogSoftmax(),List(List(64, 1000)))
static JCudnnSoftmax x937 = new JCudnnSoftmax(new int[]{64,1000}, SoftmaxAlgorithm.LOG);
// (Pooling(3,2,0,true),List(List(64, 64, 112, 112)))
static JCudnnPooling x33 = new JCudnnPooling(new int[]{64,64,112,112}, 3, 2, 0, PoolingType.MAX);
// (Pooling(7,1,0,false),List(List(64, 2048, 7, 7)))
static JCudnnPooling x923 = new JCudnnPooling(new int[]{64,2048,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
// (ReLU(),List(List(64, 1024, 14, 14)))
static JCudnnActivation x490 = new JCudnnActivation(new int[]{64,1024,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(64, 128, 28, 28)))
static JCudnnActivation x235 = new JCudnnActivation(new int[]{64,128,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(64, 2048, 7, 7)))
static JCudnnActivation x812 = new JCudnnActivation(new int[]{64,2048,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(64, 256, 14, 14)))
static JCudnnActivation x457 = new JCudnnActivation(new int[]{64,256,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(64, 256, 55, 55)))
static JCudnnActivation x96 = new JCudnnActivation(new int[]{64,256,55,55}, ActivationMode.RELU);
// (ReLU(),List(List(64, 512, 28, 28)))
static JCudnnActivation x268 = new JCudnnActivation(new int[]{64,512,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(64, 512, 7, 7)))
static JCudnnActivation x779 = new JCudnnActivation(new int[]{64,512,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(64, 64, 112, 112)))
static JCudnnActivation x30 = new JCudnnActivation(new int[]{64,64,112,112}, ActivationMode.RELU);
// (ReLU(),List(List(64, 64, 55, 55)))
static JCudnnActivation x64 = new JCudnnActivation(new int[]{64,64,55,55}, ActivationMode.RELU);
// 1_bn_bias
static JCudaTensor x26 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/1_bn_bias").asJCudaTensor();
// 1_bn_scale
static JCudaTensor x25 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/1_bn_scale").asJCudaTensor();
// 1_cv_B
static JCudaTensor x19 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 1_cv_W
static JCudaTensor x18 = JTensor.randomFloat(-0.11664237f, 0.11664237f, 64, 3, 7, 7).load(network_dir + "/1_cv_W").asJCudaTensor();
// 2a1_bn_bias
static JCudaTensor x53 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2a1_bn_bias").asJCudaTensor();
// 2a1_bn_scale
static JCudaTensor x52 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2a1_bn_scale").asJCudaTensor();
// 2a1_cv_B
static JCudaTensor x39 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2a1_cv_W
static JCudaTensor x38 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2a1_cv_W").asJCudaTensor();
// 2a2_a_bn_bias
static JCudaTensor x60 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2a2_a_bn_bias").asJCudaTensor();
// 2a2_a_bn_scale
static JCudaTensor x59 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2a2_a_bn_scale").asJCudaTensor();
// 2a2_a_cv_B
static JCudaTensor x46 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2a2_a_cv_W
static JCudaTensor x45 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/2a2_a_cv_W").asJCudaTensor();
// 2a2_b_bn_bias
static JCudaTensor x77 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2a2_b_bn_bias").asJCudaTensor();
// 2a2_b_bn_scale
static JCudaTensor x76 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2a2_b_bn_scale").asJCudaTensor();
// 2a2_b_cv_B
static JCudaTensor x70 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2a2_b_cv_W
static JCudaTensor x69 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2a2_b_cv_W").asJCudaTensor();
// 2a2_c_bn_bias
static JCudaTensor x92 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_bias").asJCudaTensor();
// 2a2_c_bn_scale
static JCudaTensor x91 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_scale").asJCudaTensor();
// 2a2_c_cv_B
static JCudaTensor x86 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2a2_c_cv_W
static JCudaTensor x85 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2a2_c_cv_W").asJCudaTensor();
// 2b_a_bn_bias
static JCudaTensor x116 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_bias").asJCudaTensor();
// 2b_a_bn_scale
static JCudaTensor x115 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_scale").asJCudaTensor();
// 2b_a_cv_B
static JCudaTensor x109 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2b_a_cv_W
static JCudaTensor x108 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2b_a_cv_W").asJCudaTensor();
// 2b_b_bn_bias
static JCudaTensor x131 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_bias").asJCudaTensor();
// 2b_b_bn_scale
static JCudaTensor x130 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_scale").asJCudaTensor();
// 2b_b_cv_B
static JCudaTensor x125 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2b_b_cv_W
static JCudaTensor x124 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2b_b_cv_W").asJCudaTensor();
// 2b_c_bn_bias
static JCudaTensor x146 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_bias").asJCudaTensor();
// 2b_c_bn_scale
static JCudaTensor x145 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_scale").asJCudaTensor();
// 2b_c_cv_B
static JCudaTensor x140 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2b_c_cv_W
static JCudaTensor x139 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2b_c_cv_W").asJCudaTensor();
// 2c_a_bn_bias
static JCudaTensor x166 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_bias").asJCudaTensor();
// 2c_a_bn_scale
static JCudaTensor x165 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_scale").asJCudaTensor();
// 2c_a_cv_B
static JCudaTensor x160 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2c_a_cv_W
static JCudaTensor x159 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2c_a_cv_W").asJCudaTensor();
// 2c_b_bn_bias
static JCudaTensor x181 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_bias").asJCudaTensor();
// 2c_b_bn_scale
static JCudaTensor x180 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_scale").asJCudaTensor();
// 2c_b_cv_B
static JCudaTensor x175 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2c_b_cv_W
static JCudaTensor x174 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2c_b_cv_W").asJCudaTensor();
// 2c_c_bn_bias
static JCudaTensor x196 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_bias").asJCudaTensor();
// 2c_c_bn_scale
static JCudaTensor x195 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_scale").asJCudaTensor();
// 2c_c_cv_B
static JCudaTensor x190 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2c_c_cv_W
static JCudaTensor x189 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2c_c_cv_W").asJCudaTensor();
// 3a1_bn_bias
static JCudaTensor x231 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_bias").asJCudaTensor();
// 3a1_bn_scale
static JCudaTensor x230 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_scale").asJCudaTensor();
// 3a1_cv_B
static JCudaTensor x217 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3a1_cv_W
static JCudaTensor x216 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 512, 256, 1, 1).load(network_dir + "/3a1_cv_W").asJCudaTensor();
// 3a2_a_bn_bias
static JCudaTensor x224 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_bias").asJCudaTensor();
// 3a2_a_bn_scale
static JCudaTensor x223 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_scale").asJCudaTensor();
// 3a2_a_cv_B
static JCudaTensor x210 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3a2_a_cv_W
static JCudaTensor x209 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/3a2_a_cv_W").asJCudaTensor();
// 3a2_b_bn_bias
static JCudaTensor x248 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_bias").asJCudaTensor();
// 3a2_b_bn_scale
static JCudaTensor x247 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_scale").asJCudaTensor();
// 3a2_b_cv_B
static JCudaTensor x241 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3a2_b_cv_W
static JCudaTensor x240 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3a2_b_cv_W").asJCudaTensor();
// 3a2_c_bn_bias
static JCudaTensor x264 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_bias").asJCudaTensor();
// 3a2_c_bn_scale
static JCudaTensor x263 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_scale").asJCudaTensor();
// 3a2_c_cv_B
static JCudaTensor x257 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3a2_c_cv_W
static JCudaTensor x256 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3a2_c_cv_W").asJCudaTensor();
// 3b_a_bn_bias
static JCudaTensor x288 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_bias").asJCudaTensor();
// 3b_a_bn_scale
static JCudaTensor x287 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_scale").asJCudaTensor();
// 3b_a_cv_B
static JCudaTensor x281 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3b_a_cv_W
static JCudaTensor x280 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3b_a_cv_W").asJCudaTensor();
// 3b_b_bn_bias
static JCudaTensor x303 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_bias").asJCudaTensor();
// 3b_b_bn_scale
static JCudaTensor x302 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_scale").asJCudaTensor();
// 3b_b_cv_B
static JCudaTensor x297 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3b_b_cv_W
static JCudaTensor x296 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3b_b_cv_W").asJCudaTensor();
// 3b_c_bn_bias
static JCudaTensor x318 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_bias").asJCudaTensor();
// 3b_c_bn_scale
static JCudaTensor x317 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_scale").asJCudaTensor();
// 3b_c_cv_B
static JCudaTensor x312 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3b_c_cv_W
static JCudaTensor x311 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3b_c_cv_W").asJCudaTensor();
// 3c_a_bn_bias
static JCudaTensor x338 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_bias").asJCudaTensor();
// 3c_a_bn_scale
static JCudaTensor x337 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_scale").asJCudaTensor();
// 3c_a_cv_B
static JCudaTensor x332 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3c_a_cv_W
static JCudaTensor x331 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3c_a_cv_W").asJCudaTensor();
// 3c_b_bn_bias
static JCudaTensor x353 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_bias").asJCudaTensor();
// 3c_b_bn_scale
static JCudaTensor x352 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_scale").asJCudaTensor();
// 3c_b_cv_B
static JCudaTensor x347 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3c_b_cv_W
static JCudaTensor x346 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3c_b_cv_W").asJCudaTensor();
// 3c_c_bn_bias
static JCudaTensor x368 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_bias").asJCudaTensor();
// 3c_c_bn_scale
static JCudaTensor x367 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_scale").asJCudaTensor();
// 3c_c_cv_B
static JCudaTensor x362 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3c_c_cv_W
static JCudaTensor x361 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3c_c_cv_W").asJCudaTensor();
// 3d_a_bn_bias
static JCudaTensor x388 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_bias").asJCudaTensor();
// 3d_a_bn_scale
static JCudaTensor x387 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_scale").asJCudaTensor();
// 3d_a_cv_B
static JCudaTensor x382 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3d_a_cv_W
static JCudaTensor x381 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3d_a_cv_W").asJCudaTensor();
// 3d_b_bn_bias
static JCudaTensor x403 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_bias").asJCudaTensor();
// 3d_b_bn_scale
static JCudaTensor x402 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_scale").asJCudaTensor();
// 3d_b_cv_B
static JCudaTensor x397 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3d_b_cv_W
static JCudaTensor x396 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3d_b_cv_W").asJCudaTensor();
// 3d_c_bn_bias
static JCudaTensor x418 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_bias").asJCudaTensor();
// 3d_c_bn_scale
static JCudaTensor x417 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_scale").asJCudaTensor();
// 3d_c_cv_B
static JCudaTensor x412 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3d_c_cv_W
static JCudaTensor x411 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3d_c_cv_W").asJCudaTensor();
// 4a1_bn_bias
static JCudaTensor x446 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_bias").asJCudaTensor();
// 4a1_bn_scale
static JCudaTensor x445 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_scale").asJCudaTensor();
// 4a1_cv_B
static JCudaTensor x439 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4a1_cv_W
static JCudaTensor x438 = JTensor.randomFloat(-0.0625f, 0.0625f, 1024, 512, 1, 1).load(network_dir + "/4a1_cv_W").asJCudaTensor();
// 4a2_a_bn_bias
static JCudaTensor x453 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_bias").asJCudaTensor();
// 4a2_a_bn_scale
static JCudaTensor x452 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_scale").asJCudaTensor();
// 4a2_a_cv_B
static JCudaTensor x432 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4a2_a_cv_W
static JCudaTensor x431 = JTensor.randomFloat(-0.0625f, 0.0625f, 256, 512, 1, 1).load(network_dir + "/4a2_a_cv_W").asJCudaTensor();
// 4a2_b_bn_bias
static JCudaTensor x470 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_bias").asJCudaTensor();
// 4a2_b_bn_scale
static JCudaTensor x469 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_scale").asJCudaTensor();
// 4a2_b_cv_B
static JCudaTensor x463 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4a2_b_cv_W
static JCudaTensor x462 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4a2_b_cv_W").asJCudaTensor();
// 4a2_c_bn_bias
static JCudaTensor x486 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_bias").asJCudaTensor();
// 4a2_c_bn_scale
static JCudaTensor x485 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_scale").asJCudaTensor();
// 4a2_c_cv_B
static JCudaTensor x479 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4a2_c_cv_W
static JCudaTensor x478 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4a2_c_cv_W").asJCudaTensor();
// 4b_a_bn_bias
static JCudaTensor x510 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_bias").asJCudaTensor();
// 4b_a_bn_scale
static JCudaTensor x509 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_scale").asJCudaTensor();
// 4b_a_cv_B
static JCudaTensor x503 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4b_a_cv_W
static JCudaTensor x502 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4b_a_cv_W").asJCudaTensor();
// 4b_b_bn_bias
static JCudaTensor x525 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_bias").asJCudaTensor();
// 4b_b_bn_scale
static JCudaTensor x524 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_scale").asJCudaTensor();
// 4b_b_cv_B
static JCudaTensor x519 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4b_b_cv_W
static JCudaTensor x518 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4b_b_cv_W").asJCudaTensor();
// 4b_c_bn_bias
static JCudaTensor x540 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_bias").asJCudaTensor();
// 4b_c_bn_scale
static JCudaTensor x539 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_scale").asJCudaTensor();
// 4b_c_cv_B
static JCudaTensor x534 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4b_c_cv_W
static JCudaTensor x533 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4b_c_cv_W").asJCudaTensor();
// 4c_a_bn_bias
static JCudaTensor x560 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_bias").asJCudaTensor();
// 4c_a_bn_scale
static JCudaTensor x559 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_scale").asJCudaTensor();
// 4c_a_cv_B
static JCudaTensor x554 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4c_a_cv_W
static JCudaTensor x553 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4c_a_cv_W").asJCudaTensor();
// 4c_b_bn_bias
static JCudaTensor x575 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_bias").asJCudaTensor();
// 4c_b_bn_scale
static JCudaTensor x574 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_scale").asJCudaTensor();
// 4c_b_cv_B
static JCudaTensor x569 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4c_b_cv_W
static JCudaTensor x568 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4c_b_cv_W").asJCudaTensor();
// 4c_c_bn_bias
static JCudaTensor x590 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_bias").asJCudaTensor();
// 4c_c_bn_scale
static JCudaTensor x589 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_scale").asJCudaTensor();
// 4c_c_cv_B
static JCudaTensor x584 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4c_c_cv_W
static JCudaTensor x583 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4c_c_cv_W").asJCudaTensor();
// 4d_a_bn_bias
static JCudaTensor x610 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_bias").asJCudaTensor();
// 4d_a_bn_scale
static JCudaTensor x609 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_scale").asJCudaTensor();
// 4d_a_cv_B
static JCudaTensor x604 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4d_a_cv_W
static JCudaTensor x603 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4d_a_cv_W").asJCudaTensor();
// 4d_b_bn_bias
static JCudaTensor x625 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_bias").asJCudaTensor();
// 4d_b_bn_scale
static JCudaTensor x624 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_scale").asJCudaTensor();
// 4d_b_cv_B
static JCudaTensor x619 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4d_b_cv_W
static JCudaTensor x618 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4d_b_cv_W").asJCudaTensor();
// 4d_c_bn_bias
static JCudaTensor x640 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_bias").asJCudaTensor();
// 4d_c_bn_scale
static JCudaTensor x639 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_scale").asJCudaTensor();
// 4d_c_cv_B
static JCudaTensor x634 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4d_c_cv_W
static JCudaTensor x633 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4d_c_cv_W").asJCudaTensor();
// 4e_a_bn_bias
static JCudaTensor x660 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_bias").asJCudaTensor();
// 4e_a_bn_scale
static JCudaTensor x659 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_scale").asJCudaTensor();
// 4e_a_cv_B
static JCudaTensor x654 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4e_a_cv_W
static JCudaTensor x653 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4e_a_cv_W").asJCudaTensor();
// 4e_b_bn_bias
static JCudaTensor x675 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_bias").asJCudaTensor();
// 4e_b_bn_scale
static JCudaTensor x674 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_scale").asJCudaTensor();
// 4e_b_cv_B
static JCudaTensor x669 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4e_b_cv_W
static JCudaTensor x668 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4e_b_cv_W").asJCudaTensor();
// 4e_c_bn_bias
static JCudaTensor x690 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_bias").asJCudaTensor();
// 4e_c_bn_scale
static JCudaTensor x689 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_scale").asJCudaTensor();
// 4e_c_cv_B
static JCudaTensor x684 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4e_c_cv_W
static JCudaTensor x683 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4e_c_cv_W").asJCudaTensor();
// 4f_a_bn_bias
static JCudaTensor x710 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_bias").asJCudaTensor();
// 4f_a_bn_scale
static JCudaTensor x709 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_scale").asJCudaTensor();
// 4f_a_cv_B
static JCudaTensor x704 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4f_a_cv_W
static JCudaTensor x703 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4f_a_cv_W").asJCudaTensor();
// 4f_b_bn_bias
static JCudaTensor x725 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_bias").asJCudaTensor();
// 4f_b_bn_scale
static JCudaTensor x724 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_scale").asJCudaTensor();
// 4f_b_cv_B
static JCudaTensor x719 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4f_b_cv_W
static JCudaTensor x718 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4f_b_cv_W").asJCudaTensor();
// 4f_c_bn_bias
static JCudaTensor x740 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_bias").asJCudaTensor();
// 4f_c_bn_scale
static JCudaTensor x739 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_scale").asJCudaTensor();
// 4f_c_cv_B
static JCudaTensor x734 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4f_c_cv_W
static JCudaTensor x733 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4f_c_cv_W").asJCudaTensor();
// 5a1_bn_bias
static JCudaTensor x768 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_bias").asJCudaTensor();
// 5a1_bn_scale
static JCudaTensor x767 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_scale").asJCudaTensor();
// 5a1_cv_B
static JCudaTensor x754 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5a1_cv_W
static JCudaTensor x753 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 2048, 1024, 1, 1).load(network_dir + "/5a1_cv_W").asJCudaTensor();
// 5a2_a_bn_bias
static JCudaTensor x775 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_bias").asJCudaTensor();
// 5a2_a_bn_scale
static JCudaTensor x774 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_scale").asJCudaTensor();
// 5a2_a_cv_B
static JCudaTensor x761 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5a2_a_cv_W
static JCudaTensor x760 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 512, 1024, 1, 1).load(network_dir + "/5a2_a_cv_W").asJCudaTensor();
// 5a2_b_bn_bias
static JCudaTensor x792 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_bias").asJCudaTensor();
// 5a2_b_bn_scale
static JCudaTensor x791 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_scale").asJCudaTensor();
// 5a2_b_cv_B
static JCudaTensor x785 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5a2_b_cv_W
static JCudaTensor x784 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5a2_b_cv_W").asJCudaTensor();
// 5a2_c_bn_bias
static JCudaTensor x808 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_bias").asJCudaTensor();
// 5a2_c_bn_scale
static JCudaTensor x807 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_scale").asJCudaTensor();
// 5a2_c_cv_B
static JCudaTensor x801 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5a2_c_cv_W
static JCudaTensor x800 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5a2_c_cv_W").asJCudaTensor();
// 5b_a_bn_bias
static JCudaTensor x832 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_bias").asJCudaTensor();
// 5b_a_bn_scale
static JCudaTensor x831 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_scale").asJCudaTensor();
// 5b_a_cv_B
static JCudaTensor x825 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5b_a_cv_W
static JCudaTensor x824 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5b_a_cv_W").asJCudaTensor();
// 5b_b_bn_bias
static JCudaTensor x847 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_bias").asJCudaTensor();
// 5b_b_bn_scale
static JCudaTensor x846 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_scale").asJCudaTensor();
// 5b_b_cv_B
static JCudaTensor x841 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5b_b_cv_W
static JCudaTensor x840 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5b_b_cv_W").asJCudaTensor();
// 5b_c_bn_bias
static JCudaTensor x862 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_bias").asJCudaTensor();
// 5b_c_bn_scale
static JCudaTensor x861 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_scale").asJCudaTensor();
// 5b_c_cv_B
static JCudaTensor x856 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5b_c_cv_W
static JCudaTensor x855 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5b_c_cv_W").asJCudaTensor();
// 5c_a_bn_bias
static JCudaTensor x882 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_bias").asJCudaTensor();
// 5c_a_bn_scale
static JCudaTensor x881 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_scale").asJCudaTensor();
// 5c_a_cv_B
static JCudaTensor x876 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5c_a_cv_W
static JCudaTensor x875 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5c_a_cv_W").asJCudaTensor();
// 5c_b_bn_bias
static JCudaTensor x897 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_bias").asJCudaTensor();
// 5c_b_bn_scale
static JCudaTensor x896 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_scale").asJCudaTensor();
// 5c_b_cv_B
static JCudaTensor x891 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5c_b_cv_W
static JCudaTensor x890 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5c_b_cv_W").asJCudaTensor();
// 5c_c_bn_bias
static JCudaTensor x912 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_bias").asJCudaTensor();
// 5c_c_bn_scale
static JCudaTensor x911 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_scale").asJCudaTensor();
// 5c_c_cv_B
static JCudaTensor x906 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5c_c_cv_W
static JCudaTensor x905 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5c_c_cv_W").asJCudaTensor();
// BatchSum(((Sum(X209317) / |64|) / 10))
static float x4302;
// V_1_bn_bias
static JCudaTensor x3525 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_1_bn_scale
static JCudaTensor x3530 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_1_cv_W
static JCudaTensor x3535 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
// V_2a1_bn_bias
static JCudaTensor x3372 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a1_bn_scale
static JCudaTensor x3377 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a1_cv_W
static JCudaTensor x3367 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_2a2_a_bn_bias
static JCudaTensor x3487 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_a_bn_scale
static JCudaTensor x3482 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_a_cv_W
static JCudaTensor x3476 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
// V_2a2_b_bn_bias
static JCudaTensor x3426 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_b_bn_scale
static JCudaTensor x3440 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_b_cv_W
static JCudaTensor x3434 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
// V_2a2_c_bn_bias
static JCudaTensor x3357 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a2_c_bn_scale
static JCudaTensor x3362 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a2_c_cv_W
static JCudaTensor x3352 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_2b_a_bn_bias
static JCudaTensor x3293 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_a_bn_scale
static JCudaTensor x3279 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_a_cv_W
static JCudaTensor x3287 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_2b_b_bn_bias
static JCudaTensor x3241 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_b_bn_scale
static JCudaTensor x3233 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_b_cv_W
static JCudaTensor x3246 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
// V_2b_c_bn_bias
static JCudaTensor x3195 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2b_c_bn_scale
static JCudaTensor x3187 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2b_c_cv_W
static JCudaTensor x3200 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_2c_a_bn_bias
static JCudaTensor x3147 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_a_bn_scale
static JCudaTensor x3133 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_a_cv_W
static JCudaTensor x3141 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_2c_b_bn_bias
static JCudaTensor x3096 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_b_bn_scale
static JCudaTensor x3101 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_b_cv_W
static JCudaTensor x3090 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
// V_2c_c_bn_bias
static JCudaTensor x3044 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2c_c_bn_scale
static JCudaTensor x3049 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2c_c_cv_W
static JCudaTensor x3054 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_3a1_bn_bias
static JCudaTensor x2870 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a1_bn_scale
static JCudaTensor x2889 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a1_cv_W
static JCudaTensor x2875 = JTensor.constFloat(0.0f, 512, 256, 1, 1).asJCudaTensor();
// V_3a2_a_bn_bias
static JCudaTensor x3005 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_a_bn_scale
static JCudaTensor x3000 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_a_cv_W
static JCudaTensor x2994 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_3a2_b_bn_bias
static JCudaTensor x2944 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_b_bn_scale
static JCudaTensor x2949 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_b_cv_W
static JCudaTensor x2957 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3a2_c_bn_bias
static JCudaTensor x2894 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a2_c_bn_scale
static JCudaTensor x2899 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a2_c_cv_W
static JCudaTensor x2880 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_3b_a_bn_bias
static JCudaTensor x2802 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_a_bn_scale
static JCudaTensor x2797 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_a_cv_W
static JCudaTensor x2810 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
// V_3b_b_bn_bias
static JCudaTensor x2756 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_b_bn_scale
static JCudaTensor x2751 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_b_cv_W
static JCudaTensor x2764 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3b_c_bn_bias
static JCudaTensor x2705 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3b_c_bn_scale
static JCudaTensor x2710 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3b_c_cv_W
static JCudaTensor x2718 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_3c_a_bn_bias
static JCudaTensor x2660 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_a_bn_scale
static JCudaTensor x2665 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_a_cv_W
static JCudaTensor x2654 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
// V_3c_b_bn_bias
static JCudaTensor x2619 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_b_bn_scale
static JCudaTensor x2608 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_b_cv_W
static JCudaTensor x2613 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3c_c_bn_bias
static JCudaTensor x2559 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3c_c_bn_scale
static JCudaTensor x2573 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3c_c_cv_W
static JCudaTensor x2567 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_3d_a_bn_bias
static JCudaTensor x2513 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_a_bn_scale
static JCudaTensor x2505 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_a_cv_W
static JCudaTensor x2518 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
// V_3d_b_bn_bias
static JCudaTensor x2462 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_b_bn_scale
static JCudaTensor x2467 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_b_cv_W
static JCudaTensor x2472 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3d_c_bn_bias
static JCudaTensor x2422 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3d_c_bn_scale
static JCudaTensor x2427 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3d_c_cv_W
static JCudaTensor x2416 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_4a1_bn_bias
static JCudaTensor x2248 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a1_bn_scale
static JCudaTensor x2275 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a1_cv_W
static JCudaTensor x2253 = JTensor.constFloat(0.0f, 1024, 512, 1, 1).asJCudaTensor();
// V_4a2_a_bn_bias
static JCudaTensor x2372 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_a_bn_scale
static JCudaTensor x2377 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_a_cv_W
static JCudaTensor x2366 = JTensor.constFloat(0.0f, 256, 512, 1, 1).asJCudaTensor();
// V_4a2_b_bn_bias
static JCudaTensor x2330 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_b_bn_scale
static JCudaTensor x2325 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_b_cv_W
static JCudaTensor x2319 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4a2_c_bn_bias
static JCudaTensor x2259 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a2_c_bn_scale
static JCudaTensor x2270 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a2_c_cv_W
static JCudaTensor x2264 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4b_a_bn_bias
static JCudaTensor x2178 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_a_bn_scale
static JCudaTensor x2183 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_a_cv_W
static JCudaTensor x2172 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4b_b_bn_bias
static JCudaTensor x2137 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_b_bn_scale
static JCudaTensor x2132 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_b_cv_W
static JCudaTensor x2126 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4b_c_bn_bias
static JCudaTensor x2077 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4b_c_bn_scale
static JCudaTensor x2091 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4b_c_cv_W
static JCudaTensor x2085 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4c_a_bn_bias
static JCudaTensor x2026 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_a_bn_scale
static JCudaTensor x2031 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_a_cv_W
static JCudaTensor x2036 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4c_b_bn_bias
static JCudaTensor x1986 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_b_bn_scale
static JCudaTensor x1991 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_b_cv_W
static JCudaTensor x1980 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4c_c_bn_bias
static JCudaTensor x1931 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4c_c_bn_scale
static JCudaTensor x1945 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4c_c_cv_W
static JCudaTensor x1939 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4d_a_bn_bias
static JCudaTensor x1877 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_a_bn_scale
static JCudaTensor x1891 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_a_cv_W
static JCudaTensor x1885 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4d_b_bn_bias
static JCudaTensor x1845 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_b_bn_scale
static JCudaTensor x1840 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_b_cv_W
static JCudaTensor x1834 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4d_c_bn_bias
static JCudaTensor x1790 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4d_c_bn_scale
static JCudaTensor x1785 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4d_c_cv_W
static JCudaTensor x1798 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4e_a_bn_bias
static JCudaTensor x1745 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_a_bn_scale
static JCudaTensor x1740 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_a_cv_W
static JCudaTensor x1734 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4e_b_bn_bias
static JCudaTensor x1685 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_b_bn_scale
static JCudaTensor x1690 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_b_cv_W
static JCudaTensor x1698 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4e_c_bn_bias
static JCudaTensor x1653 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4e_c_bn_scale
static JCudaTensor x1648 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4e_c_cv_W
static JCudaTensor x1642 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4f_a_bn_bias
static JCudaTensor x1594 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_a_bn_scale
static JCudaTensor x1599 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_a_cv_W
static JCudaTensor x1588 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4f_b_bn_bias
static JCudaTensor x1539 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_b_bn_scale
static JCudaTensor x1544 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_b_cv_W
static JCudaTensor x1552 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4f_c_bn_bias
static JCudaTensor x1493 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4f_c_bn_scale
static JCudaTensor x1507 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4f_c_cv_W
static JCudaTensor x1501 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_5a1_bn_bias
static JCudaTensor x1322 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a1_bn_scale
static JCudaTensor x1335 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a1_cv_W
static JCudaTensor x1344 = JTensor.constFloat(0.0f, 2048, 1024, 1, 1).asJCudaTensor();
// V_5a2_a_bn_bias
static JCudaTensor x1442 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_a_bn_scale
static JCudaTensor x1447 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_a_cv_W
static JCudaTensor x1456 = JTensor.constFloat(0.0f, 512, 1024, 1, 1).asJCudaTensor();
// V_5a2_b_bn_bias
static JCudaTensor x1410 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_b_bn_scale
static JCudaTensor x1405 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_b_cv_W
static JCudaTensor x1399 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_5a2_c_bn_bias
static JCudaTensor x1355 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a2_c_bn_scale
static JCudaTensor x1350 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a2_c_cv_W
static JCudaTensor x1327 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
// V_5b_a_bn_bias
static JCudaTensor x1263 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_a_bn_scale
static JCudaTensor x1252 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_a_cv_W
static JCudaTensor x1257 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
// V_5b_b_bn_bias
static JCudaTensor x1203 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_b_bn_scale
static JCudaTensor x1208 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_b_cv_W
static JCudaTensor x1216 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_5b_c_bn_bias
static JCudaTensor x1171 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5b_c_bn_scale
static JCudaTensor x1166 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5b_c_cv_W
static JCudaTensor x1160 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
// V_5c_a_bn_bias
static JCudaTensor x1103 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_a_bn_scale
static JCudaTensor x1111 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_a_cv_W
static JCudaTensor x1116 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
// V_5c_b_bn_bias
static JCudaTensor x1057 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_b_bn_scale
static JCudaTensor x1071 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_b_cv_W
static JCudaTensor x1065 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_5c_c_bn_bias
static JCudaTensor x1025 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5c_c_bn_scale
static JCudaTensor x1020 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5c_c_cv_W
static JCudaTensor x1014 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
// V_fc_B
static JCudaTensor x971 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc_W
static JCudaTensor x966 = JTensor.constFloat(0.0f, 1000, 2048).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// fc_B
static JCudaTensor x934 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
// fc_W
static JCudaTensor x930 = JTensor.randomFloat(-0.03125f, 0.03125f, 1000, 2048).load(network_dir + "/fc_W").asJCudaTensor();

public static void main(String[] args){
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
test();
x26.save(network_dir + "/1_bn_bias");
x25.save(network_dir + "/1_bn_scale");
x18.save(network_dir + "/1_cv_W");
x53.save(network_dir + "/2a1_bn_bias");
x52.save(network_dir + "/2a1_bn_scale");
x38.save(network_dir + "/2a1_cv_W");
x60.save(network_dir + "/2a2_a_bn_bias");
x59.save(network_dir + "/2a2_a_bn_scale");
x45.save(network_dir + "/2a2_a_cv_W");
x77.save(network_dir + "/2a2_b_bn_bias");
x76.save(network_dir + "/2a2_b_bn_scale");
x69.save(network_dir + "/2a2_b_cv_W");
x92.save(network_dir + "/2a2_c_bn_bias");
x91.save(network_dir + "/2a2_c_bn_scale");
x85.save(network_dir + "/2a2_c_cv_W");
x116.save(network_dir + "/2b_a_bn_bias");
x115.save(network_dir + "/2b_a_bn_scale");
x108.save(network_dir + "/2b_a_cv_W");
x131.save(network_dir + "/2b_b_bn_bias");
x130.save(network_dir + "/2b_b_bn_scale");
x124.save(network_dir + "/2b_b_cv_W");
x146.save(network_dir + "/2b_c_bn_bias");
x145.save(network_dir + "/2b_c_bn_scale");
x139.save(network_dir + "/2b_c_cv_W");
x166.save(network_dir + "/2c_a_bn_bias");
x165.save(network_dir + "/2c_a_bn_scale");
x159.save(network_dir + "/2c_a_cv_W");
x181.save(network_dir + "/2c_b_bn_bias");
x180.save(network_dir + "/2c_b_bn_scale");
x174.save(network_dir + "/2c_b_cv_W");
x196.save(network_dir + "/2c_c_bn_bias");
x195.save(network_dir + "/2c_c_bn_scale");
x189.save(network_dir + "/2c_c_cv_W");
x231.save(network_dir + "/3a1_bn_bias");
x230.save(network_dir + "/3a1_bn_scale");
x216.save(network_dir + "/3a1_cv_W");
x224.save(network_dir + "/3a2_a_bn_bias");
x223.save(network_dir + "/3a2_a_bn_scale");
x209.save(network_dir + "/3a2_a_cv_W");
x248.save(network_dir + "/3a2_b_bn_bias");
x247.save(network_dir + "/3a2_b_bn_scale");
x240.save(network_dir + "/3a2_b_cv_W");
x264.save(network_dir + "/3a2_c_bn_bias");
x263.save(network_dir + "/3a2_c_bn_scale");
x256.save(network_dir + "/3a2_c_cv_W");
x288.save(network_dir + "/3b_a_bn_bias");
x287.save(network_dir + "/3b_a_bn_scale");
x280.save(network_dir + "/3b_a_cv_W");
x303.save(network_dir + "/3b_b_bn_bias");
x302.save(network_dir + "/3b_b_bn_scale");
x296.save(network_dir + "/3b_b_cv_W");
x318.save(network_dir + "/3b_c_bn_bias");
x317.save(network_dir + "/3b_c_bn_scale");
x311.save(network_dir + "/3b_c_cv_W");
x338.save(network_dir + "/3c_a_bn_bias");
x337.save(network_dir + "/3c_a_bn_scale");
x331.save(network_dir + "/3c_a_cv_W");
x353.save(network_dir + "/3c_b_bn_bias");
x352.save(network_dir + "/3c_b_bn_scale");
x346.save(network_dir + "/3c_b_cv_W");
x368.save(network_dir + "/3c_c_bn_bias");
x367.save(network_dir + "/3c_c_bn_scale");
x361.save(network_dir + "/3c_c_cv_W");
x388.save(network_dir + "/3d_a_bn_bias");
x387.save(network_dir + "/3d_a_bn_scale");
x381.save(network_dir + "/3d_a_cv_W");
x403.save(network_dir + "/3d_b_bn_bias");
x402.save(network_dir + "/3d_b_bn_scale");
x396.save(network_dir + "/3d_b_cv_W");
x418.save(network_dir + "/3d_c_bn_bias");
x417.save(network_dir + "/3d_c_bn_scale");
x411.save(network_dir + "/3d_c_cv_W");
x446.save(network_dir + "/4a1_bn_bias");
x445.save(network_dir + "/4a1_bn_scale");
x438.save(network_dir + "/4a1_cv_W");
x453.save(network_dir + "/4a2_a_bn_bias");
x452.save(network_dir + "/4a2_a_bn_scale");
x431.save(network_dir + "/4a2_a_cv_W");
x470.save(network_dir + "/4a2_b_bn_bias");
x469.save(network_dir + "/4a2_b_bn_scale");
x462.save(network_dir + "/4a2_b_cv_W");
x486.save(network_dir + "/4a2_c_bn_bias");
x485.save(network_dir + "/4a2_c_bn_scale");
x478.save(network_dir + "/4a2_c_cv_W");
x510.save(network_dir + "/4b_a_bn_bias");
x509.save(network_dir + "/4b_a_bn_scale");
x502.save(network_dir + "/4b_a_cv_W");
x525.save(network_dir + "/4b_b_bn_bias");
x524.save(network_dir + "/4b_b_bn_scale");
x518.save(network_dir + "/4b_b_cv_W");
x540.save(network_dir + "/4b_c_bn_bias");
x539.save(network_dir + "/4b_c_bn_scale");
x533.save(network_dir + "/4b_c_cv_W");
x560.save(network_dir + "/4c_a_bn_bias");
x559.save(network_dir + "/4c_a_bn_scale");
x553.save(network_dir + "/4c_a_cv_W");
x575.save(network_dir + "/4c_b_bn_bias");
x574.save(network_dir + "/4c_b_bn_scale");
x568.save(network_dir + "/4c_b_cv_W");
x590.save(network_dir + "/4c_c_bn_bias");
x589.save(network_dir + "/4c_c_bn_scale");
x583.save(network_dir + "/4c_c_cv_W");
x610.save(network_dir + "/4d_a_bn_bias");
x609.save(network_dir + "/4d_a_bn_scale");
x603.save(network_dir + "/4d_a_cv_W");
x625.save(network_dir + "/4d_b_bn_bias");
x624.save(network_dir + "/4d_b_bn_scale");
x618.save(network_dir + "/4d_b_cv_W");
x640.save(network_dir + "/4d_c_bn_bias");
x639.save(network_dir + "/4d_c_bn_scale");
x633.save(network_dir + "/4d_c_cv_W");
x660.save(network_dir + "/4e_a_bn_bias");
x659.save(network_dir + "/4e_a_bn_scale");
x653.save(network_dir + "/4e_a_cv_W");
x675.save(network_dir + "/4e_b_bn_bias");
x674.save(network_dir + "/4e_b_bn_scale");
x668.save(network_dir + "/4e_b_cv_W");
x690.save(network_dir + "/4e_c_bn_bias");
x689.save(network_dir + "/4e_c_bn_scale");
x683.save(network_dir + "/4e_c_cv_W");
x710.save(network_dir + "/4f_a_bn_bias");
x709.save(network_dir + "/4f_a_bn_scale");
x703.save(network_dir + "/4f_a_cv_W");
x725.save(network_dir + "/4f_b_bn_bias");
x724.save(network_dir + "/4f_b_bn_scale");
x718.save(network_dir + "/4f_b_cv_W");
x740.save(network_dir + "/4f_c_bn_bias");
x739.save(network_dir + "/4f_c_bn_scale");
x733.save(network_dir + "/4f_c_cv_W");
x768.save(network_dir + "/5a1_bn_bias");
x767.save(network_dir + "/5a1_bn_scale");
x753.save(network_dir + "/5a1_cv_W");
x775.save(network_dir + "/5a2_a_bn_bias");
x774.save(network_dir + "/5a2_a_bn_scale");
x760.save(network_dir + "/5a2_a_cv_W");
x792.save(network_dir + "/5a2_b_bn_bias");
x791.save(network_dir + "/5a2_b_bn_scale");
x784.save(network_dir + "/5a2_b_cv_W");
x808.save(network_dir + "/5a2_c_bn_bias");
x807.save(network_dir + "/5a2_c_bn_scale");
x800.save(network_dir + "/5a2_c_cv_W");
x832.save(network_dir + "/5b_a_bn_bias");
x831.save(network_dir + "/5b_a_bn_scale");
x824.save(network_dir + "/5b_a_cv_W");
x847.save(network_dir + "/5b_b_bn_bias");
x846.save(network_dir + "/5b_b_bn_scale");
x840.save(network_dir + "/5b_b_cv_W");
x862.save(network_dir + "/5b_c_bn_bias");
x861.save(network_dir + "/5b_c_bn_scale");
x855.save(network_dir + "/5b_c_cv_W");
x882.save(network_dir + "/5c_a_bn_bias");
x881.save(network_dir + "/5c_a_bn_scale");
x875.save(network_dir + "/5c_a_cv_W");
x897.save(network_dir + "/5c_b_bn_bias");
x896.save(network_dir + "/5c_b_bn_scale");
x890.save(network_dir + "/5c_b_cv_W");
x912.save(network_dir + "/5c_c_bn_bias");
x911.save(network_dir + "/5c_c_bn_scale");
x905.save(network_dir + "/5c_c_cv_W");
x934.save(network_dir + "/fc_B");
x930.save(network_dir + "/fc_W");
x381.free();
x3530.free();
x3377.free();
x934.free();
x753.free();
x469.free();
x53.free();
x733.free();
x1642.free();
x131.free();
x2091.free();
x317.free();
x3090.free();
x2377.free();
x1501.free();
x2810.free();
x1945.free();
x116.free();
x683.free();
x767.free();
x2031.free();
x825.free();
x2330.free();
x2462.free();
x855.free();
x774.free();
x3241.free();
x2957.free();
x337.free();
x453.free();
x140.free();
x906.free();
x2764.free();
x3147.free();
x3044.free();
x890.free();
x241.free();
x703.free();
x3476.free();
x1991.free();
x2567.free();
x847.free();
x831.free();
x534.free();
x180.free();
x1355.free();
x856.free();
x709.free();
x1539.free();
x1166.free();
x39.free();
x1685.free();
x2472.free();
x841.free();
x1552.free();
x1057.free();
x784.free();
x3054.free();
x1986.free();
x1931.free();
x3246.free();
x2253.free();
x397.free();
x1790.free();
x247.free();
x224.free();
x2608.free();
x718.free();
x1493.free();
x1257.free();
x331.free();
x775.free();
x2275.free();
x18.free();
x19.free();
x175.free();
x124.free();
x396.free();
x740.free();
x891.free();
x2270.free();
x1594.free();
x1160.free();
x674.free();
x3482.free();
x1014.free();
x734.free();
x739.free();
x125.free();
x1171.free();
x689.free();
x130.free();
x2870.free();
x3200.free();
x2613.free();
x159.free();
x280.free();
x59.free();
x896.free();
x710.free();
x2132.free();
x875.free();
x486.free();
x2513.free();
x352.free();
x368.free();
x231.free();
x230.free();
x554.free();
x181.free();
x518.free();
x2319.free();
x296.free();
x1252.free();
x584.free();
x332.free();
x2619.free();
x353.free();
x311.free();
x1980.free();
x624.free();
x704.free();
x2172.free();
x1939.free();
x668.free();
x634.free();
x445.free();
x2751.free();
x1335.free();
x2325.free();
x297.free();
x1891.free();
x2994.free();
x2126.free();
x145.free();
x1203.free();
x2372.free();
x618.free();
x801.free();
x824.free();
x1025.free();
x3357.free();
x609.free();
x619.free();
x2026.free();
x2654.free();
x590.free();
x2137.free();
x70.free();
x403.free();
x675.free();
x77.free();
x263.free();
x248.free();
x633.free();
x2756.free();
x25.free();
x196.free();
x966.free();
x3233.free();
x911.free();
x1785.free();
x574.free();
x1103.free();
x2366.free();
x195.free();
x1798.free();
x2505.free();
x719.free();
x3525.free();
x690.free();
x432.free();
x519.free();
x165.free();
x1327.free();
x26.free();
x1065.free();
x256.free();
x510.free();
x2660.free();
x660.free();
x257.free();
x2889.free();
x761.free();
x1020.free();
x1648.free();
x338.free();
x2427.free();
x470.free();
x1690.free();
x318.free();
x971.free();
x85.free();
x724.free();
x553.free();
x367.free();
x3101.free();
x1399.free();
x881.free();
x2718.free();
x388.free();
x223.free();
x3279.free();
x166.free();
x45.free();
x288.free();
x478.free();
x60.free();
x832.free();
x2705.free();
x768.free();
x1834.free();
x791.free();
x479.free();
x2518.free();
x2183.free();
x1599.free();
x2894.free();
x1877.free();
x86.free();
x684.free();
x808.free();
x2467.free();
x3000.free();
x792.free();
x2949.free();
x217.free();
x625.free();
x603.free();
x1447.free();
x930.free();
x3141.free();
x92.free();
x639.free();
x108.free();
x91.free();
x362.free();
x402.free();
x1116.free();
x52.free();
x382.free();
x640.free();
x1410.free();
x2259.free();
x1344.free();
x1405.free();
x3287.free();
x2797.free();
x303.free();
x46.free();
x725.free();
x2802.free();
x3049.free();
x846.free();
x139.free();
x439.free();
x3440.free();
x861.free();
x2036.free();
x2178.free();
x2573.free();
x2077.free();
x1698.free();
x3362.free();
x38.free();
x264.free();
x446.free();
x312.free();
x3367.free();
x69.free();
x1745.free();
x115.free();
x3133.free();
x575.free();
x2665.free();
x569.free();
x503.free();
x438.free();
x1208.free();
x160.free();
x146.free();
x1216.free();
x800.free();
x109.free();
x568.free();
x346.free();
x540.free();
x1442.free();
x3005.free();
x3434.free();
x2944.free();
x418.free();
x1071.free();
x862.free();
x1885.free();
x2875.free();
x287.free();
x281.free();
x1588.free();
x840.free();
x524.free();
x897.free();
x502.free();
x589.free();
x785.free();
x653.free();
x905.free();
x3352.free();
x3195.free();
x3187.free();
x1845.free();
x3535.free();
x533.free();
x216.free();
x209.free();
x604.free();
x417.free();
x3293.free();
x882.free();
x2899.free();
x659.free();
x463.free();
x411.free();
x559.free();
x1544.free();
x2710.free();
x1734.free();
x1456.free();
x2880.free();
x347.free();
x1653.free();
x412.free();
x583.free();
x2559.free();
x76.free();
x912.free();
x2248.free();
x1263.free();
x190.free();
x876.free();
x3426.free();
x485.free();
x1740.free();
x2264.free();
x1507.free();
x525.free();
x560.free();
x760.free();
x2085.free();
x654.free();
x2422.free();
x807.free();
x1322.free();
x509.free();
x174.free();
x1840.free();
x189.free();
x452.free();
x302.free();
x210.free();
x3372.free();
x1350.free();
x754.free();
x539.free();
x3096.free();
x3487.free();
x669.free();
x240.free();
x610.free();
x431.free();
x1111.free();
x2416.free();
x462.free();
x387.free();
x361.free();
x61.free();
x937.free();
x167.free();
x812.free();
x404.free();
x242.free();
x776.free();
x232.free();
x741.free();
x526.free();
x235.free();
x769.free();
x561.free();
x898.free();
x369.free();
x64.free();
x661.free();
x147.free();
x96.free();
x319.free();
x40.free();
x487.free();
x71.free();
x591.free();
x27.free();
x339.free();
x913.free();
x110.free();
x54.free();
x576.free();
x833.free();
x447.free();
x78.free();
x354.free();
x225.free();
x20.free();
x923.free();
x93.free();
x490.free();
x268.free();
x480.free();
x457.free();
x848.free();
x863.free();
x433.free();
x883.free();
x691.free();
x30.free();
x762.free();
x182.free();
x793.free();
x419.free();
x511.free();
x626.free();
x454.free();
x258.free();
x541.free();
x676.free();
x211.free();
x809.free();
x464.free();
x779.free();
x826.free();
x33.free();
x132.free();
x755.free();
x289.free();
x282.free();
x786.free();
x265.free();
x711.free();
x641.free();
x471.free();
x504.free();
x726.free();
x389.free();
x218.free();
x47.free();
x197.free();
x440.free();
x802.free();
x117.free();
x611.free();
x304.free();
x249.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void train() {
 for(int x5=0; x5<train_itr; x5++) {
JTensorFloatTuple x6 =  x1.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X197 = Cuda(X)
JCudaTensor x7;
JTensorFloat x8;
x8 = x3;
x7 = x8.asJCudaTensor();

// val X395 = Cuda(Indicator(Y, 1000))
JCudaTensor x9;
JTensorFloat x10;
x10 = x4.asIndicator(1000);
x9 = x10.asJCudaTensor();

// val X1139 = - X395.copy
JCudaTensor x11;
JCudaTensor x12;
float x13;
x12 = x9;
x12 = x12.clone();
x13 = -1;
x11 = x12.times_i(x13);

// val X198 = Convolv(2,3)(X197,1_cv_W,1_cv_B)
JCudaTensor x14;
JCudaTensor x15, x16, x17;
x15 = x7;
x16 = x18;
x17 = x19;
x14 = x20.forward(x15, x16, x17);

// val X199 = BatchNorm(1_bn)(X198,1_bn_scale,1_bn_bias)
JCudaTensor x21;
JCudaTensor x22, x23, x24;
x22 = x14;
x23 = x25;
x24 = x26;
x21 = x27.forward(x22, x23, x24);

// val X200 = ReLU()(X199)
JCudaTensor x28;
JCudaTensor x29;
x29 = x21;
x28 = x30.forward(x29);

// val X201 = Pooling(3,2,0,true)(X200)
JCudaTensor x31;
JCudaTensor x32;
x32 = x28;
x31 = x33.forward(x32);

// val X202 = Convolv(1,0)(X201,2a1_cv_W,2a1_cv_B)
JCudaTensor x34;
JCudaTensor x35, x36, x37;
x35 = x31;
x36 = x38;
x37 = x39;
x34 = x40.forward(x35, x36, x37);

// val X205 = Convolv(1,0)(X201,2a2_a_cv_W,2a2_a_cv_B)
JCudaTensor x41;
JCudaTensor x42, x43, x44;
x42 = x31;
x43 = x45;
x44 = x46;
x41 = x47.forward(x42, x43, x44);

// val X203 = BatchNorm(2a1_bn)(X202,2a1_bn_scale,2a1_bn_bias)
JCudaTensor x48;
JCudaTensor x49, x50, x51;
x49 = x34;
x50 = x52;
x51 = x53;
x48 = x54.forward(x49, x50, x51);

// val X206 = BatchNorm(2a2_a_bn)(X205,2a2_a_bn_scale,2a2_a_bn_bias)
JCudaTensor x55;
JCudaTensor x56, x57, x58;
x56 = x41;
x57 = x59;
x58 = x60;
x55 = x61.forward(x56, x57, x58);

// val X207 = ReLU()(X206)
JCudaTensor x62;
JCudaTensor x63;
x63 = x55;
x62 = x64.forward(x63);

// val X208 = Convolv(1,1)(X207,2a2_b_cv_W,2a2_b_cv_B)
JCudaTensor x65;
JCudaTensor x66, x67, x68;
x66 = x62;
x67 = x69;
x68 = x70;
x65 = x71.forward(x66, x67, x68);

// val X209 = BatchNorm(2a2_b_bn)(X208,2a2_b_bn_scale,2a2_b_bn_bias)
JCudaTensor x72;
JCudaTensor x73, x74, x75;
x73 = x65;
x74 = x76;
x75 = x77;
x72 = x78.forward(x73, x74, x75);

// val X210 = ReLU()(X209)
JCudaTensor x79;
JCudaTensor x80;
x80 = x72;
x79 = x64.forward(x80);

// val X211 = Convolv(1,0)(X210,2a2_c_cv_W,2a2_c_cv_B)
JCudaTensor x81;
JCudaTensor x82, x83, x84;
x82 = x79;
x83 = x85;
x84 = x86;
x81 = x40.forward(x82, x83, x84);

// val X212 = BatchNorm(2a2_c_bn)(X211,2a2_c_bn_scale,2a2_c_bn_bias)
JCudaTensor x87;
JCudaTensor x88, x89, x90;
x88 = x81;
x89 = x91;
x90 = x92;
x87 = x93.forward(x88, x89, x90);

// val X204 = ReLU()(X203)
JCudaTensor x94;
JCudaTensor x95;
x95 = x48;
x94 = x96.forward(x95);

// val X213 = ReLU()(X212)
JCudaTensor x97;
JCudaTensor x98;
x98 = x87;
x97 = x96.forward(x98);

// val X214 = (X204.copy + X213)
JCudaTensor x99;
JCudaTensor x100, x101;
x100 = x94;
x100 = x100.clone();
x101 = x97;
x99 = x100.plus_i(x101);

// val X215 = ReLU()(X214)
JCudaTensor x102;
JCudaTensor x103;
x103 = x99;
x102 = x96.forward(x103);

// val X216 = Convolv(1,0)(X215,2b_a_cv_W,2b_a_cv_B)
JCudaTensor x104;
JCudaTensor x105, x106, x107;
x105 = x102;
x106 = x108;
x107 = x109;
x104 = x110.forward(x105, x106, x107);

// val X217 = BatchNorm(2b_a_bn)(X216,2b_a_bn_scale,2b_a_bn_bias)
JCudaTensor x111;
JCudaTensor x112, x113, x114;
x112 = x104;
x113 = x115;
x114 = x116;
x111 = x117.forward(x112, x113, x114);

// val X218 = ReLU()(X217)
JCudaTensor x118;
JCudaTensor x119;
x119 = x111;
x118 = x64.forward(x119);

// val X219 = Convolv(1,1)(X218,2b_b_cv_W,2b_b_cv_B)
JCudaTensor x120;
JCudaTensor x121, x122, x123;
x121 = x118;
x122 = x124;
x123 = x125;
x120 = x71.forward(x121, x122, x123);

// val X220 = BatchNorm(2b_b_bn)(X219,2b_b_bn_scale,2b_b_bn_bias)
JCudaTensor x126;
JCudaTensor x127, x128, x129;
x127 = x120;
x128 = x130;
x129 = x131;
x126 = x132.forward(x127, x128, x129);

// val X221 = ReLU()(X220)
JCudaTensor x133;
JCudaTensor x134;
x134 = x126;
x133 = x64.forward(x134);

// val X222 = Convolv(1,0)(X221,2b_c_cv_W,2b_c_cv_B)
JCudaTensor x135;
JCudaTensor x136, x137, x138;
x136 = x133;
x137 = x139;
x138 = x140;
x135 = x40.forward(x136, x137, x138);

// val X223 = BatchNorm(2b_c_bn)(X222,2b_c_bn_scale,2b_c_bn_bias)
JCudaTensor x141;
JCudaTensor x142, x143, x144;
x142 = x135;
x143 = x145;
x144 = x146;
x141 = x147.forward(x142, x143, x144);

// val X224 = ReLU()(X223)
JCudaTensor x148;
JCudaTensor x149;
x149 = x141;
x148 = x96.forward(x149);

// val X225 = (X224.copy + X215)
JCudaTensor x150;
JCudaTensor x151, x152;
x151 = x148;
x151 = x151.clone();
x152 = x102;
x150 = x151.plus_i(x152);

// val X226 = ReLU()(X225)
JCudaTensor x153;
JCudaTensor x154;
x154 = x150;
x153 = x96.forward(x154);

// val X227 = Convolv(1,0)(X226,2c_a_cv_W,2c_a_cv_B)
JCudaTensor x155;
JCudaTensor x156, x157, x158;
x156 = x153;
x157 = x159;
x158 = x160;
x155 = x110.forward(x156, x157, x158);

// val X228 = BatchNorm(2c_a_bn)(X227,2c_a_bn_scale,2c_a_bn_bias)
JCudaTensor x161;
JCudaTensor x162, x163, x164;
x162 = x155;
x163 = x165;
x164 = x166;
x161 = x167.forward(x162, x163, x164);

// val X229 = ReLU()(X228)
JCudaTensor x168;
JCudaTensor x169;
x169 = x161;
x168 = x64.forward(x169);

// val X230 = Convolv(1,1)(X229,2c_b_cv_W,2c_b_cv_B)
JCudaTensor x170;
JCudaTensor x171, x172, x173;
x171 = x168;
x172 = x174;
x173 = x175;
x170 = x71.forward(x171, x172, x173);

// val X231 = BatchNorm(2c_b_bn)(X230,2c_b_bn_scale,2c_b_bn_bias)
JCudaTensor x176;
JCudaTensor x177, x178, x179;
x177 = x170;
x178 = x180;
x179 = x181;
x176 = x182.forward(x177, x178, x179);

// val X232 = ReLU()(X231)
JCudaTensor x183;
JCudaTensor x184;
x184 = x176;
x183 = x64.forward(x184);

// val X233 = Convolv(1,0)(X232,2c_c_cv_W,2c_c_cv_B)
JCudaTensor x185;
JCudaTensor x186, x187, x188;
x186 = x183;
x187 = x189;
x188 = x190;
x185 = x40.forward(x186, x187, x188);

// val X234 = BatchNorm(2c_c_bn)(X233,2c_c_bn_scale,2c_c_bn_bias)
JCudaTensor x191;
JCudaTensor x192, x193, x194;
x192 = x185;
x193 = x195;
x194 = x196;
x191 = x197.forward(x192, x193, x194);

// val X235 = ReLU()(X234)
JCudaTensor x198;
JCudaTensor x199;
x199 = x191;
x198 = x96.forward(x199);

// val X236 = (X235.copy + X226)
JCudaTensor x200;
JCudaTensor x201, x202;
x201 = x198;
x201 = x201.clone();
x202 = x153;
x200 = x201.plus_i(x202);

// val X237 = ReLU()(X236)
JCudaTensor x203;
JCudaTensor x204;
x204 = x200;
x203 = x96.forward(x204);

// val X241 = Convolv(2,0)(X237,3a2_a_cv_W,3a2_a_cv_B)
JCudaTensor x205;
JCudaTensor x206, x207, x208;
x206 = x203;
x207 = x209;
x208 = x210;
x205 = x211.forward(x206, x207, x208);

// val X238 = Convolv(2,0)(X237,3a1_cv_W,3a1_cv_B)
JCudaTensor x212;
JCudaTensor x213, x214, x215;
x213 = x203;
x214 = x216;
x215 = x217;
x212 = x218.forward(x213, x214, x215);

// val X242 = BatchNorm(3a2_a_bn)(X241,3a2_a_bn_scale,3a2_a_bn_bias)
JCudaTensor x219;
JCudaTensor x220, x221, x222;
x220 = x205;
x221 = x223;
x222 = x224;
x219 = x225.forward(x220, x221, x222);

// val X239 = BatchNorm(3a1_bn)(X238,3a1_bn_scale,3a1_bn_bias)
JCudaTensor x226;
JCudaTensor x227, x228, x229;
x227 = x212;
x228 = x230;
x229 = x231;
x226 = x232.forward(x227, x228, x229);

// val X243 = ReLU()(X242)
JCudaTensor x233;
JCudaTensor x234;
x234 = x219;
x233 = x235.forward(x234);

// val X244 = Convolv(1,1)(X243,3a2_b_cv_W,3a2_b_cv_B)
JCudaTensor x236;
JCudaTensor x237, x238, x239;
x237 = x233;
x238 = x240;
x239 = x241;
x236 = x242.forward(x237, x238, x239);

// val X245 = BatchNorm(3a2_b_bn)(X244,3a2_b_bn_scale,3a2_b_bn_bias)
JCudaTensor x243;
JCudaTensor x244, x245, x246;
x244 = x236;
x245 = x247;
x246 = x248;
x243 = x249.forward(x244, x245, x246);

// val X246 = ReLU()(X245)
JCudaTensor x250;
JCudaTensor x251;
x251 = x243;
x250 = x235.forward(x251);

// val X247 = Convolv(1,0)(X246,3a2_c_cv_W,3a2_c_cv_B)
JCudaTensor x252;
JCudaTensor x253, x254, x255;
x253 = x250;
x254 = x256;
x255 = x257;
x252 = x258.forward(x253, x254, x255);

// val X248 = BatchNorm(3a2_c_bn)(X247,3a2_c_bn_scale,3a2_c_bn_bias)
JCudaTensor x259;
JCudaTensor x260, x261, x262;
x260 = x252;
x261 = x263;
x262 = x264;
x259 = x265.forward(x260, x261, x262);

// val X240 = ReLU()(X239)
JCudaTensor x266;
JCudaTensor x267;
x267 = x226;
x266 = x268.forward(x267);

// val X249 = ReLU()(X248)
JCudaTensor x269;
JCudaTensor x270;
x270 = x259;
x269 = x268.forward(x270);

// val X250 = (X240.copy + X249)
JCudaTensor x271;
JCudaTensor x272, x273;
x272 = x266;
x272 = x272.clone();
x273 = x269;
x271 = x272.plus_i(x273);

// val X251 = ReLU()(X250)
JCudaTensor x274;
JCudaTensor x275;
x275 = x271;
x274 = x268.forward(x275);

// val X252 = Convolv(1,0)(X251,3b_a_cv_W,3b_a_cv_B)
JCudaTensor x276;
JCudaTensor x277, x278, x279;
x277 = x274;
x278 = x280;
x279 = x281;
x276 = x282.forward(x277, x278, x279);

// val X253 = BatchNorm(3b_a_bn)(X252,3b_a_bn_scale,3b_a_bn_bias)
JCudaTensor x283;
JCudaTensor x284, x285, x286;
x284 = x276;
x285 = x287;
x286 = x288;
x283 = x289.forward(x284, x285, x286);

// val X254 = ReLU()(X253)
JCudaTensor x290;
JCudaTensor x291;
x291 = x283;
x290 = x235.forward(x291);

// val X255 = Convolv(1,1)(X254,3b_b_cv_W,3b_b_cv_B)
JCudaTensor x292;
JCudaTensor x293, x294, x295;
x293 = x290;
x294 = x296;
x295 = x297;
x292 = x242.forward(x293, x294, x295);

// val X256 = BatchNorm(3b_b_bn)(X255,3b_b_bn_scale,3b_b_bn_bias)
JCudaTensor x298;
JCudaTensor x299, x300, x301;
x299 = x292;
x300 = x302;
x301 = x303;
x298 = x304.forward(x299, x300, x301);

// val X257 = ReLU()(X256)
JCudaTensor x305;
JCudaTensor x306;
x306 = x298;
x305 = x235.forward(x306);

// val X258 = Convolv(1,0)(X257,3b_c_cv_W,3b_c_cv_B)
JCudaTensor x307;
JCudaTensor x308, x309, x310;
x308 = x305;
x309 = x311;
x310 = x312;
x307 = x258.forward(x308, x309, x310);

// val X259 = BatchNorm(3b_c_bn)(X258,3b_c_bn_scale,3b_c_bn_bias)
JCudaTensor x313;
JCudaTensor x314, x315, x316;
x314 = x307;
x315 = x317;
x316 = x318;
x313 = x319.forward(x314, x315, x316);

// val X260 = ReLU()(X259)
JCudaTensor x320;
JCudaTensor x321;
x321 = x313;
x320 = x268.forward(x321);

// val X261 = (X260.copy + X251)
JCudaTensor x322;
JCudaTensor x323, x324;
x323 = x320;
x323 = x323.clone();
x324 = x274;
x322 = x323.plus_i(x324);

// val X262 = ReLU()(X261)
JCudaTensor x325;
JCudaTensor x326;
x326 = x322;
x325 = x268.forward(x326);

// val X263 = Convolv(1,0)(X262,3c_a_cv_W,3c_a_cv_B)
JCudaTensor x327;
JCudaTensor x328, x329, x330;
x328 = x325;
x329 = x331;
x330 = x332;
x327 = x282.forward(x328, x329, x330);

// val X264 = BatchNorm(3c_a_bn)(X263,3c_a_bn_scale,3c_a_bn_bias)
JCudaTensor x333;
JCudaTensor x334, x335, x336;
x334 = x327;
x335 = x337;
x336 = x338;
x333 = x339.forward(x334, x335, x336);

// val X265 = ReLU()(X264)
JCudaTensor x340;
JCudaTensor x341;
x341 = x333;
x340 = x235.forward(x341);

// val X266 = Convolv(1,1)(X265,3c_b_cv_W,3c_b_cv_B)
JCudaTensor x342;
JCudaTensor x343, x344, x345;
x343 = x340;
x344 = x346;
x345 = x347;
x342 = x242.forward(x343, x344, x345);

// val X267 = BatchNorm(3c_b_bn)(X266,3c_b_bn_scale,3c_b_bn_bias)
JCudaTensor x348;
JCudaTensor x349, x350, x351;
x349 = x342;
x350 = x352;
x351 = x353;
x348 = x354.forward(x349, x350, x351);

// val X268 = ReLU()(X267)
JCudaTensor x355;
JCudaTensor x356;
x356 = x348;
x355 = x235.forward(x356);

// val X269 = Convolv(1,0)(X268,3c_c_cv_W,3c_c_cv_B)
JCudaTensor x357;
JCudaTensor x358, x359, x360;
x358 = x355;
x359 = x361;
x360 = x362;
x357 = x258.forward(x358, x359, x360);

// val X270 = BatchNorm(3c_c_bn)(X269,3c_c_bn_scale,3c_c_bn_bias)
JCudaTensor x363;
JCudaTensor x364, x365, x366;
x364 = x357;
x365 = x367;
x366 = x368;
x363 = x369.forward(x364, x365, x366);

// val X271 = ReLU()(X270)
JCudaTensor x370;
JCudaTensor x371;
x371 = x363;
x370 = x268.forward(x371);

// val X272 = (X271.copy + X262)
JCudaTensor x372;
JCudaTensor x373, x374;
x373 = x370;
x373 = x373.clone();
x374 = x325;
x372 = x373.plus_i(x374);

// val X273 = ReLU()(X272)
JCudaTensor x375;
JCudaTensor x376;
x376 = x372;
x375 = x268.forward(x376);

// val X274 = Convolv(1,0)(X273,3d_a_cv_W,3d_a_cv_B)
JCudaTensor x377;
JCudaTensor x378, x379, x380;
x378 = x375;
x379 = x381;
x380 = x382;
x377 = x282.forward(x378, x379, x380);

// val X275 = BatchNorm(3d_a_bn)(X274,3d_a_bn_scale,3d_a_bn_bias)
JCudaTensor x383;
JCudaTensor x384, x385, x386;
x384 = x377;
x385 = x387;
x386 = x388;
x383 = x389.forward(x384, x385, x386);

// val X276 = ReLU()(X275)
JCudaTensor x390;
JCudaTensor x391;
x391 = x383;
x390 = x235.forward(x391);

// val X277 = Convolv(1,1)(X276,3d_b_cv_W,3d_b_cv_B)
JCudaTensor x392;
JCudaTensor x393, x394, x395;
x393 = x390;
x394 = x396;
x395 = x397;
x392 = x242.forward(x393, x394, x395);

// val X278 = BatchNorm(3d_b_bn)(X277,3d_b_bn_scale,3d_b_bn_bias)
JCudaTensor x398;
JCudaTensor x399, x400, x401;
x399 = x392;
x400 = x402;
x401 = x403;
x398 = x404.forward(x399, x400, x401);

// val X279 = ReLU()(X278)
JCudaTensor x405;
JCudaTensor x406;
x406 = x398;
x405 = x235.forward(x406);

// val X280 = Convolv(1,0)(X279,3d_c_cv_W,3d_c_cv_B)
JCudaTensor x407;
JCudaTensor x408, x409, x410;
x408 = x405;
x409 = x411;
x410 = x412;
x407 = x258.forward(x408, x409, x410);

// val X281 = BatchNorm(3d_c_bn)(X280,3d_c_bn_scale,3d_c_bn_bias)
JCudaTensor x413;
JCudaTensor x414, x415, x416;
x414 = x407;
x415 = x417;
x416 = x418;
x413 = x419.forward(x414, x415, x416);

// val X282 = ReLU()(X281)
JCudaTensor x420;
JCudaTensor x421;
x421 = x413;
x420 = x268.forward(x421);

// val X283 = (X282.copy + X273)
JCudaTensor x422;
JCudaTensor x423, x424;
x423 = x420;
x423 = x423.clone();
x424 = x375;
x422 = x423.plus_i(x424);

// val X284 = ReLU()(X283)
JCudaTensor x425;
JCudaTensor x426;
x426 = x422;
x425 = x268.forward(x426);

// val X288 = Convolv(2,0)(X284,4a2_a_cv_W,4a2_a_cv_B)
JCudaTensor x427;
JCudaTensor x428, x429, x430;
x428 = x425;
x429 = x431;
x430 = x432;
x427 = x433.forward(x428, x429, x430);

// val X285 = Convolv(2,0)(X284,4a1_cv_W,4a1_cv_B)
JCudaTensor x434;
JCudaTensor x435, x436, x437;
x435 = x425;
x436 = x438;
x437 = x439;
x434 = x440.forward(x435, x436, x437);

// val X286 = BatchNorm(4a1_bn)(X285,4a1_bn_scale,4a1_bn_bias)
JCudaTensor x441;
JCudaTensor x442, x443, x444;
x442 = x434;
x443 = x445;
x444 = x446;
x441 = x447.forward(x442, x443, x444);

// val X289 = BatchNorm(4a2_a_bn)(X288,4a2_a_bn_scale,4a2_a_bn_bias)
JCudaTensor x448;
JCudaTensor x449, x450, x451;
x449 = x427;
x450 = x452;
x451 = x453;
x448 = x454.forward(x449, x450, x451);

// val X290 = ReLU()(X289)
JCudaTensor x455;
JCudaTensor x456;
x456 = x448;
x455 = x457.forward(x456);

// val X291 = Convolv(1,1)(X290,4a2_b_cv_W,4a2_b_cv_B)
JCudaTensor x458;
JCudaTensor x459, x460, x461;
x459 = x455;
x460 = x462;
x461 = x463;
x458 = x464.forward(x459, x460, x461);

// val X292 = BatchNorm(4a2_b_bn)(X291,4a2_b_bn_scale,4a2_b_bn_bias)
JCudaTensor x465;
JCudaTensor x466, x467, x468;
x466 = x458;
x467 = x469;
x468 = x470;
x465 = x471.forward(x466, x467, x468);

// val X293 = ReLU()(X292)
JCudaTensor x472;
JCudaTensor x473;
x473 = x465;
x472 = x457.forward(x473);

// val X294 = Convolv(1,0)(X293,4a2_c_cv_W,4a2_c_cv_B)
JCudaTensor x474;
JCudaTensor x475, x476, x477;
x475 = x472;
x476 = x478;
x477 = x479;
x474 = x480.forward(x475, x476, x477);

// val X295 = BatchNorm(4a2_c_bn)(X294,4a2_c_bn_scale,4a2_c_bn_bias)
JCudaTensor x481;
JCudaTensor x482, x483, x484;
x482 = x474;
x483 = x485;
x484 = x486;
x481 = x487.forward(x482, x483, x484);

// val X287 = ReLU()(X286)
JCudaTensor x488;
JCudaTensor x489;
x489 = x441;
x488 = x490.forward(x489);

// val X296 = ReLU()(X295)
JCudaTensor x491;
JCudaTensor x492;
x492 = x481;
x491 = x490.forward(x492);

// val X297 = (X287.copy + X296)
JCudaTensor x493;
JCudaTensor x494, x495;
x494 = x488;
x494 = x494.clone();
x495 = x491;
x493 = x494.plus_i(x495);

// val X298 = ReLU()(X297)
JCudaTensor x496;
JCudaTensor x497;
x497 = x493;
x496 = x490.forward(x497);

// val X299 = Convolv(1,0)(X298,4b_a_cv_W,4b_a_cv_B)
JCudaTensor x498;
JCudaTensor x499, x500, x501;
x499 = x496;
x500 = x502;
x501 = x503;
x498 = x504.forward(x499, x500, x501);

// val X300 = BatchNorm(4b_a_bn)(X299,4b_a_bn_scale,4b_a_bn_bias)
JCudaTensor x505;
JCudaTensor x506, x507, x508;
x506 = x498;
x507 = x509;
x508 = x510;
x505 = x511.forward(x506, x507, x508);

// val X301 = ReLU()(X300)
JCudaTensor x512;
JCudaTensor x513;
x513 = x505;
x512 = x457.forward(x513);

// val X302 = Convolv(1,1)(X301,4b_b_cv_W,4b_b_cv_B)
JCudaTensor x514;
JCudaTensor x515, x516, x517;
x515 = x512;
x516 = x518;
x517 = x519;
x514 = x464.forward(x515, x516, x517);

// val X303 = BatchNorm(4b_b_bn)(X302,4b_b_bn_scale,4b_b_bn_bias)
JCudaTensor x520;
JCudaTensor x521, x522, x523;
x521 = x514;
x522 = x524;
x523 = x525;
x520 = x526.forward(x521, x522, x523);

// val X304 = ReLU()(X303)
JCudaTensor x527;
JCudaTensor x528;
x528 = x520;
x527 = x457.forward(x528);

// val X305 = Convolv(1,0)(X304,4b_c_cv_W,4b_c_cv_B)
JCudaTensor x529;
JCudaTensor x530, x531, x532;
x530 = x527;
x531 = x533;
x532 = x534;
x529 = x480.forward(x530, x531, x532);

// val X306 = BatchNorm(4b_c_bn)(X305,4b_c_bn_scale,4b_c_bn_bias)
JCudaTensor x535;
JCudaTensor x536, x537, x538;
x536 = x529;
x537 = x539;
x538 = x540;
x535 = x541.forward(x536, x537, x538);

// val X307 = ReLU()(X306)
JCudaTensor x542;
JCudaTensor x543;
x543 = x535;
x542 = x490.forward(x543);

// val X308 = (X307.copy + X298)
JCudaTensor x544;
JCudaTensor x545, x546;
x545 = x542;
x545 = x545.clone();
x546 = x496;
x544 = x545.plus_i(x546);

// val X309 = ReLU()(X308)
JCudaTensor x547;
JCudaTensor x548;
x548 = x544;
x547 = x490.forward(x548);

// val X310 = Convolv(1,0)(X309,4c_a_cv_W,4c_a_cv_B)
JCudaTensor x549;
JCudaTensor x550, x551, x552;
x550 = x547;
x551 = x553;
x552 = x554;
x549 = x504.forward(x550, x551, x552);

// val X311 = BatchNorm(4c_a_bn)(X310,4c_a_bn_scale,4c_a_bn_bias)
JCudaTensor x555;
JCudaTensor x556, x557, x558;
x556 = x549;
x557 = x559;
x558 = x560;
x555 = x561.forward(x556, x557, x558);

// val X312 = ReLU()(X311)
JCudaTensor x562;
JCudaTensor x563;
x563 = x555;
x562 = x457.forward(x563);

// val X313 = Convolv(1,1)(X312,4c_b_cv_W,4c_b_cv_B)
JCudaTensor x564;
JCudaTensor x565, x566, x567;
x565 = x562;
x566 = x568;
x567 = x569;
x564 = x464.forward(x565, x566, x567);

// val X314 = BatchNorm(4c_b_bn)(X313,4c_b_bn_scale,4c_b_bn_bias)
JCudaTensor x570;
JCudaTensor x571, x572, x573;
x571 = x564;
x572 = x574;
x573 = x575;
x570 = x576.forward(x571, x572, x573);

// val X315 = ReLU()(X314)
JCudaTensor x577;
JCudaTensor x578;
x578 = x570;
x577 = x457.forward(x578);

// val X316 = Convolv(1,0)(X315,4c_c_cv_W,4c_c_cv_B)
JCudaTensor x579;
JCudaTensor x580, x581, x582;
x580 = x577;
x581 = x583;
x582 = x584;
x579 = x480.forward(x580, x581, x582);

// val X317 = BatchNorm(4c_c_bn)(X316,4c_c_bn_scale,4c_c_bn_bias)
JCudaTensor x585;
JCudaTensor x586, x587, x588;
x586 = x579;
x587 = x589;
x588 = x590;
x585 = x591.forward(x586, x587, x588);

// val X318 = ReLU()(X317)
JCudaTensor x592;
JCudaTensor x593;
x593 = x585;
x592 = x490.forward(x593);

// val X319 = (X318.copy + X309)
JCudaTensor x594;
JCudaTensor x595, x596;
x595 = x592;
x595 = x595.clone();
x596 = x547;
x594 = x595.plus_i(x596);

// val X320 = ReLU()(X319)
JCudaTensor x597;
JCudaTensor x598;
x598 = x594;
x597 = x490.forward(x598);

// val X321 = Convolv(1,0)(X320,4d_a_cv_W,4d_a_cv_B)
JCudaTensor x599;
JCudaTensor x600, x601, x602;
x600 = x597;
x601 = x603;
x602 = x604;
x599 = x504.forward(x600, x601, x602);

// val X322 = BatchNorm(4d_a_bn)(X321,4d_a_bn_scale,4d_a_bn_bias)
JCudaTensor x605;
JCudaTensor x606, x607, x608;
x606 = x599;
x607 = x609;
x608 = x610;
x605 = x611.forward(x606, x607, x608);

// val X323 = ReLU()(X322)
JCudaTensor x612;
JCudaTensor x613;
x613 = x605;
x612 = x457.forward(x613);

// val X324 = Convolv(1,1)(X323,4d_b_cv_W,4d_b_cv_B)
JCudaTensor x614;
JCudaTensor x615, x616, x617;
x615 = x612;
x616 = x618;
x617 = x619;
x614 = x464.forward(x615, x616, x617);

// val X325 = BatchNorm(4d_b_bn)(X324,4d_b_bn_scale,4d_b_bn_bias)
JCudaTensor x620;
JCudaTensor x621, x622, x623;
x621 = x614;
x622 = x624;
x623 = x625;
x620 = x626.forward(x621, x622, x623);

// val X326 = ReLU()(X325)
JCudaTensor x627;
JCudaTensor x628;
x628 = x620;
x627 = x457.forward(x628);

// val X327 = Convolv(1,0)(X326,4d_c_cv_W,4d_c_cv_B)
JCudaTensor x629;
JCudaTensor x630, x631, x632;
x630 = x627;
x631 = x633;
x632 = x634;
x629 = x480.forward(x630, x631, x632);

// val X328 = BatchNorm(4d_c_bn)(X327,4d_c_bn_scale,4d_c_bn_bias)
JCudaTensor x635;
JCudaTensor x636, x637, x638;
x636 = x629;
x637 = x639;
x638 = x640;
x635 = x641.forward(x636, x637, x638);

// val X329 = ReLU()(X328)
JCudaTensor x642;
JCudaTensor x643;
x643 = x635;
x642 = x490.forward(x643);

// val X330 = (X329.copy + X320)
JCudaTensor x644;
JCudaTensor x645, x646;
x645 = x642;
x645 = x645.clone();
x646 = x597;
x644 = x645.plus_i(x646);

// val X331 = ReLU()(X330)
JCudaTensor x647;
JCudaTensor x648;
x648 = x644;
x647 = x490.forward(x648);

// val X332 = Convolv(1,0)(X331,4e_a_cv_W,4e_a_cv_B)
JCudaTensor x649;
JCudaTensor x650, x651, x652;
x650 = x647;
x651 = x653;
x652 = x654;
x649 = x504.forward(x650, x651, x652);

// val X333 = BatchNorm(4e_a_bn)(X332,4e_a_bn_scale,4e_a_bn_bias)
JCudaTensor x655;
JCudaTensor x656, x657, x658;
x656 = x649;
x657 = x659;
x658 = x660;
x655 = x661.forward(x656, x657, x658);

// val X334 = ReLU()(X333)
JCudaTensor x662;
JCudaTensor x663;
x663 = x655;
x662 = x457.forward(x663);

// val X335 = Convolv(1,1)(X334,4e_b_cv_W,4e_b_cv_B)
JCudaTensor x664;
JCudaTensor x665, x666, x667;
x665 = x662;
x666 = x668;
x667 = x669;
x664 = x464.forward(x665, x666, x667);

// val X336 = BatchNorm(4e_b_bn)(X335,4e_b_bn_scale,4e_b_bn_bias)
JCudaTensor x670;
JCudaTensor x671, x672, x673;
x671 = x664;
x672 = x674;
x673 = x675;
x670 = x676.forward(x671, x672, x673);

// val X337 = ReLU()(X336)
JCudaTensor x677;
JCudaTensor x678;
x678 = x670;
x677 = x457.forward(x678);

// val X338 = Convolv(1,0)(X337,4e_c_cv_W,4e_c_cv_B)
JCudaTensor x679;
JCudaTensor x680, x681, x682;
x680 = x677;
x681 = x683;
x682 = x684;
x679 = x480.forward(x680, x681, x682);

// val X339 = BatchNorm(4e_c_bn)(X338,4e_c_bn_scale,4e_c_bn_bias)
JCudaTensor x685;
JCudaTensor x686, x687, x688;
x686 = x679;
x687 = x689;
x688 = x690;
x685 = x691.forward(x686, x687, x688);

// val X340 = ReLU()(X339)
JCudaTensor x692;
JCudaTensor x693;
x693 = x685;
x692 = x490.forward(x693);

// val X341 = (X340.copy + X331)
JCudaTensor x694;
JCudaTensor x695, x696;
x695 = x692;
x695 = x695.clone();
x696 = x647;
x694 = x695.plus_i(x696);

// val X342 = ReLU()(X341)
JCudaTensor x697;
JCudaTensor x698;
x698 = x694;
x697 = x490.forward(x698);

// val X343 = Convolv(1,0)(X342,4f_a_cv_W,4f_a_cv_B)
JCudaTensor x699;
JCudaTensor x700, x701, x702;
x700 = x697;
x701 = x703;
x702 = x704;
x699 = x504.forward(x700, x701, x702);

// val X344 = BatchNorm(4f_a_bn)(X343,4f_a_bn_scale,4f_a_bn_bias)
JCudaTensor x705;
JCudaTensor x706, x707, x708;
x706 = x699;
x707 = x709;
x708 = x710;
x705 = x711.forward(x706, x707, x708);

// val X345 = ReLU()(X344)
JCudaTensor x712;
JCudaTensor x713;
x713 = x705;
x712 = x457.forward(x713);

// val X346 = Convolv(1,1)(X345,4f_b_cv_W,4f_b_cv_B)
JCudaTensor x714;
JCudaTensor x715, x716, x717;
x715 = x712;
x716 = x718;
x717 = x719;
x714 = x464.forward(x715, x716, x717);

// val X347 = BatchNorm(4f_b_bn)(X346,4f_b_bn_scale,4f_b_bn_bias)
JCudaTensor x720;
JCudaTensor x721, x722, x723;
x721 = x714;
x722 = x724;
x723 = x725;
x720 = x726.forward(x721, x722, x723);

// val X348 = ReLU()(X347)
JCudaTensor x727;
JCudaTensor x728;
x728 = x720;
x727 = x457.forward(x728);

// val X349 = Convolv(1,0)(X348,4f_c_cv_W,4f_c_cv_B)
JCudaTensor x729;
JCudaTensor x730, x731, x732;
x730 = x727;
x731 = x733;
x732 = x734;
x729 = x480.forward(x730, x731, x732);

// val X350 = BatchNorm(4f_c_bn)(X349,4f_c_bn_scale,4f_c_bn_bias)
JCudaTensor x735;
JCudaTensor x736, x737, x738;
x736 = x729;
x737 = x739;
x738 = x740;
x735 = x741.forward(x736, x737, x738);

// val X351 = ReLU()(X350)
JCudaTensor x742;
JCudaTensor x743;
x743 = x735;
x742 = x490.forward(x743);

// val X352 = (X351.copy + X342)
JCudaTensor x744;
JCudaTensor x745, x746;
x745 = x742;
x745 = x745.clone();
x746 = x697;
x744 = x745.plus_i(x746);

// val X353 = ReLU()(X352)
JCudaTensor x747;
JCudaTensor x748;
x748 = x744;
x747 = x490.forward(x748);

// val X354 = Convolv(2,0)(X353,5a1_cv_W,5a1_cv_B)
JCudaTensor x749;
JCudaTensor x750, x751, x752;
x750 = x747;
x751 = x753;
x752 = x754;
x749 = x755.forward(x750, x751, x752);

// val X357 = Convolv(2,0)(X353,5a2_a_cv_W,5a2_a_cv_B)
JCudaTensor x756;
JCudaTensor x757, x758, x759;
x757 = x747;
x758 = x760;
x759 = x761;
x756 = x762.forward(x757, x758, x759);

// val X355 = BatchNorm(5a1_bn)(X354,5a1_bn_scale,5a1_bn_bias)
JCudaTensor x763;
JCudaTensor x764, x765, x766;
x764 = x749;
x765 = x767;
x766 = x768;
x763 = x769.forward(x764, x765, x766);

// val X358 = BatchNorm(5a2_a_bn)(X357,5a2_a_bn_scale,5a2_a_bn_bias)
JCudaTensor x770;
JCudaTensor x771, x772, x773;
x771 = x756;
x772 = x774;
x773 = x775;
x770 = x776.forward(x771, x772, x773);

// val X359 = ReLU()(X358)
JCudaTensor x777;
JCudaTensor x778;
x778 = x770;
x777 = x779.forward(x778);

// val X360 = Convolv(1,1)(X359,5a2_b_cv_W,5a2_b_cv_B)
JCudaTensor x780;
JCudaTensor x781, x782, x783;
x781 = x777;
x782 = x784;
x783 = x785;
x780 = x786.forward(x781, x782, x783);

// val X361 = BatchNorm(5a2_b_bn)(X360,5a2_b_bn_scale,5a2_b_bn_bias)
JCudaTensor x787;
JCudaTensor x788, x789, x790;
x788 = x780;
x789 = x791;
x790 = x792;
x787 = x793.forward(x788, x789, x790);

// val X362 = ReLU()(X361)
JCudaTensor x794;
JCudaTensor x795;
x795 = x787;
x794 = x779.forward(x795);

// val X363 = Convolv(1,0)(X362,5a2_c_cv_W,5a2_c_cv_B)
JCudaTensor x796;
JCudaTensor x797, x798, x799;
x797 = x794;
x798 = x800;
x799 = x801;
x796 = x802.forward(x797, x798, x799);

// val X364 = BatchNorm(5a2_c_bn)(X363,5a2_c_bn_scale,5a2_c_bn_bias)
JCudaTensor x803;
JCudaTensor x804, x805, x806;
x804 = x796;
x805 = x807;
x806 = x808;
x803 = x809.forward(x804, x805, x806);

// val X356 = ReLU()(X355)
JCudaTensor x810;
JCudaTensor x811;
x811 = x763;
x810 = x812.forward(x811);

// val X365 = ReLU()(X364)
JCudaTensor x813;
JCudaTensor x814;
x814 = x803;
x813 = x812.forward(x814);

// val X366 = (X356.copy + X365)
JCudaTensor x815;
JCudaTensor x816, x817;
x816 = x810;
x816 = x816.clone();
x817 = x813;
x815 = x816.plus_i(x817);

// val X367 = ReLU()(X366)
JCudaTensor x818;
JCudaTensor x819;
x819 = x815;
x818 = x812.forward(x819);

// val X368 = Convolv(1,0)(X367,5b_a_cv_W,5b_a_cv_B)
JCudaTensor x820;
JCudaTensor x821, x822, x823;
x821 = x818;
x822 = x824;
x823 = x825;
x820 = x826.forward(x821, x822, x823);

// val X369 = BatchNorm(5b_a_bn)(X368,5b_a_bn_scale,5b_a_bn_bias)
JCudaTensor x827;
JCudaTensor x828, x829, x830;
x828 = x820;
x829 = x831;
x830 = x832;
x827 = x833.forward(x828, x829, x830);

// val X370 = ReLU()(X369)
JCudaTensor x834;
JCudaTensor x835;
x835 = x827;
x834 = x779.forward(x835);

// val X371 = Convolv(1,1)(X370,5b_b_cv_W,5b_b_cv_B)
JCudaTensor x836;
JCudaTensor x837, x838, x839;
x837 = x834;
x838 = x840;
x839 = x841;
x836 = x786.forward(x837, x838, x839);

// val X372 = BatchNorm(5b_b_bn)(X371,5b_b_bn_scale,5b_b_bn_bias)
JCudaTensor x842;
JCudaTensor x843, x844, x845;
x843 = x836;
x844 = x846;
x845 = x847;
x842 = x848.forward(x843, x844, x845);

// val X373 = ReLU()(X372)
JCudaTensor x849;
JCudaTensor x850;
x850 = x842;
x849 = x779.forward(x850);

// val X374 = Convolv(1,0)(X373,5b_c_cv_W,5b_c_cv_B)
JCudaTensor x851;
JCudaTensor x852, x853, x854;
x852 = x849;
x853 = x855;
x854 = x856;
x851 = x802.forward(x852, x853, x854);

// val X375 = BatchNorm(5b_c_bn)(X374,5b_c_bn_scale,5b_c_bn_bias)
JCudaTensor x857;
JCudaTensor x858, x859, x860;
x858 = x851;
x859 = x861;
x860 = x862;
x857 = x863.forward(x858, x859, x860);

// val X376 = ReLU()(X375)
JCudaTensor x864;
JCudaTensor x865;
x865 = x857;
x864 = x812.forward(x865);

// val X377 = (X376.copy + X367)
JCudaTensor x866;
JCudaTensor x867, x868;
x867 = x864;
x867 = x867.clone();
x868 = x818;
x866 = x867.plus_i(x868);

// val X378 = ReLU()(X377)
JCudaTensor x869;
JCudaTensor x870;
x870 = x866;
x869 = x812.forward(x870);

// val X379 = Convolv(1,0)(X378,5c_a_cv_W,5c_a_cv_B)
JCudaTensor x871;
JCudaTensor x872, x873, x874;
x872 = x869;
x873 = x875;
x874 = x876;
x871 = x826.forward(x872, x873, x874);

// val X380 = BatchNorm(5c_a_bn)(X379,5c_a_bn_scale,5c_a_bn_bias)
JCudaTensor x877;
JCudaTensor x878, x879, x880;
x878 = x871;
x879 = x881;
x880 = x882;
x877 = x883.forward(x878, x879, x880);

// val X381 = ReLU()(X380)
JCudaTensor x884;
JCudaTensor x885;
x885 = x877;
x884 = x779.forward(x885);

// val X382 = Convolv(1,1)(X381,5c_b_cv_W,5c_b_cv_B)
JCudaTensor x886;
JCudaTensor x887, x888, x889;
x887 = x884;
x888 = x890;
x889 = x891;
x886 = x786.forward(x887, x888, x889);

// val X383 = BatchNorm(5c_b_bn)(X382,5c_b_bn_scale,5c_b_bn_bias)
JCudaTensor x892;
JCudaTensor x893, x894, x895;
x893 = x886;
x894 = x896;
x895 = x897;
x892 = x898.forward(x893, x894, x895);

// val X384 = ReLU()(X383)
JCudaTensor x899;
JCudaTensor x900;
x900 = x892;
x899 = x779.forward(x900);

// val X385 = Convolv(1,0)(X384,5c_c_cv_W,5c_c_cv_B)
JCudaTensor x901;
JCudaTensor x902, x903, x904;
x902 = x899;
x903 = x905;
x904 = x906;
x901 = x802.forward(x902, x903, x904);

// val X386 = BatchNorm(5c_c_bn)(X385,5c_c_bn_scale,5c_c_bn_bias)
JCudaTensor x907;
JCudaTensor x908, x909, x910;
x908 = x901;
x909 = x911;
x910 = x912;
x907 = x913.forward(x908, x909, x910);

// val X387 = ReLU()(X386)
JCudaTensor x914;
JCudaTensor x915;
x915 = x907;
x914 = x812.forward(x915);

// val X388 = (X387.copy + X378)
JCudaTensor x916;
JCudaTensor x917, x918;
x917 = x914;
x917 = x917.clone();
x918 = x869;
x916 = x917.plus_i(x918);

// val X389 = ReLU()(X388)
JCudaTensor x919;
JCudaTensor x920;
x920 = x916;
x919 = x812.forward(x920);

// val X390 = Pooling(7,1,0,false)(X389)
JCudaTensor x921;
JCudaTensor x922;
x922 = x919;
x921 = x923.forward(x922);

// val X391 = (X390[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x924;
JCudaMatrix x925;
JCudaMatrix x926;
JCudaTensor x927;
JCudaTensor x928;
x928 = x921;
x927 = x928.flatten(1, new int[]{2048, 1, 1});
x925 = x927.asMatrix(1, true);
JCudaTensor x929;
x929 = x930;
x926 = x929.asMatrix(1, true);
x924 = x925.times(x926);

// val X393 = (X391 + (i) => fc_B)
JCudaTensor x931;
JCudaTensor x932, x933;
x932 = x924;
x933 = x934;
x931 = x933.copy(64, x932);

// val X394 = LogSoftmax()(X393)
JCudaTensor x935;
JCudaTensor x936;
x936 = x931;
x935 = x937.forward(x936);

// Dealloc(X393)
JCudaTensor x938;
x938 = x931;
x938.free();

// val X1140 = (X1139 / |64|)
JCudaTensor x939;
JCudaTensor x940;
float x941;
x940 = x11;
float x942;
x942 = 64;
x941 = 1 / x942;
x939 = x940.times_i(x941);

// Print(((0 - (X395 . X394)) / |64|))
float x943;
float x944;
float x945;
float x946;
JCudaTensor x947, x948;
x947 = x9;
x948 = x935;
x946 = x947.dot(x948);
x944 = - x946;
x945 = 64;
x943 = x944 / x945;
System.out.println(x5 + " " + x943);
if (Float.isNaN(x943)) { System.exit(-1); }

// Dealloc(X395)
JCudaTensor x949;
x949 = x9;
x949.free();

// val X1339 = X1140 * d_LogSoftmax()(X394)/d_X393
JCudaTensor x950;
JCudaTensor x951, x952;
x951 = x939;
x952 = x935;
x950 = x937.backward(x951, x952);

// Dealloc(X1140)
JCudaTensor x953;
x953 = x939;
x953.free();

// Dealloc(X394)
JCudaTensor x954;
x954 = x935;
x954.free();

// val m1 = (i19) => fc_W[@, i19]
JCudaMatrix x955;
JCudaTensor x956;
x956 = x930;
x955 = x956.asMatrix(1, false);

// val m160 = (i96656) => X1339[@, i96656]
JCudaMatrix x957;
JCudaTensor x958;
x958 = x950;
x957 = x958.asMatrix(1, false);

// val m162 = (i96664) => X390[1><3][@, i96664]
JCudaMatrix x959;
JCudaTensor x960;
JCudaTensor x961;
x961 = x921;
x960 = x961.flatten(1, new int[]{2048, 1, 1});
x959 = x960.asMatrix(1, false);

// val X1533 = (X1339)(i18 | @) * m1
JCudaTensor x962;
JCudaMatrix x963;
JCudaMatrix x964;
JCudaTensor x965;
x965 = x950;
x963 = x965.asMatrix(1, true);
x964 = x955;
x962 = x963.times(x964);

// V_fc_W <~~ m160 * m162
float x967, x968;
x967 = lrn_rate_1;
x968 = momentum;
JCudaMatrix x969;
JCudaMatrix x970;
x969 = x957;
x970 = x959;
x969.times(x970, x966, x967, x968);

// V_fc_B <~~ Sum(m160)
float x972, x973;
x972 = lrn_rate_1;
x973 = momentum;
JCudaMatrix x974;
x974 = x957;
x974.sum(x971, x972, x973);

// Dealloc(X1339)
JCudaTensor x975;
x975 = x950;
x975.free();

// val X1535 = X1533[1<>3] * d_Pooling(7,1,0,false)(X390,X389)/d_X389
JCudaTensor x976;
JCudaTensor x977, x978, x979;
JCudaTensor x980;
x980 = x962;
x977 = x980.unflatten(1, new int[]{2048, 1, 1});
x978 = x921;
x979 = x919;
x976 = x923.backward(x977, x978, x979);

// Dealloc(X1533)
JCudaTensor x981;
x981 = x962;
x981.free();

// Dealloc(X390)
JCudaTensor x982;
x982 = x921;
x982.free();

// fc_W <~~ V_fc_W
float x983, x984;
x983 = 1;
x984 = decay_1;
JCudaTensor x985;
x985 = x966;
x930.update(x985, x983, x984);

// fc_B <~~ V_fc_B
float x986, x987;
x986 = 1;
x987 = decay_1;
JCudaTensor x988;
x988 = x971;
x934.update(x988, x986, x987);

// val X1572 = X1535 * d_ReLU()(X389)/d_X388
JCudaTensor x989;
JCudaTensor x990, x991;
x990 = x976;
x991 = x919;
x989 = x812.backward(x990, x991);

// Dealloc(X389)
JCudaTensor x992;
x992 = x919;
x992.free();

// val X1582 = X1572.copy * d_ReLU()(X387)/d_X386
JCudaTensor x993;
JCudaTensor x994, x995;
x994 = x989;
x994 = x994.clone();
x995 = x914;
x993 = x812.backward(x994, x995);

// Dealloc(X387)
JCudaTensor x996;
x996 = x914;
x996.free();

// val X206559 = X1582 * d_BatchNorm(5c_c_bn)(X385,5c_c_bn_scale)/d_5c_c_bn_bias
JCudaTensor x997;
JCudaTensor x998, x999, x1000;
x998 = x993;
x999 = x901;
x1000 = x911;
JCudaTensor[] x1001 = x913.backward(x998,x999,x1000);
x997 = x1001[2];

// val X1583 = X1582 * d_BatchNorm(5c_c_bn)(X385,5c_c_bn_scale)/d_X385
JCudaTensor x1002;
x1002 = x1001[0];

// val X207230 = X1582 * d_BatchNorm(5c_c_bn)(X385,5c_c_bn_scale)/d_5c_c_bn_scale
JCudaTensor x1006;
x1006 = x1001[1];

// Dealloc(X385)
JCudaTensor x1010;
x1010 = x901;
x1010.free();

// val X1584 = X1583 * d_Convolv(1,0)(5c_c_cv_W)/d_X384
JCudaTensor x1011;
JCudaTensor x1012, x1013;
x1012 = x1002;
x1013 = x905;
x1011 = x802.backward_data(x1012, x1013);

// V_5c_c_cv_W <~~ X1583 * d_Convolv(1,0)(X384)/d_5c_c_cv_W
float x1015, x1016;
x1015 = lrn_rate_1;
x1016 = momentum;
JCudaTensor x1017, x1018;
x1017 = x1002;
x1018 = x899;
x802.backward_filter(x1017, x1018, x1014, x1015, x1016);

// Dealloc(X1583)
JCudaTensor x1019;
x1019 = x1002;
x1019.free();

// V_5c_c_bn_scale <~~ X207230
float x1021, x1022;
x1021 = lrn_rate_1;
x1022 = momentum;
JCudaTensor x1023;
x1023 = x1006;
x1020.update(x1023, x1021, x1022);

// Dealloc(X207230)
JCudaTensor x1024;
x1024 = x1006;
x1024.free();

// V_5c_c_bn_bias <~~ X206559
float x1026, x1027;
x1026 = lrn_rate_1;
x1027 = momentum;
JCudaTensor x1028;
x1028 = x997;
x1025.update(x1028, x1026, x1027);

// Dealloc(X206559)
JCudaTensor x1029;
x1029 = x997;
x1029.free();

// 5c_c_cv_W <~~ V_5c_c_cv_W
float x1030, x1031;
x1030 = 1;
x1031 = decay_1;
JCudaTensor x1032;
x1032 = x1014;
x905.update(x1032, x1030, x1031);

// 5c_c_bn_scale <~~ V_5c_c_bn_scale
float x1033, x1034;
x1033 = 1;
x1034 = decay_1;
JCudaTensor x1035;
x1035 = x1020;
x911.update(x1035, x1033, x1034);

// 5c_c_bn_bias <~~ V_5c_c_bn_bias
float x1036, x1037;
x1036 = 1;
x1037 = decay_1;
JCudaTensor x1038;
x1038 = x1025;
x912.update(x1038, x1036, x1037);

// val X1588 = X1584 * d_ReLU()(X384)/d_X383
JCudaTensor x1039;
JCudaTensor x1040, x1041;
x1040 = x1011;
x1041 = x899;
x1039 = x779.backward(x1040, x1041);

// Dealloc(X384)
JCudaTensor x1042;
x1042 = x899;
x1042.free();

// val X205202 = X1588 * d_BatchNorm(5c_b_bn)(X382,5c_b_bn_scale)/d_5c_b_bn_scale
JCudaTensor x1043;
JCudaTensor x1044, x1045, x1046;
x1044 = x1039;
x1045 = x886;
x1046 = x896;
JCudaTensor[] x1047 = x898.backward(x1044,x1045,x1046);
x1043 = x1047[1];

// val X1589 = X1588 * d_BatchNorm(5c_b_bn)(X382,5c_b_bn_scale)/d_X382
JCudaTensor x1048;
x1048 = x1047[0];

// val X204518 = X1588 * d_BatchNorm(5c_b_bn)(X382,5c_b_bn_scale)/d_5c_b_bn_bias
JCudaTensor x1052;
x1052 = x1047[2];

// Dealloc(X382)
JCudaTensor x1056;
x1056 = x886;
x1056.free();

// V_5c_b_bn_bias <~~ X204518
float x1058, x1059;
x1058 = lrn_rate_1;
x1059 = momentum;
JCudaTensor x1060;
x1060 = x1052;
x1057.update(x1060, x1058, x1059);

// Dealloc(X204518)
JCudaTensor x1061;
x1061 = x1052;
x1061.free();

// val X1590 = X1589 * d_Convolv(1,1)(5c_b_cv_W)/d_X381
JCudaTensor x1062;
JCudaTensor x1063, x1064;
x1063 = x1048;
x1064 = x890;
x1062 = x786.backward_data(x1063, x1064);

// V_5c_b_cv_W <~~ X1589 * d_Convolv(1,1)(X381)/d_5c_b_cv_W
float x1066, x1067;
x1066 = lrn_rate_1;
x1067 = momentum;
JCudaTensor x1068, x1069;
x1068 = x1048;
x1069 = x884;
x786.backward_filter(x1068, x1069, x1065, x1066, x1067);

// Dealloc(X1589)
JCudaTensor x1070;
x1070 = x1048;
x1070.free();

// V_5c_b_bn_scale <~~ X205202
float x1072, x1073;
x1072 = lrn_rate_1;
x1073 = momentum;
JCudaTensor x1074;
x1074 = x1043;
x1071.update(x1074, x1072, x1073);

// Dealloc(X205202)
JCudaTensor x1075;
x1075 = x1043;
x1075.free();

// 5c_b_bn_bias <~~ V_5c_b_bn_bias
float x1076, x1077;
x1076 = 1;
x1077 = decay_1;
JCudaTensor x1078;
x1078 = x1057;
x897.update(x1078, x1076, x1077);

// 5c_b_cv_W <~~ V_5c_b_cv_W
float x1079, x1080;
x1079 = 1;
x1080 = decay_1;
JCudaTensor x1081;
x1081 = x1065;
x890.update(x1081, x1079, x1080);

// 5c_b_bn_scale <~~ V_5c_b_bn_scale
float x1082, x1083;
x1082 = 1;
x1083 = decay_1;
JCudaTensor x1084;
x1084 = x1071;
x896.update(x1084, x1082, x1083);

// val X1594 = X1590 * d_ReLU()(X381)/d_X380
JCudaTensor x1085;
JCudaTensor x1086, x1087;
x1086 = x1062;
x1087 = x884;
x1085 = x779.backward(x1086, x1087);

// Dealloc(X381)
JCudaTensor x1088;
x1088 = x884;
x1088.free();

// val X202438 = X1594 * d_BatchNorm(5c_a_bn)(X379,5c_a_bn_scale)/d_5c_a_bn_bias
JCudaTensor x1089;
JCudaTensor x1090, x1091, x1092;
x1090 = x1085;
x1091 = x871;
x1092 = x881;
JCudaTensor[] x1093 = x883.backward(x1090,x1091,x1092);
x1089 = x1093[2];

// val X1595 = X1594 * d_BatchNorm(5c_a_bn)(X379,5c_a_bn_scale)/d_X379
JCudaTensor x1094;
x1094 = x1093[0];

// val X203135 = X1594 * d_BatchNorm(5c_a_bn)(X379,5c_a_bn_scale)/d_5c_a_bn_scale
JCudaTensor x1098;
x1098 = x1093[1];

// Dealloc(X379)
JCudaTensor x1102;
x1102 = x871;
x1102.free();

// V_5c_a_bn_bias <~~ X202438
float x1104, x1105;
x1104 = lrn_rate_1;
x1105 = momentum;
JCudaTensor x1106;
x1106 = x1089;
x1103.update(x1106, x1104, x1105);

// Dealloc(X202438)
JCudaTensor x1107;
x1107 = x1089;
x1107.free();

// val X1596 = X1595 * d_Convolv(1,0)(5c_a_cv_W)/d_X378
JCudaTensor x1108;
JCudaTensor x1109, x1110;
x1109 = x1094;
x1110 = x875;
x1108 = x826.backward_data(x1109, x1110);

// V_5c_a_bn_scale <~~ X203135
float x1112, x1113;
x1112 = lrn_rate_1;
x1113 = momentum;
JCudaTensor x1114;
x1114 = x1098;
x1111.update(x1114, x1112, x1113);

// Dealloc(X203135)
JCudaTensor x1115;
x1115 = x1098;
x1115.free();

// V_5c_a_cv_W <~~ X1595 * d_Convolv(1,0)(X378)/d_5c_a_cv_W
float x1117, x1118;
x1117 = lrn_rate_1;
x1118 = momentum;
JCudaTensor x1119, x1120;
x1119 = x1094;
x1120 = x869;
x826.backward_filter(x1119, x1120, x1116, x1117, x1118);

// Dealloc(X1595)
JCudaTensor x1121;
x1121 = x1094;
x1121.free();

// 5c_a_bn_bias <~~ V_5c_a_bn_bias
float x1122, x1123;
x1122 = 1;
x1123 = decay_1;
JCudaTensor x1124;
x1124 = x1103;
x882.update(x1124, x1122, x1123);

// 5c_a_bn_scale <~~ V_5c_a_bn_scale
float x1125, x1126;
x1125 = 1;
x1126 = decay_1;
JCudaTensor x1127;
x1127 = x1111;
x881.update(x1127, x1125, x1126);

// 5c_a_cv_W <~~ V_5c_a_cv_W
float x1128, x1129;
x1128 = 1;
x1129 = decay_1;
JCudaTensor x1130;
x1130 = x1116;
x875.update(x1130, x1128, x1129);

// val X1597 = (X1596 + X1572)
JCudaTensor x1131;
JCudaTensor x1132, x1133;
x1132 = x1108;
x1133 = x989;
x1131 = x1132.plus_i(x1133);

// Dealloc(X1572)
JCudaTensor x1134;
x1134 = x989;
x1134.free();

// val X1609 = X1597 * d_ReLU()(X378)/d_X377
JCudaTensor x1135;
JCudaTensor x1136, x1137;
x1136 = x1131;
x1137 = x869;
x1135 = x812.backward(x1136, x1137);

// Dealloc(X378)
JCudaTensor x1138;
x1138 = x869;
x1138.free();

// val X1619 = X1609.copy * d_ReLU()(X376)/d_X375
JCudaTensor x1139;
JCudaTensor x1140, x1141;
x1140 = x1135;
x1140 = x1140.clone();
x1141 = x864;
x1139 = x812.backward(x1140, x1141);

// Dealloc(X376)
JCudaTensor x1142;
x1142 = x864;
x1142.free();

// val X201001 = X1619 * d_BatchNorm(5b_c_bn)(X374,5b_c_bn_scale)/d_5b_c_bn_scale
JCudaTensor x1143;
JCudaTensor x1144, x1145, x1146;
x1144 = x1139;
x1145 = x851;
x1146 = x861;
JCudaTensor[] x1147 = x863.backward(x1144,x1145,x1146);
x1143 = x1147[1];

// val X1620 = X1619 * d_BatchNorm(5b_c_bn)(X374,5b_c_bn_scale)/d_X374
JCudaTensor x1148;
x1148 = x1147[0];

// val X200263 = X1619 * d_BatchNorm(5b_c_bn)(X374,5b_c_bn_scale)/d_5b_c_bn_bias
JCudaTensor x1152;
x1152 = x1147[2];

// Dealloc(X374)
JCudaTensor x1156;
x1156 = x851;
x1156.free();

// val X1621 = X1620 * d_Convolv(1,0)(5b_c_cv_W)/d_X373
JCudaTensor x1157;
JCudaTensor x1158, x1159;
x1158 = x1148;
x1159 = x855;
x1157 = x802.backward_data(x1158, x1159);

// V_5b_c_cv_W <~~ X1620 * d_Convolv(1,0)(X373)/d_5b_c_cv_W
float x1161, x1162;
x1161 = lrn_rate_1;
x1162 = momentum;
JCudaTensor x1163, x1164;
x1163 = x1148;
x1164 = x849;
x802.backward_filter(x1163, x1164, x1160, x1161, x1162);

// Dealloc(X1620)
JCudaTensor x1165;
x1165 = x1148;
x1165.free();

// V_5b_c_bn_scale <~~ X201001
float x1167, x1168;
x1167 = lrn_rate_1;
x1168 = momentum;
JCudaTensor x1169;
x1169 = x1143;
x1166.update(x1169, x1167, x1168);

// Dealloc(X201001)
JCudaTensor x1170;
x1170 = x1143;
x1170.free();

// V_5b_c_bn_bias <~~ X200263
float x1172, x1173;
x1172 = lrn_rate_1;
x1173 = momentum;
JCudaTensor x1174;
x1174 = x1152;
x1171.update(x1174, x1172, x1173);

// Dealloc(X200263)
JCudaTensor x1175;
x1175 = x1152;
x1175.free();

// 5b_c_cv_W <~~ V_5b_c_cv_W
float x1176, x1177;
x1176 = 1;
x1177 = decay_1;
JCudaTensor x1178;
x1178 = x1160;
x855.update(x1178, x1176, x1177);

// 5b_c_bn_scale <~~ V_5b_c_bn_scale
float x1179, x1180;
x1179 = 1;
x1180 = decay_1;
JCudaTensor x1181;
x1181 = x1166;
x861.update(x1181, x1179, x1180);

// 5b_c_bn_bias <~~ V_5b_c_bn_bias
float x1182, x1183;
x1182 = 1;
x1183 = decay_1;
JCudaTensor x1184;
x1184 = x1171;
x862.update(x1184, x1182, x1183);

// val X1625 = X1621 * d_ReLU()(X373)/d_X372
JCudaTensor x1185;
JCudaTensor x1186, x1187;
x1186 = x1157;
x1187 = x849;
x1185 = x779.backward(x1186, x1187);

// Dealloc(X373)
JCudaTensor x1188;
x1188 = x849;
x1188.free();

// val X198772 = X1625 * d_BatchNorm(5b_b_bn)(X371,5b_b_bn_scale)/d_5b_b_bn_scale
JCudaTensor x1189;
JCudaTensor x1190, x1191, x1192;
x1190 = x1185;
x1191 = x836;
x1192 = x846;
JCudaTensor[] x1193 = x848.backward(x1190,x1191,x1192);
x1189 = x1193[1];

// val X198021 = X1625 * d_BatchNorm(5b_b_bn)(X371,5b_b_bn_scale)/d_5b_b_bn_bias
JCudaTensor x1194;
x1194 = x1193[2];

// val X1626 = X1625 * d_BatchNorm(5b_b_bn)(X371,5b_b_bn_scale)/d_X371
JCudaTensor x1198;
x1198 = x1193[0];

// Dealloc(X371)
JCudaTensor x1202;
x1202 = x836;
x1202.free();

// V_5b_b_bn_bias <~~ X198021
float x1204, x1205;
x1204 = lrn_rate_1;
x1205 = momentum;
JCudaTensor x1206;
x1206 = x1194;
x1203.update(x1206, x1204, x1205);

// Dealloc(X198021)
JCudaTensor x1207;
x1207 = x1194;
x1207.free();

// V_5b_b_bn_scale <~~ X198772
float x1209, x1210;
x1209 = lrn_rate_1;
x1210 = momentum;
JCudaTensor x1211;
x1211 = x1189;
x1208.update(x1211, x1209, x1210);

// Dealloc(X198772)
JCudaTensor x1212;
x1212 = x1189;
x1212.free();

// val X1627 = X1626 * d_Convolv(1,1)(5b_b_cv_W)/d_X370
JCudaTensor x1213;
JCudaTensor x1214, x1215;
x1214 = x1198;
x1215 = x840;
x1213 = x786.backward_data(x1214, x1215);

// V_5b_b_cv_W <~~ X1626 * d_Convolv(1,1)(X370)/d_5b_b_cv_W
float x1217, x1218;
x1217 = lrn_rate_1;
x1218 = momentum;
JCudaTensor x1219, x1220;
x1219 = x1198;
x1220 = x834;
x786.backward_filter(x1219, x1220, x1216, x1217, x1218);

// Dealloc(X1626)
JCudaTensor x1221;
x1221 = x1198;
x1221.free();

// 5b_b_bn_bias <~~ V_5b_b_bn_bias
float x1222, x1223;
x1222 = 1;
x1223 = decay_1;
JCudaTensor x1224;
x1224 = x1203;
x847.update(x1224, x1222, x1223);

// 5b_b_bn_scale <~~ V_5b_b_bn_scale
float x1225, x1226;
x1225 = 1;
x1226 = decay_1;
JCudaTensor x1227;
x1227 = x1208;
x846.update(x1227, x1225, x1226);

// 5b_b_cv_W <~~ V_5b_b_cv_W
float x1228, x1229;
x1228 = 1;
x1229 = decay_1;
JCudaTensor x1230;
x1230 = x1216;
x840.update(x1230, x1228, x1229);

// val X1631 = X1627 * d_ReLU()(X370)/d_X369
JCudaTensor x1231;
JCudaTensor x1232, x1233;
x1232 = x1213;
x1233 = x834;
x1231 = x779.backward(x1232, x1233);

// Dealloc(X370)
JCudaTensor x1234;
x1234 = x834;
x1234.free();

// val X196504 = X1631 * d_BatchNorm(5b_a_bn)(X368,5b_a_bn_scale)/d_5b_a_bn_scale
JCudaTensor x1235;
JCudaTensor x1236, x1237, x1238;
x1236 = x1231;
x1237 = x820;
x1238 = x831;
JCudaTensor[] x1239 = x833.backward(x1236,x1237,x1238);
x1235 = x1239[1];

// val X195740 = X1631 * d_BatchNorm(5b_a_bn)(X368,5b_a_bn_scale)/d_5b_a_bn_bias
JCudaTensor x1240;
x1240 = x1239[2];

// val X1632 = X1631 * d_BatchNorm(5b_a_bn)(X368,5b_a_bn_scale)/d_X368
JCudaTensor x1244;
x1244 = x1239[0];

// Dealloc(X368)
JCudaTensor x1248;
x1248 = x820;
x1248.free();

// val X1633 = X1632 * d_Convolv(1,0)(5b_a_cv_W)/d_X367
JCudaTensor x1249;
JCudaTensor x1250, x1251;
x1250 = x1244;
x1251 = x824;
x1249 = x826.backward_data(x1250, x1251);

// V_5b_a_bn_scale <~~ X196504
float x1253, x1254;
x1253 = lrn_rate_1;
x1254 = momentum;
JCudaTensor x1255;
x1255 = x1235;
x1252.update(x1255, x1253, x1254);

// Dealloc(X196504)
JCudaTensor x1256;
x1256 = x1235;
x1256.free();

// V_5b_a_cv_W <~~ X1632 * d_Convolv(1,0)(X367)/d_5b_a_cv_W
float x1258, x1259;
x1258 = lrn_rate_1;
x1259 = momentum;
JCudaTensor x1260, x1261;
x1260 = x1244;
x1261 = x818;
x826.backward_filter(x1260, x1261, x1257, x1258, x1259);

// Dealloc(X1632)
JCudaTensor x1262;
x1262 = x1244;
x1262.free();

// V_5b_a_bn_bias <~~ X195740
float x1264, x1265;
x1264 = lrn_rate_1;
x1265 = momentum;
JCudaTensor x1266;
x1266 = x1240;
x1263.update(x1266, x1264, x1265);

// Dealloc(X195740)
JCudaTensor x1267;
x1267 = x1240;
x1267.free();

// 5b_a_bn_scale <~~ V_5b_a_bn_scale
float x1268, x1269;
x1268 = 1;
x1269 = decay_1;
JCudaTensor x1270;
x1270 = x1252;
x831.update(x1270, x1268, x1269);

// 5b_a_cv_W <~~ V_5b_a_cv_W
float x1271, x1272;
x1271 = 1;
x1272 = decay_1;
JCudaTensor x1273;
x1273 = x1257;
x824.update(x1273, x1271, x1272);

// 5b_a_bn_bias <~~ V_5b_a_bn_bias
float x1274, x1275;
x1274 = 1;
x1275 = decay_1;
JCudaTensor x1276;
x1276 = x1263;
x832.update(x1276, x1274, x1275);

// val X1634 = (X1633 + X1609)
JCudaTensor x1277;
JCudaTensor x1278, x1279;
x1278 = x1249;
x1279 = x1135;
x1277 = x1278.plus_i(x1279);

// Dealloc(X1609)
JCudaTensor x1280;
x1280 = x1135;
x1280.free();

// val X1649 = X1634 * d_ReLU()(X367)/d_X366
JCudaTensor x1281;
JCudaTensor x1282, x1283;
x1282 = x1277;
x1283 = x818;
x1281 = x812.backward(x1282, x1283);

// Dealloc(X367)
JCudaTensor x1284;
x1284 = x818;
x1284.free();

// val X1665 = X1649.copy * d_ReLU()(X365)/d_X364
JCudaTensor x1285;
JCudaTensor x1286, x1287;
x1286 = x1281;
x1286 = x1286.clone();
x1287 = x813;
x1285 = x812.backward(x1286, x1287);

// Dealloc(X365)
JCudaTensor x1288;
x1288 = x813;
x1288.free();

// val X1653 = X1649.copy * d_ReLU()(X356)/d_X355
JCudaTensor x1289;
JCudaTensor x1290, x1291;
x1290 = x1281;
x1290 = x1290.clone();
x1291 = x810;
x1289 = x812.backward(x1290, x1291);

// Dealloc(X1649)
JCudaTensor x1292;
x1292 = x1281;
x1292.free();

// Dealloc(X356)
JCudaTensor x1293;
x1293 = x810;
x1293.free();

// val X194166 = X1665 * d_BatchNorm(5a2_c_bn)(X363,5a2_c_bn_scale)/d_5a2_c_bn_scale
JCudaTensor x1294;
JCudaTensor x1295, x1296, x1297;
x1295 = x1285;
x1296 = x796;
x1297 = x807;
JCudaTensor[] x1298 = x809.backward(x1295,x1296,x1297);
x1294 = x1298[1];

// val X185981 = X1653 * d_BatchNorm(5a1_bn)(X354,5a1_bn_scale)/d_5a1_bn_bias
JCudaTensor x1299;
JCudaTensor x1300, x1301, x1302;
x1300 = x1289;
x1301 = x749;
x1302 = x767;
JCudaTensor[] x1303 = x769.backward(x1300,x1301,x1302);
x1299 = x1303[2];

// val X1654 = X1653 * d_BatchNorm(5a1_bn)(X354,5a1_bn_scale)/d_X354
JCudaTensor x1304;
x1304 = x1303[0];

// val X193358 = X1665 * d_BatchNorm(5a2_c_bn)(X363,5a2_c_bn_scale)/d_5a2_c_bn_bias
JCudaTensor x1308;
x1308 = x1298[2];

// val X186780 = X1653 * d_BatchNorm(5a1_bn)(X354,5a1_bn_scale)/d_5a1_bn_scale
JCudaTensor x1312;
x1312 = x1303[1];

// Dealloc(X354)
JCudaTensor x1316;
x1316 = x749;
x1316.free();

// val X1666 = X1665 * d_BatchNorm(5a2_c_bn)(X363,5a2_c_bn_scale)/d_X363
JCudaTensor x1317;
x1317 = x1298[0];

// Dealloc(X363)
JCudaTensor x1321;
x1321 = x796;
x1321.free();

// V_5a1_bn_bias <~~ X185981
float x1323, x1324;
x1323 = lrn_rate_1;
x1324 = momentum;
JCudaTensor x1325;
x1325 = x1299;
x1322.update(x1325, x1323, x1324);

// Dealloc(X185981)
JCudaTensor x1326;
x1326 = x1299;
x1326.free();

// V_5a2_c_cv_W <~~ X1666 * d_Convolv(1,0)(X362)/d_5a2_c_cv_W
float x1328, x1329;
x1328 = lrn_rate_1;
x1329 = momentum;
JCudaTensor x1330, x1331;
x1330 = x1317;
x1331 = x794;
x802.backward_filter(x1330, x1331, x1327, x1328, x1329);

// val X1655 = X1654 * d_Convolv(2,0)(5a1_cv_W)/d_X353
JCudaTensor x1332;
JCudaTensor x1333, x1334;
x1333 = x1304;
x1334 = x753;
x1332 = x755.backward_data(x1333, x1334);

// V_5a1_bn_scale <~~ X186780
float x1336, x1337;
x1336 = lrn_rate_1;
x1337 = momentum;
JCudaTensor x1338;
x1338 = x1312;
x1335.update(x1338, x1336, x1337);

// Dealloc(X186780)
JCudaTensor x1339;
x1339 = x1312;
x1339.free();

// val X1667 = X1666 * d_Convolv(1,0)(5a2_c_cv_W)/d_X362
JCudaTensor x1340;
JCudaTensor x1341, x1342;
x1341 = x1317;
x1342 = x800;
x1340 = x802.backward_data(x1341, x1342);

// Dealloc(X1666)
JCudaTensor x1343;
x1343 = x1317;
x1343.free();

// V_5a1_cv_W <~~ X1654 * d_Convolv(2,0)(X353)/d_5a1_cv_W
float x1345, x1346;
x1345 = lrn_rate_1;
x1346 = momentum;
JCudaTensor x1347, x1348;
x1347 = x1304;
x1348 = x747;
x755.backward_filter(x1347, x1348, x1344, x1345, x1346);

// Dealloc(X1654)
JCudaTensor x1349;
x1349 = x1304;
x1349.free();

// V_5a2_c_bn_scale <~~ X194166
float x1351, x1352;
x1351 = lrn_rate_1;
x1352 = momentum;
JCudaTensor x1353;
x1353 = x1294;
x1350.update(x1353, x1351, x1352);

// Dealloc(X194166)
JCudaTensor x1354;
x1354 = x1294;
x1354.free();

// V_5a2_c_bn_bias <~~ X193358
float x1356, x1357;
x1356 = lrn_rate_1;
x1357 = momentum;
JCudaTensor x1358;
x1358 = x1308;
x1355.update(x1358, x1356, x1357);

// Dealloc(X193358)
JCudaTensor x1359;
x1359 = x1308;
x1359.free();

// 5a1_bn_scale <~~ V_5a1_bn_scale
float x1360, x1361;
x1360 = 1;
x1361 = decay_1;
JCudaTensor x1362;
x1362 = x1335;
x767.update(x1362, x1360, x1361);

// 5a2_c_bn_scale <~~ V_5a2_c_bn_scale
float x1363, x1364;
x1363 = 1;
x1364 = decay_1;
JCudaTensor x1365;
x1365 = x1350;
x807.update(x1365, x1363, x1364);

// 5a2_c_bn_bias <~~ V_5a2_c_bn_bias
float x1366, x1367;
x1366 = 1;
x1367 = decay_1;
JCudaTensor x1368;
x1368 = x1355;
x808.update(x1368, x1366, x1367);

// 5a2_c_cv_W <~~ V_5a2_c_cv_W
float x1369, x1370;
x1369 = 1;
x1370 = decay_1;
JCudaTensor x1371;
x1371 = x1327;
x800.update(x1371, x1369, x1370);

// 5a1_bn_bias <~~ V_5a1_bn_bias
float x1372, x1373;
x1372 = 1;
x1373 = decay_1;
JCudaTensor x1374;
x1374 = x1322;
x768.update(x1374, x1372, x1373);

// 5a1_cv_W <~~ V_5a1_cv_W
float x1375, x1376;
x1375 = 1;
x1376 = decay_1;
JCudaTensor x1377;
x1377 = x1344;
x753.update(x1377, x1375, x1376);

// val X1671 = X1667 * d_ReLU()(X362)/d_X361
JCudaTensor x1378;
JCudaTensor x1379, x1380;
x1379 = x1340;
x1380 = x794;
x1378 = x779.backward(x1379, x1380);

// Dealloc(X362)
JCudaTensor x1381;
x1381 = x794;
x1381.free();

// val X190906 = X1671 * d_BatchNorm(5a2_b_bn)(X360,5a2_b_bn_scale)/d_5a2_b_bn_bias
JCudaTensor x1382;
JCudaTensor x1383, x1384, x1385;
x1383 = x1378;
x1384 = x780;
x1385 = x791;
JCudaTensor[] x1386 = x793.backward(x1383,x1384,x1385);
x1382 = x1386[2];

// val X1672 = X1671 * d_BatchNorm(5a2_b_bn)(X360,5a2_b_bn_scale)/d_X360
JCudaTensor x1387;
x1387 = x1386[0];

// val X191727 = X1671 * d_BatchNorm(5a2_b_bn)(X360,5a2_b_bn_scale)/d_5a2_b_bn_scale
JCudaTensor x1391;
x1391 = x1386[1];

// Dealloc(X360)
JCudaTensor x1395;
x1395 = x780;
x1395.free();

// val X1673 = X1672 * d_Convolv(1,1)(5a2_b_cv_W)/d_X359
JCudaTensor x1396;
JCudaTensor x1397, x1398;
x1397 = x1387;
x1398 = x784;
x1396 = x786.backward_data(x1397, x1398);

// V_5a2_b_cv_W <~~ X1672 * d_Convolv(1,1)(X359)/d_5a2_b_cv_W
float x1400, x1401;
x1400 = lrn_rate_1;
x1401 = momentum;
JCudaTensor x1402, x1403;
x1402 = x1387;
x1403 = x777;
x786.backward_filter(x1402, x1403, x1399, x1400, x1401);

// Dealloc(X1672)
JCudaTensor x1404;
x1404 = x1387;
x1404.free();

// V_5a2_b_bn_scale <~~ X191727
float x1406, x1407;
x1406 = lrn_rate_1;
x1407 = momentum;
JCudaTensor x1408;
x1408 = x1391;
x1405.update(x1408, x1406, x1407);

// Dealloc(X191727)
JCudaTensor x1409;
x1409 = x1391;
x1409.free();

// V_5a2_b_bn_bias <~~ X190906
float x1411, x1412;
x1411 = lrn_rate_1;
x1412 = momentum;
JCudaTensor x1413;
x1413 = x1382;
x1410.update(x1413, x1411, x1412);

// Dealloc(X190906)
JCudaTensor x1414;
x1414 = x1382;
x1414.free();

// 5a2_b_cv_W <~~ V_5a2_b_cv_W
float x1415, x1416;
x1415 = 1;
x1416 = decay_1;
JCudaTensor x1417;
x1417 = x1399;
x784.update(x1417, x1415, x1416);

// 5a2_b_bn_scale <~~ V_5a2_b_bn_scale
float x1418, x1419;
x1418 = 1;
x1419 = decay_1;
JCudaTensor x1420;
x1420 = x1405;
x791.update(x1420, x1418, x1419);

// 5a2_b_bn_bias <~~ V_5a2_b_bn_bias
float x1421, x1422;
x1421 = 1;
x1422 = decay_1;
JCudaTensor x1423;
x1423 = x1410;
x792.update(x1423, x1421, x1422);

// val X1677 = X1673 * d_ReLU()(X359)/d_X358
JCudaTensor x1424;
JCudaTensor x1425, x1426;
x1425 = x1396;
x1426 = x777;
x1424 = x779.backward(x1425, x1426);

// Dealloc(X359)
JCudaTensor x1427;
x1427 = x777;
x1427.free();

// val X188415 = X1677 * d_BatchNorm(5a2_a_bn)(X357,5a2_a_bn_scale)/d_5a2_a_bn_bias
JCudaTensor x1428;
JCudaTensor x1429, x1430, x1431;
x1429 = x1424;
x1430 = x756;
x1431 = x774;
JCudaTensor[] x1432 = x776.backward(x1429,x1430,x1431);
x1428 = x1432[2];

// val X1678 = X1677 * d_BatchNorm(5a2_a_bn)(X357,5a2_a_bn_scale)/d_X357
JCudaTensor x1433;
x1433 = x1432[0];

// val X189249 = X1677 * d_BatchNorm(5a2_a_bn)(X357,5a2_a_bn_scale)/d_5a2_a_bn_scale
JCudaTensor x1437;
x1437 = x1432[1];

// Dealloc(X357)
JCudaTensor x1441;
x1441 = x756;
x1441.free();

// V_5a2_a_bn_bias <~~ X188415
float x1443, x1444;
x1443 = lrn_rate_1;
x1444 = momentum;
JCudaTensor x1445;
x1445 = x1428;
x1442.update(x1445, x1443, x1444);

// Dealloc(X188415)
JCudaTensor x1446;
x1446 = x1428;
x1446.free();

// V_5a2_a_bn_scale <~~ X189249
float x1448, x1449;
x1448 = lrn_rate_1;
x1449 = momentum;
JCudaTensor x1450;
x1450 = x1437;
x1447.update(x1450, x1448, x1449);

// Dealloc(X189249)
JCudaTensor x1451;
x1451 = x1437;
x1451.free();

// val X1680 = (X1655 + X1678 * d_Convolv(2,0)(5a2_a_cv_W)/d_X353)
JCudaTensor x1452;
JCudaTensor x1453;
x1453 = x1332;
JCudaTensor x1454, x1455;
x1454 = x1433;
x1455 = x760;
x1452 = x762.backward_data(x1454,x1455, x1453);

// V_5a2_a_cv_W <~~ X1678 * d_Convolv(2,0)(X353)/d_5a2_a_cv_W
float x1457, x1458;
x1457 = lrn_rate_1;
x1458 = momentum;
JCudaTensor x1459, x1460;
x1459 = x1433;
x1460 = x747;
x762.backward_filter(x1459, x1460, x1456, x1457, x1458);

// Dealloc(X1678)
JCudaTensor x1461;
x1461 = x1433;
x1461.free();

// 5a2_a_bn_bias <~~ V_5a2_a_bn_bias
float x1462, x1463;
x1462 = 1;
x1463 = decay_1;
JCudaTensor x1464;
x1464 = x1442;
x775.update(x1464, x1462, x1463);

// 5a2_a_bn_scale <~~ V_5a2_a_bn_scale
float x1465, x1466;
x1465 = 1;
x1466 = decay_1;
JCudaTensor x1467;
x1467 = x1447;
x774.update(x1467, x1465, x1466);

// 5a2_a_cv_W <~~ V_5a2_a_cv_W
float x1468, x1469;
x1468 = 1;
x1469 = decay_1;
JCudaTensor x1470;
x1470 = x1456;
x760.update(x1470, x1468, x1469);

// val X1750 = X1680 * d_ReLU()(X353)/d_X352
JCudaTensor x1471;
JCudaTensor x1472, x1473;
x1472 = x1452;
x1473 = x747;
x1471 = x490.backward(x1472, x1473);

// Dealloc(X353)
JCudaTensor x1474;
x1474 = x747;
x1474.free();

// val X1760 = X1750.copy * d_ReLU()(X351)/d_X350
JCudaTensor x1475;
JCudaTensor x1476, x1477;
x1476 = x1471;
x1476 = x1476.clone();
x1477 = x742;
x1475 = x490.backward(x1476, x1477);

// Dealloc(X351)
JCudaTensor x1478;
x1478 = x742;
x1478.free();

// val X184228 = X1760 * d_BatchNorm(4f_c_bn)(X349,4f_c_bn_scale)/d_4f_c_bn_scale
JCudaTensor x1479;
JCudaTensor x1480, x1481, x1482;
x1480 = x1475;
x1481 = x729;
x1482 = x739;
JCudaTensor[] x1483 = x741.backward(x1480,x1481,x1482);
x1479 = x1483[1];

// val X183276 = X1760 * d_BatchNorm(4f_c_bn)(X349,4f_c_bn_scale)/d_4f_c_bn_bias
JCudaTensor x1484;
x1484 = x1483[2];

// val X1761 = X1760 * d_BatchNorm(4f_c_bn)(X349,4f_c_bn_scale)/d_X349
JCudaTensor x1488;
x1488 = x1483[0];

// Dealloc(X349)
JCudaTensor x1492;
x1492 = x729;
x1492.free();

// V_4f_c_bn_bias <~~ X183276
float x1494, x1495;
x1494 = lrn_rate_1;
x1495 = momentum;
JCudaTensor x1496;
x1496 = x1484;
x1493.update(x1496, x1494, x1495);

// Dealloc(X183276)
JCudaTensor x1497;
x1497 = x1484;
x1497.free();

// val X1762 = X1761 * d_Convolv(1,0)(4f_c_cv_W)/d_X348
JCudaTensor x1498;
JCudaTensor x1499, x1500;
x1499 = x1488;
x1500 = x733;
x1498 = x480.backward_data(x1499, x1500);

// V_4f_c_cv_W <~~ X1761 * d_Convolv(1,0)(X348)/d_4f_c_cv_W
float x1502, x1503;
x1502 = lrn_rate_1;
x1503 = momentum;
JCudaTensor x1504, x1505;
x1504 = x1488;
x1505 = x727;
x480.backward_filter(x1504, x1505, x1501, x1502, x1503);

// Dealloc(X1761)
JCudaTensor x1506;
x1506 = x1488;
x1506.free();

// V_4f_c_bn_scale <~~ X184228
float x1508, x1509;
x1508 = lrn_rate_1;
x1509 = momentum;
JCudaTensor x1510;
x1510 = x1479;
x1507.update(x1510, x1508, x1509);

// Dealloc(X184228)
JCudaTensor x1511;
x1511 = x1479;
x1511.free();

// 4f_c_bn_bias <~~ V_4f_c_bn_bias
float x1512, x1513;
x1512 = 1;
x1513 = decay_1;
JCudaTensor x1514;
x1514 = x1493;
x740.update(x1514, x1512, x1513);

// 4f_c_cv_W <~~ V_4f_c_cv_W
float x1515, x1516;
x1515 = 1;
x1516 = decay_1;
JCudaTensor x1517;
x1517 = x1501;
x733.update(x1517, x1515, x1516);

// 4f_c_bn_scale <~~ V_4f_c_bn_scale
float x1518, x1519;
x1518 = 1;
x1519 = decay_1;
JCudaTensor x1520;
x1520 = x1507;
x739.update(x1520, x1518, x1519);

// val X1766 = X1762 * d_ReLU()(X348)/d_X347
JCudaTensor x1521;
JCudaTensor x1522, x1523;
x1522 = x1498;
x1523 = x727;
x1521 = x457.backward(x1522, x1523);

// Dealloc(X348)
JCudaTensor x1524;
x1524 = x727;
x1524.free();

// val X1767 = X1766 * d_BatchNorm(4f_b_bn)(X346,4f_b_bn_scale)/d_X346
JCudaTensor x1525;
JCudaTensor x1526, x1527, x1528;
x1526 = x1521;
x1527 = x714;
x1528 = x724;
JCudaTensor[] x1529 = x726.backward(x1526,x1527,x1528);
x1525 = x1529[0];

// val X181357 = X1766 * d_BatchNorm(4f_b_bn)(X346,4f_b_bn_scale)/d_4f_b_bn_scale
JCudaTensor x1530;
x1530 = x1529[1];

// val X180392 = X1766 * d_BatchNorm(4f_b_bn)(X346,4f_b_bn_scale)/d_4f_b_bn_bias
JCudaTensor x1534;
x1534 = x1529[2];

// Dealloc(X346)
JCudaTensor x1538;
x1538 = x714;
x1538.free();

// V_4f_b_bn_bias <~~ X180392
float x1540, x1541;
x1540 = lrn_rate_1;
x1541 = momentum;
JCudaTensor x1542;
x1542 = x1534;
x1539.update(x1542, x1540, x1541);

// Dealloc(X180392)
JCudaTensor x1543;
x1543 = x1534;
x1543.free();

// V_4f_b_bn_scale <~~ X181357
float x1545, x1546;
x1545 = lrn_rate_1;
x1546 = momentum;
JCudaTensor x1547;
x1547 = x1530;
x1544.update(x1547, x1545, x1546);

// Dealloc(X181357)
JCudaTensor x1548;
x1548 = x1530;
x1548.free();

// val X1768 = X1767 * d_Convolv(1,1)(4f_b_cv_W)/d_X345
JCudaTensor x1549;
JCudaTensor x1550, x1551;
x1550 = x1525;
x1551 = x718;
x1549 = x464.backward_data(x1550, x1551);

// V_4f_b_cv_W <~~ X1767 * d_Convolv(1,1)(X345)/d_4f_b_cv_W
float x1553, x1554;
x1553 = lrn_rate_1;
x1554 = momentum;
JCudaTensor x1555, x1556;
x1555 = x1525;
x1556 = x712;
x464.backward_filter(x1555, x1556, x1552, x1553, x1554);

// Dealloc(X1767)
JCudaTensor x1557;
x1557 = x1525;
x1557.free();

// 4f_b_bn_bias <~~ V_4f_b_bn_bias
float x1558, x1559;
x1558 = 1;
x1559 = decay_1;
JCudaTensor x1560;
x1560 = x1539;
x725.update(x1560, x1558, x1559);

// 4f_b_bn_scale <~~ V_4f_b_bn_scale
float x1561, x1562;
x1561 = 1;
x1562 = decay_1;
JCudaTensor x1563;
x1563 = x1544;
x724.update(x1563, x1561, x1562);

// 4f_b_cv_W <~~ V_4f_b_cv_W
float x1564, x1565;
x1564 = 1;
x1565 = decay_1;
JCudaTensor x1566;
x1566 = x1552;
x718.update(x1566, x1564, x1565);

// val X1772 = X1768 * d_ReLU()(X345)/d_X344
JCudaTensor x1567;
JCudaTensor x1568, x1569;
x1568 = x1549;
x1569 = x712;
x1567 = x457.backward(x1568, x1569);

// Dealloc(X345)
JCudaTensor x1570;
x1570 = x712;
x1570.free();

// val X177469 = X1772 * d_BatchNorm(4f_a_bn)(X343,4f_a_bn_scale)/d_4f_a_bn_bias
JCudaTensor x1571;
JCudaTensor x1572, x1573, x1574;
x1572 = x1567;
x1573 = x699;
x1574 = x709;
JCudaTensor[] x1575 = x711.backward(x1572,x1573,x1574);
x1571 = x1575[2];

// val X1773 = X1772 * d_BatchNorm(4f_a_bn)(X343,4f_a_bn_scale)/d_X343
JCudaTensor x1576;
x1576 = x1575[0];

// val X178447 = X1772 * d_BatchNorm(4f_a_bn)(X343,4f_a_bn_scale)/d_4f_a_bn_scale
JCudaTensor x1580;
x1580 = x1575[1];

// Dealloc(X343)
JCudaTensor x1584;
x1584 = x699;
x1584.free();

// val X1774 = X1773 * d_Convolv(1,0)(4f_a_cv_W)/d_X342
JCudaTensor x1585;
JCudaTensor x1586, x1587;
x1586 = x1576;
x1587 = x703;
x1585 = x504.backward_data(x1586, x1587);

// V_4f_a_cv_W <~~ X1773 * d_Convolv(1,0)(X342)/d_4f_a_cv_W
float x1589, x1590;
x1589 = lrn_rate_1;
x1590 = momentum;
JCudaTensor x1591, x1592;
x1591 = x1576;
x1592 = x697;
x504.backward_filter(x1591, x1592, x1588, x1589, x1590);

// Dealloc(X1773)
JCudaTensor x1593;
x1593 = x1576;
x1593.free();

// V_4f_a_bn_bias <~~ X177469
float x1595, x1596;
x1595 = lrn_rate_1;
x1596 = momentum;
JCudaTensor x1597;
x1597 = x1571;
x1594.update(x1597, x1595, x1596);

// Dealloc(X177469)
JCudaTensor x1598;
x1598 = x1571;
x1598.free();

// V_4f_a_bn_scale <~~ X178447
float x1600, x1601;
x1600 = lrn_rate_1;
x1601 = momentum;
JCudaTensor x1602;
x1602 = x1580;
x1599.update(x1602, x1600, x1601);

// Dealloc(X178447)
JCudaTensor x1603;
x1603 = x1580;
x1603.free();

// 4f_a_cv_W <~~ V_4f_a_cv_W
float x1604, x1605;
x1604 = 1;
x1605 = decay_1;
JCudaTensor x1606;
x1606 = x1588;
x703.update(x1606, x1604, x1605);

// 4f_a_bn_bias <~~ V_4f_a_bn_bias
float x1607, x1608;
x1607 = 1;
x1608 = decay_1;
JCudaTensor x1609;
x1609 = x1594;
x710.update(x1609, x1607, x1608);

// 4f_a_bn_scale <~~ V_4f_a_bn_scale
float x1610, x1611;
x1610 = 1;
x1611 = decay_1;
JCudaTensor x1612;
x1612 = x1599;
x709.update(x1612, x1610, x1611);

// val X1775 = (X1774 + X1750)
JCudaTensor x1613;
JCudaTensor x1614, x1615;
x1614 = x1585;
x1615 = x1471;
x1613 = x1614.plus_i(x1615);

// Dealloc(X1750)
JCudaTensor x1616;
x1616 = x1471;
x1616.free();

// val X1787 = X1775 * d_ReLU()(X342)/d_X341
JCudaTensor x1617;
JCudaTensor x1618, x1619;
x1618 = x1613;
x1619 = x697;
x1617 = x490.backward(x1618, x1619);

// Dealloc(X342)
JCudaTensor x1620;
x1620 = x697;
x1620.free();

// val X1797 = X1787.copy * d_ReLU()(X340)/d_X339
JCudaTensor x1621;
JCudaTensor x1622, x1623;
x1622 = x1617;
x1622 = x1622.clone();
x1623 = x692;
x1621 = x490.backward(x1622, x1623);

// Dealloc(X340)
JCudaTensor x1624;
x1624 = x692;
x1624.free();

// val X174451 = X1797 * d_BatchNorm(4e_c_bn)(X338,4e_c_bn_scale)/d_4e_c_bn_bias
JCudaTensor x1625;
JCudaTensor x1626, x1627, x1628;
x1626 = x1621;
x1627 = x679;
x1628 = x689;
JCudaTensor[] x1629 = x691.backward(x1626,x1627,x1628);
x1625 = x1629[2];

// val X1798 = X1797 * d_BatchNorm(4e_c_bn)(X338,4e_c_bn_scale)/d_X338
JCudaTensor x1630;
x1630 = x1629[0];

// val X175470 = X1797 * d_BatchNorm(4e_c_bn)(X338,4e_c_bn_scale)/d_4e_c_bn_scale
JCudaTensor x1634;
x1634 = x1629[1];

// Dealloc(X338)
JCudaTensor x1638;
x1638 = x679;
x1638.free();

// val X1799 = X1798 * d_Convolv(1,0)(4e_c_cv_W)/d_X337
JCudaTensor x1639;
JCudaTensor x1640, x1641;
x1640 = x1630;
x1641 = x683;
x1639 = x480.backward_data(x1640, x1641);

// V_4e_c_cv_W <~~ X1798 * d_Convolv(1,0)(X337)/d_4e_c_cv_W
float x1643, x1644;
x1643 = lrn_rate_1;
x1644 = momentum;
JCudaTensor x1645, x1646;
x1645 = x1630;
x1646 = x677;
x480.backward_filter(x1645, x1646, x1642, x1643, x1644);

// Dealloc(X1798)
JCudaTensor x1647;
x1647 = x1630;
x1647.free();

// V_4e_c_bn_scale <~~ X175470
float x1649, x1650;
x1649 = lrn_rate_1;
x1650 = momentum;
JCudaTensor x1651;
x1651 = x1634;
x1648.update(x1651, x1649, x1650);

// Dealloc(X175470)
JCudaTensor x1652;
x1652 = x1634;
x1652.free();

// V_4e_c_bn_bias <~~ X174451
float x1654, x1655;
x1654 = lrn_rate_1;
x1655 = momentum;
JCudaTensor x1656;
x1656 = x1625;
x1653.update(x1656, x1654, x1655);

// Dealloc(X174451)
JCudaTensor x1657;
x1657 = x1625;
x1657.free();

// 4e_c_cv_W <~~ V_4e_c_cv_W
float x1658, x1659;
x1658 = 1;
x1659 = decay_1;
JCudaTensor x1660;
x1660 = x1642;
x683.update(x1660, x1658, x1659);

// 4e_c_bn_scale <~~ V_4e_c_bn_scale
float x1661, x1662;
x1661 = 1;
x1662 = decay_1;
JCudaTensor x1663;
x1663 = x1648;
x689.update(x1663, x1661, x1662);

// 4e_c_bn_bias <~~ V_4e_c_bn_bias
float x1664, x1665;
x1664 = 1;
x1665 = decay_1;
JCudaTensor x1666;
x1666 = x1653;
x690.update(x1666, x1664, x1665);

// val X1803 = X1799 * d_ReLU()(X337)/d_X336
JCudaTensor x1667;
JCudaTensor x1668, x1669;
x1668 = x1639;
x1669 = x677;
x1667 = x457.backward(x1668, x1669);

// Dealloc(X337)
JCudaTensor x1670;
x1670 = x677;
x1670.free();

// val X1804 = X1803 * d_BatchNorm(4e_b_bn)(X335,4e_b_bn_scale)/d_X335
JCudaTensor x1671;
JCudaTensor x1672, x1673, x1674;
x1672 = x1667;
x1673 = x664;
x1674 = x674;
JCudaTensor[] x1675 = x676.backward(x1672,x1673,x1674);
x1671 = x1675[0];

// val X172398 = X1803 * d_BatchNorm(4e_b_bn)(X335,4e_b_bn_scale)/d_4e_b_bn_scale
JCudaTensor x1676;
x1676 = x1675[1];

// val X171366 = X1803 * d_BatchNorm(4e_b_bn)(X335,4e_b_bn_scale)/d_4e_b_bn_bias
JCudaTensor x1680;
x1680 = x1675[2];

// Dealloc(X335)
JCudaTensor x1684;
x1684 = x664;
x1684.free();

// V_4e_b_bn_bias <~~ X171366
float x1686, x1687;
x1686 = lrn_rate_1;
x1687 = momentum;
JCudaTensor x1688;
x1688 = x1680;
x1685.update(x1688, x1686, x1687);

// Dealloc(X171366)
JCudaTensor x1689;
x1689 = x1680;
x1689.free();

// V_4e_b_bn_scale <~~ X172398
float x1691, x1692;
x1691 = lrn_rate_1;
x1692 = momentum;
JCudaTensor x1693;
x1693 = x1676;
x1690.update(x1693, x1691, x1692);

// Dealloc(X172398)
JCudaTensor x1694;
x1694 = x1676;
x1694.free();

// val X1805 = X1804 * d_Convolv(1,1)(4e_b_cv_W)/d_X334
JCudaTensor x1695;
JCudaTensor x1696, x1697;
x1696 = x1671;
x1697 = x668;
x1695 = x464.backward_data(x1696, x1697);

// V_4e_b_cv_W <~~ X1804 * d_Convolv(1,1)(X334)/d_4e_b_cv_W
float x1699, x1700;
x1699 = lrn_rate_1;
x1700 = momentum;
JCudaTensor x1701, x1702;
x1701 = x1671;
x1702 = x662;
x464.backward_filter(x1701, x1702, x1698, x1699, x1700);

// Dealloc(X1804)
JCudaTensor x1703;
x1703 = x1671;
x1703.free();

// 4e_b_bn_bias <~~ V_4e_b_bn_bias
float x1704, x1705;
x1704 = 1;
x1705 = decay_1;
JCudaTensor x1706;
x1706 = x1685;
x675.update(x1706, x1704, x1705);

// 4e_b_bn_scale <~~ V_4e_b_bn_scale
float x1707, x1708;
x1707 = 1;
x1708 = decay_1;
JCudaTensor x1709;
x1709 = x1690;
x674.update(x1709, x1707, x1708);

// 4e_b_cv_W <~~ V_4e_b_cv_W
float x1710, x1711;
x1710 = 1;
x1711 = decay_1;
JCudaTensor x1712;
x1712 = x1698;
x668.update(x1712, x1710, x1711);

// val X1809 = X1805 * d_ReLU()(X334)/d_X333
JCudaTensor x1713;
JCudaTensor x1714, x1715;
x1714 = x1695;
x1715 = x662;
x1713 = x457.backward(x1714, x1715);

// Dealloc(X334)
JCudaTensor x1716;
x1716 = x662;
x1716.free();

// val X169287 = X1809 * d_BatchNorm(4e_a_bn)(X332,4e_a_bn_scale)/d_4e_a_bn_scale
JCudaTensor x1717;
JCudaTensor x1718, x1719, x1720;
x1718 = x1713;
x1719 = x649;
x1720 = x659;
JCudaTensor[] x1721 = x661.backward(x1718,x1719,x1720);
x1717 = x1721[1];

// val X168242 = X1809 * d_BatchNorm(4e_a_bn)(X332,4e_a_bn_scale)/d_4e_a_bn_bias
JCudaTensor x1722;
x1722 = x1721[2];

// val X1810 = X1809 * d_BatchNorm(4e_a_bn)(X332,4e_a_bn_scale)/d_X332
JCudaTensor x1726;
x1726 = x1721[0];

// Dealloc(X332)
JCudaTensor x1730;
x1730 = x649;
x1730.free();

// val X1811 = X1810 * d_Convolv(1,0)(4e_a_cv_W)/d_X331
JCudaTensor x1731;
JCudaTensor x1732, x1733;
x1732 = x1726;
x1733 = x653;
x1731 = x504.backward_data(x1732, x1733);

// V_4e_a_cv_W <~~ X1810 * d_Convolv(1,0)(X331)/d_4e_a_cv_W
float x1735, x1736;
x1735 = lrn_rate_1;
x1736 = momentum;
JCudaTensor x1737, x1738;
x1737 = x1726;
x1738 = x647;
x504.backward_filter(x1737, x1738, x1734, x1735, x1736);

// Dealloc(X1810)
JCudaTensor x1739;
x1739 = x1726;
x1739.free();

// V_4e_a_bn_scale <~~ X169287
float x1741, x1742;
x1741 = lrn_rate_1;
x1742 = momentum;
JCudaTensor x1743;
x1743 = x1717;
x1740.update(x1743, x1741, x1742);

// Dealloc(X169287)
JCudaTensor x1744;
x1744 = x1717;
x1744.free();

// V_4e_a_bn_bias <~~ X168242
float x1746, x1747;
x1746 = lrn_rate_1;
x1747 = momentum;
JCudaTensor x1748;
x1748 = x1722;
x1745.update(x1748, x1746, x1747);

// Dealloc(X168242)
JCudaTensor x1749;
x1749 = x1722;
x1749.free();

// 4e_a_cv_W <~~ V_4e_a_cv_W
float x1750, x1751;
x1750 = 1;
x1751 = decay_1;
JCudaTensor x1752;
x1752 = x1734;
x653.update(x1752, x1750, x1751);

// 4e_a_bn_scale <~~ V_4e_a_bn_scale
float x1753, x1754;
x1753 = 1;
x1754 = decay_1;
JCudaTensor x1755;
x1755 = x1740;
x659.update(x1755, x1753, x1754);

// 4e_a_bn_bias <~~ V_4e_a_bn_bias
float x1756, x1757;
x1756 = 1;
x1757 = decay_1;
JCudaTensor x1758;
x1758 = x1745;
x660.update(x1758, x1756, x1757);

// val X1812 = (X1811 + X1787)
JCudaTensor x1759;
JCudaTensor x1760, x1761;
x1760 = x1731;
x1761 = x1617;
x1759 = x1760.plus_i(x1761);

// Dealloc(X1787)
JCudaTensor x1762;
x1762 = x1617;
x1762.free();

// val X1824 = X1812 * d_ReLU()(X331)/d_X330
JCudaTensor x1763;
JCudaTensor x1764, x1765;
x1764 = x1759;
x1765 = x647;
x1763 = x490.backward(x1764, x1765);

// Dealloc(X331)
JCudaTensor x1766;
x1766 = x647;
x1766.free();

// val X1834 = X1824.copy * d_ReLU()(X329)/d_X328
JCudaTensor x1767;
JCudaTensor x1768, x1769;
x1768 = x1763;
x1768 = x1768.clone();
x1769 = x642;
x1767 = x490.backward(x1768, x1769);

// Dealloc(X329)
JCudaTensor x1770;
x1770 = x642;
x1770.free();

// val X166109 = X1834 * d_BatchNorm(4d_c_bn)(X327,4d_c_bn_scale)/d_4d_c_bn_scale
JCudaTensor x1771;
JCudaTensor x1772, x1773, x1774;
x1772 = x1767;
x1773 = x629;
x1774 = x639;
JCudaTensor[] x1775 = x641.backward(x1772,x1773,x1774);
x1771 = x1775[1];

// val X165023 = X1834 * d_BatchNorm(4d_c_bn)(X327,4d_c_bn_scale)/d_4d_c_bn_bias
JCudaTensor x1776;
x1776 = x1775[2];

// val X1835 = X1834 * d_BatchNorm(4d_c_bn)(X327,4d_c_bn_scale)/d_X327
JCudaTensor x1780;
x1780 = x1775[0];

// Dealloc(X327)
JCudaTensor x1784;
x1784 = x629;
x1784.free();

// V_4d_c_bn_scale <~~ X166109
float x1786, x1787;
x1786 = lrn_rate_1;
x1787 = momentum;
JCudaTensor x1788;
x1788 = x1771;
x1785.update(x1788, x1786, x1787);

// Dealloc(X166109)
JCudaTensor x1789;
x1789 = x1771;
x1789.free();

// V_4d_c_bn_bias <~~ X165023
float x1791, x1792;
x1791 = lrn_rate_1;
x1792 = momentum;
JCudaTensor x1793;
x1793 = x1776;
x1790.update(x1793, x1791, x1792);

// Dealloc(X165023)
JCudaTensor x1794;
x1794 = x1776;
x1794.free();

// val X1836 = X1835 * d_Convolv(1,0)(4d_c_cv_W)/d_X326
JCudaTensor x1795;
JCudaTensor x1796, x1797;
x1796 = x1780;
x1797 = x633;
x1795 = x480.backward_data(x1796, x1797);

// V_4d_c_cv_W <~~ X1835 * d_Convolv(1,0)(X326)/d_4d_c_cv_W
float x1799, x1800;
x1799 = lrn_rate_1;
x1800 = momentum;
JCudaTensor x1801, x1802;
x1801 = x1780;
x1802 = x627;
x480.backward_filter(x1801, x1802, x1798, x1799, x1800);

// Dealloc(X1835)
JCudaTensor x1803;
x1803 = x1780;
x1803.free();

// 4d_c_bn_scale <~~ V_4d_c_bn_scale
float x1804, x1805;
x1804 = 1;
x1805 = decay_1;
JCudaTensor x1806;
x1806 = x1785;
x639.update(x1806, x1804, x1805);

// 4d_c_bn_bias <~~ V_4d_c_bn_bias
float x1807, x1808;
x1807 = 1;
x1808 = decay_1;
JCudaTensor x1809;
x1809 = x1790;
x640.update(x1809, x1807, x1808);

// 4d_c_cv_W <~~ V_4d_c_cv_W
float x1810, x1811;
x1810 = 1;
x1811 = decay_1;
JCudaTensor x1812;
x1812 = x1798;
x633.update(x1812, x1810, x1811);

// val X1840 = X1836 * d_ReLU()(X326)/d_X325
JCudaTensor x1813;
JCudaTensor x1814, x1815;
x1814 = x1795;
x1815 = x627;
x1813 = x457.backward(x1814, x1815);

// Dealloc(X326)
JCudaTensor x1816;
x1816 = x627;
x1816.free();

// val X1841 = X1840 * d_BatchNorm(4d_b_bn)(X324,4d_b_bn_scale)/d_X324
JCudaTensor x1817;
JCudaTensor x1818, x1819, x1820;
x1818 = x1813;
x1819 = x614;
x1820 = x624;
JCudaTensor[] x1821 = x626.backward(x1818,x1819,x1820);
x1817 = x1821[0];

// val X161737 = X1840 * d_BatchNorm(4d_b_bn)(X324,4d_b_bn_scale)/d_4d_b_bn_bias
JCudaTensor x1822;
x1822 = x1821[2];

// val X162836 = X1840 * d_BatchNorm(4d_b_bn)(X324,4d_b_bn_scale)/d_4d_b_bn_scale
JCudaTensor x1826;
x1826 = x1821[1];

// Dealloc(X324)
JCudaTensor x1830;
x1830 = x614;
x1830.free();

// val X1842 = X1841 * d_Convolv(1,1)(4d_b_cv_W)/d_X323
JCudaTensor x1831;
JCudaTensor x1832, x1833;
x1832 = x1817;
x1833 = x618;
x1831 = x464.backward_data(x1832, x1833);

// V_4d_b_cv_W <~~ X1841 * d_Convolv(1,1)(X323)/d_4d_b_cv_W
float x1835, x1836;
x1835 = lrn_rate_1;
x1836 = momentum;
JCudaTensor x1837, x1838;
x1837 = x1817;
x1838 = x612;
x464.backward_filter(x1837, x1838, x1834, x1835, x1836);

// Dealloc(X1841)
JCudaTensor x1839;
x1839 = x1817;
x1839.free();

// V_4d_b_bn_scale <~~ X162836
float x1841, x1842;
x1841 = lrn_rate_1;
x1842 = momentum;
JCudaTensor x1843;
x1843 = x1826;
x1840.update(x1843, x1841, x1842);

// Dealloc(X162836)
JCudaTensor x1844;
x1844 = x1826;
x1844.free();

// V_4d_b_bn_bias <~~ X161737
float x1846, x1847;
x1846 = lrn_rate_1;
x1847 = momentum;
JCudaTensor x1848;
x1848 = x1822;
x1845.update(x1848, x1846, x1847);

// Dealloc(X161737)
JCudaTensor x1849;
x1849 = x1822;
x1849.free();

// 4d_b_cv_W <~~ V_4d_b_cv_W
float x1850, x1851;
x1850 = 1;
x1851 = decay_1;
JCudaTensor x1852;
x1852 = x1834;
x618.update(x1852, x1850, x1851);

// 4d_b_bn_scale <~~ V_4d_b_bn_scale
float x1853, x1854;
x1853 = 1;
x1854 = decay_1;
JCudaTensor x1855;
x1855 = x1840;
x624.update(x1855, x1853, x1854);

// 4d_b_bn_bias <~~ V_4d_b_bn_bias
float x1856, x1857;
x1856 = 1;
x1857 = decay_1;
JCudaTensor x1858;
x1858 = x1845;
x625.update(x1858, x1856, x1857);

// val X1846 = X1842 * d_ReLU()(X323)/d_X322
JCudaTensor x1859;
JCudaTensor x1860, x1861;
x1860 = x1831;
x1861 = x612;
x1859 = x457.backward(x1860, x1861);

// Dealloc(X323)
JCudaTensor x1862;
x1862 = x612;
x1862.free();

// val X1847 = X1846 * d_BatchNorm(4d_a_bn)(X321,4d_a_bn_scale)/d_X321
JCudaTensor x1863;
JCudaTensor x1864, x1865, x1866;
x1864 = x1859;
x1865 = x599;
x1866 = x609;
JCudaTensor[] x1867 = x611.backward(x1864,x1865,x1866);
x1863 = x1867[0];

// val X159524 = X1846 * d_BatchNorm(4d_a_bn)(X321,4d_a_bn_scale)/d_4d_a_bn_scale
JCudaTensor x1868;
x1868 = x1867[1];

// val X158412 = X1846 * d_BatchNorm(4d_a_bn)(X321,4d_a_bn_scale)/d_4d_a_bn_bias
JCudaTensor x1872;
x1872 = x1867[2];

// Dealloc(X321)
JCudaTensor x1876;
x1876 = x599;
x1876.free();

// V_4d_a_bn_bias <~~ X158412
float x1878, x1879;
x1878 = lrn_rate_1;
x1879 = momentum;
JCudaTensor x1880;
x1880 = x1872;
x1877.update(x1880, x1878, x1879);

// Dealloc(X158412)
JCudaTensor x1881;
x1881 = x1872;
x1881.free();

// val X1848 = X1847 * d_Convolv(1,0)(4d_a_cv_W)/d_X320
JCudaTensor x1882;
JCudaTensor x1883, x1884;
x1883 = x1863;
x1884 = x603;
x1882 = x504.backward_data(x1883, x1884);

// V_4d_a_cv_W <~~ X1847 * d_Convolv(1,0)(X320)/d_4d_a_cv_W
float x1886, x1887;
x1886 = lrn_rate_1;
x1887 = momentum;
JCudaTensor x1888, x1889;
x1888 = x1863;
x1889 = x597;
x504.backward_filter(x1888, x1889, x1885, x1886, x1887);

// Dealloc(X1847)
JCudaTensor x1890;
x1890 = x1863;
x1890.free();

// V_4d_a_bn_scale <~~ X159524
float x1892, x1893;
x1892 = lrn_rate_1;
x1893 = momentum;
JCudaTensor x1894;
x1894 = x1868;
x1891.update(x1894, x1892, x1893);

// Dealloc(X159524)
JCudaTensor x1895;
x1895 = x1868;
x1895.free();

// 4d_a_bn_bias <~~ V_4d_a_bn_bias
float x1896, x1897;
x1896 = 1;
x1897 = decay_1;
JCudaTensor x1898;
x1898 = x1877;
x610.update(x1898, x1896, x1897);

// 4d_a_cv_W <~~ V_4d_a_cv_W
float x1899, x1900;
x1899 = 1;
x1900 = decay_1;
JCudaTensor x1901;
x1901 = x1885;
x603.update(x1901, x1899, x1900);

// 4d_a_bn_scale <~~ V_4d_a_bn_scale
float x1902, x1903;
x1902 = 1;
x1903 = decay_1;
JCudaTensor x1904;
x1904 = x1891;
x609.update(x1904, x1902, x1903);

// val X1849 = (X1848 + X1824)
JCudaTensor x1905;
JCudaTensor x1906, x1907;
x1906 = x1882;
x1907 = x1763;
x1905 = x1906.plus_i(x1907);

// Dealloc(X1824)
JCudaTensor x1908;
x1908 = x1763;
x1908.free();

// val X1861 = X1849 * d_ReLU()(X320)/d_X319
JCudaTensor x1909;
JCudaTensor x1910, x1911;
x1910 = x1905;
x1911 = x597;
x1909 = x490.backward(x1910, x1911);

// Dealloc(X320)
JCudaTensor x1912;
x1912 = x597;
x1912.free();

// val X1871 = X1861.copy * d_ReLU()(X318)/d_X317
JCudaTensor x1913;
JCudaTensor x1914, x1915;
x1914 = x1909;
x1914 = x1914.clone();
x1915 = x592;
x1913 = x490.backward(x1914, x1915);

// Dealloc(X318)
JCudaTensor x1916;
x1916 = x592;
x1916.free();

// val X156145 = X1871 * d_BatchNorm(4c_c_bn)(X316,4c_c_bn_scale)/d_4c_c_bn_scale
JCudaTensor x1917;
JCudaTensor x1918, x1919, x1920;
x1918 = x1913;
x1919 = x579;
x1920 = x589;
JCudaTensor[] x1921 = x591.backward(x1918,x1919,x1920);
x1917 = x1921[1];

// val X1872 = X1871 * d_BatchNorm(4c_c_bn)(X316,4c_c_bn_scale)/d_X316
JCudaTensor x1922;
x1922 = x1921[0];

// val X154992 = X1871 * d_BatchNorm(4c_c_bn)(X316,4c_c_bn_scale)/d_4c_c_bn_bias
JCudaTensor x1926;
x1926 = x1921[2];

// Dealloc(X316)
JCudaTensor x1930;
x1930 = x579;
x1930.free();

// V_4c_c_bn_bias <~~ X154992
float x1932, x1933;
x1932 = lrn_rate_1;
x1933 = momentum;
JCudaTensor x1934;
x1934 = x1926;
x1931.update(x1934, x1932, x1933);

// Dealloc(X154992)
JCudaTensor x1935;
x1935 = x1926;
x1935.free();

// val X1873 = X1872 * d_Convolv(1,0)(4c_c_cv_W)/d_X315
JCudaTensor x1936;
JCudaTensor x1937, x1938;
x1937 = x1922;
x1938 = x583;
x1936 = x480.backward_data(x1937, x1938);

// V_4c_c_cv_W <~~ X1872 * d_Convolv(1,0)(X315)/d_4c_c_cv_W
float x1940, x1941;
x1940 = lrn_rate_1;
x1941 = momentum;
JCudaTensor x1942, x1943;
x1942 = x1922;
x1943 = x577;
x480.backward_filter(x1942, x1943, x1939, x1940, x1941);

// Dealloc(X1872)
JCudaTensor x1944;
x1944 = x1922;
x1944.free();

// V_4c_c_bn_scale <~~ X156145
float x1946, x1947;
x1946 = lrn_rate_1;
x1947 = momentum;
JCudaTensor x1948;
x1948 = x1917;
x1945.update(x1948, x1946, x1947);

// Dealloc(X156145)
JCudaTensor x1949;
x1949 = x1917;
x1949.free();

// 4c_c_bn_bias <~~ V_4c_c_bn_bias
float x1950, x1951;
x1950 = 1;
x1951 = decay_1;
JCudaTensor x1952;
x1952 = x1931;
x590.update(x1952, x1950, x1951);

// 4c_c_cv_W <~~ V_4c_c_cv_W
float x1953, x1954;
x1953 = 1;
x1954 = decay_1;
JCudaTensor x1955;
x1955 = x1939;
x583.update(x1955, x1953, x1954);

// 4c_c_bn_scale <~~ V_4c_c_bn_scale
float x1956, x1957;
x1956 = 1;
x1957 = decay_1;
JCudaTensor x1958;
x1958 = x1945;
x589.update(x1958, x1956, x1957);

// val X1877 = X1873 * d_ReLU()(X315)/d_X314
JCudaTensor x1959;
JCudaTensor x1960, x1961;
x1960 = x1936;
x1961 = x577;
x1959 = x457.backward(x1960, x1961);

// Dealloc(X315)
JCudaTensor x1962;
x1962 = x577;
x1962.free();

// val X151505 = X1877 * d_BatchNorm(4c_b_bn)(X313,4c_b_bn_scale)/d_4c_b_bn_bias
JCudaTensor x1963;
JCudaTensor x1964, x1965, x1966;
x1964 = x1959;
x1965 = x564;
x1966 = x574;
JCudaTensor[] x1967 = x576.backward(x1964,x1965,x1966);
x1963 = x1967[2];

// val X1878 = X1877 * d_BatchNorm(4c_b_bn)(X313,4c_b_bn_scale)/d_X313
JCudaTensor x1968;
x1968 = x1967[0];

// val X152671 = X1877 * d_BatchNorm(4c_b_bn)(X313,4c_b_bn_scale)/d_4c_b_bn_scale
JCudaTensor x1972;
x1972 = x1967[1];

// Dealloc(X313)
JCudaTensor x1976;
x1976 = x564;
x1976.free();

// val X1879 = X1878 * d_Convolv(1,1)(4c_b_cv_W)/d_X312
JCudaTensor x1977;
JCudaTensor x1978, x1979;
x1978 = x1968;
x1979 = x568;
x1977 = x464.backward_data(x1978, x1979);

// V_4c_b_cv_W <~~ X1878 * d_Convolv(1,1)(X312)/d_4c_b_cv_W
float x1981, x1982;
x1981 = lrn_rate_1;
x1982 = momentum;
JCudaTensor x1983, x1984;
x1983 = x1968;
x1984 = x562;
x464.backward_filter(x1983, x1984, x1980, x1981, x1982);

// Dealloc(X1878)
JCudaTensor x1985;
x1985 = x1968;
x1985.free();

// V_4c_b_bn_bias <~~ X151505
float x1987, x1988;
x1987 = lrn_rate_1;
x1988 = momentum;
JCudaTensor x1989;
x1989 = x1963;
x1986.update(x1989, x1987, x1988);

// Dealloc(X151505)
JCudaTensor x1990;
x1990 = x1963;
x1990.free();

// V_4c_b_bn_scale <~~ X152671
float x1992, x1993;
x1992 = lrn_rate_1;
x1993 = momentum;
JCudaTensor x1994;
x1994 = x1972;
x1991.update(x1994, x1992, x1993);

// Dealloc(X152671)
JCudaTensor x1995;
x1995 = x1972;
x1995.free();

// 4c_b_cv_W <~~ V_4c_b_cv_W
float x1996, x1997;
x1996 = 1;
x1997 = decay_1;
JCudaTensor x1998;
x1998 = x1980;
x568.update(x1998, x1996, x1997);

// 4c_b_bn_bias <~~ V_4c_b_bn_bias
float x1999, x2000;
x1999 = 1;
x2000 = decay_1;
JCudaTensor x2001;
x2001 = x1986;
x575.update(x2001, x1999, x2000);

// 4c_b_bn_scale <~~ V_4c_b_bn_scale
float x2002, x2003;
x2002 = 1;
x2003 = decay_1;
JCudaTensor x2004;
x2004 = x1991;
x574.update(x2004, x2002, x2003);

// val X1883 = X1879 * d_ReLU()(X312)/d_X311
JCudaTensor x2005;
JCudaTensor x2006, x2007;
x2006 = x1977;
x2007 = x562;
x2005 = x457.backward(x2006, x2007);

// Dealloc(X312)
JCudaTensor x2008;
x2008 = x562;
x2008.free();

// val X147979 = X1883 * d_BatchNorm(4c_a_bn)(X310,4c_a_bn_scale)/d_4c_a_bn_bias
JCudaTensor x2009;
JCudaTensor x2010, x2011, x2012;
x2010 = x2005;
x2011 = x549;
x2012 = x559;
JCudaTensor[] x2013 = x561.backward(x2010,x2011,x2012);
x2009 = x2013[2];

// val X1884 = X1883 * d_BatchNorm(4c_a_bn)(X310,4c_a_bn_scale)/d_X310
JCudaTensor x2014;
x2014 = x2013[0];

// val X149158 = X1883 * d_BatchNorm(4c_a_bn)(X310,4c_a_bn_scale)/d_4c_a_bn_scale
JCudaTensor x2018;
x2018 = x2013[1];

// Dealloc(X310)
JCudaTensor x2022;
x2022 = x549;
x2022.free();

// val X1885 = X1884 * d_Convolv(1,0)(4c_a_cv_W)/d_X309
JCudaTensor x2023;
JCudaTensor x2024, x2025;
x2024 = x2014;
x2025 = x553;
x2023 = x504.backward_data(x2024, x2025);

// V_4c_a_bn_bias <~~ X147979
float x2027, x2028;
x2027 = lrn_rate_1;
x2028 = momentum;
JCudaTensor x2029;
x2029 = x2009;
x2026.update(x2029, x2027, x2028);

// Dealloc(X147979)
JCudaTensor x2030;
x2030 = x2009;
x2030.free();

// V_4c_a_bn_scale <~~ X149158
float x2032, x2033;
x2032 = lrn_rate_1;
x2033 = momentum;
JCudaTensor x2034;
x2034 = x2018;
x2031.update(x2034, x2032, x2033);

// Dealloc(X149158)
JCudaTensor x2035;
x2035 = x2018;
x2035.free();

// V_4c_a_cv_W <~~ X1884 * d_Convolv(1,0)(X309)/d_4c_a_cv_W
float x2037, x2038;
x2037 = lrn_rate_1;
x2038 = momentum;
JCudaTensor x2039, x2040;
x2039 = x2014;
x2040 = x547;
x504.backward_filter(x2039, x2040, x2036, x2037, x2038);

// Dealloc(X1884)
JCudaTensor x2041;
x2041 = x2014;
x2041.free();

// 4c_a_bn_bias <~~ V_4c_a_bn_bias
float x2042, x2043;
x2042 = 1;
x2043 = decay_1;
JCudaTensor x2044;
x2044 = x2026;
x560.update(x2044, x2042, x2043);

// 4c_a_bn_scale <~~ V_4c_a_bn_scale
float x2045, x2046;
x2045 = 1;
x2046 = decay_1;
JCudaTensor x2047;
x2047 = x2031;
x559.update(x2047, x2045, x2046);

// 4c_a_cv_W <~~ V_4c_a_cv_W
float x2048, x2049;
x2048 = 1;
x2049 = decay_1;
JCudaTensor x2050;
x2050 = x2036;
x553.update(x2050, x2048, x2049);

// val X1886 = (X1885 + X1861)
JCudaTensor x2051;
JCudaTensor x2052, x2053;
x2052 = x2023;
x2053 = x1909;
x2051 = x2052.plus_i(x2053);

// Dealloc(X1861)
JCudaTensor x2054;
x2054 = x1909;
x2054.free();

// val X1898 = X1886 * d_ReLU()(X309)/d_X308
JCudaTensor x2055;
JCudaTensor x2056, x2057;
x2056 = x2051;
x2057 = x547;
x2055 = x490.backward(x2056, x2057);

// Dealloc(X309)
JCudaTensor x2058;
x2058 = x547;
x2058.free();

// val X1908 = X1898.copy * d_ReLU()(X307)/d_X306
JCudaTensor x2059;
JCudaTensor x2060, x2061;
x2060 = x2055;
x2060 = x2060.clone();
x2061 = x542;
x2059 = x490.backward(x2060, x2061);

// Dealloc(X307)
JCudaTensor x2062;
x2062 = x542;
x2062.free();

// val X1909 = X1908 * d_BatchNorm(4b_c_bn)(X305,4b_c_bn_scale)/d_X305
JCudaTensor x2063;
JCudaTensor x2064, x2065, x2066;
x2064 = x2059;
x2065 = x529;
x2066 = x539;
JCudaTensor[] x2067 = x541.backward(x2064,x2065,x2066);
x2063 = x2067[0];

// val X144358 = X1908 * d_BatchNorm(4b_c_bn)(X305,4b_c_bn_scale)/d_4b_c_bn_bias
JCudaTensor x2068;
x2068 = x2067[2];

// val X145578 = X1908 * d_BatchNorm(4b_c_bn)(X305,4b_c_bn_scale)/d_4b_c_bn_scale
JCudaTensor x2072;
x2072 = x2067[1];

// Dealloc(X305)
JCudaTensor x2076;
x2076 = x529;
x2076.free();

// V_4b_c_bn_bias <~~ X144358
float x2078, x2079;
x2078 = lrn_rate_1;
x2079 = momentum;
JCudaTensor x2080;
x2080 = x2068;
x2077.update(x2080, x2078, x2079);

// Dealloc(X144358)
JCudaTensor x2081;
x2081 = x2068;
x2081.free();

// val X1910 = X1909 * d_Convolv(1,0)(4b_c_cv_W)/d_X304
JCudaTensor x2082;
JCudaTensor x2083, x2084;
x2083 = x2063;
x2084 = x533;
x2082 = x480.backward_data(x2083, x2084);

// V_4b_c_cv_W <~~ X1909 * d_Convolv(1,0)(X304)/d_4b_c_cv_W
float x2086, x2087;
x2086 = lrn_rate_1;
x2087 = momentum;
JCudaTensor x2088, x2089;
x2088 = x2063;
x2089 = x527;
x480.backward_filter(x2088, x2089, x2085, x2086, x2087);

// Dealloc(X1909)
JCudaTensor x2090;
x2090 = x2063;
x2090.free();

// V_4b_c_bn_scale <~~ X145578
float x2092, x2093;
x2092 = lrn_rate_1;
x2093 = momentum;
JCudaTensor x2094;
x2094 = x2072;
x2091.update(x2094, x2092, x2093);

// Dealloc(X145578)
JCudaTensor x2095;
x2095 = x2072;
x2095.free();

// 4b_c_bn_bias <~~ V_4b_c_bn_bias
float x2096, x2097;
x2096 = 1;
x2097 = decay_1;
JCudaTensor x2098;
x2098 = x2077;
x540.update(x2098, x2096, x2097);

// 4b_c_cv_W <~~ V_4b_c_cv_W
float x2099, x2100;
x2099 = 1;
x2100 = decay_1;
JCudaTensor x2101;
x2101 = x2085;
x533.update(x2101, x2099, x2100);

// 4b_c_bn_scale <~~ V_4b_c_bn_scale
float x2102, x2103;
x2102 = 1;
x2103 = decay_1;
JCudaTensor x2104;
x2104 = x2091;
x539.update(x2104, x2102, x2103);

// val X1914 = X1910 * d_ReLU()(X304)/d_X303
JCudaTensor x2105;
JCudaTensor x2106, x2107;
x2106 = x2082;
x2107 = x527;
x2105 = x457.backward(x2106, x2107);

// Dealloc(X304)
JCudaTensor x2108;
x2108 = x527;
x2108.free();

// val X141903 = X1914 * d_BatchNorm(4b_b_bn)(X302,4b_b_bn_scale)/d_4b_b_bn_scale
JCudaTensor x2109;
JCudaTensor x2110, x2111, x2112;
x2110 = x2105;
x2111 = x514;
x2112 = x524;
JCudaTensor[] x2113 = x526.backward(x2110,x2111,x2112);
x2109 = x2113[1];

// val X140670 = X1914 * d_BatchNorm(4b_b_bn)(X302,4b_b_bn_scale)/d_4b_b_bn_bias
JCudaTensor x2114;
x2114 = x2113[2];

// val X1915 = X1914 * d_BatchNorm(4b_b_bn)(X302,4b_b_bn_scale)/d_X302
JCudaTensor x2118;
x2118 = x2113[0];

// Dealloc(X302)
JCudaTensor x2122;
x2122 = x514;
x2122.free();

// val X1916 = X1915 * d_Convolv(1,1)(4b_b_cv_W)/d_X301
JCudaTensor x2123;
JCudaTensor x2124, x2125;
x2124 = x2118;
x2125 = x518;
x2123 = x464.backward_data(x2124, x2125);

// V_4b_b_cv_W <~~ X1915 * d_Convolv(1,1)(X301)/d_4b_b_cv_W
float x2127, x2128;
x2127 = lrn_rate_1;
x2128 = momentum;
JCudaTensor x2129, x2130;
x2129 = x2118;
x2130 = x512;
x464.backward_filter(x2129, x2130, x2126, x2127, x2128);

// Dealloc(X1915)
JCudaTensor x2131;
x2131 = x2118;
x2131.free();

// V_4b_b_bn_scale <~~ X141903
float x2133, x2134;
x2133 = lrn_rate_1;
x2134 = momentum;
JCudaTensor x2135;
x2135 = x2109;
x2132.update(x2135, x2133, x2134);

// Dealloc(X141903)
JCudaTensor x2136;
x2136 = x2109;
x2136.free();

// V_4b_b_bn_bias <~~ X140670
float x2138, x2139;
x2138 = lrn_rate_1;
x2139 = momentum;
JCudaTensor x2140;
x2140 = x2114;
x2137.update(x2140, x2138, x2139);

// Dealloc(X140670)
JCudaTensor x2141;
x2141 = x2114;
x2141.free();

// 4b_b_cv_W <~~ V_4b_b_cv_W
float x2142, x2143;
x2142 = 1;
x2143 = decay_1;
JCudaTensor x2144;
x2144 = x2126;
x518.update(x2144, x2142, x2143);

// 4b_b_bn_scale <~~ V_4b_b_bn_scale
float x2145, x2146;
x2145 = 1;
x2146 = decay_1;
JCudaTensor x2147;
x2147 = x2132;
x524.update(x2147, x2145, x2146);

// 4b_b_bn_bias <~~ V_4b_b_bn_bias
float x2148, x2149;
x2148 = 1;
x2149 = decay_1;
JCudaTensor x2150;
x2150 = x2137;
x525.update(x2150, x2148, x2149);

// val X1920 = X1916 * d_ReLU()(X301)/d_X300
JCudaTensor x2151;
JCudaTensor x2152, x2153;
x2152 = x2123;
x2153 = x512;
x2151 = x457.backward(x2152, x2153);

// Dealloc(X301)
JCudaTensor x2154;
x2154 = x512;
x2154.free();

// val X138189 = X1920 * d_BatchNorm(4b_a_bn)(X299,4b_a_bn_scale)/d_4b_a_bn_scale
JCudaTensor x2155;
JCudaTensor x2156, x2157, x2158;
x2156 = x2151;
x2157 = x498;
x2158 = x509;
JCudaTensor[] x2159 = x511.backward(x2156,x2157,x2158);
x2155 = x2159[1];

// val X1921 = X1920 * d_BatchNorm(4b_a_bn)(X299,4b_a_bn_scale)/d_X299
JCudaTensor x2160;
x2160 = x2159[0];

// val X136943 = X1920 * d_BatchNorm(4b_a_bn)(X299,4b_a_bn_scale)/d_4b_a_bn_bias
JCudaTensor x2164;
x2164 = x2159[2];

// Dealloc(X299)
JCudaTensor x2168;
x2168 = x498;
x2168.free();

// val X1922 = X1921 * d_Convolv(1,0)(4b_a_cv_W)/d_X298
JCudaTensor x2169;
JCudaTensor x2170, x2171;
x2170 = x2160;
x2171 = x502;
x2169 = x504.backward_data(x2170, x2171);

// V_4b_a_cv_W <~~ X1921 * d_Convolv(1,0)(X298)/d_4b_a_cv_W
float x2173, x2174;
x2173 = lrn_rate_1;
x2174 = momentum;
JCudaTensor x2175, x2176;
x2175 = x2160;
x2176 = x496;
x504.backward_filter(x2175, x2176, x2172, x2173, x2174);

// Dealloc(X1921)
JCudaTensor x2177;
x2177 = x2160;
x2177.free();

// V_4b_a_bn_bias <~~ X136943
float x2179, x2180;
x2179 = lrn_rate_1;
x2180 = momentum;
JCudaTensor x2181;
x2181 = x2164;
x2178.update(x2181, x2179, x2180);

// Dealloc(X136943)
JCudaTensor x2182;
x2182 = x2164;
x2182.free();

// V_4b_a_bn_scale <~~ X138189
float x2184, x2185;
x2184 = lrn_rate_1;
x2185 = momentum;
JCudaTensor x2186;
x2186 = x2155;
x2183.update(x2186, x2184, x2185);

// Dealloc(X138189)
JCudaTensor x2187;
x2187 = x2155;
x2187.free();

// 4b_a_cv_W <~~ V_4b_a_cv_W
float x2188, x2189;
x2188 = 1;
x2189 = decay_1;
JCudaTensor x2190;
x2190 = x2172;
x502.update(x2190, x2188, x2189);

// 4b_a_bn_bias <~~ V_4b_a_bn_bias
float x2191, x2192;
x2191 = 1;
x2192 = decay_1;
JCudaTensor x2193;
x2193 = x2178;
x510.update(x2193, x2191, x2192);

// 4b_a_bn_scale <~~ V_4b_a_bn_scale
float x2194, x2195;
x2194 = 1;
x2195 = decay_1;
JCudaTensor x2196;
x2196 = x2183;
x509.update(x2196, x2194, x2195);

// val X1923 = (X1922 + X1898)
JCudaTensor x2197;
JCudaTensor x2198, x2199;
x2198 = x2169;
x2199 = x2055;
x2197 = x2198.plus_i(x2199);

// Dealloc(X1898)
JCudaTensor x2200;
x2200 = x2055;
x2200.free();

// val X1938 = X1923 * d_ReLU()(X298)/d_X297
JCudaTensor x2201;
JCudaTensor x2202, x2203;
x2202 = x2197;
x2203 = x496;
x2201 = x490.backward(x2202, x2203);

// Dealloc(X298)
JCudaTensor x2204;
x2204 = x496;
x2204.free();

// val X1954 = X1938.copy * d_ReLU()(X296)/d_X295
JCudaTensor x2205;
JCudaTensor x2206, x2207;
x2206 = x2201;
x2206 = x2206.clone();
x2207 = x491;
x2205 = x490.backward(x2206, x2207);

// Dealloc(X296)
JCudaTensor x2208;
x2208 = x491;
x2208.free();

// val X1942 = X1938.copy * d_ReLU()(X287)/d_X286
JCudaTensor x2209;
JCudaTensor x2210, x2211;
x2210 = x2201;
x2210 = x2210.clone();
x2211 = x488;
x2209 = x490.backward(x2210, x2211);

// Dealloc(X1938)
JCudaTensor x2212;
x2212 = x2201;
x2212.free();

// Dealloc(X287)
JCudaTensor x2213;
x2213 = x488;
x2213.free();

// val X1955 = X1954 * d_BatchNorm(4a2_c_bn)(X294,4a2_c_bn_scale)/d_X294
JCudaTensor x2214;
JCudaTensor x2215, x2216, x2217;
x2215 = x2205;
x2216 = x474;
x2217 = x485;
JCudaTensor[] x2218 = x487.backward(x2215,x2216,x2217);
x2214 = x2218[0];

// val X134405 = X1954 * d_BatchNorm(4a2_c_bn)(X294,4a2_c_bn_scale)/d_4a2_c_bn_scale
JCudaTensor x2219;
x2219 = x2218[1];

// val X121400 = X1942 * d_BatchNorm(4a1_bn)(X285,4a1_bn_scale)/d_4a1_bn_bias
JCudaTensor x2223;
JCudaTensor x2224, x2225, x2226;
x2224 = x2209;
x2225 = x434;
x2226 = x445;
JCudaTensor[] x2227 = x447.backward(x2224,x2225,x2226);
x2223 = x2227[2];

// val X122681 = X1942 * d_BatchNorm(4a1_bn)(X285,4a1_bn_scale)/d_4a1_bn_scale
JCudaTensor x2228;
x2228 = x2227[1];

// val X1943 = X1942 * d_BatchNorm(4a1_bn)(X285,4a1_bn_scale)/d_X285
JCudaTensor x2232;
x2232 = x2227[0];

// Dealloc(X285)
JCudaTensor x2236;
x2236 = x434;
x2236.free();

// val X133115 = X1954 * d_BatchNorm(4a2_c_bn)(X294,4a2_c_bn_scale)/d_4a2_c_bn_bias
JCudaTensor x2237;
x2237 = x2218[2];

// Dealloc(X294)
JCudaTensor x2241;
x2241 = x474;
x2241.free();

// val X1944 = X1943 * d_Convolv(2,0)(4a1_cv_W)/d_X284
JCudaTensor x2242;
JCudaTensor x2243, x2244;
x2243 = x2232;
x2244 = x438;
x2242 = x440.backward_data(x2243, x2244);

// val X1956 = X1955 * d_Convolv(1,0)(4a2_c_cv_W)/d_X293
JCudaTensor x2245;
JCudaTensor x2246, x2247;
x2246 = x2214;
x2247 = x478;
x2245 = x480.backward_data(x2246, x2247);

// V_4a1_bn_bias <~~ X121400
float x2249, x2250;
x2249 = lrn_rate_1;
x2250 = momentum;
JCudaTensor x2251;
x2251 = x2223;
x2248.update(x2251, x2249, x2250);

// Dealloc(X121400)
JCudaTensor x2252;
x2252 = x2223;
x2252.free();

// V_4a1_cv_W <~~ X1943 * d_Convolv(2,0)(X284)/d_4a1_cv_W
float x2254, x2255;
x2254 = lrn_rate_1;
x2255 = momentum;
JCudaTensor x2256, x2257;
x2256 = x2232;
x2257 = x425;
x440.backward_filter(x2256, x2257, x2253, x2254, x2255);

// Dealloc(X1943)
JCudaTensor x2258;
x2258 = x2232;
x2258.free();

// V_4a2_c_bn_bias <~~ X133115
float x2260, x2261;
x2260 = lrn_rate_1;
x2261 = momentum;
JCudaTensor x2262;
x2262 = x2237;
x2259.update(x2262, x2260, x2261);

// Dealloc(X133115)
JCudaTensor x2263;
x2263 = x2237;
x2263.free();

// V_4a2_c_cv_W <~~ X1955 * d_Convolv(1,0)(X293)/d_4a2_c_cv_W
float x2265, x2266;
x2265 = lrn_rate_1;
x2266 = momentum;
JCudaTensor x2267, x2268;
x2267 = x2214;
x2268 = x472;
x480.backward_filter(x2267, x2268, x2264, x2265, x2266);

// Dealloc(X1955)
JCudaTensor x2269;
x2269 = x2214;
x2269.free();

// V_4a2_c_bn_scale <~~ X134405
float x2271, x2272;
x2271 = lrn_rate_1;
x2272 = momentum;
JCudaTensor x2273;
x2273 = x2219;
x2270.update(x2273, x2271, x2272);

// Dealloc(X134405)
JCudaTensor x2274;
x2274 = x2219;
x2274.free();

// V_4a1_bn_scale <~~ X122681
float x2276, x2277;
x2276 = lrn_rate_1;
x2277 = momentum;
JCudaTensor x2278;
x2278 = x2228;
x2275.update(x2278, x2276, x2277);

// Dealloc(X122681)
JCudaTensor x2279;
x2279 = x2228;
x2279.free();

// 4a1_bn_scale <~~ V_4a1_bn_scale
float x2280, x2281;
x2280 = 1;
x2281 = decay_1;
JCudaTensor x2282;
x2282 = x2275;
x445.update(x2282, x2280, x2281);

// 4a2_c_bn_scale <~~ V_4a2_c_bn_scale
float x2283, x2284;
x2283 = 1;
x2284 = decay_1;
JCudaTensor x2285;
x2285 = x2270;
x485.update(x2285, x2283, x2284);

// 4a1_bn_bias <~~ V_4a1_bn_bias
float x2286, x2287;
x2286 = 1;
x2287 = decay_1;
JCudaTensor x2288;
x2288 = x2248;
x446.update(x2288, x2286, x2287);

// 4a1_cv_W <~~ V_4a1_cv_W
float x2289, x2290;
x2289 = 1;
x2290 = decay_1;
JCudaTensor x2291;
x2291 = x2253;
x438.update(x2291, x2289, x2290);

// 4a2_c_bn_bias <~~ V_4a2_c_bn_bias
float x2292, x2293;
x2292 = 1;
x2293 = decay_1;
JCudaTensor x2294;
x2294 = x2259;
x486.update(x2294, x2292, x2293);

// 4a2_c_cv_W <~~ V_4a2_c_cv_W
float x2295, x2296;
x2295 = 1;
x2296 = decay_1;
JCudaTensor x2297;
x2297 = x2264;
x478.update(x2297, x2295, x2296);

// val X1960 = X1956 * d_ReLU()(X293)/d_X292
JCudaTensor x2298;
JCudaTensor x2299, x2300;
x2299 = x2245;
x2300 = x472;
x2298 = x457.backward(x2299, x2300);

// Dealloc(X293)
JCudaTensor x2301;
x2301 = x472;
x2301.free();

// val X129217 = X1960 * d_BatchNorm(4a2_b_bn)(X291,4a2_b_bn_scale)/d_4a2_b_bn_bias
JCudaTensor x2302;
JCudaTensor x2303, x2304, x2305;
x2303 = x2298;
x2304 = x458;
x2305 = x469;
JCudaTensor[] x2306 = x471.backward(x2303,x2304,x2305);
x2302 = x2306[2];

// val X1961 = X1960 * d_BatchNorm(4a2_b_bn)(X291,4a2_b_bn_scale)/d_X291
JCudaTensor x2307;
x2307 = x2306[0];

// val X130520 = X1960 * d_BatchNorm(4a2_b_bn)(X291,4a2_b_bn_scale)/d_4a2_b_bn_scale
JCudaTensor x2311;
x2311 = x2306[1];

// Dealloc(X291)
JCudaTensor x2315;
x2315 = x458;
x2315.free();

// val X1962 = X1961 * d_Convolv(1,1)(4a2_b_cv_W)/d_X290
JCudaTensor x2316;
JCudaTensor x2317, x2318;
x2317 = x2307;
x2318 = x462;
x2316 = x464.backward_data(x2317, x2318);

// V_4a2_b_cv_W <~~ X1961 * d_Convolv(1,1)(X290)/d_4a2_b_cv_W
float x2320, x2321;
x2320 = lrn_rate_1;
x2321 = momentum;
JCudaTensor x2322, x2323;
x2322 = x2307;
x2323 = x455;
x464.backward_filter(x2322, x2323, x2319, x2320, x2321);

// Dealloc(X1961)
JCudaTensor x2324;
x2324 = x2307;
x2324.free();

// V_4a2_b_bn_scale <~~ X130520
float x2326, x2327;
x2326 = lrn_rate_1;
x2327 = momentum;
JCudaTensor x2328;
x2328 = x2311;
x2325.update(x2328, x2326, x2327);

// Dealloc(X130520)
JCudaTensor x2329;
x2329 = x2311;
x2329.free();

// V_4a2_b_bn_bias <~~ X129217
float x2331, x2332;
x2331 = lrn_rate_1;
x2332 = momentum;
JCudaTensor x2333;
x2333 = x2302;
x2330.update(x2333, x2331, x2332);

// Dealloc(X129217)
JCudaTensor x2334;
x2334 = x2302;
x2334.free();

// 4a2_b_cv_W <~~ V_4a2_b_cv_W
float x2335, x2336;
x2335 = 1;
x2336 = decay_1;
JCudaTensor x2337;
x2337 = x2319;
x462.update(x2337, x2335, x2336);

// 4a2_b_bn_scale <~~ V_4a2_b_bn_scale
float x2338, x2339;
x2338 = 1;
x2339 = decay_1;
JCudaTensor x2340;
x2340 = x2325;
x469.update(x2340, x2338, x2339);

// 4a2_b_bn_bias <~~ V_4a2_b_bn_bias
float x2341, x2342;
x2341 = 1;
x2342 = decay_1;
JCudaTensor x2343;
x2343 = x2330;
x470.update(x2343, x2341, x2342);

// val X1966 = X1962 * d_ReLU()(X290)/d_X289
JCudaTensor x2344;
JCudaTensor x2345, x2346;
x2345 = x2316;
x2346 = x455;
x2344 = x457.backward(x2345, x2346);

// Dealloc(X290)
JCudaTensor x2347;
x2347 = x455;
x2347.free();

// val X125280 = X1966 * d_BatchNorm(4a2_a_bn)(X288,4a2_a_bn_scale)/d_4a2_a_bn_bias
JCudaTensor x2348;
JCudaTensor x2349, x2350, x2351;
x2349 = x2344;
x2350 = x427;
x2351 = x452;
JCudaTensor[] x2352 = x454.backward(x2349,x2350,x2351);
x2348 = x2352[2];

// val X1967 = X1966 * d_BatchNorm(4a2_a_bn)(X288,4a2_a_bn_scale)/d_X288
JCudaTensor x2353;
x2353 = x2352[0];

// val X126596 = X1966 * d_BatchNorm(4a2_a_bn)(X288,4a2_a_bn_scale)/d_4a2_a_bn_scale
JCudaTensor x2357;
x2357 = x2352[1];

// Dealloc(X288)
JCudaTensor x2361;
x2361 = x427;
x2361.free();

// val X1969 = (X1944 + X1967 * d_Convolv(2,0)(4a2_a_cv_W)/d_X284)
JCudaTensor x2362;
JCudaTensor x2363;
x2363 = x2242;
JCudaTensor x2364, x2365;
x2364 = x2353;
x2365 = x431;
x2362 = x433.backward_data(x2364,x2365, x2363);

// V_4a2_a_cv_W <~~ X1967 * d_Convolv(2,0)(X284)/d_4a2_a_cv_W
float x2367, x2368;
x2367 = lrn_rate_1;
x2368 = momentum;
JCudaTensor x2369, x2370;
x2369 = x2353;
x2370 = x425;
x433.backward_filter(x2369, x2370, x2366, x2367, x2368);

// Dealloc(X1967)
JCudaTensor x2371;
x2371 = x2353;
x2371.free();

// V_4a2_a_bn_bias <~~ X125280
float x2373, x2374;
x2373 = lrn_rate_1;
x2374 = momentum;
JCudaTensor x2375;
x2375 = x2348;
x2372.update(x2375, x2373, x2374);

// Dealloc(X125280)
JCudaTensor x2376;
x2376 = x2348;
x2376.free();

// V_4a2_a_bn_scale <~~ X126596
float x2378, x2379;
x2378 = lrn_rate_1;
x2379 = momentum;
JCudaTensor x2380;
x2380 = x2357;
x2377.update(x2380, x2378, x2379);

// Dealloc(X126596)
JCudaTensor x2381;
x2381 = x2357;
x2381.free();

// 4a2_a_cv_W <~~ V_4a2_a_cv_W
float x2382, x2383;
x2382 = 1;
x2383 = decay_1;
JCudaTensor x2384;
x2384 = x2366;
x431.update(x2384, x2382, x2383);

// 4a2_a_bn_bias <~~ V_4a2_a_bn_bias
float x2385, x2386;
x2385 = 1;
x2386 = decay_1;
JCudaTensor x2387;
x2387 = x2372;
x453.update(x2387, x2385, x2386);

// 4a2_a_bn_scale <~~ V_4a2_a_bn_scale
float x2388, x2389;
x2388 = 1;
x2389 = decay_1;
JCudaTensor x2390;
x2390 = x2377;
x452.update(x2390, x2388, x2389);

// val X2017 = X1969 * d_ReLU()(X284)/d_X283
JCudaTensor x2391;
JCudaTensor x2392, x2393;
x2392 = x2362;
x2393 = x425;
x2391 = x268.backward(x2392, x2393);

// Dealloc(X284)
JCudaTensor x2394;
x2394 = x425;
x2394.free();

// val X2027 = X2017.copy * d_ReLU()(X282)/d_X281
JCudaTensor x2395;
JCudaTensor x2396, x2397;
x2396 = x2391;
x2396 = x2396.clone();
x2397 = x420;
x2395 = x268.backward(x2396, x2397);

// Dealloc(X282)
JCudaTensor x2398;
x2398 = x420;
x2398.free();

// val X117297 = X2027 * d_BatchNorm(3d_c_bn)(X280,3d_c_bn_scale)/d_3d_c_bn_bias
JCudaTensor x2399;
JCudaTensor x2400, x2401, x2402;
x2400 = x2395;
x2401 = x407;
x2402 = x417;
JCudaTensor[] x2403 = x419.backward(x2400,x2401,x2402);
x2399 = x2403[2];

// val X118707 = X2027 * d_BatchNorm(3d_c_bn)(X280,3d_c_bn_scale)/d_3d_c_bn_scale
JCudaTensor x2404;
x2404 = x2403[1];

// val X2028 = X2027 * d_BatchNorm(3d_c_bn)(X280,3d_c_bn_scale)/d_X280
JCudaTensor x2408;
x2408 = x2403[0];

// Dealloc(X280)
JCudaTensor x2412;
x2412 = x407;
x2412.free();

// val X2029 = X2028 * d_Convolv(1,0)(3d_c_cv_W)/d_X279
JCudaTensor x2413;
JCudaTensor x2414, x2415;
x2414 = x2408;
x2415 = x411;
x2413 = x258.backward_data(x2414, x2415);

// V_3d_c_cv_W <~~ X2028 * d_Convolv(1,0)(X279)/d_3d_c_cv_W
float x2417, x2418;
x2417 = lrn_rate_1;
x2418 = momentum;
JCudaTensor x2419, x2420;
x2419 = x2408;
x2420 = x405;
x258.backward_filter(x2419, x2420, x2416, x2417, x2418);

// Dealloc(X2028)
JCudaTensor x2421;
x2421 = x2408;
x2421.free();

// V_3d_c_bn_bias <~~ X117297
float x2423, x2424;
x2423 = lrn_rate_1;
x2424 = momentum;
JCudaTensor x2425;
x2425 = x2399;
x2422.update(x2425, x2423, x2424);

// Dealloc(X117297)
JCudaTensor x2426;
x2426 = x2399;
x2426.free();

// V_3d_c_bn_scale <~~ X118707
float x2428, x2429;
x2428 = lrn_rate_1;
x2429 = momentum;
JCudaTensor x2430;
x2430 = x2404;
x2427.update(x2430, x2428, x2429);

// Dealloc(X118707)
JCudaTensor x2431;
x2431 = x2404;
x2431.free();

// 3d_c_cv_W <~~ V_3d_c_cv_W
float x2432, x2433;
x2432 = 1;
x2433 = decay_1;
JCudaTensor x2434;
x2434 = x2416;
x411.update(x2434, x2432, x2433);

// 3d_c_bn_bias <~~ V_3d_c_bn_bias
float x2435, x2436;
x2435 = 1;
x2436 = decay_1;
JCudaTensor x2437;
x2437 = x2422;
x418.update(x2437, x2435, x2436);

// 3d_c_bn_scale <~~ V_3d_c_bn_scale
float x2438, x2439;
x2438 = 1;
x2439 = decay_1;
JCudaTensor x2440;
x2440 = x2427;
x417.update(x2440, x2438, x2439);

// val X2033 = X2029 * d_ReLU()(X279)/d_X278
JCudaTensor x2441;
JCudaTensor x2442, x2443;
x2442 = x2413;
x2443 = x405;
x2441 = x235.backward(x2442, x2443);

// Dealloc(X279)
JCudaTensor x2444;
x2444 = x405;
x2444.free();

// val X113039 = X2033 * d_BatchNorm(3d_b_bn)(X277,3d_b_bn_scale)/d_3d_b_bn_bias
JCudaTensor x2445;
JCudaTensor x2446, x2447, x2448;
x2446 = x2441;
x2447 = x392;
x2448 = x402;
JCudaTensor[] x2449 = x404.backward(x2446,x2447,x2448);
x2445 = x2449[2];

// val X2034 = X2033 * d_BatchNorm(3d_b_bn)(X277,3d_b_bn_scale)/d_X277
JCudaTensor x2450;
x2450 = x2449[0];

// val X114462 = X2033 * d_BatchNorm(3d_b_bn)(X277,3d_b_bn_scale)/d_3d_b_bn_scale
JCudaTensor x2454;
x2454 = x2449[1];

// Dealloc(X277)
JCudaTensor x2458;
x2458 = x392;
x2458.free();

// val X2035 = X2034 * d_Convolv(1,1)(3d_b_cv_W)/d_X276
JCudaTensor x2459;
JCudaTensor x2460, x2461;
x2460 = x2450;
x2461 = x396;
x2459 = x242.backward_data(x2460, x2461);

// V_3d_b_bn_bias <~~ X113039
float x2463, x2464;
x2463 = lrn_rate_1;
x2464 = momentum;
JCudaTensor x2465;
x2465 = x2445;
x2462.update(x2465, x2463, x2464);

// Dealloc(X113039)
JCudaTensor x2466;
x2466 = x2445;
x2466.free();

// V_3d_b_bn_scale <~~ X114462
float x2468, x2469;
x2468 = lrn_rate_1;
x2469 = momentum;
JCudaTensor x2470;
x2470 = x2454;
x2467.update(x2470, x2468, x2469);

// Dealloc(X114462)
JCudaTensor x2471;
x2471 = x2454;
x2471.free();

// V_3d_b_cv_W <~~ X2034 * d_Convolv(1,1)(X276)/d_3d_b_cv_W
float x2473, x2474;
x2473 = lrn_rate_1;
x2474 = momentum;
JCudaTensor x2475, x2476;
x2475 = x2450;
x2476 = x390;
x242.backward_filter(x2475, x2476, x2472, x2473, x2474);

// Dealloc(X2034)
JCudaTensor x2477;
x2477 = x2450;
x2477.free();

// 3d_b_bn_bias <~~ V_3d_b_bn_bias
float x2478, x2479;
x2478 = 1;
x2479 = decay_1;
JCudaTensor x2480;
x2480 = x2462;
x403.update(x2480, x2478, x2479);

// 3d_b_bn_scale <~~ V_3d_b_bn_scale
float x2481, x2482;
x2481 = 1;
x2482 = decay_1;
JCudaTensor x2483;
x2483 = x2467;
x402.update(x2483, x2481, x2482);

// 3d_b_cv_W <~~ V_3d_b_cv_W
float x2484, x2485;
x2484 = 1;
x2485 = decay_1;
JCudaTensor x2486;
x2486 = x2472;
x396.update(x2486, x2484, x2485);

// val X2039 = X2035 * d_ReLU()(X276)/d_X275
JCudaTensor x2487;
JCudaTensor x2488, x2489;
x2488 = x2459;
x2489 = x390;
x2487 = x235.backward(x2488, x2489);

// Dealloc(X276)
JCudaTensor x2490;
x2490 = x390;
x2490.free();

// val X108742 = X2039 * d_BatchNorm(3d_a_bn)(X274,3d_a_bn_scale)/d_3d_a_bn_bias
JCudaTensor x2491;
JCudaTensor x2492, x2493, x2494;
x2492 = x2487;
x2493 = x377;
x2494 = x387;
JCudaTensor[] x2495 = x389.backward(x2492,x2493,x2494);
x2491 = x2495[2];

// val X2040 = X2039 * d_BatchNorm(3d_a_bn)(X274,3d_a_bn_scale)/d_X274
JCudaTensor x2496;
x2496 = x2495[0];

// val X110178 = X2039 * d_BatchNorm(3d_a_bn)(X274,3d_a_bn_scale)/d_3d_a_bn_scale
JCudaTensor x2500;
x2500 = x2495[1];

// Dealloc(X274)
JCudaTensor x2504;
x2504 = x377;
x2504.free();

// V_3d_a_bn_scale <~~ X110178
float x2506, x2507;
x2506 = lrn_rate_1;
x2507 = momentum;
JCudaTensor x2508;
x2508 = x2500;
x2505.update(x2508, x2506, x2507);

// Dealloc(X110178)
JCudaTensor x2509;
x2509 = x2500;
x2509.free();

// val X2041 = X2040 * d_Convolv(1,0)(3d_a_cv_W)/d_X273
JCudaTensor x2510;
JCudaTensor x2511, x2512;
x2511 = x2496;
x2512 = x381;
x2510 = x282.backward_data(x2511, x2512);

// V_3d_a_bn_bias <~~ X108742
float x2514, x2515;
x2514 = lrn_rate_1;
x2515 = momentum;
JCudaTensor x2516;
x2516 = x2491;
x2513.update(x2516, x2514, x2515);

// Dealloc(X108742)
JCudaTensor x2517;
x2517 = x2491;
x2517.free();

// V_3d_a_cv_W <~~ X2040 * d_Convolv(1,0)(X273)/d_3d_a_cv_W
float x2519, x2520;
x2519 = lrn_rate_1;
x2520 = momentum;
JCudaTensor x2521, x2522;
x2521 = x2496;
x2522 = x375;
x282.backward_filter(x2521, x2522, x2518, x2519, x2520);

// Dealloc(X2040)
JCudaTensor x2523;
x2523 = x2496;
x2523.free();

// 3d_a_bn_scale <~~ V_3d_a_bn_scale
float x2524, x2525;
x2524 = 1;
x2525 = decay_1;
JCudaTensor x2526;
x2526 = x2505;
x387.update(x2526, x2524, x2525);

// 3d_a_bn_bias <~~ V_3d_a_bn_bias
float x2527, x2528;
x2527 = 1;
x2528 = decay_1;
JCudaTensor x2529;
x2529 = x2513;
x388.update(x2529, x2527, x2528);

// 3d_a_cv_W <~~ V_3d_a_cv_W
float x2530, x2531;
x2530 = 1;
x2531 = decay_1;
JCudaTensor x2532;
x2532 = x2518;
x381.update(x2532, x2530, x2531);

// val X2042 = (X2041 + X2017)
JCudaTensor x2533;
JCudaTensor x2534, x2535;
x2534 = x2510;
x2535 = x2391;
x2533 = x2534.plus_i(x2535);

// Dealloc(X2017)
JCudaTensor x2536;
x2536 = x2391;
x2536.free();

// val X2054 = X2042 * d_ReLU()(X273)/d_X272
JCudaTensor x2537;
JCudaTensor x2538, x2539;
x2538 = x2533;
x2539 = x375;
x2537 = x268.backward(x2538, x2539);

// Dealloc(X273)
JCudaTensor x2540;
x2540 = x375;
x2540.free();

// val X2064 = X2054.copy * d_ReLU()(X271)/d_X270
JCudaTensor x2541;
JCudaTensor x2542, x2543;
x2542 = x2537;
x2542 = x2542.clone();
x2543 = x370;
x2541 = x268.backward(x2542, x2543);

// Dealloc(X271)
JCudaTensor x2544;
x2544 = x370;
x2544.free();

// val X2065 = X2064 * d_BatchNorm(3c_c_bn)(X269,3c_c_bn_scale)/d_X269
JCudaTensor x2545;
JCudaTensor x2546, x2547, x2548;
x2546 = x2541;
x2547 = x357;
x2548 = x367;
JCudaTensor[] x2549 = x369.backward(x2546,x2547,x2548);
x2545 = x2549[0];

// val X105827 = X2064 * d_BatchNorm(3c_c_bn)(X269,3c_c_bn_scale)/d_3c_c_bn_scale
JCudaTensor x2550;
x2550 = x2549[1];

// val X104350 = X2064 * d_BatchNorm(3c_c_bn)(X269,3c_c_bn_scale)/d_3c_c_bn_bias
JCudaTensor x2554;
x2554 = x2549[2];

// Dealloc(X269)
JCudaTensor x2558;
x2558 = x357;
x2558.free();

// V_3c_c_bn_bias <~~ X104350
float x2560, x2561;
x2560 = lrn_rate_1;
x2561 = momentum;
JCudaTensor x2562;
x2562 = x2554;
x2559.update(x2562, x2560, x2561);

// Dealloc(X104350)
JCudaTensor x2563;
x2563 = x2554;
x2563.free();

// val X2066 = X2065 * d_Convolv(1,0)(3c_c_cv_W)/d_X268
JCudaTensor x2564;
JCudaTensor x2565, x2566;
x2565 = x2545;
x2566 = x361;
x2564 = x258.backward_data(x2565, x2566);

// V_3c_c_cv_W <~~ X2065 * d_Convolv(1,0)(X268)/d_3c_c_cv_W
float x2568, x2569;
x2568 = lrn_rate_1;
x2569 = momentum;
JCudaTensor x2570, x2571;
x2570 = x2545;
x2571 = x355;
x258.backward_filter(x2570, x2571, x2567, x2568, x2569);

// Dealloc(X2065)
JCudaTensor x2572;
x2572 = x2545;
x2572.free();

// V_3c_c_bn_scale <~~ X105827
float x2574, x2575;
x2574 = lrn_rate_1;
x2575 = momentum;
JCudaTensor x2576;
x2576 = x2550;
x2573.update(x2576, x2574, x2575);

// Dealloc(X105827)
JCudaTensor x2577;
x2577 = x2550;
x2577.free();

// 3c_c_bn_bias <~~ V_3c_c_bn_bias
float x2578, x2579;
x2578 = 1;
x2579 = decay_1;
JCudaTensor x2580;
x2580 = x2559;
x368.update(x2580, x2578, x2579);

// 3c_c_cv_W <~~ V_3c_c_cv_W
float x2581, x2582;
x2581 = 1;
x2582 = decay_1;
JCudaTensor x2583;
x2583 = x2567;
x361.update(x2583, x2581, x2582);

// 3c_c_bn_scale <~~ V_3c_c_bn_scale
float x2584, x2585;
x2584 = 1;
x2585 = decay_1;
JCudaTensor x2586;
x2586 = x2573;
x367.update(x2586, x2584, x2585);

// val X2070 = X2066 * d_ReLU()(X268)/d_X267
JCudaTensor x2587;
JCudaTensor x2588, x2589;
x2588 = x2564;
x2589 = x355;
x2587 = x235.backward(x2588, x2589);

// Dealloc(X268)
JCudaTensor x2590;
x2590 = x355;
x2590.free();

// val X101381 = X2070 * d_BatchNorm(3c_b_bn)(X266,3c_b_bn_scale)/d_3c_b_bn_scale
JCudaTensor x2591;
JCudaTensor x2592, x2593, x2594;
x2592 = x2587;
x2593 = x342;
x2594 = x352;
JCudaTensor[] x2595 = x354.backward(x2592,x2593,x2594);
x2591 = x2595[1];

// val X2071 = X2070 * d_BatchNorm(3c_b_bn)(X266,3c_b_bn_scale)/d_X266
JCudaTensor x2596;
x2596 = x2595[0];

// val X99891 = X2070 * d_BatchNorm(3c_b_bn)(X266,3c_b_bn_scale)/d_3c_b_bn_bias
JCudaTensor x2600;
x2600 = x2595[2];

// Dealloc(X266)
JCudaTensor x2604;
x2604 = x342;
x2604.free();

// val X2072 = X2071 * d_Convolv(1,1)(3c_b_cv_W)/d_X265
JCudaTensor x2605;
JCudaTensor x2606, x2607;
x2606 = x2596;
x2607 = x346;
x2605 = x242.backward_data(x2606, x2607);

// V_3c_b_bn_scale <~~ X101381
float x2609, x2610;
x2609 = lrn_rate_1;
x2610 = momentum;
JCudaTensor x2611;
x2611 = x2591;
x2608.update(x2611, x2609, x2610);

// Dealloc(X101381)
JCudaTensor x2612;
x2612 = x2591;
x2612.free();

// V_3c_b_cv_W <~~ X2071 * d_Convolv(1,1)(X265)/d_3c_b_cv_W
float x2614, x2615;
x2614 = lrn_rate_1;
x2615 = momentum;
JCudaTensor x2616, x2617;
x2616 = x2596;
x2617 = x340;
x242.backward_filter(x2616, x2617, x2613, x2614, x2615);

// Dealloc(X2071)
JCudaTensor x2618;
x2618 = x2596;
x2618.free();

// V_3c_b_bn_bias <~~ X99891
float x2620, x2621;
x2620 = lrn_rate_1;
x2621 = momentum;
JCudaTensor x2622;
x2622 = x2600;
x2619.update(x2622, x2620, x2621);

// Dealloc(X99891)
JCudaTensor x2623;
x2623 = x2600;
x2623.free();

// 3c_b_bn_scale <~~ V_3c_b_bn_scale
float x2624, x2625;
x2624 = 1;
x2625 = decay_1;
JCudaTensor x2626;
x2626 = x2608;
x352.update(x2626, x2624, x2625);

// 3c_b_cv_W <~~ V_3c_b_cv_W
float x2627, x2628;
x2627 = 1;
x2628 = decay_1;
JCudaTensor x2629;
x2629 = x2613;
x346.update(x2629, x2627, x2628);

// 3c_b_bn_bias <~~ V_3c_b_bn_bias
float x2630, x2631;
x2630 = 1;
x2631 = decay_1;
JCudaTensor x2632;
x2632 = x2619;
x353.update(x2632, x2630, x2631);

// val X2076 = X2072 * d_ReLU()(X265)/d_X264
JCudaTensor x2633;
JCudaTensor x2634, x2635;
x2634 = x2605;
x2635 = x340;
x2633 = x235.backward(x2634, x2635);

// Dealloc(X265)
JCudaTensor x2636;
x2636 = x340;
x2636.free();

// val X95393 = X2076 * d_BatchNorm(3c_a_bn)(X263,3c_a_bn_scale)/d_3c_a_bn_bias
JCudaTensor x2637;
JCudaTensor x2638, x2639, x2640;
x2638 = x2633;
x2639 = x327;
x2640 = x337;
JCudaTensor[] x2641 = x339.backward(x2638,x2639,x2640);
x2637 = x2641[2];

// val X96896 = X2076 * d_BatchNorm(3c_a_bn)(X263,3c_a_bn_scale)/d_3c_a_bn_scale
JCudaTensor x2642;
x2642 = x2641[1];

// val X2077 = X2076 * d_BatchNorm(3c_a_bn)(X263,3c_a_bn_scale)/d_X263
JCudaTensor x2646;
x2646 = x2641[0];

// Dealloc(X263)
JCudaTensor x2650;
x2650 = x327;
x2650.free();

// val X2078 = X2077 * d_Convolv(1,0)(3c_a_cv_W)/d_X262
JCudaTensor x2651;
JCudaTensor x2652, x2653;
x2652 = x2646;
x2653 = x331;
x2651 = x282.backward_data(x2652, x2653);

// V_3c_a_cv_W <~~ X2077 * d_Convolv(1,0)(X262)/d_3c_a_cv_W
float x2655, x2656;
x2655 = lrn_rate_1;
x2656 = momentum;
JCudaTensor x2657, x2658;
x2657 = x2646;
x2658 = x325;
x282.backward_filter(x2657, x2658, x2654, x2655, x2656);

// Dealloc(X2077)
JCudaTensor x2659;
x2659 = x2646;
x2659.free();

// V_3c_a_bn_bias <~~ X95393
float x2661, x2662;
x2661 = lrn_rate_1;
x2662 = momentum;
JCudaTensor x2663;
x2663 = x2637;
x2660.update(x2663, x2661, x2662);

// Dealloc(X95393)
JCudaTensor x2664;
x2664 = x2637;
x2664.free();

// V_3c_a_bn_scale <~~ X96896
float x2666, x2667;
x2666 = lrn_rate_1;
x2667 = momentum;
JCudaTensor x2668;
x2668 = x2642;
x2665.update(x2668, x2666, x2667);

// Dealloc(X96896)
JCudaTensor x2669;
x2669 = x2642;
x2669.free();

// 3c_a_cv_W <~~ V_3c_a_cv_W
float x2670, x2671;
x2670 = 1;
x2671 = decay_1;
JCudaTensor x2672;
x2672 = x2654;
x331.update(x2672, x2670, x2671);

// 3c_a_bn_bias <~~ V_3c_a_bn_bias
float x2673, x2674;
x2673 = 1;
x2674 = decay_1;
JCudaTensor x2675;
x2675 = x2660;
x338.update(x2675, x2673, x2674);

// 3c_a_bn_scale <~~ V_3c_a_bn_scale
float x2676, x2677;
x2676 = 1;
x2677 = decay_1;
JCudaTensor x2678;
x2678 = x2665;
x337.update(x2678, x2676, x2677);

// val X2079 = (X2078 + X2054)
JCudaTensor x2679;
JCudaTensor x2680, x2681;
x2680 = x2651;
x2681 = x2537;
x2679 = x2680.plus_i(x2681);

// Dealloc(X2054)
JCudaTensor x2682;
x2682 = x2537;
x2682.free();

// val X2091 = X2079 * d_ReLU()(X262)/d_X261
JCudaTensor x2683;
JCudaTensor x2684, x2685;
x2684 = x2679;
x2685 = x325;
x2683 = x268.backward(x2684, x2685);

// Dealloc(X262)
JCudaTensor x2686;
x2686 = x325;
x2686.free();

// val X2101 = X2091.copy * d_ReLU()(X260)/d_X259
JCudaTensor x2687;
JCudaTensor x2688, x2689;
x2688 = x2683;
x2688 = x2688.clone();
x2689 = x320;
x2687 = x268.backward(x2688, x2689);

// Dealloc(X260)
JCudaTensor x2690;
x2690 = x320;
x2690.free();

// val X2102 = X2101 * d_BatchNorm(3b_c_bn)(X258,3b_c_bn_scale)/d_X258
JCudaTensor x2691;
JCudaTensor x2692, x2693, x2694;
x2692 = x2687;
x2693 = x307;
x2694 = x317;
JCudaTensor[] x2695 = x319.backward(x2692,x2693,x2694);
x2691 = x2695[0];

// val X92344 = X2101 * d_BatchNorm(3b_c_bn)(X258,3b_c_bn_scale)/d_3b_c_bn_scale
JCudaTensor x2696;
x2696 = x2695[1];

// val X90800 = X2101 * d_BatchNorm(3b_c_bn)(X258,3b_c_bn_scale)/d_3b_c_bn_bias
JCudaTensor x2700;
x2700 = x2695[2];

// Dealloc(X258)
JCudaTensor x2704;
x2704 = x307;
x2704.free();

// V_3b_c_bn_bias <~~ X90800
float x2706, x2707;
x2706 = lrn_rate_1;
x2707 = momentum;
JCudaTensor x2708;
x2708 = x2700;
x2705.update(x2708, x2706, x2707);

// Dealloc(X90800)
JCudaTensor x2709;
x2709 = x2700;
x2709.free();

// V_3b_c_bn_scale <~~ X92344
float x2711, x2712;
x2711 = lrn_rate_1;
x2712 = momentum;
JCudaTensor x2713;
x2713 = x2696;
x2710.update(x2713, x2711, x2712);

// Dealloc(X92344)
JCudaTensor x2714;
x2714 = x2696;
x2714.free();

// val X2103 = X2102 * d_Convolv(1,0)(3b_c_cv_W)/d_X257
JCudaTensor x2715;
JCudaTensor x2716, x2717;
x2716 = x2691;
x2717 = x311;
x2715 = x258.backward_data(x2716, x2717);

// V_3b_c_cv_W <~~ X2102 * d_Convolv(1,0)(X257)/d_3b_c_cv_W
float x2719, x2720;
x2719 = lrn_rate_1;
x2720 = momentum;
JCudaTensor x2721, x2722;
x2721 = x2691;
x2722 = x305;
x258.backward_filter(x2721, x2722, x2718, x2719, x2720);

// Dealloc(X2102)
JCudaTensor x2723;
x2723 = x2691;
x2723.free();

// 3b_c_bn_bias <~~ V_3b_c_bn_bias
float x2724, x2725;
x2724 = 1;
x2725 = decay_1;
JCudaTensor x2726;
x2726 = x2705;
x318.update(x2726, x2724, x2725);

// 3b_c_bn_scale <~~ V_3b_c_bn_scale
float x2727, x2728;
x2727 = 1;
x2728 = decay_1;
JCudaTensor x2729;
x2729 = x2710;
x317.update(x2729, x2727, x2728);

// 3b_c_cv_W <~~ V_3b_c_cv_W
float x2730, x2731;
x2730 = 1;
x2731 = decay_1;
JCudaTensor x2732;
x2732 = x2718;
x311.update(x2732, x2730, x2731);

// val X2107 = X2103 * d_ReLU()(X257)/d_X256
JCudaTensor x2733;
JCudaTensor x2734, x2735;
x2734 = x2715;
x2735 = x305;
x2733 = x235.backward(x2734, x2735);

// Dealloc(X257)
JCudaTensor x2736;
x2736 = x305;
x2736.free();

// val X86140 = X2107 * d_BatchNorm(3b_b_bn)(X255,3b_b_bn_scale)/d_3b_b_bn_bias
JCudaTensor x2737;
JCudaTensor x2738, x2739, x2740;
x2738 = x2733;
x2739 = x292;
x2740 = x302;
JCudaTensor[] x2741 = x304.backward(x2738,x2739,x2740);
x2737 = x2741[2];

// val X2108 = X2107 * d_BatchNorm(3b_b_bn)(X255,3b_b_bn_scale)/d_X255
JCudaTensor x2742;
x2742 = x2741[0];

// val X87697 = X2107 * d_BatchNorm(3b_b_bn)(X255,3b_b_bn_scale)/d_3b_b_bn_scale
JCudaTensor x2746;
x2746 = x2741[1];

// Dealloc(X255)
JCudaTensor x2750;
x2750 = x292;
x2750.free();

// V_3b_b_bn_scale <~~ X87697
float x2752, x2753;
x2752 = lrn_rate_1;
x2753 = momentum;
JCudaTensor x2754;
x2754 = x2746;
x2751.update(x2754, x2752, x2753);

// Dealloc(X87697)
JCudaTensor x2755;
x2755 = x2746;
x2755.free();

// V_3b_b_bn_bias <~~ X86140
float x2757, x2758;
x2757 = lrn_rate_1;
x2758 = momentum;
JCudaTensor x2759;
x2759 = x2737;
x2756.update(x2759, x2757, x2758);

// Dealloc(X86140)
JCudaTensor x2760;
x2760 = x2737;
x2760.free();

// val X2109 = X2108 * d_Convolv(1,1)(3b_b_cv_W)/d_X254
JCudaTensor x2761;
JCudaTensor x2762, x2763;
x2762 = x2742;
x2763 = x296;
x2761 = x242.backward_data(x2762, x2763);

// V_3b_b_cv_W <~~ X2108 * d_Convolv(1,1)(X254)/d_3b_b_cv_W
float x2765, x2766;
x2765 = lrn_rate_1;
x2766 = momentum;
JCudaTensor x2767, x2768;
x2767 = x2742;
x2768 = x290;
x242.backward_filter(x2767, x2768, x2764, x2765, x2766);

// Dealloc(X2108)
JCudaTensor x2769;
x2769 = x2742;
x2769.free();

// 3b_b_bn_scale <~~ V_3b_b_bn_scale
float x2770, x2771;
x2770 = 1;
x2771 = decay_1;
JCudaTensor x2772;
x2772 = x2751;
x302.update(x2772, x2770, x2771);

// 3b_b_bn_bias <~~ V_3b_b_bn_bias
float x2773, x2774;
x2773 = 1;
x2774 = decay_1;
JCudaTensor x2775;
x2775 = x2756;
x303.update(x2775, x2773, x2774);

// 3b_b_cv_W <~~ V_3b_b_cv_W
float x2776, x2777;
x2776 = 1;
x2777 = decay_1;
JCudaTensor x2778;
x2778 = x2764;
x296.update(x2778, x2776, x2777);

// val X2113 = X2109 * d_ReLU()(X254)/d_X253
JCudaTensor x2779;
JCudaTensor x2780, x2781;
x2780 = x2761;
x2781 = x290;
x2779 = x235.backward(x2780, x2781);

// Dealloc(X254)
JCudaTensor x2782;
x2782 = x290;
x2782.free();

// val X83011 = X2113 * d_BatchNorm(3b_a_bn)(X252,3b_a_bn_scale)/d_3b_a_bn_scale
JCudaTensor x2783;
JCudaTensor x2784, x2785, x2786;
x2784 = x2779;
x2785 = x276;
x2786 = x287;
JCudaTensor[] x2787 = x289.backward(x2784,x2785,x2786);
x2783 = x2787[1];

// val X2114 = X2113 * d_BatchNorm(3b_a_bn)(X252,3b_a_bn_scale)/d_X252
JCudaTensor x2788;
x2788 = x2787[0];

// val X81441 = X2113 * d_BatchNorm(3b_a_bn)(X252,3b_a_bn_scale)/d_3b_a_bn_bias
JCudaTensor x2792;
x2792 = x2787[2];

// Dealloc(X252)
JCudaTensor x2796;
x2796 = x276;
x2796.free();

// V_3b_a_bn_scale <~~ X83011
float x2798, x2799;
x2798 = lrn_rate_1;
x2799 = momentum;
JCudaTensor x2800;
x2800 = x2783;
x2797.update(x2800, x2798, x2799);

// Dealloc(X83011)
JCudaTensor x2801;
x2801 = x2783;
x2801.free();

// V_3b_a_bn_bias <~~ X81441
float x2803, x2804;
x2803 = lrn_rate_1;
x2804 = momentum;
JCudaTensor x2805;
x2805 = x2792;
x2802.update(x2805, x2803, x2804);

// Dealloc(X81441)
JCudaTensor x2806;
x2806 = x2792;
x2806.free();

// val X2115 = X2114 * d_Convolv(1,0)(3b_a_cv_W)/d_X251
JCudaTensor x2807;
JCudaTensor x2808, x2809;
x2808 = x2788;
x2809 = x280;
x2807 = x282.backward_data(x2808, x2809);

// V_3b_a_cv_W <~~ X2114 * d_Convolv(1,0)(X251)/d_3b_a_cv_W
float x2811, x2812;
x2811 = lrn_rate_1;
x2812 = momentum;
JCudaTensor x2813, x2814;
x2813 = x2788;
x2814 = x274;
x282.backward_filter(x2813, x2814, x2810, x2811, x2812);

// Dealloc(X2114)
JCudaTensor x2815;
x2815 = x2788;
x2815.free();

// 3b_a_bn_scale <~~ V_3b_a_bn_scale
float x2816, x2817;
x2816 = 1;
x2817 = decay_1;
JCudaTensor x2818;
x2818 = x2797;
x287.update(x2818, x2816, x2817);

// 3b_a_bn_bias <~~ V_3b_a_bn_bias
float x2819, x2820;
x2819 = 1;
x2820 = decay_1;
JCudaTensor x2821;
x2821 = x2802;
x288.update(x2821, x2819, x2820);

// 3b_a_cv_W <~~ V_3b_a_cv_W
float x2822, x2823;
x2822 = 1;
x2823 = decay_1;
JCudaTensor x2824;
x2824 = x2810;
x280.update(x2824, x2822, x2823);

// val X2116 = (X2115 + X2091)
JCudaTensor x2825;
JCudaTensor x2826, x2827;
x2826 = x2807;
x2827 = x2683;
x2825 = x2826.plus_i(x2827);

// Dealloc(X2091)
JCudaTensor x2828;
x2828 = x2683;
x2828.free();

// val X2131 = X2116 * d_ReLU()(X251)/d_X250
JCudaTensor x2829;
JCudaTensor x2830, x2831;
x2830 = x2825;
x2831 = x274;
x2829 = x268.backward(x2830, x2831);

// Dealloc(X251)
JCudaTensor x2832;
x2832 = x274;
x2832.free();

// val X2135 = X2131.copy * d_ReLU()(X240)/d_X239
JCudaTensor x2833;
JCudaTensor x2834, x2835;
x2834 = x2829;
x2834 = x2834.clone();
x2835 = x266;
x2833 = x268.backward(x2834, x2835);

// Dealloc(X240)
JCudaTensor x2836;
x2836 = x266;
x2836.free();

// val X2147 = X2131.copy * d_ReLU()(X249)/d_X248
JCudaTensor x2837;
JCudaTensor x2838, x2839;
x2838 = x2829;
x2838 = x2838.clone();
x2839 = x269;
x2837 = x268.backward(x2838, x2839);

// Dealloc(X2131)
JCudaTensor x2840;
x2840 = x2829;
x2840.free();

// Dealloc(X249)
JCudaTensor x2841;
x2841 = x269;
x2841.free();

// val X63615 = X2135 * d_BatchNorm(3a1_bn)(X238,3a1_bn_scale)/d_3a1_bn_scale
JCudaTensor x2842;
JCudaTensor x2843, x2844, x2845;
x2843 = x2833;
x2844 = x212;
x2845 = x230;
JCudaTensor[] x2846 = x232.backward(x2843,x2844,x2845);
x2842 = x2846[1];

// val X2136 = X2135 * d_BatchNorm(3a1_bn)(X238,3a1_bn_scale)/d_X238
JCudaTensor x2847;
x2847 = x2846[0];

// val X78255 = X2147 * d_BatchNorm(3a2_c_bn)(X247,3a2_c_bn_scale)/d_3a2_c_bn_scale
JCudaTensor x2851;
JCudaTensor x2852, x2853, x2854;
x2852 = x2837;
x2853 = x252;
x2854 = x263;
JCudaTensor[] x2855 = x265.backward(x2852,x2853,x2854);
x2851 = x2855[1];

// val X2148 = X2147 * d_BatchNorm(3a2_c_bn)(X247,3a2_c_bn_scale)/d_X247
JCudaTensor x2856;
x2856 = x2855[0];

// val X76641 = X2147 * d_BatchNorm(3a2_c_bn)(X247,3a2_c_bn_scale)/d_3a2_c_bn_bias
JCudaTensor x2860;
x2860 = x2855[2];

// Dealloc(X247)
JCudaTensor x2864;
x2864 = x252;
x2864.free();

// val X62010 = X2135 * d_BatchNorm(3a1_bn)(X238,3a1_bn_scale)/d_3a1_bn_bias
JCudaTensor x2865;
x2865 = x2846[2];

// Dealloc(X238)
JCudaTensor x2869;
x2869 = x212;
x2869.free();

// V_3a1_bn_bias <~~ X62010
float x2871, x2872;
x2871 = lrn_rate_1;
x2872 = momentum;
JCudaTensor x2873;
x2873 = x2865;
x2870.update(x2873, x2871, x2872);

// Dealloc(X62010)
JCudaTensor x2874;
x2874 = x2865;
x2874.free();

// V_3a1_cv_W <~~ X2136 * d_Convolv(2,0)(X237)/d_3a1_cv_W
float x2876, x2877;
x2876 = lrn_rate_1;
x2877 = momentum;
JCudaTensor x2878, x2879;
x2878 = x2847;
x2879 = x203;
x218.backward_filter(x2878, x2879, x2875, x2876, x2877);

// V_3a2_c_cv_W <~~ X2148 * d_Convolv(1,0)(X246)/d_3a2_c_cv_W
float x2881, x2882;
x2881 = lrn_rate_1;
x2882 = momentum;
JCudaTensor x2883, x2884;
x2883 = x2856;
x2884 = x250;
x258.backward_filter(x2883, x2884, x2880, x2881, x2882);

// val X2137 = X2136 * d_Convolv(2,0)(3a1_cv_W)/d_X237
JCudaTensor x2885;
JCudaTensor x2886, x2887;
x2886 = x2847;
x2887 = x216;
x2885 = x218.backward_data(x2886, x2887);

// Dealloc(X2136)
JCudaTensor x2888;
x2888 = x2847;
x2888.free();

// V_3a1_bn_scale <~~ X63615
float x2890, x2891;
x2890 = lrn_rate_1;
x2891 = momentum;
JCudaTensor x2892;
x2892 = x2842;
x2889.update(x2892, x2890, x2891);

// Dealloc(X63615)
JCudaTensor x2893;
x2893 = x2842;
x2893.free();

// V_3a2_c_bn_bias <~~ X76641
float x2895, x2896;
x2895 = lrn_rate_1;
x2896 = momentum;
JCudaTensor x2897;
x2897 = x2860;
x2894.update(x2897, x2895, x2896);

// Dealloc(X76641)
JCudaTensor x2898;
x2898 = x2860;
x2898.free();

// V_3a2_c_bn_scale <~~ X78255
float x2900, x2901;
x2900 = lrn_rate_1;
x2901 = momentum;
JCudaTensor x2902;
x2902 = x2851;
x2899.update(x2902, x2900, x2901);

// Dealloc(X78255)
JCudaTensor x2903;
x2903 = x2851;
x2903.free();

// val X2149 = X2148 * d_Convolv(1,0)(3a2_c_cv_W)/d_X246
JCudaTensor x2904;
JCudaTensor x2905, x2906;
x2905 = x2856;
x2906 = x256;
x2904 = x258.backward_data(x2905, x2906);

// Dealloc(X2148)
JCudaTensor x2907;
x2907 = x2856;
x2907.free();

// 3a1_bn_scale <~~ V_3a1_bn_scale
float x2908, x2909;
x2908 = 1;
x2909 = decay_1;
JCudaTensor x2910;
x2910 = x2889;
x230.update(x2910, x2908, x2909);

// 3a2_c_bn_scale <~~ V_3a2_c_bn_scale
float x2911, x2912;
x2911 = 1;
x2912 = decay_1;
JCudaTensor x2913;
x2913 = x2899;
x263.update(x2913, x2911, x2912);

// 3a1_cv_W <~~ V_3a1_cv_W
float x2914, x2915;
x2914 = 1;
x2915 = decay_1;
JCudaTensor x2916;
x2916 = x2875;
x216.update(x2916, x2914, x2915);

// 3a2_c_bn_bias <~~ V_3a2_c_bn_bias
float x2917, x2918;
x2917 = 1;
x2918 = decay_1;
JCudaTensor x2919;
x2919 = x2894;
x264.update(x2919, x2917, x2918);

// 3a2_c_cv_W <~~ V_3a2_c_cv_W
float x2920, x2921;
x2920 = 1;
x2921 = decay_1;
JCudaTensor x2922;
x2922 = x2880;
x256.update(x2922, x2920, x2921);

// 3a1_bn_bias <~~ V_3a1_bn_bias
float x2923, x2924;
x2923 = 1;
x2924 = decay_1;
JCudaTensor x2925;
x2925 = x2870;
x231.update(x2925, x2923, x2924);

// val X2153 = X2149 * d_ReLU()(X246)/d_X245
JCudaTensor x2926;
JCudaTensor x2927, x2928;
x2927 = x2904;
x2928 = x250;
x2926 = x235.backward(x2927, x2928);

// Dealloc(X246)
JCudaTensor x2929;
x2929 = x250;
x2929.free();

// val X2154 = X2153 * d_BatchNorm(3a2_b_bn)(X244,3a2_b_bn_scale)/d_X244
JCudaTensor x2930;
JCudaTensor x2931, x2932, x2933;
x2931 = x2926;
x2932 = x236;
x2933 = x247;
JCudaTensor[] x2934 = x249.backward(x2931,x2932,x2933);
x2930 = x2934[0];

// val X71771 = X2153 * d_BatchNorm(3a2_b_bn)(X244,3a2_b_bn_scale)/d_3a2_b_bn_bias
JCudaTensor x2935;
x2935 = x2934[2];

// val X73398 = X2153 * d_BatchNorm(3a2_b_bn)(X244,3a2_b_bn_scale)/d_3a2_b_bn_scale
JCudaTensor x2939;
x2939 = x2934[1];

// Dealloc(X244)
JCudaTensor x2943;
x2943 = x236;
x2943.free();

// V_3a2_b_bn_bias <~~ X71771
float x2945, x2946;
x2945 = lrn_rate_1;
x2946 = momentum;
JCudaTensor x2947;
x2947 = x2935;
x2944.update(x2947, x2945, x2946);

// Dealloc(X71771)
JCudaTensor x2948;
x2948 = x2935;
x2948.free();

// V_3a2_b_bn_scale <~~ X73398
float x2950, x2951;
x2950 = lrn_rate_1;
x2951 = momentum;
JCudaTensor x2952;
x2952 = x2939;
x2949.update(x2952, x2950, x2951);

// Dealloc(X73398)
JCudaTensor x2953;
x2953 = x2939;
x2953.free();

// val X2155 = X2154 * d_Convolv(1,1)(3a2_b_cv_W)/d_X243
JCudaTensor x2954;
JCudaTensor x2955, x2956;
x2955 = x2930;
x2956 = x240;
x2954 = x242.backward_data(x2955, x2956);

// V_3a2_b_cv_W <~~ X2154 * d_Convolv(1,1)(X243)/d_3a2_b_cv_W
float x2958, x2959;
x2958 = lrn_rate_1;
x2959 = momentum;
JCudaTensor x2960, x2961;
x2960 = x2930;
x2961 = x233;
x242.backward_filter(x2960, x2961, x2957, x2958, x2959);

// Dealloc(X2154)
JCudaTensor x2962;
x2962 = x2930;
x2962.free();

// 3a2_b_bn_bias <~~ V_3a2_b_bn_bias
float x2963, x2964;
x2963 = 1;
x2964 = decay_1;
JCudaTensor x2965;
x2965 = x2944;
x248.update(x2965, x2963, x2964);

// 3a2_b_bn_scale <~~ V_3a2_b_bn_scale
float x2966, x2967;
x2966 = 1;
x2967 = decay_1;
JCudaTensor x2968;
x2968 = x2949;
x247.update(x2968, x2966, x2967);

// 3a2_b_cv_W <~~ V_3a2_b_cv_W
float x2969, x2970;
x2969 = 1;
x2970 = decay_1;
JCudaTensor x2971;
x2971 = x2957;
x240.update(x2971, x2969, x2970);

// val X2159 = X2155 * d_ReLU()(X243)/d_X242
JCudaTensor x2972;
JCudaTensor x2973, x2974;
x2973 = x2954;
x2974 = x233;
x2972 = x235.backward(x2973, x2974);

// Dealloc(X243)
JCudaTensor x2975;
x2975 = x233;
x2975.free();

// val X2160 = X2159 * d_BatchNorm(3a2_a_bn)(X241,3a2_a_bn_scale)/d_X241
JCudaTensor x2976;
JCudaTensor x2977, x2978, x2979;
x2977 = x2972;
x2978 = x205;
x2979 = x223;
JCudaTensor[] x2980 = x225.backward(x2977,x2978,x2979);
x2976 = x2980[0];

// val X68502 = X2159 * d_BatchNorm(3a2_a_bn)(X241,3a2_a_bn_scale)/d_3a2_a_bn_scale
JCudaTensor x2981;
x2981 = x2980[1];

// val X66862 = X2159 * d_BatchNorm(3a2_a_bn)(X241,3a2_a_bn_scale)/d_3a2_a_bn_bias
JCudaTensor x2985;
x2985 = x2980[2];

// Dealloc(X241)
JCudaTensor x2989;
x2989 = x205;
x2989.free();

// val X2162 = (X2137 + X2160 * d_Convolv(2,0)(3a2_a_cv_W)/d_X237)
JCudaTensor x2990;
JCudaTensor x2991;
x2991 = x2885;
JCudaTensor x2992, x2993;
x2992 = x2976;
x2993 = x209;
x2990 = x211.backward_data(x2992,x2993, x2991);

// V_3a2_a_cv_W <~~ X2160 * d_Convolv(2,0)(X237)/d_3a2_a_cv_W
float x2995, x2996;
x2995 = lrn_rate_1;
x2996 = momentum;
JCudaTensor x2997, x2998;
x2997 = x2976;
x2998 = x203;
x211.backward_filter(x2997, x2998, x2994, x2995, x2996);

// Dealloc(X2160)
JCudaTensor x2999;
x2999 = x2976;
x2999.free();

// V_3a2_a_bn_scale <~~ X68502
float x3001, x3002;
x3001 = lrn_rate_1;
x3002 = momentum;
JCudaTensor x3003;
x3003 = x2981;
x3000.update(x3003, x3001, x3002);

// Dealloc(X68502)
JCudaTensor x3004;
x3004 = x2981;
x3004.free();

// V_3a2_a_bn_bias <~~ X66862
float x3006, x3007;
x3006 = lrn_rate_1;
x3007 = momentum;
JCudaTensor x3008;
x3008 = x2985;
x3005.update(x3008, x3006, x3007);

// Dealloc(X66862)
JCudaTensor x3009;
x3009 = x2985;
x3009.free();

// 3a2_a_cv_W <~~ V_3a2_a_cv_W
float x3010, x3011;
x3010 = 1;
x3011 = decay_1;
JCudaTensor x3012;
x3012 = x2994;
x209.update(x3012, x3010, x3011);

// 3a2_a_bn_scale <~~ V_3a2_a_bn_scale
float x3013, x3014;
x3013 = 1;
x3014 = decay_1;
JCudaTensor x3015;
x3015 = x3000;
x223.update(x3015, x3013, x3014);

// 3a2_a_bn_bias <~~ V_3a2_a_bn_bias
float x3016, x3017;
x3016 = 1;
x3017 = decay_1;
JCudaTensor x3018;
x3018 = x3005;
x224.update(x3018, x3016, x3017);

// val X2199 = X2162 * d_ReLU()(X237)/d_X236
JCudaTensor x3019;
JCudaTensor x3020, x3021;
x3020 = x2990;
x3021 = x203;
x3019 = x96.backward(x3020, x3021);

// Dealloc(X237)
JCudaTensor x3022;
x3022 = x203;
x3022.free();

// val X2209 = X2199.copy * d_ReLU()(X235)/d_X234
JCudaTensor x3023;
JCudaTensor x3024, x3025;
x3024 = x3019;
x3024 = x3024.clone();
x3025 = x198;
x3023 = x96.backward(x3024, x3025);

// Dealloc(X235)
JCudaTensor x3026;
x3026 = x198;
x3026.free();

// val X58681 = X2209 * d_BatchNorm(2c_c_bn)(X233,2c_c_bn_scale)/d_2c_c_bn_scale
JCudaTensor x3027;
JCudaTensor x3028, x3029, x3030;
x3028 = x3023;
x3029 = x185;
x3030 = x195;
JCudaTensor[] x3031 = x197.backward(x3028,x3029,x3030);
x3027 = x3031[1];

// val X2210 = X2209 * d_BatchNorm(2c_c_bn)(X233,2c_c_bn_scale)/d_X233
JCudaTensor x3032;
x3032 = x3031[0];

// val X56959 = X2209 * d_BatchNorm(2c_c_bn)(X233,2c_c_bn_scale)/d_2c_c_bn_bias
JCudaTensor x3036;
x3036 = x3031[2];

// Dealloc(X233)
JCudaTensor x3040;
x3040 = x185;
x3040.free();

// val X2211 = X2210 * d_Convolv(1,0)(2c_c_cv_W)/d_X232
JCudaTensor x3041;
JCudaTensor x3042, x3043;
x3042 = x3032;
x3043 = x189;
x3041 = x40.backward_data(x3042, x3043);

// V_2c_c_bn_bias <~~ X56959
float x3045, x3046;
x3045 = lrn_rate_1;
x3046 = momentum;
JCudaTensor x3047;
x3047 = x3036;
x3044.update(x3047, x3045, x3046);

// Dealloc(X56959)
JCudaTensor x3048;
x3048 = x3036;
x3048.free();

// V_2c_c_bn_scale <~~ X58681
float x3050, x3051;
x3050 = lrn_rate_1;
x3051 = momentum;
JCudaTensor x3052;
x3052 = x3027;
x3049.update(x3052, x3050, x3051);

// Dealloc(X58681)
JCudaTensor x3053;
x3053 = x3027;
x3053.free();

// V_2c_c_cv_W <~~ X2210 * d_Convolv(1,0)(X232)/d_2c_c_cv_W
float x3055, x3056;
x3055 = lrn_rate_1;
x3056 = momentum;
JCudaTensor x3057, x3058;
x3057 = x3032;
x3058 = x183;
x40.backward_filter(x3057, x3058, x3054, x3055, x3056);

// Dealloc(X2210)
JCudaTensor x3059;
x3059 = x3032;
x3059.free();

// 2c_c_bn_bias <~~ V_2c_c_bn_bias
float x3060, x3061;
x3060 = 1;
x3061 = decay_1;
JCudaTensor x3062;
x3062 = x3044;
x196.update(x3062, x3060, x3061);

// 2c_c_bn_scale <~~ V_2c_c_bn_scale
float x3063, x3064;
x3063 = 1;
x3064 = decay_1;
JCudaTensor x3065;
x3065 = x3049;
x195.update(x3065, x3063, x3064);

// 2c_c_cv_W <~~ V_2c_c_cv_W
float x3066, x3067;
x3066 = 1;
x3067 = decay_1;
JCudaTensor x3068;
x3068 = x3054;
x189.update(x3068, x3066, x3067);

// val X2215 = X2211 * d_ReLU()(X232)/d_X231
JCudaTensor x3069;
JCudaTensor x3070, x3071;
x3070 = x3041;
x3071 = x183;
x3069 = x64.backward(x3070, x3071);

// Dealloc(X232)
JCudaTensor x3072;
x3072 = x183;
x3072.free();

// val X2216 = X2215 * d_BatchNorm(2c_b_bn)(X230,2c_b_bn_scale)/d_X230
JCudaTensor x3073;
JCudaTensor x3074, x3075, x3076;
x3074 = x3069;
x3075 = x170;
x3076 = x180;
JCudaTensor[] x3077 = x182.backward(x3074,x3075,x3076);
x3073 = x3077[0];

// val X53500 = X2215 * d_BatchNorm(2c_b_bn)(X230,2c_b_bn_scale)/d_2c_b_bn_scale
JCudaTensor x3078;
x3078 = x3077[1];

// val X51765 = X2215 * d_BatchNorm(2c_b_bn)(X230,2c_b_bn_scale)/d_2c_b_bn_bias
JCudaTensor x3082;
x3082 = x3077[2];

// Dealloc(X230)
JCudaTensor x3086;
x3086 = x170;
x3086.free();

// val X2217 = X2216 * d_Convolv(1,1)(2c_b_cv_W)/d_X229
JCudaTensor x3087;
JCudaTensor x3088, x3089;
x3088 = x3073;
x3089 = x174;
x3087 = x71.backward_data(x3088, x3089);

// V_2c_b_cv_W <~~ X2216 * d_Convolv(1,1)(X229)/d_2c_b_cv_W
float x3091, x3092;
x3091 = lrn_rate_1;
x3092 = momentum;
JCudaTensor x3093, x3094;
x3093 = x3073;
x3094 = x168;
x71.backward_filter(x3093, x3094, x3090, x3091, x3092);

// Dealloc(X2216)
JCudaTensor x3095;
x3095 = x3073;
x3095.free();

// V_2c_b_bn_bias <~~ X51765
float x3097, x3098;
x3097 = lrn_rate_1;
x3098 = momentum;
JCudaTensor x3099;
x3099 = x3082;
x3096.update(x3099, x3097, x3098);

// Dealloc(X51765)
JCudaTensor x3100;
x3100 = x3082;
x3100.free();

// V_2c_b_bn_scale <~~ X53500
float x3102, x3103;
x3102 = lrn_rate_1;
x3103 = momentum;
JCudaTensor x3104;
x3104 = x3078;
x3101.update(x3104, x3102, x3103);

// Dealloc(X53500)
JCudaTensor x3105;
x3105 = x3078;
x3105.free();

// 2c_b_cv_W <~~ V_2c_b_cv_W
float x3106, x3107;
x3106 = 1;
x3107 = decay_1;
JCudaTensor x3108;
x3108 = x3090;
x174.update(x3108, x3106, x3107);

// 2c_b_bn_bias <~~ V_2c_b_bn_bias
float x3109, x3110;
x3109 = 1;
x3110 = decay_1;
JCudaTensor x3111;
x3111 = x3096;
x181.update(x3111, x3109, x3110);

// 2c_b_bn_scale <~~ V_2c_b_bn_scale
float x3112, x3113;
x3112 = 1;
x3113 = decay_1;
JCudaTensor x3114;
x3114 = x3101;
x180.update(x3114, x3112, x3113);

// val X2221 = X2217 * d_ReLU()(X229)/d_X228
JCudaTensor x3115;
JCudaTensor x3116, x3117;
x3116 = x3087;
x3117 = x168;
x3115 = x64.backward(x3116, x3117);

// Dealloc(X229)
JCudaTensor x3118;
x3118 = x168;
x3118.free();

// val X2222 = X2221 * d_BatchNorm(2c_a_bn)(X227,2c_a_bn_scale)/d_X227
JCudaTensor x3119;
JCudaTensor x3120, x3121, x3122;
x3120 = x3115;
x3121 = x155;
x3122 = x165;
JCudaTensor[] x3123 = x167.backward(x3120,x3121,x3122);
x3119 = x3123[0];

// val X48280 = X2221 * d_BatchNorm(2c_a_bn)(X227,2c_a_bn_scale)/d_2c_a_bn_scale
JCudaTensor x3124;
x3124 = x3123[1];

// val X46532 = X2221 * d_BatchNorm(2c_a_bn)(X227,2c_a_bn_scale)/d_2c_a_bn_bias
JCudaTensor x3128;
x3128 = x3123[2];

// Dealloc(X227)
JCudaTensor x3132;
x3132 = x155;
x3132.free();

// V_2c_a_bn_scale <~~ X48280
float x3134, x3135;
x3134 = lrn_rate_1;
x3135 = momentum;
JCudaTensor x3136;
x3136 = x3124;
x3133.update(x3136, x3134, x3135);

// Dealloc(X48280)
JCudaTensor x3137;
x3137 = x3124;
x3137.free();

// val X2223 = X2222 * d_Convolv(1,0)(2c_a_cv_W)/d_X226
JCudaTensor x3138;
JCudaTensor x3139, x3140;
x3139 = x3119;
x3140 = x159;
x3138 = x110.backward_data(x3139, x3140);

// V_2c_a_cv_W <~~ X2222 * d_Convolv(1,0)(X226)/d_2c_a_cv_W
float x3142, x3143;
x3142 = lrn_rate_1;
x3143 = momentum;
JCudaTensor x3144, x3145;
x3144 = x3119;
x3145 = x153;
x110.backward_filter(x3144, x3145, x3141, x3142, x3143);

// Dealloc(X2222)
JCudaTensor x3146;
x3146 = x3119;
x3146.free();

// V_2c_a_bn_bias <~~ X46532
float x3148, x3149;
x3148 = lrn_rate_1;
x3149 = momentum;
JCudaTensor x3150;
x3150 = x3128;
x3147.update(x3150, x3148, x3149);

// Dealloc(X46532)
JCudaTensor x3151;
x3151 = x3128;
x3151.free();

// 2c_a_bn_scale <~~ V_2c_a_bn_scale
float x3152, x3153;
x3152 = 1;
x3153 = decay_1;
JCudaTensor x3154;
x3154 = x3133;
x165.update(x3154, x3152, x3153);

// 2c_a_cv_W <~~ V_2c_a_cv_W
float x3155, x3156;
x3155 = 1;
x3156 = decay_1;
JCudaTensor x3157;
x3157 = x3141;
x159.update(x3157, x3155, x3156);

// 2c_a_bn_bias <~~ V_2c_a_bn_bias
float x3158, x3159;
x3158 = 1;
x3159 = decay_1;
JCudaTensor x3160;
x3160 = x3147;
x166.update(x3160, x3158, x3159);

// val X2224 = (X2223 + X2199)
JCudaTensor x3161;
JCudaTensor x3162, x3163;
x3162 = x3138;
x3163 = x3019;
x3161 = x3162.plus_i(x3163);

// Dealloc(X2199)
JCudaTensor x3164;
x3164 = x3019;
x3164.free();

// val X2236 = X2224 * d_ReLU()(X226)/d_X225
JCudaTensor x3165;
JCudaTensor x3166, x3167;
x3166 = x3161;
x3167 = x153;
x3165 = x96.backward(x3166, x3167);

// Dealloc(X226)
JCudaTensor x3168;
x3168 = x153;
x3168.free();

// val X2246 = X2236.copy * d_ReLU()(X224)/d_X223
JCudaTensor x3169;
JCudaTensor x3170, x3171;
x3170 = x3165;
x3170 = x3170.clone();
x3171 = x148;
x3169 = x96.backward(x3170, x3171);

// Dealloc(X224)
JCudaTensor x3172;
x3172 = x148;
x3172.free();

// val X2247 = X2246 * d_BatchNorm(2b_c_bn)(X222,2b_c_bn_scale)/d_X222
JCudaTensor x3173;
JCudaTensor x3174, x3175, x3176;
x3174 = x3169;
x3175 = x135;
x3176 = x145;
JCudaTensor[] x3177 = x147.backward(x3174,x3175,x3176);
x3173 = x3177[0];

// val X41204 = X2246 * d_BatchNorm(2b_c_bn)(X222,2b_c_bn_scale)/d_2b_c_bn_bias
JCudaTensor x3178;
x3178 = x3177[2];

// val X42993 = X2246 * d_BatchNorm(2b_c_bn)(X222,2b_c_bn_scale)/d_2b_c_bn_scale
JCudaTensor x3182;
x3182 = x3177[1];

// Dealloc(X222)
JCudaTensor x3186;
x3186 = x135;
x3186.free();

// V_2b_c_bn_scale <~~ X42993
float x3188, x3189;
x3188 = lrn_rate_1;
x3189 = momentum;
JCudaTensor x3190;
x3190 = x3182;
x3187.update(x3190, x3188, x3189);

// Dealloc(X42993)
JCudaTensor x3191;
x3191 = x3182;
x3191.free();

// val X2248 = X2247 * d_Convolv(1,0)(2b_c_cv_W)/d_X221
JCudaTensor x3192;
JCudaTensor x3193, x3194;
x3193 = x3173;
x3194 = x139;
x3192 = x40.backward_data(x3193, x3194);

// V_2b_c_bn_bias <~~ X41204
float x3196, x3197;
x3196 = lrn_rate_1;
x3197 = momentum;
JCudaTensor x3198;
x3198 = x3178;
x3195.update(x3198, x3196, x3197);

// Dealloc(X41204)
JCudaTensor x3199;
x3199 = x3178;
x3199.free();

// V_2b_c_cv_W <~~ X2247 * d_Convolv(1,0)(X221)/d_2b_c_cv_W
float x3201, x3202;
x3201 = lrn_rate_1;
x3202 = momentum;
JCudaTensor x3203, x3204;
x3203 = x3173;
x3204 = x133;
x40.backward_filter(x3203, x3204, x3200, x3201, x3202);

// Dealloc(X2247)
JCudaTensor x3205;
x3205 = x3173;
x3205.free();

// 2b_c_bn_scale <~~ V_2b_c_bn_scale
float x3206, x3207;
x3206 = 1;
x3207 = decay_1;
JCudaTensor x3208;
x3208 = x3187;
x145.update(x3208, x3206, x3207);

// 2b_c_bn_bias <~~ V_2b_c_bn_bias
float x3209, x3210;
x3209 = 1;
x3210 = decay_1;
JCudaTensor x3211;
x3211 = x3195;
x146.update(x3211, x3209, x3210);

// 2b_c_cv_W <~~ V_2b_c_cv_W
float x3212, x3213;
x3212 = 1;
x3213 = decay_1;
JCudaTensor x3214;
x3214 = x3200;
x139.update(x3214, x3212, x3213);

// val X2252 = X2248 * d_ReLU()(X221)/d_X220
JCudaTensor x3215;
JCudaTensor x3216, x3217;
x3216 = x3192;
x3217 = x133;
x3215 = x64.backward(x3216, x3217);

// Dealloc(X221)
JCudaTensor x3218;
x3218 = x133;
x3218.free();

// val X35809 = X2252 * d_BatchNorm(2b_b_bn)(X219,2b_b_bn_scale)/d_2b_b_bn_bias
JCudaTensor x3219;
JCudaTensor x3220, x3221, x3222;
x3220 = x3215;
x3221 = x120;
x3222 = x130;
JCudaTensor[] x3223 = x132.backward(x3220,x3221,x3222);
x3219 = x3223[2];

// val X37611 = X2252 * d_BatchNorm(2b_b_bn)(X219,2b_b_bn_scale)/d_2b_b_bn_scale
JCudaTensor x3224;
x3224 = x3223[1];

// val X2253 = X2252 * d_BatchNorm(2b_b_bn)(X219,2b_b_bn_scale)/d_X219
JCudaTensor x3228;
x3228 = x3223[0];

// Dealloc(X219)
JCudaTensor x3232;
x3232 = x120;
x3232.free();

// V_2b_b_bn_scale <~~ X37611
float x3234, x3235;
x3234 = lrn_rate_1;
x3235 = momentum;
JCudaTensor x3236;
x3236 = x3224;
x3233.update(x3236, x3234, x3235);

// Dealloc(X37611)
JCudaTensor x3237;
x3237 = x3224;
x3237.free();

// val X2254 = X2253 * d_Convolv(1,1)(2b_b_cv_W)/d_X218
JCudaTensor x3238;
JCudaTensor x3239, x3240;
x3239 = x3228;
x3240 = x124;
x3238 = x71.backward_data(x3239, x3240);

// V_2b_b_bn_bias <~~ X35809
float x3242, x3243;
x3242 = lrn_rate_1;
x3243 = momentum;
JCudaTensor x3244;
x3244 = x3219;
x3241.update(x3244, x3242, x3243);

// Dealloc(X35809)
JCudaTensor x3245;
x3245 = x3219;
x3245.free();

// V_2b_b_cv_W <~~ X2253 * d_Convolv(1,1)(X218)/d_2b_b_cv_W
float x3247, x3248;
x3247 = lrn_rate_1;
x3248 = momentum;
JCudaTensor x3249, x3250;
x3249 = x3228;
x3250 = x118;
x71.backward_filter(x3249, x3250, x3246, x3247, x3248);

// Dealloc(X2253)
JCudaTensor x3251;
x3251 = x3228;
x3251.free();

// 2b_b_bn_scale <~~ V_2b_b_bn_scale
float x3252, x3253;
x3252 = 1;
x3253 = decay_1;
JCudaTensor x3254;
x3254 = x3233;
x130.update(x3254, x3252, x3253);

// 2b_b_bn_bias <~~ V_2b_b_bn_bias
float x3255, x3256;
x3255 = 1;
x3256 = decay_1;
JCudaTensor x3257;
x3257 = x3241;
x131.update(x3257, x3255, x3256);

// 2b_b_cv_W <~~ V_2b_b_cv_W
float x3258, x3259;
x3258 = 1;
x3259 = decay_1;
JCudaTensor x3260;
x3260 = x3246;
x124.update(x3260, x3258, x3259);

// val X2258 = X2254 * d_ReLU()(X218)/d_X217
JCudaTensor x3261;
JCudaTensor x3262, x3263;
x3262 = x3238;
x3263 = x118;
x3261 = x64.backward(x3262, x3263);

// Dealloc(X218)
JCudaTensor x3264;
x3264 = x118;
x3264.free();

// val X30375 = X2258 * d_BatchNorm(2b_a_bn)(X216,2b_a_bn_scale)/d_2b_a_bn_bias
JCudaTensor x3265;
JCudaTensor x3266, x3267, x3268;
x3266 = x3261;
x3267 = x104;
x3268 = x115;
JCudaTensor[] x3269 = x117.backward(x3266,x3267,x3268);
x3265 = x3269[2];

// val X2259 = X2258 * d_BatchNorm(2b_a_bn)(X216,2b_a_bn_scale)/d_X216
JCudaTensor x3270;
x3270 = x3269[0];

// val X32190 = X2258 * d_BatchNorm(2b_a_bn)(X216,2b_a_bn_scale)/d_2b_a_bn_scale
JCudaTensor x3274;
x3274 = x3269[1];

// Dealloc(X216)
JCudaTensor x3278;
x3278 = x104;
x3278.free();

// V_2b_a_bn_scale <~~ X32190
float x3280, x3281;
x3280 = lrn_rate_1;
x3281 = momentum;
JCudaTensor x3282;
x3282 = x3274;
x3279.update(x3282, x3280, x3281);

// Dealloc(X32190)
JCudaTensor x3283;
x3283 = x3274;
x3283.free();

// val X2260 = X2259 * d_Convolv(1,0)(2b_a_cv_W)/d_X215
JCudaTensor x3284;
JCudaTensor x3285, x3286;
x3285 = x3270;
x3286 = x108;
x3284 = x110.backward_data(x3285, x3286);

// V_2b_a_cv_W <~~ X2259 * d_Convolv(1,0)(X215)/d_2b_a_cv_W
float x3288, x3289;
x3288 = lrn_rate_1;
x3289 = momentum;
JCudaTensor x3290, x3291;
x3290 = x3270;
x3291 = x102;
x110.backward_filter(x3290, x3291, x3287, x3288, x3289);

// Dealloc(X2259)
JCudaTensor x3292;
x3292 = x3270;
x3292.free();

// V_2b_a_bn_bias <~~ X30375
float x3294, x3295;
x3294 = lrn_rate_1;
x3295 = momentum;
JCudaTensor x3296;
x3296 = x3265;
x3293.update(x3296, x3294, x3295);

// Dealloc(X30375)
JCudaTensor x3297;
x3297 = x3265;
x3297.free();

// 2b_a_bn_scale <~~ V_2b_a_bn_scale
float x3298, x3299;
x3298 = 1;
x3299 = decay_1;
JCudaTensor x3300;
x3300 = x3279;
x115.update(x3300, x3298, x3299);

// 2b_a_cv_W <~~ V_2b_a_cv_W
float x3301, x3302;
x3301 = 1;
x3302 = decay_1;
JCudaTensor x3303;
x3303 = x3287;
x108.update(x3303, x3301, x3302);

// 2b_a_bn_bias <~~ V_2b_a_bn_bias
float x3304, x3305;
x3304 = 1;
x3305 = decay_1;
JCudaTensor x3306;
x3306 = x3293;
x116.update(x3306, x3304, x3305);

// val X2261 = (X2260 + X2236)
JCudaTensor x3307;
JCudaTensor x3308, x3309;
x3308 = x3284;
x3309 = x3165;
x3307 = x3308.plus_i(x3309);

// Dealloc(X2236)
JCudaTensor x3310;
x3310 = x3165;
x3310.free();

// val X2276 = X2261 * d_ReLU()(X215)/d_X214
JCudaTensor x3311;
JCudaTensor x3312, x3313;
x3312 = x3307;
x3313 = x102;
x3311 = x96.backward(x3312, x3313);

// Dealloc(X215)
JCudaTensor x3314;
x3314 = x102;
x3314.free();

// val X2292 = X2276.copy * d_ReLU()(X213)/d_X212
JCudaTensor x3315;
JCudaTensor x3316, x3317;
x3316 = x3311;
x3316 = x3316.clone();
x3317 = x97;
x3315 = x96.backward(x3316, x3317);

// Dealloc(X213)
JCudaTensor x3318;
x3318 = x97;
x3318.free();

// val X2280 = X2276.copy * d_ReLU()(X204)/d_X203
JCudaTensor x3319;
JCudaTensor x3320, x3321;
x3320 = x3311;
x3320 = x3320.clone();
x3321 = x94;
x3319 = x96.backward(x3320, x3321);

// Dealloc(X2276)
JCudaTensor x3322;
x3322 = x3311;
x3322.free();

// Dealloc(X204)
JCudaTensor x3323;
x3323 = x94;
x3323.free();

// val X2293 = X2292 * d_BatchNorm(2a2_c_bn)(X211,2a2_c_bn_scale)/d_X211
JCudaTensor x3324;
JCudaTensor x3325, x3326, x3327;
x3325 = x3315;
x3326 = x81;
x3327 = x91;
JCudaTensor[] x3328 = x93.backward(x3325,x3326,x3327);
x3324 = x3328[0];

// val X2281 = X2280 * d_BatchNorm(2a1_bn)(X202,2a1_bn_scale)/d_X202
JCudaTensor x3329;
JCudaTensor x3330, x3331, x3332;
x3330 = x3319;
x3331 = x34;
x3332 = x52;
JCudaTensor[] x3333 = x54.backward(x3330,x3331,x3332);
x3329 = x3333[0];

// val X9854 = X2280 * d_BatchNorm(2a1_bn)(X202,2a1_bn_scale)/d_2a1_bn_scale
JCudaTensor x3334;
x3334 = x3333[1];

// val X26699 = X2292 * d_BatchNorm(2a2_c_bn)(X211,2a2_c_bn_scale)/d_2a2_c_bn_scale
JCudaTensor x3338;
x3338 = x3328[1];

// val X24840 = X2292 * d_BatchNorm(2a2_c_bn)(X211,2a2_c_bn_scale)/d_2a2_c_bn_bias
JCudaTensor x3342;
x3342 = x3328[2];

// Dealloc(X211)
JCudaTensor x3346;
x3346 = x81;
x3346.free();

// val X8004 = X2280 * d_BatchNorm(2a1_bn)(X202,2a1_bn_scale)/d_2a1_bn_bias
JCudaTensor x3347;
x3347 = x3333[2];

// Dealloc(X202)
JCudaTensor x3351;
x3351 = x34;
x3351.free();

// V_2a2_c_cv_W <~~ X2293 * d_Convolv(1,0)(X210)/d_2a2_c_cv_W
float x3353, x3354;
x3353 = lrn_rate_1;
x3354 = momentum;
JCudaTensor x3355, x3356;
x3355 = x3324;
x3356 = x79;
x40.backward_filter(x3355, x3356, x3352, x3353, x3354);

// V_2a2_c_bn_bias <~~ X24840
float x3358, x3359;
x3358 = lrn_rate_1;
x3359 = momentum;
JCudaTensor x3360;
x3360 = x3342;
x3357.update(x3360, x3358, x3359);

// Dealloc(X24840)
JCudaTensor x3361;
x3361 = x3342;
x3361.free();

// V_2a2_c_bn_scale <~~ X26699
float x3363, x3364;
x3363 = lrn_rate_1;
x3364 = momentum;
JCudaTensor x3365;
x3365 = x3338;
x3362.update(x3365, x3363, x3364);

// Dealloc(X26699)
JCudaTensor x3366;
x3366 = x3338;
x3366.free();

// V_2a1_cv_W <~~ X2281 * d_Convolv(1,0)(X201)/d_2a1_cv_W
float x3368, x3369;
x3368 = lrn_rate_1;
x3369 = momentum;
JCudaTensor x3370, x3371;
x3370 = x3329;
x3371 = x31;
x40.backward_filter(x3370, x3371, x3367, x3368, x3369);

// V_2a1_bn_bias <~~ X8004
float x3373, x3374;
x3373 = lrn_rate_1;
x3374 = momentum;
JCudaTensor x3375;
x3375 = x3347;
x3372.update(x3375, x3373, x3374);

// Dealloc(X8004)
JCudaTensor x3376;
x3376 = x3347;
x3376.free();

// V_2a1_bn_scale <~~ X9854
float x3378, x3379;
x3378 = lrn_rate_1;
x3379 = momentum;
JCudaTensor x3380;
x3380 = x3334;
x3377.update(x3380, x3378, x3379);

// Dealloc(X9854)
JCudaTensor x3381;
x3381 = x3334;
x3381.free();

// val X2282 = X2281 * d_Convolv(1,0)(2a1_cv_W)/d_X201
JCudaTensor x3382;
JCudaTensor x3383, x3384;
x3383 = x3329;
x3384 = x38;
x3382 = x40.backward_data(x3383, x3384);

// Dealloc(X2281)
JCudaTensor x3385;
x3385 = x3329;
x3385.free();

// val X2294 = X2293 * d_Convolv(1,0)(2a2_c_cv_W)/d_X210
JCudaTensor x3386;
JCudaTensor x3387, x3388;
x3387 = x3324;
x3388 = x85;
x3386 = x40.backward_data(x3387, x3388);

// Dealloc(X2293)
JCudaTensor x3389;
x3389 = x3324;
x3389.free();

// 2a1_bn_scale <~~ V_2a1_bn_scale
float x3390, x3391;
x3390 = 1;
x3391 = decay_1;
JCudaTensor x3392;
x3392 = x3377;
x52.update(x3392, x3390, x3391);

// 2a2_c_cv_W <~~ V_2a2_c_cv_W
float x3393, x3394;
x3393 = 1;
x3394 = decay_1;
JCudaTensor x3395;
x3395 = x3352;
x85.update(x3395, x3393, x3394);

// 2a2_c_bn_bias <~~ V_2a2_c_bn_bias
float x3396, x3397;
x3396 = 1;
x3397 = decay_1;
JCudaTensor x3398;
x3398 = x3357;
x92.update(x3398, x3396, x3397);

// 2a1_bn_bias <~~ V_2a1_bn_bias
float x3399, x3400;
x3399 = 1;
x3400 = decay_1;
JCudaTensor x3401;
x3401 = x3372;
x53.update(x3401, x3399, x3400);

// 2a2_c_bn_scale <~~ V_2a2_c_bn_scale
float x3402, x3403;
x3402 = 1;
x3403 = decay_1;
JCudaTensor x3404;
x3404 = x3362;
x91.update(x3404, x3402, x3403);

// 2a1_cv_W <~~ V_2a1_cv_W
float x3405, x3406;
x3405 = 1;
x3406 = decay_1;
JCudaTensor x3407;
x3407 = x3367;
x38.update(x3407, x3405, x3406);

// val X2298 = X2294 * d_ReLU()(X210)/d_X209
JCudaTensor x3408;
JCudaTensor x3409, x3410;
x3409 = x3386;
x3410 = x79;
x3408 = x64.backward(x3409, x3410);

// Dealloc(X210)
JCudaTensor x3411;
x3411 = x79;
x3411.free();

// val X21107 = X2298 * d_BatchNorm(2a2_b_bn)(X208,2a2_b_bn_scale)/d_2a2_b_bn_scale
JCudaTensor x3412;
JCudaTensor x3413, x3414, x3415;
x3413 = x3408;
x3414 = x65;
x3415 = x76;
JCudaTensor[] x3416 = x78.backward(x3413,x3414,x3415);
x3412 = x3416[1];

// val X2299 = X2298 * d_BatchNorm(2a2_b_bn)(X208,2a2_b_bn_scale)/d_X208
JCudaTensor x3417;
x3417 = x3416[0];

// val X19235 = X2298 * d_BatchNorm(2a2_b_bn)(X208,2a2_b_bn_scale)/d_2a2_b_bn_bias
JCudaTensor x3421;
x3421 = x3416[2];

// Dealloc(X208)
JCudaTensor x3425;
x3425 = x65;
x3425.free();

// V_2a2_b_bn_bias <~~ X19235
float x3427, x3428;
x3427 = lrn_rate_1;
x3428 = momentum;
JCudaTensor x3429;
x3429 = x3421;
x3426.update(x3429, x3427, x3428);

// Dealloc(X19235)
JCudaTensor x3430;
x3430 = x3421;
x3430.free();

// val X2300 = X2299 * d_Convolv(1,1)(2a2_b_cv_W)/d_X207
JCudaTensor x3431;
JCudaTensor x3432, x3433;
x3432 = x3417;
x3433 = x69;
x3431 = x71.backward_data(x3432, x3433);

// V_2a2_b_cv_W <~~ X2299 * d_Convolv(1,1)(X207)/d_2a2_b_cv_W
float x3435, x3436;
x3435 = lrn_rate_1;
x3436 = momentum;
JCudaTensor x3437, x3438;
x3437 = x3417;
x3438 = x62;
x71.backward_filter(x3437, x3438, x3434, x3435, x3436);

// Dealloc(X2299)
JCudaTensor x3439;
x3439 = x3417;
x3439.free();

// V_2a2_b_bn_scale <~~ X21107
float x3441, x3442;
x3441 = lrn_rate_1;
x3442 = momentum;
JCudaTensor x3443;
x3443 = x3412;
x3440.update(x3443, x3441, x3442);

// Dealloc(X21107)
JCudaTensor x3444;
x3444 = x3412;
x3444.free();

// 2a2_b_bn_bias <~~ V_2a2_b_bn_bias
float x3445, x3446;
x3445 = 1;
x3446 = decay_1;
JCudaTensor x3447;
x3447 = x3426;
x77.update(x3447, x3445, x3446);

// 2a2_b_cv_W <~~ V_2a2_b_cv_W
float x3448, x3449;
x3448 = 1;
x3449 = decay_1;
JCudaTensor x3450;
x3450 = x3434;
x69.update(x3450, x3448, x3449);

// 2a2_b_bn_scale <~~ V_2a2_b_bn_scale
float x3451, x3452;
x3451 = 1;
x3452 = decay_1;
JCudaTensor x3453;
x3453 = x3440;
x76.update(x3453, x3451, x3452);

// val X2304 = X2300 * d_ReLU()(X207)/d_X206
JCudaTensor x3454;
JCudaTensor x3455, x3456;
x3455 = x3431;
x3456 = x62;
x3454 = x64.backward(x3455, x3456);

// Dealloc(X207)
JCudaTensor x3457;
x3457 = x62;
x3457.free();

// val X13591 = X2304 * d_BatchNorm(2a2_a_bn)(X205,2a2_a_bn_scale)/d_2a2_a_bn_bias
JCudaTensor x3458;
JCudaTensor x3459, x3460, x3461;
x3459 = x3454;
x3460 = x41;
x3461 = x59;
JCudaTensor[] x3462 = x61.backward(x3459,x3460,x3461);
x3458 = x3462[2];

// val X2305 = X2304 * d_BatchNorm(2a2_a_bn)(X205,2a2_a_bn_scale)/d_X205
JCudaTensor x3463;
x3463 = x3462[0];

// val X15476 = X2304 * d_BatchNorm(2a2_a_bn)(X205,2a2_a_bn_scale)/d_2a2_a_bn_scale
JCudaTensor x3467;
x3467 = x3462[1];

// Dealloc(X205)
JCudaTensor x3471;
x3471 = x41;
x3471.free();

// val X2307 = (X2282 + X2305 * d_Convolv(1,0)(2a2_a_cv_W)/d_X201)
JCudaTensor x3472;
JCudaTensor x3473;
x3473 = x3382;
JCudaTensor x3474, x3475;
x3474 = x3463;
x3475 = x45;
x3472 = x47.backward_data(x3474,x3475, x3473);

// V_2a2_a_cv_W <~~ X2305 * d_Convolv(1,0)(X201)/d_2a2_a_cv_W
float x3477, x3478;
x3477 = lrn_rate_1;
x3478 = momentum;
JCudaTensor x3479, x3480;
x3479 = x3463;
x3480 = x31;
x47.backward_filter(x3479, x3480, x3476, x3477, x3478);

// Dealloc(X2305)
JCudaTensor x3481;
x3481 = x3463;
x3481.free();

// V_2a2_a_bn_scale <~~ X15476
float x3483, x3484;
x3483 = lrn_rate_1;
x3484 = momentum;
JCudaTensor x3485;
x3485 = x3467;
x3482.update(x3485, x3483, x3484);

// Dealloc(X15476)
JCudaTensor x3486;
x3486 = x3467;
x3486.free();

// V_2a2_a_bn_bias <~~ X13591
float x3488, x3489;
x3488 = lrn_rate_1;
x3489 = momentum;
JCudaTensor x3490;
x3490 = x3458;
x3487.update(x3490, x3488, x3489);

// Dealloc(X13591)
JCudaTensor x3491;
x3491 = x3458;
x3491.free();

// 2a2_a_cv_W <~~ V_2a2_a_cv_W
float x3492, x3493;
x3492 = 1;
x3493 = decay_1;
JCudaTensor x3494;
x3494 = x3476;
x45.update(x3494, x3492, x3493);

// 2a2_a_bn_scale <~~ V_2a2_a_bn_scale
float x3495, x3496;
x3495 = 1;
x3496 = decay_1;
JCudaTensor x3497;
x3497 = x3482;
x59.update(x3497, x3495, x3496);

// 2a2_a_bn_bias <~~ V_2a2_a_bn_bias
float x3498, x3499;
x3498 = 1;
x3499 = decay_1;
JCudaTensor x3500;
x3500 = x3487;
x60.update(x3500, x3498, x3499);

// val X2309 = X2307 * d_Pooling(3,2,0,true)(X201,X200)/d_X200
JCudaTensor x3501;
JCudaTensor x3502, x3503, x3504;
x3502 = x3472;
x3503 = x31;
x3504 = x28;
x3501 = x33.backward(x3502, x3503, x3504);

// Dealloc(X2307)
JCudaTensor x3505;
x3505 = x3472;
x3505.free();

// Dealloc(X201)
JCudaTensor x3506;
x3506 = x31;
x3506.free();

// val X2313 = X2309 * d_ReLU()(X200)/d_X199
JCudaTensor x3507;
JCudaTensor x3508, x3509;
x3508 = x3501;
x3509 = x28;
x3507 = x30.backward(x3508, x3509);

// Dealloc(X200)
JCudaTensor x3510;
x3510 = x28;
x3510.free();

// val X6153 = X2313 * d_BatchNorm(1_bn)(X198,1_bn_scale)/d_X198
JCudaTensor x3511;
JCudaTensor x3512, x3513, x3514;
x3512 = x3507;
x3513 = x14;
x3514 = x25;
JCudaTensor[] x3515 = x27.backward(x3512,x3513,x3514);
x3511 = x3515[0];

// val X4233 = X2313 * d_BatchNorm(1_bn)(X198,1_bn_scale)/d_1_bn_scale
JCudaTensor x3516;
x3516 = x3515[1];

// val X2314 = X2313 * d_BatchNorm(1_bn)(X198,1_bn_scale)/d_1_bn_bias
JCudaTensor x3520;
x3520 = x3515[2];

// Dealloc(X198)
JCudaTensor x3524;
x3524 = x14;
x3524.free();

// V_1_bn_bias <~~ X2314
float x3526, x3527;
x3526 = lrn_rate_1;
x3527 = momentum;
JCudaTensor x3528;
x3528 = x3520;
x3525.update(x3528, x3526, x3527);

// Dealloc(X2314)
JCudaTensor x3529;
x3529 = x3520;
x3529.free();

// V_1_bn_scale <~~ X4233
float x3531, x3532;
x3531 = lrn_rate_1;
x3532 = momentum;
JCudaTensor x3533;
x3533 = x3516;
x3530.update(x3533, x3531, x3532);

// Dealloc(X4233)
JCudaTensor x3534;
x3534 = x3516;
x3534.free();

// V_1_cv_W <~~ X6153 * d_Convolv(2,3)(X197)/d_1_cv_W
float x3536, x3537;
x3536 = lrn_rate_1;
x3537 = momentum;
JCudaTensor x3538, x3539;
x3538 = x3511;
x3539 = x7;
x20.backward_filter(x3538, x3539, x3535, x3536, x3537);

// Dealloc(X6153)
JCudaTensor x3540;
x3540 = x3511;
x3540.free();

// Dealloc(X197)
JCudaTensor x3541;
x3541 = x7;
x3541.free();

// 1_bn_bias <~~ V_1_bn_bias
float x3542, x3543;
x3542 = 1;
x3543 = decay_1;
JCudaTensor x3544;
x3544 = x3525;
x26.update(x3544, x3542, x3543);

// 1_bn_scale <~~ V_1_bn_scale
float x3545, x3546;
x3545 = 1;
x3546 = decay_1;
JCudaTensor x3547;
x3547 = x3530;
x25.update(x3547, x3545, x3546);

// 1_cv_W <~~ V_1_cv_W
float x3548, x3549;
x3548 = 1;
x3549 = decay_1;
JCudaTensor x3550;
x3550 = x3535;
x18.update(x3550, x3548, x3549);

}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X209116 = Cuda(X)
JCudaTensor x3551;
JTensorFloat x3552;
x3552 = x3;
x3551 = x3552.asJCudaTensor();

// val X209117 = Convolv(2,3)(X209116,1_cv_W,1_cv_B)
JCudaTensor x3553;
JCudaTensor x3554, x3555, x3556;
x3554 = x3551;
x3555 = x18;
x3556 = x19;
x3553 = x20.forward(x3554, x3555, x3556);

// Dealloc(X209116)
JCudaTensor x3557;
x3557 = x3551;
x3557.free();

// val X209118 = BatchNorm(1_bn)(X209117,1_bn_scale,1_bn_bias)
JCudaTensor x3558;
JCudaTensor x3559, x3560, x3561;
x3559 = x3553;
x3560 = x25;
x3561 = x26;
x3558 = x27.forward_inference(x3559, x3560, x3561);

// Dealloc(X209117)
JCudaTensor x3562;
x3562 = x3553;
x3562.free();

// val X209119 = ReLU()(X209118)
JCudaTensor x3563;
JCudaTensor x3564;
x3564 = x3558;
x3563 = x30.forward(x3564);

// val X209120 = Pooling(3,2,0,true)(X209119)
JCudaTensor x3565;
JCudaTensor x3566;
x3566 = x3563;
x3565 = x33.forward(x3566);

// Dealloc(X209119)
JCudaTensor x3567;
x3567 = x3563;
x3567.free();

// val X209124 = Convolv(1,0)(X209120,2a2_a_cv_W,2a2_a_cv_B)
JCudaTensor x3568;
JCudaTensor x3569, x3570, x3571;
x3569 = x3565;
x3570 = x45;
x3571 = x46;
x3568 = x47.forward(x3569, x3570, x3571);

// val X209121 = Convolv(1,0)(X209120,2a1_cv_W,2a1_cv_B)
JCudaTensor x3572;
JCudaTensor x3573, x3574, x3575;
x3573 = x3565;
x3574 = x38;
x3575 = x39;
x3572 = x40.forward(x3573, x3574, x3575);

// Dealloc(X209120)
JCudaTensor x3576;
x3576 = x3565;
x3576.free();

// val X209122 = BatchNorm(2a1_bn)(X209121,2a1_bn_scale,2a1_bn_bias)
JCudaTensor x3577;
JCudaTensor x3578, x3579, x3580;
x3578 = x3572;
x3579 = x52;
x3580 = x53;
x3577 = x54.forward_inference(x3578, x3579, x3580);

// Dealloc(X209121)
JCudaTensor x3581;
x3581 = x3572;
x3581.free();

// val X209125 = BatchNorm(2a2_a_bn)(X209124,2a2_a_bn_scale,2a2_a_bn_bias)
JCudaTensor x3582;
JCudaTensor x3583, x3584, x3585;
x3583 = x3568;
x3584 = x59;
x3585 = x60;
x3582 = x61.forward_inference(x3583, x3584, x3585);

// Dealloc(X209124)
JCudaTensor x3586;
x3586 = x3568;
x3586.free();

// val X209126 = ReLU()(X209125)
JCudaTensor x3587;
JCudaTensor x3588;
x3588 = x3582;
x3587 = x64.forward(x3588);

// val X209127 = Convolv(1,1)(X209126,2a2_b_cv_W,2a2_b_cv_B)
JCudaTensor x3589;
JCudaTensor x3590, x3591, x3592;
x3590 = x3587;
x3591 = x69;
x3592 = x70;
x3589 = x71.forward(x3590, x3591, x3592);

// Dealloc(X209126)
JCudaTensor x3593;
x3593 = x3587;
x3593.free();

// val X209128 = BatchNorm(2a2_b_bn)(X209127,2a2_b_bn_scale,2a2_b_bn_bias)
JCudaTensor x3594;
JCudaTensor x3595, x3596, x3597;
x3595 = x3589;
x3596 = x76;
x3597 = x77;
x3594 = x78.forward_inference(x3595, x3596, x3597);

// Dealloc(X209127)
JCudaTensor x3598;
x3598 = x3589;
x3598.free();

// val X209129 = ReLU()(X209128)
JCudaTensor x3599;
JCudaTensor x3600;
x3600 = x3594;
x3599 = x64.forward(x3600);

// val X209130 = Convolv(1,0)(X209129,2a2_c_cv_W,2a2_c_cv_B)
JCudaTensor x3601;
JCudaTensor x3602, x3603, x3604;
x3602 = x3599;
x3603 = x85;
x3604 = x86;
x3601 = x40.forward(x3602, x3603, x3604);

// Dealloc(X209129)
JCudaTensor x3605;
x3605 = x3599;
x3605.free();

// val X209131 = BatchNorm(2a2_c_bn)(X209130,2a2_c_bn_scale,2a2_c_bn_bias)
JCudaTensor x3606;
JCudaTensor x3607, x3608, x3609;
x3607 = x3601;
x3608 = x91;
x3609 = x92;
x3606 = x93.forward_inference(x3607, x3608, x3609);

// Dealloc(X209130)
JCudaTensor x3610;
x3610 = x3601;
x3610.free();

// val X209123 = ReLU()(X209122)
JCudaTensor x3611;
JCudaTensor x3612;
x3612 = x3577;
x3611 = x96.forward(x3612);

// val X209132 = ReLU()(X209131)
JCudaTensor x3613;
JCudaTensor x3614;
x3614 = x3606;
x3613 = x96.forward(x3614);

// val X209133 = (X209123 + X209132)
JCudaTensor x3615;
JCudaTensor x3616, x3617;
x3616 = x3611;
x3617 = x3613;
x3615 = x3616.plus_i(x3617);

// Dealloc(X209132)
JCudaTensor x3618;
x3618 = x3613;
x3618.free();

// val X209134 = ReLU()(X209133)
JCudaTensor x3619;
JCudaTensor x3620;
x3620 = x3615;
x3619 = x96.forward(x3620);

// val X209135 = Convolv(1,0)(X209134,2b_a_cv_W,2b_a_cv_B)
JCudaTensor x3621;
JCudaTensor x3622, x3623, x3624;
x3622 = x3619;
x3623 = x108;
x3624 = x109;
x3621 = x110.forward(x3622, x3623, x3624);

// val X209136 = BatchNorm(2b_a_bn)(X209135,2b_a_bn_scale,2b_a_bn_bias)
JCudaTensor x3625;
JCudaTensor x3626, x3627, x3628;
x3626 = x3621;
x3627 = x115;
x3628 = x116;
x3625 = x117.forward_inference(x3626, x3627, x3628);

// Dealloc(X209135)
JCudaTensor x3629;
x3629 = x3621;
x3629.free();

// val X209137 = ReLU()(X209136)
JCudaTensor x3630;
JCudaTensor x3631;
x3631 = x3625;
x3630 = x64.forward(x3631);

// val X209138 = Convolv(1,1)(X209137,2b_b_cv_W,2b_b_cv_B)
JCudaTensor x3632;
JCudaTensor x3633, x3634, x3635;
x3633 = x3630;
x3634 = x124;
x3635 = x125;
x3632 = x71.forward(x3633, x3634, x3635);

// Dealloc(X209137)
JCudaTensor x3636;
x3636 = x3630;
x3636.free();

// val X209139 = BatchNorm(2b_b_bn)(X209138,2b_b_bn_scale,2b_b_bn_bias)
JCudaTensor x3637;
JCudaTensor x3638, x3639, x3640;
x3638 = x3632;
x3639 = x130;
x3640 = x131;
x3637 = x132.forward_inference(x3638, x3639, x3640);

// Dealloc(X209138)
JCudaTensor x3641;
x3641 = x3632;
x3641.free();

// val X209140 = ReLU()(X209139)
JCudaTensor x3642;
JCudaTensor x3643;
x3643 = x3637;
x3642 = x64.forward(x3643);

// val X209141 = Convolv(1,0)(X209140,2b_c_cv_W,2b_c_cv_B)
JCudaTensor x3644;
JCudaTensor x3645, x3646, x3647;
x3645 = x3642;
x3646 = x139;
x3647 = x140;
x3644 = x40.forward(x3645, x3646, x3647);

// Dealloc(X209140)
JCudaTensor x3648;
x3648 = x3642;
x3648.free();

// val X209142 = BatchNorm(2b_c_bn)(X209141,2b_c_bn_scale,2b_c_bn_bias)
JCudaTensor x3649;
JCudaTensor x3650, x3651, x3652;
x3650 = x3644;
x3651 = x145;
x3652 = x146;
x3649 = x147.forward_inference(x3650, x3651, x3652);

// Dealloc(X209141)
JCudaTensor x3653;
x3653 = x3644;
x3653.free();

// val X209143 = ReLU()(X209142)
JCudaTensor x3654;
JCudaTensor x3655;
x3655 = x3649;
x3654 = x96.forward(x3655);

// val X209144 = (X209143 + X209134)
JCudaTensor x3656;
JCudaTensor x3657, x3658;
x3657 = x3654;
x3658 = x3619;
x3656 = x3657.plus_i(x3658);

// Dealloc(X209134)
JCudaTensor x3659;
x3659 = x3619;
x3659.free();

// val X209145 = ReLU()(X209144)
JCudaTensor x3660;
JCudaTensor x3661;
x3661 = x3656;
x3660 = x96.forward(x3661);

// val X209146 = Convolv(1,0)(X209145,2c_a_cv_W,2c_a_cv_B)
JCudaTensor x3662;
JCudaTensor x3663, x3664, x3665;
x3663 = x3660;
x3664 = x159;
x3665 = x160;
x3662 = x110.forward(x3663, x3664, x3665);

// val X209147 = BatchNorm(2c_a_bn)(X209146,2c_a_bn_scale,2c_a_bn_bias)
JCudaTensor x3666;
JCudaTensor x3667, x3668, x3669;
x3667 = x3662;
x3668 = x165;
x3669 = x166;
x3666 = x167.forward_inference(x3667, x3668, x3669);

// Dealloc(X209146)
JCudaTensor x3670;
x3670 = x3662;
x3670.free();

// val X209148 = ReLU()(X209147)
JCudaTensor x3671;
JCudaTensor x3672;
x3672 = x3666;
x3671 = x64.forward(x3672);

// val X209149 = Convolv(1,1)(X209148,2c_b_cv_W,2c_b_cv_B)
JCudaTensor x3673;
JCudaTensor x3674, x3675, x3676;
x3674 = x3671;
x3675 = x174;
x3676 = x175;
x3673 = x71.forward(x3674, x3675, x3676);

// Dealloc(X209148)
JCudaTensor x3677;
x3677 = x3671;
x3677.free();

// val X209150 = BatchNorm(2c_b_bn)(X209149,2c_b_bn_scale,2c_b_bn_bias)
JCudaTensor x3678;
JCudaTensor x3679, x3680, x3681;
x3679 = x3673;
x3680 = x180;
x3681 = x181;
x3678 = x182.forward_inference(x3679, x3680, x3681);

// Dealloc(X209149)
JCudaTensor x3682;
x3682 = x3673;
x3682.free();

// val X209151 = ReLU()(X209150)
JCudaTensor x3683;
JCudaTensor x3684;
x3684 = x3678;
x3683 = x64.forward(x3684);

// val X209152 = Convolv(1,0)(X209151,2c_c_cv_W,2c_c_cv_B)
JCudaTensor x3685;
JCudaTensor x3686, x3687, x3688;
x3686 = x3683;
x3687 = x189;
x3688 = x190;
x3685 = x40.forward(x3686, x3687, x3688);

// Dealloc(X209151)
JCudaTensor x3689;
x3689 = x3683;
x3689.free();

// val X209153 = BatchNorm(2c_c_bn)(X209152,2c_c_bn_scale,2c_c_bn_bias)
JCudaTensor x3690;
JCudaTensor x3691, x3692, x3693;
x3691 = x3685;
x3692 = x195;
x3693 = x196;
x3690 = x197.forward_inference(x3691, x3692, x3693);

// Dealloc(X209152)
JCudaTensor x3694;
x3694 = x3685;
x3694.free();

// val X209154 = ReLU()(X209153)
JCudaTensor x3695;
JCudaTensor x3696;
x3696 = x3690;
x3695 = x96.forward(x3696);

// val X209155 = (X209154 + X209145)
JCudaTensor x3697;
JCudaTensor x3698, x3699;
x3698 = x3695;
x3699 = x3660;
x3697 = x3698.plus_i(x3699);

// Dealloc(X209145)
JCudaTensor x3700;
x3700 = x3660;
x3700.free();

// val X209156 = ReLU()(X209155)
JCudaTensor x3701;
JCudaTensor x3702;
x3702 = x3697;
x3701 = x96.forward(x3702);

// val X209157 = Convolv(2,0)(X209156,3a1_cv_W,3a1_cv_B)
JCudaTensor x3703;
JCudaTensor x3704, x3705, x3706;
x3704 = x3701;
x3705 = x216;
x3706 = x217;
x3703 = x218.forward(x3704, x3705, x3706);

// val X209160 = Convolv(2,0)(X209156,3a2_a_cv_W,3a2_a_cv_B)
JCudaTensor x3707;
JCudaTensor x3708, x3709, x3710;
x3708 = x3701;
x3709 = x209;
x3710 = x210;
x3707 = x211.forward(x3708, x3709, x3710);

// Dealloc(X209156)
JCudaTensor x3711;
x3711 = x3701;
x3711.free();

// val X209161 = BatchNorm(3a2_a_bn)(X209160,3a2_a_bn_scale,3a2_a_bn_bias)
JCudaTensor x3712;
JCudaTensor x3713, x3714, x3715;
x3713 = x3707;
x3714 = x223;
x3715 = x224;
x3712 = x225.forward_inference(x3713, x3714, x3715);

// Dealloc(X209160)
JCudaTensor x3716;
x3716 = x3707;
x3716.free();

// val X209158 = BatchNorm(3a1_bn)(X209157,3a1_bn_scale,3a1_bn_bias)
JCudaTensor x3717;
JCudaTensor x3718, x3719, x3720;
x3718 = x3703;
x3719 = x230;
x3720 = x231;
x3717 = x232.forward_inference(x3718, x3719, x3720);

// Dealloc(X209157)
JCudaTensor x3721;
x3721 = x3703;
x3721.free();

// val X209162 = ReLU()(X209161)
JCudaTensor x3722;
JCudaTensor x3723;
x3723 = x3712;
x3722 = x235.forward(x3723);

// val X209163 = Convolv(1,1)(X209162,3a2_b_cv_W,3a2_b_cv_B)
JCudaTensor x3724;
JCudaTensor x3725, x3726, x3727;
x3725 = x3722;
x3726 = x240;
x3727 = x241;
x3724 = x242.forward(x3725, x3726, x3727);

// Dealloc(X209162)
JCudaTensor x3728;
x3728 = x3722;
x3728.free();

// val X209164 = BatchNorm(3a2_b_bn)(X209163,3a2_b_bn_scale,3a2_b_bn_bias)
JCudaTensor x3729;
JCudaTensor x3730, x3731, x3732;
x3730 = x3724;
x3731 = x247;
x3732 = x248;
x3729 = x249.forward_inference(x3730, x3731, x3732);

// Dealloc(X209163)
JCudaTensor x3733;
x3733 = x3724;
x3733.free();

// val X209165 = ReLU()(X209164)
JCudaTensor x3734;
JCudaTensor x3735;
x3735 = x3729;
x3734 = x235.forward(x3735);

// val X209166 = Convolv(1,0)(X209165,3a2_c_cv_W,3a2_c_cv_B)
JCudaTensor x3736;
JCudaTensor x3737, x3738, x3739;
x3737 = x3734;
x3738 = x256;
x3739 = x257;
x3736 = x258.forward(x3737, x3738, x3739);

// Dealloc(X209165)
JCudaTensor x3740;
x3740 = x3734;
x3740.free();

// val X209167 = BatchNorm(3a2_c_bn)(X209166,3a2_c_bn_scale,3a2_c_bn_bias)
JCudaTensor x3741;
JCudaTensor x3742, x3743, x3744;
x3742 = x3736;
x3743 = x263;
x3744 = x264;
x3741 = x265.forward_inference(x3742, x3743, x3744);

// Dealloc(X209166)
JCudaTensor x3745;
x3745 = x3736;
x3745.free();

// val X209159 = ReLU()(X209158)
JCudaTensor x3746;
JCudaTensor x3747;
x3747 = x3717;
x3746 = x268.forward(x3747);

// val X209168 = ReLU()(X209167)
JCudaTensor x3748;
JCudaTensor x3749;
x3749 = x3741;
x3748 = x268.forward(x3749);

// val X209169 = (X209159 + X209168)
JCudaTensor x3750;
JCudaTensor x3751, x3752;
x3751 = x3746;
x3752 = x3748;
x3750 = x3751.plus_i(x3752);

// Dealloc(X209168)
JCudaTensor x3753;
x3753 = x3748;
x3753.free();

// val X209170 = ReLU()(X209169)
JCudaTensor x3754;
JCudaTensor x3755;
x3755 = x3750;
x3754 = x268.forward(x3755);

// val X209171 = Convolv(1,0)(X209170,3b_a_cv_W,3b_a_cv_B)
JCudaTensor x3756;
JCudaTensor x3757, x3758, x3759;
x3757 = x3754;
x3758 = x280;
x3759 = x281;
x3756 = x282.forward(x3757, x3758, x3759);

// val X209172 = BatchNorm(3b_a_bn)(X209171,3b_a_bn_scale,3b_a_bn_bias)
JCudaTensor x3760;
JCudaTensor x3761, x3762, x3763;
x3761 = x3756;
x3762 = x287;
x3763 = x288;
x3760 = x289.forward_inference(x3761, x3762, x3763);

// Dealloc(X209171)
JCudaTensor x3764;
x3764 = x3756;
x3764.free();

// val X209173 = ReLU()(X209172)
JCudaTensor x3765;
JCudaTensor x3766;
x3766 = x3760;
x3765 = x235.forward(x3766);

// val X209174 = Convolv(1,1)(X209173,3b_b_cv_W,3b_b_cv_B)
JCudaTensor x3767;
JCudaTensor x3768, x3769, x3770;
x3768 = x3765;
x3769 = x296;
x3770 = x297;
x3767 = x242.forward(x3768, x3769, x3770);

// Dealloc(X209173)
JCudaTensor x3771;
x3771 = x3765;
x3771.free();

// val X209175 = BatchNorm(3b_b_bn)(X209174,3b_b_bn_scale,3b_b_bn_bias)
JCudaTensor x3772;
JCudaTensor x3773, x3774, x3775;
x3773 = x3767;
x3774 = x302;
x3775 = x303;
x3772 = x304.forward_inference(x3773, x3774, x3775);

// Dealloc(X209174)
JCudaTensor x3776;
x3776 = x3767;
x3776.free();

// val X209176 = ReLU()(X209175)
JCudaTensor x3777;
JCudaTensor x3778;
x3778 = x3772;
x3777 = x235.forward(x3778);

// val X209177 = Convolv(1,0)(X209176,3b_c_cv_W,3b_c_cv_B)
JCudaTensor x3779;
JCudaTensor x3780, x3781, x3782;
x3780 = x3777;
x3781 = x311;
x3782 = x312;
x3779 = x258.forward(x3780, x3781, x3782);

// Dealloc(X209176)
JCudaTensor x3783;
x3783 = x3777;
x3783.free();

// val X209178 = BatchNorm(3b_c_bn)(X209177,3b_c_bn_scale,3b_c_bn_bias)
JCudaTensor x3784;
JCudaTensor x3785, x3786, x3787;
x3785 = x3779;
x3786 = x317;
x3787 = x318;
x3784 = x319.forward_inference(x3785, x3786, x3787);

// Dealloc(X209177)
JCudaTensor x3788;
x3788 = x3779;
x3788.free();

// val X209179 = ReLU()(X209178)
JCudaTensor x3789;
JCudaTensor x3790;
x3790 = x3784;
x3789 = x268.forward(x3790);

// val X209180 = (X209179 + X209170)
JCudaTensor x3791;
JCudaTensor x3792, x3793;
x3792 = x3789;
x3793 = x3754;
x3791 = x3792.plus_i(x3793);

// Dealloc(X209170)
JCudaTensor x3794;
x3794 = x3754;
x3794.free();

// val X209181 = ReLU()(X209180)
JCudaTensor x3795;
JCudaTensor x3796;
x3796 = x3791;
x3795 = x268.forward(x3796);

// val X209182 = Convolv(1,0)(X209181,3c_a_cv_W,3c_a_cv_B)
JCudaTensor x3797;
JCudaTensor x3798, x3799, x3800;
x3798 = x3795;
x3799 = x331;
x3800 = x332;
x3797 = x282.forward(x3798, x3799, x3800);

// val X209183 = BatchNorm(3c_a_bn)(X209182,3c_a_bn_scale,3c_a_bn_bias)
JCudaTensor x3801;
JCudaTensor x3802, x3803, x3804;
x3802 = x3797;
x3803 = x337;
x3804 = x338;
x3801 = x339.forward_inference(x3802, x3803, x3804);

// Dealloc(X209182)
JCudaTensor x3805;
x3805 = x3797;
x3805.free();

// val X209184 = ReLU()(X209183)
JCudaTensor x3806;
JCudaTensor x3807;
x3807 = x3801;
x3806 = x235.forward(x3807);

// val X209185 = Convolv(1,1)(X209184,3c_b_cv_W,3c_b_cv_B)
JCudaTensor x3808;
JCudaTensor x3809, x3810, x3811;
x3809 = x3806;
x3810 = x346;
x3811 = x347;
x3808 = x242.forward(x3809, x3810, x3811);

// Dealloc(X209184)
JCudaTensor x3812;
x3812 = x3806;
x3812.free();

// val X209186 = BatchNorm(3c_b_bn)(X209185,3c_b_bn_scale,3c_b_bn_bias)
JCudaTensor x3813;
JCudaTensor x3814, x3815, x3816;
x3814 = x3808;
x3815 = x352;
x3816 = x353;
x3813 = x354.forward_inference(x3814, x3815, x3816);

// Dealloc(X209185)
JCudaTensor x3817;
x3817 = x3808;
x3817.free();

// val X209187 = ReLU()(X209186)
JCudaTensor x3818;
JCudaTensor x3819;
x3819 = x3813;
x3818 = x235.forward(x3819);

// val X209188 = Convolv(1,0)(X209187,3c_c_cv_W,3c_c_cv_B)
JCudaTensor x3820;
JCudaTensor x3821, x3822, x3823;
x3821 = x3818;
x3822 = x361;
x3823 = x362;
x3820 = x258.forward(x3821, x3822, x3823);

// Dealloc(X209187)
JCudaTensor x3824;
x3824 = x3818;
x3824.free();

// val X209189 = BatchNorm(3c_c_bn)(X209188,3c_c_bn_scale,3c_c_bn_bias)
JCudaTensor x3825;
JCudaTensor x3826, x3827, x3828;
x3826 = x3820;
x3827 = x367;
x3828 = x368;
x3825 = x369.forward_inference(x3826, x3827, x3828);

// Dealloc(X209188)
JCudaTensor x3829;
x3829 = x3820;
x3829.free();

// val X209190 = ReLU()(X209189)
JCudaTensor x3830;
JCudaTensor x3831;
x3831 = x3825;
x3830 = x268.forward(x3831);

// val X209191 = (X209190 + X209181)
JCudaTensor x3832;
JCudaTensor x3833, x3834;
x3833 = x3830;
x3834 = x3795;
x3832 = x3833.plus_i(x3834);

// Dealloc(X209181)
JCudaTensor x3835;
x3835 = x3795;
x3835.free();

// val X209192 = ReLU()(X209191)
JCudaTensor x3836;
JCudaTensor x3837;
x3837 = x3832;
x3836 = x268.forward(x3837);

// val X209193 = Convolv(1,0)(X209192,3d_a_cv_W,3d_a_cv_B)
JCudaTensor x3838;
JCudaTensor x3839, x3840, x3841;
x3839 = x3836;
x3840 = x381;
x3841 = x382;
x3838 = x282.forward(x3839, x3840, x3841);

// val X209194 = BatchNorm(3d_a_bn)(X209193,3d_a_bn_scale,3d_a_bn_bias)
JCudaTensor x3842;
JCudaTensor x3843, x3844, x3845;
x3843 = x3838;
x3844 = x387;
x3845 = x388;
x3842 = x389.forward_inference(x3843, x3844, x3845);

// Dealloc(X209193)
JCudaTensor x3846;
x3846 = x3838;
x3846.free();

// val X209195 = ReLU()(X209194)
JCudaTensor x3847;
JCudaTensor x3848;
x3848 = x3842;
x3847 = x235.forward(x3848);

// val X209196 = Convolv(1,1)(X209195,3d_b_cv_W,3d_b_cv_B)
JCudaTensor x3849;
JCudaTensor x3850, x3851, x3852;
x3850 = x3847;
x3851 = x396;
x3852 = x397;
x3849 = x242.forward(x3850, x3851, x3852);

// Dealloc(X209195)
JCudaTensor x3853;
x3853 = x3847;
x3853.free();

// val X209197 = BatchNorm(3d_b_bn)(X209196,3d_b_bn_scale,3d_b_bn_bias)
JCudaTensor x3854;
JCudaTensor x3855, x3856, x3857;
x3855 = x3849;
x3856 = x402;
x3857 = x403;
x3854 = x404.forward_inference(x3855, x3856, x3857);

// Dealloc(X209196)
JCudaTensor x3858;
x3858 = x3849;
x3858.free();

// val X209198 = ReLU()(X209197)
JCudaTensor x3859;
JCudaTensor x3860;
x3860 = x3854;
x3859 = x235.forward(x3860);

// val X209199 = Convolv(1,0)(X209198,3d_c_cv_W,3d_c_cv_B)
JCudaTensor x3861;
JCudaTensor x3862, x3863, x3864;
x3862 = x3859;
x3863 = x411;
x3864 = x412;
x3861 = x258.forward(x3862, x3863, x3864);

// Dealloc(X209198)
JCudaTensor x3865;
x3865 = x3859;
x3865.free();

// val X209200 = BatchNorm(3d_c_bn)(X209199,3d_c_bn_scale,3d_c_bn_bias)
JCudaTensor x3866;
JCudaTensor x3867, x3868, x3869;
x3867 = x3861;
x3868 = x417;
x3869 = x418;
x3866 = x419.forward_inference(x3867, x3868, x3869);

// Dealloc(X209199)
JCudaTensor x3870;
x3870 = x3861;
x3870.free();

// val X209201 = ReLU()(X209200)
JCudaTensor x3871;
JCudaTensor x3872;
x3872 = x3866;
x3871 = x268.forward(x3872);

// val X209202 = (X209201 + X209192)
JCudaTensor x3873;
JCudaTensor x3874, x3875;
x3874 = x3871;
x3875 = x3836;
x3873 = x3874.plus_i(x3875);

// Dealloc(X209192)
JCudaTensor x3876;
x3876 = x3836;
x3876.free();

// val X209203 = ReLU()(X209202)
JCudaTensor x3877;
JCudaTensor x3878;
x3878 = x3873;
x3877 = x268.forward(x3878);

// val X209204 = Convolv(2,0)(X209203,4a1_cv_W,4a1_cv_B)
JCudaTensor x3879;
JCudaTensor x3880, x3881, x3882;
x3880 = x3877;
x3881 = x438;
x3882 = x439;
x3879 = x440.forward(x3880, x3881, x3882);

// val X209207 = Convolv(2,0)(X209203,4a2_a_cv_W,4a2_a_cv_B)
JCudaTensor x3883;
JCudaTensor x3884, x3885, x3886;
x3884 = x3877;
x3885 = x431;
x3886 = x432;
x3883 = x433.forward(x3884, x3885, x3886);

// Dealloc(X209203)
JCudaTensor x3887;
x3887 = x3877;
x3887.free();

// val X209208 = BatchNorm(4a2_a_bn)(X209207,4a2_a_bn_scale,4a2_a_bn_bias)
JCudaTensor x3888;
JCudaTensor x3889, x3890, x3891;
x3889 = x3883;
x3890 = x452;
x3891 = x453;
x3888 = x454.forward_inference(x3889, x3890, x3891);

// Dealloc(X209207)
JCudaTensor x3892;
x3892 = x3883;
x3892.free();

// val X209205 = BatchNorm(4a1_bn)(X209204,4a1_bn_scale,4a1_bn_bias)
JCudaTensor x3893;
JCudaTensor x3894, x3895, x3896;
x3894 = x3879;
x3895 = x445;
x3896 = x446;
x3893 = x447.forward_inference(x3894, x3895, x3896);

// Dealloc(X209204)
JCudaTensor x3897;
x3897 = x3879;
x3897.free();

// val X209209 = ReLU()(X209208)
JCudaTensor x3898;
JCudaTensor x3899;
x3899 = x3888;
x3898 = x457.forward(x3899);

// val X209210 = Convolv(1,1)(X209209,4a2_b_cv_W,4a2_b_cv_B)
JCudaTensor x3900;
JCudaTensor x3901, x3902, x3903;
x3901 = x3898;
x3902 = x462;
x3903 = x463;
x3900 = x464.forward(x3901, x3902, x3903);

// Dealloc(X209209)
JCudaTensor x3904;
x3904 = x3898;
x3904.free();

// val X209211 = BatchNorm(4a2_b_bn)(X209210,4a2_b_bn_scale,4a2_b_bn_bias)
JCudaTensor x3905;
JCudaTensor x3906, x3907, x3908;
x3906 = x3900;
x3907 = x469;
x3908 = x470;
x3905 = x471.forward_inference(x3906, x3907, x3908);

// Dealloc(X209210)
JCudaTensor x3909;
x3909 = x3900;
x3909.free();

// val X209212 = ReLU()(X209211)
JCudaTensor x3910;
JCudaTensor x3911;
x3911 = x3905;
x3910 = x457.forward(x3911);

// val X209213 = Convolv(1,0)(X209212,4a2_c_cv_W,4a2_c_cv_B)
JCudaTensor x3912;
JCudaTensor x3913, x3914, x3915;
x3913 = x3910;
x3914 = x478;
x3915 = x479;
x3912 = x480.forward(x3913, x3914, x3915);

// Dealloc(X209212)
JCudaTensor x3916;
x3916 = x3910;
x3916.free();

// val X209214 = BatchNorm(4a2_c_bn)(X209213,4a2_c_bn_scale,4a2_c_bn_bias)
JCudaTensor x3917;
JCudaTensor x3918, x3919, x3920;
x3918 = x3912;
x3919 = x485;
x3920 = x486;
x3917 = x487.forward_inference(x3918, x3919, x3920);

// Dealloc(X209213)
JCudaTensor x3921;
x3921 = x3912;
x3921.free();

// val X209206 = ReLU()(X209205)
JCudaTensor x3922;
JCudaTensor x3923;
x3923 = x3893;
x3922 = x490.forward(x3923);

// val X209215 = ReLU()(X209214)
JCudaTensor x3924;
JCudaTensor x3925;
x3925 = x3917;
x3924 = x490.forward(x3925);

// val X209216 = (X209206 + X209215)
JCudaTensor x3926;
JCudaTensor x3927, x3928;
x3927 = x3922;
x3928 = x3924;
x3926 = x3927.plus_i(x3928);

// Dealloc(X209215)
JCudaTensor x3929;
x3929 = x3924;
x3929.free();

// val X209217 = ReLU()(X209216)
JCudaTensor x3930;
JCudaTensor x3931;
x3931 = x3926;
x3930 = x490.forward(x3931);

// val X209218 = Convolv(1,0)(X209217,4b_a_cv_W,4b_a_cv_B)
JCudaTensor x3932;
JCudaTensor x3933, x3934, x3935;
x3933 = x3930;
x3934 = x502;
x3935 = x503;
x3932 = x504.forward(x3933, x3934, x3935);

// val X209219 = BatchNorm(4b_a_bn)(X209218,4b_a_bn_scale,4b_a_bn_bias)
JCudaTensor x3936;
JCudaTensor x3937, x3938, x3939;
x3937 = x3932;
x3938 = x509;
x3939 = x510;
x3936 = x511.forward_inference(x3937, x3938, x3939);

// Dealloc(X209218)
JCudaTensor x3940;
x3940 = x3932;
x3940.free();

// val X209220 = ReLU()(X209219)
JCudaTensor x3941;
JCudaTensor x3942;
x3942 = x3936;
x3941 = x457.forward(x3942);

// val X209221 = Convolv(1,1)(X209220,4b_b_cv_W,4b_b_cv_B)
JCudaTensor x3943;
JCudaTensor x3944, x3945, x3946;
x3944 = x3941;
x3945 = x518;
x3946 = x519;
x3943 = x464.forward(x3944, x3945, x3946);

// Dealloc(X209220)
JCudaTensor x3947;
x3947 = x3941;
x3947.free();

// val X209222 = BatchNorm(4b_b_bn)(X209221,4b_b_bn_scale,4b_b_bn_bias)
JCudaTensor x3948;
JCudaTensor x3949, x3950, x3951;
x3949 = x3943;
x3950 = x524;
x3951 = x525;
x3948 = x526.forward_inference(x3949, x3950, x3951);

// Dealloc(X209221)
JCudaTensor x3952;
x3952 = x3943;
x3952.free();

// val X209223 = ReLU()(X209222)
JCudaTensor x3953;
JCudaTensor x3954;
x3954 = x3948;
x3953 = x457.forward(x3954);

// val X209224 = Convolv(1,0)(X209223,4b_c_cv_W,4b_c_cv_B)
JCudaTensor x3955;
JCudaTensor x3956, x3957, x3958;
x3956 = x3953;
x3957 = x533;
x3958 = x534;
x3955 = x480.forward(x3956, x3957, x3958);

// Dealloc(X209223)
JCudaTensor x3959;
x3959 = x3953;
x3959.free();

// val X209225 = BatchNorm(4b_c_bn)(X209224,4b_c_bn_scale,4b_c_bn_bias)
JCudaTensor x3960;
JCudaTensor x3961, x3962, x3963;
x3961 = x3955;
x3962 = x539;
x3963 = x540;
x3960 = x541.forward_inference(x3961, x3962, x3963);

// Dealloc(X209224)
JCudaTensor x3964;
x3964 = x3955;
x3964.free();

// val X209226 = ReLU()(X209225)
JCudaTensor x3965;
JCudaTensor x3966;
x3966 = x3960;
x3965 = x490.forward(x3966);

// val X209227 = (X209226 + X209217)
JCudaTensor x3967;
JCudaTensor x3968, x3969;
x3968 = x3965;
x3969 = x3930;
x3967 = x3968.plus_i(x3969);

// Dealloc(X209217)
JCudaTensor x3970;
x3970 = x3930;
x3970.free();

// val X209228 = ReLU()(X209227)
JCudaTensor x3971;
JCudaTensor x3972;
x3972 = x3967;
x3971 = x490.forward(x3972);

// val X209229 = Convolv(1,0)(X209228,4c_a_cv_W,4c_a_cv_B)
JCudaTensor x3973;
JCudaTensor x3974, x3975, x3976;
x3974 = x3971;
x3975 = x553;
x3976 = x554;
x3973 = x504.forward(x3974, x3975, x3976);

// val X209230 = BatchNorm(4c_a_bn)(X209229,4c_a_bn_scale,4c_a_bn_bias)
JCudaTensor x3977;
JCudaTensor x3978, x3979, x3980;
x3978 = x3973;
x3979 = x559;
x3980 = x560;
x3977 = x561.forward_inference(x3978, x3979, x3980);

// Dealloc(X209229)
JCudaTensor x3981;
x3981 = x3973;
x3981.free();

// val X209231 = ReLU()(X209230)
JCudaTensor x3982;
JCudaTensor x3983;
x3983 = x3977;
x3982 = x457.forward(x3983);

// val X209232 = Convolv(1,1)(X209231,4c_b_cv_W,4c_b_cv_B)
JCudaTensor x3984;
JCudaTensor x3985, x3986, x3987;
x3985 = x3982;
x3986 = x568;
x3987 = x569;
x3984 = x464.forward(x3985, x3986, x3987);

// Dealloc(X209231)
JCudaTensor x3988;
x3988 = x3982;
x3988.free();

// val X209233 = BatchNorm(4c_b_bn)(X209232,4c_b_bn_scale,4c_b_bn_bias)
JCudaTensor x3989;
JCudaTensor x3990, x3991, x3992;
x3990 = x3984;
x3991 = x574;
x3992 = x575;
x3989 = x576.forward_inference(x3990, x3991, x3992);

// Dealloc(X209232)
JCudaTensor x3993;
x3993 = x3984;
x3993.free();

// val X209234 = ReLU()(X209233)
JCudaTensor x3994;
JCudaTensor x3995;
x3995 = x3989;
x3994 = x457.forward(x3995);

// val X209235 = Convolv(1,0)(X209234,4c_c_cv_W,4c_c_cv_B)
JCudaTensor x3996;
JCudaTensor x3997, x3998, x3999;
x3997 = x3994;
x3998 = x583;
x3999 = x584;
x3996 = x480.forward(x3997, x3998, x3999);

// Dealloc(X209234)
JCudaTensor x4000;
x4000 = x3994;
x4000.free();

// val X209236 = BatchNorm(4c_c_bn)(X209235,4c_c_bn_scale,4c_c_bn_bias)
JCudaTensor x4001;
JCudaTensor x4002, x4003, x4004;
x4002 = x3996;
x4003 = x589;
x4004 = x590;
x4001 = x591.forward_inference(x4002, x4003, x4004);

// Dealloc(X209235)
JCudaTensor x4005;
x4005 = x3996;
x4005.free();

// val X209237 = ReLU()(X209236)
JCudaTensor x4006;
JCudaTensor x4007;
x4007 = x4001;
x4006 = x490.forward(x4007);

// val X209238 = (X209237 + X209228)
JCudaTensor x4008;
JCudaTensor x4009, x4010;
x4009 = x4006;
x4010 = x3971;
x4008 = x4009.plus_i(x4010);

// Dealloc(X209228)
JCudaTensor x4011;
x4011 = x3971;
x4011.free();

// val X209239 = ReLU()(X209238)
JCudaTensor x4012;
JCudaTensor x4013;
x4013 = x4008;
x4012 = x490.forward(x4013);

// val X209240 = Convolv(1,0)(X209239,4d_a_cv_W,4d_a_cv_B)
JCudaTensor x4014;
JCudaTensor x4015, x4016, x4017;
x4015 = x4012;
x4016 = x603;
x4017 = x604;
x4014 = x504.forward(x4015, x4016, x4017);

// val X209241 = BatchNorm(4d_a_bn)(X209240,4d_a_bn_scale,4d_a_bn_bias)
JCudaTensor x4018;
JCudaTensor x4019, x4020, x4021;
x4019 = x4014;
x4020 = x609;
x4021 = x610;
x4018 = x611.forward_inference(x4019, x4020, x4021);

// Dealloc(X209240)
JCudaTensor x4022;
x4022 = x4014;
x4022.free();

// val X209242 = ReLU()(X209241)
JCudaTensor x4023;
JCudaTensor x4024;
x4024 = x4018;
x4023 = x457.forward(x4024);

// val X209243 = Convolv(1,1)(X209242,4d_b_cv_W,4d_b_cv_B)
JCudaTensor x4025;
JCudaTensor x4026, x4027, x4028;
x4026 = x4023;
x4027 = x618;
x4028 = x619;
x4025 = x464.forward(x4026, x4027, x4028);

// Dealloc(X209242)
JCudaTensor x4029;
x4029 = x4023;
x4029.free();

// val X209244 = BatchNorm(4d_b_bn)(X209243,4d_b_bn_scale,4d_b_bn_bias)
JCudaTensor x4030;
JCudaTensor x4031, x4032, x4033;
x4031 = x4025;
x4032 = x624;
x4033 = x625;
x4030 = x626.forward_inference(x4031, x4032, x4033);

// Dealloc(X209243)
JCudaTensor x4034;
x4034 = x4025;
x4034.free();

// val X209245 = ReLU()(X209244)
JCudaTensor x4035;
JCudaTensor x4036;
x4036 = x4030;
x4035 = x457.forward(x4036);

// val X209246 = Convolv(1,0)(X209245,4d_c_cv_W,4d_c_cv_B)
JCudaTensor x4037;
JCudaTensor x4038, x4039, x4040;
x4038 = x4035;
x4039 = x633;
x4040 = x634;
x4037 = x480.forward(x4038, x4039, x4040);

// Dealloc(X209245)
JCudaTensor x4041;
x4041 = x4035;
x4041.free();

// val X209247 = BatchNorm(4d_c_bn)(X209246,4d_c_bn_scale,4d_c_bn_bias)
JCudaTensor x4042;
JCudaTensor x4043, x4044, x4045;
x4043 = x4037;
x4044 = x639;
x4045 = x640;
x4042 = x641.forward_inference(x4043, x4044, x4045);

// Dealloc(X209246)
JCudaTensor x4046;
x4046 = x4037;
x4046.free();

// val X209248 = ReLU()(X209247)
JCudaTensor x4047;
JCudaTensor x4048;
x4048 = x4042;
x4047 = x490.forward(x4048);

// val X209249 = (X209248 + X209239)
JCudaTensor x4049;
JCudaTensor x4050, x4051;
x4050 = x4047;
x4051 = x4012;
x4049 = x4050.plus_i(x4051);

// Dealloc(X209239)
JCudaTensor x4052;
x4052 = x4012;
x4052.free();

// val X209250 = ReLU()(X209249)
JCudaTensor x4053;
JCudaTensor x4054;
x4054 = x4049;
x4053 = x490.forward(x4054);

// val X209251 = Convolv(1,0)(X209250,4e_a_cv_W,4e_a_cv_B)
JCudaTensor x4055;
JCudaTensor x4056, x4057, x4058;
x4056 = x4053;
x4057 = x653;
x4058 = x654;
x4055 = x504.forward(x4056, x4057, x4058);

// val X209252 = BatchNorm(4e_a_bn)(X209251,4e_a_bn_scale,4e_a_bn_bias)
JCudaTensor x4059;
JCudaTensor x4060, x4061, x4062;
x4060 = x4055;
x4061 = x659;
x4062 = x660;
x4059 = x661.forward_inference(x4060, x4061, x4062);

// Dealloc(X209251)
JCudaTensor x4063;
x4063 = x4055;
x4063.free();

// val X209253 = ReLU()(X209252)
JCudaTensor x4064;
JCudaTensor x4065;
x4065 = x4059;
x4064 = x457.forward(x4065);

// val X209254 = Convolv(1,1)(X209253,4e_b_cv_W,4e_b_cv_B)
JCudaTensor x4066;
JCudaTensor x4067, x4068, x4069;
x4067 = x4064;
x4068 = x668;
x4069 = x669;
x4066 = x464.forward(x4067, x4068, x4069);

// Dealloc(X209253)
JCudaTensor x4070;
x4070 = x4064;
x4070.free();

// val X209255 = BatchNorm(4e_b_bn)(X209254,4e_b_bn_scale,4e_b_bn_bias)
JCudaTensor x4071;
JCudaTensor x4072, x4073, x4074;
x4072 = x4066;
x4073 = x674;
x4074 = x675;
x4071 = x676.forward_inference(x4072, x4073, x4074);

// Dealloc(X209254)
JCudaTensor x4075;
x4075 = x4066;
x4075.free();

// val X209256 = ReLU()(X209255)
JCudaTensor x4076;
JCudaTensor x4077;
x4077 = x4071;
x4076 = x457.forward(x4077);

// val X209257 = Convolv(1,0)(X209256,4e_c_cv_W,4e_c_cv_B)
JCudaTensor x4078;
JCudaTensor x4079, x4080, x4081;
x4079 = x4076;
x4080 = x683;
x4081 = x684;
x4078 = x480.forward(x4079, x4080, x4081);

// Dealloc(X209256)
JCudaTensor x4082;
x4082 = x4076;
x4082.free();

// val X209258 = BatchNorm(4e_c_bn)(X209257,4e_c_bn_scale,4e_c_bn_bias)
JCudaTensor x4083;
JCudaTensor x4084, x4085, x4086;
x4084 = x4078;
x4085 = x689;
x4086 = x690;
x4083 = x691.forward_inference(x4084, x4085, x4086);

// Dealloc(X209257)
JCudaTensor x4087;
x4087 = x4078;
x4087.free();

// val X209259 = ReLU()(X209258)
JCudaTensor x4088;
JCudaTensor x4089;
x4089 = x4083;
x4088 = x490.forward(x4089);

// val X209260 = (X209259 + X209250)
JCudaTensor x4090;
JCudaTensor x4091, x4092;
x4091 = x4088;
x4092 = x4053;
x4090 = x4091.plus_i(x4092);

// Dealloc(X209250)
JCudaTensor x4093;
x4093 = x4053;
x4093.free();

// val X209261 = ReLU()(X209260)
JCudaTensor x4094;
JCudaTensor x4095;
x4095 = x4090;
x4094 = x490.forward(x4095);

// val X209262 = Convolv(1,0)(X209261,4f_a_cv_W,4f_a_cv_B)
JCudaTensor x4096;
JCudaTensor x4097, x4098, x4099;
x4097 = x4094;
x4098 = x703;
x4099 = x704;
x4096 = x504.forward(x4097, x4098, x4099);

// val X209263 = BatchNorm(4f_a_bn)(X209262,4f_a_bn_scale,4f_a_bn_bias)
JCudaTensor x4100;
JCudaTensor x4101, x4102, x4103;
x4101 = x4096;
x4102 = x709;
x4103 = x710;
x4100 = x711.forward_inference(x4101, x4102, x4103);

// Dealloc(X209262)
JCudaTensor x4104;
x4104 = x4096;
x4104.free();

// val X209264 = ReLU()(X209263)
JCudaTensor x4105;
JCudaTensor x4106;
x4106 = x4100;
x4105 = x457.forward(x4106);

// val X209265 = Convolv(1,1)(X209264,4f_b_cv_W,4f_b_cv_B)
JCudaTensor x4107;
JCudaTensor x4108, x4109, x4110;
x4108 = x4105;
x4109 = x718;
x4110 = x719;
x4107 = x464.forward(x4108, x4109, x4110);

// Dealloc(X209264)
JCudaTensor x4111;
x4111 = x4105;
x4111.free();

// val X209266 = BatchNorm(4f_b_bn)(X209265,4f_b_bn_scale,4f_b_bn_bias)
JCudaTensor x4112;
JCudaTensor x4113, x4114, x4115;
x4113 = x4107;
x4114 = x724;
x4115 = x725;
x4112 = x726.forward_inference(x4113, x4114, x4115);

// Dealloc(X209265)
JCudaTensor x4116;
x4116 = x4107;
x4116.free();

// val X209267 = ReLU()(X209266)
JCudaTensor x4117;
JCudaTensor x4118;
x4118 = x4112;
x4117 = x457.forward(x4118);

// val X209268 = Convolv(1,0)(X209267,4f_c_cv_W,4f_c_cv_B)
JCudaTensor x4119;
JCudaTensor x4120, x4121, x4122;
x4120 = x4117;
x4121 = x733;
x4122 = x734;
x4119 = x480.forward(x4120, x4121, x4122);

// Dealloc(X209267)
JCudaTensor x4123;
x4123 = x4117;
x4123.free();

// val X209269 = BatchNorm(4f_c_bn)(X209268,4f_c_bn_scale,4f_c_bn_bias)
JCudaTensor x4124;
JCudaTensor x4125, x4126, x4127;
x4125 = x4119;
x4126 = x739;
x4127 = x740;
x4124 = x741.forward_inference(x4125, x4126, x4127);

// Dealloc(X209268)
JCudaTensor x4128;
x4128 = x4119;
x4128.free();

// val X209270 = ReLU()(X209269)
JCudaTensor x4129;
JCudaTensor x4130;
x4130 = x4124;
x4129 = x490.forward(x4130);

// val X209271 = (X209270 + X209261)
JCudaTensor x4131;
JCudaTensor x4132, x4133;
x4132 = x4129;
x4133 = x4094;
x4131 = x4132.plus_i(x4133);

// Dealloc(X209261)
JCudaTensor x4134;
x4134 = x4094;
x4134.free();

// val X209272 = ReLU()(X209271)
JCudaTensor x4135;
JCudaTensor x4136;
x4136 = x4131;
x4135 = x490.forward(x4136);

// val X209276 = Convolv(2,0)(X209272,5a2_a_cv_W,5a2_a_cv_B)
JCudaTensor x4137;
JCudaTensor x4138, x4139, x4140;
x4138 = x4135;
x4139 = x760;
x4140 = x761;
x4137 = x762.forward(x4138, x4139, x4140);

// val X209273 = Convolv(2,0)(X209272,5a1_cv_W,5a1_cv_B)
JCudaTensor x4141;
JCudaTensor x4142, x4143, x4144;
x4142 = x4135;
x4143 = x753;
x4144 = x754;
x4141 = x755.forward(x4142, x4143, x4144);

// Dealloc(X209272)
JCudaTensor x4145;
x4145 = x4135;
x4145.free();

// val X209277 = BatchNorm(5a2_a_bn)(X209276,5a2_a_bn_scale,5a2_a_bn_bias)
JCudaTensor x4146;
JCudaTensor x4147, x4148, x4149;
x4147 = x4137;
x4148 = x774;
x4149 = x775;
x4146 = x776.forward_inference(x4147, x4148, x4149);

// Dealloc(X209276)
JCudaTensor x4150;
x4150 = x4137;
x4150.free();

// val X209274 = BatchNorm(5a1_bn)(X209273,5a1_bn_scale,5a1_bn_bias)
JCudaTensor x4151;
JCudaTensor x4152, x4153, x4154;
x4152 = x4141;
x4153 = x767;
x4154 = x768;
x4151 = x769.forward_inference(x4152, x4153, x4154);

// Dealloc(X209273)
JCudaTensor x4155;
x4155 = x4141;
x4155.free();

// val X209278 = ReLU()(X209277)
JCudaTensor x4156;
JCudaTensor x4157;
x4157 = x4146;
x4156 = x779.forward(x4157);

// val X209279 = Convolv(1,1)(X209278,5a2_b_cv_W,5a2_b_cv_B)
JCudaTensor x4158;
JCudaTensor x4159, x4160, x4161;
x4159 = x4156;
x4160 = x784;
x4161 = x785;
x4158 = x786.forward(x4159, x4160, x4161);

// Dealloc(X209278)
JCudaTensor x4162;
x4162 = x4156;
x4162.free();

// val X209280 = BatchNorm(5a2_b_bn)(X209279,5a2_b_bn_scale,5a2_b_bn_bias)
JCudaTensor x4163;
JCudaTensor x4164, x4165, x4166;
x4164 = x4158;
x4165 = x791;
x4166 = x792;
x4163 = x793.forward_inference(x4164, x4165, x4166);

// Dealloc(X209279)
JCudaTensor x4167;
x4167 = x4158;
x4167.free();

// val X209281 = ReLU()(X209280)
JCudaTensor x4168;
JCudaTensor x4169;
x4169 = x4163;
x4168 = x779.forward(x4169);

// val X209282 = Convolv(1,0)(X209281,5a2_c_cv_W,5a2_c_cv_B)
JCudaTensor x4170;
JCudaTensor x4171, x4172, x4173;
x4171 = x4168;
x4172 = x800;
x4173 = x801;
x4170 = x802.forward(x4171, x4172, x4173);

// Dealloc(X209281)
JCudaTensor x4174;
x4174 = x4168;
x4174.free();

// val X209283 = BatchNorm(5a2_c_bn)(X209282,5a2_c_bn_scale,5a2_c_bn_bias)
JCudaTensor x4175;
JCudaTensor x4176, x4177, x4178;
x4176 = x4170;
x4177 = x807;
x4178 = x808;
x4175 = x809.forward_inference(x4176, x4177, x4178);

// Dealloc(X209282)
JCudaTensor x4179;
x4179 = x4170;
x4179.free();

// val X209275 = ReLU()(X209274)
JCudaTensor x4180;
JCudaTensor x4181;
x4181 = x4151;
x4180 = x812.forward(x4181);

// val X209284 = ReLU()(X209283)
JCudaTensor x4182;
JCudaTensor x4183;
x4183 = x4175;
x4182 = x812.forward(x4183);

// val X209285 = (X209275 + X209284)
JCudaTensor x4184;
JCudaTensor x4185, x4186;
x4185 = x4180;
x4186 = x4182;
x4184 = x4185.plus_i(x4186);

// Dealloc(X209284)
JCudaTensor x4187;
x4187 = x4182;
x4187.free();

// val X209286 = ReLU()(X209285)
JCudaTensor x4188;
JCudaTensor x4189;
x4189 = x4184;
x4188 = x812.forward(x4189);

// val X209287 = Convolv(1,0)(X209286,5b_a_cv_W,5b_a_cv_B)
JCudaTensor x4190;
JCudaTensor x4191, x4192, x4193;
x4191 = x4188;
x4192 = x824;
x4193 = x825;
x4190 = x826.forward(x4191, x4192, x4193);

// val X209288 = BatchNorm(5b_a_bn)(X209287,5b_a_bn_scale,5b_a_bn_bias)
JCudaTensor x4194;
JCudaTensor x4195, x4196, x4197;
x4195 = x4190;
x4196 = x831;
x4197 = x832;
x4194 = x833.forward_inference(x4195, x4196, x4197);

// Dealloc(X209287)
JCudaTensor x4198;
x4198 = x4190;
x4198.free();

// val X209289 = ReLU()(X209288)
JCudaTensor x4199;
JCudaTensor x4200;
x4200 = x4194;
x4199 = x779.forward(x4200);

// val X209290 = Convolv(1,1)(X209289,5b_b_cv_W,5b_b_cv_B)
JCudaTensor x4201;
JCudaTensor x4202, x4203, x4204;
x4202 = x4199;
x4203 = x840;
x4204 = x841;
x4201 = x786.forward(x4202, x4203, x4204);

// Dealloc(X209289)
JCudaTensor x4205;
x4205 = x4199;
x4205.free();

// val X209291 = BatchNorm(5b_b_bn)(X209290,5b_b_bn_scale,5b_b_bn_bias)
JCudaTensor x4206;
JCudaTensor x4207, x4208, x4209;
x4207 = x4201;
x4208 = x846;
x4209 = x847;
x4206 = x848.forward_inference(x4207, x4208, x4209);

// Dealloc(X209290)
JCudaTensor x4210;
x4210 = x4201;
x4210.free();

// val X209292 = ReLU()(X209291)
JCudaTensor x4211;
JCudaTensor x4212;
x4212 = x4206;
x4211 = x779.forward(x4212);

// val X209293 = Convolv(1,0)(X209292,5b_c_cv_W,5b_c_cv_B)
JCudaTensor x4213;
JCudaTensor x4214, x4215, x4216;
x4214 = x4211;
x4215 = x855;
x4216 = x856;
x4213 = x802.forward(x4214, x4215, x4216);

// Dealloc(X209292)
JCudaTensor x4217;
x4217 = x4211;
x4217.free();

// val X209294 = BatchNorm(5b_c_bn)(X209293,5b_c_bn_scale,5b_c_bn_bias)
JCudaTensor x4218;
JCudaTensor x4219, x4220, x4221;
x4219 = x4213;
x4220 = x861;
x4221 = x862;
x4218 = x863.forward_inference(x4219, x4220, x4221);

// Dealloc(X209293)
JCudaTensor x4222;
x4222 = x4213;
x4222.free();

// val X209295 = ReLU()(X209294)
JCudaTensor x4223;
JCudaTensor x4224;
x4224 = x4218;
x4223 = x812.forward(x4224);

// val X209296 = (X209295 + X209286)
JCudaTensor x4225;
JCudaTensor x4226, x4227;
x4226 = x4223;
x4227 = x4188;
x4225 = x4226.plus_i(x4227);

// Dealloc(X209286)
JCudaTensor x4228;
x4228 = x4188;
x4228.free();

// val X209297 = ReLU()(X209296)
JCudaTensor x4229;
JCudaTensor x4230;
x4230 = x4225;
x4229 = x812.forward(x4230);

// val X209298 = Convolv(1,0)(X209297,5c_a_cv_W,5c_a_cv_B)
JCudaTensor x4231;
JCudaTensor x4232, x4233, x4234;
x4232 = x4229;
x4233 = x875;
x4234 = x876;
x4231 = x826.forward(x4232, x4233, x4234);

// val X209299 = BatchNorm(5c_a_bn)(X209298,5c_a_bn_scale,5c_a_bn_bias)
JCudaTensor x4235;
JCudaTensor x4236, x4237, x4238;
x4236 = x4231;
x4237 = x881;
x4238 = x882;
x4235 = x883.forward_inference(x4236, x4237, x4238);

// Dealloc(X209298)
JCudaTensor x4239;
x4239 = x4231;
x4239.free();

// val X209300 = ReLU()(X209299)
JCudaTensor x4240;
JCudaTensor x4241;
x4241 = x4235;
x4240 = x779.forward(x4241);

// val X209301 = Convolv(1,1)(X209300,5c_b_cv_W,5c_b_cv_B)
JCudaTensor x4242;
JCudaTensor x4243, x4244, x4245;
x4243 = x4240;
x4244 = x890;
x4245 = x891;
x4242 = x786.forward(x4243, x4244, x4245);

// Dealloc(X209300)
JCudaTensor x4246;
x4246 = x4240;
x4246.free();

// val X209302 = BatchNorm(5c_b_bn)(X209301,5c_b_bn_scale,5c_b_bn_bias)
JCudaTensor x4247;
JCudaTensor x4248, x4249, x4250;
x4248 = x4242;
x4249 = x896;
x4250 = x897;
x4247 = x898.forward_inference(x4248, x4249, x4250);

// Dealloc(X209301)
JCudaTensor x4251;
x4251 = x4242;
x4251.free();

// val X209303 = ReLU()(X209302)
JCudaTensor x4252;
JCudaTensor x4253;
x4253 = x4247;
x4252 = x779.forward(x4253);

// val X209304 = Convolv(1,0)(X209303,5c_c_cv_W,5c_c_cv_B)
JCudaTensor x4254;
JCudaTensor x4255, x4256, x4257;
x4255 = x4252;
x4256 = x905;
x4257 = x906;
x4254 = x802.forward(x4255, x4256, x4257);

// Dealloc(X209303)
JCudaTensor x4258;
x4258 = x4252;
x4258.free();

// val X209305 = BatchNorm(5c_c_bn)(X209304,5c_c_bn_scale,5c_c_bn_bias)
JCudaTensor x4259;
JCudaTensor x4260, x4261, x4262;
x4260 = x4254;
x4261 = x911;
x4262 = x912;
x4259 = x913.forward_inference(x4260, x4261, x4262);

// Dealloc(X209304)
JCudaTensor x4263;
x4263 = x4254;
x4263.free();

// val X209306 = ReLU()(X209305)
JCudaTensor x4264;
JCudaTensor x4265;
x4265 = x4259;
x4264 = x812.forward(x4265);

// val X209307 = (X209306 + X209297)
JCudaTensor x4266;
JCudaTensor x4267, x4268;
x4267 = x4264;
x4268 = x4229;
x4266 = x4267.plus_i(x4268);

// Dealloc(X209297)
JCudaTensor x4269;
x4269 = x4229;
x4269.free();

// val X209308 = ReLU()(X209307)
JCudaTensor x4270;
JCudaTensor x4271;
x4271 = x4266;
x4270 = x812.forward(x4271);

// val X209309 = Pooling(7,1,0,false)(X209308)
JCudaTensor x4272;
JCudaTensor x4273;
x4273 = x4270;
x4272 = x923.forward(x4273);

// Dealloc(X209308)
JCudaTensor x4274;
x4274 = x4270;
x4274.free();

// val X209310 = (X209309[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x4275;
JCudaMatrix x4276;
JCudaMatrix x4277;
JCudaTensor x4278;
JCudaTensor x4279;
x4279 = x4272;
x4278 = x4279.flatten(1, new int[]{2048, 1, 1});
x4276 = x4278.asMatrix(1, true);
JCudaTensor x4280;
x4280 = x930;
x4277 = x4280.asMatrix(1, true);
x4275 = x4276.times(x4277);

// Dealloc(X209309)
JCudaTensor x4281;
x4281 = x4272;
x4281.free();

// val X209312 = (X209310 + (i) => fc_B)
JCudaTensor x4282;
JCudaTensor x4283, x4284;
x4283 = x4275;
x4284 = x934;
x4282 = x4284.copy(64, x4283);

// val X209313 = Cuda(Indicator(Y, 1000))
JCudaTensor x4285;
JTensorFloat x4286;
x4286 = x4.asIndicator(1000);
x4285 = x4286.asJCudaTensor();

// val X209314 = X209313 .* X209312
JCudaTensor x4287;
JCudaTensor x4288, x4289;
x4288 = x4285;
x4289 = x4282;
x4287 = x4288.times_i(x4289);

// val X209315 = Sum((X209314)(i13 | @))
JCudaTensor x4290;
JCudaMatrix x4291;
JCudaTensor x4292;
x4292 = x4287;
x4291 = x4292.asMatrix(1, true);
x4290 = x4291.sum();

// Dealloc(X209314)
JCudaTensor x4293;
x4293 = x4287;
x4293.free();

// val X209316 = Max((X209312)(i13 | @))
JCudaTensor x4294;
JCudaMatrix x4295;
JCudaTensor x4296;
x4296 = x4282;
x4295 = x4296.asMatrix(1, true);
x4294 = x4295.max();

// Dealloc(X209312)
JCudaTensor x4297;
x4297 = x4282;
x4297.free();

// val X209317 = 1{X209315 == X209316}
JCudaTensor x4298;
JCudaTensor x4299, x4300;
x4299 = x4290;
x4300 = x4294;
x4298 = x4299.eq(x4300);

// Dealloc(X209316)
JCudaTensor x4301;
x4301 = x4294;
x4301.free();

// BatchSum(((Sum(X209317) / |64|) / 10))
float x4303;
float x4304;
float x4305;
float x4306;
float x4307;
JCudaTensor x4308;
x4308 = x4298;
x4306 = x4308.sum();
x4307 = 64;
x4304 = x4306 / x4307;
x4305 = 10;
x4303 = x4304 / x4305;
x4302 += x4303;

// Print((Sum(X209317) / |64|))
float x4309;
float x4310;
float x4311;
JCudaTensor x4312;
x4312 = x4298;
x4310 = x4312.sum();
x4311 = 64;
x4309 = x4310 / x4311;
System.out.println(x5 + " test precision "  + x4309);

// Dealloc(X209317)
JCudaTensor x4313;
x4313 = x4298;
x4313.free();

}
System.out.println(x4302); 
}

}