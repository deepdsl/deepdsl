package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;

// This file is for inference only, which needs trained parameters.
public class Resnet_infer {
static{
// comment the first or both lines below for memory efficient mode
JCudaTensor.enableMemoryCache();
JCudaTensor.enableWorkspaceCache();
}
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/resnet";
// test_data_path
static String test_data_path = "dataset/imagenet/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 10;
// test_size
static int test_size = 10000;

// (BatchNorm(1_bn),List(List(64, 64, 112, 112), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x21 = new JCudnnBatchNorm(network_dir + "/1_bn", new int[]{64,64,112,112});
// (BatchNorm(2a1_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x51 = new JCudnnBatchNorm(network_dir + "/2a1_bn", new int[]{64,256,55,55});
// (BatchNorm(2a2_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x59 = new JCudnnBatchNorm(network_dir + "/2a2_a_bn", new int[]{64,64,55,55});
// (BatchNorm(2a2_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x78 = new JCudnnBatchNorm(network_dir + "/2a2_b_bn", new int[]{64,64,55,55});
// (BatchNorm(2a2_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x95 = new JCudnnBatchNorm(network_dir + "/2a2_c_bn", new int[]{64,256,55,55});
// (BatchNorm(2b_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x121 = new JCudnnBatchNorm(network_dir + "/2b_a_bn", new int[]{64,64,55,55});
// (BatchNorm(2b_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x138 = new JCudnnBatchNorm(network_dir + "/2b_b_bn", new int[]{64,64,55,55});
// (BatchNorm(2b_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x155 = new JCudnnBatchNorm(network_dir + "/2b_c_bn", new int[]{64,256,55,55});
// (BatchNorm(2c_a_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x177 = new JCudnnBatchNorm(network_dir + "/2c_a_bn", new int[]{64,64,55,55});
// (BatchNorm(2c_b_bn),List(List(64, 64, 55, 55), List(1, 64, 1, 1), List(1, 64, 1, 1)))
static JCudnnBatchNorm x194 = new JCudnnBatchNorm(network_dir + "/2c_b_bn", new int[]{64,64,55,55});
// (BatchNorm(2c_c_bn),List(List(64, 256, 55, 55), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x211 = new JCudnnBatchNorm(network_dir + "/2c_c_bn", new int[]{64,256,55,55});
// (BatchNorm(3a1_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x250 = new JCudnnBatchNorm(network_dir + "/3a1_bn", new int[]{64,512,28,28});
// (BatchNorm(3a2_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x242 = new JCudnnBatchNorm(network_dir + "/3a2_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3a2_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x269 = new JCudnnBatchNorm(network_dir + "/3a2_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3a2_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x287 = new JCudnnBatchNorm(network_dir + "/3a2_c_bn", new int[]{64,512,28,28});
// (BatchNorm(3b_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x313 = new JCudnnBatchNorm(network_dir + "/3b_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3b_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x330 = new JCudnnBatchNorm(network_dir + "/3b_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3b_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x347 = new JCudnnBatchNorm(network_dir + "/3b_c_bn", new int[]{64,512,28,28});
// (BatchNorm(3c_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x369 = new JCudnnBatchNorm(network_dir + "/3c_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3c_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x386 = new JCudnnBatchNorm(network_dir + "/3c_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3c_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x403 = new JCudnnBatchNorm(network_dir + "/3c_c_bn", new int[]{64,512,28,28});
// (BatchNorm(3d_a_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x425 = new JCudnnBatchNorm(network_dir + "/3d_a_bn", new int[]{64,128,28,28});
// (BatchNorm(3d_b_bn),List(List(64, 128, 28, 28), List(1, 128, 1, 1), List(1, 128, 1, 1)))
static JCudnnBatchNorm x442 = new JCudnnBatchNorm(network_dir + "/3d_b_bn", new int[]{64,128,28,28});
// (BatchNorm(3d_c_bn),List(List(64, 512, 28, 28), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x459 = new JCudnnBatchNorm(network_dir + "/3d_c_bn", new int[]{64,512,28,28});
// (BatchNorm(4a1_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x490 = new JCudnnBatchNorm(network_dir + "/4a1_bn", new int[]{64,1024,14,14});
// (BatchNorm(4a2_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x498 = new JCudnnBatchNorm(network_dir + "/4a2_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4a2_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x517 = new JCudnnBatchNorm(network_dir + "/4a2_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4a2_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x535 = new JCudnnBatchNorm(network_dir + "/4a2_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4b_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x561 = new JCudnnBatchNorm(network_dir + "/4b_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4b_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x578 = new JCudnnBatchNorm(network_dir + "/4b_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4b_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x595 = new JCudnnBatchNorm(network_dir + "/4b_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4c_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x617 = new JCudnnBatchNorm(network_dir + "/4c_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4c_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x634 = new JCudnnBatchNorm(network_dir + "/4c_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4c_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x651 = new JCudnnBatchNorm(network_dir + "/4c_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4d_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x673 = new JCudnnBatchNorm(network_dir + "/4d_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4d_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x690 = new JCudnnBatchNorm(network_dir + "/4d_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4d_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x707 = new JCudnnBatchNorm(network_dir + "/4d_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4e_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x729 = new JCudnnBatchNorm(network_dir + "/4e_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4e_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x746 = new JCudnnBatchNorm(network_dir + "/4e_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4e_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x763 = new JCudnnBatchNorm(network_dir + "/4e_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(4f_a_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x785 = new JCudnnBatchNorm(network_dir + "/4f_a_bn", new int[]{64,256,14,14});
// (BatchNorm(4f_b_bn),List(List(64, 256, 14, 14), List(1, 256, 1, 1), List(1, 256, 1, 1)))
static JCudnnBatchNorm x802 = new JCudnnBatchNorm(network_dir + "/4f_b_bn", new int[]{64,256,14,14});
// (BatchNorm(4f_c_bn),List(List(64, 1024, 14, 14), List(1, 1024, 1, 1), List(1, 1024, 1, 1)))
static JCudnnBatchNorm x819 = new JCudnnBatchNorm(network_dir + "/4f_c_bn", new int[]{64,1024,14,14});
// (BatchNorm(5a1_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x850 = new JCudnnBatchNorm(network_dir + "/5a1_bn", new int[]{64,2048,7,7});
// (BatchNorm(5a2_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x858 = new JCudnnBatchNorm(network_dir + "/5a2_a_bn", new int[]{64,512,7,7});
// (BatchNorm(5a2_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x877 = new JCudnnBatchNorm(network_dir + "/5a2_b_bn", new int[]{64,512,7,7});
// (BatchNorm(5a2_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x895 = new JCudnnBatchNorm(network_dir + "/5a2_c_bn", new int[]{64,2048,7,7});
// (BatchNorm(5b_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x921 = new JCudnnBatchNorm(network_dir + "/5b_a_bn", new int[]{64,512,7,7});
// (BatchNorm(5b_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x938 = new JCudnnBatchNorm(network_dir + "/5b_b_bn", new int[]{64,512,7,7});
// (BatchNorm(5b_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x955 = new JCudnnBatchNorm(network_dir + "/5b_c_bn", new int[]{64,2048,7,7});
// (BatchNorm(5c_a_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x977 = new JCudnnBatchNorm(network_dir + "/5c_a_bn", new int[]{64,512,7,7});
// (BatchNorm(5c_b_bn),List(List(64, 512, 7, 7), List(1, 512, 1, 1), List(1, 512, 1, 1)))
static JCudnnBatchNorm x994 = new JCudnnBatchNorm(network_dir + "/5c_b_bn", new int[]{64,512,7,7});
// (BatchNorm(5c_c_bn),List(List(64, 2048, 7, 7), List(1, 2048, 1, 1), List(1, 2048, 1, 1)))
static JCudnnBatchNorm x1011 = new JCudnnBatchNorm(network_dir + "/5c_c_bn", new int[]{64,2048,7,7});
// (Convolv(1,0),List(List(64, 1024, 14, 14), List(256, 1024, 1, 1), List(256)))
static JCudnnConvolution x554 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(List(64, 128, 28, 28), List(512, 128, 1, 1), List(512)))
static JCudnnConvolution x279 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(List(64, 2048, 7, 7), List(512, 2048, 1, 1), List(512)))
static JCudnnConvolution x914 = new JCudnnConvolution(new int[]{64,2048,7,7},new int[]{512,2048,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(List(64, 256, 14, 14), List(1024, 256, 1, 1), List(1024)))
static JCudnnConvolution x527 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(List(64, 256, 55, 55), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x114 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(64, 512, 28, 28), List(128, 512, 1, 1), List(128)))
static JCudnnConvolution x306 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{128,512,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(List(64, 512, 7, 7), List(2048, 512, 1, 1), List(2048)))
static JCudnnConvolution x887 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{2048,512,1,1},new int[]{2048}, 1, 0);
// (Convolv(1,0),List(List(64, 64, 55, 55), List(256, 64, 1, 1), List(256)))
static JCudnnConvolution x36 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(List(64, 64, 55, 55), List(64, 64, 1, 1), List(64)))
static JCudnnConvolution x43 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,1,1},new int[]{64}, 1, 0);
// (Convolv(1,1),List(List(64, 128, 28, 28), List(128, 128, 3, 3), List(128)))
static JCudnnConvolution x261 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(64, 256, 14, 14), List(256, 256, 3, 3), List(256)))
static JCudnnConvolution x509 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(List(64, 512, 7, 7), List(512, 512, 3, 3), List(512)))
static JCudnnConvolution x869 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(List(64, 64, 55, 55), List(64, 64, 3, 3), List(64)))
static JCudnnConvolution x70 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,3,3},new int[]{64}, 1, 1);
// (Convolv(2,0),List(List(64, 1024, 14, 14), List(2048, 1024, 1, 1), List(2048)))
static JCudnnConvolution x842 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{2048,1024,1,1},new int[]{2048}, 2, 0);
// (Convolv(2,0),List(List(64, 1024, 14, 14), List(512, 1024, 1, 1), List(512)))
static JCudnnConvolution x835 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{512,1024,1,1},new int[]{512}, 2, 0);
// (Convolv(2,0),List(List(64, 256, 55, 55), List(128, 256, 1, 1), List(128)))
static JCudnnConvolution x234 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{128,256,1,1},new int[]{128}, 2, 0);
// (Convolv(2,0),List(List(64, 256, 55, 55), List(512, 256, 1, 1), List(512)))
static JCudnnConvolution x227 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{512,256,1,1},new int[]{512}, 2, 0);
// (Convolv(2,0),List(List(64, 512, 28, 28), List(1024, 512, 1, 1), List(1024)))
static JCudnnConvolution x482 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{1024,512,1,1},new int[]{1024}, 2, 0);
// (Convolv(2,0),List(List(64, 512, 28, 28), List(256, 512, 1, 1), List(256)))
static JCudnnConvolution x475 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{256,512,1,1},new int[]{256}, 2, 0);
// (Convolv(2,3),List(List(64, 3, 224, 224), List(64, 3, 7, 7), List(64)))
static JCudnnConvolution x13 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
// (Imagenet,false)
static ImagenetFactory x1 = ImagenetFactory.getFactory(test_data_path, test_size, new int[]{64, 3, 224, 224}, 1000, true);
// (Pooling(3,2,0,true),List(List(64, 64, 112, 112)))
static JCudnnPooling x28 = new JCudnnPooling(new int[]{64,64,112,112}, 3, 2, 0, PoolingType.MAX);
// (Pooling(7,1,0,false),List(List(64, 2048, 7, 7)))
static JCudnnPooling x1023 = new JCudnnPooling(new int[]{64,2048,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
// (ReLU(),List(List(64, 1024, 14, 14)))
static JCudnnActivation x539 = new JCudnnActivation(new int[]{64,1024,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(64, 128, 28, 28)))
static JCudnnActivation x254 = new JCudnnActivation(new int[]{64,128,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(64, 2048, 7, 7)))
static JCudnnActivation x899 = new JCudnnActivation(new int[]{64,2048,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(64, 256, 14, 14)))
static JCudnnActivation x502 = new JCudnnActivation(new int[]{64,256,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(64, 256, 55, 55)))
static JCudnnActivation x99 = new JCudnnActivation(new int[]{64,256,55,55}, ActivationMode.RELU);
// (ReLU(),List(List(64, 512, 28, 28)))
static JCudnnActivation x291 = new JCudnnActivation(new int[]{64,512,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(64, 512, 7, 7)))
static JCudnnActivation x862 = new JCudnnActivation(new int[]{64,512,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(64, 64, 112, 112)))
static JCudnnActivation x25 = new JCudnnActivation(new int[]{64,64,112,112}, ActivationMode.RELU);
// (ReLU(),List(List(64, 64, 55, 55)))
static JCudnnActivation x63 = new JCudnnActivation(new int[]{64,64,55,55}, ActivationMode.RELU);
// 1_bn_bias
static JCudaTensor x20 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/1_bn_bias").asJCudaTensor();
// 1_bn_scale
static JCudaTensor x19 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/1_bn_scale").asJCudaTensor();
// 1_cv_B
static JCudaTensor x12 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 1_cv_W
static JCudaTensor x11 = JTensor.randomFloat(-0.11664237f, 0.11664237f, 64, 3, 7, 7).load(network_dir + "/1_cv_W").asJCudaTensor();
// 2a1_bn_bias
static JCudaTensor x50 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2a1_bn_bias").asJCudaTensor();
// 2a1_bn_scale
static JCudaTensor x49 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2a1_bn_scale").asJCudaTensor();
// 2a1_cv_B
static JCudaTensor x35 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2a1_cv_W
static JCudaTensor x34 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2a1_cv_W").asJCudaTensor();
// 2a2_a_bn_bias
static JCudaTensor x58 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2a2_a_bn_bias").asJCudaTensor();
// 2a2_a_bn_scale
static JCudaTensor x57 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2a2_a_bn_scale").asJCudaTensor();
// 2a2_a_cv_B
static JCudaTensor x42 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2a2_a_cv_W
static JCudaTensor x41 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/2a2_a_cv_W").asJCudaTensor();
// 2a2_b_bn_bias
static JCudaTensor x77 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2a2_b_bn_bias").asJCudaTensor();
// 2a2_b_bn_scale
static JCudaTensor x76 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2a2_b_bn_scale").asJCudaTensor();
// 2a2_b_cv_B
static JCudaTensor x69 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2a2_b_cv_W
static JCudaTensor x68 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2a2_b_cv_W").asJCudaTensor();
// 2a2_c_bn_bias
static JCudaTensor x94 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_bias").asJCudaTensor();
// 2a2_c_bn_scale
static JCudaTensor x93 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_scale").asJCudaTensor();
// 2a2_c_cv_B
static JCudaTensor x87 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2a2_c_cv_W
static JCudaTensor x86 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2a2_c_cv_W").asJCudaTensor();
// 2b_a_bn_bias
static JCudaTensor x120 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_bias").asJCudaTensor();
// 2b_a_bn_scale
static JCudaTensor x119 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_scale").asJCudaTensor();
// 2b_a_cv_B
static JCudaTensor x113 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2b_a_cv_W
static JCudaTensor x112 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2b_a_cv_W").asJCudaTensor();
// 2b_b_bn_bias
static JCudaTensor x137 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_bias").asJCudaTensor();
// 2b_b_bn_scale
static JCudaTensor x136 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_scale").asJCudaTensor();
// 2b_b_cv_B
static JCudaTensor x130 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2b_b_cv_W
static JCudaTensor x129 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2b_b_cv_W").asJCudaTensor();
// 2b_c_bn_bias
static JCudaTensor x154 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_bias").asJCudaTensor();
// 2b_c_bn_scale
static JCudaTensor x153 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_scale").asJCudaTensor();
// 2b_c_cv_B
static JCudaTensor x147 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2b_c_cv_W
static JCudaTensor x146 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2b_c_cv_W").asJCudaTensor();
// 2c_a_bn_bias
static JCudaTensor x176 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_bias").asJCudaTensor();
// 2c_a_bn_scale
static JCudaTensor x175 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_scale").asJCudaTensor();
// 2c_a_cv_B
static JCudaTensor x170 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2c_a_cv_W
static JCudaTensor x169 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2c_a_cv_W").asJCudaTensor();
// 2c_b_bn_bias
static JCudaTensor x193 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_bias").asJCudaTensor();
// 2c_b_bn_scale
static JCudaTensor x192 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_scale").asJCudaTensor();
// 2c_b_cv_B
static JCudaTensor x186 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2c_b_cv_W
static JCudaTensor x185 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2c_b_cv_W").asJCudaTensor();
// 2c_c_bn_bias
static JCudaTensor x210 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_bias").asJCudaTensor();
// 2c_c_bn_scale
static JCudaTensor x209 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_scale").asJCudaTensor();
// 2c_c_cv_B
static JCudaTensor x203 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2c_c_cv_W
static JCudaTensor x202 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2c_c_cv_W").asJCudaTensor();
// 3a1_bn_bias
static JCudaTensor x249 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_bias").asJCudaTensor();
// 3a1_bn_scale
static JCudaTensor x248 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_scale").asJCudaTensor();
// 3a1_cv_B
static JCudaTensor x226 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3a1_cv_W
static JCudaTensor x225 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 512, 256, 1, 1).load(network_dir + "/3a1_cv_W").asJCudaTensor();
// 3a2_a_bn_bias
static JCudaTensor x241 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_bias").asJCudaTensor();
// 3a2_a_bn_scale
static JCudaTensor x240 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_scale").asJCudaTensor();
// 3a2_a_cv_B
static JCudaTensor x233 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3a2_a_cv_W
static JCudaTensor x232 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/3a2_a_cv_W").asJCudaTensor();
// 3a2_b_bn_bias
static JCudaTensor x268 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_bias").asJCudaTensor();
// 3a2_b_bn_scale
static JCudaTensor x267 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_scale").asJCudaTensor();
// 3a2_b_cv_B
static JCudaTensor x260 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3a2_b_cv_W
static JCudaTensor x259 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3a2_b_cv_W").asJCudaTensor();
// 3a2_c_bn_bias
static JCudaTensor x286 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_bias").asJCudaTensor();
// 3a2_c_bn_scale
static JCudaTensor x285 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_scale").asJCudaTensor();
// 3a2_c_cv_B
static JCudaTensor x278 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3a2_c_cv_W
static JCudaTensor x277 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3a2_c_cv_W").asJCudaTensor();
// 3b_a_bn_bias
static JCudaTensor x312 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_bias").asJCudaTensor();
// 3b_a_bn_scale
static JCudaTensor x311 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_scale").asJCudaTensor();
// 3b_a_cv_B
static JCudaTensor x305 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3b_a_cv_W
static JCudaTensor x304 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3b_a_cv_W").asJCudaTensor();
// 3b_b_bn_bias
static JCudaTensor x329 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_bias").asJCudaTensor();
// 3b_b_bn_scale
static JCudaTensor x328 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_scale").asJCudaTensor();
// 3b_b_cv_B
static JCudaTensor x322 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3b_b_cv_W
static JCudaTensor x321 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3b_b_cv_W").asJCudaTensor();
// 3b_c_bn_bias
static JCudaTensor x346 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_bias").asJCudaTensor();
// 3b_c_bn_scale
static JCudaTensor x345 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_scale").asJCudaTensor();
// 3b_c_cv_B
static JCudaTensor x339 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3b_c_cv_W
static JCudaTensor x338 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3b_c_cv_W").asJCudaTensor();
// 3c_a_bn_bias
static JCudaTensor x368 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_bias").asJCudaTensor();
// 3c_a_bn_scale
static JCudaTensor x367 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_scale").asJCudaTensor();
// 3c_a_cv_B
static JCudaTensor x362 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3c_a_cv_W
static JCudaTensor x361 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3c_a_cv_W").asJCudaTensor();
// 3c_b_bn_bias
static JCudaTensor x385 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_bias").asJCudaTensor();
// 3c_b_bn_scale
static JCudaTensor x384 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_scale").asJCudaTensor();
// 3c_b_cv_B
static JCudaTensor x378 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3c_b_cv_W
static JCudaTensor x377 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3c_b_cv_W").asJCudaTensor();
// 3c_c_bn_bias
static JCudaTensor x402 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_bias").asJCudaTensor();
// 3c_c_bn_scale
static JCudaTensor x401 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_scale").asJCudaTensor();
// 3c_c_cv_B
static JCudaTensor x395 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3c_c_cv_W
static JCudaTensor x394 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3c_c_cv_W").asJCudaTensor();
// 3d_a_bn_bias
static JCudaTensor x424 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_bias").asJCudaTensor();
// 3d_a_bn_scale
static JCudaTensor x423 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_scale").asJCudaTensor();
// 3d_a_cv_B
static JCudaTensor x418 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3d_a_cv_W
static JCudaTensor x417 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3d_a_cv_W").asJCudaTensor();
// 3d_b_bn_bias
static JCudaTensor x441 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_bias").asJCudaTensor();
// 3d_b_bn_scale
static JCudaTensor x440 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_scale").asJCudaTensor();
// 3d_b_cv_B
static JCudaTensor x434 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3d_b_cv_W
static JCudaTensor x433 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3d_b_cv_W").asJCudaTensor();
// 3d_c_bn_bias
static JCudaTensor x458 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_bias").asJCudaTensor();
// 3d_c_bn_scale
static JCudaTensor x457 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_scale").asJCudaTensor();
// 3d_c_cv_B
static JCudaTensor x451 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3d_c_cv_W
static JCudaTensor x450 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3d_c_cv_W").asJCudaTensor();
// 4a1_bn_bias
static JCudaTensor x489 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_bias").asJCudaTensor();
// 4a1_bn_scale
static JCudaTensor x488 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_scale").asJCudaTensor();
// 4a1_cv_B
static JCudaTensor x481 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4a1_cv_W
static JCudaTensor x480 = JTensor.randomFloat(-0.0625f, 0.0625f, 1024, 512, 1, 1).load(network_dir + "/4a1_cv_W").asJCudaTensor();
// 4a2_a_bn_bias
static JCudaTensor x497 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_bias").asJCudaTensor();
// 4a2_a_bn_scale
static JCudaTensor x496 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_scale").asJCudaTensor();
// 4a2_a_cv_B
static JCudaTensor x474 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4a2_a_cv_W
static JCudaTensor x473 = JTensor.randomFloat(-0.0625f, 0.0625f, 256, 512, 1, 1).load(network_dir + "/4a2_a_cv_W").asJCudaTensor();
// 4a2_b_bn_bias
static JCudaTensor x516 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_bias").asJCudaTensor();
// 4a2_b_bn_scale
static JCudaTensor x515 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_scale").asJCudaTensor();
// 4a2_b_cv_B
static JCudaTensor x508 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4a2_b_cv_W
static JCudaTensor x507 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4a2_b_cv_W").asJCudaTensor();
// 4a2_c_bn_bias
static JCudaTensor x534 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_bias").asJCudaTensor();
// 4a2_c_bn_scale
static JCudaTensor x533 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_scale").asJCudaTensor();
// 4a2_c_cv_B
static JCudaTensor x526 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4a2_c_cv_W
static JCudaTensor x525 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4a2_c_cv_W").asJCudaTensor();
// 4b_a_bn_bias
static JCudaTensor x560 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_bias").asJCudaTensor();
// 4b_a_bn_scale
static JCudaTensor x559 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_scale").asJCudaTensor();
// 4b_a_cv_B
static JCudaTensor x553 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4b_a_cv_W
static JCudaTensor x552 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4b_a_cv_W").asJCudaTensor();
// 4b_b_bn_bias
static JCudaTensor x577 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_bias").asJCudaTensor();
// 4b_b_bn_scale
static JCudaTensor x576 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_scale").asJCudaTensor();
// 4b_b_cv_B
static JCudaTensor x570 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4b_b_cv_W
static JCudaTensor x569 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4b_b_cv_W").asJCudaTensor();
// 4b_c_bn_bias
static JCudaTensor x594 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_bias").asJCudaTensor();
// 4b_c_bn_scale
static JCudaTensor x593 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_scale").asJCudaTensor();
// 4b_c_cv_B
static JCudaTensor x587 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4b_c_cv_W
static JCudaTensor x586 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4b_c_cv_W").asJCudaTensor();
// 4c_a_bn_bias
static JCudaTensor x616 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_bias").asJCudaTensor();
// 4c_a_bn_scale
static JCudaTensor x615 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_scale").asJCudaTensor();
// 4c_a_cv_B
static JCudaTensor x610 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4c_a_cv_W
static JCudaTensor x609 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4c_a_cv_W").asJCudaTensor();
// 4c_b_bn_bias
static JCudaTensor x633 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_bias").asJCudaTensor();
// 4c_b_bn_scale
static JCudaTensor x632 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_scale").asJCudaTensor();
// 4c_b_cv_B
static JCudaTensor x626 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4c_b_cv_W
static JCudaTensor x625 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4c_b_cv_W").asJCudaTensor();
// 4c_c_bn_bias
static JCudaTensor x650 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_bias").asJCudaTensor();
// 4c_c_bn_scale
static JCudaTensor x649 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_scale").asJCudaTensor();
// 4c_c_cv_B
static JCudaTensor x643 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4c_c_cv_W
static JCudaTensor x642 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4c_c_cv_W").asJCudaTensor();
// 4d_a_bn_bias
static JCudaTensor x672 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_bias").asJCudaTensor();
// 4d_a_bn_scale
static JCudaTensor x671 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_scale").asJCudaTensor();
// 4d_a_cv_B
static JCudaTensor x666 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4d_a_cv_W
static JCudaTensor x665 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4d_a_cv_W").asJCudaTensor();
// 4d_b_bn_bias
static JCudaTensor x689 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_bias").asJCudaTensor();
// 4d_b_bn_scale
static JCudaTensor x688 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_scale").asJCudaTensor();
// 4d_b_cv_B
static JCudaTensor x682 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4d_b_cv_W
static JCudaTensor x681 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4d_b_cv_W").asJCudaTensor();
// 4d_c_bn_bias
static JCudaTensor x706 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_bias").asJCudaTensor();
// 4d_c_bn_scale
static JCudaTensor x705 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_scale").asJCudaTensor();
// 4d_c_cv_B
static JCudaTensor x699 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4d_c_cv_W
static JCudaTensor x698 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4d_c_cv_W").asJCudaTensor();
// 4e_a_bn_bias
static JCudaTensor x728 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_bias").asJCudaTensor();
// 4e_a_bn_scale
static JCudaTensor x727 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_scale").asJCudaTensor();
// 4e_a_cv_B
static JCudaTensor x722 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4e_a_cv_W
static JCudaTensor x721 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4e_a_cv_W").asJCudaTensor();
// 4e_b_bn_bias
static JCudaTensor x745 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_bias").asJCudaTensor();
// 4e_b_bn_scale
static JCudaTensor x744 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_scale").asJCudaTensor();
// 4e_b_cv_B
static JCudaTensor x738 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4e_b_cv_W
static JCudaTensor x737 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4e_b_cv_W").asJCudaTensor();
// 4e_c_bn_bias
static JCudaTensor x762 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_bias").asJCudaTensor();
// 4e_c_bn_scale
static JCudaTensor x761 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_scale").asJCudaTensor();
// 4e_c_cv_B
static JCudaTensor x755 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4e_c_cv_W
static JCudaTensor x754 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4e_c_cv_W").asJCudaTensor();
// 4f_a_bn_bias
static JCudaTensor x784 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_bias").asJCudaTensor();
// 4f_a_bn_scale
static JCudaTensor x783 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_scale").asJCudaTensor();
// 4f_a_cv_B
static JCudaTensor x778 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4f_a_cv_W
static JCudaTensor x777 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4f_a_cv_W").asJCudaTensor();
// 4f_b_bn_bias
static JCudaTensor x801 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_bias").asJCudaTensor();
// 4f_b_bn_scale
static JCudaTensor x800 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_scale").asJCudaTensor();
// 4f_b_cv_B
static JCudaTensor x794 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4f_b_cv_W
static JCudaTensor x793 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4f_b_cv_W").asJCudaTensor();
// 4f_c_bn_bias
static JCudaTensor x818 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_bias").asJCudaTensor();
// 4f_c_bn_scale
static JCudaTensor x817 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_scale").asJCudaTensor();
// 4f_c_cv_B
static JCudaTensor x811 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4f_c_cv_W
static JCudaTensor x810 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4f_c_cv_W").asJCudaTensor();
// 5a1_bn_bias
static JCudaTensor x849 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_bias").asJCudaTensor();
// 5a1_bn_scale
static JCudaTensor x848 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_scale").asJCudaTensor();
// 5a1_cv_B
static JCudaTensor x841 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5a1_cv_W
static JCudaTensor x840 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 2048, 1024, 1, 1).load(network_dir + "/5a1_cv_W").asJCudaTensor();
// 5a2_a_bn_bias
static JCudaTensor x857 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_bias").asJCudaTensor();
// 5a2_a_bn_scale
static JCudaTensor x856 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_scale").asJCudaTensor();
// 5a2_a_cv_B
static JCudaTensor x834 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5a2_a_cv_W
static JCudaTensor x833 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 512, 1024, 1, 1).load(network_dir + "/5a2_a_cv_W").asJCudaTensor();
// 5a2_b_bn_bias
static JCudaTensor x876 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_bias").asJCudaTensor();
// 5a2_b_bn_scale
static JCudaTensor x875 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_scale").asJCudaTensor();
// 5a2_b_cv_B
static JCudaTensor x868 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5a2_b_cv_W
static JCudaTensor x867 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5a2_b_cv_W").asJCudaTensor();
// 5a2_c_bn_bias
static JCudaTensor x894 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_bias").asJCudaTensor();
// 5a2_c_bn_scale
static JCudaTensor x893 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_scale").asJCudaTensor();
// 5a2_c_cv_B
static JCudaTensor x886 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5a2_c_cv_W
static JCudaTensor x885 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5a2_c_cv_W").asJCudaTensor();
// 5b_a_bn_bias
static JCudaTensor x920 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_bias").asJCudaTensor();
// 5b_a_bn_scale
static JCudaTensor x919 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_scale").asJCudaTensor();
// 5b_a_cv_B
static JCudaTensor x913 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5b_a_cv_W
static JCudaTensor x912 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5b_a_cv_W").asJCudaTensor();
// 5b_b_bn_bias
static JCudaTensor x937 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_bias").asJCudaTensor();
// 5b_b_bn_scale
static JCudaTensor x936 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_scale").asJCudaTensor();
// 5b_b_cv_B
static JCudaTensor x930 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5b_b_cv_W
static JCudaTensor x929 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5b_b_cv_W").asJCudaTensor();
// 5b_c_bn_bias
static JCudaTensor x954 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_bias").asJCudaTensor();
// 5b_c_bn_scale
static JCudaTensor x953 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_scale").asJCudaTensor();
// 5b_c_cv_B
static JCudaTensor x947 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5b_c_cv_W
static JCudaTensor x946 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5b_c_cv_W").asJCudaTensor();
// 5c_a_bn_bias
static JCudaTensor x976 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_bias").asJCudaTensor();
// 5c_a_bn_scale
static JCudaTensor x975 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_scale").asJCudaTensor();
// 5c_a_cv_B
static JCudaTensor x970 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5c_a_cv_W
static JCudaTensor x969 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5c_a_cv_W").asJCudaTensor();
// 5c_b_bn_bias
static JCudaTensor x993 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_bias").asJCudaTensor();
// 5c_b_bn_scale
static JCudaTensor x992 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_scale").asJCudaTensor();
// 5c_b_cv_B
static JCudaTensor x986 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5c_b_cv_W
static JCudaTensor x985 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5c_b_cv_W").asJCudaTensor();
// 5c_c_bn_bias
static JCudaTensor x1010 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_bias").asJCudaTensor();
// 5c_c_bn_scale
static JCudaTensor x1009 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_scale").asJCudaTensor();
// 5c_c_cv_B
static JCudaTensor x1003 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5c_c_cv_W
static JCudaTensor x1002 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5c_c_cv_W").asJCudaTensor();
// X
static JTensorFloat x2;
// fc_B
static JCudaTensor x1036 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
// fc_W
static JCudaTensor x1031 = JTensor.randomFloat(-0.03125f, 0.03125f, 1000, 2048).load(network_dir + "/fc_W").asJCudaTensor();

public static void main(String[] args){
test();
x705.free();
x937.free();
x643.free();
x515.free();
x993.free();
x119.free();
x424.free();
x321.free();
x136.free();
x305.free();
x992.free();
x507.free();
x232.free();
x744.free();
x474.free();
x241.free();
x260.free();
x526.free();
x594.free();
x94.free();
x534.free();
x857.free();
x856.free();
x154.free();
x841.free();
x385.free();
x784.free();
x481.free();
x552.free();
x169.free();
x34.free();
x817.free();
x488.free();
x19.free();
x175.free();
x587.free();
x665.free();
x894.free();
x285.free();
x147.free();
x689.free();
x130.free();
x947.free();
x1009.free();
x112.free();
x441.free();
x875.free();
x368.free();
x377.free();
x473.free();
x810.free();
x451.free();
x226.free();
x867.free();
x311.free();
x49.free();
x35.free();
x176.free();
x811.free();
x985.free();
x868.free();
x58.free();
x936.free();
x50.free();
x186.free();
x801.free();
x609.free();
x267.free();
x1010.free();
x339.free();
x777.free();
x975.free();
x137.free();
x384.free();
x77.free();
x681.free();
x722.free();
x913.free();
x794.free();
x1003.free();
x248.free();
x919.free();
x633.free();
x170.free();
x727.free();
x203.free();
x699.free();
x885.free();
x401.free();
x12.free();
x761.free();
x338.free();
x576.free();
x553.free();
x367.free();
x192.free();
x893.free();
x946.free();
x833.free();
x849.free();
x570.free();
x259.free();
x86.free();
x153.free();
x625.free();
x682.free();
x930.free();
x225.free();
x120.free();
x378.free();
x395.free();
x278.free();
x41.free();
x362.free();
x113.free();
x1031.free();
x402.free();
x920.free();
x615.free();
x1002.free();
x20.free();
x886.free();
x986.free();
x93.free();
x57.free();
x969.free();
x329.free();
x268.free();
x616.free();
x642.free();
x577.free();
x480.free();
x457.free();
x848.free();
x312.free();
x672.free();
x286.free();
x69.free();
x496.free();
x433.free();
x233.free();
x202.free();
x1036.free();
x569.free();
x929.free();
x737.free();
x745.free();
x129.free();
x688.free();
x146.free();
x800.free();
x762.free();
x346.free();
x418.free();
x834.free();
x793.free();
x193.free();
x666.free();
x840.free();
x516.free();
x458.free();
x328.free();
x626.free();
x954.free();
x345.free();
x721.free();
x650.free();
x87.free();
x533.free();
x322.free();
x209.free();
x508.free();
x417.free();
x559.free();
x593.free();
x450.free();
x755.free();
x68.free();
x434.free();
x649.free();
x497.free();
x632.free();
x728.free();
x738.free();
x42.free();
x818.free();
x76.free();
x912.free();
x394.free();
x876.free();
x706.free();
x976.free();
x525.free();
x560.free();
x671.free();
x953.free();
x783.free();
x185.free();
x698.free();
x11.free();
x277.free();
x586.free();
x210.free();
x489.free();
x754.free();
x423.free();
x440.free();
x778.free();
x304.free();
x240.free();
x610.free();
x249.free();
x970.free();
x361.free();
x1023.free();
x877.free();
x850.free();
x242.free();
x819.free();
x227.free();
x561.free();
x138.free();
x269.free();
x51.free();
x369.free();
x250.free();
x386.free();
x59.free();
x63.free();
x482.free();
x554.free();
x895.free();
x634.free();
x977.free();
x36.free();
x475.free();
x858.free();
x177.free();
x313.free();
x70.free();
x403.free();
x869.free();
x938.free();
x955.free();
x25.free();
x442.free();
x535.free();
x690.free();
x99.free();
x261.free();
x763.free();
x1011.free();
x914.free();
x835.free();
x899.free();
x121.free();
x78.free();
x729.free();
x425.free();
x921.free();
x490.free();
x746.free();
x498.free();
x291.free();
x13.free();
x306.free();
x28.free();
x155.free();
x617.free();
x707.free();
x527.free();
x862.free();
x287.free();
x578.free();
x279.free();
x502.free();
x785.free();
x211.free();
x459.free();
x194.free();
x95.free();
x114.free();
x254.free();
x347.free();
x517.free();
x651.free();
x43.free();
x21.free();
x994.free();
x509.free();
x595.free();
x539.free();
x802.free();
x887.free();
x330.free();
x842.free();
x234.free();
x673.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void test() {
 for(int x3=0; x3<test_itr; x3++) {
JTensorFloatTuple x4 =  x1.nextFloat();
x2 = x4.image;

// val X9012 = Cuda(X)
JCudaTensor x5;
JTensorFloat x6;
x6 = x2;
x5 = x6.asJCudaTensor();

// val X9013 = Convolv(2,3)(X9012,1_cv_W,1_cv_B)
JCudaTensor x7;
JCudaTensor x8, x9, x10;
x8 = x5;
x9 = x11;
x10 = x12;
x7 = x13.forward(x8, x9, x10);

// Dealloc(X9012)
JCudaTensor x14;
x14 = x5;
x14.free();

// val X9014 = BatchNorm(1_bn)(X9013,1_bn_scale,1_bn_bias)
JCudaTensor x15;
JCudaTensor x16, x17, x18;
x16 = x7;
x17 = x19;
x18 = x20;
x15 = x21.forward_inference(x16, x17, x18);

// Dealloc(X9013)
JCudaTensor x22;
x22 = x7;
x22.free();

// val X9015 = ReLU()(X9014)
JCudaTensor x23;
JCudaTensor x24;
x24 = x15;
x23 = x25.forward(x24);

// val X9016 = Pooling(3,2,0,true)(X9015)
JCudaTensor x26;
JCudaTensor x27;
x27 = x23;
x26 = x28.forward(x27);

// Dealloc(X9015)
JCudaTensor x29;
x29 = x23;
x29.free();

// val X9017 = Convolv(1,0)(X9016,2a1_cv_W,2a1_cv_B)
JCudaTensor x30;
JCudaTensor x31, x32, x33;
x31 = x26;
x32 = x34;
x33 = x35;
x30 = x36.forward(x31, x32, x33);

// val X9020 = Convolv(1,0)(X9016,2a2_a_cv_W,2a2_a_cv_B)
JCudaTensor x37;
JCudaTensor x38, x39, x40;
x38 = x26;
x39 = x41;
x40 = x42;
x37 = x43.forward(x38, x39, x40);

// Dealloc(X9016)
JCudaTensor x44;
x44 = x26;
x44.free();

// val X9018 = BatchNorm(2a1_bn)(X9017,2a1_bn_scale,2a1_bn_bias)
JCudaTensor x45;
JCudaTensor x46, x47, x48;
x46 = x30;
x47 = x49;
x48 = x50;
x45 = x51.forward_inference(x46, x47, x48);

// Dealloc(X9017)
JCudaTensor x52;
x52 = x30;
x52.free();

// val X9021 = BatchNorm(2a2_a_bn)(X9020,2a2_a_bn_scale,2a2_a_bn_bias)
JCudaTensor x53;
JCudaTensor x54, x55, x56;
x54 = x37;
x55 = x57;
x56 = x58;
x53 = x59.forward_inference(x54, x55, x56);

// Dealloc(X9020)
JCudaTensor x60;
x60 = x37;
x60.free();

// val X9022 = ReLU()(X9021)
JCudaTensor x61;
JCudaTensor x62;
x62 = x53;
x61 = x63.forward(x62);

// val X9023 = Convolv(1,1)(X9022,2a2_b_cv_W,2a2_b_cv_B)
JCudaTensor x64;
JCudaTensor x65, x66, x67;
x65 = x61;
x66 = x68;
x67 = x69;
x64 = x70.forward(x65, x66, x67);

// Dealloc(X9022)
JCudaTensor x71;
x71 = x61;
x71.free();

// val X9024 = BatchNorm(2a2_b_bn)(X9023,2a2_b_bn_scale,2a2_b_bn_bias)
JCudaTensor x72;
JCudaTensor x73, x74, x75;
x73 = x64;
x74 = x76;
x75 = x77;
x72 = x78.forward_inference(x73, x74, x75);

// Dealloc(X9023)
JCudaTensor x79;
x79 = x64;
x79.free();

// val X9025 = ReLU()(X9024)
JCudaTensor x80;
JCudaTensor x81;
x81 = x72;
x80 = x63.forward(x81);

// val X9026 = Convolv(1,0)(X9025,2a2_c_cv_W,2a2_c_cv_B)
JCudaTensor x82;
JCudaTensor x83, x84, x85;
x83 = x80;
x84 = x86;
x85 = x87;
x82 = x36.forward(x83, x84, x85);

// Dealloc(X9025)
JCudaTensor x88;
x88 = x80;
x88.free();

// val X9027 = BatchNorm(2a2_c_bn)(X9026,2a2_c_bn_scale,2a2_c_bn_bias)
JCudaTensor x89;
JCudaTensor x90, x91, x92;
x90 = x82;
x91 = x93;
x92 = x94;
x89 = x95.forward_inference(x90, x91, x92);

// Dealloc(X9026)
JCudaTensor x96;
x96 = x82;
x96.free();

// val X9019 = ReLU()(X9018)
JCudaTensor x97;
JCudaTensor x98;
x98 = x45;
x97 = x99.forward(x98);

// val X9028 = ReLU()(X9027)
JCudaTensor x100;
JCudaTensor x101;
x101 = x89;
x100 = x99.forward(x101);

// val X9029 = (X9019 + X9028)
JCudaTensor x102;
JCudaTensor x103, x104;
x103 = x97;
x104 = x100;
x102 = x103.plus_i(x104);

// Dealloc(X9028)
JCudaTensor x105;
x105 = x100;
x105.free();

// val X9030 = ReLU()(X9029)
JCudaTensor x106;
JCudaTensor x107;
x107 = x102;
x106 = x99.forward(x107);

// val X9031 = Convolv(1,0)(X9030,2b_a_cv_W,2b_a_cv_B)
JCudaTensor x108;
JCudaTensor x109, x110, x111;
x109 = x106;
x110 = x112;
x111 = x113;
x108 = x114.forward(x109, x110, x111);

// val X9032 = BatchNorm(2b_a_bn)(X9031,2b_a_bn_scale,2b_a_bn_bias)
JCudaTensor x115;
JCudaTensor x116, x117, x118;
x116 = x108;
x117 = x119;
x118 = x120;
x115 = x121.forward_inference(x116, x117, x118);

// Dealloc(X9031)
JCudaTensor x122;
x122 = x108;
x122.free();

// val X9033 = ReLU()(X9032)
JCudaTensor x123;
JCudaTensor x124;
x124 = x115;
x123 = x63.forward(x124);

// val X9034 = Convolv(1,1)(X9033,2b_b_cv_W,2b_b_cv_B)
JCudaTensor x125;
JCudaTensor x126, x127, x128;
x126 = x123;
x127 = x129;
x128 = x130;
x125 = x70.forward(x126, x127, x128);

// Dealloc(X9033)
JCudaTensor x131;
x131 = x123;
x131.free();

// val X9035 = BatchNorm(2b_b_bn)(X9034,2b_b_bn_scale,2b_b_bn_bias)
JCudaTensor x132;
JCudaTensor x133, x134, x135;
x133 = x125;
x134 = x136;
x135 = x137;
x132 = x138.forward_inference(x133, x134, x135);

// Dealloc(X9034)
JCudaTensor x139;
x139 = x125;
x139.free();

// val X9036 = ReLU()(X9035)
JCudaTensor x140;
JCudaTensor x141;
x141 = x132;
x140 = x63.forward(x141);

// val X9037 = Convolv(1,0)(X9036,2b_c_cv_W,2b_c_cv_B)
JCudaTensor x142;
JCudaTensor x143, x144, x145;
x143 = x140;
x144 = x146;
x145 = x147;
x142 = x36.forward(x143, x144, x145);

// Dealloc(X9036)
JCudaTensor x148;
x148 = x140;
x148.free();

// val X9038 = BatchNorm(2b_c_bn)(X9037,2b_c_bn_scale,2b_c_bn_bias)
JCudaTensor x149;
JCudaTensor x150, x151, x152;
x150 = x142;
x151 = x153;
x152 = x154;
x149 = x155.forward_inference(x150, x151, x152);

// Dealloc(X9037)
JCudaTensor x156;
x156 = x142;
x156.free();

// val X9039 = ReLU()(X9038)
JCudaTensor x157;
JCudaTensor x158;
x158 = x149;
x157 = x99.forward(x158);

// val X9040 = (X9039 + X9030)
JCudaTensor x159;
JCudaTensor x160, x161;
x160 = x157;
x161 = x106;
x159 = x160.plus_i(x161);

// Dealloc(X9030)
JCudaTensor x162;
x162 = x106;
x162.free();

// val X9041 = ReLU()(X9040)
JCudaTensor x163;
JCudaTensor x164;
x164 = x159;
x163 = x99.forward(x164);

// val X9042 = Convolv(1,0)(X9041,2c_a_cv_W,2c_a_cv_B)
JCudaTensor x165;
JCudaTensor x166, x167, x168;
x166 = x163;
x167 = x169;
x168 = x170;
x165 = x114.forward(x166, x167, x168);

// val X9043 = BatchNorm(2c_a_bn)(X9042,2c_a_bn_scale,2c_a_bn_bias)
JCudaTensor x171;
JCudaTensor x172, x173, x174;
x172 = x165;
x173 = x175;
x174 = x176;
x171 = x177.forward_inference(x172, x173, x174);

// Dealloc(X9042)
JCudaTensor x178;
x178 = x165;
x178.free();

// val X9044 = ReLU()(X9043)
JCudaTensor x179;
JCudaTensor x180;
x180 = x171;
x179 = x63.forward(x180);

// val X9045 = Convolv(1,1)(X9044,2c_b_cv_W,2c_b_cv_B)
JCudaTensor x181;
JCudaTensor x182, x183, x184;
x182 = x179;
x183 = x185;
x184 = x186;
x181 = x70.forward(x182, x183, x184);

// Dealloc(X9044)
JCudaTensor x187;
x187 = x179;
x187.free();

// val X9046 = BatchNorm(2c_b_bn)(X9045,2c_b_bn_scale,2c_b_bn_bias)
JCudaTensor x188;
JCudaTensor x189, x190, x191;
x189 = x181;
x190 = x192;
x191 = x193;
x188 = x194.forward_inference(x189, x190, x191);

// Dealloc(X9045)
JCudaTensor x195;
x195 = x181;
x195.free();

// val X9047 = ReLU()(X9046)
JCudaTensor x196;
JCudaTensor x197;
x197 = x188;
x196 = x63.forward(x197);

// val X9048 = Convolv(1,0)(X9047,2c_c_cv_W,2c_c_cv_B)
JCudaTensor x198;
JCudaTensor x199, x200, x201;
x199 = x196;
x200 = x202;
x201 = x203;
x198 = x36.forward(x199, x200, x201);

// Dealloc(X9047)
JCudaTensor x204;
x204 = x196;
x204.free();

// val X9049 = BatchNorm(2c_c_bn)(X9048,2c_c_bn_scale,2c_c_bn_bias)
JCudaTensor x205;
JCudaTensor x206, x207, x208;
x206 = x198;
x207 = x209;
x208 = x210;
x205 = x211.forward_inference(x206, x207, x208);

// Dealloc(X9048)
JCudaTensor x212;
x212 = x198;
x212.free();

// val X9050 = ReLU()(X9049)
JCudaTensor x213;
JCudaTensor x214;
x214 = x205;
x213 = x99.forward(x214);

// val X9051 = (X9050 + X9041)
JCudaTensor x215;
JCudaTensor x216, x217;
x216 = x213;
x217 = x163;
x215 = x216.plus_i(x217);

// Dealloc(X9041)
JCudaTensor x218;
x218 = x163;
x218.free();

// val X9052 = ReLU()(X9051)
JCudaTensor x219;
JCudaTensor x220;
x220 = x215;
x219 = x99.forward(x220);

// val X9053 = Convolv(2,0)(X9052,3a1_cv_W,3a1_cv_B)
JCudaTensor x221;
JCudaTensor x222, x223, x224;
x222 = x219;
x223 = x225;
x224 = x226;
x221 = x227.forward(x222, x223, x224);

// val X9056 = Convolv(2,0)(X9052,3a2_a_cv_W,3a2_a_cv_B)
JCudaTensor x228;
JCudaTensor x229, x230, x231;
x229 = x219;
x230 = x232;
x231 = x233;
x228 = x234.forward(x229, x230, x231);

// Dealloc(X9052)
JCudaTensor x235;
x235 = x219;
x235.free();

// val X9057 = BatchNorm(3a2_a_bn)(X9056,3a2_a_bn_scale,3a2_a_bn_bias)
JCudaTensor x236;
JCudaTensor x237, x238, x239;
x237 = x228;
x238 = x240;
x239 = x241;
x236 = x242.forward_inference(x237, x238, x239);

// Dealloc(X9056)
JCudaTensor x243;
x243 = x228;
x243.free();

// val X9054 = BatchNorm(3a1_bn)(X9053,3a1_bn_scale,3a1_bn_bias)
JCudaTensor x244;
JCudaTensor x245, x246, x247;
x245 = x221;
x246 = x248;
x247 = x249;
x244 = x250.forward_inference(x245, x246, x247);

// Dealloc(X9053)
JCudaTensor x251;
x251 = x221;
x251.free();

// val X9058 = ReLU()(X9057)
JCudaTensor x252;
JCudaTensor x253;
x253 = x236;
x252 = x254.forward(x253);

// val X9059 = Convolv(1,1)(X9058,3a2_b_cv_W,3a2_b_cv_B)
JCudaTensor x255;
JCudaTensor x256, x257, x258;
x256 = x252;
x257 = x259;
x258 = x260;
x255 = x261.forward(x256, x257, x258);

// Dealloc(X9058)
JCudaTensor x262;
x262 = x252;
x262.free();

// val X9060 = BatchNorm(3a2_b_bn)(X9059,3a2_b_bn_scale,3a2_b_bn_bias)
JCudaTensor x263;
JCudaTensor x264, x265, x266;
x264 = x255;
x265 = x267;
x266 = x268;
x263 = x269.forward_inference(x264, x265, x266);

// Dealloc(X9059)
JCudaTensor x270;
x270 = x255;
x270.free();

// val X9061 = ReLU()(X9060)
JCudaTensor x271;
JCudaTensor x272;
x272 = x263;
x271 = x254.forward(x272);

// val X9062 = Convolv(1,0)(X9061,3a2_c_cv_W,3a2_c_cv_B)
JCudaTensor x273;
JCudaTensor x274, x275, x276;
x274 = x271;
x275 = x277;
x276 = x278;
x273 = x279.forward(x274, x275, x276);

// Dealloc(X9061)
JCudaTensor x280;
x280 = x271;
x280.free();

// val X9063 = BatchNorm(3a2_c_bn)(X9062,3a2_c_bn_scale,3a2_c_bn_bias)
JCudaTensor x281;
JCudaTensor x282, x283, x284;
x282 = x273;
x283 = x285;
x284 = x286;
x281 = x287.forward_inference(x282, x283, x284);

// Dealloc(X9062)
JCudaTensor x288;
x288 = x273;
x288.free();

// val X9055 = ReLU()(X9054)
JCudaTensor x289;
JCudaTensor x290;
x290 = x244;
x289 = x291.forward(x290);

// val X9064 = ReLU()(X9063)
JCudaTensor x292;
JCudaTensor x293;
x293 = x281;
x292 = x291.forward(x293);

// val X9065 = (X9055 + X9064)
JCudaTensor x294;
JCudaTensor x295, x296;
x295 = x289;
x296 = x292;
x294 = x295.plus_i(x296);

// Dealloc(X9064)
JCudaTensor x297;
x297 = x292;
x297.free();

// val X9066 = ReLU()(X9065)
JCudaTensor x298;
JCudaTensor x299;
x299 = x294;
x298 = x291.forward(x299);

// val X9067 = Convolv(1,0)(X9066,3b_a_cv_W,3b_a_cv_B)
JCudaTensor x300;
JCudaTensor x301, x302, x303;
x301 = x298;
x302 = x304;
x303 = x305;
x300 = x306.forward(x301, x302, x303);

// val X9068 = BatchNorm(3b_a_bn)(X9067,3b_a_bn_scale,3b_a_bn_bias)
JCudaTensor x307;
JCudaTensor x308, x309, x310;
x308 = x300;
x309 = x311;
x310 = x312;
x307 = x313.forward_inference(x308, x309, x310);

// Dealloc(X9067)
JCudaTensor x314;
x314 = x300;
x314.free();

// val X9069 = ReLU()(X9068)
JCudaTensor x315;
JCudaTensor x316;
x316 = x307;
x315 = x254.forward(x316);

// val X9070 = Convolv(1,1)(X9069,3b_b_cv_W,3b_b_cv_B)
JCudaTensor x317;
JCudaTensor x318, x319, x320;
x318 = x315;
x319 = x321;
x320 = x322;
x317 = x261.forward(x318, x319, x320);

// Dealloc(X9069)
JCudaTensor x323;
x323 = x315;
x323.free();

// val X9071 = BatchNorm(3b_b_bn)(X9070,3b_b_bn_scale,3b_b_bn_bias)
JCudaTensor x324;
JCudaTensor x325, x326, x327;
x325 = x317;
x326 = x328;
x327 = x329;
x324 = x330.forward_inference(x325, x326, x327);

// Dealloc(X9070)
JCudaTensor x331;
x331 = x317;
x331.free();

// val X9072 = ReLU()(X9071)
JCudaTensor x332;
JCudaTensor x333;
x333 = x324;
x332 = x254.forward(x333);

// val X9073 = Convolv(1,0)(X9072,3b_c_cv_W,3b_c_cv_B)
JCudaTensor x334;
JCudaTensor x335, x336, x337;
x335 = x332;
x336 = x338;
x337 = x339;
x334 = x279.forward(x335, x336, x337);

// Dealloc(X9072)
JCudaTensor x340;
x340 = x332;
x340.free();

// val X9074 = BatchNorm(3b_c_bn)(X9073,3b_c_bn_scale,3b_c_bn_bias)
JCudaTensor x341;
JCudaTensor x342, x343, x344;
x342 = x334;
x343 = x345;
x344 = x346;
x341 = x347.forward_inference(x342, x343, x344);

// Dealloc(X9073)
JCudaTensor x348;
x348 = x334;
x348.free();

// val X9075 = ReLU()(X9074)
JCudaTensor x349;
JCudaTensor x350;
x350 = x341;
x349 = x291.forward(x350);

// val X9076 = (X9075 + X9066)
JCudaTensor x351;
JCudaTensor x352, x353;
x352 = x349;
x353 = x298;
x351 = x352.plus_i(x353);

// Dealloc(X9066)
JCudaTensor x354;
x354 = x298;
x354.free();

// val X9077 = ReLU()(X9076)
JCudaTensor x355;
JCudaTensor x356;
x356 = x351;
x355 = x291.forward(x356);

// val X9078 = Convolv(1,0)(X9077,3c_a_cv_W,3c_a_cv_B)
JCudaTensor x357;
JCudaTensor x358, x359, x360;
x358 = x355;
x359 = x361;
x360 = x362;
x357 = x306.forward(x358, x359, x360);

// val X9079 = BatchNorm(3c_a_bn)(X9078,3c_a_bn_scale,3c_a_bn_bias)
JCudaTensor x363;
JCudaTensor x364, x365, x366;
x364 = x357;
x365 = x367;
x366 = x368;
x363 = x369.forward_inference(x364, x365, x366);

// Dealloc(X9078)
JCudaTensor x370;
x370 = x357;
x370.free();

// val X9080 = ReLU()(X9079)
JCudaTensor x371;
JCudaTensor x372;
x372 = x363;
x371 = x254.forward(x372);

// val X9081 = Convolv(1,1)(X9080,3c_b_cv_W,3c_b_cv_B)
JCudaTensor x373;
JCudaTensor x374, x375, x376;
x374 = x371;
x375 = x377;
x376 = x378;
x373 = x261.forward(x374, x375, x376);

// Dealloc(X9080)
JCudaTensor x379;
x379 = x371;
x379.free();

// val X9082 = BatchNorm(3c_b_bn)(X9081,3c_b_bn_scale,3c_b_bn_bias)
JCudaTensor x380;
JCudaTensor x381, x382, x383;
x381 = x373;
x382 = x384;
x383 = x385;
x380 = x386.forward_inference(x381, x382, x383);

// Dealloc(X9081)
JCudaTensor x387;
x387 = x373;
x387.free();

// val X9083 = ReLU()(X9082)
JCudaTensor x388;
JCudaTensor x389;
x389 = x380;
x388 = x254.forward(x389);

// val X9084 = Convolv(1,0)(X9083,3c_c_cv_W,3c_c_cv_B)
JCudaTensor x390;
JCudaTensor x391, x392, x393;
x391 = x388;
x392 = x394;
x393 = x395;
x390 = x279.forward(x391, x392, x393);

// Dealloc(X9083)
JCudaTensor x396;
x396 = x388;
x396.free();

// val X9085 = BatchNorm(3c_c_bn)(X9084,3c_c_bn_scale,3c_c_bn_bias)
JCudaTensor x397;
JCudaTensor x398, x399, x400;
x398 = x390;
x399 = x401;
x400 = x402;
x397 = x403.forward_inference(x398, x399, x400);

// Dealloc(X9084)
JCudaTensor x404;
x404 = x390;
x404.free();

// val X9086 = ReLU()(X9085)
JCudaTensor x405;
JCudaTensor x406;
x406 = x397;
x405 = x291.forward(x406);

// val X9087 = (X9086 + X9077)
JCudaTensor x407;
JCudaTensor x408, x409;
x408 = x405;
x409 = x355;
x407 = x408.plus_i(x409);

// Dealloc(X9077)
JCudaTensor x410;
x410 = x355;
x410.free();

// val X9088 = ReLU()(X9087)
JCudaTensor x411;
JCudaTensor x412;
x412 = x407;
x411 = x291.forward(x412);

// val X9089 = Convolv(1,0)(X9088,3d_a_cv_W,3d_a_cv_B)
JCudaTensor x413;
JCudaTensor x414, x415, x416;
x414 = x411;
x415 = x417;
x416 = x418;
x413 = x306.forward(x414, x415, x416);

// val X9090 = BatchNorm(3d_a_bn)(X9089,3d_a_bn_scale,3d_a_bn_bias)
JCudaTensor x419;
JCudaTensor x420, x421, x422;
x420 = x413;
x421 = x423;
x422 = x424;
x419 = x425.forward_inference(x420, x421, x422);

// Dealloc(X9089)
JCudaTensor x426;
x426 = x413;
x426.free();

// val X9091 = ReLU()(X9090)
JCudaTensor x427;
JCudaTensor x428;
x428 = x419;
x427 = x254.forward(x428);

// val X9092 = Convolv(1,1)(X9091,3d_b_cv_W,3d_b_cv_B)
JCudaTensor x429;
JCudaTensor x430, x431, x432;
x430 = x427;
x431 = x433;
x432 = x434;
x429 = x261.forward(x430, x431, x432);

// Dealloc(X9091)
JCudaTensor x435;
x435 = x427;
x435.free();

// val X9093 = BatchNorm(3d_b_bn)(X9092,3d_b_bn_scale,3d_b_bn_bias)
JCudaTensor x436;
JCudaTensor x437, x438, x439;
x437 = x429;
x438 = x440;
x439 = x441;
x436 = x442.forward_inference(x437, x438, x439);

// Dealloc(X9092)
JCudaTensor x443;
x443 = x429;
x443.free();

// val X9094 = ReLU()(X9093)
JCudaTensor x444;
JCudaTensor x445;
x445 = x436;
x444 = x254.forward(x445);

// val X9095 = Convolv(1,0)(X9094,3d_c_cv_W,3d_c_cv_B)
JCudaTensor x446;
JCudaTensor x447, x448, x449;
x447 = x444;
x448 = x450;
x449 = x451;
x446 = x279.forward(x447, x448, x449);

// Dealloc(X9094)
JCudaTensor x452;
x452 = x444;
x452.free();

// val X9096 = BatchNorm(3d_c_bn)(X9095,3d_c_bn_scale,3d_c_bn_bias)
JCudaTensor x453;
JCudaTensor x454, x455, x456;
x454 = x446;
x455 = x457;
x456 = x458;
x453 = x459.forward_inference(x454, x455, x456);

// Dealloc(X9095)
JCudaTensor x460;
x460 = x446;
x460.free();

// val X9097 = ReLU()(X9096)
JCudaTensor x461;
JCudaTensor x462;
x462 = x453;
x461 = x291.forward(x462);

// val X9098 = (X9097 + X9088)
JCudaTensor x463;
JCudaTensor x464, x465;
x464 = x461;
x465 = x411;
x463 = x464.plus_i(x465);

// Dealloc(X9088)
JCudaTensor x466;
x466 = x411;
x466.free();

// val X9099 = ReLU()(X9098)
JCudaTensor x467;
JCudaTensor x468;
x468 = x463;
x467 = x291.forward(x468);

// val X9103 = Convolv(2,0)(X9099,4a2_a_cv_W,4a2_a_cv_B)
JCudaTensor x469;
JCudaTensor x470, x471, x472;
x470 = x467;
x471 = x473;
x472 = x474;
x469 = x475.forward(x470, x471, x472);

// val X9100 = Convolv(2,0)(X9099,4a1_cv_W,4a1_cv_B)
JCudaTensor x476;
JCudaTensor x477, x478, x479;
x477 = x467;
x478 = x480;
x479 = x481;
x476 = x482.forward(x477, x478, x479);

// Dealloc(X9099)
JCudaTensor x483;
x483 = x467;
x483.free();

// val X9101 = BatchNorm(4a1_bn)(X9100,4a1_bn_scale,4a1_bn_bias)
JCudaTensor x484;
JCudaTensor x485, x486, x487;
x485 = x476;
x486 = x488;
x487 = x489;
x484 = x490.forward_inference(x485, x486, x487);

// Dealloc(X9100)
JCudaTensor x491;
x491 = x476;
x491.free();

// val X9104 = BatchNorm(4a2_a_bn)(X9103,4a2_a_bn_scale,4a2_a_bn_bias)
JCudaTensor x492;
JCudaTensor x493, x494, x495;
x493 = x469;
x494 = x496;
x495 = x497;
x492 = x498.forward_inference(x493, x494, x495);

// Dealloc(X9103)
JCudaTensor x499;
x499 = x469;
x499.free();

// val X9105 = ReLU()(X9104)
JCudaTensor x500;
JCudaTensor x501;
x501 = x492;
x500 = x502.forward(x501);

// val X9106 = Convolv(1,1)(X9105,4a2_b_cv_W,4a2_b_cv_B)
JCudaTensor x503;
JCudaTensor x504, x505, x506;
x504 = x500;
x505 = x507;
x506 = x508;
x503 = x509.forward(x504, x505, x506);

// Dealloc(X9105)
JCudaTensor x510;
x510 = x500;
x510.free();

// val X9107 = BatchNorm(4a2_b_bn)(X9106,4a2_b_bn_scale,4a2_b_bn_bias)
JCudaTensor x511;
JCudaTensor x512, x513, x514;
x512 = x503;
x513 = x515;
x514 = x516;
x511 = x517.forward_inference(x512, x513, x514);

// Dealloc(X9106)
JCudaTensor x518;
x518 = x503;
x518.free();

// val X9108 = ReLU()(X9107)
JCudaTensor x519;
JCudaTensor x520;
x520 = x511;
x519 = x502.forward(x520);

// val X9109 = Convolv(1,0)(X9108,4a2_c_cv_W,4a2_c_cv_B)
JCudaTensor x521;
JCudaTensor x522, x523, x524;
x522 = x519;
x523 = x525;
x524 = x526;
x521 = x527.forward(x522, x523, x524);

// Dealloc(X9108)
JCudaTensor x528;
x528 = x519;
x528.free();

// val X9110 = BatchNorm(4a2_c_bn)(X9109,4a2_c_bn_scale,4a2_c_bn_bias)
JCudaTensor x529;
JCudaTensor x530, x531, x532;
x530 = x521;
x531 = x533;
x532 = x534;
x529 = x535.forward_inference(x530, x531, x532);

// Dealloc(X9109)
JCudaTensor x536;
x536 = x521;
x536.free();

// val X9102 = ReLU()(X9101)
JCudaTensor x537;
JCudaTensor x538;
x538 = x484;
x537 = x539.forward(x538);

// val X9111 = ReLU()(X9110)
JCudaTensor x540;
JCudaTensor x541;
x541 = x529;
x540 = x539.forward(x541);

// val X9112 = (X9102 + X9111)
JCudaTensor x542;
JCudaTensor x543, x544;
x543 = x537;
x544 = x540;
x542 = x543.plus_i(x544);

// Dealloc(X9111)
JCudaTensor x545;
x545 = x540;
x545.free();

// val X9113 = ReLU()(X9112)
JCudaTensor x546;
JCudaTensor x547;
x547 = x542;
x546 = x539.forward(x547);

// val X9114 = Convolv(1,0)(X9113,4b_a_cv_W,4b_a_cv_B)
JCudaTensor x548;
JCudaTensor x549, x550, x551;
x549 = x546;
x550 = x552;
x551 = x553;
x548 = x554.forward(x549, x550, x551);

// val X9115 = BatchNorm(4b_a_bn)(X9114,4b_a_bn_scale,4b_a_bn_bias)
JCudaTensor x555;
JCudaTensor x556, x557, x558;
x556 = x548;
x557 = x559;
x558 = x560;
x555 = x561.forward_inference(x556, x557, x558);

// Dealloc(X9114)
JCudaTensor x562;
x562 = x548;
x562.free();

// val X9116 = ReLU()(X9115)
JCudaTensor x563;
JCudaTensor x564;
x564 = x555;
x563 = x502.forward(x564);

// val X9117 = Convolv(1,1)(X9116,4b_b_cv_W,4b_b_cv_B)
JCudaTensor x565;
JCudaTensor x566, x567, x568;
x566 = x563;
x567 = x569;
x568 = x570;
x565 = x509.forward(x566, x567, x568);

// Dealloc(X9116)
JCudaTensor x571;
x571 = x563;
x571.free();

// val X9118 = BatchNorm(4b_b_bn)(X9117,4b_b_bn_scale,4b_b_bn_bias)
JCudaTensor x572;
JCudaTensor x573, x574, x575;
x573 = x565;
x574 = x576;
x575 = x577;
x572 = x578.forward_inference(x573, x574, x575);

// Dealloc(X9117)
JCudaTensor x579;
x579 = x565;
x579.free();

// val X9119 = ReLU()(X9118)
JCudaTensor x580;
JCudaTensor x581;
x581 = x572;
x580 = x502.forward(x581);

// val X9120 = Convolv(1,0)(X9119,4b_c_cv_W,4b_c_cv_B)
JCudaTensor x582;
JCudaTensor x583, x584, x585;
x583 = x580;
x584 = x586;
x585 = x587;
x582 = x527.forward(x583, x584, x585);

// Dealloc(X9119)
JCudaTensor x588;
x588 = x580;
x588.free();

// val X9121 = BatchNorm(4b_c_bn)(X9120,4b_c_bn_scale,4b_c_bn_bias)
JCudaTensor x589;
JCudaTensor x590, x591, x592;
x590 = x582;
x591 = x593;
x592 = x594;
x589 = x595.forward_inference(x590, x591, x592);

// Dealloc(X9120)
JCudaTensor x596;
x596 = x582;
x596.free();

// val X9122 = ReLU()(X9121)
JCudaTensor x597;
JCudaTensor x598;
x598 = x589;
x597 = x539.forward(x598);

// val X9123 = (X9122 + X9113)
JCudaTensor x599;
JCudaTensor x600, x601;
x600 = x597;
x601 = x546;
x599 = x600.plus_i(x601);

// Dealloc(X9113)
JCudaTensor x602;
x602 = x546;
x602.free();

// val X9124 = ReLU()(X9123)
JCudaTensor x603;
JCudaTensor x604;
x604 = x599;
x603 = x539.forward(x604);

// val X9125 = Convolv(1,0)(X9124,4c_a_cv_W,4c_a_cv_B)
JCudaTensor x605;
JCudaTensor x606, x607, x608;
x606 = x603;
x607 = x609;
x608 = x610;
x605 = x554.forward(x606, x607, x608);

// val X9126 = BatchNorm(4c_a_bn)(X9125,4c_a_bn_scale,4c_a_bn_bias)
JCudaTensor x611;
JCudaTensor x612, x613, x614;
x612 = x605;
x613 = x615;
x614 = x616;
x611 = x617.forward_inference(x612, x613, x614);

// Dealloc(X9125)
JCudaTensor x618;
x618 = x605;
x618.free();

// val X9127 = ReLU()(X9126)
JCudaTensor x619;
JCudaTensor x620;
x620 = x611;
x619 = x502.forward(x620);

// val X9128 = Convolv(1,1)(X9127,4c_b_cv_W,4c_b_cv_B)
JCudaTensor x621;
JCudaTensor x622, x623, x624;
x622 = x619;
x623 = x625;
x624 = x626;
x621 = x509.forward(x622, x623, x624);

// Dealloc(X9127)
JCudaTensor x627;
x627 = x619;
x627.free();

// val X9129 = BatchNorm(4c_b_bn)(X9128,4c_b_bn_scale,4c_b_bn_bias)
JCudaTensor x628;
JCudaTensor x629, x630, x631;
x629 = x621;
x630 = x632;
x631 = x633;
x628 = x634.forward_inference(x629, x630, x631);

// Dealloc(X9128)
JCudaTensor x635;
x635 = x621;
x635.free();

// val X9130 = ReLU()(X9129)
JCudaTensor x636;
JCudaTensor x637;
x637 = x628;
x636 = x502.forward(x637);

// val X9131 = Convolv(1,0)(X9130,4c_c_cv_W,4c_c_cv_B)
JCudaTensor x638;
JCudaTensor x639, x640, x641;
x639 = x636;
x640 = x642;
x641 = x643;
x638 = x527.forward(x639, x640, x641);

// Dealloc(X9130)
JCudaTensor x644;
x644 = x636;
x644.free();

// val X9132 = BatchNorm(4c_c_bn)(X9131,4c_c_bn_scale,4c_c_bn_bias)
JCudaTensor x645;
JCudaTensor x646, x647, x648;
x646 = x638;
x647 = x649;
x648 = x650;
x645 = x651.forward_inference(x646, x647, x648);

// Dealloc(X9131)
JCudaTensor x652;
x652 = x638;
x652.free();

// val X9133 = ReLU()(X9132)
JCudaTensor x653;
JCudaTensor x654;
x654 = x645;
x653 = x539.forward(x654);

// val X9134 = (X9133 + X9124)
JCudaTensor x655;
JCudaTensor x656, x657;
x656 = x653;
x657 = x603;
x655 = x656.plus_i(x657);

// Dealloc(X9124)
JCudaTensor x658;
x658 = x603;
x658.free();

// val X9135 = ReLU()(X9134)
JCudaTensor x659;
JCudaTensor x660;
x660 = x655;
x659 = x539.forward(x660);

// val X9136 = Convolv(1,0)(X9135,4d_a_cv_W,4d_a_cv_B)
JCudaTensor x661;
JCudaTensor x662, x663, x664;
x662 = x659;
x663 = x665;
x664 = x666;
x661 = x554.forward(x662, x663, x664);

// val X9137 = BatchNorm(4d_a_bn)(X9136,4d_a_bn_scale,4d_a_bn_bias)
JCudaTensor x667;
JCudaTensor x668, x669, x670;
x668 = x661;
x669 = x671;
x670 = x672;
x667 = x673.forward_inference(x668, x669, x670);

// Dealloc(X9136)
JCudaTensor x674;
x674 = x661;
x674.free();

// val X9138 = ReLU()(X9137)
JCudaTensor x675;
JCudaTensor x676;
x676 = x667;
x675 = x502.forward(x676);

// val X9139 = Convolv(1,1)(X9138,4d_b_cv_W,4d_b_cv_B)
JCudaTensor x677;
JCudaTensor x678, x679, x680;
x678 = x675;
x679 = x681;
x680 = x682;
x677 = x509.forward(x678, x679, x680);

// Dealloc(X9138)
JCudaTensor x683;
x683 = x675;
x683.free();

// val X9140 = BatchNorm(4d_b_bn)(X9139,4d_b_bn_scale,4d_b_bn_bias)
JCudaTensor x684;
JCudaTensor x685, x686, x687;
x685 = x677;
x686 = x688;
x687 = x689;
x684 = x690.forward_inference(x685, x686, x687);

// Dealloc(X9139)
JCudaTensor x691;
x691 = x677;
x691.free();

// val X9141 = ReLU()(X9140)
JCudaTensor x692;
JCudaTensor x693;
x693 = x684;
x692 = x502.forward(x693);

// val X9142 = Convolv(1,0)(X9141,4d_c_cv_W,4d_c_cv_B)
JCudaTensor x694;
JCudaTensor x695, x696, x697;
x695 = x692;
x696 = x698;
x697 = x699;
x694 = x527.forward(x695, x696, x697);

// Dealloc(X9141)
JCudaTensor x700;
x700 = x692;
x700.free();

// val X9143 = BatchNorm(4d_c_bn)(X9142,4d_c_bn_scale,4d_c_bn_bias)
JCudaTensor x701;
JCudaTensor x702, x703, x704;
x702 = x694;
x703 = x705;
x704 = x706;
x701 = x707.forward_inference(x702, x703, x704);

// Dealloc(X9142)
JCudaTensor x708;
x708 = x694;
x708.free();

// val X9144 = ReLU()(X9143)
JCudaTensor x709;
JCudaTensor x710;
x710 = x701;
x709 = x539.forward(x710);

// val X9145 = (X9144 + X9135)
JCudaTensor x711;
JCudaTensor x712, x713;
x712 = x709;
x713 = x659;
x711 = x712.plus_i(x713);

// Dealloc(X9135)
JCudaTensor x714;
x714 = x659;
x714.free();

// val X9146 = ReLU()(X9145)
JCudaTensor x715;
JCudaTensor x716;
x716 = x711;
x715 = x539.forward(x716);

// val X9147 = Convolv(1,0)(X9146,4e_a_cv_W,4e_a_cv_B)
JCudaTensor x717;
JCudaTensor x718, x719, x720;
x718 = x715;
x719 = x721;
x720 = x722;
x717 = x554.forward(x718, x719, x720);

// val X9148 = BatchNorm(4e_a_bn)(X9147,4e_a_bn_scale,4e_a_bn_bias)
JCudaTensor x723;
JCudaTensor x724, x725, x726;
x724 = x717;
x725 = x727;
x726 = x728;
x723 = x729.forward_inference(x724, x725, x726);

// Dealloc(X9147)
JCudaTensor x730;
x730 = x717;
x730.free();

// val X9149 = ReLU()(X9148)
JCudaTensor x731;
JCudaTensor x732;
x732 = x723;
x731 = x502.forward(x732);

// val X9150 = Convolv(1,1)(X9149,4e_b_cv_W,4e_b_cv_B)
JCudaTensor x733;
JCudaTensor x734, x735, x736;
x734 = x731;
x735 = x737;
x736 = x738;
x733 = x509.forward(x734, x735, x736);

// Dealloc(X9149)
JCudaTensor x739;
x739 = x731;
x739.free();

// val X9151 = BatchNorm(4e_b_bn)(X9150,4e_b_bn_scale,4e_b_bn_bias)
JCudaTensor x740;
JCudaTensor x741, x742, x743;
x741 = x733;
x742 = x744;
x743 = x745;
x740 = x746.forward_inference(x741, x742, x743);

// Dealloc(X9150)
JCudaTensor x747;
x747 = x733;
x747.free();

// val X9152 = ReLU()(X9151)
JCudaTensor x748;
JCudaTensor x749;
x749 = x740;
x748 = x502.forward(x749);

// val X9153 = Convolv(1,0)(X9152,4e_c_cv_W,4e_c_cv_B)
JCudaTensor x750;
JCudaTensor x751, x752, x753;
x751 = x748;
x752 = x754;
x753 = x755;
x750 = x527.forward(x751, x752, x753);

// Dealloc(X9152)
JCudaTensor x756;
x756 = x748;
x756.free();

// val X9154 = BatchNorm(4e_c_bn)(X9153,4e_c_bn_scale,4e_c_bn_bias)
JCudaTensor x757;
JCudaTensor x758, x759, x760;
x758 = x750;
x759 = x761;
x760 = x762;
x757 = x763.forward_inference(x758, x759, x760);

// Dealloc(X9153)
JCudaTensor x764;
x764 = x750;
x764.free();

// val X9155 = ReLU()(X9154)
JCudaTensor x765;
JCudaTensor x766;
x766 = x757;
x765 = x539.forward(x766);

// val X9156 = (X9155 + X9146)
JCudaTensor x767;
JCudaTensor x768, x769;
x768 = x765;
x769 = x715;
x767 = x768.plus_i(x769);

// Dealloc(X9146)
JCudaTensor x770;
x770 = x715;
x770.free();

// val X9157 = ReLU()(X9156)
JCudaTensor x771;
JCudaTensor x772;
x772 = x767;
x771 = x539.forward(x772);

// val X9158 = Convolv(1,0)(X9157,4f_a_cv_W,4f_a_cv_B)
JCudaTensor x773;
JCudaTensor x774, x775, x776;
x774 = x771;
x775 = x777;
x776 = x778;
x773 = x554.forward(x774, x775, x776);

// val X9159 = BatchNorm(4f_a_bn)(X9158,4f_a_bn_scale,4f_a_bn_bias)
JCudaTensor x779;
JCudaTensor x780, x781, x782;
x780 = x773;
x781 = x783;
x782 = x784;
x779 = x785.forward_inference(x780, x781, x782);

// Dealloc(X9158)
JCudaTensor x786;
x786 = x773;
x786.free();

// val X9160 = ReLU()(X9159)
JCudaTensor x787;
JCudaTensor x788;
x788 = x779;
x787 = x502.forward(x788);

// val X9161 = Convolv(1,1)(X9160,4f_b_cv_W,4f_b_cv_B)
JCudaTensor x789;
JCudaTensor x790, x791, x792;
x790 = x787;
x791 = x793;
x792 = x794;
x789 = x509.forward(x790, x791, x792);

// Dealloc(X9160)
JCudaTensor x795;
x795 = x787;
x795.free();

// val X9162 = BatchNorm(4f_b_bn)(X9161,4f_b_bn_scale,4f_b_bn_bias)
JCudaTensor x796;
JCudaTensor x797, x798, x799;
x797 = x789;
x798 = x800;
x799 = x801;
x796 = x802.forward_inference(x797, x798, x799);

// Dealloc(X9161)
JCudaTensor x803;
x803 = x789;
x803.free();

// val X9163 = ReLU()(X9162)
JCudaTensor x804;
JCudaTensor x805;
x805 = x796;
x804 = x502.forward(x805);

// val X9164 = Convolv(1,0)(X9163,4f_c_cv_W,4f_c_cv_B)
JCudaTensor x806;
JCudaTensor x807, x808, x809;
x807 = x804;
x808 = x810;
x809 = x811;
x806 = x527.forward(x807, x808, x809);

// Dealloc(X9163)
JCudaTensor x812;
x812 = x804;
x812.free();

// val X9165 = BatchNorm(4f_c_bn)(X9164,4f_c_bn_scale,4f_c_bn_bias)
JCudaTensor x813;
JCudaTensor x814, x815, x816;
x814 = x806;
x815 = x817;
x816 = x818;
x813 = x819.forward_inference(x814, x815, x816);

// Dealloc(X9164)
JCudaTensor x820;
x820 = x806;
x820.free();

// val X9166 = ReLU()(X9165)
JCudaTensor x821;
JCudaTensor x822;
x822 = x813;
x821 = x539.forward(x822);

// val X9167 = (X9166 + X9157)
JCudaTensor x823;
JCudaTensor x824, x825;
x824 = x821;
x825 = x771;
x823 = x824.plus_i(x825);

// Dealloc(X9157)
JCudaTensor x826;
x826 = x771;
x826.free();

// val X9168 = ReLU()(X9167)
JCudaTensor x827;
JCudaTensor x828;
x828 = x823;
x827 = x539.forward(x828);

// val X9172 = Convolv(2,0)(X9168,5a2_a_cv_W,5a2_a_cv_B)
JCudaTensor x829;
JCudaTensor x830, x831, x832;
x830 = x827;
x831 = x833;
x832 = x834;
x829 = x835.forward(x830, x831, x832);

// val X9169 = Convolv(2,0)(X9168,5a1_cv_W,5a1_cv_B)
JCudaTensor x836;
JCudaTensor x837, x838, x839;
x837 = x827;
x838 = x840;
x839 = x841;
x836 = x842.forward(x837, x838, x839);

// Dealloc(X9168)
JCudaTensor x843;
x843 = x827;
x843.free();

// val X9170 = BatchNorm(5a1_bn)(X9169,5a1_bn_scale,5a1_bn_bias)
JCudaTensor x844;
JCudaTensor x845, x846, x847;
x845 = x836;
x846 = x848;
x847 = x849;
x844 = x850.forward_inference(x845, x846, x847);

// Dealloc(X9169)
JCudaTensor x851;
x851 = x836;
x851.free();

// val X9173 = BatchNorm(5a2_a_bn)(X9172,5a2_a_bn_scale,5a2_a_bn_bias)
JCudaTensor x852;
JCudaTensor x853, x854, x855;
x853 = x829;
x854 = x856;
x855 = x857;
x852 = x858.forward_inference(x853, x854, x855);

// Dealloc(X9172)
JCudaTensor x859;
x859 = x829;
x859.free();

// val X9174 = ReLU()(X9173)
JCudaTensor x860;
JCudaTensor x861;
x861 = x852;
x860 = x862.forward(x861);

// val X9175 = Convolv(1,1)(X9174,5a2_b_cv_W,5a2_b_cv_B)
JCudaTensor x863;
JCudaTensor x864, x865, x866;
x864 = x860;
x865 = x867;
x866 = x868;
x863 = x869.forward(x864, x865, x866);

// Dealloc(X9174)
JCudaTensor x870;
x870 = x860;
x870.free();

// val X9176 = BatchNorm(5a2_b_bn)(X9175,5a2_b_bn_scale,5a2_b_bn_bias)
JCudaTensor x871;
JCudaTensor x872, x873, x874;
x872 = x863;
x873 = x875;
x874 = x876;
x871 = x877.forward_inference(x872, x873, x874);

// Dealloc(X9175)
JCudaTensor x878;
x878 = x863;
x878.free();

// val X9177 = ReLU()(X9176)
JCudaTensor x879;
JCudaTensor x880;
x880 = x871;
x879 = x862.forward(x880);

// val X9178 = Convolv(1,0)(X9177,5a2_c_cv_W,5a2_c_cv_B)
JCudaTensor x881;
JCudaTensor x882, x883, x884;
x882 = x879;
x883 = x885;
x884 = x886;
x881 = x887.forward(x882, x883, x884);

// Dealloc(X9177)
JCudaTensor x888;
x888 = x879;
x888.free();

// val X9179 = BatchNorm(5a2_c_bn)(X9178,5a2_c_bn_scale,5a2_c_bn_bias)
JCudaTensor x889;
JCudaTensor x890, x891, x892;
x890 = x881;
x891 = x893;
x892 = x894;
x889 = x895.forward_inference(x890, x891, x892);

// Dealloc(X9178)
JCudaTensor x896;
x896 = x881;
x896.free();

// val X9171 = ReLU()(X9170)
JCudaTensor x897;
JCudaTensor x898;
x898 = x844;
x897 = x899.forward(x898);

// val X9180 = ReLU()(X9179)
JCudaTensor x900;
JCudaTensor x901;
x901 = x889;
x900 = x899.forward(x901);

// val X9181 = (X9171 + X9180)
JCudaTensor x902;
JCudaTensor x903, x904;
x903 = x897;
x904 = x900;
x902 = x903.plus_i(x904);

// Dealloc(X9180)
JCudaTensor x905;
x905 = x900;
x905.free();

// val X9182 = ReLU()(X9181)
JCudaTensor x906;
JCudaTensor x907;
x907 = x902;
x906 = x899.forward(x907);

// val X9183 = Convolv(1,0)(X9182,5b_a_cv_W,5b_a_cv_B)
JCudaTensor x908;
JCudaTensor x909, x910, x911;
x909 = x906;
x910 = x912;
x911 = x913;
x908 = x914.forward(x909, x910, x911);

// val X9184 = BatchNorm(5b_a_bn)(X9183,5b_a_bn_scale,5b_a_bn_bias)
JCudaTensor x915;
JCudaTensor x916, x917, x918;
x916 = x908;
x917 = x919;
x918 = x920;
x915 = x921.forward_inference(x916, x917, x918);

// Dealloc(X9183)
JCudaTensor x922;
x922 = x908;
x922.free();

// val X9185 = ReLU()(X9184)
JCudaTensor x923;
JCudaTensor x924;
x924 = x915;
x923 = x862.forward(x924);

// val X9186 = Convolv(1,1)(X9185,5b_b_cv_W,5b_b_cv_B)
JCudaTensor x925;
JCudaTensor x926, x927, x928;
x926 = x923;
x927 = x929;
x928 = x930;
x925 = x869.forward(x926, x927, x928);

// Dealloc(X9185)
JCudaTensor x931;
x931 = x923;
x931.free();

// val X9187 = BatchNorm(5b_b_bn)(X9186,5b_b_bn_scale,5b_b_bn_bias)
JCudaTensor x932;
JCudaTensor x933, x934, x935;
x933 = x925;
x934 = x936;
x935 = x937;
x932 = x938.forward_inference(x933, x934, x935);

// Dealloc(X9186)
JCudaTensor x939;
x939 = x925;
x939.free();

// val X9188 = ReLU()(X9187)
JCudaTensor x940;
JCudaTensor x941;
x941 = x932;
x940 = x862.forward(x941);

// val X9189 = Convolv(1,0)(X9188,5b_c_cv_W,5b_c_cv_B)
JCudaTensor x942;
JCudaTensor x943, x944, x945;
x943 = x940;
x944 = x946;
x945 = x947;
x942 = x887.forward(x943, x944, x945);

// Dealloc(X9188)
JCudaTensor x948;
x948 = x940;
x948.free();

// val X9190 = BatchNorm(5b_c_bn)(X9189,5b_c_bn_scale,5b_c_bn_bias)
JCudaTensor x949;
JCudaTensor x950, x951, x952;
x950 = x942;
x951 = x953;
x952 = x954;
x949 = x955.forward_inference(x950, x951, x952);

// Dealloc(X9189)
JCudaTensor x956;
x956 = x942;
x956.free();

// val X9191 = ReLU()(X9190)
JCudaTensor x957;
JCudaTensor x958;
x958 = x949;
x957 = x899.forward(x958);

// val X9192 = (X9191 + X9182)
JCudaTensor x959;
JCudaTensor x960, x961;
x960 = x957;
x961 = x906;
x959 = x960.plus_i(x961);

// Dealloc(X9182)
JCudaTensor x962;
x962 = x906;
x962.free();

// val X9193 = ReLU()(X9192)
JCudaTensor x963;
JCudaTensor x964;
x964 = x959;
x963 = x899.forward(x964);

// val X9194 = Convolv(1,0)(X9193,5c_a_cv_W,5c_a_cv_B)
JCudaTensor x965;
JCudaTensor x966, x967, x968;
x966 = x963;
x967 = x969;
x968 = x970;
x965 = x914.forward(x966, x967, x968);

// val X9195 = BatchNorm(5c_a_bn)(X9194,5c_a_bn_scale,5c_a_bn_bias)
JCudaTensor x971;
JCudaTensor x972, x973, x974;
x972 = x965;
x973 = x975;
x974 = x976;
x971 = x977.forward_inference(x972, x973, x974);

// Dealloc(X9194)
JCudaTensor x978;
x978 = x965;
x978.free();

// val X9196 = ReLU()(X9195)
JCudaTensor x979;
JCudaTensor x980;
x980 = x971;
x979 = x862.forward(x980);

// val X9197 = Convolv(1,1)(X9196,5c_b_cv_W,5c_b_cv_B)
JCudaTensor x981;
JCudaTensor x982, x983, x984;
x982 = x979;
x983 = x985;
x984 = x986;
x981 = x869.forward(x982, x983, x984);

// Dealloc(X9196)
JCudaTensor x987;
x987 = x979;
x987.free();

// val X9198 = BatchNorm(5c_b_bn)(X9197,5c_b_bn_scale,5c_b_bn_bias)
JCudaTensor x988;
JCudaTensor x989, x990, x991;
x989 = x981;
x990 = x992;
x991 = x993;
x988 = x994.forward_inference(x989, x990, x991);

// Dealloc(X9197)
JCudaTensor x995;
x995 = x981;
x995.free();

// val X9199 = ReLU()(X9198)
JCudaTensor x996;
JCudaTensor x997;
x997 = x988;
x996 = x862.forward(x997);

// val X9200 = Convolv(1,0)(X9199,5c_c_cv_W,5c_c_cv_B)
JCudaTensor x998;
JCudaTensor x999, x1000, x1001;
x999 = x996;
x1000 = x1002;
x1001 = x1003;
x998 = x887.forward(x999, x1000, x1001);

// Dealloc(X9199)
JCudaTensor x1004;
x1004 = x996;
x1004.free();

// val X9201 = BatchNorm(5c_c_bn)(X9200,5c_c_bn_scale,5c_c_bn_bias)
JCudaTensor x1005;
JCudaTensor x1006, x1007, x1008;
x1006 = x998;
x1007 = x1009;
x1008 = x1010;
x1005 = x1011.forward_inference(x1006, x1007, x1008);

// Dealloc(X9200)
JCudaTensor x1012;
x1012 = x998;
x1012.free();

// val X9202 = ReLU()(X9201)
JCudaTensor x1013;
JCudaTensor x1014;
x1014 = x1005;
x1013 = x899.forward(x1014);

// val X9203 = (X9202 + X9193)
JCudaTensor x1015;
JCudaTensor x1016, x1017;
x1016 = x1013;
x1017 = x963;
x1015 = x1016.plus_i(x1017);

// Dealloc(X9193)
JCudaTensor x1018;
x1018 = x963;
x1018.free();

// val X9204 = ReLU()(X9203)
JCudaTensor x1019;
JCudaTensor x1020;
x1020 = x1015;
x1019 = x899.forward(x1020);

// val X9205 = Pooling(7,1,0,false)(X9204)
JCudaTensor x1021;
JCudaTensor x1022;
x1022 = x1019;
x1021 = x1023.forward(x1022);

// Dealloc(X9204)
JCudaTensor x1024;
x1024 = x1019;
x1024.free();

// val X9206 = (X9205[1><3])(i1 | @) * (fc_W)(i2 | @)
JCudaTensor x1025;
JCudaMatrix x1026;
JCudaMatrix x1027;
JCudaTensor x1028;
JCudaTensor x1029;
x1029 = x1021;
x1028 = x1029.flatten(1, new int[]{2048, 1, 1});
x1026 = x1028.asMatrix(1, true);
JCudaTensor x1030;
x1030 = x1031;
x1027 = x1030.asMatrix(1, true);
x1025 = x1026.times(x1027);

// Dealloc(X9205)
JCudaTensor x1032;
x1032 = x1021;
x1032.free();

// val X9208 = (X9206 + (i1) => fc_B)
JCudaTensor x1033;
JCudaTensor x1034, x1035;
x1034 = x1025;
x1035 = x1036;
x1033 = x1035.copy(64, x1034);

// Prediction(X9208)
JCudaTensor x1037;
x1037 = x1033;
System.out.println(x3 + " inference " + java.util.Arrays.toString(x1037.prediction()));

// Dealloc(X9208)
JCudaTensor x1038;
x1038 = x1033;
x1038.free();

}
 
}

}