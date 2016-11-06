package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.tensor.*;
import deepdsl.util.*;


public class Rcnn {
static{ JCudaTensor.enableMemoryCache();}
// decay_1
static float decay_1 = 0.999995f;
// lrn_rate_1
static float lrn_rate_1 = -0.01f;
// momentum
static float momentum = 0.9f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/rcnn";
// test_data_path
static String test_data_path = "dataset/imagenet224/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 10;
// train_data_path
static String train_data_path = "dataset/imagenet224/ilsvrc12_train_lmdb";
// train_itr
static int train_itr = 100;

// (BatchNorm(),List(64, 1024, 14, 14))
static JCudnnBatchNorm x804 = new JCudnnBatchNorm(new int[]{64,1024,14,14});
// (BatchNorm(),List(64, 1024, 14, 14))
static JCudnnBatchNorm x748 = new JCudnnBatchNorm(new int[]{64,1024,14,14});
// (BatchNorm(),List(64, 1024, 14, 14))
static JCudnnBatchNorm x692 = new JCudnnBatchNorm(new int[]{64,1024,14,14});
// (BatchNorm(),List(64, 1024, 14, 14))
static JCudnnBatchNorm x636 = new JCudnnBatchNorm(new int[]{64,1024,14,14});
// (BatchNorm(),List(64, 1024, 14, 14))
static JCudnnBatchNorm x580 = new JCudnnBatchNorm(new int[]{64,1024,14,14});
// (BatchNorm(),List(64, 1024, 14, 14))
static JCudnnBatchNorm x521 = new JCudnnBatchNorm(new int[]{64,1024,14,14});
// (BatchNorm(),List(64, 1024, 14, 14))
static JCudnnBatchNorm x480 = new JCudnnBatchNorm(new int[]{64,1024,14,14});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x434 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x417 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x378 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x361 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x322 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x263 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x305 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 128, 28, 28))
static JCudnnBatchNorm x246 = new JCudnnBatchNorm(new int[]{64,128,28,28});
// (BatchNorm(),List(64, 2048, 7, 7))
static JCudnnBatchNorm x989 = new JCudnnBatchNorm(new int[]{64,2048,7,7});
// (BatchNorm(),List(64, 2048, 7, 7))
static JCudnnBatchNorm x933 = new JCudnnBatchNorm(new int[]{64,2048,7,7});
// (BatchNorm(),List(64, 2048, 7, 7))
static JCudnnBatchNorm x874 = new JCudnnBatchNorm(new int[]{64,2048,7,7});
// (BatchNorm(),List(64, 2048, 7, 7))
static JCudnnBatchNorm x840 = new JCudnnBatchNorm(new int[]{64,2048,7,7});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x787 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x770 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x731 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x714 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x675 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x658 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x619 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x602 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x563 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x504 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x546 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 14, 14))
static JCudnnBatchNorm x487 = new JCudnnBatchNorm(new int[]{64,256,14,14});
// (BatchNorm(),List(64, 256, 55, 55))
static JCudnnBatchNorm x210 = new JCudnnBatchNorm(new int[]{64,256,55,55});
// (BatchNorm(),List(64, 256, 55, 55))
static JCudnnBatchNorm x154 = new JCudnnBatchNorm(new int[]{64,256,55,55});
// (BatchNorm(),List(64, 256, 55, 55))
static JCudnnBatchNorm x95 = new JCudnnBatchNorm(new int[]{64,256,55,55});
// (BatchNorm(),List(64, 256, 55, 55))
static JCudnnBatchNorm x54 = new JCudnnBatchNorm(new int[]{64,256,55,55});
// (BatchNorm(),List(64, 512, 7, 7))
static JCudnnBatchNorm x972 = new JCudnnBatchNorm(new int[]{64,512,7,7});
// (BatchNorm(),List(64, 512, 7, 7))
static JCudnnBatchNorm x955 = new JCudnnBatchNorm(new int[]{64,512,7,7});
// (BatchNorm(),List(64, 512, 7, 7))
static JCudnnBatchNorm x916 = new JCudnnBatchNorm(new int[]{64,512,7,7});
// (BatchNorm(),List(64, 512, 7, 7))
static JCudnnBatchNorm x857 = new JCudnnBatchNorm(new int[]{64,512,7,7});
// (BatchNorm(),List(64, 512, 7, 7))
static JCudnnBatchNorm x899 = new JCudnnBatchNorm(new int[]{64,512,7,7});
// (BatchNorm(),List(64, 512, 7, 7))
static JCudnnBatchNorm x833 = new JCudnnBatchNorm(new int[]{64,512,7,7});
// (BatchNorm(),List(64, 512, 28, 28))
static JCudnnBatchNorm x451 = new JCudnnBatchNorm(new int[]{64,512,28,28});
// (BatchNorm(),List(64, 512, 28, 28))
static JCudnnBatchNorm x395 = new JCudnnBatchNorm(new int[]{64,512,28,28});
// (BatchNorm(),List(64, 512, 28, 28))
static JCudnnBatchNorm x339 = new JCudnnBatchNorm(new int[]{64,512,28,28});
// (BatchNorm(),List(64, 512, 28, 28))
static JCudnnBatchNorm x280 = new JCudnnBatchNorm(new int[]{64,512,28,28});
// (BatchNorm(),List(64, 512, 28, 28))
static JCudnnBatchNorm x239 = new JCudnnBatchNorm(new int[]{64,512,28,28});
// (BatchNorm(),List(64, 64, 55, 55))
static JCudnnBatchNorm x193 = new JCudnnBatchNorm(new int[]{64,64,55,55});
// (BatchNorm(),List(64, 64, 55, 55))
static JCudnnBatchNorm x176 = new JCudnnBatchNorm(new int[]{64,64,55,55});
// (BatchNorm(),List(64, 64, 55, 55))
static JCudnnBatchNorm x137 = new JCudnnBatchNorm(new int[]{64,64,55,55});
// (BatchNorm(),List(64, 64, 55, 55))
static JCudnnBatchNorm x78 = new JCudnnBatchNorm(new int[]{64,64,55,55});
// (BatchNorm(),List(64, 64, 55, 55))
static JCudnnBatchNorm x120 = new JCudnnBatchNorm(new int[]{64,64,55,55});
// (BatchNorm(),List(64, 64, 55, 55))
static JCudnnBatchNorm x61 = new JCudnnBatchNorm(new int[]{64,64,55,55});
// (BatchNorm(),List(64, 64, 112, 112))
static JCudnnBatchNorm x27 = new JCudnnBatchNorm(new int[]{64,64,112,112});
// (Convolv(1,0),List(64, 1024, 14, 14))
static JCudnnConvolution x797 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(64, 1024, 14, 14))
static JCudnnConvolution x741 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(64, 1024, 14, 14))
static JCudnnConvolution x685 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(64, 1024, 14, 14))
static JCudnnConvolution x629 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(64, 1024, 14, 14))
static JCudnnConvolution x573 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(64, 1024, 14, 14))
static JCudnnConvolution x514 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{1024,256,1,1},new int[]{1024}, 1, 0);
// (Convolv(1,0),List(64, 128, 28, 28))
static JCudnnConvolution x410 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{128,512,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(64, 128, 28, 28))
static JCudnnConvolution x354 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{128,512,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(64, 128, 28, 28))
static JCudnnConvolution x298 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{128,512,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(64, 2048, 7, 7))
static JCudnnConvolution x982 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{2048,512,1,1},new int[]{2048}, 1, 0);
// (Convolv(1,0),List(64, 2048, 7, 7))
static JCudnnConvolution x926 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{2048,512,1,1},new int[]{2048}, 1, 0);
// (Convolv(1,0),List(64, 2048, 7, 7))
static JCudnnConvolution x867 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{2048,512,1,1},new int[]{2048}, 1, 0);
// (Convolv(1,0),List(64, 256, 14, 14))
static JCudnnConvolution x763 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 14, 14))
static JCudnnConvolution x707 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 14, 14))
static JCudnnConvolution x651 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 14, 14))
static JCudnnConvolution x595 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 14, 14))
static JCudnnConvolution x539 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{256,1024,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 55, 55))
static JCudnnConvolution x203 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 55, 55))
static JCudnnConvolution x147 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 55, 55))
static JCudnnConvolution x88 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 256, 55, 55))
static JCudnnConvolution x40 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{256,64,1,1},new int[]{256}, 1, 0);
// (Convolv(1,0),List(64, 512, 7, 7))
static JCudnnConvolution x948 = new JCudnnConvolution(new int[]{64,2048,7,7},new int[]{512,2048,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(64, 512, 7, 7))
static JCudnnConvolution x892 = new JCudnnConvolution(new int[]{64,2048,7,7},new int[]{512,2048,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(64, 512, 28, 28))
static JCudnnConvolution x444 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(64, 512, 28, 28))
static JCudnnConvolution x388 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(64, 512, 28, 28))
static JCudnnConvolution x332 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(64, 512, 28, 28))
static JCudnnConvolution x273 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{512,128,1,1},new int[]{512}, 1, 0);
// (Convolv(1,0),List(64, 64, 55, 55))
static JCudnnConvolution x169 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(64, 64, 55, 55))
static JCudnnConvolution x113 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(64, 64, 55, 55))
static JCudnnConvolution x47 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,1,1},new int[]{64}, 1, 0);
// (Convolv(1,1),List(64, 128, 28, 28))
static JCudnnConvolution x427 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(64, 128, 28, 28))
static JCudnnConvolution x371 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(64, 128, 28, 28))
static JCudnnConvolution x315 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(64, 128, 28, 28))
static JCudnnConvolution x256 = new JCudnnConvolution(new int[]{64,128,28,28},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(64, 256, 14, 14))
static JCudnnConvolution x780 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 256, 14, 14))
static JCudnnConvolution x724 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 256, 14, 14))
static JCudnnConvolution x668 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 256, 14, 14))
static JCudnnConvolution x612 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 256, 14, 14))
static JCudnnConvolution x556 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 256, 14, 14))
static JCudnnConvolution x497 = new JCudnnConvolution(new int[]{64,256,14,14},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(64, 512, 7, 7))
static JCudnnConvolution x965 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 512, 7, 7))
static JCudnnConvolution x909 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 512, 7, 7))
static JCudnnConvolution x850 = new JCudnnConvolution(new int[]{64,512,7,7},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(64, 64, 55, 55))
static JCudnnConvolution x186 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,3,3},new int[]{64}, 1, 1);
// (Convolv(1,1),List(64, 64, 55, 55))
static JCudnnConvolution x130 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,3,3},new int[]{64}, 1, 1);
// (Convolv(1,1),List(64, 64, 55, 55))
static JCudnnConvolution x71 = new JCudnnConvolution(new int[]{64,64,55,55},new int[]{64,64,3,3},new int[]{64}, 1, 1);
// (Convolv(2,0),List(64, 1024, 14, 14))
static JCudnnConvolution x466 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{1024,512,1,1},new int[]{1024}, 2, 0);
// (Convolv(2,0),List(64, 128, 28, 28))
static JCudnnConvolution x225 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{128,256,1,1},new int[]{128}, 2, 0);
// (Convolv(2,0),List(64, 2048, 7, 7))
static JCudnnConvolution x819 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{2048,1024,1,1},new int[]{2048}, 2, 0);
// (Convolv(2,0),List(64, 256, 14, 14))
static JCudnnConvolution x473 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{256,512,1,1},new int[]{256}, 2, 0);
// (Convolv(2,0),List(64, 512, 7, 7))
static JCudnnConvolution x826 = new JCudnnConvolution(new int[]{64,1024,14,14},new int[]{512,1024,1,1},new int[]{512}, 2, 0);
// (Convolv(2,0),List(64, 512, 28, 28))
static JCudnnConvolution x232 = new JCudnnConvolution(new int[]{64,256,55,55},new int[]{512,256,1,1},new int[]{512}, 2, 0);
// (Convolv(2,3),List(64, 64, 112, 112))
static JCudnnConvolution x20 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
// (LMDB,false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, 640, new int[]{64, 3, 224, 224});
// (LMDB,true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, 6400, new int[]{64, 3, 224, 224});
// (LogSoftmax(),List(64, 1000))
static JCudnnSoftmax x1014 = new JCudnnSoftmax(new int[]{64,1000}, 2);
// (Pooling(3,2,0,true),List(64, 64, 55, 55))
static JCudnnPooling x33 = new JCudnnPooling(new int[]{64,64,112,112}, 3, 2, 0, 0);
// (Pooling(7,1,0,false),List(64, 2048, 1, 1))
static JCudnnPooling x1000 = new JCudnnPooling(new int[]{64,2048,7,7}, 7, 1, 0, 2);
// (ReLU(),List(64, 1024, 14, 14))
static JCudnnActivation x807 = new JCudnnActivation(new int[]{64,1024,14,14}, 1);
// (ReLU(),List(64, 1024, 14, 14))
static JCudnnActivation x751 = new JCudnnActivation(new int[]{64,1024,14,14}, 1);
// (ReLU(),List(64, 1024, 14, 14))
static JCudnnActivation x695 = new JCudnnActivation(new int[]{64,1024,14,14}, 1);
// (ReLU(),List(64, 1024, 14, 14))
static JCudnnActivation x639 = new JCudnnActivation(new int[]{64,1024,14,14}, 1);
// (ReLU(),List(64, 1024, 14, 14))
static JCudnnActivation x583 = new JCudnnActivation(new int[]{64,1024,14,14}, 1);
// (ReLU(),List(64, 1024, 14, 14))
static JCudnnActivation x527 = new JCudnnActivation(new int[]{64,1024,14,14}, 1);
// (ReLU(),List(64, 1024, 14, 14))
static JCudnnActivation x524 = new JCudnnActivation(new int[]{64,1024,14,14}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x437 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x420 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x381 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x364 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x325 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x266 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x308 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 128, 28, 28))
static JCudnnActivation x249 = new JCudnnActivation(new int[]{64,128,28,28}, 1);
// (ReLU(),List(64, 2048, 7, 7))
static JCudnnActivation x992 = new JCudnnActivation(new int[]{64,2048,7,7}, 1);
// (ReLU(),List(64, 2048, 7, 7))
static JCudnnActivation x936 = new JCudnnActivation(new int[]{64,2048,7,7}, 1);
// (ReLU(),List(64, 2048, 7, 7))
static JCudnnActivation x880 = new JCudnnActivation(new int[]{64,2048,7,7}, 1);
// (ReLU(),List(64, 2048, 7, 7))
static JCudnnActivation x877 = new JCudnnActivation(new int[]{64,2048,7,7}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x790 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x773 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x734 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x717 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x678 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x661 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x622 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x605 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x566 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x507 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x549 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 14, 14))
static JCudnnActivation x490 = new JCudnnActivation(new int[]{64,256,14,14}, 1);
// (ReLU(),List(64, 256, 55, 55))
static JCudnnActivation x213 = new JCudnnActivation(new int[]{64,256,55,55}, 1);
// (ReLU(),List(64, 256, 55, 55))
static JCudnnActivation x157 = new JCudnnActivation(new int[]{64,256,55,55}, 1);
// (ReLU(),List(64, 256, 55, 55))
static JCudnnActivation x101 = new JCudnnActivation(new int[]{64,256,55,55}, 1);
// (ReLU(),List(64, 256, 55, 55))
static JCudnnActivation x98 = new JCudnnActivation(new int[]{64,256,55,55}, 1);
// (ReLU(),List(64, 512, 7, 7))
static JCudnnActivation x975 = new JCudnnActivation(new int[]{64,512,7,7}, 1);
// (ReLU(),List(64, 512, 7, 7))
static JCudnnActivation x958 = new JCudnnActivation(new int[]{64,512,7,7}, 1);
// (ReLU(),List(64, 512, 7, 7))
static JCudnnActivation x919 = new JCudnnActivation(new int[]{64,512,7,7}, 1);
// (ReLU(),List(64, 512, 7, 7))
static JCudnnActivation x860 = new JCudnnActivation(new int[]{64,512,7,7}, 1);
// (ReLU(),List(64, 512, 7, 7))
static JCudnnActivation x902 = new JCudnnActivation(new int[]{64,512,7,7}, 1);
// (ReLU(),List(64, 512, 7, 7))
static JCudnnActivation x843 = new JCudnnActivation(new int[]{64,512,7,7}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x454 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x398 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x342 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x286 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 512, 28, 28))
static JCudnnActivation x283 = new JCudnnActivation(new int[]{64,512,28,28}, 1);
// (ReLU(),List(64, 64, 55, 55))
static JCudnnActivation x196 = new JCudnnActivation(new int[]{64,64,55,55}, 1);
// (ReLU(),List(64, 64, 55, 55))
static JCudnnActivation x179 = new JCudnnActivation(new int[]{64,64,55,55}, 1);
// (ReLU(),List(64, 64, 55, 55))
static JCudnnActivation x140 = new JCudnnActivation(new int[]{64,64,55,55}, 1);
// (ReLU(),List(64, 64, 55, 55))
static JCudnnActivation x81 = new JCudnnActivation(new int[]{64,64,55,55}, 1);
// (ReLU(),List(64, 64, 55, 55))
static JCudnnActivation x123 = new JCudnnActivation(new int[]{64,64,55,55}, 1);
// (ReLU(),List(64, 64, 55, 55))
static JCudnnActivation x64 = new JCudnnActivation(new int[]{64,64,55,55}, 1);
// (ReLU(),List(64, 64, 112, 112))
static JCudnnActivation x30 = new JCudnnActivation(new int[]{64,64,112,112}, 1);
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
static JCudaTensor x94 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_bias").asJCudaTensor();
// 2a2_c_bn_scale
static JCudaTensor x93 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2a2_c_bn_scale").asJCudaTensor();
// 2a2_c_cv_B
static JCudaTensor x87 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2a2_c_cv_W
static JCudaTensor x86 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2a2_c_cv_W").asJCudaTensor();
// 2b_a_bn_bias
static JCudaTensor x119 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_bias").asJCudaTensor();
// 2b_a_bn_scale
static JCudaTensor x118 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_a_bn_scale").asJCudaTensor();
// 2b_a_cv_B
static JCudaTensor x112 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2b_a_cv_W
static JCudaTensor x111 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2b_a_cv_W").asJCudaTensor();
// 2b_b_bn_bias
static JCudaTensor x136 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_bias").asJCudaTensor();
// 2b_b_bn_scale
static JCudaTensor x135 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2b_b_bn_scale").asJCudaTensor();
// 2b_b_cv_B
static JCudaTensor x129 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2b_b_cv_W
static JCudaTensor x128 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2b_b_cv_W").asJCudaTensor();
// 2b_c_bn_bias
static JCudaTensor x153 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_bias").asJCudaTensor();
// 2b_c_bn_scale
static JCudaTensor x152 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2b_c_bn_scale").asJCudaTensor();
// 2b_c_cv_B
static JCudaTensor x146 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2b_c_cv_W
static JCudaTensor x145 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2b_c_cv_W").asJCudaTensor();
// 2c_a_bn_bias
static JCudaTensor x175 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_bias").asJCudaTensor();
// 2c_a_bn_scale
static JCudaTensor x174 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_a_bn_scale").asJCudaTensor();
// 2c_a_cv_B
static JCudaTensor x168 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2c_a_cv_W
static JCudaTensor x167 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/2c_a_cv_W").asJCudaTensor();
// 2c_b_bn_bias
static JCudaTensor x192 = JTensor.constFloat(0.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_bias").asJCudaTensor();
// 2c_b_bn_scale
static JCudaTensor x191 = JTensor.constFloat(1.0f, 1, 64, 1, 1).load(network_dir + "/2c_b_bn_scale").asJCudaTensor();
// 2c_b_cv_B
static JCudaTensor x185 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// 2c_b_cv_W
static JCudaTensor x184 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/2c_b_cv_W").asJCudaTensor();
// 2c_c_bn_bias
static JCudaTensor x209 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_bias").asJCudaTensor();
// 2c_c_bn_scale
static JCudaTensor x208 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/2c_c_bn_scale").asJCudaTensor();
// 2c_c_cv_B
static JCudaTensor x202 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 2c_c_cv_W
static JCudaTensor x201 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 256, 64, 1, 1).load(network_dir + "/2c_c_cv_W").asJCudaTensor();
// 3a1_bn_bias
static JCudaTensor x238 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_bias").asJCudaTensor();
// 3a1_bn_scale
static JCudaTensor x237 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a1_bn_scale").asJCudaTensor();
// 3a1_cv_B
static JCudaTensor x231 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3a1_cv_W
static JCudaTensor x230 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 512, 256, 1, 1).load(network_dir + "/3a1_cv_W").asJCudaTensor();
// 3a2_a_bn_bias
static JCudaTensor x245 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_bias").asJCudaTensor();
// 3a2_a_bn_scale
static JCudaTensor x244 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_a_bn_scale").asJCudaTensor();
// 3a2_a_cv_B
static JCudaTensor x224 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3a2_a_cv_W
static JCudaTensor x223 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/3a2_a_cv_W").asJCudaTensor();
// 3a2_b_bn_bias
static JCudaTensor x262 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_bias").asJCudaTensor();
// 3a2_b_bn_scale
static JCudaTensor x261 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3a2_b_bn_scale").asJCudaTensor();
// 3a2_b_cv_B
static JCudaTensor x255 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3a2_b_cv_W
static JCudaTensor x254 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3a2_b_cv_W").asJCudaTensor();
// 3a2_c_bn_bias
static JCudaTensor x279 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_bias").asJCudaTensor();
// 3a2_c_bn_scale
static JCudaTensor x278 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3a2_c_bn_scale").asJCudaTensor();
// 3a2_c_cv_B
static JCudaTensor x272 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3a2_c_cv_W
static JCudaTensor x271 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3a2_c_cv_W").asJCudaTensor();
// 3b_a_bn_bias
static JCudaTensor x304 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_bias").asJCudaTensor();
// 3b_a_bn_scale
static JCudaTensor x303 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_a_bn_scale").asJCudaTensor();
// 3b_a_cv_B
static JCudaTensor x297 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3b_a_cv_W
static JCudaTensor x296 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3b_a_cv_W").asJCudaTensor();
// 3b_b_bn_bias
static JCudaTensor x321 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_bias").asJCudaTensor();
// 3b_b_bn_scale
static JCudaTensor x320 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3b_b_bn_scale").asJCudaTensor();
// 3b_b_cv_B
static JCudaTensor x314 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3b_b_cv_W
static JCudaTensor x313 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3b_b_cv_W").asJCudaTensor();
// 3b_c_bn_bias
static JCudaTensor x338 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_bias").asJCudaTensor();
// 3b_c_bn_scale
static JCudaTensor x337 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3b_c_bn_scale").asJCudaTensor();
// 3b_c_cv_B
static JCudaTensor x331 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3b_c_cv_W
static JCudaTensor x330 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3b_c_cv_W").asJCudaTensor();
// 3c_a_bn_bias
static JCudaTensor x360 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_bias").asJCudaTensor();
// 3c_a_bn_scale
static JCudaTensor x359 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_a_bn_scale").asJCudaTensor();
// 3c_a_cv_B
static JCudaTensor x353 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3c_a_cv_W
static JCudaTensor x352 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3c_a_cv_W").asJCudaTensor();
// 3c_b_bn_bias
static JCudaTensor x377 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_bias").asJCudaTensor();
// 3c_b_bn_scale
static JCudaTensor x376 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3c_b_bn_scale").asJCudaTensor();
// 3c_b_cv_B
static JCudaTensor x370 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3c_b_cv_W
static JCudaTensor x369 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3c_b_cv_W").asJCudaTensor();
// 3c_c_bn_bias
static JCudaTensor x394 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_bias").asJCudaTensor();
// 3c_c_bn_scale
static JCudaTensor x393 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3c_c_bn_scale").asJCudaTensor();
// 3c_c_cv_B
static JCudaTensor x387 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3c_c_cv_W
static JCudaTensor x386 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3c_c_cv_W").asJCudaTensor();
// 3d_a_bn_bias
static JCudaTensor x416 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_bias").asJCudaTensor();
// 3d_a_bn_scale
static JCudaTensor x415 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_a_bn_scale").asJCudaTensor();
// 3d_a_cv_B
static JCudaTensor x409 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3d_a_cv_W
static JCudaTensor x408 = JTensor.randomFloat(-0.0625f, 0.0625f, 128, 512, 1, 1).load(network_dir + "/3d_a_cv_W").asJCudaTensor();
// 3d_b_bn_bias
static JCudaTensor x433 = JTensor.constFloat(0.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_bias").asJCudaTensor();
// 3d_b_bn_scale
static JCudaTensor x432 = JTensor.constFloat(1.0f, 1, 128, 1, 1).load(network_dir + "/3d_b_bn_scale").asJCudaTensor();
// 3d_b_cv_B
static JCudaTensor x426 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// 3d_b_cv_W
static JCudaTensor x425 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/3d_b_cv_W").asJCudaTensor();
// 3d_c_bn_bias
static JCudaTensor x450 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_bias").asJCudaTensor();
// 3d_c_bn_scale
static JCudaTensor x449 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/3d_c_bn_scale").asJCudaTensor();
// 3d_c_cv_B
static JCudaTensor x443 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 3d_c_cv_W
static JCudaTensor x442 = JTensor.randomFloat(-0.125f, 0.125f, 512, 128, 1, 1).load(network_dir + "/3d_c_cv_W").asJCudaTensor();
// 4a1_bn_bias
static JCudaTensor x479 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_bias").asJCudaTensor();
// 4a1_bn_scale
static JCudaTensor x478 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a1_bn_scale").asJCudaTensor();
// 4a1_cv_B
static JCudaTensor x465 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4a1_cv_W
static JCudaTensor x464 = JTensor.randomFloat(-0.0625f, 0.0625f, 1024, 512, 1, 1).load(network_dir + "/4a1_cv_W").asJCudaTensor();
// 4a2_a_bn_bias
static JCudaTensor x486 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_bias").asJCudaTensor();
// 4a2_a_bn_scale
static JCudaTensor x485 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_a_bn_scale").asJCudaTensor();
// 4a2_a_cv_B
static JCudaTensor x472 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4a2_a_cv_W
static JCudaTensor x471 = JTensor.randomFloat(-0.0625f, 0.0625f, 256, 512, 1, 1).load(network_dir + "/4a2_a_cv_W").asJCudaTensor();
// 4a2_b_bn_bias
static JCudaTensor x503 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_bias").asJCudaTensor();
// 4a2_b_bn_scale
static JCudaTensor x502 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4a2_b_bn_scale").asJCudaTensor();
// 4a2_b_cv_B
static JCudaTensor x496 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4a2_b_cv_W
static JCudaTensor x495 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4a2_b_cv_W").asJCudaTensor();
// 4a2_c_bn_bias
static JCudaTensor x520 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_bias").asJCudaTensor();
// 4a2_c_bn_scale
static JCudaTensor x519 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4a2_c_bn_scale").asJCudaTensor();
// 4a2_c_cv_B
static JCudaTensor x513 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4a2_c_cv_W
static JCudaTensor x512 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4a2_c_cv_W").asJCudaTensor();
// 4b_a_bn_bias
static JCudaTensor x545 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_bias").asJCudaTensor();
// 4b_a_bn_scale
static JCudaTensor x544 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_a_bn_scale").asJCudaTensor();
// 4b_a_cv_B
static JCudaTensor x538 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4b_a_cv_W
static JCudaTensor x537 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4b_a_cv_W").asJCudaTensor();
// 4b_b_bn_bias
static JCudaTensor x562 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_bias").asJCudaTensor();
// 4b_b_bn_scale
static JCudaTensor x561 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4b_b_bn_scale").asJCudaTensor();
// 4b_b_cv_B
static JCudaTensor x555 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4b_b_cv_W
static JCudaTensor x554 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4b_b_cv_W").asJCudaTensor();
// 4b_c_bn_bias
static JCudaTensor x579 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_bias").asJCudaTensor();
// 4b_c_bn_scale
static JCudaTensor x578 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4b_c_bn_scale").asJCudaTensor();
// 4b_c_cv_B
static JCudaTensor x572 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4b_c_cv_W
static JCudaTensor x571 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4b_c_cv_W").asJCudaTensor();
// 4c_a_bn_bias
static JCudaTensor x601 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_bias").asJCudaTensor();
// 4c_a_bn_scale
static JCudaTensor x600 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_a_bn_scale").asJCudaTensor();
// 4c_a_cv_B
static JCudaTensor x594 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4c_a_cv_W
static JCudaTensor x593 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4c_a_cv_W").asJCudaTensor();
// 4c_b_bn_bias
static JCudaTensor x618 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_bias").asJCudaTensor();
// 4c_b_bn_scale
static JCudaTensor x617 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4c_b_bn_scale").asJCudaTensor();
// 4c_b_cv_B
static JCudaTensor x611 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4c_b_cv_W
static JCudaTensor x610 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4c_b_cv_W").asJCudaTensor();
// 4c_c_bn_bias
static JCudaTensor x635 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_bias").asJCudaTensor();
// 4c_c_bn_scale
static JCudaTensor x634 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4c_c_bn_scale").asJCudaTensor();
// 4c_c_cv_B
static JCudaTensor x628 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4c_c_cv_W
static JCudaTensor x627 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4c_c_cv_W").asJCudaTensor();
// 4d_a_bn_bias
static JCudaTensor x657 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_bias").asJCudaTensor();
// 4d_a_bn_scale
static JCudaTensor x656 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_a_bn_scale").asJCudaTensor();
// 4d_a_cv_B
static JCudaTensor x650 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4d_a_cv_W
static JCudaTensor x649 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4d_a_cv_W").asJCudaTensor();
// 4d_b_bn_bias
static JCudaTensor x674 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_bias").asJCudaTensor();
// 4d_b_bn_scale
static JCudaTensor x673 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4d_b_bn_scale").asJCudaTensor();
// 4d_b_cv_B
static JCudaTensor x667 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4d_b_cv_W
static JCudaTensor x666 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4d_b_cv_W").asJCudaTensor();
// 4d_c_bn_bias
static JCudaTensor x691 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_bias").asJCudaTensor();
// 4d_c_bn_scale
static JCudaTensor x690 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4d_c_bn_scale").asJCudaTensor();
// 4d_c_cv_B
static JCudaTensor x684 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4d_c_cv_W
static JCudaTensor x683 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4d_c_cv_W").asJCudaTensor();
// 4e_a_bn_bias
static JCudaTensor x713 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_bias").asJCudaTensor();
// 4e_a_bn_scale
static JCudaTensor x712 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_a_bn_scale").asJCudaTensor();
// 4e_a_cv_B
static JCudaTensor x706 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4e_a_cv_W
static JCudaTensor x705 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4e_a_cv_W").asJCudaTensor();
// 4e_b_bn_bias
static JCudaTensor x730 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_bias").asJCudaTensor();
// 4e_b_bn_scale
static JCudaTensor x729 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4e_b_bn_scale").asJCudaTensor();
// 4e_b_cv_B
static JCudaTensor x723 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4e_b_cv_W
static JCudaTensor x722 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4e_b_cv_W").asJCudaTensor();
// 4e_c_bn_bias
static JCudaTensor x747 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_bias").asJCudaTensor();
// 4e_c_bn_scale
static JCudaTensor x746 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4e_c_bn_scale").asJCudaTensor();
// 4e_c_cv_B
static JCudaTensor x740 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4e_c_cv_W
static JCudaTensor x739 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4e_c_cv_W").asJCudaTensor();
// 4f_a_bn_bias
static JCudaTensor x769 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_bias").asJCudaTensor();
// 4f_a_bn_scale
static JCudaTensor x768 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_a_bn_scale").asJCudaTensor();
// 4f_a_cv_B
static JCudaTensor x762 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4f_a_cv_W
static JCudaTensor x761 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 256, 1024, 1, 1).load(network_dir + "/4f_a_cv_W").asJCudaTensor();
// 4f_b_bn_bias
static JCudaTensor x786 = JTensor.constFloat(0.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_bias").asJCudaTensor();
// 4f_b_bn_scale
static JCudaTensor x785 = JTensor.constFloat(1.0f, 1, 256, 1, 1).load(network_dir + "/4f_b_bn_scale").asJCudaTensor();
// 4f_b_cv_B
static JCudaTensor x779 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
// 4f_b_cv_W
static JCudaTensor x778 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/4f_b_cv_W").asJCudaTensor();
// 4f_c_bn_bias
static JCudaTensor x803 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_bias").asJCudaTensor();
// 4f_c_bn_scale
static JCudaTensor x802 = JTensor.constFloat(1.0f, 1, 1024, 1, 1).load(network_dir + "/4f_c_bn_scale").asJCudaTensor();
// 4f_c_cv_B
static JCudaTensor x796 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// 4f_c_cv_W
static JCudaTensor x795 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1024, 256, 1, 1).load(network_dir + "/4f_c_cv_W").asJCudaTensor();
// 5a1_bn_bias
static JCudaTensor x839 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_bias").asJCudaTensor();
// 5a1_bn_scale
static JCudaTensor x838 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a1_bn_scale").asJCudaTensor();
// 5a1_cv_B
static JCudaTensor x818 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5a1_cv_W
static JCudaTensor x817 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 2048, 1024, 1, 1).load(network_dir + "/5a1_cv_W").asJCudaTensor();
// 5a2_a_bn_bias
static JCudaTensor x832 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_bias").asJCudaTensor();
// 5a2_a_bn_scale
static JCudaTensor x831 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_a_bn_scale").asJCudaTensor();
// 5a2_a_cv_B
static JCudaTensor x825 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5a2_a_cv_W
static JCudaTensor x824 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 512, 1024, 1, 1).load(network_dir + "/5a2_a_cv_W").asJCudaTensor();
// 5a2_b_bn_bias
static JCudaTensor x856 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_bias").asJCudaTensor();
// 5a2_b_bn_scale
static JCudaTensor x855 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5a2_b_bn_scale").asJCudaTensor();
// 5a2_b_cv_B
static JCudaTensor x849 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5a2_b_cv_W
static JCudaTensor x848 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5a2_b_cv_W").asJCudaTensor();
// 5a2_c_bn_bias
static JCudaTensor x873 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_bias").asJCudaTensor();
// 5a2_c_bn_scale
static JCudaTensor x872 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5a2_c_bn_scale").asJCudaTensor();
// 5a2_c_cv_B
static JCudaTensor x866 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5a2_c_cv_W
static JCudaTensor x865 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5a2_c_cv_W").asJCudaTensor();
// 5b_a_bn_bias
static JCudaTensor x898 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_bias").asJCudaTensor();
// 5b_a_bn_scale
static JCudaTensor x897 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_a_bn_scale").asJCudaTensor();
// 5b_a_cv_B
static JCudaTensor x891 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5b_a_cv_W
static JCudaTensor x890 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5b_a_cv_W").asJCudaTensor();
// 5b_b_bn_bias
static JCudaTensor x915 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_bias").asJCudaTensor();
// 5b_b_bn_scale
static JCudaTensor x914 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5b_b_bn_scale").asJCudaTensor();
// 5b_b_cv_B
static JCudaTensor x908 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5b_b_cv_W
static JCudaTensor x907 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5b_b_cv_W").asJCudaTensor();
// 5b_c_bn_bias
static JCudaTensor x932 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_bias").asJCudaTensor();
// 5b_c_bn_scale
static JCudaTensor x931 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5b_c_bn_scale").asJCudaTensor();
// 5b_c_cv_B
static JCudaTensor x925 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5b_c_cv_W
static JCudaTensor x924 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5b_c_cv_W").asJCudaTensor();
// 5c_a_bn_bias
static JCudaTensor x954 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_bias").asJCudaTensor();
// 5c_a_bn_scale
static JCudaTensor x953 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_a_bn_scale").asJCudaTensor();
// 5c_a_cv_B
static JCudaTensor x947 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5c_a_cv_W
static JCudaTensor x946 = JTensor.randomFloat(-0.03125f, 0.03125f, 512, 2048, 1, 1).load(network_dir + "/5c_a_cv_W").asJCudaTensor();
// 5c_b_bn_bias
static JCudaTensor x971 = JTensor.constFloat(0.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_bias").asJCudaTensor();
// 5c_b_bn_scale
static JCudaTensor x970 = JTensor.constFloat(1.0f, 1, 512, 1, 1).load(network_dir + "/5c_b_bn_scale").asJCudaTensor();
// 5c_b_cv_B
static JCudaTensor x964 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
// 5c_b_cv_W
static JCudaTensor x963 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/5c_b_cv_W").asJCudaTensor();
// 5c_c_bn_bias
static JCudaTensor x988 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_bias").asJCudaTensor();
// 5c_c_bn_scale
static JCudaTensor x987 = JTensor.constFloat(1.0f, 1, 2048, 1, 1).load(network_dir + "/5c_c_bn_scale").asJCudaTensor();
// 5c_c_cv_B
static JCudaTensor x981 = JTensor.constFloat(0.0f, 2048).asJCudaTensor();
// 5c_c_cv_W
static JCudaTensor x980 = JTensor.randomFloat(-0.0625f, 0.0625f, 2048, 512, 1, 1).load(network_dir + "/5c_c_cv_W").asJCudaTensor();
// V_1_bn_bias
static JCudaTensor x3602 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_1_bn_scale
static JCudaTensor x3607 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_1_cv_W
static JCudaTensor x3612 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
// V_2a1_bn_bias
static JCudaTensor x3438 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a1_bn_scale
static JCudaTensor x3458 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a1_cv_W
static JCudaTensor x3448 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_2a2_a_bn_bias
static JCudaTensor x3564 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_a_bn_scale
static JCudaTensor x3559 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_a_cv_W
static JCudaTensor x3553 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
// V_2a2_b_bn_bias
static JCudaTensor x3503 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_b_bn_scale
static JCudaTensor x3517 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2a2_b_cv_W
static JCudaTensor x3511 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
// V_2a2_c_bn_bias
static JCudaTensor x3443 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a2_c_bn_scale
static JCudaTensor x3453 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2a2_c_cv_W
static JCudaTensor x3432 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_2b_a_bn_bias
static JCudaTensor x3370 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_a_bn_scale
static JCudaTensor x3356 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_a_cv_W
static JCudaTensor x3364 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_2b_b_bn_bias
static JCudaTensor x3318 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_b_bn_scale
static JCudaTensor x3313 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2b_b_cv_W
static JCudaTensor x3323 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
// V_2b_c_bn_bias
static JCudaTensor x3272 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2b_c_bn_scale
static JCudaTensor x3264 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2b_c_cv_W
static JCudaTensor x3277 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_2c_a_bn_bias
static JCudaTensor x3224 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_a_bn_scale
static JCudaTensor x3210 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_a_cv_W
static JCudaTensor x3218 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_2c_b_bn_bias
static JCudaTensor x3173 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_b_bn_scale
static JCudaTensor x3178 = JTensor.constFloat(0.0f, 1, 64, 1, 1).asJCudaTensor();
// V_2c_b_cv_W
static JCudaTensor x3167 = JTensor.constFloat(0.0f, 64, 64, 3, 3).asJCudaTensor();
// V_2c_c_bn_bias
static JCudaTensor x3118 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2c_c_bn_scale
static JCudaTensor x3123 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_2c_c_cv_W
static JCudaTensor x3131 = JTensor.constFloat(0.0f, 256, 64, 1, 1).asJCudaTensor();
// V_3a1_bn_bias
static JCudaTensor x2952 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a1_bn_scale
static JCudaTensor x2966 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a1_cv_W
static JCudaTensor x2957 = JTensor.constFloat(0.0f, 512, 256, 1, 1).asJCudaTensor();
// V_3a2_a_bn_bias
static JCudaTensor x3082 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_a_bn_scale
static JCudaTensor x3077 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_a_cv_W
static JCudaTensor x3071 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_3a2_b_bn_bias
static JCudaTensor x3021 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_b_bn_scale
static JCudaTensor x3026 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3a2_b_cv_W
static JCudaTensor x3034 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3a2_c_bn_bias
static JCudaTensor x2971 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a2_c_bn_scale
static JCudaTensor x2980 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3a2_c_cv_W
static JCudaTensor x2947 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_3b_a_bn_bias
static JCudaTensor x2879 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_a_bn_scale
static JCudaTensor x2874 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_a_cv_W
static JCudaTensor x2887 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
// V_3b_b_bn_bias
static JCudaTensor x2833 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_b_bn_scale
static JCudaTensor x2828 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3b_b_cv_W
static JCudaTensor x2841 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3b_c_bn_bias
static JCudaTensor x2782 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3b_c_bn_scale
static JCudaTensor x2787 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3b_c_cv_W
static JCudaTensor x2795 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_3c_a_bn_bias
static JCudaTensor x2737 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_a_bn_scale
static JCudaTensor x2742 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_a_cv_W
static JCudaTensor x2731 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
// V_3c_b_bn_bias
static JCudaTensor x2696 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_b_bn_scale
static JCudaTensor x2682 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3c_b_cv_W
static JCudaTensor x2690 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3c_c_bn_bias
static JCudaTensor x2636 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3c_c_bn_scale
static JCudaTensor x2650 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3c_c_cv_W
static JCudaTensor x2644 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_3d_a_bn_bias
static JCudaTensor x2590 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_a_bn_scale
static JCudaTensor x2582 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_a_cv_W
static JCudaTensor x2595 = JTensor.constFloat(0.0f, 128, 512, 1, 1).asJCudaTensor();
// V_3d_b_bn_bias
static JCudaTensor x2539 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_b_bn_scale
static JCudaTensor x2544 = JTensor.constFloat(0.0f, 1, 128, 1, 1).asJCudaTensor();
// V_3d_b_cv_W
static JCudaTensor x2549 = JTensor.constFloat(0.0f, 128, 128, 3, 3).asJCudaTensor();
// V_3d_c_bn_bias
static JCudaTensor x2499 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3d_c_bn_scale
static JCudaTensor x2504 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_3d_c_cv_W
static JCudaTensor x2493 = JTensor.constFloat(0.0f, 512, 128, 1, 1).asJCudaTensor();
// V_4a1_bn_bias
static JCudaTensor x2319 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a1_bn_scale
static JCudaTensor x2324 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a1_cv_W
static JCudaTensor x2348 = JTensor.constFloat(0.0f, 1024, 512, 1, 1).asJCudaTensor();
// V_4a2_a_bn_bias
static JCudaTensor x2449 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_a_bn_scale
static JCudaTensor x2454 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_a_cv_W
static JCudaTensor x2443 = JTensor.constFloat(0.0f, 256, 512, 1, 1).asJCudaTensor();
// V_4a2_b_bn_bias
static JCudaTensor x2407 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_b_bn_scale
static JCudaTensor x2402 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4a2_b_cv_W
static JCudaTensor x2396 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4a2_c_bn_bias
static JCudaTensor x2343 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a2_c_bn_scale
static JCudaTensor x2329 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4a2_c_cv_W
static JCudaTensor x2337 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4b_a_bn_bias
static JCudaTensor x2255 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_a_bn_scale
static JCudaTensor x2260 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_a_cv_W
static JCudaTensor x2249 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4b_b_bn_bias
static JCudaTensor x2214 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_b_bn_scale
static JCudaTensor x2209 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4b_b_cv_W
static JCudaTensor x2203 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4b_c_bn_bias
static JCudaTensor x2154 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4b_c_bn_scale
static JCudaTensor x2168 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4b_c_cv_W
static JCudaTensor x2162 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4c_a_bn_bias
static JCudaTensor x2103 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_a_bn_scale
static JCudaTensor x2108 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_a_cv_W
static JCudaTensor x2113 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4c_b_bn_bias
static JCudaTensor x2063 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_b_bn_scale
static JCudaTensor x2068 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4c_b_cv_W
static JCudaTensor x2057 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4c_c_bn_bias
static JCudaTensor x2008 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4c_c_bn_scale
static JCudaTensor x2022 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4c_c_cv_W
static JCudaTensor x2016 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4d_a_bn_bias
static JCudaTensor x1954 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_a_bn_scale
static JCudaTensor x1968 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_a_cv_W
static JCudaTensor x1962 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4d_b_bn_bias
static JCudaTensor x1922 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_b_bn_scale
static JCudaTensor x1917 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4d_b_cv_W
static JCudaTensor x1911 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4d_c_bn_bias
static JCudaTensor x1870 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4d_c_bn_scale
static JCudaTensor x1862 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4d_c_cv_W
static JCudaTensor x1875 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4e_a_bn_bias
static JCudaTensor x1822 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_a_bn_scale
static JCudaTensor x1817 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_a_cv_W
static JCudaTensor x1811 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4e_b_bn_bias
static JCudaTensor x1765 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_b_bn_scale
static JCudaTensor x1770 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4e_b_cv_W
static JCudaTensor x1775 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4e_c_bn_bias
static JCudaTensor x1730 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4e_c_bn_scale
static JCudaTensor x1725 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4e_c_cv_W
static JCudaTensor x1719 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_4f_a_bn_bias
static JCudaTensor x1671 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_a_bn_scale
static JCudaTensor x1676 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_a_cv_W
static JCudaTensor x1665 = JTensor.constFloat(0.0f, 256, 1024, 1, 1).asJCudaTensor();
// V_4f_b_bn_bias
static JCudaTensor x1616 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_b_bn_scale
static JCudaTensor x1624 = JTensor.constFloat(0.0f, 1, 256, 1, 1).asJCudaTensor();
// V_4f_b_cv_W
static JCudaTensor x1629 = JTensor.constFloat(0.0f, 256, 256, 3, 3).asJCudaTensor();
// V_4f_c_bn_bias
static JCudaTensor x1570 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4f_c_bn_scale
static JCudaTensor x1584 = JTensor.constFloat(0.0f, 1, 1024, 1, 1).asJCudaTensor();
// V_4f_c_cv_W
static JCudaTensor x1578 = JTensor.constFloat(0.0f, 1024, 256, 1, 1).asJCudaTensor();
// V_5a1_bn_bias
static JCudaTensor x1404 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a1_bn_scale
static JCudaTensor x1428 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a1_cv_W
static JCudaTensor x1409 = JTensor.constFloat(0.0f, 2048, 1024, 1, 1).asJCudaTensor();
// V_5a2_a_bn_bias
static JCudaTensor x1519 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_a_bn_scale
static JCudaTensor x1524 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_a_cv_W
static JCudaTensor x1533 = JTensor.constFloat(0.0f, 512, 1024, 1, 1).asJCudaTensor();
// V_5a2_b_bn_bias
static JCudaTensor x1487 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_b_bn_scale
static JCudaTensor x1482 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5a2_b_cv_W
static JCudaTensor x1476 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_5a2_c_bn_bias
static JCudaTensor x1399 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a2_c_bn_scale
static JCudaTensor x1423 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5a2_c_cv_W
static JCudaTensor x1414 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
// V_5b_a_bn_bias
static JCudaTensor x1340 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_a_bn_scale
static JCudaTensor x1326 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_a_cv_W
static JCudaTensor x1334 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
// V_5b_b_bn_bias
static JCudaTensor x1280 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_b_bn_scale
static JCudaTensor x1285 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5b_b_cv_W
static JCudaTensor x1293 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_5b_c_bn_bias
static JCudaTensor x1248 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5b_c_bn_scale
static JCudaTensor x1243 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5b_c_cv_W
static JCudaTensor x1237 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
// V_5c_a_bn_bias
static JCudaTensor x1180 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_a_bn_scale
static JCudaTensor x1185 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_a_cv_W
static JCudaTensor x1193 = JTensor.constFloat(0.0f, 512, 2048, 1, 1).asJCudaTensor();
// V_5c_b_bn_bias
static JCudaTensor x1134 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_b_bn_scale
static JCudaTensor x1148 = JTensor.constFloat(0.0f, 1, 512, 1, 1).asJCudaTensor();
// V_5c_b_cv_W
static JCudaTensor x1142 = JTensor.constFloat(0.0f, 512, 512, 3, 3).asJCudaTensor();
// V_5c_c_bn_bias
static JCudaTensor x1102 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5c_c_bn_scale
static JCudaTensor x1097 = JTensor.constFloat(0.0f, 1, 2048, 1, 1).asJCudaTensor();
// V_5c_c_cv_W
static JCudaTensor x1091 = JTensor.constFloat(0.0f, 2048, 512, 1, 1).asJCudaTensor();
// V_fc_B
static JCudaTensor x1055 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc_W
static JCudaTensor x1043 = JTensor.constFloat(0.0f, 1000, 2048).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// fc_B
static JCudaTensor x1011 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
// fc_W
static JCudaTensor x1007 = JTensor.randomFloat(-0.03125f, 0.03125f, 1000, 2048).load(network_dir + "/fc_W").asJCudaTensor();

public static void main(String[] args){
ArithStats.isStats = false;
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
System.out.println(ArithStats.outStats());
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
x94.save(network_dir + "/2a2_c_bn_bias");
x93.save(network_dir + "/2a2_c_bn_scale");
x86.save(network_dir + "/2a2_c_cv_W");
x119.save(network_dir + "/2b_a_bn_bias");
x118.save(network_dir + "/2b_a_bn_scale");
x111.save(network_dir + "/2b_a_cv_W");
x136.save(network_dir + "/2b_b_bn_bias");
x135.save(network_dir + "/2b_b_bn_scale");
x128.save(network_dir + "/2b_b_cv_W");
x153.save(network_dir + "/2b_c_bn_bias");
x152.save(network_dir + "/2b_c_bn_scale");
x145.save(network_dir + "/2b_c_cv_W");
x175.save(network_dir + "/2c_a_bn_bias");
x174.save(network_dir + "/2c_a_bn_scale");
x167.save(network_dir + "/2c_a_cv_W");
x192.save(network_dir + "/2c_b_bn_bias");
x191.save(network_dir + "/2c_b_bn_scale");
x184.save(network_dir + "/2c_b_cv_W");
x209.save(network_dir + "/2c_c_bn_bias");
x208.save(network_dir + "/2c_c_bn_scale");
x201.save(network_dir + "/2c_c_cv_W");
x238.save(network_dir + "/3a1_bn_bias");
x237.save(network_dir + "/3a1_bn_scale");
x230.save(network_dir + "/3a1_cv_W");
x245.save(network_dir + "/3a2_a_bn_bias");
x244.save(network_dir + "/3a2_a_bn_scale");
x223.save(network_dir + "/3a2_a_cv_W");
x262.save(network_dir + "/3a2_b_bn_bias");
x261.save(network_dir + "/3a2_b_bn_scale");
x254.save(network_dir + "/3a2_b_cv_W");
x279.save(network_dir + "/3a2_c_bn_bias");
x278.save(network_dir + "/3a2_c_bn_scale");
x271.save(network_dir + "/3a2_c_cv_W");
x304.save(network_dir + "/3b_a_bn_bias");
x303.save(network_dir + "/3b_a_bn_scale");
x296.save(network_dir + "/3b_a_cv_W");
x321.save(network_dir + "/3b_b_bn_bias");
x320.save(network_dir + "/3b_b_bn_scale");
x313.save(network_dir + "/3b_b_cv_W");
x338.save(network_dir + "/3b_c_bn_bias");
x337.save(network_dir + "/3b_c_bn_scale");
x330.save(network_dir + "/3b_c_cv_W");
x360.save(network_dir + "/3c_a_bn_bias");
x359.save(network_dir + "/3c_a_bn_scale");
x352.save(network_dir + "/3c_a_cv_W");
x377.save(network_dir + "/3c_b_bn_bias");
x376.save(network_dir + "/3c_b_bn_scale");
x369.save(network_dir + "/3c_b_cv_W");
x394.save(network_dir + "/3c_c_bn_bias");
x393.save(network_dir + "/3c_c_bn_scale");
x386.save(network_dir + "/3c_c_cv_W");
x416.save(network_dir + "/3d_a_bn_bias");
x415.save(network_dir + "/3d_a_bn_scale");
x408.save(network_dir + "/3d_a_cv_W");
x433.save(network_dir + "/3d_b_bn_bias");
x432.save(network_dir + "/3d_b_bn_scale");
x425.save(network_dir + "/3d_b_cv_W");
x450.save(network_dir + "/3d_c_bn_bias");
x449.save(network_dir + "/3d_c_bn_scale");
x442.save(network_dir + "/3d_c_cv_W");
x479.save(network_dir + "/4a1_bn_bias");
x478.save(network_dir + "/4a1_bn_scale");
x464.save(network_dir + "/4a1_cv_W");
x486.save(network_dir + "/4a2_a_bn_bias");
x485.save(network_dir + "/4a2_a_bn_scale");
x471.save(network_dir + "/4a2_a_cv_W");
x503.save(network_dir + "/4a2_b_bn_bias");
x502.save(network_dir + "/4a2_b_bn_scale");
x495.save(network_dir + "/4a2_b_cv_W");
x520.save(network_dir + "/4a2_c_bn_bias");
x519.save(network_dir + "/4a2_c_bn_scale");
x512.save(network_dir + "/4a2_c_cv_W");
x545.save(network_dir + "/4b_a_bn_bias");
x544.save(network_dir + "/4b_a_bn_scale");
x537.save(network_dir + "/4b_a_cv_W");
x562.save(network_dir + "/4b_b_bn_bias");
x561.save(network_dir + "/4b_b_bn_scale");
x554.save(network_dir + "/4b_b_cv_W");
x579.save(network_dir + "/4b_c_bn_bias");
x578.save(network_dir + "/4b_c_bn_scale");
x571.save(network_dir + "/4b_c_cv_W");
x601.save(network_dir + "/4c_a_bn_bias");
x600.save(network_dir + "/4c_a_bn_scale");
x593.save(network_dir + "/4c_a_cv_W");
x618.save(network_dir + "/4c_b_bn_bias");
x617.save(network_dir + "/4c_b_bn_scale");
x610.save(network_dir + "/4c_b_cv_W");
x635.save(network_dir + "/4c_c_bn_bias");
x634.save(network_dir + "/4c_c_bn_scale");
x627.save(network_dir + "/4c_c_cv_W");
x657.save(network_dir + "/4d_a_bn_bias");
x656.save(network_dir + "/4d_a_bn_scale");
x649.save(network_dir + "/4d_a_cv_W");
x674.save(network_dir + "/4d_b_bn_bias");
x673.save(network_dir + "/4d_b_bn_scale");
x666.save(network_dir + "/4d_b_cv_W");
x691.save(network_dir + "/4d_c_bn_bias");
x690.save(network_dir + "/4d_c_bn_scale");
x683.save(network_dir + "/4d_c_cv_W");
x713.save(network_dir + "/4e_a_bn_bias");
x712.save(network_dir + "/4e_a_bn_scale");
x705.save(network_dir + "/4e_a_cv_W");
x730.save(network_dir + "/4e_b_bn_bias");
x729.save(network_dir + "/4e_b_bn_scale");
x722.save(network_dir + "/4e_b_cv_W");
x747.save(network_dir + "/4e_c_bn_bias");
x746.save(network_dir + "/4e_c_bn_scale");
x739.save(network_dir + "/4e_c_cv_W");
x769.save(network_dir + "/4f_a_bn_bias");
x768.save(network_dir + "/4f_a_bn_scale");
x761.save(network_dir + "/4f_a_cv_W");
x786.save(network_dir + "/4f_b_bn_bias");
x785.save(network_dir + "/4f_b_bn_scale");
x778.save(network_dir + "/4f_b_cv_W");
x803.save(network_dir + "/4f_c_bn_bias");
x802.save(network_dir + "/4f_c_bn_scale");
x795.save(network_dir + "/4f_c_cv_W");
x839.save(network_dir + "/5a1_bn_bias");
x838.save(network_dir + "/5a1_bn_scale");
x817.save(network_dir + "/5a1_cv_W");
x832.save(network_dir + "/5a2_a_bn_bias");
x831.save(network_dir + "/5a2_a_bn_scale");
x824.save(network_dir + "/5a2_a_cv_W");
x856.save(network_dir + "/5a2_b_bn_bias");
x855.save(network_dir + "/5a2_b_bn_scale");
x848.save(network_dir + "/5a2_b_cv_W");
x873.save(network_dir + "/5a2_c_bn_bias");
x872.save(network_dir + "/5a2_c_bn_scale");
x865.save(network_dir + "/5a2_c_cv_W");
x898.save(network_dir + "/5b_a_bn_bias");
x897.save(network_dir + "/5b_a_bn_scale");
x890.save(network_dir + "/5b_a_cv_W");
x915.save(network_dir + "/5b_b_bn_bias");
x914.save(network_dir + "/5b_b_bn_scale");
x907.save(network_dir + "/5b_b_cv_W");
x932.save(network_dir + "/5b_c_bn_bias");
x931.save(network_dir + "/5b_c_bn_scale");
x924.save(network_dir + "/5b_c_cv_W");
x954.save(network_dir + "/5c_a_bn_bias");
x953.save(network_dir + "/5c_a_bn_scale");
x946.save(network_dir + "/5c_a_cv_W");
x971.save(network_dir + "/5c_b_bn_bias");
x970.save(network_dir + "/5c_b_bn_scale");
x963.save(network_dir + "/5c_b_cv_W");
x988.save(network_dir + "/5c_c_bn_bias");
x987.save(network_dir + "/5c_c_bn_scale");
x980.save(network_dir + "/5c_c_cv_W");
x1011.save(network_dir + "/fc_B");
x1007.save(network_dir + "/fc_W");
x705.free();
x53.free();
x2068.free();
x2947.free();
x2650.free();
x2209.free();
x167.free();
x2022.free();
x119.free();
x683.free();
x1193.free();
x825.free();
x1180.free();
x855.free();
x2108.free();
x1476.free();
x321.free();
x136.free();
x572.free();
x376.free();
x2957.free();
x337.free();
x2016.free();
x1428.free();
x2595.free();
x980.free();
x3071.free();
x1102.free();
x1423.free();
x987.free();
x184.free();
x191.free();
x890.free();
x2590.free();
x831.free();
x594.free();
x94.free();
x2407.free();
x2787.free();
x1340.free();
x964.free();
x3313.free();
x1616.free();
x963.free();
x856.free();
x3370.free();
x769.free();
x3511.free();
x3210.free();
x561.free();
x39.free();
x2454.free();
x1578.free();
x747.free();
x657.free();
x128.free();
x667.free();
x898.free();
x1097.free();
x472.free();
x369.free();
x866.free();
x224.free();
x931.free();
x817.free();
x331.free();
x1409.free();
x3356.free();
x18.free();
x19.free();
x175.free();
x2499.free();
x740.free();
x891.free();
x1917.free();
x3503.free();
x2539.free();
x393.free();
x1770.free();
x237.free();
x674.free();
x3218.free();
x1922.free();
x915.free();
x739.free();
x2737.free();
x2828.free();
x1280.free();
x723.free();
x947.free();
x238.free();
x112.free();
x386.free();
x59.free();
x1285.free();
x3432.free();
x3458.free();
x2782.free();
x486.free();
x352.free();
x231.free();
x2544.free();
x377.free();
x2214.free();
x230.free();
x554.free();
x2636.free();
x2319.free();
x2952.free();
x296.free();
x255.free();
x2063.free();
x152.free();
x370.free();
x3224.free();
x1968.free();
x353.free();
x1293.free();
x1334.free();
x634.free();
x1237.free();
x520.free();
x3517.free();
x908.free();
x297.free();
x1822.free();
x2449.free();
x271.free();
x1091.free();
x145.free();
x618.free();
x1629.free();
x3264.free();
x656.free();
x824.free();
x981.free();
x555.free();
x3034.free();
x313.free();
x3173.free();
x70.free();
x2731.free();
x1148.free();
x77.free();
x1482.free();
x722.free();
x513.free();
x3167.free();
x1533.free();
x1142.free();
x25.free();
x2504.free();
x2329.free();
x442.free();
x1962.free();
x1725.free();
x3438.free();
x2887.free();
x3118.free();
x2337.free();
x839.free();
x1524.free();
x690.free();
x432.free();
x519.free();
x1862.free();
x795.free();
x1404.free();
x512.free();
x3131.free();
x712.free();
x26.free();
x261.free();
x761.free();
x2324.free();
x338.free();
x3448.free();
x3123.free();
x201.free();
x1055.free();
x2696.free();
x971.free();
x1011.free();
x1570.free();
x2795.free();
x545.free();
x2162.free();
x1399.free();
x192.free();
x2348.free();
x2168.free();
x914.free();
x223.free();
x1775.free();
x946.free();
x45.free();
x272.free();
x478.free();
x60.free();
x832.free();
x768.free();
x849.free();
x479.free();
x544.free();
x838.free();
x86.free();
x2582.free();
x1765.free();
x684.free();
x729.free();
x153.free();
x3026.free();
x2644.free();
x628.free();
x415.free();
x3443.free();
x1007.free();
x730.free();
x278.free();
x2971.free();
x2343.free();
x52.free();
x538.free();
x865.free();
x3612.free();
x425.free();
x359.free();
x416.free();
x443.free();
x635.free();
x93.free();
x303.free();
x1671.free();
x3364.free();
x627.free();
x46.free();
x2255.free();
x1326.free();
x426.free();
x2154.free();
x3077.free();
x1414.free();
x2833.free();
x2879.free();
x746.free();
x600.free();
x314.free();
x872.free();
x2874.free();
x3178.free();
x1911.free();
x2742.free();
x38.free();
x848.free();
x409.free();
x69.free();
x496.free();
x3021.free();
x465.free();
x433.free();
x1811.free();
x202.free();
x691.free();
x617.free();
x503.free();
x129.free();
x562.free();
x146.free();
x762.free();
x924.free();
x118.free();
x1185.free();
x873.free();
x1719.free();
x579.free();
x925.free();
x2113.free();
x3318.free();
x2057.free();
x1665.free();
x571.free();
x666.free();
x1875.free();
x3559.free();
x3453.free();
x2493.free();
x244.free();
x449.free();
x954.free();
x897.free();
x578.free();
x279.free();
x502.free();
x3602.free();
x785.free();
x2443.free();
x245.free();
x2841.free();
x650.free();
x87.free();
x209.free();
x464.free();
x796.free();
x601.free();
x779.free();
x1043.free();
x2980.free();
x262.free();
x2260.free();
x593.free();
x450.free();
x3607.free();
x111.free();
x208.free();
x649.free();
x2690.free();
x2249.free();
x254.free();
x1870.free();
x2103.free();
x495.free();
x1134.free();
x3277.free();
x786.free();
x818.free();
x76.free();
x360.free();
x394.free();
x485.free();
x3272.free();
x907.free();
x1954.free();
x706.free();
x408.free();
x988.free();
x135.free();
x953.free();
x2402.free();
x185.free();
x1624.free();
x1730.free();
x932.free();
x471.free();
x174.free();
x2966.free();
x2396.free();
x1676.free();
x713.free();
x1243.free();
x3564.free();
x537.free();
x803.free();
x2549.free();
x1817.free();
x2682.free();
x802.free();
x3553.free();
x320.free();
x330.free();
x1584.free();
x2203.free();
x168.free();
x3082.free();
x778.free();
x611.free();
x304.free();
x1519.free();
x610.free();
x3323.free();
x1248.free();
x1487.free();
x2008.free();
x970.free();
x387.free();
x673.free();
x381.free();
x61.free();
x843.free();
x770.free();
x342.free();
x364.free();
x731.free();
x714.free();
x573.free();
x546.free();
x877.free();
x850.free();
x305.free();
x140.free();
x992.free();
x780.free();
x507.free();
x819.free();
x232.free();
x933.free();
x741.free();
x602.free();
x892.free();
x857.free();
x154.free();
x466.free();
x678.free();
x169.free();
x64.free();
x371.free();
x790.free();
x605.free();
x239.free();
x661.free();
x514.free();
x549.free();
x1000.free();
x1014.free();
x734.free();
x147.free();
x130.free();
x420.free();
x98.free();
x280.free();
x213.free();
x123.free();
x958.free();
x563.free();
x622.free();
x473.free();
x451.free();
x332.free();
x965.free();
x867.free();
x40.free();
x668.free();
x176.free();
x936.free();
x487.free();
x71.free();
x692.free();
x186.free();
x880.free();
x27.free();
x444.free();
x619.free();
x339.free();
x982.free();
x308.free();
x989.free();
x975.free();
x137.free();
x675.free();
x263.free();
x955.free();
x919.free();
x196.free();
x658.free();
x636.free();
x685.free();
x203.free();
x54.free();
x315.free();
x437.free();
x256.free();
x398.free();
x763.free();
x926.free();
x612.free();
x724.free();
x804.free();
x388.free();
x773.free();
x410.free();
x787.free();
x909.free();
x833.free();
x899.free();
x580.free();
x78.free();
x354.free();
x266.free();
x566.free();
x88.free();
x639.free();
x225.free();
x120.free();
x378.free();
x395.free();
x113.free();
x20.free();
x521.free();
x490.free();
x480.free();
x916.free();
x972.free();
x286.free();
x298.free();
x30.free();
x325.free();
x707.free();
x527.free();
x797.free();
x948.free();
x193.free();
x840.free();
x524.free();
x695.free();
x454.free();
x322.free();
x629.free();
x417.free();
x157.free();
x874.free();
x826.free();
x33.free();
x101.free();
x434.free();
x95.free();
x751.free();
x246.free();
x497.free();
x902.free();
x81.free();
x427.free();
x273.free();
x583.free();
x651.free();
x556.free();
x283.free();
x807.free();
x717.free();
x748.free();
x595.free();
x504.free();
x210.free();
x47.free();
x860.free();
x539.free();
x249.free();
x179.free();
x361.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void train() {
 for(int x5=0; x5<train_itr; x5++) {
JTensorFloatTuple x6 =  x1.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X396 = Cuda(Indicator(Y, 1000))
JCudaTensor x7;
JTensorFloat x8;
x8 = x4.asIndicator(1000);
x7 = x8.asJCudaTensor();

// val X198 = Cuda(X)
JCudaTensor x9;
JTensorFloat x10;
x10 = x3;
x9 = x10.asJCudaTensor();

// val X1140 = - X396.copy
JCudaTensor x11;
JCudaTensor x12;
float x13;
x12 = x7;
x12 = x12.clone();
x13 = -1;
x11 = x12.times_i(x13);

// val X199 = Convolv(2,3)(X198,1_cv_W,1_cv_B)
JCudaTensor x14;
JCudaTensor x15, x16, x17;
x15 = x9;
x16 = x18;
x17 = x19;
x14 = x20.forward(x15,x16,x17);

// val X200 = BatchNorm()(X199,1_bn_scale,1_bn_bias)
JCudaTensor x21;
JCudaTensor x22, x23, x24;
x22 = x14;
x23 = x25;
x24 = x26;
x21 = x27.forward(x22,x23,x24);

// val X201 = ReLU()(X200)
JCudaTensor x28;
JCudaTensor x29;
x29 = x21;
x28 = x30.forward(x29);

// val X202 = Pooling(3,2,0,true)(X201)
JCudaTensor x31;
JCudaTensor x32;
x32 = x28;
x31 = x33.forward(x32);

// val X203 = Convolv(1,0)(X202,2a1_cv_W,2a1_cv_B)
JCudaTensor x34;
JCudaTensor x35, x36, x37;
x35 = x31;
x36 = x38;
x37 = x39;
x34 = x40.forward(x35,x36,x37);

// val X206 = Convolv(1,0)(X202,2a2_a_cv_W,2a2_a_cv_B)
JCudaTensor x41;
JCudaTensor x42, x43, x44;
x42 = x31;
x43 = x45;
x44 = x46;
x41 = x47.forward(x42,x43,x44);

// val X204 = BatchNorm()(X203,2a1_bn_scale,2a1_bn_bias)
JCudaTensor x48;
JCudaTensor x49, x50, x51;
x49 = x34;
x50 = x52;
x51 = x53;
x48 = x54.forward(x49,x50,x51);

// val X207 = BatchNorm()(X206,2a2_a_bn_scale,2a2_a_bn_bias)
JCudaTensor x55;
JCudaTensor x56, x57, x58;
x56 = x41;
x57 = x59;
x58 = x60;
x55 = x61.forward(x56,x57,x58);

// val X208 = ReLU()(X207)
JCudaTensor x62;
JCudaTensor x63;
x63 = x55;
x62 = x64.forward(x63);

// val X209 = Convolv(1,1)(X208,2a2_b_cv_W,2a2_b_cv_B)
JCudaTensor x65;
JCudaTensor x66, x67, x68;
x66 = x62;
x67 = x69;
x68 = x70;
x65 = x71.forward(x66,x67,x68);

// val X210 = BatchNorm()(X209,2a2_b_bn_scale,2a2_b_bn_bias)
JCudaTensor x72;
JCudaTensor x73, x74, x75;
x73 = x65;
x74 = x76;
x75 = x77;
x72 = x78.forward(x73,x74,x75);

// val X211 = ReLU()(X210)
JCudaTensor x79;
JCudaTensor x80;
x80 = x72;
x79 = x81.forward(x80);

// val X212 = Convolv(1,0)(X211,2a2_c_cv_W,2a2_c_cv_B)
JCudaTensor x82;
JCudaTensor x83, x84, x85;
x83 = x79;
x84 = x86;
x85 = x87;
x82 = x88.forward(x83,x84,x85);

// val X213 = BatchNorm()(X212,2a2_c_bn_scale,2a2_c_bn_bias)
JCudaTensor x89;
JCudaTensor x90, x91, x92;
x90 = x82;
x91 = x93;
x92 = x94;
x89 = x95.forward(x90,x91,x92);

// val X205 = ReLU()(X204)
JCudaTensor x96;
JCudaTensor x97;
x97 = x48;
x96 = x98.forward(x97);

// val X214 = ReLU()(X213)
JCudaTensor x99;
JCudaTensor x100;
x100 = x89;
x99 = x101.forward(x100);

// val X215 = (X205.copy + X214)
JCudaTensor x102;
JCudaTensor x103, x104;
x103 = x96;
x103 = x103.clone();
x104 = x99;
x102 = x103.plus_i(x104);

// val X216 = ReLU()(X215)
JCudaTensor x105;
JCudaTensor x106;
x106 = x102;
x105 = x98.forward(x106);

// val X217 = Convolv(1,0)(X216,2b_a_cv_W,2b_a_cv_B)
JCudaTensor x107;
JCudaTensor x108, x109, x110;
x108 = x105;
x109 = x111;
x110 = x112;
x107 = x113.forward(x108,x109,x110);

// val X218 = BatchNorm()(X217,2b_a_bn_scale,2b_a_bn_bias)
JCudaTensor x114;
JCudaTensor x115, x116, x117;
x115 = x107;
x116 = x118;
x117 = x119;
x114 = x120.forward(x115,x116,x117);

// val X219 = ReLU()(X218)
JCudaTensor x121;
JCudaTensor x122;
x122 = x114;
x121 = x123.forward(x122);

// val X220 = Convolv(1,1)(X219,2b_b_cv_W,2b_b_cv_B)
JCudaTensor x124;
JCudaTensor x125, x126, x127;
x125 = x121;
x126 = x128;
x127 = x129;
x124 = x130.forward(x125,x126,x127);

// val X221 = BatchNorm()(X220,2b_b_bn_scale,2b_b_bn_bias)
JCudaTensor x131;
JCudaTensor x132, x133, x134;
x132 = x124;
x133 = x135;
x134 = x136;
x131 = x137.forward(x132,x133,x134);

// val X222 = ReLU()(X221)
JCudaTensor x138;
JCudaTensor x139;
x139 = x131;
x138 = x140.forward(x139);

// val X223 = Convolv(1,0)(X222,2b_c_cv_W,2b_c_cv_B)
JCudaTensor x141;
JCudaTensor x142, x143, x144;
x142 = x138;
x143 = x145;
x144 = x146;
x141 = x147.forward(x142,x143,x144);

// val X224 = BatchNorm()(X223,2b_c_bn_scale,2b_c_bn_bias)
JCudaTensor x148;
JCudaTensor x149, x150, x151;
x149 = x141;
x150 = x152;
x151 = x153;
x148 = x154.forward(x149,x150,x151);

// val X225 = ReLU()(X224)
JCudaTensor x155;
JCudaTensor x156;
x156 = x148;
x155 = x157.forward(x156);

// val X226 = (X225.copy + X216)
JCudaTensor x158;
JCudaTensor x159, x160;
x159 = x155;
x159 = x159.clone();
x160 = x105;
x158 = x159.plus_i(x160);

// val X227 = ReLU()(X226)
JCudaTensor x161;
JCudaTensor x162;
x162 = x158;
x161 = x157.forward(x162);

// val X228 = Convolv(1,0)(X227,2c_a_cv_W,2c_a_cv_B)
JCudaTensor x163;
JCudaTensor x164, x165, x166;
x164 = x161;
x165 = x167;
x166 = x168;
x163 = x169.forward(x164,x165,x166);

// val X229 = BatchNorm()(X228,2c_a_bn_scale,2c_a_bn_bias)
JCudaTensor x170;
JCudaTensor x171, x172, x173;
x171 = x163;
x172 = x174;
x173 = x175;
x170 = x176.forward(x171,x172,x173);

// val X230 = ReLU()(X229)
JCudaTensor x177;
JCudaTensor x178;
x178 = x170;
x177 = x179.forward(x178);

// val X231 = Convolv(1,1)(X230,2c_b_cv_W,2c_b_cv_B)
JCudaTensor x180;
JCudaTensor x181, x182, x183;
x181 = x177;
x182 = x184;
x183 = x185;
x180 = x186.forward(x181,x182,x183);

// val X232 = BatchNorm()(X231,2c_b_bn_scale,2c_b_bn_bias)
JCudaTensor x187;
JCudaTensor x188, x189, x190;
x188 = x180;
x189 = x191;
x190 = x192;
x187 = x193.forward(x188,x189,x190);

// val X233 = ReLU()(X232)
JCudaTensor x194;
JCudaTensor x195;
x195 = x187;
x194 = x196.forward(x195);

// val X234 = Convolv(1,0)(X233,2c_c_cv_W,2c_c_cv_B)
JCudaTensor x197;
JCudaTensor x198, x199, x200;
x198 = x194;
x199 = x201;
x200 = x202;
x197 = x203.forward(x198,x199,x200);

// val X235 = BatchNorm()(X234,2c_c_bn_scale,2c_c_bn_bias)
JCudaTensor x204;
JCudaTensor x205, x206, x207;
x205 = x197;
x206 = x208;
x207 = x209;
x204 = x210.forward(x205,x206,x207);

// val X236 = ReLU()(X235)
JCudaTensor x211;
JCudaTensor x212;
x212 = x204;
x211 = x213.forward(x212);

// val X237 = (X236.copy + X227)
JCudaTensor x214;
JCudaTensor x215, x216;
x215 = x211;
x215 = x215.clone();
x216 = x161;
x214 = x215.plus_i(x216);

// val X238 = ReLU()(X237)
JCudaTensor x217;
JCudaTensor x218;
x218 = x214;
x217 = x213.forward(x218);

// val X242 = Convolv(2,0)(X238,3a2_a_cv_W,3a2_a_cv_B)
JCudaTensor x219;
JCudaTensor x220, x221, x222;
x220 = x217;
x221 = x223;
x222 = x224;
x219 = x225.forward(x220,x221,x222);

// val X239 = Convolv(2,0)(X238,3a1_cv_W,3a1_cv_B)
JCudaTensor x226;
JCudaTensor x227, x228, x229;
x227 = x217;
x228 = x230;
x229 = x231;
x226 = x232.forward(x227,x228,x229);

// val X240 = BatchNorm()(X239,3a1_bn_scale,3a1_bn_bias)
JCudaTensor x233;
JCudaTensor x234, x235, x236;
x234 = x226;
x235 = x237;
x236 = x238;
x233 = x239.forward(x234,x235,x236);

// val X243 = BatchNorm()(X242,3a2_a_bn_scale,3a2_a_bn_bias)
JCudaTensor x240;
JCudaTensor x241, x242, x243;
x241 = x219;
x242 = x244;
x243 = x245;
x240 = x246.forward(x241,x242,x243);

// val X244 = ReLU()(X243)
JCudaTensor x247;
JCudaTensor x248;
x248 = x240;
x247 = x249.forward(x248);

// val X245 = Convolv(1,1)(X244,3a2_b_cv_W,3a2_b_cv_B)
JCudaTensor x250;
JCudaTensor x251, x252, x253;
x251 = x247;
x252 = x254;
x253 = x255;
x250 = x256.forward(x251,x252,x253);

// val X246 = BatchNorm()(X245,3a2_b_bn_scale,3a2_b_bn_bias)
JCudaTensor x257;
JCudaTensor x258, x259, x260;
x258 = x250;
x259 = x261;
x260 = x262;
x257 = x263.forward(x258,x259,x260);

// val X247 = ReLU()(X246)
JCudaTensor x264;
JCudaTensor x265;
x265 = x257;
x264 = x266.forward(x265);

// val X248 = Convolv(1,0)(X247,3a2_c_cv_W,3a2_c_cv_B)
JCudaTensor x267;
JCudaTensor x268, x269, x270;
x268 = x264;
x269 = x271;
x270 = x272;
x267 = x273.forward(x268,x269,x270);

// val X249 = BatchNorm()(X248,3a2_c_bn_scale,3a2_c_bn_bias)
JCudaTensor x274;
JCudaTensor x275, x276, x277;
x275 = x267;
x276 = x278;
x277 = x279;
x274 = x280.forward(x275,x276,x277);

// val X241 = ReLU()(X240)
JCudaTensor x281;
JCudaTensor x282;
x282 = x233;
x281 = x283.forward(x282);

// val X250 = ReLU()(X249)
JCudaTensor x284;
JCudaTensor x285;
x285 = x274;
x284 = x286.forward(x285);

// val X251 = (X241.copy + X250)
JCudaTensor x287;
JCudaTensor x288, x289;
x288 = x281;
x288 = x288.clone();
x289 = x284;
x287 = x288.plus_i(x289);

// val X252 = ReLU()(X251)
JCudaTensor x290;
JCudaTensor x291;
x291 = x287;
x290 = x283.forward(x291);

// val X253 = Convolv(1,0)(X252,3b_a_cv_W,3b_a_cv_B)
JCudaTensor x292;
JCudaTensor x293, x294, x295;
x293 = x290;
x294 = x296;
x295 = x297;
x292 = x298.forward(x293,x294,x295);

// val X254 = BatchNorm()(X253,3b_a_bn_scale,3b_a_bn_bias)
JCudaTensor x299;
JCudaTensor x300, x301, x302;
x300 = x292;
x301 = x303;
x302 = x304;
x299 = x305.forward(x300,x301,x302);

// val X255 = ReLU()(X254)
JCudaTensor x306;
JCudaTensor x307;
x307 = x299;
x306 = x308.forward(x307);

// val X256 = Convolv(1,1)(X255,3b_b_cv_W,3b_b_cv_B)
JCudaTensor x309;
JCudaTensor x310, x311, x312;
x310 = x306;
x311 = x313;
x312 = x314;
x309 = x315.forward(x310,x311,x312);

// val X257 = BatchNorm()(X256,3b_b_bn_scale,3b_b_bn_bias)
JCudaTensor x316;
JCudaTensor x317, x318, x319;
x317 = x309;
x318 = x320;
x319 = x321;
x316 = x322.forward(x317,x318,x319);

// val X258 = ReLU()(X257)
JCudaTensor x323;
JCudaTensor x324;
x324 = x316;
x323 = x325.forward(x324);

// val X259 = Convolv(1,0)(X258,3b_c_cv_W,3b_c_cv_B)
JCudaTensor x326;
JCudaTensor x327, x328, x329;
x327 = x323;
x328 = x330;
x329 = x331;
x326 = x332.forward(x327,x328,x329);

// val X260 = BatchNorm()(X259,3b_c_bn_scale,3b_c_bn_bias)
JCudaTensor x333;
JCudaTensor x334, x335, x336;
x334 = x326;
x335 = x337;
x336 = x338;
x333 = x339.forward(x334,x335,x336);

// val X261 = ReLU()(X260)
JCudaTensor x340;
JCudaTensor x341;
x341 = x333;
x340 = x342.forward(x341);

// val X262 = (X261.copy + X252)
JCudaTensor x343;
JCudaTensor x344, x345;
x344 = x340;
x344 = x344.clone();
x345 = x290;
x343 = x344.plus_i(x345);

// val X263 = ReLU()(X262)
JCudaTensor x346;
JCudaTensor x347;
x347 = x343;
x346 = x342.forward(x347);

// val X264 = Convolv(1,0)(X263,3c_a_cv_W,3c_a_cv_B)
JCudaTensor x348;
JCudaTensor x349, x350, x351;
x349 = x346;
x350 = x352;
x351 = x353;
x348 = x354.forward(x349,x350,x351);

// val X265 = BatchNorm()(X264,3c_a_bn_scale,3c_a_bn_bias)
JCudaTensor x355;
JCudaTensor x356, x357, x358;
x356 = x348;
x357 = x359;
x358 = x360;
x355 = x361.forward(x356,x357,x358);

// val X266 = ReLU()(X265)
JCudaTensor x362;
JCudaTensor x363;
x363 = x355;
x362 = x364.forward(x363);

// val X267 = Convolv(1,1)(X266,3c_b_cv_W,3c_b_cv_B)
JCudaTensor x365;
JCudaTensor x366, x367, x368;
x366 = x362;
x367 = x369;
x368 = x370;
x365 = x371.forward(x366,x367,x368);

// val X268 = BatchNorm()(X267,3c_b_bn_scale,3c_b_bn_bias)
JCudaTensor x372;
JCudaTensor x373, x374, x375;
x373 = x365;
x374 = x376;
x375 = x377;
x372 = x378.forward(x373,x374,x375);

// val X269 = ReLU()(X268)
JCudaTensor x379;
JCudaTensor x380;
x380 = x372;
x379 = x381.forward(x380);

// val X270 = Convolv(1,0)(X269,3c_c_cv_W,3c_c_cv_B)
JCudaTensor x382;
JCudaTensor x383, x384, x385;
x383 = x379;
x384 = x386;
x385 = x387;
x382 = x388.forward(x383,x384,x385);

// val X271 = BatchNorm()(X270,3c_c_bn_scale,3c_c_bn_bias)
JCudaTensor x389;
JCudaTensor x390, x391, x392;
x390 = x382;
x391 = x393;
x392 = x394;
x389 = x395.forward(x390,x391,x392);

// val X272 = ReLU()(X271)
JCudaTensor x396;
JCudaTensor x397;
x397 = x389;
x396 = x398.forward(x397);

// val X273 = (X272.copy + X263)
JCudaTensor x399;
JCudaTensor x400, x401;
x400 = x396;
x400 = x400.clone();
x401 = x346;
x399 = x400.plus_i(x401);

// val X274 = ReLU()(X273)
JCudaTensor x402;
JCudaTensor x403;
x403 = x399;
x402 = x398.forward(x403);

// val X275 = Convolv(1,0)(X274,3d_a_cv_W,3d_a_cv_B)
JCudaTensor x404;
JCudaTensor x405, x406, x407;
x405 = x402;
x406 = x408;
x407 = x409;
x404 = x410.forward(x405,x406,x407);

// val X276 = BatchNorm()(X275,3d_a_bn_scale,3d_a_bn_bias)
JCudaTensor x411;
JCudaTensor x412, x413, x414;
x412 = x404;
x413 = x415;
x414 = x416;
x411 = x417.forward(x412,x413,x414);

// val X277 = ReLU()(X276)
JCudaTensor x418;
JCudaTensor x419;
x419 = x411;
x418 = x420.forward(x419);

// val X278 = Convolv(1,1)(X277,3d_b_cv_W,3d_b_cv_B)
JCudaTensor x421;
JCudaTensor x422, x423, x424;
x422 = x418;
x423 = x425;
x424 = x426;
x421 = x427.forward(x422,x423,x424);

// val X279 = BatchNorm()(X278,3d_b_bn_scale,3d_b_bn_bias)
JCudaTensor x428;
JCudaTensor x429, x430, x431;
x429 = x421;
x430 = x432;
x431 = x433;
x428 = x434.forward(x429,x430,x431);

// val X280 = ReLU()(X279)
JCudaTensor x435;
JCudaTensor x436;
x436 = x428;
x435 = x437.forward(x436);

// val X281 = Convolv(1,0)(X280,3d_c_cv_W,3d_c_cv_B)
JCudaTensor x438;
JCudaTensor x439, x440, x441;
x439 = x435;
x440 = x442;
x441 = x443;
x438 = x444.forward(x439,x440,x441);

// val X282 = BatchNorm()(X281,3d_c_bn_scale,3d_c_bn_bias)
JCudaTensor x445;
JCudaTensor x446, x447, x448;
x446 = x438;
x447 = x449;
x448 = x450;
x445 = x451.forward(x446,x447,x448);

// val X283 = ReLU()(X282)
JCudaTensor x452;
JCudaTensor x453;
x453 = x445;
x452 = x454.forward(x453);

// val X284 = (X283.copy + X274)
JCudaTensor x455;
JCudaTensor x456, x457;
x456 = x452;
x456 = x456.clone();
x457 = x402;
x455 = x456.plus_i(x457);

// val X285 = ReLU()(X284)
JCudaTensor x458;
JCudaTensor x459;
x459 = x455;
x458 = x454.forward(x459);

// val X286 = Convolv(2,0)(X285,4a1_cv_W,4a1_cv_B)
JCudaTensor x460;
JCudaTensor x461, x462, x463;
x461 = x458;
x462 = x464;
x463 = x465;
x460 = x466.forward(x461,x462,x463);

// val X289 = Convolv(2,0)(X285,4a2_a_cv_W,4a2_a_cv_B)
JCudaTensor x467;
JCudaTensor x468, x469, x470;
x468 = x458;
x469 = x471;
x470 = x472;
x467 = x473.forward(x468,x469,x470);

// val X287 = BatchNorm()(X286,4a1_bn_scale,4a1_bn_bias)
JCudaTensor x474;
JCudaTensor x475, x476, x477;
x475 = x460;
x476 = x478;
x477 = x479;
x474 = x480.forward(x475,x476,x477);

// val X290 = BatchNorm()(X289,4a2_a_bn_scale,4a2_a_bn_bias)
JCudaTensor x481;
JCudaTensor x482, x483, x484;
x482 = x467;
x483 = x485;
x484 = x486;
x481 = x487.forward(x482,x483,x484);

// val X291 = ReLU()(X290)
JCudaTensor x488;
JCudaTensor x489;
x489 = x481;
x488 = x490.forward(x489);

// val X292 = Convolv(1,1)(X291,4a2_b_cv_W,4a2_b_cv_B)
JCudaTensor x491;
JCudaTensor x492, x493, x494;
x492 = x488;
x493 = x495;
x494 = x496;
x491 = x497.forward(x492,x493,x494);

// val X293 = BatchNorm()(X292,4a2_b_bn_scale,4a2_b_bn_bias)
JCudaTensor x498;
JCudaTensor x499, x500, x501;
x499 = x491;
x500 = x502;
x501 = x503;
x498 = x504.forward(x499,x500,x501);

// val X294 = ReLU()(X293)
JCudaTensor x505;
JCudaTensor x506;
x506 = x498;
x505 = x507.forward(x506);

// val X295 = Convolv(1,0)(X294,4a2_c_cv_W,4a2_c_cv_B)
JCudaTensor x508;
JCudaTensor x509, x510, x511;
x509 = x505;
x510 = x512;
x511 = x513;
x508 = x514.forward(x509,x510,x511);

// val X296 = BatchNorm()(X295,4a2_c_bn_scale,4a2_c_bn_bias)
JCudaTensor x515;
JCudaTensor x516, x517, x518;
x516 = x508;
x517 = x519;
x518 = x520;
x515 = x521.forward(x516,x517,x518);

// val X288 = ReLU()(X287)
JCudaTensor x522;
JCudaTensor x523;
x523 = x474;
x522 = x524.forward(x523);

// val X297 = ReLU()(X296)
JCudaTensor x525;
JCudaTensor x526;
x526 = x515;
x525 = x527.forward(x526);

// val X298 = (X288.copy + X297)
JCudaTensor x528;
JCudaTensor x529, x530;
x529 = x522;
x529 = x529.clone();
x530 = x525;
x528 = x529.plus_i(x530);

// val X299 = ReLU()(X298)
JCudaTensor x531;
JCudaTensor x532;
x532 = x528;
x531 = x524.forward(x532);

// val X300 = Convolv(1,0)(X299,4b_a_cv_W,4b_a_cv_B)
JCudaTensor x533;
JCudaTensor x534, x535, x536;
x534 = x531;
x535 = x537;
x536 = x538;
x533 = x539.forward(x534,x535,x536);

// val X301 = BatchNorm()(X300,4b_a_bn_scale,4b_a_bn_bias)
JCudaTensor x540;
JCudaTensor x541, x542, x543;
x541 = x533;
x542 = x544;
x543 = x545;
x540 = x546.forward(x541,x542,x543);

// val X302 = ReLU()(X301)
JCudaTensor x547;
JCudaTensor x548;
x548 = x540;
x547 = x549.forward(x548);

// val X303 = Convolv(1,1)(X302,4b_b_cv_W,4b_b_cv_B)
JCudaTensor x550;
JCudaTensor x551, x552, x553;
x551 = x547;
x552 = x554;
x553 = x555;
x550 = x556.forward(x551,x552,x553);

// val X304 = BatchNorm()(X303,4b_b_bn_scale,4b_b_bn_bias)
JCudaTensor x557;
JCudaTensor x558, x559, x560;
x558 = x550;
x559 = x561;
x560 = x562;
x557 = x563.forward(x558,x559,x560);

// val X305 = ReLU()(X304)
JCudaTensor x564;
JCudaTensor x565;
x565 = x557;
x564 = x566.forward(x565);

// val X306 = Convolv(1,0)(X305,4b_c_cv_W,4b_c_cv_B)
JCudaTensor x567;
JCudaTensor x568, x569, x570;
x568 = x564;
x569 = x571;
x570 = x572;
x567 = x573.forward(x568,x569,x570);

// val X307 = BatchNorm()(X306,4b_c_bn_scale,4b_c_bn_bias)
JCudaTensor x574;
JCudaTensor x575, x576, x577;
x575 = x567;
x576 = x578;
x577 = x579;
x574 = x580.forward(x575,x576,x577);

// val X308 = ReLU()(X307)
JCudaTensor x581;
JCudaTensor x582;
x582 = x574;
x581 = x583.forward(x582);

// val X309 = (X308.copy + X299)
JCudaTensor x584;
JCudaTensor x585, x586;
x585 = x581;
x585 = x585.clone();
x586 = x531;
x584 = x585.plus_i(x586);

// val X310 = ReLU()(X309)
JCudaTensor x587;
JCudaTensor x588;
x588 = x584;
x587 = x583.forward(x588);

// val X311 = Convolv(1,0)(X310,4c_a_cv_W,4c_a_cv_B)
JCudaTensor x589;
JCudaTensor x590, x591, x592;
x590 = x587;
x591 = x593;
x592 = x594;
x589 = x595.forward(x590,x591,x592);

// val X312 = BatchNorm()(X311,4c_a_bn_scale,4c_a_bn_bias)
JCudaTensor x596;
JCudaTensor x597, x598, x599;
x597 = x589;
x598 = x600;
x599 = x601;
x596 = x602.forward(x597,x598,x599);

// val X313 = ReLU()(X312)
JCudaTensor x603;
JCudaTensor x604;
x604 = x596;
x603 = x605.forward(x604);

// val X314 = Convolv(1,1)(X313,4c_b_cv_W,4c_b_cv_B)
JCudaTensor x606;
JCudaTensor x607, x608, x609;
x607 = x603;
x608 = x610;
x609 = x611;
x606 = x612.forward(x607,x608,x609);

// val X315 = BatchNorm()(X314,4c_b_bn_scale,4c_b_bn_bias)
JCudaTensor x613;
JCudaTensor x614, x615, x616;
x614 = x606;
x615 = x617;
x616 = x618;
x613 = x619.forward(x614,x615,x616);

// val X316 = ReLU()(X315)
JCudaTensor x620;
JCudaTensor x621;
x621 = x613;
x620 = x622.forward(x621);

// val X317 = Convolv(1,0)(X316,4c_c_cv_W,4c_c_cv_B)
JCudaTensor x623;
JCudaTensor x624, x625, x626;
x624 = x620;
x625 = x627;
x626 = x628;
x623 = x629.forward(x624,x625,x626);

// val X318 = BatchNorm()(X317,4c_c_bn_scale,4c_c_bn_bias)
JCudaTensor x630;
JCudaTensor x631, x632, x633;
x631 = x623;
x632 = x634;
x633 = x635;
x630 = x636.forward(x631,x632,x633);

// val X319 = ReLU()(X318)
JCudaTensor x637;
JCudaTensor x638;
x638 = x630;
x637 = x639.forward(x638);

// val X320 = (X319.copy + X310)
JCudaTensor x640;
JCudaTensor x641, x642;
x641 = x637;
x641 = x641.clone();
x642 = x587;
x640 = x641.plus_i(x642);

// val X321 = ReLU()(X320)
JCudaTensor x643;
JCudaTensor x644;
x644 = x640;
x643 = x639.forward(x644);

// val X322 = Convolv(1,0)(X321,4d_a_cv_W,4d_a_cv_B)
JCudaTensor x645;
JCudaTensor x646, x647, x648;
x646 = x643;
x647 = x649;
x648 = x650;
x645 = x651.forward(x646,x647,x648);

// val X323 = BatchNorm()(X322,4d_a_bn_scale,4d_a_bn_bias)
JCudaTensor x652;
JCudaTensor x653, x654, x655;
x653 = x645;
x654 = x656;
x655 = x657;
x652 = x658.forward(x653,x654,x655);

// val X324 = ReLU()(X323)
JCudaTensor x659;
JCudaTensor x660;
x660 = x652;
x659 = x661.forward(x660);

// val X325 = Convolv(1,1)(X324,4d_b_cv_W,4d_b_cv_B)
JCudaTensor x662;
JCudaTensor x663, x664, x665;
x663 = x659;
x664 = x666;
x665 = x667;
x662 = x668.forward(x663,x664,x665);

// val X326 = BatchNorm()(X325,4d_b_bn_scale,4d_b_bn_bias)
JCudaTensor x669;
JCudaTensor x670, x671, x672;
x670 = x662;
x671 = x673;
x672 = x674;
x669 = x675.forward(x670,x671,x672);

// val X327 = ReLU()(X326)
JCudaTensor x676;
JCudaTensor x677;
x677 = x669;
x676 = x678.forward(x677);

// val X328 = Convolv(1,0)(X327,4d_c_cv_W,4d_c_cv_B)
JCudaTensor x679;
JCudaTensor x680, x681, x682;
x680 = x676;
x681 = x683;
x682 = x684;
x679 = x685.forward(x680,x681,x682);

// val X329 = BatchNorm()(X328,4d_c_bn_scale,4d_c_bn_bias)
JCudaTensor x686;
JCudaTensor x687, x688, x689;
x687 = x679;
x688 = x690;
x689 = x691;
x686 = x692.forward(x687,x688,x689);

// val X330 = ReLU()(X329)
JCudaTensor x693;
JCudaTensor x694;
x694 = x686;
x693 = x695.forward(x694);

// val X331 = (X330.copy + X321)
JCudaTensor x696;
JCudaTensor x697, x698;
x697 = x693;
x697 = x697.clone();
x698 = x643;
x696 = x697.plus_i(x698);

// val X332 = ReLU()(X331)
JCudaTensor x699;
JCudaTensor x700;
x700 = x696;
x699 = x695.forward(x700);

// val X333 = Convolv(1,0)(X332,4e_a_cv_W,4e_a_cv_B)
JCudaTensor x701;
JCudaTensor x702, x703, x704;
x702 = x699;
x703 = x705;
x704 = x706;
x701 = x707.forward(x702,x703,x704);

// val X334 = BatchNorm()(X333,4e_a_bn_scale,4e_a_bn_bias)
JCudaTensor x708;
JCudaTensor x709, x710, x711;
x709 = x701;
x710 = x712;
x711 = x713;
x708 = x714.forward(x709,x710,x711);

// val X335 = ReLU()(X334)
JCudaTensor x715;
JCudaTensor x716;
x716 = x708;
x715 = x717.forward(x716);

// val X336 = Convolv(1,1)(X335,4e_b_cv_W,4e_b_cv_B)
JCudaTensor x718;
JCudaTensor x719, x720, x721;
x719 = x715;
x720 = x722;
x721 = x723;
x718 = x724.forward(x719,x720,x721);

// val X337 = BatchNorm()(X336,4e_b_bn_scale,4e_b_bn_bias)
JCudaTensor x725;
JCudaTensor x726, x727, x728;
x726 = x718;
x727 = x729;
x728 = x730;
x725 = x731.forward(x726,x727,x728);

// val X338 = ReLU()(X337)
JCudaTensor x732;
JCudaTensor x733;
x733 = x725;
x732 = x734.forward(x733);

// val X339 = Convolv(1,0)(X338,4e_c_cv_W,4e_c_cv_B)
JCudaTensor x735;
JCudaTensor x736, x737, x738;
x736 = x732;
x737 = x739;
x738 = x740;
x735 = x741.forward(x736,x737,x738);

// val X340 = BatchNorm()(X339,4e_c_bn_scale,4e_c_bn_bias)
JCudaTensor x742;
JCudaTensor x743, x744, x745;
x743 = x735;
x744 = x746;
x745 = x747;
x742 = x748.forward(x743,x744,x745);

// val X341 = ReLU()(X340)
JCudaTensor x749;
JCudaTensor x750;
x750 = x742;
x749 = x751.forward(x750);

// val X342 = (X341.copy + X332)
JCudaTensor x752;
JCudaTensor x753, x754;
x753 = x749;
x753 = x753.clone();
x754 = x699;
x752 = x753.plus_i(x754);

// val X343 = ReLU()(X342)
JCudaTensor x755;
JCudaTensor x756;
x756 = x752;
x755 = x751.forward(x756);

// val X344 = Convolv(1,0)(X343,4f_a_cv_W,4f_a_cv_B)
JCudaTensor x757;
JCudaTensor x758, x759, x760;
x758 = x755;
x759 = x761;
x760 = x762;
x757 = x763.forward(x758,x759,x760);

// val X345 = BatchNorm()(X344,4f_a_bn_scale,4f_a_bn_bias)
JCudaTensor x764;
JCudaTensor x765, x766, x767;
x765 = x757;
x766 = x768;
x767 = x769;
x764 = x770.forward(x765,x766,x767);

// val X346 = ReLU()(X345)
JCudaTensor x771;
JCudaTensor x772;
x772 = x764;
x771 = x773.forward(x772);

// val X347 = Convolv(1,1)(X346,4f_b_cv_W,4f_b_cv_B)
JCudaTensor x774;
JCudaTensor x775, x776, x777;
x775 = x771;
x776 = x778;
x777 = x779;
x774 = x780.forward(x775,x776,x777);

// val X348 = BatchNorm()(X347,4f_b_bn_scale,4f_b_bn_bias)
JCudaTensor x781;
JCudaTensor x782, x783, x784;
x782 = x774;
x783 = x785;
x784 = x786;
x781 = x787.forward(x782,x783,x784);

// val X349 = ReLU()(X348)
JCudaTensor x788;
JCudaTensor x789;
x789 = x781;
x788 = x790.forward(x789);

// val X350 = Convolv(1,0)(X349,4f_c_cv_W,4f_c_cv_B)
JCudaTensor x791;
JCudaTensor x792, x793, x794;
x792 = x788;
x793 = x795;
x794 = x796;
x791 = x797.forward(x792,x793,x794);

// val X351 = BatchNorm()(X350,4f_c_bn_scale,4f_c_bn_bias)
JCudaTensor x798;
JCudaTensor x799, x800, x801;
x799 = x791;
x800 = x802;
x801 = x803;
x798 = x804.forward(x799,x800,x801);

// val X352 = ReLU()(X351)
JCudaTensor x805;
JCudaTensor x806;
x806 = x798;
x805 = x807.forward(x806);

// val X353 = (X352.copy + X343)
JCudaTensor x808;
JCudaTensor x809, x810;
x809 = x805;
x809 = x809.clone();
x810 = x755;
x808 = x809.plus_i(x810);

// val X354 = ReLU()(X353)
JCudaTensor x811;
JCudaTensor x812;
x812 = x808;
x811 = x807.forward(x812);

// val X355 = Convolv(2,0)(X354,5a1_cv_W,5a1_cv_B)
JCudaTensor x813;
JCudaTensor x814, x815, x816;
x814 = x811;
x815 = x817;
x816 = x818;
x813 = x819.forward(x814,x815,x816);

// val X358 = Convolv(2,0)(X354,5a2_a_cv_W,5a2_a_cv_B)
JCudaTensor x820;
JCudaTensor x821, x822, x823;
x821 = x811;
x822 = x824;
x823 = x825;
x820 = x826.forward(x821,x822,x823);

// val X359 = BatchNorm()(X358,5a2_a_bn_scale,5a2_a_bn_bias)
JCudaTensor x827;
JCudaTensor x828, x829, x830;
x828 = x820;
x829 = x831;
x830 = x832;
x827 = x833.forward(x828,x829,x830);

// val X356 = BatchNorm()(X355,5a1_bn_scale,5a1_bn_bias)
JCudaTensor x834;
JCudaTensor x835, x836, x837;
x835 = x813;
x836 = x838;
x837 = x839;
x834 = x840.forward(x835,x836,x837);

// val X360 = ReLU()(X359)
JCudaTensor x841;
JCudaTensor x842;
x842 = x827;
x841 = x843.forward(x842);

// val X361 = Convolv(1,1)(X360,5a2_b_cv_W,5a2_b_cv_B)
JCudaTensor x844;
JCudaTensor x845, x846, x847;
x845 = x841;
x846 = x848;
x847 = x849;
x844 = x850.forward(x845,x846,x847);

// val X362 = BatchNorm()(X361,5a2_b_bn_scale,5a2_b_bn_bias)
JCudaTensor x851;
JCudaTensor x852, x853, x854;
x852 = x844;
x853 = x855;
x854 = x856;
x851 = x857.forward(x852,x853,x854);

// val X363 = ReLU()(X362)
JCudaTensor x858;
JCudaTensor x859;
x859 = x851;
x858 = x860.forward(x859);

// val X364 = Convolv(1,0)(X363,5a2_c_cv_W,5a2_c_cv_B)
JCudaTensor x861;
JCudaTensor x862, x863, x864;
x862 = x858;
x863 = x865;
x864 = x866;
x861 = x867.forward(x862,x863,x864);

// val X365 = BatchNorm()(X364,5a2_c_bn_scale,5a2_c_bn_bias)
JCudaTensor x868;
JCudaTensor x869, x870, x871;
x869 = x861;
x870 = x872;
x871 = x873;
x868 = x874.forward(x869,x870,x871);

// val X357 = ReLU()(X356)
JCudaTensor x875;
JCudaTensor x876;
x876 = x834;
x875 = x877.forward(x876);

// val X366 = ReLU()(X365)
JCudaTensor x878;
JCudaTensor x879;
x879 = x868;
x878 = x880.forward(x879);

// val X367 = (X357.copy + X366)
JCudaTensor x881;
JCudaTensor x882, x883;
x882 = x875;
x882 = x882.clone();
x883 = x878;
x881 = x882.plus_i(x883);

// val X368 = ReLU()(X367)
JCudaTensor x884;
JCudaTensor x885;
x885 = x881;
x884 = x877.forward(x885);

// val X369 = Convolv(1,0)(X368,5b_a_cv_W,5b_a_cv_B)
JCudaTensor x886;
JCudaTensor x887, x888, x889;
x887 = x884;
x888 = x890;
x889 = x891;
x886 = x892.forward(x887,x888,x889);

// val X370 = BatchNorm()(X369,5b_a_bn_scale,5b_a_bn_bias)
JCudaTensor x893;
JCudaTensor x894, x895, x896;
x894 = x886;
x895 = x897;
x896 = x898;
x893 = x899.forward(x894,x895,x896);

// val X371 = ReLU()(X370)
JCudaTensor x900;
JCudaTensor x901;
x901 = x893;
x900 = x902.forward(x901);

// val X372 = Convolv(1,1)(X371,5b_b_cv_W,5b_b_cv_B)
JCudaTensor x903;
JCudaTensor x904, x905, x906;
x904 = x900;
x905 = x907;
x906 = x908;
x903 = x909.forward(x904,x905,x906);

// val X373 = BatchNorm()(X372,5b_b_bn_scale,5b_b_bn_bias)
JCudaTensor x910;
JCudaTensor x911, x912, x913;
x911 = x903;
x912 = x914;
x913 = x915;
x910 = x916.forward(x911,x912,x913);

// val X374 = ReLU()(X373)
JCudaTensor x917;
JCudaTensor x918;
x918 = x910;
x917 = x919.forward(x918);

// val X375 = Convolv(1,0)(X374,5b_c_cv_W,5b_c_cv_B)
JCudaTensor x920;
JCudaTensor x921, x922, x923;
x921 = x917;
x922 = x924;
x923 = x925;
x920 = x926.forward(x921,x922,x923);

// val X376 = BatchNorm()(X375,5b_c_bn_scale,5b_c_bn_bias)
JCudaTensor x927;
JCudaTensor x928, x929, x930;
x928 = x920;
x929 = x931;
x930 = x932;
x927 = x933.forward(x928,x929,x930);

// val X377 = ReLU()(X376)
JCudaTensor x934;
JCudaTensor x935;
x935 = x927;
x934 = x936.forward(x935);

// val X378 = (X377.copy + X368)
JCudaTensor x937;
JCudaTensor x938, x939;
x938 = x934;
x938 = x938.clone();
x939 = x884;
x937 = x938.plus_i(x939);

// val X379 = ReLU()(X378)
JCudaTensor x940;
JCudaTensor x941;
x941 = x937;
x940 = x936.forward(x941);

// val X380 = Convolv(1,0)(X379,5c_a_cv_W,5c_a_cv_B)
JCudaTensor x942;
JCudaTensor x943, x944, x945;
x943 = x940;
x944 = x946;
x945 = x947;
x942 = x948.forward(x943,x944,x945);

// val X381 = BatchNorm()(X380,5c_a_bn_scale,5c_a_bn_bias)
JCudaTensor x949;
JCudaTensor x950, x951, x952;
x950 = x942;
x951 = x953;
x952 = x954;
x949 = x955.forward(x950,x951,x952);

// val X382 = ReLU()(X381)
JCudaTensor x956;
JCudaTensor x957;
x957 = x949;
x956 = x958.forward(x957);

// val X383 = Convolv(1,1)(X382,5c_b_cv_W,5c_b_cv_B)
JCudaTensor x959;
JCudaTensor x960, x961, x962;
x960 = x956;
x961 = x963;
x962 = x964;
x959 = x965.forward(x960,x961,x962);

// val X384 = BatchNorm()(X383,5c_b_bn_scale,5c_b_bn_bias)
JCudaTensor x966;
JCudaTensor x967, x968, x969;
x967 = x959;
x968 = x970;
x969 = x971;
x966 = x972.forward(x967,x968,x969);

// val X385 = ReLU()(X384)
JCudaTensor x973;
JCudaTensor x974;
x974 = x966;
x973 = x975.forward(x974);

// val X386 = Convolv(1,0)(X385,5c_c_cv_W,5c_c_cv_B)
JCudaTensor x976;
JCudaTensor x977, x978, x979;
x977 = x973;
x978 = x980;
x979 = x981;
x976 = x982.forward(x977,x978,x979);

// val X387 = BatchNorm()(X386,5c_c_bn_scale,5c_c_bn_bias)
JCudaTensor x983;
JCudaTensor x984, x985, x986;
x984 = x976;
x985 = x987;
x986 = x988;
x983 = x989.forward(x984,x985,x986);

// val X388 = ReLU()(X387)
JCudaTensor x990;
JCudaTensor x991;
x991 = x983;
x990 = x992.forward(x991);

// val X389 = (X388.copy + X379)
JCudaTensor x993;
JCudaTensor x994, x995;
x994 = x990;
x994 = x994.clone();
x995 = x940;
x993 = x994.plus_i(x995);

// val X390 = ReLU()(X389)
JCudaTensor x996;
JCudaTensor x997;
x997 = x993;
x996 = x992.forward(x997);

// val X391 = Pooling(7,1,0,false)(X390)
JCudaTensor x998;
JCudaTensor x999;
x999 = x996;
x998 = x1000.forward(x999);

// val X392 = (X391[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x1001;
JCudaMatrix x1002;
JCudaMatrix x1003;
JCudaTensor x1004;
JCudaTensor x1005;
x1005 = x998;
x1004 = x1005.flatten(1, new int[]{2048, 1, 1});
x1002 = x1004.asMatrix(1, true);
JCudaTensor x1006;
x1006 = x1007;
x1003 = x1006.asMatrix(1, true);
x1001 = x1002.times(x1003);

// val X394 = (X392 + (i) => fc_B)
JCudaTensor x1008;
JCudaTensor x1009, x1010;
x1009 = x1001;
x1010 = x1011;
x1008 = x1010.copy(64, x1009);

// val X395 = LogSoftmax()(X394)
JCudaTensor x1012;
JCudaTensor x1013;
x1013 = x1008;
x1012 = x1014.forward(x1013);

// Dealloc(X394)
JCudaTensor x1015;
x1015 = x1008;
x1015.free();

// val X1141 = (X1140 / |64|)
JCudaTensor x1016;
JCudaTensor x1017;
float x1018;
x1017 = x11;
float x1019;
x1019 = 64;
x1018 = 1 / x1019;
x1016 = x1017.times_i(x1018);

// Print(((0 - (X396 . X395)) / |64|))
float x1020;
float x1021;
float x1022;
float x1023;
JCudaTensor x1024, x1025;
x1024 = x7;
x1025 = x1012;
x1023 = x1024.dot(x1025);
x1021 = - x1023;
x1022 = 64;
x1020 = x1021 / x1022;
System.out.println(x5 + " " + x1020);
if (Float.isNaN(x1020)) { System.exit(-1); }

// Dealloc(X396)
JCudaTensor x1026;
x1026 = x7;
x1026.free();

// val X1340 = X1141 * d_LogSoftmax()(X395)/d_X394
JCudaTensor x1027;
JCudaTensor x1028, x1029;
x1028 = x1016;
x1029 = x1012;
x1027 = x1014.backward(x1028,x1029);

// Dealloc(X1141)
JCudaTensor x1030;
x1030 = x1016;
x1030.free();

// Dealloc(X395)
JCudaTensor x1031;
x1031 = x1012;
x1031.free();

// val m1 = (i3689) => fc_W[@, i3689]
JCudaMatrix x1032;
JCudaTensor x1033;
x1033 = x1007;
x1032 = x1033.asMatrix(1, false);

// val X1534 = (X1340)(i3688 | @) * m1
JCudaTensor x1034;
JCudaMatrix x1035;
JCudaMatrix x1036;
JCudaTensor x1037;
x1037 = x1027;
x1035 = x1037.asMatrix(1, true);
x1036 = x1032;
x1034 = x1035.times(x1036);

// val m160 = (i2202233) => X1340[@, i2202233]
JCudaMatrix x1038;
JCudaTensor x1039;
x1039 = x1027;
x1038 = x1039.asMatrix(1, false);

// val m162 = (i2208600) => X391[1><3][@, i2208600]
JCudaMatrix x1040;
JCudaTensor x1041;
JCudaTensor x1042;
x1042 = x998;
x1041 = x1042.flatten(1, new int[]{2048, 1, 1});
x1040 = x1041.asMatrix(1, false);

// V_fc_W <~~ m160 * m162
float x1044, x1045;
x1044 = lrn_rate_1;
x1045 = momentum;
JCudaMatrix x1046;
JCudaMatrix x1047;
x1046 = x1038;
x1047 = x1040;
x1046.times(x1047, x1043, x1044, x1045);

// val X1536 = X1534[1<>3] * d_Pooling(7,1,0,false)(X391,X390)/d_X390
JCudaTensor x1048;
JCudaTensor x1049, x1050, x1051;
JCudaTensor x1052;
x1052 = x1034;
x1049 = x1052.unflatten(1, new int[]{2048, 1, 1});
x1050 = x998;
x1051 = x996;
x1048 = x1000.backward(x1049,x1050,x1051);

// Dealloc(X1534)
JCudaTensor x1053;
x1053 = x1034;
x1053.free();

// Dealloc(X391)
JCudaTensor x1054;
x1054 = x998;
x1054.free();

// V_fc_B <~~ Sum(m160)
float x1056, x1057;
x1056 = lrn_rate_1;
x1057 = momentum;
JCudaMatrix x1058;
x1058 = x1038;
x1058.sum(x1055, x1056, x1057);

// Dealloc(X1340)
JCudaTensor x1059;
x1059 = x1027;
x1059.free();

// fc_W <~~ V_fc_W
float x1060, x1061;
x1060 = 1;
x1061 = decay_1;
JCudaTensor x1062;
x1062 = x1043;
x1007.update(x1062, x1060, x1061);


// fc_B <~~ V_fc_B
float x1063, x1064;
x1063 = 1;
x1064 = decay_1;
JCudaTensor x1065;
x1065 = x1055;
x1011.update(x1065, x1063, x1064);


// val X1573 = X1536 * d_ReLU()(X390)/d_X389
JCudaTensor x1066;
JCudaTensor x1067, x1068;
x1067 = x1048;
x1068 = x996;
x1066 = x992.backward(x1067,x1068);

// Dealloc(X390)
JCudaTensor x1069;
x1069 = x996;
x1069.free();

// val X1583 = X1573.copy * d_ReLU()(X388)/d_X387
JCudaTensor x1070;
JCudaTensor x1071, x1072;
x1071 = x1066;
x1071 = x1071.clone();
x1072 = x990;
x1070 = x992.backward(x1071,x1072);

// Dealloc(X388)
JCudaTensor x1073;
x1073 = x990;
x1073.free();

// val X206560 = X1583 * d_BatchNorm()(X386,5c_c_bn_scale)/d_5c_c_bn_bias
JCudaTensor x1074;
JCudaTensor x1075, x1076, x1077;
x1075 = x1070;
x1076 = x976;
x1077 = x987;
JCudaTensor[] x1078 = x989.backward(x1075,x1076,x1077);
x1074 = x1078[2];

// val X1584 = X1583 * d_BatchNorm()(X386,5c_c_bn_scale)/d_X386
JCudaTensor x1079;
x1079 = x1078[0];

// val X207231 = X1583 * d_BatchNorm()(X386,5c_c_bn_scale)/d_5c_c_bn_scale
JCudaTensor x1083;
x1083 = x1078[1];

// Dealloc(X386)
JCudaTensor x1087;
x1087 = x976;
x1087.free();

// val X1585 = X1584 * d_Convolv(1,0)(5c_c_cv_W)/d_X385
JCudaTensor x1088;
JCudaTensor x1089, x1090;
x1089 = x1079;
x1090 = x980;
x1088 = x982.backward_data(x1089,x1090);

// V_5c_c_cv_W <~~ X1584 * d_Convolv(1,0)(X385)/d_5c_c_cv_W
float x1092, x1093;
x1092 = lrn_rate_1;
x1093 = momentum;
JCudaTensor x1094, x1095;
x1094 = x1079;
x1095 = x973;
x982.backward_filter(x1094,x1095, x1091, x1092, x1093);

// Dealloc(X1584)
JCudaTensor x1096;
x1096 = x1079;
x1096.free();

// V_5c_c_bn_scale <~~ X207231
float x1098, x1099;
x1098 = lrn_rate_1;
x1099 = momentum;
JCudaTensor x1100;
x1100 = x1083;
x1097.update(x1100, x1098, x1099);


// Dealloc(X207231)
JCudaTensor x1101;
x1101 = x1083;
x1101.free();

// V_5c_c_bn_bias <~~ X206560
float x1103, x1104;
x1103 = lrn_rate_1;
x1104 = momentum;
JCudaTensor x1105;
x1105 = x1074;
x1102.update(x1105, x1103, x1104);


// Dealloc(X206560)
JCudaTensor x1106;
x1106 = x1074;
x1106.free();

// 5c_c_cv_W <~~ V_5c_c_cv_W
float x1107, x1108;
x1107 = 1;
x1108 = decay_1;
JCudaTensor x1109;
x1109 = x1091;
x980.update(x1109, x1107, x1108);


// 5c_c_bn_scale <~~ V_5c_c_bn_scale
float x1110, x1111;
x1110 = 1;
x1111 = decay_1;
JCudaTensor x1112;
x1112 = x1097;
x987.update(x1112, x1110, x1111);


// 5c_c_bn_bias <~~ V_5c_c_bn_bias
float x1113, x1114;
x1113 = 1;
x1114 = decay_1;
JCudaTensor x1115;
x1115 = x1102;
x988.update(x1115, x1113, x1114);


// val X1589 = X1585 * d_ReLU()(X385)/d_X384
JCudaTensor x1116;
JCudaTensor x1117, x1118;
x1117 = x1088;
x1118 = x973;
x1116 = x975.backward(x1117,x1118);

// Dealloc(X385)
JCudaTensor x1119;
x1119 = x973;
x1119.free();

// val X1590 = X1589 * d_BatchNorm()(X383,5c_b_bn_scale)/d_X383
JCudaTensor x1120;
JCudaTensor x1121, x1122, x1123;
x1121 = x1116;
x1122 = x959;
x1123 = x970;
JCudaTensor[] x1124 = x972.backward(x1121,x1122,x1123);
x1120 = x1124[0];

// val X205203 = X1589 * d_BatchNorm()(X383,5c_b_bn_scale)/d_5c_b_bn_scale
JCudaTensor x1125;
x1125 = x1124[1];

// val X204519 = X1589 * d_BatchNorm()(X383,5c_b_bn_scale)/d_5c_b_bn_bias
JCudaTensor x1129;
x1129 = x1124[2];

// Dealloc(X383)
JCudaTensor x1133;
x1133 = x959;
x1133.free();

// V_5c_b_bn_bias <~~ X204519
float x1135, x1136;
x1135 = lrn_rate_1;
x1136 = momentum;
JCudaTensor x1137;
x1137 = x1129;
x1134.update(x1137, x1135, x1136);


// Dealloc(X204519)
JCudaTensor x1138;
x1138 = x1129;
x1138.free();

// val X1591 = X1590 * d_Convolv(1,1)(5c_b_cv_W)/d_X382
JCudaTensor x1139;
JCudaTensor x1140, x1141;
x1140 = x1120;
x1141 = x963;
x1139 = x965.backward_data(x1140,x1141);

// V_5c_b_cv_W <~~ X1590 * d_Convolv(1,1)(X382)/d_5c_b_cv_W
float x1143, x1144;
x1143 = lrn_rate_1;
x1144 = momentum;
JCudaTensor x1145, x1146;
x1145 = x1120;
x1146 = x956;
x965.backward_filter(x1145,x1146, x1142, x1143, x1144);

// Dealloc(X1590)
JCudaTensor x1147;
x1147 = x1120;
x1147.free();

// V_5c_b_bn_scale <~~ X205203
float x1149, x1150;
x1149 = lrn_rate_1;
x1150 = momentum;
JCudaTensor x1151;
x1151 = x1125;
x1148.update(x1151, x1149, x1150);


// Dealloc(X205203)
JCudaTensor x1152;
x1152 = x1125;
x1152.free();

// 5c_b_bn_bias <~~ V_5c_b_bn_bias
float x1153, x1154;
x1153 = 1;
x1154 = decay_1;
JCudaTensor x1155;
x1155 = x1134;
x971.update(x1155, x1153, x1154);


// 5c_b_cv_W <~~ V_5c_b_cv_W
float x1156, x1157;
x1156 = 1;
x1157 = decay_1;
JCudaTensor x1158;
x1158 = x1142;
x963.update(x1158, x1156, x1157);


// 5c_b_bn_scale <~~ V_5c_b_bn_scale
float x1159, x1160;
x1159 = 1;
x1160 = decay_1;
JCudaTensor x1161;
x1161 = x1148;
x970.update(x1161, x1159, x1160);


// val X1595 = X1591 * d_ReLU()(X382)/d_X381
JCudaTensor x1162;
JCudaTensor x1163, x1164;
x1163 = x1139;
x1164 = x956;
x1162 = x958.backward(x1163,x1164);

// Dealloc(X382)
JCudaTensor x1165;
x1165 = x956;
x1165.free();

// val X1596 = X1595 * d_BatchNorm()(X380,5c_a_bn_scale)/d_X380
JCudaTensor x1166;
JCudaTensor x1167, x1168, x1169;
x1167 = x1162;
x1168 = x942;
x1169 = x953;
JCudaTensor[] x1170 = x955.backward(x1167,x1168,x1169);
x1166 = x1170[0];

// val X202439 = X1595 * d_BatchNorm()(X380,5c_a_bn_scale)/d_5c_a_bn_bias
JCudaTensor x1171;
x1171 = x1170[2];

// val X203136 = X1595 * d_BatchNorm()(X380,5c_a_bn_scale)/d_5c_a_bn_scale
JCudaTensor x1175;
x1175 = x1170[1];

// Dealloc(X380)
JCudaTensor x1179;
x1179 = x942;
x1179.free();

// V_5c_a_bn_bias <~~ X202439
float x1181, x1182;
x1181 = lrn_rate_1;
x1182 = momentum;
JCudaTensor x1183;
x1183 = x1171;
x1180.update(x1183, x1181, x1182);


// Dealloc(X202439)
JCudaTensor x1184;
x1184 = x1171;
x1184.free();

// V_5c_a_bn_scale <~~ X203136
float x1186, x1187;
x1186 = lrn_rate_1;
x1187 = momentum;
JCudaTensor x1188;
x1188 = x1175;
x1185.update(x1188, x1186, x1187);


// Dealloc(X203136)
JCudaTensor x1189;
x1189 = x1175;
x1189.free();

// val X1597 = X1596 * d_Convolv(1,0)(5c_a_cv_W)/d_X379
JCudaTensor x1190;
JCudaTensor x1191, x1192;
x1191 = x1166;
x1192 = x946;
x1190 = x948.backward_data(x1191,x1192);

// V_5c_a_cv_W <~~ X1596 * d_Convolv(1,0)(X379)/d_5c_a_cv_W
float x1194, x1195;
x1194 = lrn_rate_1;
x1195 = momentum;
JCudaTensor x1196, x1197;
x1196 = x1166;
x1197 = x940;
x948.backward_filter(x1196,x1197, x1193, x1194, x1195);

// Dealloc(X1596)
JCudaTensor x1198;
x1198 = x1166;
x1198.free();

// 5c_a_bn_bias <~~ V_5c_a_bn_bias
float x1199, x1200;
x1199 = 1;
x1200 = decay_1;
JCudaTensor x1201;
x1201 = x1180;
x954.update(x1201, x1199, x1200);


// 5c_a_bn_scale <~~ V_5c_a_bn_scale
float x1202, x1203;
x1202 = 1;
x1203 = decay_1;
JCudaTensor x1204;
x1204 = x1185;
x953.update(x1204, x1202, x1203);


// 5c_a_cv_W <~~ V_5c_a_cv_W
float x1205, x1206;
x1205 = 1;
x1206 = decay_1;
JCudaTensor x1207;
x1207 = x1193;
x946.update(x1207, x1205, x1206);


// val X1598 = (X1597 + X1573)
JCudaTensor x1208;
JCudaTensor x1209, x1210;
x1209 = x1190;
x1210 = x1066;
x1208 = x1209.plus_i(x1210);

// Dealloc(X1573)
JCudaTensor x1211;
x1211 = x1066;
x1211.free();

// val X1610 = X1598 * d_ReLU()(X379)/d_X378
JCudaTensor x1212;
JCudaTensor x1213, x1214;
x1213 = x1208;
x1214 = x940;
x1212 = x936.backward(x1213,x1214);

// Dealloc(X379)
JCudaTensor x1215;
x1215 = x940;
x1215.free();

// val X1620 = X1610.copy * d_ReLU()(X377)/d_X376
JCudaTensor x1216;
JCudaTensor x1217, x1218;
x1217 = x1212;
x1217 = x1217.clone();
x1218 = x934;
x1216 = x936.backward(x1217,x1218);

// Dealloc(X377)
JCudaTensor x1219;
x1219 = x934;
x1219.free();

// val X200264 = X1620 * d_BatchNorm()(X375,5b_c_bn_scale)/d_5b_c_bn_bias
JCudaTensor x1220;
JCudaTensor x1221, x1222, x1223;
x1221 = x1216;
x1222 = x920;
x1223 = x931;
JCudaTensor[] x1224 = x933.backward(x1221,x1222,x1223);
x1220 = x1224[2];

// val X1621 = X1620 * d_BatchNorm()(X375,5b_c_bn_scale)/d_X375
JCudaTensor x1225;
x1225 = x1224[0];

// val X201002 = X1620 * d_BatchNorm()(X375,5b_c_bn_scale)/d_5b_c_bn_scale
JCudaTensor x1229;
x1229 = x1224[1];

// Dealloc(X375)
JCudaTensor x1233;
x1233 = x920;
x1233.free();

// val X1622 = X1621 * d_Convolv(1,0)(5b_c_cv_W)/d_X374
JCudaTensor x1234;
JCudaTensor x1235, x1236;
x1235 = x1225;
x1236 = x924;
x1234 = x926.backward_data(x1235,x1236);

// V_5b_c_cv_W <~~ X1621 * d_Convolv(1,0)(X374)/d_5b_c_cv_W
float x1238, x1239;
x1238 = lrn_rate_1;
x1239 = momentum;
JCudaTensor x1240, x1241;
x1240 = x1225;
x1241 = x917;
x926.backward_filter(x1240,x1241, x1237, x1238, x1239);

// Dealloc(X1621)
JCudaTensor x1242;
x1242 = x1225;
x1242.free();

// V_5b_c_bn_scale <~~ X201002
float x1244, x1245;
x1244 = lrn_rate_1;
x1245 = momentum;
JCudaTensor x1246;
x1246 = x1229;
x1243.update(x1246, x1244, x1245);


// Dealloc(X201002)
JCudaTensor x1247;
x1247 = x1229;
x1247.free();

// V_5b_c_bn_bias <~~ X200264
float x1249, x1250;
x1249 = lrn_rate_1;
x1250 = momentum;
JCudaTensor x1251;
x1251 = x1220;
x1248.update(x1251, x1249, x1250);


// Dealloc(X200264)
JCudaTensor x1252;
x1252 = x1220;
x1252.free();

// 5b_c_cv_W <~~ V_5b_c_cv_W
float x1253, x1254;
x1253 = 1;
x1254 = decay_1;
JCudaTensor x1255;
x1255 = x1237;
x924.update(x1255, x1253, x1254);


// 5b_c_bn_scale <~~ V_5b_c_bn_scale
float x1256, x1257;
x1256 = 1;
x1257 = decay_1;
JCudaTensor x1258;
x1258 = x1243;
x931.update(x1258, x1256, x1257);


// 5b_c_bn_bias <~~ V_5b_c_bn_bias
float x1259, x1260;
x1259 = 1;
x1260 = decay_1;
JCudaTensor x1261;
x1261 = x1248;
x932.update(x1261, x1259, x1260);


// val X1626 = X1622 * d_ReLU()(X374)/d_X373
JCudaTensor x1262;
JCudaTensor x1263, x1264;
x1263 = x1234;
x1264 = x917;
x1262 = x919.backward(x1263,x1264);

// Dealloc(X374)
JCudaTensor x1265;
x1265 = x917;
x1265.free();

// val X198022 = X1626 * d_BatchNorm()(X372,5b_b_bn_scale)/d_5b_b_bn_bias
JCudaTensor x1266;
JCudaTensor x1267, x1268, x1269;
x1267 = x1262;
x1268 = x903;
x1269 = x914;
JCudaTensor[] x1270 = x916.backward(x1267,x1268,x1269);
x1266 = x1270[2];

// val X1627 = X1626 * d_BatchNorm()(X372,5b_b_bn_scale)/d_X372
JCudaTensor x1271;
x1271 = x1270[0];

// val X198773 = X1626 * d_BatchNorm()(X372,5b_b_bn_scale)/d_5b_b_bn_scale
JCudaTensor x1275;
x1275 = x1270[1];

// Dealloc(X372)
JCudaTensor x1279;
x1279 = x903;
x1279.free();

// V_5b_b_bn_bias <~~ X198022
float x1281, x1282;
x1281 = lrn_rate_1;
x1282 = momentum;
JCudaTensor x1283;
x1283 = x1266;
x1280.update(x1283, x1281, x1282);


// Dealloc(X198022)
JCudaTensor x1284;
x1284 = x1266;
x1284.free();

// V_5b_b_bn_scale <~~ X198773
float x1286, x1287;
x1286 = lrn_rate_1;
x1287 = momentum;
JCudaTensor x1288;
x1288 = x1275;
x1285.update(x1288, x1286, x1287);


// Dealloc(X198773)
JCudaTensor x1289;
x1289 = x1275;
x1289.free();

// val X1628 = X1627 * d_Convolv(1,1)(5b_b_cv_W)/d_X371
JCudaTensor x1290;
JCudaTensor x1291, x1292;
x1291 = x1271;
x1292 = x907;
x1290 = x909.backward_data(x1291,x1292);

// V_5b_b_cv_W <~~ X1627 * d_Convolv(1,1)(X371)/d_5b_b_cv_W
float x1294, x1295;
x1294 = lrn_rate_1;
x1295 = momentum;
JCudaTensor x1296, x1297;
x1296 = x1271;
x1297 = x900;
x909.backward_filter(x1296,x1297, x1293, x1294, x1295);

// Dealloc(X1627)
JCudaTensor x1298;
x1298 = x1271;
x1298.free();

// 5b_b_bn_bias <~~ V_5b_b_bn_bias
float x1299, x1300;
x1299 = 1;
x1300 = decay_1;
JCudaTensor x1301;
x1301 = x1280;
x915.update(x1301, x1299, x1300);


// 5b_b_bn_scale <~~ V_5b_b_bn_scale
float x1302, x1303;
x1302 = 1;
x1303 = decay_1;
JCudaTensor x1304;
x1304 = x1285;
x914.update(x1304, x1302, x1303);


// 5b_b_cv_W <~~ V_5b_b_cv_W
float x1305, x1306;
x1305 = 1;
x1306 = decay_1;
JCudaTensor x1307;
x1307 = x1293;
x907.update(x1307, x1305, x1306);


// val X1632 = X1628 * d_ReLU()(X371)/d_X370
JCudaTensor x1308;
JCudaTensor x1309, x1310;
x1309 = x1290;
x1310 = x900;
x1308 = x902.backward(x1309,x1310);

// Dealloc(X371)
JCudaTensor x1311;
x1311 = x900;
x1311.free();

// val X1633 = X1632 * d_BatchNorm()(X369,5b_a_bn_scale)/d_X369
JCudaTensor x1312;
JCudaTensor x1313, x1314, x1315;
x1313 = x1308;
x1314 = x886;
x1315 = x897;
JCudaTensor[] x1316 = x899.backward(x1313,x1314,x1315);
x1312 = x1316[0];

// val X195741 = X1632 * d_BatchNorm()(X369,5b_a_bn_scale)/d_5b_a_bn_bias
JCudaTensor x1317;
x1317 = x1316[2];

// val X196505 = X1632 * d_BatchNorm()(X369,5b_a_bn_scale)/d_5b_a_bn_scale
JCudaTensor x1321;
x1321 = x1316[1];

// Dealloc(X369)
JCudaTensor x1325;
x1325 = x886;
x1325.free();

// V_5b_a_bn_scale <~~ X196505
float x1327, x1328;
x1327 = lrn_rate_1;
x1328 = momentum;
JCudaTensor x1329;
x1329 = x1321;
x1326.update(x1329, x1327, x1328);


// Dealloc(X196505)
JCudaTensor x1330;
x1330 = x1321;
x1330.free();

// val X1634 = X1633 * d_Convolv(1,0)(5b_a_cv_W)/d_X368
JCudaTensor x1331;
JCudaTensor x1332, x1333;
x1332 = x1312;
x1333 = x890;
x1331 = x892.backward_data(x1332,x1333);

// V_5b_a_cv_W <~~ X1633 * d_Convolv(1,0)(X368)/d_5b_a_cv_W
float x1335, x1336;
x1335 = lrn_rate_1;
x1336 = momentum;
JCudaTensor x1337, x1338;
x1337 = x1312;
x1338 = x884;
x892.backward_filter(x1337,x1338, x1334, x1335, x1336);

// Dealloc(X1633)
JCudaTensor x1339;
x1339 = x1312;
x1339.free();

// V_5b_a_bn_bias <~~ X195741
float x1341, x1342;
x1341 = lrn_rate_1;
x1342 = momentum;
JCudaTensor x1343;
x1343 = x1317;
x1340.update(x1343, x1341, x1342);


// Dealloc(X195741)
JCudaTensor x1344;
x1344 = x1317;
x1344.free();

// 5b_a_bn_scale <~~ V_5b_a_bn_scale
float x1345, x1346;
x1345 = 1;
x1346 = decay_1;
JCudaTensor x1347;
x1347 = x1326;
x897.update(x1347, x1345, x1346);


// 5b_a_cv_W <~~ V_5b_a_cv_W
float x1348, x1349;
x1348 = 1;
x1349 = decay_1;
JCudaTensor x1350;
x1350 = x1334;
x890.update(x1350, x1348, x1349);


// 5b_a_bn_bias <~~ V_5b_a_bn_bias
float x1351, x1352;
x1351 = 1;
x1352 = decay_1;
JCudaTensor x1353;
x1353 = x1340;
x898.update(x1353, x1351, x1352);


// val X1635 = (X1634 + X1610)
JCudaTensor x1354;
JCudaTensor x1355, x1356;
x1355 = x1331;
x1356 = x1212;
x1354 = x1355.plus_i(x1356);

// Dealloc(X1610)
JCudaTensor x1357;
x1357 = x1212;
x1357.free();

// val X1650 = X1635 * d_ReLU()(X368)/d_X367
JCudaTensor x1358;
JCudaTensor x1359, x1360;
x1359 = x1354;
x1360 = x884;
x1358 = x877.backward(x1359,x1360);

// Dealloc(X368)
JCudaTensor x1361;
x1361 = x884;
x1361.free();

// val X1666 = X1650.copy * d_ReLU()(X366)/d_X365
JCudaTensor x1362;
JCudaTensor x1363, x1364;
x1363 = x1358;
x1363 = x1363.clone();
x1364 = x878;
x1362 = x880.backward(x1363,x1364);

// Dealloc(X366)
JCudaTensor x1365;
x1365 = x878;
x1365.free();

// val X1654 = X1650.copy * d_ReLU()(X357)/d_X356
JCudaTensor x1366;
JCudaTensor x1367, x1368;
x1367 = x1358;
x1367 = x1367.clone();
x1368 = x875;
x1366 = x877.backward(x1367,x1368);

// Dealloc(X1650)
JCudaTensor x1369;
x1369 = x1358;
x1369.free();

// Dealloc(X357)
JCudaTensor x1370;
x1370 = x875;
x1370.free();

// val X1667 = X1666 * d_BatchNorm()(X364,5a2_c_bn_scale)/d_X364
JCudaTensor x1371;
JCudaTensor x1372, x1373, x1374;
x1372 = x1362;
x1373 = x861;
x1374 = x872;
JCudaTensor[] x1375 = x874.backward(x1372,x1373,x1374);
x1371 = x1375[0];

// val X1655 = X1654 * d_BatchNorm()(X355,5a1_bn_scale)/d_X355
JCudaTensor x1376;
JCudaTensor x1377, x1378, x1379;
x1377 = x1366;
x1378 = x813;
x1379 = x838;
JCudaTensor[] x1380 = x840.backward(x1377,x1378,x1379);
x1376 = x1380[0];

// val X186781 = X1654 * d_BatchNorm()(X355,5a1_bn_scale)/d_5a1_bn_scale
JCudaTensor x1381;
x1381 = x1380[1];

// val X193359 = X1666 * d_BatchNorm()(X364,5a2_c_bn_scale)/d_5a2_c_bn_bias
JCudaTensor x1385;
x1385 = x1375[2];

// val X185982 = X1654 * d_BatchNorm()(X355,5a1_bn_scale)/d_5a1_bn_bias
JCudaTensor x1389;
x1389 = x1380[2];

// Dealloc(X355)
JCudaTensor x1393;
x1393 = x813;
x1393.free();

// val X194167 = X1666 * d_BatchNorm()(X364,5a2_c_bn_scale)/d_5a2_c_bn_scale
JCudaTensor x1394;
x1394 = x1375[1];

// Dealloc(X364)
JCudaTensor x1398;
x1398 = x861;
x1398.free();

// V_5a2_c_bn_bias <~~ X193359
float x1400, x1401;
x1400 = lrn_rate_1;
x1401 = momentum;
JCudaTensor x1402;
x1402 = x1385;
x1399.update(x1402, x1400, x1401);


// Dealloc(X193359)
JCudaTensor x1403;
x1403 = x1385;
x1403.free();

// V_5a1_bn_bias <~~ X185982
float x1405, x1406;
x1405 = lrn_rate_1;
x1406 = momentum;
JCudaTensor x1407;
x1407 = x1389;
x1404.update(x1407, x1405, x1406);


// Dealloc(X185982)
JCudaTensor x1408;
x1408 = x1389;
x1408.free();

// V_5a1_cv_W <~~ X1655 * d_Convolv(2,0)(X354)/d_5a1_cv_W
float x1410, x1411;
x1410 = lrn_rate_1;
x1411 = momentum;
JCudaTensor x1412, x1413;
x1412 = x1376;
x1413 = x811;
x819.backward_filter(x1412,x1413, x1409, x1410, x1411);

// V_5a2_c_cv_W <~~ X1667 * d_Convolv(1,0)(X363)/d_5a2_c_cv_W
float x1415, x1416;
x1415 = lrn_rate_1;
x1416 = momentum;
JCudaTensor x1417, x1418;
x1417 = x1371;
x1418 = x858;
x867.backward_filter(x1417,x1418, x1414, x1415, x1416);

// val X1668 = X1667 * d_Convolv(1,0)(5a2_c_cv_W)/d_X363
JCudaTensor x1419;
JCudaTensor x1420, x1421;
x1420 = x1371;
x1421 = x865;
x1419 = x867.backward_data(x1420,x1421);

// Dealloc(X1667)
JCudaTensor x1422;
x1422 = x1371;
x1422.free();

// V_5a2_c_bn_scale <~~ X194167
float x1424, x1425;
x1424 = lrn_rate_1;
x1425 = momentum;
JCudaTensor x1426;
x1426 = x1394;
x1423.update(x1426, x1424, x1425);


// Dealloc(X194167)
JCudaTensor x1427;
x1427 = x1394;
x1427.free();

// V_5a1_bn_scale <~~ X186781
float x1429, x1430;
x1429 = lrn_rate_1;
x1430 = momentum;
JCudaTensor x1431;
x1431 = x1381;
x1428.update(x1431, x1429, x1430);


// Dealloc(X186781)
JCudaTensor x1432;
x1432 = x1381;
x1432.free();

// val X1656 = X1655 * d_Convolv(2,0)(5a1_cv_W)/d_X354
JCudaTensor x1433;
JCudaTensor x1434, x1435;
x1434 = x1376;
x1435 = x817;
x1433 = x819.backward_data(x1434,x1435);

// Dealloc(X1655)
JCudaTensor x1436;
x1436 = x1376;
x1436.free();

// 5a1_bn_scale <~~ V_5a1_bn_scale
float x1437, x1438;
x1437 = 1;
x1438 = decay_1;
JCudaTensor x1439;
x1439 = x1428;
x838.update(x1439, x1437, x1438);


// 5a2_c_bn_scale <~~ V_5a2_c_bn_scale
float x1440, x1441;
x1440 = 1;
x1441 = decay_1;
JCudaTensor x1442;
x1442 = x1423;
x872.update(x1442, x1440, x1441);


// 5a2_c_bn_bias <~~ V_5a2_c_bn_bias
float x1443, x1444;
x1443 = 1;
x1444 = decay_1;
JCudaTensor x1445;
x1445 = x1399;
x873.update(x1445, x1443, x1444);


// 5a2_c_cv_W <~~ V_5a2_c_cv_W
float x1446, x1447;
x1446 = 1;
x1447 = decay_1;
JCudaTensor x1448;
x1448 = x1414;
x865.update(x1448, x1446, x1447);


// 5a1_bn_bias <~~ V_5a1_bn_bias
float x1449, x1450;
x1449 = 1;
x1450 = decay_1;
JCudaTensor x1451;
x1451 = x1404;
x839.update(x1451, x1449, x1450);


// 5a1_cv_W <~~ V_5a1_cv_W
float x1452, x1453;
x1452 = 1;
x1453 = decay_1;
JCudaTensor x1454;
x1454 = x1409;
x817.update(x1454, x1452, x1453);


// val X1672 = X1668 * d_ReLU()(X363)/d_X362
JCudaTensor x1455;
JCudaTensor x1456, x1457;
x1456 = x1419;
x1457 = x858;
x1455 = x860.backward(x1456,x1457);

// Dealloc(X363)
JCudaTensor x1458;
x1458 = x858;
x1458.free();

// val X1673 = X1672 * d_BatchNorm()(X361,5a2_b_bn_scale)/d_X361
JCudaTensor x1459;
JCudaTensor x1460, x1461, x1462;
x1460 = x1455;
x1461 = x844;
x1462 = x855;
JCudaTensor[] x1463 = x857.backward(x1460,x1461,x1462);
x1459 = x1463[0];

// val X190907 = X1672 * d_BatchNorm()(X361,5a2_b_bn_scale)/d_5a2_b_bn_bias
JCudaTensor x1464;
x1464 = x1463[2];

// val X191728 = X1672 * d_BatchNorm()(X361,5a2_b_bn_scale)/d_5a2_b_bn_scale
JCudaTensor x1468;
x1468 = x1463[1];

// Dealloc(X361)
JCudaTensor x1472;
x1472 = x844;
x1472.free();

// val X1674 = X1673 * d_Convolv(1,1)(5a2_b_cv_W)/d_X360
JCudaTensor x1473;
JCudaTensor x1474, x1475;
x1474 = x1459;
x1475 = x848;
x1473 = x850.backward_data(x1474,x1475);

// V_5a2_b_cv_W <~~ X1673 * d_Convolv(1,1)(X360)/d_5a2_b_cv_W
float x1477, x1478;
x1477 = lrn_rate_1;
x1478 = momentum;
JCudaTensor x1479, x1480;
x1479 = x1459;
x1480 = x841;
x850.backward_filter(x1479,x1480, x1476, x1477, x1478);

// Dealloc(X1673)
JCudaTensor x1481;
x1481 = x1459;
x1481.free();

// V_5a2_b_bn_scale <~~ X191728
float x1483, x1484;
x1483 = lrn_rate_1;
x1484 = momentum;
JCudaTensor x1485;
x1485 = x1468;
x1482.update(x1485, x1483, x1484);


// Dealloc(X191728)
JCudaTensor x1486;
x1486 = x1468;
x1486.free();

// V_5a2_b_bn_bias <~~ X190907
float x1488, x1489;
x1488 = lrn_rate_1;
x1489 = momentum;
JCudaTensor x1490;
x1490 = x1464;
x1487.update(x1490, x1488, x1489);


// Dealloc(X190907)
JCudaTensor x1491;
x1491 = x1464;
x1491.free();

// 5a2_b_cv_W <~~ V_5a2_b_cv_W
float x1492, x1493;
x1492 = 1;
x1493 = decay_1;
JCudaTensor x1494;
x1494 = x1476;
x848.update(x1494, x1492, x1493);


// 5a2_b_bn_scale <~~ V_5a2_b_bn_scale
float x1495, x1496;
x1495 = 1;
x1496 = decay_1;
JCudaTensor x1497;
x1497 = x1482;
x855.update(x1497, x1495, x1496);


// 5a2_b_bn_bias <~~ V_5a2_b_bn_bias
float x1498, x1499;
x1498 = 1;
x1499 = decay_1;
JCudaTensor x1500;
x1500 = x1487;
x856.update(x1500, x1498, x1499);


// val X1678 = X1674 * d_ReLU()(X360)/d_X359
JCudaTensor x1501;
JCudaTensor x1502, x1503;
x1502 = x1473;
x1503 = x841;
x1501 = x843.backward(x1502,x1503);

// Dealloc(X360)
JCudaTensor x1504;
x1504 = x841;
x1504.free();

// val X1679 = X1678 * d_BatchNorm()(X358,5a2_a_bn_scale)/d_X358
JCudaTensor x1505;
JCudaTensor x1506, x1507, x1508;
x1506 = x1501;
x1507 = x820;
x1508 = x831;
JCudaTensor[] x1509 = x833.backward(x1506,x1507,x1508);
x1505 = x1509[0];

// val X188416 = X1678 * d_BatchNorm()(X358,5a2_a_bn_scale)/d_5a2_a_bn_bias
JCudaTensor x1510;
x1510 = x1509[2];

// val X189250 = X1678 * d_BatchNorm()(X358,5a2_a_bn_scale)/d_5a2_a_bn_scale
JCudaTensor x1514;
x1514 = x1509[1];

// Dealloc(X358)
JCudaTensor x1518;
x1518 = x820;
x1518.free();

// V_5a2_a_bn_bias <~~ X188416
float x1520, x1521;
x1520 = lrn_rate_1;
x1521 = momentum;
JCudaTensor x1522;
x1522 = x1510;
x1519.update(x1522, x1520, x1521);


// Dealloc(X188416)
JCudaTensor x1523;
x1523 = x1510;
x1523.free();

// V_5a2_a_bn_scale <~~ X189250
float x1525, x1526;
x1525 = lrn_rate_1;
x1526 = momentum;
JCudaTensor x1527;
x1527 = x1514;
x1524.update(x1527, x1525, x1526);


// Dealloc(X189250)
JCudaTensor x1528;
x1528 = x1514;
x1528.free();

// val X1681 = (X1656 + X1679 * d_Convolv(2,0)(5a2_a_cv_W)/d_X354)
JCudaTensor x1529;
JCudaTensor x1530;
x1530 = x1433;
JCudaTensor x1531, x1532;
x1531 = x1505;
x1532 = x824;
x1529 = x826.backward_data(x1531,x1532, x1530);

// V_5a2_a_cv_W <~~ X1679 * d_Convolv(2,0)(X354)/d_5a2_a_cv_W
float x1534, x1535;
x1534 = lrn_rate_1;
x1535 = momentum;
JCudaTensor x1536, x1537;
x1536 = x1505;
x1537 = x811;
x826.backward_filter(x1536,x1537, x1533, x1534, x1535);

// Dealloc(X1679)
JCudaTensor x1538;
x1538 = x1505;
x1538.free();

// 5a2_a_bn_bias <~~ V_5a2_a_bn_bias
float x1539, x1540;
x1539 = 1;
x1540 = decay_1;
JCudaTensor x1541;
x1541 = x1519;
x832.update(x1541, x1539, x1540);


// 5a2_a_bn_scale <~~ V_5a2_a_bn_scale
float x1542, x1543;
x1542 = 1;
x1543 = decay_1;
JCudaTensor x1544;
x1544 = x1524;
x831.update(x1544, x1542, x1543);


// 5a2_a_cv_W <~~ V_5a2_a_cv_W
float x1545, x1546;
x1545 = 1;
x1546 = decay_1;
JCudaTensor x1547;
x1547 = x1533;
x824.update(x1547, x1545, x1546);


// val X1751 = X1681 * d_ReLU()(X354)/d_X353
JCudaTensor x1548;
JCudaTensor x1549, x1550;
x1549 = x1529;
x1550 = x811;
x1548 = x807.backward(x1549,x1550);

// Dealloc(X354)
JCudaTensor x1551;
x1551 = x811;
x1551.free();

// val X1761 = X1751.copy * d_ReLU()(X352)/d_X351
JCudaTensor x1552;
JCudaTensor x1553, x1554;
x1553 = x1548;
x1553 = x1553.clone();
x1554 = x805;
x1552 = x807.backward(x1553,x1554);

// Dealloc(X352)
JCudaTensor x1555;
x1555 = x805;
x1555.free();

// val X183277 = X1761 * d_BatchNorm()(X350,4f_c_bn_scale)/d_4f_c_bn_bias
JCudaTensor x1556;
JCudaTensor x1557, x1558, x1559;
x1557 = x1552;
x1558 = x791;
x1559 = x802;
JCudaTensor[] x1560 = x804.backward(x1557,x1558,x1559);
x1556 = x1560[2];

// val X1762 = X1761 * d_BatchNorm()(X350,4f_c_bn_scale)/d_X350
JCudaTensor x1561;
x1561 = x1560[0];

// val X184229 = X1761 * d_BatchNorm()(X350,4f_c_bn_scale)/d_4f_c_bn_scale
JCudaTensor x1565;
x1565 = x1560[1];

// Dealloc(X350)
JCudaTensor x1569;
x1569 = x791;
x1569.free();

// V_4f_c_bn_bias <~~ X183277
float x1571, x1572;
x1571 = lrn_rate_1;
x1572 = momentum;
JCudaTensor x1573;
x1573 = x1556;
x1570.update(x1573, x1571, x1572);


// Dealloc(X183277)
JCudaTensor x1574;
x1574 = x1556;
x1574.free();

// val X1763 = X1762 * d_Convolv(1,0)(4f_c_cv_W)/d_X349
JCudaTensor x1575;
JCudaTensor x1576, x1577;
x1576 = x1561;
x1577 = x795;
x1575 = x797.backward_data(x1576,x1577);

// V_4f_c_cv_W <~~ X1762 * d_Convolv(1,0)(X349)/d_4f_c_cv_W
float x1579, x1580;
x1579 = lrn_rate_1;
x1580 = momentum;
JCudaTensor x1581, x1582;
x1581 = x1561;
x1582 = x788;
x797.backward_filter(x1581,x1582, x1578, x1579, x1580);

// Dealloc(X1762)
JCudaTensor x1583;
x1583 = x1561;
x1583.free();

// V_4f_c_bn_scale <~~ X184229
float x1585, x1586;
x1585 = lrn_rate_1;
x1586 = momentum;
JCudaTensor x1587;
x1587 = x1565;
x1584.update(x1587, x1585, x1586);


// Dealloc(X184229)
JCudaTensor x1588;
x1588 = x1565;
x1588.free();

// 4f_c_bn_bias <~~ V_4f_c_bn_bias
float x1589, x1590;
x1589 = 1;
x1590 = decay_1;
JCudaTensor x1591;
x1591 = x1570;
x803.update(x1591, x1589, x1590);


// 4f_c_cv_W <~~ V_4f_c_cv_W
float x1592, x1593;
x1592 = 1;
x1593 = decay_1;
JCudaTensor x1594;
x1594 = x1578;
x795.update(x1594, x1592, x1593);


// 4f_c_bn_scale <~~ V_4f_c_bn_scale
float x1595, x1596;
x1595 = 1;
x1596 = decay_1;
JCudaTensor x1597;
x1597 = x1584;
x802.update(x1597, x1595, x1596);


// val X1767 = X1763 * d_ReLU()(X349)/d_X348
JCudaTensor x1598;
JCudaTensor x1599, x1600;
x1599 = x1575;
x1600 = x788;
x1598 = x790.backward(x1599,x1600);

// Dealloc(X349)
JCudaTensor x1601;
x1601 = x788;
x1601.free();

// val X181358 = X1767 * d_BatchNorm()(X347,4f_b_bn_scale)/d_4f_b_bn_scale
JCudaTensor x1602;
JCudaTensor x1603, x1604, x1605;
x1603 = x1598;
x1604 = x774;
x1605 = x785;
JCudaTensor[] x1606 = x787.backward(x1603,x1604,x1605);
x1602 = x1606[1];

// val X1768 = X1767 * d_BatchNorm()(X347,4f_b_bn_scale)/d_X347
JCudaTensor x1607;
x1607 = x1606[0];

// val X180393 = X1767 * d_BatchNorm()(X347,4f_b_bn_scale)/d_4f_b_bn_bias
JCudaTensor x1611;
x1611 = x1606[2];

// Dealloc(X347)
JCudaTensor x1615;
x1615 = x774;
x1615.free();

// V_4f_b_bn_bias <~~ X180393
float x1617, x1618;
x1617 = lrn_rate_1;
x1618 = momentum;
JCudaTensor x1619;
x1619 = x1611;
x1616.update(x1619, x1617, x1618);


// Dealloc(X180393)
JCudaTensor x1620;
x1620 = x1611;
x1620.free();

// val X1769 = X1768 * d_Convolv(1,1)(4f_b_cv_W)/d_X346
JCudaTensor x1621;
JCudaTensor x1622, x1623;
x1622 = x1607;
x1623 = x778;
x1621 = x780.backward_data(x1622,x1623);

// V_4f_b_bn_scale <~~ X181358
float x1625, x1626;
x1625 = lrn_rate_1;
x1626 = momentum;
JCudaTensor x1627;
x1627 = x1602;
x1624.update(x1627, x1625, x1626);


// Dealloc(X181358)
JCudaTensor x1628;
x1628 = x1602;
x1628.free();

// V_4f_b_cv_W <~~ X1768 * d_Convolv(1,1)(X346)/d_4f_b_cv_W
float x1630, x1631;
x1630 = lrn_rate_1;
x1631 = momentum;
JCudaTensor x1632, x1633;
x1632 = x1607;
x1633 = x771;
x780.backward_filter(x1632,x1633, x1629, x1630, x1631);

// Dealloc(X1768)
JCudaTensor x1634;
x1634 = x1607;
x1634.free();

// 4f_b_bn_bias <~~ V_4f_b_bn_bias
float x1635, x1636;
x1635 = 1;
x1636 = decay_1;
JCudaTensor x1637;
x1637 = x1616;
x786.update(x1637, x1635, x1636);


// 4f_b_bn_scale <~~ V_4f_b_bn_scale
float x1638, x1639;
x1638 = 1;
x1639 = decay_1;
JCudaTensor x1640;
x1640 = x1624;
x785.update(x1640, x1638, x1639);


// 4f_b_cv_W <~~ V_4f_b_cv_W
float x1641, x1642;
x1641 = 1;
x1642 = decay_1;
JCudaTensor x1643;
x1643 = x1629;
x778.update(x1643, x1641, x1642);


// val X1773 = X1769 * d_ReLU()(X346)/d_X345
JCudaTensor x1644;
JCudaTensor x1645, x1646;
x1645 = x1621;
x1646 = x771;
x1644 = x773.backward(x1645,x1646);

// Dealloc(X346)
JCudaTensor x1647;
x1647 = x771;
x1647.free();

// val X1774 = X1773 * d_BatchNorm()(X344,4f_a_bn_scale)/d_X344
JCudaTensor x1648;
JCudaTensor x1649, x1650, x1651;
x1649 = x1644;
x1650 = x757;
x1651 = x768;
JCudaTensor[] x1652 = x770.backward(x1649,x1650,x1651);
x1648 = x1652[0];

// val X178448 = X1773 * d_BatchNorm()(X344,4f_a_bn_scale)/d_4f_a_bn_scale
JCudaTensor x1653;
x1653 = x1652[1];

// val X177470 = X1773 * d_BatchNorm()(X344,4f_a_bn_scale)/d_4f_a_bn_bias
JCudaTensor x1657;
x1657 = x1652[2];

// Dealloc(X344)
JCudaTensor x1661;
x1661 = x757;
x1661.free();

// val X1775 = X1774 * d_Convolv(1,0)(4f_a_cv_W)/d_X343
JCudaTensor x1662;
JCudaTensor x1663, x1664;
x1663 = x1648;
x1664 = x761;
x1662 = x763.backward_data(x1663,x1664);

// V_4f_a_cv_W <~~ X1774 * d_Convolv(1,0)(X343)/d_4f_a_cv_W
float x1666, x1667;
x1666 = lrn_rate_1;
x1667 = momentum;
JCudaTensor x1668, x1669;
x1668 = x1648;
x1669 = x755;
x763.backward_filter(x1668,x1669, x1665, x1666, x1667);

// Dealloc(X1774)
JCudaTensor x1670;
x1670 = x1648;
x1670.free();

// V_4f_a_bn_bias <~~ X177470
float x1672, x1673;
x1672 = lrn_rate_1;
x1673 = momentum;
JCudaTensor x1674;
x1674 = x1657;
x1671.update(x1674, x1672, x1673);


// Dealloc(X177470)
JCudaTensor x1675;
x1675 = x1657;
x1675.free();

// V_4f_a_bn_scale <~~ X178448
float x1677, x1678;
x1677 = lrn_rate_1;
x1678 = momentum;
JCudaTensor x1679;
x1679 = x1653;
x1676.update(x1679, x1677, x1678);


// Dealloc(X178448)
JCudaTensor x1680;
x1680 = x1653;
x1680.free();

// 4f_a_cv_W <~~ V_4f_a_cv_W
float x1681, x1682;
x1681 = 1;
x1682 = decay_1;
JCudaTensor x1683;
x1683 = x1665;
x761.update(x1683, x1681, x1682);


// 4f_a_bn_bias <~~ V_4f_a_bn_bias
float x1684, x1685;
x1684 = 1;
x1685 = decay_1;
JCudaTensor x1686;
x1686 = x1671;
x769.update(x1686, x1684, x1685);


// 4f_a_bn_scale <~~ V_4f_a_bn_scale
float x1687, x1688;
x1687 = 1;
x1688 = decay_1;
JCudaTensor x1689;
x1689 = x1676;
x768.update(x1689, x1687, x1688);


// val X1776 = (X1775 + X1751)
JCudaTensor x1690;
JCudaTensor x1691, x1692;
x1691 = x1662;
x1692 = x1548;
x1690 = x1691.plus_i(x1692);

// Dealloc(X1751)
JCudaTensor x1693;
x1693 = x1548;
x1693.free();

// val X1788 = X1776 * d_ReLU()(X343)/d_X342
JCudaTensor x1694;
JCudaTensor x1695, x1696;
x1695 = x1690;
x1696 = x755;
x1694 = x751.backward(x1695,x1696);

// Dealloc(X343)
JCudaTensor x1697;
x1697 = x755;
x1697.free();

// val X1798 = X1788.copy * d_ReLU()(X341)/d_X340
JCudaTensor x1698;
JCudaTensor x1699, x1700;
x1699 = x1694;
x1699 = x1699.clone();
x1700 = x749;
x1698 = x751.backward(x1699,x1700);

// Dealloc(X341)
JCudaTensor x1701;
x1701 = x749;
x1701.free();

// val X175471 = X1798 * d_BatchNorm()(X339,4e_c_bn_scale)/d_4e_c_bn_scale
JCudaTensor x1702;
JCudaTensor x1703, x1704, x1705;
x1703 = x1698;
x1704 = x735;
x1705 = x746;
JCudaTensor[] x1706 = x748.backward(x1703,x1704,x1705);
x1702 = x1706[1];

// val X1799 = X1798 * d_BatchNorm()(X339,4e_c_bn_scale)/d_X339
JCudaTensor x1707;
x1707 = x1706[0];

// val X174452 = X1798 * d_BatchNorm()(X339,4e_c_bn_scale)/d_4e_c_bn_bias
JCudaTensor x1711;
x1711 = x1706[2];

// Dealloc(X339)
JCudaTensor x1715;
x1715 = x735;
x1715.free();

// val X1800 = X1799 * d_Convolv(1,0)(4e_c_cv_W)/d_X338
JCudaTensor x1716;
JCudaTensor x1717, x1718;
x1717 = x1707;
x1718 = x739;
x1716 = x741.backward_data(x1717,x1718);

// V_4e_c_cv_W <~~ X1799 * d_Convolv(1,0)(X338)/d_4e_c_cv_W
float x1720, x1721;
x1720 = lrn_rate_1;
x1721 = momentum;
JCudaTensor x1722, x1723;
x1722 = x1707;
x1723 = x732;
x741.backward_filter(x1722,x1723, x1719, x1720, x1721);

// Dealloc(X1799)
JCudaTensor x1724;
x1724 = x1707;
x1724.free();

// V_4e_c_bn_scale <~~ X175471
float x1726, x1727;
x1726 = lrn_rate_1;
x1727 = momentum;
JCudaTensor x1728;
x1728 = x1702;
x1725.update(x1728, x1726, x1727);


// Dealloc(X175471)
JCudaTensor x1729;
x1729 = x1702;
x1729.free();

// V_4e_c_bn_bias <~~ X174452
float x1731, x1732;
x1731 = lrn_rate_1;
x1732 = momentum;
JCudaTensor x1733;
x1733 = x1711;
x1730.update(x1733, x1731, x1732);


// Dealloc(X174452)
JCudaTensor x1734;
x1734 = x1711;
x1734.free();

// 4e_c_cv_W <~~ V_4e_c_cv_W
float x1735, x1736;
x1735 = 1;
x1736 = decay_1;
JCudaTensor x1737;
x1737 = x1719;
x739.update(x1737, x1735, x1736);


// 4e_c_bn_scale <~~ V_4e_c_bn_scale
float x1738, x1739;
x1738 = 1;
x1739 = decay_1;
JCudaTensor x1740;
x1740 = x1725;
x746.update(x1740, x1738, x1739);


// 4e_c_bn_bias <~~ V_4e_c_bn_bias
float x1741, x1742;
x1741 = 1;
x1742 = decay_1;
JCudaTensor x1743;
x1743 = x1730;
x747.update(x1743, x1741, x1742);


// val X1804 = X1800 * d_ReLU()(X338)/d_X337
JCudaTensor x1744;
JCudaTensor x1745, x1746;
x1745 = x1716;
x1746 = x732;
x1744 = x734.backward(x1745,x1746);

// Dealloc(X338)
JCudaTensor x1747;
x1747 = x732;
x1747.free();

// val X172399 = X1804 * d_BatchNorm()(X336,4e_b_bn_scale)/d_4e_b_bn_scale
JCudaTensor x1748;
JCudaTensor x1749, x1750, x1751;
x1749 = x1744;
x1750 = x718;
x1751 = x729;
JCudaTensor[] x1752 = x731.backward(x1749,x1750,x1751);
x1748 = x1752[1];

// val X1805 = X1804 * d_BatchNorm()(X336,4e_b_bn_scale)/d_X336
JCudaTensor x1753;
x1753 = x1752[0];

// val X171367 = X1804 * d_BatchNorm()(X336,4e_b_bn_scale)/d_4e_b_bn_bias
JCudaTensor x1757;
x1757 = x1752[2];

// Dealloc(X336)
JCudaTensor x1761;
x1761 = x718;
x1761.free();

// val X1806 = X1805 * d_Convolv(1,1)(4e_b_cv_W)/d_X335
JCudaTensor x1762;
JCudaTensor x1763, x1764;
x1763 = x1753;
x1764 = x722;
x1762 = x724.backward_data(x1763,x1764);

// V_4e_b_bn_bias <~~ X171367
float x1766, x1767;
x1766 = lrn_rate_1;
x1767 = momentum;
JCudaTensor x1768;
x1768 = x1757;
x1765.update(x1768, x1766, x1767);


// Dealloc(X171367)
JCudaTensor x1769;
x1769 = x1757;
x1769.free();

// V_4e_b_bn_scale <~~ X172399
float x1771, x1772;
x1771 = lrn_rate_1;
x1772 = momentum;
JCudaTensor x1773;
x1773 = x1748;
x1770.update(x1773, x1771, x1772);


// Dealloc(X172399)
JCudaTensor x1774;
x1774 = x1748;
x1774.free();

// V_4e_b_cv_W <~~ X1805 * d_Convolv(1,1)(X335)/d_4e_b_cv_W
float x1776, x1777;
x1776 = lrn_rate_1;
x1777 = momentum;
JCudaTensor x1778, x1779;
x1778 = x1753;
x1779 = x715;
x724.backward_filter(x1778,x1779, x1775, x1776, x1777);

// Dealloc(X1805)
JCudaTensor x1780;
x1780 = x1753;
x1780.free();

// 4e_b_bn_bias <~~ V_4e_b_bn_bias
float x1781, x1782;
x1781 = 1;
x1782 = decay_1;
JCudaTensor x1783;
x1783 = x1765;
x730.update(x1783, x1781, x1782);


// 4e_b_bn_scale <~~ V_4e_b_bn_scale
float x1784, x1785;
x1784 = 1;
x1785 = decay_1;
JCudaTensor x1786;
x1786 = x1770;
x729.update(x1786, x1784, x1785);


// 4e_b_cv_W <~~ V_4e_b_cv_W
float x1787, x1788;
x1787 = 1;
x1788 = decay_1;
JCudaTensor x1789;
x1789 = x1775;
x722.update(x1789, x1787, x1788);


// val X1810 = X1806 * d_ReLU()(X335)/d_X334
JCudaTensor x1790;
JCudaTensor x1791, x1792;
x1791 = x1762;
x1792 = x715;
x1790 = x717.backward(x1791,x1792);

// Dealloc(X335)
JCudaTensor x1793;
x1793 = x715;
x1793.free();

// val X169288 = X1810 * d_BatchNorm()(X333,4e_a_bn_scale)/d_4e_a_bn_scale
JCudaTensor x1794;
JCudaTensor x1795, x1796, x1797;
x1795 = x1790;
x1796 = x701;
x1797 = x712;
JCudaTensor[] x1798 = x714.backward(x1795,x1796,x1797);
x1794 = x1798[1];

// val X1811 = X1810 * d_BatchNorm()(X333,4e_a_bn_scale)/d_X333
JCudaTensor x1799;
x1799 = x1798[0];

// val X168243 = X1810 * d_BatchNorm()(X333,4e_a_bn_scale)/d_4e_a_bn_bias
JCudaTensor x1803;
x1803 = x1798[2];

// Dealloc(X333)
JCudaTensor x1807;
x1807 = x701;
x1807.free();

// val X1812 = X1811 * d_Convolv(1,0)(4e_a_cv_W)/d_X332
JCudaTensor x1808;
JCudaTensor x1809, x1810;
x1809 = x1799;
x1810 = x705;
x1808 = x707.backward_data(x1809,x1810);

// V_4e_a_cv_W <~~ X1811 * d_Convolv(1,0)(X332)/d_4e_a_cv_W
float x1812, x1813;
x1812 = lrn_rate_1;
x1813 = momentum;
JCudaTensor x1814, x1815;
x1814 = x1799;
x1815 = x699;
x707.backward_filter(x1814,x1815, x1811, x1812, x1813);

// Dealloc(X1811)
JCudaTensor x1816;
x1816 = x1799;
x1816.free();

// V_4e_a_bn_scale <~~ X169288
float x1818, x1819;
x1818 = lrn_rate_1;
x1819 = momentum;
JCudaTensor x1820;
x1820 = x1794;
x1817.update(x1820, x1818, x1819);


// Dealloc(X169288)
JCudaTensor x1821;
x1821 = x1794;
x1821.free();

// V_4e_a_bn_bias <~~ X168243
float x1823, x1824;
x1823 = lrn_rate_1;
x1824 = momentum;
JCudaTensor x1825;
x1825 = x1803;
x1822.update(x1825, x1823, x1824);


// Dealloc(X168243)
JCudaTensor x1826;
x1826 = x1803;
x1826.free();

// 4e_a_cv_W <~~ V_4e_a_cv_W
float x1827, x1828;
x1827 = 1;
x1828 = decay_1;
JCudaTensor x1829;
x1829 = x1811;
x705.update(x1829, x1827, x1828);


// 4e_a_bn_scale <~~ V_4e_a_bn_scale
float x1830, x1831;
x1830 = 1;
x1831 = decay_1;
JCudaTensor x1832;
x1832 = x1817;
x712.update(x1832, x1830, x1831);


// 4e_a_bn_bias <~~ V_4e_a_bn_bias
float x1833, x1834;
x1833 = 1;
x1834 = decay_1;
JCudaTensor x1835;
x1835 = x1822;
x713.update(x1835, x1833, x1834);


// val X1813 = (X1812 + X1788)
JCudaTensor x1836;
JCudaTensor x1837, x1838;
x1837 = x1808;
x1838 = x1694;
x1836 = x1837.plus_i(x1838);

// Dealloc(X1788)
JCudaTensor x1839;
x1839 = x1694;
x1839.free();

// val X1825 = X1813 * d_ReLU()(X332)/d_X331
JCudaTensor x1840;
JCudaTensor x1841, x1842;
x1841 = x1836;
x1842 = x699;
x1840 = x695.backward(x1841,x1842);

// Dealloc(X332)
JCudaTensor x1843;
x1843 = x699;
x1843.free();

// val X1835 = X1825.copy * d_ReLU()(X330)/d_X329
JCudaTensor x1844;
JCudaTensor x1845, x1846;
x1845 = x1840;
x1845 = x1845.clone();
x1846 = x693;
x1844 = x695.backward(x1845,x1846);

// Dealloc(X330)
JCudaTensor x1847;
x1847 = x693;
x1847.free();

// val X165024 = X1835 * d_BatchNorm()(X328,4d_c_bn_scale)/d_4d_c_bn_bias
JCudaTensor x1848;
JCudaTensor x1849, x1850, x1851;
x1849 = x1844;
x1850 = x679;
x1851 = x690;
JCudaTensor[] x1852 = x692.backward(x1849,x1850,x1851);
x1848 = x1852[2];

// val X1836 = X1835 * d_BatchNorm()(X328,4d_c_bn_scale)/d_X328
JCudaTensor x1853;
x1853 = x1852[0];

// val X166110 = X1835 * d_BatchNorm()(X328,4d_c_bn_scale)/d_4d_c_bn_scale
JCudaTensor x1857;
x1857 = x1852[1];

// Dealloc(X328)
JCudaTensor x1861;
x1861 = x679;
x1861.free();

// V_4d_c_bn_scale <~~ X166110
float x1863, x1864;
x1863 = lrn_rate_1;
x1864 = momentum;
JCudaTensor x1865;
x1865 = x1857;
x1862.update(x1865, x1863, x1864);


// Dealloc(X166110)
JCudaTensor x1866;
x1866 = x1857;
x1866.free();

// val X1837 = X1836 * d_Convolv(1,0)(4d_c_cv_W)/d_X327
JCudaTensor x1867;
JCudaTensor x1868, x1869;
x1868 = x1853;
x1869 = x683;
x1867 = x685.backward_data(x1868,x1869);

// V_4d_c_bn_bias <~~ X165024
float x1871, x1872;
x1871 = lrn_rate_1;
x1872 = momentum;
JCudaTensor x1873;
x1873 = x1848;
x1870.update(x1873, x1871, x1872);


// Dealloc(X165024)
JCudaTensor x1874;
x1874 = x1848;
x1874.free();

// V_4d_c_cv_W <~~ X1836 * d_Convolv(1,0)(X327)/d_4d_c_cv_W
float x1876, x1877;
x1876 = lrn_rate_1;
x1877 = momentum;
JCudaTensor x1878, x1879;
x1878 = x1853;
x1879 = x676;
x685.backward_filter(x1878,x1879, x1875, x1876, x1877);

// Dealloc(X1836)
JCudaTensor x1880;
x1880 = x1853;
x1880.free();

// 4d_c_bn_scale <~~ V_4d_c_bn_scale
float x1881, x1882;
x1881 = 1;
x1882 = decay_1;
JCudaTensor x1883;
x1883 = x1862;
x690.update(x1883, x1881, x1882);


// 4d_c_bn_bias <~~ V_4d_c_bn_bias
float x1884, x1885;
x1884 = 1;
x1885 = decay_1;
JCudaTensor x1886;
x1886 = x1870;
x691.update(x1886, x1884, x1885);


// 4d_c_cv_W <~~ V_4d_c_cv_W
float x1887, x1888;
x1887 = 1;
x1888 = decay_1;
JCudaTensor x1889;
x1889 = x1875;
x683.update(x1889, x1887, x1888);


// val X1841 = X1837 * d_ReLU()(X327)/d_X326
JCudaTensor x1890;
JCudaTensor x1891, x1892;
x1891 = x1867;
x1892 = x676;
x1890 = x678.backward(x1891,x1892);

// Dealloc(X327)
JCudaTensor x1893;
x1893 = x676;
x1893.free();

// val X1842 = X1841 * d_BatchNorm()(X325,4d_b_bn_scale)/d_X325
JCudaTensor x1894;
JCudaTensor x1895, x1896, x1897;
x1895 = x1890;
x1896 = x662;
x1897 = x673;
JCudaTensor[] x1898 = x675.backward(x1895,x1896,x1897);
x1894 = x1898[0];

// val X161738 = X1841 * d_BatchNorm()(X325,4d_b_bn_scale)/d_4d_b_bn_bias
JCudaTensor x1899;
x1899 = x1898[2];

// val X162837 = X1841 * d_BatchNorm()(X325,4d_b_bn_scale)/d_4d_b_bn_scale
JCudaTensor x1903;
x1903 = x1898[1];

// Dealloc(X325)
JCudaTensor x1907;
x1907 = x662;
x1907.free();

// val X1843 = X1842 * d_Convolv(1,1)(4d_b_cv_W)/d_X324
JCudaTensor x1908;
JCudaTensor x1909, x1910;
x1909 = x1894;
x1910 = x666;
x1908 = x668.backward_data(x1909,x1910);

// V_4d_b_cv_W <~~ X1842 * d_Convolv(1,1)(X324)/d_4d_b_cv_W
float x1912, x1913;
x1912 = lrn_rate_1;
x1913 = momentum;
JCudaTensor x1914, x1915;
x1914 = x1894;
x1915 = x659;
x668.backward_filter(x1914,x1915, x1911, x1912, x1913);

// Dealloc(X1842)
JCudaTensor x1916;
x1916 = x1894;
x1916.free();

// V_4d_b_bn_scale <~~ X162837
float x1918, x1919;
x1918 = lrn_rate_1;
x1919 = momentum;
JCudaTensor x1920;
x1920 = x1903;
x1917.update(x1920, x1918, x1919);


// Dealloc(X162837)
JCudaTensor x1921;
x1921 = x1903;
x1921.free();

// V_4d_b_bn_bias <~~ X161738
float x1923, x1924;
x1923 = lrn_rate_1;
x1924 = momentum;
JCudaTensor x1925;
x1925 = x1899;
x1922.update(x1925, x1923, x1924);


// Dealloc(X161738)
JCudaTensor x1926;
x1926 = x1899;
x1926.free();

// 4d_b_cv_W <~~ V_4d_b_cv_W
float x1927, x1928;
x1927 = 1;
x1928 = decay_1;
JCudaTensor x1929;
x1929 = x1911;
x666.update(x1929, x1927, x1928);


// 4d_b_bn_scale <~~ V_4d_b_bn_scale
float x1930, x1931;
x1930 = 1;
x1931 = decay_1;
JCudaTensor x1932;
x1932 = x1917;
x673.update(x1932, x1930, x1931);


// 4d_b_bn_bias <~~ V_4d_b_bn_bias
float x1933, x1934;
x1933 = 1;
x1934 = decay_1;
JCudaTensor x1935;
x1935 = x1922;
x674.update(x1935, x1933, x1934);


// val X1847 = X1843 * d_ReLU()(X324)/d_X323
JCudaTensor x1936;
JCudaTensor x1937, x1938;
x1937 = x1908;
x1938 = x659;
x1936 = x661.backward(x1937,x1938);

// Dealloc(X324)
JCudaTensor x1939;
x1939 = x659;
x1939.free();

// val X1848 = X1847 * d_BatchNorm()(X322,4d_a_bn_scale)/d_X322
JCudaTensor x1940;
JCudaTensor x1941, x1942, x1943;
x1941 = x1936;
x1942 = x645;
x1943 = x656;
JCudaTensor[] x1944 = x658.backward(x1941,x1942,x1943);
x1940 = x1944[0];

// val X158413 = X1847 * d_BatchNorm()(X322,4d_a_bn_scale)/d_4d_a_bn_bias
JCudaTensor x1945;
x1945 = x1944[2];

// val X159525 = X1847 * d_BatchNorm()(X322,4d_a_bn_scale)/d_4d_a_bn_scale
JCudaTensor x1949;
x1949 = x1944[1];

// Dealloc(X322)
JCudaTensor x1953;
x1953 = x645;
x1953.free();

// V_4d_a_bn_bias <~~ X158413
float x1955, x1956;
x1955 = lrn_rate_1;
x1956 = momentum;
JCudaTensor x1957;
x1957 = x1945;
x1954.update(x1957, x1955, x1956);


// Dealloc(X158413)
JCudaTensor x1958;
x1958 = x1945;
x1958.free();

// val X1849 = X1848 * d_Convolv(1,0)(4d_a_cv_W)/d_X321
JCudaTensor x1959;
JCudaTensor x1960, x1961;
x1960 = x1940;
x1961 = x649;
x1959 = x651.backward_data(x1960,x1961);

// V_4d_a_cv_W <~~ X1848 * d_Convolv(1,0)(X321)/d_4d_a_cv_W
float x1963, x1964;
x1963 = lrn_rate_1;
x1964 = momentum;
JCudaTensor x1965, x1966;
x1965 = x1940;
x1966 = x643;
x651.backward_filter(x1965,x1966, x1962, x1963, x1964);

// Dealloc(X1848)
JCudaTensor x1967;
x1967 = x1940;
x1967.free();

// V_4d_a_bn_scale <~~ X159525
float x1969, x1970;
x1969 = lrn_rate_1;
x1970 = momentum;
JCudaTensor x1971;
x1971 = x1949;
x1968.update(x1971, x1969, x1970);


// Dealloc(X159525)
JCudaTensor x1972;
x1972 = x1949;
x1972.free();

// 4d_a_bn_bias <~~ V_4d_a_bn_bias
float x1973, x1974;
x1973 = 1;
x1974 = decay_1;
JCudaTensor x1975;
x1975 = x1954;
x657.update(x1975, x1973, x1974);


// 4d_a_cv_W <~~ V_4d_a_cv_W
float x1976, x1977;
x1976 = 1;
x1977 = decay_1;
JCudaTensor x1978;
x1978 = x1962;
x649.update(x1978, x1976, x1977);


// 4d_a_bn_scale <~~ V_4d_a_bn_scale
float x1979, x1980;
x1979 = 1;
x1980 = decay_1;
JCudaTensor x1981;
x1981 = x1968;
x656.update(x1981, x1979, x1980);


// val X1850 = (X1849 + X1825)
JCudaTensor x1982;
JCudaTensor x1983, x1984;
x1983 = x1959;
x1984 = x1840;
x1982 = x1983.plus_i(x1984);

// Dealloc(X1825)
JCudaTensor x1985;
x1985 = x1840;
x1985.free();

// val X1862 = X1850 * d_ReLU()(X321)/d_X320
JCudaTensor x1986;
JCudaTensor x1987, x1988;
x1987 = x1982;
x1988 = x643;
x1986 = x639.backward(x1987,x1988);

// Dealloc(X321)
JCudaTensor x1989;
x1989 = x643;
x1989.free();

// val X1872 = X1862.copy * d_ReLU()(X319)/d_X318
JCudaTensor x1990;
JCudaTensor x1991, x1992;
x1991 = x1986;
x1991 = x1991.clone();
x1992 = x637;
x1990 = x639.backward(x1991,x1992);

// Dealloc(X319)
JCudaTensor x1993;
x1993 = x637;
x1993.free();

// val X1873 = X1872 * d_BatchNorm()(X317,4c_c_bn_scale)/d_X317
JCudaTensor x1994;
JCudaTensor x1995, x1996, x1997;
x1995 = x1990;
x1996 = x623;
x1997 = x634;
JCudaTensor[] x1998 = x636.backward(x1995,x1996,x1997);
x1994 = x1998[0];

// val X156146 = X1872 * d_BatchNorm()(X317,4c_c_bn_scale)/d_4c_c_bn_scale
JCudaTensor x1999;
x1999 = x1998[1];

// val X154993 = X1872 * d_BatchNorm()(X317,4c_c_bn_scale)/d_4c_c_bn_bias
JCudaTensor x2003;
x2003 = x1998[2];

// Dealloc(X317)
JCudaTensor x2007;
x2007 = x623;
x2007.free();

// V_4c_c_bn_bias <~~ X154993
float x2009, x2010;
x2009 = lrn_rate_1;
x2010 = momentum;
JCudaTensor x2011;
x2011 = x2003;
x2008.update(x2011, x2009, x2010);


// Dealloc(X154993)
JCudaTensor x2012;
x2012 = x2003;
x2012.free();

// val X1874 = X1873 * d_Convolv(1,0)(4c_c_cv_W)/d_X316
JCudaTensor x2013;
JCudaTensor x2014, x2015;
x2014 = x1994;
x2015 = x627;
x2013 = x629.backward_data(x2014,x2015);

// V_4c_c_cv_W <~~ X1873 * d_Convolv(1,0)(X316)/d_4c_c_cv_W
float x2017, x2018;
x2017 = lrn_rate_1;
x2018 = momentum;
JCudaTensor x2019, x2020;
x2019 = x1994;
x2020 = x620;
x629.backward_filter(x2019,x2020, x2016, x2017, x2018);

// Dealloc(X1873)
JCudaTensor x2021;
x2021 = x1994;
x2021.free();

// V_4c_c_bn_scale <~~ X156146
float x2023, x2024;
x2023 = lrn_rate_1;
x2024 = momentum;
JCudaTensor x2025;
x2025 = x1999;
x2022.update(x2025, x2023, x2024);


// Dealloc(X156146)
JCudaTensor x2026;
x2026 = x1999;
x2026.free();

// 4c_c_bn_bias <~~ V_4c_c_bn_bias
float x2027, x2028;
x2027 = 1;
x2028 = decay_1;
JCudaTensor x2029;
x2029 = x2008;
x635.update(x2029, x2027, x2028);


// 4c_c_cv_W <~~ V_4c_c_cv_W
float x2030, x2031;
x2030 = 1;
x2031 = decay_1;
JCudaTensor x2032;
x2032 = x2016;
x627.update(x2032, x2030, x2031);


// 4c_c_bn_scale <~~ V_4c_c_bn_scale
float x2033, x2034;
x2033 = 1;
x2034 = decay_1;
JCudaTensor x2035;
x2035 = x2022;
x634.update(x2035, x2033, x2034);


// val X1878 = X1874 * d_ReLU()(X316)/d_X315
JCudaTensor x2036;
JCudaTensor x2037, x2038;
x2037 = x2013;
x2038 = x620;
x2036 = x622.backward(x2037,x2038);

// Dealloc(X316)
JCudaTensor x2039;
x2039 = x620;
x2039.free();

// val X1879 = X1878 * d_BatchNorm()(X314,4c_b_bn_scale)/d_X314
JCudaTensor x2040;
JCudaTensor x2041, x2042, x2043;
x2041 = x2036;
x2042 = x606;
x2043 = x617;
JCudaTensor[] x2044 = x619.backward(x2041,x2042,x2043);
x2040 = x2044[0];

// val X151506 = X1878 * d_BatchNorm()(X314,4c_b_bn_scale)/d_4c_b_bn_bias
JCudaTensor x2045;
x2045 = x2044[2];

// val X152672 = X1878 * d_BatchNorm()(X314,4c_b_bn_scale)/d_4c_b_bn_scale
JCudaTensor x2049;
x2049 = x2044[1];

// Dealloc(X314)
JCudaTensor x2053;
x2053 = x606;
x2053.free();

// val X1880 = X1879 * d_Convolv(1,1)(4c_b_cv_W)/d_X313
JCudaTensor x2054;
JCudaTensor x2055, x2056;
x2055 = x2040;
x2056 = x610;
x2054 = x612.backward_data(x2055,x2056);

// V_4c_b_cv_W <~~ X1879 * d_Convolv(1,1)(X313)/d_4c_b_cv_W
float x2058, x2059;
x2058 = lrn_rate_1;
x2059 = momentum;
JCudaTensor x2060, x2061;
x2060 = x2040;
x2061 = x603;
x612.backward_filter(x2060,x2061, x2057, x2058, x2059);

// Dealloc(X1879)
JCudaTensor x2062;
x2062 = x2040;
x2062.free();

// V_4c_b_bn_bias <~~ X151506
float x2064, x2065;
x2064 = lrn_rate_1;
x2065 = momentum;
JCudaTensor x2066;
x2066 = x2045;
x2063.update(x2066, x2064, x2065);


// Dealloc(X151506)
JCudaTensor x2067;
x2067 = x2045;
x2067.free();

// V_4c_b_bn_scale <~~ X152672
float x2069, x2070;
x2069 = lrn_rate_1;
x2070 = momentum;
JCudaTensor x2071;
x2071 = x2049;
x2068.update(x2071, x2069, x2070);


// Dealloc(X152672)
JCudaTensor x2072;
x2072 = x2049;
x2072.free();

// 4c_b_cv_W <~~ V_4c_b_cv_W
float x2073, x2074;
x2073 = 1;
x2074 = decay_1;
JCudaTensor x2075;
x2075 = x2057;
x610.update(x2075, x2073, x2074);


// 4c_b_bn_bias <~~ V_4c_b_bn_bias
float x2076, x2077;
x2076 = 1;
x2077 = decay_1;
JCudaTensor x2078;
x2078 = x2063;
x618.update(x2078, x2076, x2077);


// 4c_b_bn_scale <~~ V_4c_b_bn_scale
float x2079, x2080;
x2079 = 1;
x2080 = decay_1;
JCudaTensor x2081;
x2081 = x2068;
x617.update(x2081, x2079, x2080);


// val X1884 = X1880 * d_ReLU()(X313)/d_X312
JCudaTensor x2082;
JCudaTensor x2083, x2084;
x2083 = x2054;
x2084 = x603;
x2082 = x605.backward(x2083,x2084);

// Dealloc(X313)
JCudaTensor x2085;
x2085 = x603;
x2085.free();

// val X1885 = X1884 * d_BatchNorm()(X311,4c_a_bn_scale)/d_X311
JCudaTensor x2086;
JCudaTensor x2087, x2088, x2089;
x2087 = x2082;
x2088 = x589;
x2089 = x600;
JCudaTensor[] x2090 = x602.backward(x2087,x2088,x2089);
x2086 = x2090[0];

// val X147980 = X1884 * d_BatchNorm()(X311,4c_a_bn_scale)/d_4c_a_bn_bias
JCudaTensor x2091;
x2091 = x2090[2];

// val X149159 = X1884 * d_BatchNorm()(X311,4c_a_bn_scale)/d_4c_a_bn_scale
JCudaTensor x2095;
x2095 = x2090[1];

// Dealloc(X311)
JCudaTensor x2099;
x2099 = x589;
x2099.free();

// val X1886 = X1885 * d_Convolv(1,0)(4c_a_cv_W)/d_X310
JCudaTensor x2100;
JCudaTensor x2101, x2102;
x2101 = x2086;
x2102 = x593;
x2100 = x595.backward_data(x2101,x2102);

// V_4c_a_bn_bias <~~ X147980
float x2104, x2105;
x2104 = lrn_rate_1;
x2105 = momentum;
JCudaTensor x2106;
x2106 = x2091;
x2103.update(x2106, x2104, x2105);


// Dealloc(X147980)
JCudaTensor x2107;
x2107 = x2091;
x2107.free();

// V_4c_a_bn_scale <~~ X149159
float x2109, x2110;
x2109 = lrn_rate_1;
x2110 = momentum;
JCudaTensor x2111;
x2111 = x2095;
x2108.update(x2111, x2109, x2110);


// Dealloc(X149159)
JCudaTensor x2112;
x2112 = x2095;
x2112.free();

// V_4c_a_cv_W <~~ X1885 * d_Convolv(1,0)(X310)/d_4c_a_cv_W
float x2114, x2115;
x2114 = lrn_rate_1;
x2115 = momentum;
JCudaTensor x2116, x2117;
x2116 = x2086;
x2117 = x587;
x595.backward_filter(x2116,x2117, x2113, x2114, x2115);

// Dealloc(X1885)
JCudaTensor x2118;
x2118 = x2086;
x2118.free();

// 4c_a_bn_bias <~~ V_4c_a_bn_bias
float x2119, x2120;
x2119 = 1;
x2120 = decay_1;
JCudaTensor x2121;
x2121 = x2103;
x601.update(x2121, x2119, x2120);


// 4c_a_bn_scale <~~ V_4c_a_bn_scale
float x2122, x2123;
x2122 = 1;
x2123 = decay_1;
JCudaTensor x2124;
x2124 = x2108;
x600.update(x2124, x2122, x2123);


// 4c_a_cv_W <~~ V_4c_a_cv_W
float x2125, x2126;
x2125 = 1;
x2126 = decay_1;
JCudaTensor x2127;
x2127 = x2113;
x593.update(x2127, x2125, x2126);


// val X1887 = (X1886 + X1862)
JCudaTensor x2128;
JCudaTensor x2129, x2130;
x2129 = x2100;
x2130 = x1986;
x2128 = x2129.plus_i(x2130);

// Dealloc(X1862)
JCudaTensor x2131;
x2131 = x1986;
x2131.free();

// val X1899 = X1887 * d_ReLU()(X310)/d_X309
JCudaTensor x2132;
JCudaTensor x2133, x2134;
x2133 = x2128;
x2134 = x587;
x2132 = x583.backward(x2133,x2134);

// Dealloc(X310)
JCudaTensor x2135;
x2135 = x587;
x2135.free();

// val X1909 = X1899.copy * d_ReLU()(X308)/d_X307
JCudaTensor x2136;
JCudaTensor x2137, x2138;
x2137 = x2132;
x2137 = x2137.clone();
x2138 = x581;
x2136 = x583.backward(x2137,x2138);

// Dealloc(X308)
JCudaTensor x2139;
x2139 = x581;
x2139.free();

// val X1910 = X1909 * d_BatchNorm()(X306,4b_c_bn_scale)/d_X306
JCudaTensor x2140;
JCudaTensor x2141, x2142, x2143;
x2141 = x2136;
x2142 = x567;
x2143 = x578;
JCudaTensor[] x2144 = x580.backward(x2141,x2142,x2143);
x2140 = x2144[0];

// val X144359 = X1909 * d_BatchNorm()(X306,4b_c_bn_scale)/d_4b_c_bn_bias
JCudaTensor x2145;
x2145 = x2144[2];

// val X145579 = X1909 * d_BatchNorm()(X306,4b_c_bn_scale)/d_4b_c_bn_scale
JCudaTensor x2149;
x2149 = x2144[1];

// Dealloc(X306)
JCudaTensor x2153;
x2153 = x567;
x2153.free();

// V_4b_c_bn_bias <~~ X144359
float x2155, x2156;
x2155 = lrn_rate_1;
x2156 = momentum;
JCudaTensor x2157;
x2157 = x2145;
x2154.update(x2157, x2155, x2156);


// Dealloc(X144359)
JCudaTensor x2158;
x2158 = x2145;
x2158.free();

// val X1911 = X1910 * d_Convolv(1,0)(4b_c_cv_W)/d_X305
JCudaTensor x2159;
JCudaTensor x2160, x2161;
x2160 = x2140;
x2161 = x571;
x2159 = x573.backward_data(x2160,x2161);

// V_4b_c_cv_W <~~ X1910 * d_Convolv(1,0)(X305)/d_4b_c_cv_W
float x2163, x2164;
x2163 = lrn_rate_1;
x2164 = momentum;
JCudaTensor x2165, x2166;
x2165 = x2140;
x2166 = x564;
x573.backward_filter(x2165,x2166, x2162, x2163, x2164);

// Dealloc(X1910)
JCudaTensor x2167;
x2167 = x2140;
x2167.free();

// V_4b_c_bn_scale <~~ X145579
float x2169, x2170;
x2169 = lrn_rate_1;
x2170 = momentum;
JCudaTensor x2171;
x2171 = x2149;
x2168.update(x2171, x2169, x2170);


// Dealloc(X145579)
JCudaTensor x2172;
x2172 = x2149;
x2172.free();

// 4b_c_bn_bias <~~ V_4b_c_bn_bias
float x2173, x2174;
x2173 = 1;
x2174 = decay_1;
JCudaTensor x2175;
x2175 = x2154;
x579.update(x2175, x2173, x2174);


// 4b_c_cv_W <~~ V_4b_c_cv_W
float x2176, x2177;
x2176 = 1;
x2177 = decay_1;
JCudaTensor x2178;
x2178 = x2162;
x571.update(x2178, x2176, x2177);


// 4b_c_bn_scale <~~ V_4b_c_bn_scale
float x2179, x2180;
x2179 = 1;
x2180 = decay_1;
JCudaTensor x2181;
x2181 = x2168;
x578.update(x2181, x2179, x2180);


// val X1915 = X1911 * d_ReLU()(X305)/d_X304
JCudaTensor x2182;
JCudaTensor x2183, x2184;
x2183 = x2159;
x2184 = x564;
x2182 = x566.backward(x2183,x2184);

// Dealloc(X305)
JCudaTensor x2185;
x2185 = x564;
x2185.free();

// val X141904 = X1915 * d_BatchNorm()(X303,4b_b_bn_scale)/d_4b_b_bn_scale
JCudaTensor x2186;
JCudaTensor x2187, x2188, x2189;
x2187 = x2182;
x2188 = x550;
x2189 = x561;
JCudaTensor[] x2190 = x563.backward(x2187,x2188,x2189);
x2186 = x2190[1];

// val X1916 = X1915 * d_BatchNorm()(X303,4b_b_bn_scale)/d_X303
JCudaTensor x2191;
x2191 = x2190[0];

// val X140671 = X1915 * d_BatchNorm()(X303,4b_b_bn_scale)/d_4b_b_bn_bias
JCudaTensor x2195;
x2195 = x2190[2];

// Dealloc(X303)
JCudaTensor x2199;
x2199 = x550;
x2199.free();

// val X1917 = X1916 * d_Convolv(1,1)(4b_b_cv_W)/d_X302
JCudaTensor x2200;
JCudaTensor x2201, x2202;
x2201 = x2191;
x2202 = x554;
x2200 = x556.backward_data(x2201,x2202);

// V_4b_b_cv_W <~~ X1916 * d_Convolv(1,1)(X302)/d_4b_b_cv_W
float x2204, x2205;
x2204 = lrn_rate_1;
x2205 = momentum;
JCudaTensor x2206, x2207;
x2206 = x2191;
x2207 = x547;
x556.backward_filter(x2206,x2207, x2203, x2204, x2205);

// Dealloc(X1916)
JCudaTensor x2208;
x2208 = x2191;
x2208.free();

// V_4b_b_bn_scale <~~ X141904
float x2210, x2211;
x2210 = lrn_rate_1;
x2211 = momentum;
JCudaTensor x2212;
x2212 = x2186;
x2209.update(x2212, x2210, x2211);


// Dealloc(X141904)
JCudaTensor x2213;
x2213 = x2186;
x2213.free();

// V_4b_b_bn_bias <~~ X140671
float x2215, x2216;
x2215 = lrn_rate_1;
x2216 = momentum;
JCudaTensor x2217;
x2217 = x2195;
x2214.update(x2217, x2215, x2216);


// Dealloc(X140671)
JCudaTensor x2218;
x2218 = x2195;
x2218.free();

// 4b_b_cv_W <~~ V_4b_b_cv_W
float x2219, x2220;
x2219 = 1;
x2220 = decay_1;
JCudaTensor x2221;
x2221 = x2203;
x554.update(x2221, x2219, x2220);


// 4b_b_bn_scale <~~ V_4b_b_bn_scale
float x2222, x2223;
x2222 = 1;
x2223 = decay_1;
JCudaTensor x2224;
x2224 = x2209;
x561.update(x2224, x2222, x2223);


// 4b_b_bn_bias <~~ V_4b_b_bn_bias
float x2225, x2226;
x2225 = 1;
x2226 = decay_1;
JCudaTensor x2227;
x2227 = x2214;
x562.update(x2227, x2225, x2226);


// val X1921 = X1917 * d_ReLU()(X302)/d_X301
JCudaTensor x2228;
JCudaTensor x2229, x2230;
x2229 = x2200;
x2230 = x547;
x2228 = x549.backward(x2229,x2230);

// Dealloc(X302)
JCudaTensor x2231;
x2231 = x547;
x2231.free();

// val X1922 = X1921 * d_BatchNorm()(X300,4b_a_bn_scale)/d_X300
JCudaTensor x2232;
JCudaTensor x2233, x2234, x2235;
x2233 = x2228;
x2234 = x533;
x2235 = x544;
JCudaTensor[] x2236 = x546.backward(x2233,x2234,x2235);
x2232 = x2236[0];

// val X136944 = X1921 * d_BatchNorm()(X300,4b_a_bn_scale)/d_4b_a_bn_bias
JCudaTensor x2237;
x2237 = x2236[2];

// val X138190 = X1921 * d_BatchNorm()(X300,4b_a_bn_scale)/d_4b_a_bn_scale
JCudaTensor x2241;
x2241 = x2236[1];

// Dealloc(X300)
JCudaTensor x2245;
x2245 = x533;
x2245.free();

// val X1923 = X1922 * d_Convolv(1,0)(4b_a_cv_W)/d_X299
JCudaTensor x2246;
JCudaTensor x2247, x2248;
x2247 = x2232;
x2248 = x537;
x2246 = x539.backward_data(x2247,x2248);

// V_4b_a_cv_W <~~ X1922 * d_Convolv(1,0)(X299)/d_4b_a_cv_W
float x2250, x2251;
x2250 = lrn_rate_1;
x2251 = momentum;
JCudaTensor x2252, x2253;
x2252 = x2232;
x2253 = x531;
x539.backward_filter(x2252,x2253, x2249, x2250, x2251);

// Dealloc(X1922)
JCudaTensor x2254;
x2254 = x2232;
x2254.free();

// V_4b_a_bn_bias <~~ X136944
float x2256, x2257;
x2256 = lrn_rate_1;
x2257 = momentum;
JCudaTensor x2258;
x2258 = x2237;
x2255.update(x2258, x2256, x2257);


// Dealloc(X136944)
JCudaTensor x2259;
x2259 = x2237;
x2259.free();

// V_4b_a_bn_scale <~~ X138190
float x2261, x2262;
x2261 = lrn_rate_1;
x2262 = momentum;
JCudaTensor x2263;
x2263 = x2241;
x2260.update(x2263, x2261, x2262);


// Dealloc(X138190)
JCudaTensor x2264;
x2264 = x2241;
x2264.free();

// 4b_a_cv_W <~~ V_4b_a_cv_W
float x2265, x2266;
x2265 = 1;
x2266 = decay_1;
JCudaTensor x2267;
x2267 = x2249;
x537.update(x2267, x2265, x2266);


// 4b_a_bn_bias <~~ V_4b_a_bn_bias
float x2268, x2269;
x2268 = 1;
x2269 = decay_1;
JCudaTensor x2270;
x2270 = x2255;
x545.update(x2270, x2268, x2269);


// 4b_a_bn_scale <~~ V_4b_a_bn_scale
float x2271, x2272;
x2271 = 1;
x2272 = decay_1;
JCudaTensor x2273;
x2273 = x2260;
x544.update(x2273, x2271, x2272);


// val X1924 = (X1923 + X1899)
JCudaTensor x2274;
JCudaTensor x2275, x2276;
x2275 = x2246;
x2276 = x2132;
x2274 = x2275.plus_i(x2276);

// Dealloc(X1899)
JCudaTensor x2277;
x2277 = x2132;
x2277.free();

// val X1939 = X1924 * d_ReLU()(X299)/d_X298
JCudaTensor x2278;
JCudaTensor x2279, x2280;
x2279 = x2274;
x2280 = x531;
x2278 = x524.backward(x2279,x2280);

// Dealloc(X299)
JCudaTensor x2281;
x2281 = x531;
x2281.free();

// val X1943 = X1939.copy * d_ReLU()(X288)/d_X287
JCudaTensor x2282;
JCudaTensor x2283, x2284;
x2283 = x2278;
x2283 = x2283.clone();
x2284 = x522;
x2282 = x524.backward(x2283,x2284);

// Dealloc(X288)
JCudaTensor x2285;
x2285 = x522;
x2285.free();

// val X1955 = X1939.copy * d_ReLU()(X297)/d_X296
JCudaTensor x2286;
JCudaTensor x2287, x2288;
x2287 = x2278;
x2287 = x2287.clone();
x2288 = x525;
x2286 = x527.backward(x2287,x2288);

// Dealloc(X1939)
JCudaTensor x2289;
x2289 = x2278;
x2289.free();

// Dealloc(X297)
JCudaTensor x2290;
x2290 = x525;
x2290.free();

// val X122682 = X1943 * d_BatchNorm()(X286,4a1_bn_scale)/d_4a1_bn_scale
JCudaTensor x2291;
JCudaTensor x2292, x2293, x2294;
x2292 = x2282;
x2293 = x460;
x2294 = x478;
JCudaTensor[] x2295 = x480.backward(x2292,x2293,x2294);
x2291 = x2295[1];

// val X134406 = X1955 * d_BatchNorm()(X295,4a2_c_bn_scale)/d_4a2_c_bn_scale
JCudaTensor x2296;
JCudaTensor x2297, x2298, x2299;
x2297 = x2286;
x2298 = x508;
x2299 = x519;
JCudaTensor[] x2300 = x521.backward(x2297,x2298,x2299);
x2296 = x2300[1];

// val X1956 = X1955 * d_BatchNorm()(X295,4a2_c_bn_scale)/d_X295
JCudaTensor x2301;
x2301 = x2300[0];

// val X1944 = X1943 * d_BatchNorm()(X286,4a1_bn_scale)/d_X286
JCudaTensor x2305;
x2305 = x2295[0];

// val X121401 = X1943 * d_BatchNorm()(X286,4a1_bn_scale)/d_4a1_bn_bias
JCudaTensor x2309;
x2309 = x2295[2];

// Dealloc(X286)
JCudaTensor x2313;
x2313 = x460;
x2313.free();

// val X133116 = X1955 * d_BatchNorm()(X295,4a2_c_bn_scale)/d_4a2_c_bn_bias
JCudaTensor x2314;
x2314 = x2300[2];

// Dealloc(X295)
JCudaTensor x2318;
x2318 = x508;
x2318.free();

// V_4a1_bn_bias <~~ X121401
float x2320, x2321;
x2320 = lrn_rate_1;
x2321 = momentum;
JCudaTensor x2322;
x2322 = x2309;
x2319.update(x2322, x2320, x2321);


// Dealloc(X121401)
JCudaTensor x2323;
x2323 = x2309;
x2323.free();

// V_4a1_bn_scale <~~ X122682
float x2325, x2326;
x2325 = lrn_rate_1;
x2326 = momentum;
JCudaTensor x2327;
x2327 = x2291;
x2324.update(x2327, x2325, x2326);


// Dealloc(X122682)
JCudaTensor x2328;
x2328 = x2291;
x2328.free();

// V_4a2_c_bn_scale <~~ X134406
float x2330, x2331;
x2330 = lrn_rate_1;
x2331 = momentum;
JCudaTensor x2332;
x2332 = x2296;
x2329.update(x2332, x2330, x2331);


// Dealloc(X134406)
JCudaTensor x2333;
x2333 = x2296;
x2333.free();

// val X1957 = X1956 * d_Convolv(1,0)(4a2_c_cv_W)/d_X294
JCudaTensor x2334;
JCudaTensor x2335, x2336;
x2335 = x2301;
x2336 = x512;
x2334 = x514.backward_data(x2335,x2336);

// V_4a2_c_cv_W <~~ X1956 * d_Convolv(1,0)(X294)/d_4a2_c_cv_W
float x2338, x2339;
x2338 = lrn_rate_1;
x2339 = momentum;
JCudaTensor x2340, x2341;
x2340 = x2301;
x2341 = x505;
x514.backward_filter(x2340,x2341, x2337, x2338, x2339);

// Dealloc(X1956)
JCudaTensor x2342;
x2342 = x2301;
x2342.free();

// V_4a2_c_bn_bias <~~ X133116
float x2344, x2345;
x2344 = lrn_rate_1;
x2345 = momentum;
JCudaTensor x2346;
x2346 = x2314;
x2343.update(x2346, x2344, x2345);


// Dealloc(X133116)
JCudaTensor x2347;
x2347 = x2314;
x2347.free();

// V_4a1_cv_W <~~ X1944 * d_Convolv(2,0)(X285)/d_4a1_cv_W
float x2349, x2350;
x2349 = lrn_rate_1;
x2350 = momentum;
JCudaTensor x2351, x2352;
x2351 = x2305;
x2352 = x458;
x466.backward_filter(x2351,x2352, x2348, x2349, x2350);

// val X1945 = X1944 * d_Convolv(2,0)(4a1_cv_W)/d_X285
JCudaTensor x2353;
JCudaTensor x2354, x2355;
x2354 = x2305;
x2355 = x464;
x2353 = x466.backward_data(x2354,x2355);

// Dealloc(X1944)
JCudaTensor x2356;
x2356 = x2305;
x2356.free();

// 4a1_bn_scale <~~ V_4a1_bn_scale
float x2357, x2358;
x2357 = 1;
x2358 = decay_1;
JCudaTensor x2359;
x2359 = x2324;
x478.update(x2359, x2357, x2358);


// 4a2_c_bn_scale <~~ V_4a2_c_bn_scale
float x2360, x2361;
x2360 = 1;
x2361 = decay_1;
JCudaTensor x2362;
x2362 = x2329;
x519.update(x2362, x2360, x2361);


// 4a1_bn_bias <~~ V_4a1_bn_bias
float x2363, x2364;
x2363 = 1;
x2364 = decay_1;
JCudaTensor x2365;
x2365 = x2319;
x479.update(x2365, x2363, x2364);


// 4a1_cv_W <~~ V_4a1_cv_W
float x2366, x2367;
x2366 = 1;
x2367 = decay_1;
JCudaTensor x2368;
x2368 = x2348;
x464.update(x2368, x2366, x2367);


// 4a2_c_bn_bias <~~ V_4a2_c_bn_bias
float x2369, x2370;
x2369 = 1;
x2370 = decay_1;
JCudaTensor x2371;
x2371 = x2343;
x520.update(x2371, x2369, x2370);


// 4a2_c_cv_W <~~ V_4a2_c_cv_W
float x2372, x2373;
x2372 = 1;
x2373 = decay_1;
JCudaTensor x2374;
x2374 = x2337;
x512.update(x2374, x2372, x2373);


// val X1961 = X1957 * d_ReLU()(X294)/d_X293
JCudaTensor x2375;
JCudaTensor x2376, x2377;
x2376 = x2334;
x2377 = x505;
x2375 = x507.backward(x2376,x2377);

// Dealloc(X294)
JCudaTensor x2378;
x2378 = x505;
x2378.free();

// val X130521 = X1961 * d_BatchNorm()(X292,4a2_b_bn_scale)/d_4a2_b_bn_scale
JCudaTensor x2379;
JCudaTensor x2380, x2381, x2382;
x2380 = x2375;
x2381 = x491;
x2382 = x502;
JCudaTensor[] x2383 = x504.backward(x2380,x2381,x2382);
x2379 = x2383[1];

// val X1962 = X1961 * d_BatchNorm()(X292,4a2_b_bn_scale)/d_X292
JCudaTensor x2384;
x2384 = x2383[0];

// val X129218 = X1961 * d_BatchNorm()(X292,4a2_b_bn_scale)/d_4a2_b_bn_bias
JCudaTensor x2388;
x2388 = x2383[2];

// Dealloc(X292)
JCudaTensor x2392;
x2392 = x491;
x2392.free();

// val X1963 = X1962 * d_Convolv(1,1)(4a2_b_cv_W)/d_X291
JCudaTensor x2393;
JCudaTensor x2394, x2395;
x2394 = x2384;
x2395 = x495;
x2393 = x497.backward_data(x2394,x2395);

// V_4a2_b_cv_W <~~ X1962 * d_Convolv(1,1)(X291)/d_4a2_b_cv_W
float x2397, x2398;
x2397 = lrn_rate_1;
x2398 = momentum;
JCudaTensor x2399, x2400;
x2399 = x2384;
x2400 = x488;
x497.backward_filter(x2399,x2400, x2396, x2397, x2398);

// Dealloc(X1962)
JCudaTensor x2401;
x2401 = x2384;
x2401.free();

// V_4a2_b_bn_scale <~~ X130521
float x2403, x2404;
x2403 = lrn_rate_1;
x2404 = momentum;
JCudaTensor x2405;
x2405 = x2379;
x2402.update(x2405, x2403, x2404);


// Dealloc(X130521)
JCudaTensor x2406;
x2406 = x2379;
x2406.free();

// V_4a2_b_bn_bias <~~ X129218
float x2408, x2409;
x2408 = lrn_rate_1;
x2409 = momentum;
JCudaTensor x2410;
x2410 = x2388;
x2407.update(x2410, x2408, x2409);


// Dealloc(X129218)
JCudaTensor x2411;
x2411 = x2388;
x2411.free();

// 4a2_b_cv_W <~~ V_4a2_b_cv_W
float x2412, x2413;
x2412 = 1;
x2413 = decay_1;
JCudaTensor x2414;
x2414 = x2396;
x495.update(x2414, x2412, x2413);


// 4a2_b_bn_scale <~~ V_4a2_b_bn_scale
float x2415, x2416;
x2415 = 1;
x2416 = decay_1;
JCudaTensor x2417;
x2417 = x2402;
x502.update(x2417, x2415, x2416);


// 4a2_b_bn_bias <~~ V_4a2_b_bn_bias
float x2418, x2419;
x2418 = 1;
x2419 = decay_1;
JCudaTensor x2420;
x2420 = x2407;
x503.update(x2420, x2418, x2419);


// val X1967 = X1963 * d_ReLU()(X291)/d_X290
JCudaTensor x2421;
JCudaTensor x2422, x2423;
x2422 = x2393;
x2423 = x488;
x2421 = x490.backward(x2422,x2423);

// Dealloc(X291)
JCudaTensor x2424;
x2424 = x488;
x2424.free();

// val X125281 = X1967 * d_BatchNorm()(X289,4a2_a_bn_scale)/d_4a2_a_bn_bias
JCudaTensor x2425;
JCudaTensor x2426, x2427, x2428;
x2426 = x2421;
x2427 = x467;
x2428 = x485;
JCudaTensor[] x2429 = x487.backward(x2426,x2427,x2428);
x2425 = x2429[2];

// val X1968 = X1967 * d_BatchNorm()(X289,4a2_a_bn_scale)/d_X289
JCudaTensor x2430;
x2430 = x2429[0];

// val X126597 = X1967 * d_BatchNorm()(X289,4a2_a_bn_scale)/d_4a2_a_bn_scale
JCudaTensor x2434;
x2434 = x2429[1];

// Dealloc(X289)
JCudaTensor x2438;
x2438 = x467;
x2438.free();

// val X1970 = (X1945 + X1968 * d_Convolv(2,0)(4a2_a_cv_W)/d_X285)
JCudaTensor x2439;
JCudaTensor x2440;
x2440 = x2353;
JCudaTensor x2441, x2442;
x2441 = x2430;
x2442 = x471;
x2439 = x473.backward_data(x2441,x2442, x2440);

// V_4a2_a_cv_W <~~ X1968 * d_Convolv(2,0)(X285)/d_4a2_a_cv_W
float x2444, x2445;
x2444 = lrn_rate_1;
x2445 = momentum;
JCudaTensor x2446, x2447;
x2446 = x2430;
x2447 = x458;
x473.backward_filter(x2446,x2447, x2443, x2444, x2445);

// Dealloc(X1968)
JCudaTensor x2448;
x2448 = x2430;
x2448.free();

// V_4a2_a_bn_bias <~~ X125281
float x2450, x2451;
x2450 = lrn_rate_1;
x2451 = momentum;
JCudaTensor x2452;
x2452 = x2425;
x2449.update(x2452, x2450, x2451);


// Dealloc(X125281)
JCudaTensor x2453;
x2453 = x2425;
x2453.free();

// V_4a2_a_bn_scale <~~ X126597
float x2455, x2456;
x2455 = lrn_rate_1;
x2456 = momentum;
JCudaTensor x2457;
x2457 = x2434;
x2454.update(x2457, x2455, x2456);


// Dealloc(X126597)
JCudaTensor x2458;
x2458 = x2434;
x2458.free();

// 4a2_a_cv_W <~~ V_4a2_a_cv_W
float x2459, x2460;
x2459 = 1;
x2460 = decay_1;
JCudaTensor x2461;
x2461 = x2443;
x471.update(x2461, x2459, x2460);


// 4a2_a_bn_bias <~~ V_4a2_a_bn_bias
float x2462, x2463;
x2462 = 1;
x2463 = decay_1;
JCudaTensor x2464;
x2464 = x2449;
x486.update(x2464, x2462, x2463);


// 4a2_a_bn_scale <~~ V_4a2_a_bn_scale
float x2465, x2466;
x2465 = 1;
x2466 = decay_1;
JCudaTensor x2467;
x2467 = x2454;
x485.update(x2467, x2465, x2466);


// val X2018 = X1970 * d_ReLU()(X285)/d_X284
JCudaTensor x2468;
JCudaTensor x2469, x2470;
x2469 = x2439;
x2470 = x458;
x2468 = x454.backward(x2469,x2470);

// Dealloc(X285)
JCudaTensor x2471;
x2471 = x458;
x2471.free();

// val X2028 = X2018.copy * d_ReLU()(X283)/d_X282
JCudaTensor x2472;
JCudaTensor x2473, x2474;
x2473 = x2468;
x2473 = x2473.clone();
x2474 = x452;
x2472 = x454.backward(x2473,x2474);

// Dealloc(X283)
JCudaTensor x2475;
x2475 = x452;
x2475.free();

// val X118708 = X2028 * d_BatchNorm()(X281,3d_c_bn_scale)/d_3d_c_bn_scale
JCudaTensor x2476;
JCudaTensor x2477, x2478, x2479;
x2477 = x2472;
x2478 = x438;
x2479 = x449;
JCudaTensor[] x2480 = x451.backward(x2477,x2478,x2479);
x2476 = x2480[1];

// val X117298 = X2028 * d_BatchNorm()(X281,3d_c_bn_scale)/d_3d_c_bn_bias
JCudaTensor x2481;
x2481 = x2480[2];

// val X2029 = X2028 * d_BatchNorm()(X281,3d_c_bn_scale)/d_X281
JCudaTensor x2485;
x2485 = x2480[0];

// Dealloc(X281)
JCudaTensor x2489;
x2489 = x438;
x2489.free();

// val X2030 = X2029 * d_Convolv(1,0)(3d_c_cv_W)/d_X280
JCudaTensor x2490;
JCudaTensor x2491, x2492;
x2491 = x2485;
x2492 = x442;
x2490 = x444.backward_data(x2491,x2492);

// V_3d_c_cv_W <~~ X2029 * d_Convolv(1,0)(X280)/d_3d_c_cv_W
float x2494, x2495;
x2494 = lrn_rate_1;
x2495 = momentum;
JCudaTensor x2496, x2497;
x2496 = x2485;
x2497 = x435;
x444.backward_filter(x2496,x2497, x2493, x2494, x2495);

// Dealloc(X2029)
JCudaTensor x2498;
x2498 = x2485;
x2498.free();

// V_3d_c_bn_bias <~~ X117298
float x2500, x2501;
x2500 = lrn_rate_1;
x2501 = momentum;
JCudaTensor x2502;
x2502 = x2481;
x2499.update(x2502, x2500, x2501);


// Dealloc(X117298)
JCudaTensor x2503;
x2503 = x2481;
x2503.free();

// V_3d_c_bn_scale <~~ X118708
float x2505, x2506;
x2505 = lrn_rate_1;
x2506 = momentum;
JCudaTensor x2507;
x2507 = x2476;
x2504.update(x2507, x2505, x2506);


// Dealloc(X118708)
JCudaTensor x2508;
x2508 = x2476;
x2508.free();

// 3d_c_cv_W <~~ V_3d_c_cv_W
float x2509, x2510;
x2509 = 1;
x2510 = decay_1;
JCudaTensor x2511;
x2511 = x2493;
x442.update(x2511, x2509, x2510);


// 3d_c_bn_bias <~~ V_3d_c_bn_bias
float x2512, x2513;
x2512 = 1;
x2513 = decay_1;
JCudaTensor x2514;
x2514 = x2499;
x450.update(x2514, x2512, x2513);


// 3d_c_bn_scale <~~ V_3d_c_bn_scale
float x2515, x2516;
x2515 = 1;
x2516 = decay_1;
JCudaTensor x2517;
x2517 = x2504;
x449.update(x2517, x2515, x2516);


// val X2034 = X2030 * d_ReLU()(X280)/d_X279
JCudaTensor x2518;
JCudaTensor x2519, x2520;
x2519 = x2490;
x2520 = x435;
x2518 = x437.backward(x2519,x2520);

// Dealloc(X280)
JCudaTensor x2521;
x2521 = x435;
x2521.free();

// val X114463 = X2034 * d_BatchNorm()(X278,3d_b_bn_scale)/d_3d_b_bn_scale
JCudaTensor x2522;
JCudaTensor x2523, x2524, x2525;
x2523 = x2518;
x2524 = x421;
x2525 = x432;
JCudaTensor[] x2526 = x434.backward(x2523,x2524,x2525);
x2522 = x2526[1];

// val X2035 = X2034 * d_BatchNorm()(X278,3d_b_bn_scale)/d_X278
JCudaTensor x2527;
x2527 = x2526[0];

// val X113040 = X2034 * d_BatchNorm()(X278,3d_b_bn_scale)/d_3d_b_bn_bias
JCudaTensor x2531;
x2531 = x2526[2];

// Dealloc(X278)
JCudaTensor x2535;
x2535 = x421;
x2535.free();

// val X2036 = X2035 * d_Convolv(1,1)(3d_b_cv_W)/d_X277
JCudaTensor x2536;
JCudaTensor x2537, x2538;
x2537 = x2527;
x2538 = x425;
x2536 = x427.backward_data(x2537,x2538);

// V_3d_b_bn_bias <~~ X113040
float x2540, x2541;
x2540 = lrn_rate_1;
x2541 = momentum;
JCudaTensor x2542;
x2542 = x2531;
x2539.update(x2542, x2540, x2541);


// Dealloc(X113040)
JCudaTensor x2543;
x2543 = x2531;
x2543.free();

// V_3d_b_bn_scale <~~ X114463
float x2545, x2546;
x2545 = lrn_rate_1;
x2546 = momentum;
JCudaTensor x2547;
x2547 = x2522;
x2544.update(x2547, x2545, x2546);


// Dealloc(X114463)
JCudaTensor x2548;
x2548 = x2522;
x2548.free();

// V_3d_b_cv_W <~~ X2035 * d_Convolv(1,1)(X277)/d_3d_b_cv_W
float x2550, x2551;
x2550 = lrn_rate_1;
x2551 = momentum;
JCudaTensor x2552, x2553;
x2552 = x2527;
x2553 = x418;
x427.backward_filter(x2552,x2553, x2549, x2550, x2551);

// Dealloc(X2035)
JCudaTensor x2554;
x2554 = x2527;
x2554.free();

// 3d_b_bn_bias <~~ V_3d_b_bn_bias
float x2555, x2556;
x2555 = 1;
x2556 = decay_1;
JCudaTensor x2557;
x2557 = x2539;
x433.update(x2557, x2555, x2556);


// 3d_b_bn_scale <~~ V_3d_b_bn_scale
float x2558, x2559;
x2558 = 1;
x2559 = decay_1;
JCudaTensor x2560;
x2560 = x2544;
x432.update(x2560, x2558, x2559);


// 3d_b_cv_W <~~ V_3d_b_cv_W
float x2561, x2562;
x2561 = 1;
x2562 = decay_1;
JCudaTensor x2563;
x2563 = x2549;
x425.update(x2563, x2561, x2562);


// val X2040 = X2036 * d_ReLU()(X277)/d_X276
JCudaTensor x2564;
JCudaTensor x2565, x2566;
x2565 = x2536;
x2566 = x418;
x2564 = x420.backward(x2565,x2566);

// Dealloc(X277)
JCudaTensor x2567;
x2567 = x418;
x2567.free();

// val X110179 = X2040 * d_BatchNorm()(X275,3d_a_bn_scale)/d_3d_a_bn_scale
JCudaTensor x2568;
JCudaTensor x2569, x2570, x2571;
x2569 = x2564;
x2570 = x404;
x2571 = x415;
JCudaTensor[] x2572 = x417.backward(x2569,x2570,x2571);
x2568 = x2572[1];

// val X108743 = X2040 * d_BatchNorm()(X275,3d_a_bn_scale)/d_3d_a_bn_bias
JCudaTensor x2573;
x2573 = x2572[2];

// val X2041 = X2040 * d_BatchNorm()(X275,3d_a_bn_scale)/d_X275
JCudaTensor x2577;
x2577 = x2572[0];

// Dealloc(X275)
JCudaTensor x2581;
x2581 = x404;
x2581.free();

// V_3d_a_bn_scale <~~ X110179
float x2583, x2584;
x2583 = lrn_rate_1;
x2584 = momentum;
JCudaTensor x2585;
x2585 = x2568;
x2582.update(x2585, x2583, x2584);


// Dealloc(X110179)
JCudaTensor x2586;
x2586 = x2568;
x2586.free();

// val X2042 = X2041 * d_Convolv(1,0)(3d_a_cv_W)/d_X274
JCudaTensor x2587;
JCudaTensor x2588, x2589;
x2588 = x2577;
x2589 = x408;
x2587 = x410.backward_data(x2588,x2589);

// V_3d_a_bn_bias <~~ X108743
float x2591, x2592;
x2591 = lrn_rate_1;
x2592 = momentum;
JCudaTensor x2593;
x2593 = x2573;
x2590.update(x2593, x2591, x2592);


// Dealloc(X108743)
JCudaTensor x2594;
x2594 = x2573;
x2594.free();

// V_3d_a_cv_W <~~ X2041 * d_Convolv(1,0)(X274)/d_3d_a_cv_W
float x2596, x2597;
x2596 = lrn_rate_1;
x2597 = momentum;
JCudaTensor x2598, x2599;
x2598 = x2577;
x2599 = x402;
x410.backward_filter(x2598,x2599, x2595, x2596, x2597);

// Dealloc(X2041)
JCudaTensor x2600;
x2600 = x2577;
x2600.free();

// 3d_a_bn_scale <~~ V_3d_a_bn_scale
float x2601, x2602;
x2601 = 1;
x2602 = decay_1;
JCudaTensor x2603;
x2603 = x2582;
x415.update(x2603, x2601, x2602);


// 3d_a_bn_bias <~~ V_3d_a_bn_bias
float x2604, x2605;
x2604 = 1;
x2605 = decay_1;
JCudaTensor x2606;
x2606 = x2590;
x416.update(x2606, x2604, x2605);


// 3d_a_cv_W <~~ V_3d_a_cv_W
float x2607, x2608;
x2607 = 1;
x2608 = decay_1;
JCudaTensor x2609;
x2609 = x2595;
x408.update(x2609, x2607, x2608);


// val X2043 = (X2042 + X2018)
JCudaTensor x2610;
JCudaTensor x2611, x2612;
x2611 = x2587;
x2612 = x2468;
x2610 = x2611.plus_i(x2612);

// Dealloc(X2018)
JCudaTensor x2613;
x2613 = x2468;
x2613.free();

// val X2055 = X2043 * d_ReLU()(X274)/d_X273
JCudaTensor x2614;
JCudaTensor x2615, x2616;
x2615 = x2610;
x2616 = x402;
x2614 = x398.backward(x2615,x2616);

// Dealloc(X274)
JCudaTensor x2617;
x2617 = x402;
x2617.free();

// val X2065 = X2055.copy * d_ReLU()(X272)/d_X271
JCudaTensor x2618;
JCudaTensor x2619, x2620;
x2619 = x2614;
x2619 = x2619.clone();
x2620 = x396;
x2618 = x398.backward(x2619,x2620);

// Dealloc(X272)
JCudaTensor x2621;
x2621 = x396;
x2621.free();

// val X104351 = X2065 * d_BatchNorm()(X270,3c_c_bn_scale)/d_3c_c_bn_bias
JCudaTensor x2622;
JCudaTensor x2623, x2624, x2625;
x2623 = x2618;
x2624 = x382;
x2625 = x393;
JCudaTensor[] x2626 = x395.backward(x2623,x2624,x2625);
x2622 = x2626[2];

// val X2066 = X2065 * d_BatchNorm()(X270,3c_c_bn_scale)/d_X270
JCudaTensor x2627;
x2627 = x2626[0];

// val X105828 = X2065 * d_BatchNorm()(X270,3c_c_bn_scale)/d_3c_c_bn_scale
JCudaTensor x2631;
x2631 = x2626[1];

// Dealloc(X270)
JCudaTensor x2635;
x2635 = x382;
x2635.free();

// V_3c_c_bn_bias <~~ X104351
float x2637, x2638;
x2637 = lrn_rate_1;
x2638 = momentum;
JCudaTensor x2639;
x2639 = x2622;
x2636.update(x2639, x2637, x2638);


// Dealloc(X104351)
JCudaTensor x2640;
x2640 = x2622;
x2640.free();

// val X2067 = X2066 * d_Convolv(1,0)(3c_c_cv_W)/d_X269
JCudaTensor x2641;
JCudaTensor x2642, x2643;
x2642 = x2627;
x2643 = x386;
x2641 = x388.backward_data(x2642,x2643);

// V_3c_c_cv_W <~~ X2066 * d_Convolv(1,0)(X269)/d_3c_c_cv_W
float x2645, x2646;
x2645 = lrn_rate_1;
x2646 = momentum;
JCudaTensor x2647, x2648;
x2647 = x2627;
x2648 = x379;
x388.backward_filter(x2647,x2648, x2644, x2645, x2646);

// Dealloc(X2066)
JCudaTensor x2649;
x2649 = x2627;
x2649.free();

// V_3c_c_bn_scale <~~ X105828
float x2651, x2652;
x2651 = lrn_rate_1;
x2652 = momentum;
JCudaTensor x2653;
x2653 = x2631;
x2650.update(x2653, x2651, x2652);


// Dealloc(X105828)
JCudaTensor x2654;
x2654 = x2631;
x2654.free();

// 3c_c_bn_bias <~~ V_3c_c_bn_bias
float x2655, x2656;
x2655 = 1;
x2656 = decay_1;
JCudaTensor x2657;
x2657 = x2636;
x394.update(x2657, x2655, x2656);


// 3c_c_cv_W <~~ V_3c_c_cv_W
float x2658, x2659;
x2658 = 1;
x2659 = decay_1;
JCudaTensor x2660;
x2660 = x2644;
x386.update(x2660, x2658, x2659);


// 3c_c_bn_scale <~~ V_3c_c_bn_scale
float x2661, x2662;
x2661 = 1;
x2662 = decay_1;
JCudaTensor x2663;
x2663 = x2650;
x393.update(x2663, x2661, x2662);


// val X2071 = X2067 * d_ReLU()(X269)/d_X268
JCudaTensor x2664;
JCudaTensor x2665, x2666;
x2665 = x2641;
x2666 = x379;
x2664 = x381.backward(x2665,x2666);

// Dealloc(X269)
JCudaTensor x2667;
x2667 = x379;
x2667.free();

// val X2072 = X2071 * d_BatchNorm()(X267,3c_b_bn_scale)/d_X267
JCudaTensor x2668;
JCudaTensor x2669, x2670, x2671;
x2669 = x2664;
x2670 = x365;
x2671 = x376;
JCudaTensor[] x2672 = x378.backward(x2669,x2670,x2671);
x2668 = x2672[0];

// val X99892 = X2071 * d_BatchNorm()(X267,3c_b_bn_scale)/d_3c_b_bn_bias
JCudaTensor x2673;
x2673 = x2672[2];

// val X101382 = X2071 * d_BatchNorm()(X267,3c_b_bn_scale)/d_3c_b_bn_scale
JCudaTensor x2677;
x2677 = x2672[1];

// Dealloc(X267)
JCudaTensor x2681;
x2681 = x365;
x2681.free();

// V_3c_b_bn_scale <~~ X101382
float x2683, x2684;
x2683 = lrn_rate_1;
x2684 = momentum;
JCudaTensor x2685;
x2685 = x2677;
x2682.update(x2685, x2683, x2684);


// Dealloc(X101382)
JCudaTensor x2686;
x2686 = x2677;
x2686.free();

// val X2073 = X2072 * d_Convolv(1,1)(3c_b_cv_W)/d_X266
JCudaTensor x2687;
JCudaTensor x2688, x2689;
x2688 = x2668;
x2689 = x369;
x2687 = x371.backward_data(x2688,x2689);

// V_3c_b_cv_W <~~ X2072 * d_Convolv(1,1)(X266)/d_3c_b_cv_W
float x2691, x2692;
x2691 = lrn_rate_1;
x2692 = momentum;
JCudaTensor x2693, x2694;
x2693 = x2668;
x2694 = x362;
x371.backward_filter(x2693,x2694, x2690, x2691, x2692);

// Dealloc(X2072)
JCudaTensor x2695;
x2695 = x2668;
x2695.free();

// V_3c_b_bn_bias <~~ X99892
float x2697, x2698;
x2697 = lrn_rate_1;
x2698 = momentum;
JCudaTensor x2699;
x2699 = x2673;
x2696.update(x2699, x2697, x2698);


// Dealloc(X99892)
JCudaTensor x2700;
x2700 = x2673;
x2700.free();

// 3c_b_bn_scale <~~ V_3c_b_bn_scale
float x2701, x2702;
x2701 = 1;
x2702 = decay_1;
JCudaTensor x2703;
x2703 = x2682;
x376.update(x2703, x2701, x2702);


// 3c_b_cv_W <~~ V_3c_b_cv_W
float x2704, x2705;
x2704 = 1;
x2705 = decay_1;
JCudaTensor x2706;
x2706 = x2690;
x369.update(x2706, x2704, x2705);


// 3c_b_bn_bias <~~ V_3c_b_bn_bias
float x2707, x2708;
x2707 = 1;
x2708 = decay_1;
JCudaTensor x2709;
x2709 = x2696;
x377.update(x2709, x2707, x2708);


// val X2077 = X2073 * d_ReLU()(X266)/d_X265
JCudaTensor x2710;
JCudaTensor x2711, x2712;
x2711 = x2687;
x2712 = x362;
x2710 = x364.backward(x2711,x2712);

// Dealloc(X266)
JCudaTensor x2713;
x2713 = x362;
x2713.free();

// val X96897 = X2077 * d_BatchNorm()(X264,3c_a_bn_scale)/d_3c_a_bn_scale
JCudaTensor x2714;
JCudaTensor x2715, x2716, x2717;
x2715 = x2710;
x2716 = x348;
x2717 = x359;
JCudaTensor[] x2718 = x361.backward(x2715,x2716,x2717);
x2714 = x2718[1];

// val X2078 = X2077 * d_BatchNorm()(X264,3c_a_bn_scale)/d_X264
JCudaTensor x2719;
x2719 = x2718[0];

// val X95394 = X2077 * d_BatchNorm()(X264,3c_a_bn_scale)/d_3c_a_bn_bias
JCudaTensor x2723;
x2723 = x2718[2];

// Dealloc(X264)
JCudaTensor x2727;
x2727 = x348;
x2727.free();

// val X2079 = X2078 * d_Convolv(1,0)(3c_a_cv_W)/d_X263
JCudaTensor x2728;
JCudaTensor x2729, x2730;
x2729 = x2719;
x2730 = x352;
x2728 = x354.backward_data(x2729,x2730);

// V_3c_a_cv_W <~~ X2078 * d_Convolv(1,0)(X263)/d_3c_a_cv_W
float x2732, x2733;
x2732 = lrn_rate_1;
x2733 = momentum;
JCudaTensor x2734, x2735;
x2734 = x2719;
x2735 = x346;
x354.backward_filter(x2734,x2735, x2731, x2732, x2733);

// Dealloc(X2078)
JCudaTensor x2736;
x2736 = x2719;
x2736.free();

// V_3c_a_bn_bias <~~ X95394
float x2738, x2739;
x2738 = lrn_rate_1;
x2739 = momentum;
JCudaTensor x2740;
x2740 = x2723;
x2737.update(x2740, x2738, x2739);


// Dealloc(X95394)
JCudaTensor x2741;
x2741 = x2723;
x2741.free();

// V_3c_a_bn_scale <~~ X96897
float x2743, x2744;
x2743 = lrn_rate_1;
x2744 = momentum;
JCudaTensor x2745;
x2745 = x2714;
x2742.update(x2745, x2743, x2744);


// Dealloc(X96897)
JCudaTensor x2746;
x2746 = x2714;
x2746.free();

// 3c_a_cv_W <~~ V_3c_a_cv_W
float x2747, x2748;
x2747 = 1;
x2748 = decay_1;
JCudaTensor x2749;
x2749 = x2731;
x352.update(x2749, x2747, x2748);


// 3c_a_bn_bias <~~ V_3c_a_bn_bias
float x2750, x2751;
x2750 = 1;
x2751 = decay_1;
JCudaTensor x2752;
x2752 = x2737;
x360.update(x2752, x2750, x2751);


// 3c_a_bn_scale <~~ V_3c_a_bn_scale
float x2753, x2754;
x2753 = 1;
x2754 = decay_1;
JCudaTensor x2755;
x2755 = x2742;
x359.update(x2755, x2753, x2754);


// val X2080 = (X2079 + X2055)
JCudaTensor x2756;
JCudaTensor x2757, x2758;
x2757 = x2728;
x2758 = x2614;
x2756 = x2757.plus_i(x2758);

// Dealloc(X2055)
JCudaTensor x2759;
x2759 = x2614;
x2759.free();

// val X2092 = X2080 * d_ReLU()(X263)/d_X262
JCudaTensor x2760;
JCudaTensor x2761, x2762;
x2761 = x2756;
x2762 = x346;
x2760 = x342.backward(x2761,x2762);

// Dealloc(X263)
JCudaTensor x2763;
x2763 = x346;
x2763.free();

// val X2102 = X2092.copy * d_ReLU()(X261)/d_X260
JCudaTensor x2764;
JCudaTensor x2765, x2766;
x2765 = x2760;
x2765 = x2765.clone();
x2766 = x340;
x2764 = x342.backward(x2765,x2766);

// Dealloc(X261)
JCudaTensor x2767;
x2767 = x340;
x2767.free();

// val X2103 = X2102 * d_BatchNorm()(X259,3b_c_bn_scale)/d_X259
JCudaTensor x2768;
JCudaTensor x2769, x2770, x2771;
x2769 = x2764;
x2770 = x326;
x2771 = x337;
JCudaTensor[] x2772 = x339.backward(x2769,x2770,x2771);
x2768 = x2772[0];

// val X90801 = X2102 * d_BatchNorm()(X259,3b_c_bn_scale)/d_3b_c_bn_bias
JCudaTensor x2773;
x2773 = x2772[2];

// val X92345 = X2102 * d_BatchNorm()(X259,3b_c_bn_scale)/d_3b_c_bn_scale
JCudaTensor x2777;
x2777 = x2772[1];

// Dealloc(X259)
JCudaTensor x2781;
x2781 = x326;
x2781.free();

// V_3b_c_bn_bias <~~ X90801
float x2783, x2784;
x2783 = lrn_rate_1;
x2784 = momentum;
JCudaTensor x2785;
x2785 = x2773;
x2782.update(x2785, x2783, x2784);


// Dealloc(X90801)
JCudaTensor x2786;
x2786 = x2773;
x2786.free();

// V_3b_c_bn_scale <~~ X92345
float x2788, x2789;
x2788 = lrn_rate_1;
x2789 = momentum;
JCudaTensor x2790;
x2790 = x2777;
x2787.update(x2790, x2788, x2789);


// Dealloc(X92345)
JCudaTensor x2791;
x2791 = x2777;
x2791.free();

// val X2104 = X2103 * d_Convolv(1,0)(3b_c_cv_W)/d_X258
JCudaTensor x2792;
JCudaTensor x2793, x2794;
x2793 = x2768;
x2794 = x330;
x2792 = x332.backward_data(x2793,x2794);

// V_3b_c_cv_W <~~ X2103 * d_Convolv(1,0)(X258)/d_3b_c_cv_W
float x2796, x2797;
x2796 = lrn_rate_1;
x2797 = momentum;
JCudaTensor x2798, x2799;
x2798 = x2768;
x2799 = x323;
x332.backward_filter(x2798,x2799, x2795, x2796, x2797);

// Dealloc(X2103)
JCudaTensor x2800;
x2800 = x2768;
x2800.free();

// 3b_c_bn_bias <~~ V_3b_c_bn_bias
float x2801, x2802;
x2801 = 1;
x2802 = decay_1;
JCudaTensor x2803;
x2803 = x2782;
x338.update(x2803, x2801, x2802);


// 3b_c_bn_scale <~~ V_3b_c_bn_scale
float x2804, x2805;
x2804 = 1;
x2805 = decay_1;
JCudaTensor x2806;
x2806 = x2787;
x337.update(x2806, x2804, x2805);


// 3b_c_cv_W <~~ V_3b_c_cv_W
float x2807, x2808;
x2807 = 1;
x2808 = decay_1;
JCudaTensor x2809;
x2809 = x2795;
x330.update(x2809, x2807, x2808);


// val X2108 = X2104 * d_ReLU()(X258)/d_X257
JCudaTensor x2810;
JCudaTensor x2811, x2812;
x2811 = x2792;
x2812 = x323;
x2810 = x325.backward(x2811,x2812);

// Dealloc(X258)
JCudaTensor x2813;
x2813 = x323;
x2813.free();

// val X2109 = X2108 * d_BatchNorm()(X256,3b_b_bn_scale)/d_X256
JCudaTensor x2814;
JCudaTensor x2815, x2816, x2817;
x2815 = x2810;
x2816 = x309;
x2817 = x320;
JCudaTensor[] x2818 = x322.backward(x2815,x2816,x2817);
x2814 = x2818[0];

// val X87698 = X2108 * d_BatchNorm()(X256,3b_b_bn_scale)/d_3b_b_bn_scale
JCudaTensor x2819;
x2819 = x2818[1];

// val X86141 = X2108 * d_BatchNorm()(X256,3b_b_bn_scale)/d_3b_b_bn_bias
JCudaTensor x2823;
x2823 = x2818[2];

// Dealloc(X256)
JCudaTensor x2827;
x2827 = x309;
x2827.free();

// V_3b_b_bn_scale <~~ X87698
float x2829, x2830;
x2829 = lrn_rate_1;
x2830 = momentum;
JCudaTensor x2831;
x2831 = x2819;
x2828.update(x2831, x2829, x2830);


// Dealloc(X87698)
JCudaTensor x2832;
x2832 = x2819;
x2832.free();

// V_3b_b_bn_bias <~~ X86141
float x2834, x2835;
x2834 = lrn_rate_1;
x2835 = momentum;
JCudaTensor x2836;
x2836 = x2823;
x2833.update(x2836, x2834, x2835);


// Dealloc(X86141)
JCudaTensor x2837;
x2837 = x2823;
x2837.free();

// val X2110 = X2109 * d_Convolv(1,1)(3b_b_cv_W)/d_X255
JCudaTensor x2838;
JCudaTensor x2839, x2840;
x2839 = x2814;
x2840 = x313;
x2838 = x315.backward_data(x2839,x2840);

// V_3b_b_cv_W <~~ X2109 * d_Convolv(1,1)(X255)/d_3b_b_cv_W
float x2842, x2843;
x2842 = lrn_rate_1;
x2843 = momentum;
JCudaTensor x2844, x2845;
x2844 = x2814;
x2845 = x306;
x315.backward_filter(x2844,x2845, x2841, x2842, x2843);

// Dealloc(X2109)
JCudaTensor x2846;
x2846 = x2814;
x2846.free();

// 3b_b_bn_scale <~~ V_3b_b_bn_scale
float x2847, x2848;
x2847 = 1;
x2848 = decay_1;
JCudaTensor x2849;
x2849 = x2828;
x320.update(x2849, x2847, x2848);


// 3b_b_bn_bias <~~ V_3b_b_bn_bias
float x2850, x2851;
x2850 = 1;
x2851 = decay_1;
JCudaTensor x2852;
x2852 = x2833;
x321.update(x2852, x2850, x2851);


// 3b_b_cv_W <~~ V_3b_b_cv_W
float x2853, x2854;
x2853 = 1;
x2854 = decay_1;
JCudaTensor x2855;
x2855 = x2841;
x313.update(x2855, x2853, x2854);


// val X2114 = X2110 * d_ReLU()(X255)/d_X254
JCudaTensor x2856;
JCudaTensor x2857, x2858;
x2857 = x2838;
x2858 = x306;
x2856 = x308.backward(x2857,x2858);

// Dealloc(X255)
JCudaTensor x2859;
x2859 = x306;
x2859.free();

// val X2115 = X2114 * d_BatchNorm()(X253,3b_a_bn_scale)/d_X253
JCudaTensor x2860;
JCudaTensor x2861, x2862, x2863;
x2861 = x2856;
x2862 = x292;
x2863 = x303;
JCudaTensor[] x2864 = x305.backward(x2861,x2862,x2863);
x2860 = x2864[0];

// val X81442 = X2114 * d_BatchNorm()(X253,3b_a_bn_scale)/d_3b_a_bn_bias
JCudaTensor x2865;
x2865 = x2864[2];

// val X83012 = X2114 * d_BatchNorm()(X253,3b_a_bn_scale)/d_3b_a_bn_scale
JCudaTensor x2869;
x2869 = x2864[1];

// Dealloc(X253)
JCudaTensor x2873;
x2873 = x292;
x2873.free();

// V_3b_a_bn_scale <~~ X83012
float x2875, x2876;
x2875 = lrn_rate_1;
x2876 = momentum;
JCudaTensor x2877;
x2877 = x2869;
x2874.update(x2877, x2875, x2876);


// Dealloc(X83012)
JCudaTensor x2878;
x2878 = x2869;
x2878.free();

// V_3b_a_bn_bias <~~ X81442
float x2880, x2881;
x2880 = lrn_rate_1;
x2881 = momentum;
JCudaTensor x2882;
x2882 = x2865;
x2879.update(x2882, x2880, x2881);


// Dealloc(X81442)
JCudaTensor x2883;
x2883 = x2865;
x2883.free();

// val X2116 = X2115 * d_Convolv(1,0)(3b_a_cv_W)/d_X252
JCudaTensor x2884;
JCudaTensor x2885, x2886;
x2885 = x2860;
x2886 = x296;
x2884 = x298.backward_data(x2885,x2886);

// V_3b_a_cv_W <~~ X2115 * d_Convolv(1,0)(X252)/d_3b_a_cv_W
float x2888, x2889;
x2888 = lrn_rate_1;
x2889 = momentum;
JCudaTensor x2890, x2891;
x2890 = x2860;
x2891 = x290;
x298.backward_filter(x2890,x2891, x2887, x2888, x2889);

// Dealloc(X2115)
JCudaTensor x2892;
x2892 = x2860;
x2892.free();

// 3b_a_bn_scale <~~ V_3b_a_bn_scale
float x2893, x2894;
x2893 = 1;
x2894 = decay_1;
JCudaTensor x2895;
x2895 = x2874;
x303.update(x2895, x2893, x2894);


// 3b_a_bn_bias <~~ V_3b_a_bn_bias
float x2896, x2897;
x2896 = 1;
x2897 = decay_1;
JCudaTensor x2898;
x2898 = x2879;
x304.update(x2898, x2896, x2897);


// 3b_a_cv_W <~~ V_3b_a_cv_W
float x2899, x2900;
x2899 = 1;
x2900 = decay_1;
JCudaTensor x2901;
x2901 = x2887;
x296.update(x2901, x2899, x2900);


// val X2117 = (X2116 + X2092)
JCudaTensor x2902;
JCudaTensor x2903, x2904;
x2903 = x2884;
x2904 = x2760;
x2902 = x2903.plus_i(x2904);

// Dealloc(X2092)
JCudaTensor x2905;
x2905 = x2760;
x2905.free();

// val X2132 = X2117 * d_ReLU()(X252)/d_X251
JCudaTensor x2906;
JCudaTensor x2907, x2908;
x2907 = x2902;
x2908 = x290;
x2906 = x283.backward(x2907,x2908);

// Dealloc(X252)
JCudaTensor x2909;
x2909 = x290;
x2909.free();

// val X2148 = X2132.copy * d_ReLU()(X250)/d_X249
JCudaTensor x2910;
JCudaTensor x2911, x2912;
x2911 = x2906;
x2911 = x2911.clone();
x2912 = x284;
x2910 = x286.backward(x2911,x2912);

// Dealloc(X250)
JCudaTensor x2913;
x2913 = x284;
x2913.free();

// val X2136 = X2132.copy * d_ReLU()(X241)/d_X240
JCudaTensor x2914;
JCudaTensor x2915, x2916;
x2915 = x2906;
x2915 = x2915.clone();
x2916 = x281;
x2914 = x283.backward(x2915,x2916);

// Dealloc(X2132)
JCudaTensor x2917;
x2917 = x2906;
x2917.free();

// Dealloc(X241)
JCudaTensor x2918;
x2918 = x281;
x2918.free();

// val X76642 = X2148 * d_BatchNorm()(X248,3a2_c_bn_scale)/d_3a2_c_bn_bias
JCudaTensor x2919;
JCudaTensor x2920, x2921, x2922;
x2920 = x2910;
x2921 = x267;
x2922 = x278;
JCudaTensor[] x2923 = x280.backward(x2920,x2921,x2922);
x2919 = x2923[2];

// val X63616 = X2136 * d_BatchNorm()(X239,3a1_bn_scale)/d_3a1_bn_scale
JCudaTensor x2924;
JCudaTensor x2925, x2926, x2927;
x2925 = x2914;
x2926 = x226;
x2927 = x237;
JCudaTensor[] x2928 = x239.backward(x2925,x2926,x2927);
x2924 = x2928[1];

// val X2137 = X2136 * d_BatchNorm()(X239,3a1_bn_scale)/d_X239
JCudaTensor x2929;
x2929 = x2928[0];

// val X62011 = X2136 * d_BatchNorm()(X239,3a1_bn_scale)/d_3a1_bn_bias
JCudaTensor x2933;
x2933 = x2928[2];

// Dealloc(X239)
JCudaTensor x2937;
x2937 = x226;
x2937.free();

// val X78256 = X2148 * d_BatchNorm()(X248,3a2_c_bn_scale)/d_3a2_c_bn_scale
JCudaTensor x2938;
x2938 = x2923[1];

// val X2149 = X2148 * d_BatchNorm()(X248,3a2_c_bn_scale)/d_X248
JCudaTensor x2942;
x2942 = x2923[0];

// Dealloc(X248)
JCudaTensor x2946;
x2946 = x267;
x2946.free();

// V_3a2_c_cv_W <~~ X2149 * d_Convolv(1,0)(X247)/d_3a2_c_cv_W
float x2948, x2949;
x2948 = lrn_rate_1;
x2949 = momentum;
JCudaTensor x2950, x2951;
x2950 = x2942;
x2951 = x264;
x273.backward_filter(x2950,x2951, x2947, x2948, x2949);

// V_3a1_bn_bias <~~ X62011
float x2953, x2954;
x2953 = lrn_rate_1;
x2954 = momentum;
JCudaTensor x2955;
x2955 = x2933;
x2952.update(x2955, x2953, x2954);


// Dealloc(X62011)
JCudaTensor x2956;
x2956 = x2933;
x2956.free();

// V_3a1_cv_W <~~ X2137 * d_Convolv(2,0)(X238)/d_3a1_cv_W
float x2958, x2959;
x2958 = lrn_rate_1;
x2959 = momentum;
JCudaTensor x2960, x2961;
x2960 = x2929;
x2961 = x217;
x232.backward_filter(x2960,x2961, x2957, x2958, x2959);

// val X2138 = X2137 * d_Convolv(2,0)(3a1_cv_W)/d_X238
JCudaTensor x2962;
JCudaTensor x2963, x2964;
x2963 = x2929;
x2964 = x230;
x2962 = x232.backward_data(x2963,x2964);

// Dealloc(X2137)
JCudaTensor x2965;
x2965 = x2929;
x2965.free();

// V_3a1_bn_scale <~~ X63616
float x2967, x2968;
x2967 = lrn_rate_1;
x2968 = momentum;
JCudaTensor x2969;
x2969 = x2924;
x2966.update(x2969, x2967, x2968);


// Dealloc(X63616)
JCudaTensor x2970;
x2970 = x2924;
x2970.free();

// V_3a2_c_bn_bias <~~ X76642
float x2972, x2973;
x2972 = lrn_rate_1;
x2973 = momentum;
JCudaTensor x2974;
x2974 = x2919;
x2971.update(x2974, x2972, x2973);


// Dealloc(X76642)
JCudaTensor x2975;
x2975 = x2919;
x2975.free();

// val X2150 = X2149 * d_Convolv(1,0)(3a2_c_cv_W)/d_X247
JCudaTensor x2976;
JCudaTensor x2977, x2978;
x2977 = x2942;
x2978 = x271;
x2976 = x273.backward_data(x2977,x2978);

// Dealloc(X2149)
JCudaTensor x2979;
x2979 = x2942;
x2979.free();

// V_3a2_c_bn_scale <~~ X78256
float x2981, x2982;
x2981 = lrn_rate_1;
x2982 = momentum;
JCudaTensor x2983;
x2983 = x2938;
x2980.update(x2983, x2981, x2982);


// Dealloc(X78256)
JCudaTensor x2984;
x2984 = x2938;
x2984.free();

// 3a1_bn_scale <~~ V_3a1_bn_scale
float x2985, x2986;
x2985 = 1;
x2986 = decay_1;
JCudaTensor x2987;
x2987 = x2966;
x237.update(x2987, x2985, x2986);


// 3a2_c_bn_scale <~~ V_3a2_c_bn_scale
float x2988, x2989;
x2988 = 1;
x2989 = decay_1;
JCudaTensor x2990;
x2990 = x2980;
x278.update(x2990, x2988, x2989);


// 3a1_cv_W <~~ V_3a1_cv_W
float x2991, x2992;
x2991 = 1;
x2992 = decay_1;
JCudaTensor x2993;
x2993 = x2957;
x230.update(x2993, x2991, x2992);


// 3a2_c_bn_bias <~~ V_3a2_c_bn_bias
float x2994, x2995;
x2994 = 1;
x2995 = decay_1;
JCudaTensor x2996;
x2996 = x2971;
x279.update(x2996, x2994, x2995);


// 3a2_c_cv_W <~~ V_3a2_c_cv_W
float x2997, x2998;
x2997 = 1;
x2998 = decay_1;
JCudaTensor x2999;
x2999 = x2947;
x271.update(x2999, x2997, x2998);


// 3a1_bn_bias <~~ V_3a1_bn_bias
float x3000, x3001;
x3000 = 1;
x3001 = decay_1;
JCudaTensor x3002;
x3002 = x2952;
x238.update(x3002, x3000, x3001);


// val X2154 = X2150 * d_ReLU()(X247)/d_X246
JCudaTensor x3003;
JCudaTensor x3004, x3005;
x3004 = x2976;
x3005 = x264;
x3003 = x266.backward(x3004,x3005);

// Dealloc(X247)
JCudaTensor x3006;
x3006 = x264;
x3006.free();

// val X2155 = X2154 * d_BatchNorm()(X245,3a2_b_bn_scale)/d_X245
JCudaTensor x3007;
JCudaTensor x3008, x3009, x3010;
x3008 = x3003;
x3009 = x250;
x3010 = x261;
JCudaTensor[] x3011 = x263.backward(x3008,x3009,x3010);
x3007 = x3011[0];

// val X73399 = X2154 * d_BatchNorm()(X245,3a2_b_bn_scale)/d_3a2_b_bn_scale
JCudaTensor x3012;
x3012 = x3011[1];

// val X71772 = X2154 * d_BatchNorm()(X245,3a2_b_bn_scale)/d_3a2_b_bn_bias
JCudaTensor x3016;
x3016 = x3011[2];

// Dealloc(X245)
JCudaTensor x3020;
x3020 = x250;
x3020.free();

// V_3a2_b_bn_bias <~~ X71772
float x3022, x3023;
x3022 = lrn_rate_1;
x3023 = momentum;
JCudaTensor x3024;
x3024 = x3016;
x3021.update(x3024, x3022, x3023);


// Dealloc(X71772)
JCudaTensor x3025;
x3025 = x3016;
x3025.free();

// V_3a2_b_bn_scale <~~ X73399
float x3027, x3028;
x3027 = lrn_rate_1;
x3028 = momentum;
JCudaTensor x3029;
x3029 = x3012;
x3026.update(x3029, x3027, x3028);


// Dealloc(X73399)
JCudaTensor x3030;
x3030 = x3012;
x3030.free();

// val X2156 = X2155 * d_Convolv(1,1)(3a2_b_cv_W)/d_X244
JCudaTensor x3031;
JCudaTensor x3032, x3033;
x3032 = x3007;
x3033 = x254;
x3031 = x256.backward_data(x3032,x3033);

// V_3a2_b_cv_W <~~ X2155 * d_Convolv(1,1)(X244)/d_3a2_b_cv_W
float x3035, x3036;
x3035 = lrn_rate_1;
x3036 = momentum;
JCudaTensor x3037, x3038;
x3037 = x3007;
x3038 = x247;
x256.backward_filter(x3037,x3038, x3034, x3035, x3036);

// Dealloc(X2155)
JCudaTensor x3039;
x3039 = x3007;
x3039.free();

// 3a2_b_bn_bias <~~ V_3a2_b_bn_bias
float x3040, x3041;
x3040 = 1;
x3041 = decay_1;
JCudaTensor x3042;
x3042 = x3021;
x262.update(x3042, x3040, x3041);


// 3a2_b_bn_scale <~~ V_3a2_b_bn_scale
float x3043, x3044;
x3043 = 1;
x3044 = decay_1;
JCudaTensor x3045;
x3045 = x3026;
x261.update(x3045, x3043, x3044);


// 3a2_b_cv_W <~~ V_3a2_b_cv_W
float x3046, x3047;
x3046 = 1;
x3047 = decay_1;
JCudaTensor x3048;
x3048 = x3034;
x254.update(x3048, x3046, x3047);


// val X2160 = X2156 * d_ReLU()(X244)/d_X243
JCudaTensor x3049;
JCudaTensor x3050, x3051;
x3050 = x3031;
x3051 = x247;
x3049 = x249.backward(x3050,x3051);

// Dealloc(X244)
JCudaTensor x3052;
x3052 = x247;
x3052.free();

// val X66863 = X2160 * d_BatchNorm()(X242,3a2_a_bn_scale)/d_3a2_a_bn_bias
JCudaTensor x3053;
JCudaTensor x3054, x3055, x3056;
x3054 = x3049;
x3055 = x219;
x3056 = x244;
JCudaTensor[] x3057 = x246.backward(x3054,x3055,x3056);
x3053 = x3057[2];

// val X2161 = X2160 * d_BatchNorm()(X242,3a2_a_bn_scale)/d_X242
JCudaTensor x3058;
x3058 = x3057[0];

// val X68503 = X2160 * d_BatchNorm()(X242,3a2_a_bn_scale)/d_3a2_a_bn_scale
JCudaTensor x3062;
x3062 = x3057[1];

// Dealloc(X242)
JCudaTensor x3066;
x3066 = x219;
x3066.free();

// val X2163 = (X2138 + X2161 * d_Convolv(2,0)(3a2_a_cv_W)/d_X238)
JCudaTensor x3067;
JCudaTensor x3068;
x3068 = x2962;
JCudaTensor x3069, x3070;
x3069 = x3058;
x3070 = x223;
x3067 = x225.backward_data(x3069,x3070, x3068);

// V_3a2_a_cv_W <~~ X2161 * d_Convolv(2,0)(X238)/d_3a2_a_cv_W
float x3072, x3073;
x3072 = lrn_rate_1;
x3073 = momentum;
JCudaTensor x3074, x3075;
x3074 = x3058;
x3075 = x217;
x225.backward_filter(x3074,x3075, x3071, x3072, x3073);

// Dealloc(X2161)
JCudaTensor x3076;
x3076 = x3058;
x3076.free();

// V_3a2_a_bn_scale <~~ X68503
float x3078, x3079;
x3078 = lrn_rate_1;
x3079 = momentum;
JCudaTensor x3080;
x3080 = x3062;
x3077.update(x3080, x3078, x3079);


// Dealloc(X68503)
JCudaTensor x3081;
x3081 = x3062;
x3081.free();

// V_3a2_a_bn_bias <~~ X66863
float x3083, x3084;
x3083 = lrn_rate_1;
x3084 = momentum;
JCudaTensor x3085;
x3085 = x3053;
x3082.update(x3085, x3083, x3084);


// Dealloc(X66863)
JCudaTensor x3086;
x3086 = x3053;
x3086.free();

// 3a2_a_cv_W <~~ V_3a2_a_cv_W
float x3087, x3088;
x3087 = 1;
x3088 = decay_1;
JCudaTensor x3089;
x3089 = x3071;
x223.update(x3089, x3087, x3088);


// 3a2_a_bn_scale <~~ V_3a2_a_bn_scale
float x3090, x3091;
x3090 = 1;
x3091 = decay_1;
JCudaTensor x3092;
x3092 = x3077;
x244.update(x3092, x3090, x3091);


// 3a2_a_bn_bias <~~ V_3a2_a_bn_bias
float x3093, x3094;
x3093 = 1;
x3094 = decay_1;
JCudaTensor x3095;
x3095 = x3082;
x245.update(x3095, x3093, x3094);


// val X2200 = X2163 * d_ReLU()(X238)/d_X237
JCudaTensor x3096;
JCudaTensor x3097, x3098;
x3097 = x3067;
x3098 = x217;
x3096 = x213.backward(x3097,x3098);

// Dealloc(X238)
JCudaTensor x3099;
x3099 = x217;
x3099.free();

// val X2210 = X2200.copy * d_ReLU()(X236)/d_X235
JCudaTensor x3100;
JCudaTensor x3101, x3102;
x3101 = x3096;
x3101 = x3101.clone();
x3102 = x211;
x3100 = x213.backward(x3101,x3102);

// Dealloc(X236)
JCudaTensor x3103;
x3103 = x211;
x3103.free();

// val X2211 = X2210 * d_BatchNorm()(X234,2c_c_bn_scale)/d_X234
JCudaTensor x3104;
JCudaTensor x3105, x3106, x3107;
x3105 = x3100;
x3106 = x197;
x3107 = x208;
JCudaTensor[] x3108 = x210.backward(x3105,x3106,x3107);
x3104 = x3108[0];

// val X58682 = X2210 * d_BatchNorm()(X234,2c_c_bn_scale)/d_2c_c_bn_scale
JCudaTensor x3109;
x3109 = x3108[1];

// val X56960 = X2210 * d_BatchNorm()(X234,2c_c_bn_scale)/d_2c_c_bn_bias
JCudaTensor x3113;
x3113 = x3108[2];

// Dealloc(X234)
JCudaTensor x3117;
x3117 = x197;
x3117.free();

// V_2c_c_bn_bias <~~ X56960
float x3119, x3120;
x3119 = lrn_rate_1;
x3120 = momentum;
JCudaTensor x3121;
x3121 = x3113;
x3118.update(x3121, x3119, x3120);


// Dealloc(X56960)
JCudaTensor x3122;
x3122 = x3113;
x3122.free();

// V_2c_c_bn_scale <~~ X58682
float x3124, x3125;
x3124 = lrn_rate_1;
x3125 = momentum;
JCudaTensor x3126;
x3126 = x3109;
x3123.update(x3126, x3124, x3125);


// Dealloc(X58682)
JCudaTensor x3127;
x3127 = x3109;
x3127.free();

// val X2212 = X2211 * d_Convolv(1,0)(2c_c_cv_W)/d_X233
JCudaTensor x3128;
JCudaTensor x3129, x3130;
x3129 = x3104;
x3130 = x201;
x3128 = x203.backward_data(x3129,x3130);

// V_2c_c_cv_W <~~ X2211 * d_Convolv(1,0)(X233)/d_2c_c_cv_W
float x3132, x3133;
x3132 = lrn_rate_1;
x3133 = momentum;
JCudaTensor x3134, x3135;
x3134 = x3104;
x3135 = x194;
x203.backward_filter(x3134,x3135, x3131, x3132, x3133);

// Dealloc(X2211)
JCudaTensor x3136;
x3136 = x3104;
x3136.free();

// 2c_c_bn_bias <~~ V_2c_c_bn_bias
float x3137, x3138;
x3137 = 1;
x3138 = decay_1;
JCudaTensor x3139;
x3139 = x3118;
x209.update(x3139, x3137, x3138);


// 2c_c_bn_scale <~~ V_2c_c_bn_scale
float x3140, x3141;
x3140 = 1;
x3141 = decay_1;
JCudaTensor x3142;
x3142 = x3123;
x208.update(x3142, x3140, x3141);


// 2c_c_cv_W <~~ V_2c_c_cv_W
float x3143, x3144;
x3143 = 1;
x3144 = decay_1;
JCudaTensor x3145;
x3145 = x3131;
x201.update(x3145, x3143, x3144);


// val X2216 = X2212 * d_ReLU()(X233)/d_X232
JCudaTensor x3146;
JCudaTensor x3147, x3148;
x3147 = x3128;
x3148 = x194;
x3146 = x196.backward(x3147,x3148);

// Dealloc(X233)
JCudaTensor x3149;
x3149 = x194;
x3149.free();

// val X53501 = X2216 * d_BatchNorm()(X231,2c_b_bn_scale)/d_2c_b_bn_scale
JCudaTensor x3150;
JCudaTensor x3151, x3152, x3153;
x3151 = x3146;
x3152 = x180;
x3153 = x191;
JCudaTensor[] x3154 = x193.backward(x3151,x3152,x3153);
x3150 = x3154[1];

// val X51766 = X2216 * d_BatchNorm()(X231,2c_b_bn_scale)/d_2c_b_bn_bias
JCudaTensor x3155;
x3155 = x3154[2];

// val X2217 = X2216 * d_BatchNorm()(X231,2c_b_bn_scale)/d_X231
JCudaTensor x3159;
x3159 = x3154[0];

// Dealloc(X231)
JCudaTensor x3163;
x3163 = x180;
x3163.free();

// val X2218 = X2217 * d_Convolv(1,1)(2c_b_cv_W)/d_X230
JCudaTensor x3164;
JCudaTensor x3165, x3166;
x3165 = x3159;
x3166 = x184;
x3164 = x186.backward_data(x3165,x3166);

// V_2c_b_cv_W <~~ X2217 * d_Convolv(1,1)(X230)/d_2c_b_cv_W
float x3168, x3169;
x3168 = lrn_rate_1;
x3169 = momentum;
JCudaTensor x3170, x3171;
x3170 = x3159;
x3171 = x177;
x186.backward_filter(x3170,x3171, x3167, x3168, x3169);

// Dealloc(X2217)
JCudaTensor x3172;
x3172 = x3159;
x3172.free();

// V_2c_b_bn_bias <~~ X51766
float x3174, x3175;
x3174 = lrn_rate_1;
x3175 = momentum;
JCudaTensor x3176;
x3176 = x3155;
x3173.update(x3176, x3174, x3175);


// Dealloc(X51766)
JCudaTensor x3177;
x3177 = x3155;
x3177.free();

// V_2c_b_bn_scale <~~ X53501
float x3179, x3180;
x3179 = lrn_rate_1;
x3180 = momentum;
JCudaTensor x3181;
x3181 = x3150;
x3178.update(x3181, x3179, x3180);


// Dealloc(X53501)
JCudaTensor x3182;
x3182 = x3150;
x3182.free();

// 2c_b_cv_W <~~ V_2c_b_cv_W
float x3183, x3184;
x3183 = 1;
x3184 = decay_1;
JCudaTensor x3185;
x3185 = x3167;
x184.update(x3185, x3183, x3184);


// 2c_b_bn_bias <~~ V_2c_b_bn_bias
float x3186, x3187;
x3186 = 1;
x3187 = decay_1;
JCudaTensor x3188;
x3188 = x3173;
x192.update(x3188, x3186, x3187);


// 2c_b_bn_scale <~~ V_2c_b_bn_scale
float x3189, x3190;
x3189 = 1;
x3190 = decay_1;
JCudaTensor x3191;
x3191 = x3178;
x191.update(x3191, x3189, x3190);


// val X2222 = X2218 * d_ReLU()(X230)/d_X229
JCudaTensor x3192;
JCudaTensor x3193, x3194;
x3193 = x3164;
x3194 = x177;
x3192 = x179.backward(x3193,x3194);

// Dealloc(X230)
JCudaTensor x3195;
x3195 = x177;
x3195.free();

// val X46533 = X2222 * d_BatchNorm()(X228,2c_a_bn_scale)/d_2c_a_bn_bias
JCudaTensor x3196;
JCudaTensor x3197, x3198, x3199;
x3197 = x3192;
x3198 = x163;
x3199 = x174;
JCudaTensor[] x3200 = x176.backward(x3197,x3198,x3199);
x3196 = x3200[2];

// val X2223 = X2222 * d_BatchNorm()(X228,2c_a_bn_scale)/d_X228
JCudaTensor x3201;
x3201 = x3200[0];

// val X48281 = X2222 * d_BatchNorm()(X228,2c_a_bn_scale)/d_2c_a_bn_scale
JCudaTensor x3205;
x3205 = x3200[1];

// Dealloc(X228)
JCudaTensor x3209;
x3209 = x163;
x3209.free();

// V_2c_a_bn_scale <~~ X48281
float x3211, x3212;
x3211 = lrn_rate_1;
x3212 = momentum;
JCudaTensor x3213;
x3213 = x3205;
x3210.update(x3213, x3211, x3212);


// Dealloc(X48281)
JCudaTensor x3214;
x3214 = x3205;
x3214.free();

// val X2224 = X2223 * d_Convolv(1,0)(2c_a_cv_W)/d_X227
JCudaTensor x3215;
JCudaTensor x3216, x3217;
x3216 = x3201;
x3217 = x167;
x3215 = x169.backward_data(x3216,x3217);

// V_2c_a_cv_W <~~ X2223 * d_Convolv(1,0)(X227)/d_2c_a_cv_W
float x3219, x3220;
x3219 = lrn_rate_1;
x3220 = momentum;
JCudaTensor x3221, x3222;
x3221 = x3201;
x3222 = x161;
x169.backward_filter(x3221,x3222, x3218, x3219, x3220);

// Dealloc(X2223)
JCudaTensor x3223;
x3223 = x3201;
x3223.free();

// V_2c_a_bn_bias <~~ X46533
float x3225, x3226;
x3225 = lrn_rate_1;
x3226 = momentum;
JCudaTensor x3227;
x3227 = x3196;
x3224.update(x3227, x3225, x3226);


// Dealloc(X46533)
JCudaTensor x3228;
x3228 = x3196;
x3228.free();

// 2c_a_bn_scale <~~ V_2c_a_bn_scale
float x3229, x3230;
x3229 = 1;
x3230 = decay_1;
JCudaTensor x3231;
x3231 = x3210;
x174.update(x3231, x3229, x3230);


// 2c_a_cv_W <~~ V_2c_a_cv_W
float x3232, x3233;
x3232 = 1;
x3233 = decay_1;
JCudaTensor x3234;
x3234 = x3218;
x167.update(x3234, x3232, x3233);


// 2c_a_bn_bias <~~ V_2c_a_bn_bias
float x3235, x3236;
x3235 = 1;
x3236 = decay_1;
JCudaTensor x3237;
x3237 = x3224;
x175.update(x3237, x3235, x3236);


// val X2225 = (X2224 + X2200)
JCudaTensor x3238;
JCudaTensor x3239, x3240;
x3239 = x3215;
x3240 = x3096;
x3238 = x3239.plus_i(x3240);

// Dealloc(X2200)
JCudaTensor x3241;
x3241 = x3096;
x3241.free();

// val X2237 = X2225 * d_ReLU()(X227)/d_X226
JCudaTensor x3242;
JCudaTensor x3243, x3244;
x3243 = x3238;
x3244 = x161;
x3242 = x157.backward(x3243,x3244);

// Dealloc(X227)
JCudaTensor x3245;
x3245 = x161;
x3245.free();

// val X2247 = X2237.copy * d_ReLU()(X225)/d_X224
JCudaTensor x3246;
JCudaTensor x3247, x3248;
x3247 = x3242;
x3247 = x3247.clone();
x3248 = x155;
x3246 = x157.backward(x3247,x3248);

// Dealloc(X225)
JCudaTensor x3249;
x3249 = x155;
x3249.free();

// val X2248 = X2247 * d_BatchNorm()(X223,2b_c_bn_scale)/d_X223
JCudaTensor x3250;
JCudaTensor x3251, x3252, x3253;
x3251 = x3246;
x3252 = x141;
x3253 = x152;
JCudaTensor[] x3254 = x154.backward(x3251,x3252,x3253);
x3250 = x3254[0];

// val X42994 = X2247 * d_BatchNorm()(X223,2b_c_bn_scale)/d_2b_c_bn_scale
JCudaTensor x3255;
x3255 = x3254[1];

// val X41205 = X2247 * d_BatchNorm()(X223,2b_c_bn_scale)/d_2b_c_bn_bias
JCudaTensor x3259;
x3259 = x3254[2];

// Dealloc(X223)
JCudaTensor x3263;
x3263 = x141;
x3263.free();

// V_2b_c_bn_scale <~~ X42994
float x3265, x3266;
x3265 = lrn_rate_1;
x3266 = momentum;
JCudaTensor x3267;
x3267 = x3255;
x3264.update(x3267, x3265, x3266);


// Dealloc(X42994)
JCudaTensor x3268;
x3268 = x3255;
x3268.free();

// val X2249 = X2248 * d_Convolv(1,0)(2b_c_cv_W)/d_X222
JCudaTensor x3269;
JCudaTensor x3270, x3271;
x3270 = x3250;
x3271 = x145;
x3269 = x147.backward_data(x3270,x3271);

// V_2b_c_bn_bias <~~ X41205
float x3273, x3274;
x3273 = lrn_rate_1;
x3274 = momentum;
JCudaTensor x3275;
x3275 = x3259;
x3272.update(x3275, x3273, x3274);


// Dealloc(X41205)
JCudaTensor x3276;
x3276 = x3259;
x3276.free();

// V_2b_c_cv_W <~~ X2248 * d_Convolv(1,0)(X222)/d_2b_c_cv_W
float x3278, x3279;
x3278 = lrn_rate_1;
x3279 = momentum;
JCudaTensor x3280, x3281;
x3280 = x3250;
x3281 = x138;
x147.backward_filter(x3280,x3281, x3277, x3278, x3279);

// Dealloc(X2248)
JCudaTensor x3282;
x3282 = x3250;
x3282.free();

// 2b_c_bn_scale <~~ V_2b_c_bn_scale
float x3283, x3284;
x3283 = 1;
x3284 = decay_1;
JCudaTensor x3285;
x3285 = x3264;
x152.update(x3285, x3283, x3284);


// 2b_c_bn_bias <~~ V_2b_c_bn_bias
float x3286, x3287;
x3286 = 1;
x3287 = decay_1;
JCudaTensor x3288;
x3288 = x3272;
x153.update(x3288, x3286, x3287);


// 2b_c_cv_W <~~ V_2b_c_cv_W
float x3289, x3290;
x3289 = 1;
x3290 = decay_1;
JCudaTensor x3291;
x3291 = x3277;
x145.update(x3291, x3289, x3290);


// val X2253 = X2249 * d_ReLU()(X222)/d_X221
JCudaTensor x3292;
JCudaTensor x3293, x3294;
x3293 = x3269;
x3294 = x138;
x3292 = x140.backward(x3293,x3294);

// Dealloc(X222)
JCudaTensor x3295;
x3295 = x138;
x3295.free();

// val X37612 = X2253 * d_BatchNorm()(X220,2b_b_bn_scale)/d_2b_b_bn_scale
JCudaTensor x3296;
JCudaTensor x3297, x3298, x3299;
x3297 = x3292;
x3298 = x124;
x3299 = x135;
JCudaTensor[] x3300 = x137.backward(x3297,x3298,x3299);
x3296 = x3300[1];

// val X2254 = X2253 * d_BatchNorm()(X220,2b_b_bn_scale)/d_X220
JCudaTensor x3301;
x3301 = x3300[0];

// val X35810 = X2253 * d_BatchNorm()(X220,2b_b_bn_scale)/d_2b_b_bn_bias
JCudaTensor x3305;
x3305 = x3300[2];

// Dealloc(X220)
JCudaTensor x3309;
x3309 = x124;
x3309.free();

// val X2255 = X2254 * d_Convolv(1,1)(2b_b_cv_W)/d_X219
JCudaTensor x3310;
JCudaTensor x3311, x3312;
x3311 = x3301;
x3312 = x128;
x3310 = x130.backward_data(x3311,x3312);

// V_2b_b_bn_scale <~~ X37612
float x3314, x3315;
x3314 = lrn_rate_1;
x3315 = momentum;
JCudaTensor x3316;
x3316 = x3296;
x3313.update(x3316, x3314, x3315);


// Dealloc(X37612)
JCudaTensor x3317;
x3317 = x3296;
x3317.free();

// V_2b_b_bn_bias <~~ X35810
float x3319, x3320;
x3319 = lrn_rate_1;
x3320 = momentum;
JCudaTensor x3321;
x3321 = x3305;
x3318.update(x3321, x3319, x3320);


// Dealloc(X35810)
JCudaTensor x3322;
x3322 = x3305;
x3322.free();

// V_2b_b_cv_W <~~ X2254 * d_Convolv(1,1)(X219)/d_2b_b_cv_W
float x3324, x3325;
x3324 = lrn_rate_1;
x3325 = momentum;
JCudaTensor x3326, x3327;
x3326 = x3301;
x3327 = x121;
x130.backward_filter(x3326,x3327, x3323, x3324, x3325);

// Dealloc(X2254)
JCudaTensor x3328;
x3328 = x3301;
x3328.free();

// 2b_b_bn_scale <~~ V_2b_b_bn_scale
float x3329, x3330;
x3329 = 1;
x3330 = decay_1;
JCudaTensor x3331;
x3331 = x3313;
x135.update(x3331, x3329, x3330);


// 2b_b_bn_bias <~~ V_2b_b_bn_bias
float x3332, x3333;
x3332 = 1;
x3333 = decay_1;
JCudaTensor x3334;
x3334 = x3318;
x136.update(x3334, x3332, x3333);


// 2b_b_cv_W <~~ V_2b_b_cv_W
float x3335, x3336;
x3335 = 1;
x3336 = decay_1;
JCudaTensor x3337;
x3337 = x3323;
x128.update(x3337, x3335, x3336);


// val X2259 = X2255 * d_ReLU()(X219)/d_X218
JCudaTensor x3338;
JCudaTensor x3339, x3340;
x3339 = x3310;
x3340 = x121;
x3338 = x123.backward(x3339,x3340);

// Dealloc(X219)
JCudaTensor x3341;
x3341 = x121;
x3341.free();

// val X2260 = X2259 * d_BatchNorm()(X217,2b_a_bn_scale)/d_X217
JCudaTensor x3342;
JCudaTensor x3343, x3344, x3345;
x3343 = x3338;
x3344 = x107;
x3345 = x118;
JCudaTensor[] x3346 = x120.backward(x3343,x3344,x3345);
x3342 = x3346[0];

// val X32191 = X2259 * d_BatchNorm()(X217,2b_a_bn_scale)/d_2b_a_bn_scale
JCudaTensor x3347;
x3347 = x3346[1];

// val X30376 = X2259 * d_BatchNorm()(X217,2b_a_bn_scale)/d_2b_a_bn_bias
JCudaTensor x3351;
x3351 = x3346[2];

// Dealloc(X217)
JCudaTensor x3355;
x3355 = x107;
x3355.free();

// V_2b_a_bn_scale <~~ X32191
float x3357, x3358;
x3357 = lrn_rate_1;
x3358 = momentum;
JCudaTensor x3359;
x3359 = x3347;
x3356.update(x3359, x3357, x3358);


// Dealloc(X32191)
JCudaTensor x3360;
x3360 = x3347;
x3360.free();

// val X2261 = X2260 * d_Convolv(1,0)(2b_a_cv_W)/d_X216
JCudaTensor x3361;
JCudaTensor x3362, x3363;
x3362 = x3342;
x3363 = x111;
x3361 = x113.backward_data(x3362,x3363);

// V_2b_a_cv_W <~~ X2260 * d_Convolv(1,0)(X216)/d_2b_a_cv_W
float x3365, x3366;
x3365 = lrn_rate_1;
x3366 = momentum;
JCudaTensor x3367, x3368;
x3367 = x3342;
x3368 = x105;
x113.backward_filter(x3367,x3368, x3364, x3365, x3366);

// Dealloc(X2260)
JCudaTensor x3369;
x3369 = x3342;
x3369.free();

// V_2b_a_bn_bias <~~ X30376
float x3371, x3372;
x3371 = lrn_rate_1;
x3372 = momentum;
JCudaTensor x3373;
x3373 = x3351;
x3370.update(x3373, x3371, x3372);


// Dealloc(X30376)
JCudaTensor x3374;
x3374 = x3351;
x3374.free();

// 2b_a_bn_scale <~~ V_2b_a_bn_scale
float x3375, x3376;
x3375 = 1;
x3376 = decay_1;
JCudaTensor x3377;
x3377 = x3356;
x118.update(x3377, x3375, x3376);


// 2b_a_cv_W <~~ V_2b_a_cv_W
float x3378, x3379;
x3378 = 1;
x3379 = decay_1;
JCudaTensor x3380;
x3380 = x3364;
x111.update(x3380, x3378, x3379);


// 2b_a_bn_bias <~~ V_2b_a_bn_bias
float x3381, x3382;
x3381 = 1;
x3382 = decay_1;
JCudaTensor x3383;
x3383 = x3370;
x119.update(x3383, x3381, x3382);


// val X2262 = (X2261 + X2237)
JCudaTensor x3384;
JCudaTensor x3385, x3386;
x3385 = x3361;
x3386 = x3242;
x3384 = x3385.plus_i(x3386);

// Dealloc(X2237)
JCudaTensor x3387;
x3387 = x3242;
x3387.free();

// val X2277 = X2262 * d_ReLU()(X216)/d_X215
JCudaTensor x3388;
JCudaTensor x3389, x3390;
x3389 = x3384;
x3390 = x105;
x3388 = x98.backward(x3389,x3390);

// Dealloc(X216)
JCudaTensor x3391;
x3391 = x105;
x3391.free();

// val X2281 = X2277.copy * d_ReLU()(X205)/d_X204
JCudaTensor x3392;
JCudaTensor x3393, x3394;
x3393 = x3388;
x3393 = x3393.clone();
x3394 = x96;
x3392 = x98.backward(x3393,x3394);

// Dealloc(X205)
JCudaTensor x3395;
x3395 = x96;
x3395.free();

// val X2293 = X2277.copy * d_ReLU()(X214)/d_X213
JCudaTensor x3396;
JCudaTensor x3397, x3398;
x3397 = x3388;
x3397 = x3397.clone();
x3398 = x99;
x3396 = x101.backward(x3397,x3398);

// Dealloc(X2277)
JCudaTensor x3399;
x3399 = x3388;
x3399.free();

// Dealloc(X214)
JCudaTensor x3400;
x3400 = x99;
x3400.free();

// val X2282 = X2281 * d_BatchNorm()(X203,2a1_bn_scale)/d_X203
JCudaTensor x3401;
JCudaTensor x3402, x3403, x3404;
x3402 = x3392;
x3403 = x34;
x3404 = x52;
JCudaTensor[] x3405 = x54.backward(x3402,x3403,x3404);
x3401 = x3405[0];

// val X24841 = X2293 * d_BatchNorm()(X212,2a2_c_bn_scale)/d_2a2_c_bn_bias
JCudaTensor x3406;
JCudaTensor x3407, x3408, x3409;
x3407 = x3396;
x3408 = x82;
x3409 = x93;
JCudaTensor[] x3410 = x95.backward(x3407,x3408,x3409);
x3406 = x3410[2];

// val X2294 = X2293 * d_BatchNorm()(X212,2a2_c_bn_scale)/d_X212
JCudaTensor x3411;
x3411 = x3410[0];

// val X9855 = X2281 * d_BatchNorm()(X203,2a1_bn_scale)/d_2a1_bn_scale
JCudaTensor x3415;
x3415 = x3405[1];

// val X8005 = X2281 * d_BatchNorm()(X203,2a1_bn_scale)/d_2a1_bn_bias
JCudaTensor x3419;
x3419 = x3405[2];

// Dealloc(X203)
JCudaTensor x3423;
x3423 = x34;
x3423.free();

// val X26700 = X2293 * d_BatchNorm()(X212,2a2_c_bn_scale)/d_2a2_c_bn_scale
JCudaTensor x3424;
x3424 = x3410[1];

// Dealloc(X212)
JCudaTensor x3428;
x3428 = x82;
x3428.free();

// val X2295 = X2294 * d_Convolv(1,0)(2a2_c_cv_W)/d_X211
JCudaTensor x3429;
JCudaTensor x3430, x3431;
x3430 = x3411;
x3431 = x86;
x3429 = x88.backward_data(x3430,x3431);

// V_2a2_c_cv_W <~~ X2294 * d_Convolv(1,0)(X211)/d_2a2_c_cv_W
float x3433, x3434;
x3433 = lrn_rate_1;
x3434 = momentum;
JCudaTensor x3435, x3436;
x3435 = x3411;
x3436 = x79;
x88.backward_filter(x3435,x3436, x3432, x3433, x3434);

// Dealloc(X2294)
JCudaTensor x3437;
x3437 = x3411;
x3437.free();

// V_2a1_bn_bias <~~ X8005
float x3439, x3440;
x3439 = lrn_rate_1;
x3440 = momentum;
JCudaTensor x3441;
x3441 = x3419;
x3438.update(x3441, x3439, x3440);


// Dealloc(X8005)
JCudaTensor x3442;
x3442 = x3419;
x3442.free();

// V_2a2_c_bn_bias <~~ X24841
float x3444, x3445;
x3444 = lrn_rate_1;
x3445 = momentum;
JCudaTensor x3446;
x3446 = x3406;
x3443.update(x3446, x3444, x3445);


// Dealloc(X24841)
JCudaTensor x3447;
x3447 = x3406;
x3447.free();

// V_2a1_cv_W <~~ X2282 * d_Convolv(1,0)(X202)/d_2a1_cv_W
float x3449, x3450;
x3449 = lrn_rate_1;
x3450 = momentum;
JCudaTensor x3451, x3452;
x3451 = x3401;
x3452 = x31;
x40.backward_filter(x3451,x3452, x3448, x3449, x3450);

// V_2a2_c_bn_scale <~~ X26700
float x3454, x3455;
x3454 = lrn_rate_1;
x3455 = momentum;
JCudaTensor x3456;
x3456 = x3424;
x3453.update(x3456, x3454, x3455);


// Dealloc(X26700)
JCudaTensor x3457;
x3457 = x3424;
x3457.free();

// V_2a1_bn_scale <~~ X9855
float x3459, x3460;
x3459 = lrn_rate_1;
x3460 = momentum;
JCudaTensor x3461;
x3461 = x3415;
x3458.update(x3461, x3459, x3460);


// Dealloc(X9855)
JCudaTensor x3462;
x3462 = x3415;
x3462.free();

// val X2283 = X2282 * d_Convolv(1,0)(2a1_cv_W)/d_X202
JCudaTensor x3463;
JCudaTensor x3464, x3465;
x3464 = x3401;
x3465 = x38;
x3463 = x40.backward_data(x3464,x3465);

// Dealloc(X2282)
JCudaTensor x3466;
x3466 = x3401;
x3466.free();

// 2a1_bn_scale <~~ V_2a1_bn_scale
float x3467, x3468;
x3467 = 1;
x3468 = decay_1;
JCudaTensor x3469;
x3469 = x3458;
x52.update(x3469, x3467, x3468);


// 2a2_c_cv_W <~~ V_2a2_c_cv_W
float x3470, x3471;
x3470 = 1;
x3471 = decay_1;
JCudaTensor x3472;
x3472 = x3432;
x86.update(x3472, x3470, x3471);


// 2a2_c_bn_bias <~~ V_2a2_c_bn_bias
float x3473, x3474;
x3473 = 1;
x3474 = decay_1;
JCudaTensor x3475;
x3475 = x3443;
x94.update(x3475, x3473, x3474);


// 2a1_bn_bias <~~ V_2a1_bn_bias
float x3476, x3477;
x3476 = 1;
x3477 = decay_1;
JCudaTensor x3478;
x3478 = x3438;
x53.update(x3478, x3476, x3477);


// 2a2_c_bn_scale <~~ V_2a2_c_bn_scale
float x3479, x3480;
x3479 = 1;
x3480 = decay_1;
JCudaTensor x3481;
x3481 = x3453;
x93.update(x3481, x3479, x3480);


// 2a1_cv_W <~~ V_2a1_cv_W
float x3482, x3483;
x3482 = 1;
x3483 = decay_1;
JCudaTensor x3484;
x3484 = x3448;
x38.update(x3484, x3482, x3483);


// val X2299 = X2295 * d_ReLU()(X211)/d_X210
JCudaTensor x3485;
JCudaTensor x3486, x3487;
x3486 = x3429;
x3487 = x79;
x3485 = x81.backward(x3486,x3487);

// Dealloc(X211)
JCudaTensor x3488;
x3488 = x79;
x3488.free();

// val X21108 = X2299 * d_BatchNorm()(X209,2a2_b_bn_scale)/d_2a2_b_bn_scale
JCudaTensor x3489;
JCudaTensor x3490, x3491, x3492;
x3490 = x3485;
x3491 = x65;
x3492 = x76;
JCudaTensor[] x3493 = x78.backward(x3490,x3491,x3492);
x3489 = x3493[1];

// val X19236 = X2299 * d_BatchNorm()(X209,2a2_b_bn_scale)/d_2a2_b_bn_bias
JCudaTensor x3494;
x3494 = x3493[2];

// val X2300 = X2299 * d_BatchNorm()(X209,2a2_b_bn_scale)/d_X209
JCudaTensor x3498;
x3498 = x3493[0];

// Dealloc(X209)
JCudaTensor x3502;
x3502 = x65;
x3502.free();

// V_2a2_b_bn_bias <~~ X19236
float x3504, x3505;
x3504 = lrn_rate_1;
x3505 = momentum;
JCudaTensor x3506;
x3506 = x3494;
x3503.update(x3506, x3504, x3505);


// Dealloc(X19236)
JCudaTensor x3507;
x3507 = x3494;
x3507.free();

// val X2301 = X2300 * d_Convolv(1,1)(2a2_b_cv_W)/d_X208
JCudaTensor x3508;
JCudaTensor x3509, x3510;
x3509 = x3498;
x3510 = x69;
x3508 = x71.backward_data(x3509,x3510);

// V_2a2_b_cv_W <~~ X2300 * d_Convolv(1,1)(X208)/d_2a2_b_cv_W
float x3512, x3513;
x3512 = lrn_rate_1;
x3513 = momentum;
JCudaTensor x3514, x3515;
x3514 = x3498;
x3515 = x62;
x71.backward_filter(x3514,x3515, x3511, x3512, x3513);

// Dealloc(X2300)
JCudaTensor x3516;
x3516 = x3498;
x3516.free();

// V_2a2_b_bn_scale <~~ X21108
float x3518, x3519;
x3518 = lrn_rate_1;
x3519 = momentum;
JCudaTensor x3520;
x3520 = x3489;
x3517.update(x3520, x3518, x3519);


// Dealloc(X21108)
JCudaTensor x3521;
x3521 = x3489;
x3521.free();

// 2a2_b_bn_bias <~~ V_2a2_b_bn_bias
float x3522, x3523;
x3522 = 1;
x3523 = decay_1;
JCudaTensor x3524;
x3524 = x3503;
x77.update(x3524, x3522, x3523);


// 2a2_b_cv_W <~~ V_2a2_b_cv_W
float x3525, x3526;
x3525 = 1;
x3526 = decay_1;
JCudaTensor x3527;
x3527 = x3511;
x69.update(x3527, x3525, x3526);


// 2a2_b_bn_scale <~~ V_2a2_b_bn_scale
float x3528, x3529;
x3528 = 1;
x3529 = decay_1;
JCudaTensor x3530;
x3530 = x3517;
x76.update(x3530, x3528, x3529);


// val X2305 = X2301 * d_ReLU()(X208)/d_X207
JCudaTensor x3531;
JCudaTensor x3532, x3533;
x3532 = x3508;
x3533 = x62;
x3531 = x64.backward(x3532,x3533);

// Dealloc(X208)
JCudaTensor x3534;
x3534 = x62;
x3534.free();

// val X13592 = X2305 * d_BatchNorm()(X206,2a2_a_bn_scale)/d_2a2_a_bn_bias
JCudaTensor x3535;
JCudaTensor x3536, x3537, x3538;
x3536 = x3531;
x3537 = x41;
x3538 = x59;
JCudaTensor[] x3539 = x61.backward(x3536,x3537,x3538);
x3535 = x3539[2];

// val X2306 = X2305 * d_BatchNorm()(X206,2a2_a_bn_scale)/d_X206
JCudaTensor x3540;
x3540 = x3539[0];

// val X15477 = X2305 * d_BatchNorm()(X206,2a2_a_bn_scale)/d_2a2_a_bn_scale
JCudaTensor x3544;
x3544 = x3539[1];

// Dealloc(X206)
JCudaTensor x3548;
x3548 = x41;
x3548.free();

// val X2308 = (X2283 + X2306 * d_Convolv(1,0)(2a2_a_cv_W)/d_X202)
JCudaTensor x3549;
JCudaTensor x3550;
x3550 = x3463;
JCudaTensor x3551, x3552;
x3551 = x3540;
x3552 = x45;
x3549 = x47.backward_data(x3551,x3552, x3550);

// V_2a2_a_cv_W <~~ X2306 * d_Convolv(1,0)(X202)/d_2a2_a_cv_W
float x3554, x3555;
x3554 = lrn_rate_1;
x3555 = momentum;
JCudaTensor x3556, x3557;
x3556 = x3540;
x3557 = x31;
x47.backward_filter(x3556,x3557, x3553, x3554, x3555);

// Dealloc(X2306)
JCudaTensor x3558;
x3558 = x3540;
x3558.free();

// V_2a2_a_bn_scale <~~ X15477
float x3560, x3561;
x3560 = lrn_rate_1;
x3561 = momentum;
JCudaTensor x3562;
x3562 = x3544;
x3559.update(x3562, x3560, x3561);


// Dealloc(X15477)
JCudaTensor x3563;
x3563 = x3544;
x3563.free();

// V_2a2_a_bn_bias <~~ X13592
float x3565, x3566;
x3565 = lrn_rate_1;
x3566 = momentum;
JCudaTensor x3567;
x3567 = x3535;
x3564.update(x3567, x3565, x3566);


// Dealloc(X13592)
JCudaTensor x3568;
x3568 = x3535;
x3568.free();

// 2a2_a_cv_W <~~ V_2a2_a_cv_W
float x3569, x3570;
x3569 = 1;
x3570 = decay_1;
JCudaTensor x3571;
x3571 = x3553;
x45.update(x3571, x3569, x3570);


// 2a2_a_bn_scale <~~ V_2a2_a_bn_scale
float x3572, x3573;
x3572 = 1;
x3573 = decay_1;
JCudaTensor x3574;
x3574 = x3559;
x59.update(x3574, x3572, x3573);


// 2a2_a_bn_bias <~~ V_2a2_a_bn_bias
float x3575, x3576;
x3575 = 1;
x3576 = decay_1;
JCudaTensor x3577;
x3577 = x3564;
x60.update(x3577, x3575, x3576);


// val X2310 = X2308 * d_Pooling(3,2,0,true)(X202,X201)/d_X201
JCudaTensor x3578;
JCudaTensor x3579, x3580, x3581;
x3579 = x3549;
x3580 = x31;
x3581 = x28;
x3578 = x33.backward(x3579,x3580,x3581);

// Dealloc(X2308)
JCudaTensor x3582;
x3582 = x3549;
x3582.free();

// Dealloc(X202)
JCudaTensor x3583;
x3583 = x31;
x3583.free();

// val X2314 = X2310 * d_ReLU()(X201)/d_X200
JCudaTensor x3584;
JCudaTensor x3585, x3586;
x3585 = x3578;
x3586 = x28;
x3584 = x30.backward(x3585,x3586);

// Dealloc(X201)
JCudaTensor x3587;
x3587 = x28;
x3587.free();

// val X4234 = X2314 * d_BatchNorm()(X199,1_bn_scale)/d_1_bn_scale
JCudaTensor x3588;
JCudaTensor x3589, x3590, x3591;
x3589 = x3584;
x3590 = x14;
x3591 = x25;
JCudaTensor[] x3592 = x27.backward(x3589,x3590,x3591);
x3588 = x3592[1];

// val X2315 = X2314 * d_BatchNorm()(X199,1_bn_scale)/d_1_bn_bias
JCudaTensor x3593;
x3593 = x3592[2];

// val X6154 = X2314 * d_BatchNorm()(X199,1_bn_scale)/d_X199
JCudaTensor x3597;
x3597 = x3592[0];

// Dealloc(X199)
JCudaTensor x3601;
x3601 = x14;
x3601.free();

// V_1_bn_bias <~~ X2315
float x3603, x3604;
x3603 = lrn_rate_1;
x3604 = momentum;
JCudaTensor x3605;
x3605 = x3593;
x3602.update(x3605, x3603, x3604);


// Dealloc(X2315)
JCudaTensor x3606;
x3606 = x3593;
x3606.free();

// V_1_bn_scale <~~ X4234
float x3608, x3609;
x3608 = lrn_rate_1;
x3609 = momentum;
JCudaTensor x3610;
x3610 = x3588;
x3607.update(x3610, x3608, x3609);


// Dealloc(X4234)
JCudaTensor x3611;
x3611 = x3588;
x3611.free();

// V_1_cv_W <~~ X6154 * d_Convolv(2,3)(X198)/d_1_cv_W
float x3613, x3614;
x3613 = lrn_rate_1;
x3614 = momentum;
JCudaTensor x3615, x3616;
x3615 = x3597;
x3616 = x9;
x20.backward_filter(x3615,x3616, x3612, x3613, x3614);

// Dealloc(X6154)
JCudaTensor x3617;
x3617 = x3597;
x3617.free();

// Dealloc(X198)
JCudaTensor x3618;
x3618 = x9;
x3618.free();

// 1_bn_bias <~~ V_1_bn_bias
float x3619, x3620;
x3619 = 1;
x3620 = decay_1;
JCudaTensor x3621;
x3621 = x3602;
x26.update(x3621, x3619, x3620);


// 1_bn_scale <~~ V_1_bn_scale
float x3622, x3623;
x3622 = 1;
x3623 = decay_1;
JCudaTensor x3624;
x3624 = x3607;
x25.update(x3624, x3622, x3623);


// 1_cv_W <~~ V_1_cv_W
float x3625, x3626;
x3625 = 1;
x3626 = decay_1;
JCudaTensor x3627;
x3627 = x3612;
x18.update(x3627, x3625, x3626);


}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X209117 = Cuda(X)
JCudaTensor x3628;
JTensorFloat x3629;
x3629 = x3;
x3628 = x3629.asJCudaTensor();

// val X209118 = Convolv(2,3)(X209117,1_cv_W,1_cv_B)
JCudaTensor x3630;
JCudaTensor x3631, x3632, x3633;
x3631 = x3628;
x3632 = x18;
x3633 = x19;
x3630 = x20.forward(x3631,x3632,x3633);

// Dealloc(X209117)
JCudaTensor x3634;
x3634 = x3628;
x3634.free();

// val X209119 = BatchNorm()(X209118,1_bn_scale,1_bn_bias)
JCudaTensor x3635;
JCudaTensor x3636, x3637, x3638;
x3636 = x3630;
x3637 = x25;
x3638 = x26;
x3635 = x27.forward_inference(x3636,x3637,x3638);

// Dealloc(X209118)
JCudaTensor x3639;
x3639 = x3630;
x3639.free();

// val X209120 = ReLU()(X209119)
JCudaTensor x3640;
JCudaTensor x3641;
x3641 = x3635;
x3640 = x30.forward(x3641);

// val X209121 = Pooling(3,2,0,true)(X209120)
JCudaTensor x3642;
JCudaTensor x3643;
x3643 = x3640;
x3642 = x33.forward(x3643);

// Dealloc(X209120)
JCudaTensor x3644;
x3644 = x3640;
x3644.free();

// val X209125 = Convolv(1,0)(X209121,2a2_a_cv_W,2a2_a_cv_B)
JCudaTensor x3645;
JCudaTensor x3646, x3647, x3648;
x3646 = x3642;
x3647 = x45;
x3648 = x46;
x3645 = x47.forward(x3646,x3647,x3648);

// val X209122 = Convolv(1,0)(X209121,2a1_cv_W,2a1_cv_B)
JCudaTensor x3649;
JCudaTensor x3650, x3651, x3652;
x3650 = x3642;
x3651 = x38;
x3652 = x39;
x3649 = x40.forward(x3650,x3651,x3652);

// Dealloc(X209121)
JCudaTensor x3653;
x3653 = x3642;
x3653.free();

// val X209123 = BatchNorm()(X209122,2a1_bn_scale,2a1_bn_bias)
JCudaTensor x3654;
JCudaTensor x3655, x3656, x3657;
x3655 = x3649;
x3656 = x52;
x3657 = x53;
x3654 = x54.forward_inference(x3655,x3656,x3657);

// Dealloc(X209122)
JCudaTensor x3658;
x3658 = x3649;
x3658.free();

// val X209126 = BatchNorm()(X209125,2a2_a_bn_scale,2a2_a_bn_bias)
JCudaTensor x3659;
JCudaTensor x3660, x3661, x3662;
x3660 = x3645;
x3661 = x59;
x3662 = x60;
x3659 = x61.forward_inference(x3660,x3661,x3662);

// Dealloc(X209125)
JCudaTensor x3663;
x3663 = x3645;
x3663.free();

// val X209127 = ReLU()(X209126)
JCudaTensor x3664;
JCudaTensor x3665;
x3665 = x3659;
x3664 = x64.forward(x3665);

// val X209128 = Convolv(1,1)(X209127,2a2_b_cv_W,2a2_b_cv_B)
JCudaTensor x3666;
JCudaTensor x3667, x3668, x3669;
x3667 = x3664;
x3668 = x69;
x3669 = x70;
x3666 = x71.forward(x3667,x3668,x3669);

// Dealloc(X209127)
JCudaTensor x3670;
x3670 = x3664;
x3670.free();

// val X209129 = BatchNorm()(X209128,2a2_b_bn_scale,2a2_b_bn_bias)
JCudaTensor x3671;
JCudaTensor x3672, x3673, x3674;
x3672 = x3666;
x3673 = x76;
x3674 = x77;
x3671 = x78.forward_inference(x3672,x3673,x3674);

// Dealloc(X209128)
JCudaTensor x3675;
x3675 = x3666;
x3675.free();

// val X209130 = ReLU()(X209129)
JCudaTensor x3676;
JCudaTensor x3677;
x3677 = x3671;
x3676 = x81.forward(x3677);

// val X209131 = Convolv(1,0)(X209130,2a2_c_cv_W,2a2_c_cv_B)
JCudaTensor x3678;
JCudaTensor x3679, x3680, x3681;
x3679 = x3676;
x3680 = x86;
x3681 = x87;
x3678 = x88.forward(x3679,x3680,x3681);

// Dealloc(X209130)
JCudaTensor x3682;
x3682 = x3676;
x3682.free();

// val X209132 = BatchNorm()(X209131,2a2_c_bn_scale,2a2_c_bn_bias)
JCudaTensor x3683;
JCudaTensor x3684, x3685, x3686;
x3684 = x3678;
x3685 = x93;
x3686 = x94;
x3683 = x95.forward_inference(x3684,x3685,x3686);

// Dealloc(X209131)
JCudaTensor x3687;
x3687 = x3678;
x3687.free();

// val X209124 = ReLU()(X209123)
JCudaTensor x3688;
JCudaTensor x3689;
x3689 = x3654;
x3688 = x98.forward(x3689);

// val X209133 = ReLU()(X209132)
JCudaTensor x3690;
JCudaTensor x3691;
x3691 = x3683;
x3690 = x101.forward(x3691);

// val X209134 = (X209124 + X209133)
JCudaTensor x3692;
JCudaTensor x3693, x3694;
x3693 = x3688;
x3694 = x3690;
x3692 = x3693.plus_i(x3694);

// Dealloc(X209133)
JCudaTensor x3695;
x3695 = x3690;
x3695.free();

// val X209135 = ReLU()(X209134)
JCudaTensor x3696;
JCudaTensor x3697;
x3697 = x3692;
x3696 = x98.forward(x3697);

// val X209136 = Convolv(1,0)(X209135,2b_a_cv_W,2b_a_cv_B)
JCudaTensor x3698;
JCudaTensor x3699, x3700, x3701;
x3699 = x3696;
x3700 = x111;
x3701 = x112;
x3698 = x113.forward(x3699,x3700,x3701);

// val X209137 = BatchNorm()(X209136,2b_a_bn_scale,2b_a_bn_bias)
JCudaTensor x3702;
JCudaTensor x3703, x3704, x3705;
x3703 = x3698;
x3704 = x118;
x3705 = x119;
x3702 = x120.forward_inference(x3703,x3704,x3705);

// Dealloc(X209136)
JCudaTensor x3706;
x3706 = x3698;
x3706.free();

// val X209138 = ReLU()(X209137)
JCudaTensor x3707;
JCudaTensor x3708;
x3708 = x3702;
x3707 = x123.forward(x3708);

// val X209139 = Convolv(1,1)(X209138,2b_b_cv_W,2b_b_cv_B)
JCudaTensor x3709;
JCudaTensor x3710, x3711, x3712;
x3710 = x3707;
x3711 = x128;
x3712 = x129;
x3709 = x130.forward(x3710,x3711,x3712);

// Dealloc(X209138)
JCudaTensor x3713;
x3713 = x3707;
x3713.free();

// val X209140 = BatchNorm()(X209139,2b_b_bn_scale,2b_b_bn_bias)
JCudaTensor x3714;
JCudaTensor x3715, x3716, x3717;
x3715 = x3709;
x3716 = x135;
x3717 = x136;
x3714 = x137.forward_inference(x3715,x3716,x3717);

// Dealloc(X209139)
JCudaTensor x3718;
x3718 = x3709;
x3718.free();

// val X209141 = ReLU()(X209140)
JCudaTensor x3719;
JCudaTensor x3720;
x3720 = x3714;
x3719 = x140.forward(x3720);

// val X209142 = Convolv(1,0)(X209141,2b_c_cv_W,2b_c_cv_B)
JCudaTensor x3721;
JCudaTensor x3722, x3723, x3724;
x3722 = x3719;
x3723 = x145;
x3724 = x146;
x3721 = x147.forward(x3722,x3723,x3724);

// Dealloc(X209141)
JCudaTensor x3725;
x3725 = x3719;
x3725.free();

// val X209143 = BatchNorm()(X209142,2b_c_bn_scale,2b_c_bn_bias)
JCudaTensor x3726;
JCudaTensor x3727, x3728, x3729;
x3727 = x3721;
x3728 = x152;
x3729 = x153;
x3726 = x154.forward_inference(x3727,x3728,x3729);

// Dealloc(X209142)
JCudaTensor x3730;
x3730 = x3721;
x3730.free();

// val X209144 = ReLU()(X209143)
JCudaTensor x3731;
JCudaTensor x3732;
x3732 = x3726;
x3731 = x157.forward(x3732);

// val X209145 = (X209144 + X209135)
JCudaTensor x3733;
JCudaTensor x3734, x3735;
x3734 = x3731;
x3735 = x3696;
x3733 = x3734.plus_i(x3735);

// Dealloc(X209135)
JCudaTensor x3736;
x3736 = x3696;
x3736.free();

// val X209146 = ReLU()(X209145)
JCudaTensor x3737;
JCudaTensor x3738;
x3738 = x3733;
x3737 = x157.forward(x3738);

// val X209147 = Convolv(1,0)(X209146,2c_a_cv_W,2c_a_cv_B)
JCudaTensor x3739;
JCudaTensor x3740, x3741, x3742;
x3740 = x3737;
x3741 = x167;
x3742 = x168;
x3739 = x169.forward(x3740,x3741,x3742);

// val X209148 = BatchNorm()(X209147,2c_a_bn_scale,2c_a_bn_bias)
JCudaTensor x3743;
JCudaTensor x3744, x3745, x3746;
x3744 = x3739;
x3745 = x174;
x3746 = x175;
x3743 = x176.forward_inference(x3744,x3745,x3746);

// Dealloc(X209147)
JCudaTensor x3747;
x3747 = x3739;
x3747.free();

// val X209149 = ReLU()(X209148)
JCudaTensor x3748;
JCudaTensor x3749;
x3749 = x3743;
x3748 = x179.forward(x3749);

// val X209150 = Convolv(1,1)(X209149,2c_b_cv_W,2c_b_cv_B)
JCudaTensor x3750;
JCudaTensor x3751, x3752, x3753;
x3751 = x3748;
x3752 = x184;
x3753 = x185;
x3750 = x186.forward(x3751,x3752,x3753);

// Dealloc(X209149)
JCudaTensor x3754;
x3754 = x3748;
x3754.free();

// val X209151 = BatchNorm()(X209150,2c_b_bn_scale,2c_b_bn_bias)
JCudaTensor x3755;
JCudaTensor x3756, x3757, x3758;
x3756 = x3750;
x3757 = x191;
x3758 = x192;
x3755 = x193.forward_inference(x3756,x3757,x3758);

// Dealloc(X209150)
JCudaTensor x3759;
x3759 = x3750;
x3759.free();

// val X209152 = ReLU()(X209151)
JCudaTensor x3760;
JCudaTensor x3761;
x3761 = x3755;
x3760 = x196.forward(x3761);

// val X209153 = Convolv(1,0)(X209152,2c_c_cv_W,2c_c_cv_B)
JCudaTensor x3762;
JCudaTensor x3763, x3764, x3765;
x3763 = x3760;
x3764 = x201;
x3765 = x202;
x3762 = x203.forward(x3763,x3764,x3765);

// Dealloc(X209152)
JCudaTensor x3766;
x3766 = x3760;
x3766.free();

// val X209154 = BatchNorm()(X209153,2c_c_bn_scale,2c_c_bn_bias)
JCudaTensor x3767;
JCudaTensor x3768, x3769, x3770;
x3768 = x3762;
x3769 = x208;
x3770 = x209;
x3767 = x210.forward_inference(x3768,x3769,x3770);

// Dealloc(X209153)
JCudaTensor x3771;
x3771 = x3762;
x3771.free();

// val X209155 = ReLU()(X209154)
JCudaTensor x3772;
JCudaTensor x3773;
x3773 = x3767;
x3772 = x213.forward(x3773);

// val X209156 = (X209155 + X209146)
JCudaTensor x3774;
JCudaTensor x3775, x3776;
x3775 = x3772;
x3776 = x3737;
x3774 = x3775.plus_i(x3776);

// Dealloc(X209146)
JCudaTensor x3777;
x3777 = x3737;
x3777.free();

// val X209157 = ReLU()(X209156)
JCudaTensor x3778;
JCudaTensor x3779;
x3779 = x3774;
x3778 = x213.forward(x3779);

// val X209161 = Convolv(2,0)(X209157,3a2_a_cv_W,3a2_a_cv_B)
JCudaTensor x3780;
JCudaTensor x3781, x3782, x3783;
x3781 = x3778;
x3782 = x223;
x3783 = x224;
x3780 = x225.forward(x3781,x3782,x3783);

// val X209158 = Convolv(2,0)(X209157,3a1_cv_W,3a1_cv_B)
JCudaTensor x3784;
JCudaTensor x3785, x3786, x3787;
x3785 = x3778;
x3786 = x230;
x3787 = x231;
x3784 = x232.forward(x3785,x3786,x3787);

// Dealloc(X209157)
JCudaTensor x3788;
x3788 = x3778;
x3788.free();

// val X209162 = BatchNorm()(X209161,3a2_a_bn_scale,3a2_a_bn_bias)
JCudaTensor x3789;
JCudaTensor x3790, x3791, x3792;
x3790 = x3780;
x3791 = x244;
x3792 = x245;
x3789 = x246.forward_inference(x3790,x3791,x3792);

// Dealloc(X209161)
JCudaTensor x3793;
x3793 = x3780;
x3793.free();

// val X209159 = BatchNorm()(X209158,3a1_bn_scale,3a1_bn_bias)
JCudaTensor x3794;
JCudaTensor x3795, x3796, x3797;
x3795 = x3784;
x3796 = x237;
x3797 = x238;
x3794 = x239.forward_inference(x3795,x3796,x3797);

// Dealloc(X209158)
JCudaTensor x3798;
x3798 = x3784;
x3798.free();

// val X209163 = ReLU()(X209162)
JCudaTensor x3799;
JCudaTensor x3800;
x3800 = x3789;
x3799 = x249.forward(x3800);

// val X209164 = Convolv(1,1)(X209163,3a2_b_cv_W,3a2_b_cv_B)
JCudaTensor x3801;
JCudaTensor x3802, x3803, x3804;
x3802 = x3799;
x3803 = x254;
x3804 = x255;
x3801 = x256.forward(x3802,x3803,x3804);

// Dealloc(X209163)
JCudaTensor x3805;
x3805 = x3799;
x3805.free();

// val X209165 = BatchNorm()(X209164,3a2_b_bn_scale,3a2_b_bn_bias)
JCudaTensor x3806;
JCudaTensor x3807, x3808, x3809;
x3807 = x3801;
x3808 = x261;
x3809 = x262;
x3806 = x263.forward_inference(x3807,x3808,x3809);

// Dealloc(X209164)
JCudaTensor x3810;
x3810 = x3801;
x3810.free();

// val X209166 = ReLU()(X209165)
JCudaTensor x3811;
JCudaTensor x3812;
x3812 = x3806;
x3811 = x266.forward(x3812);

// val X209167 = Convolv(1,0)(X209166,3a2_c_cv_W,3a2_c_cv_B)
JCudaTensor x3813;
JCudaTensor x3814, x3815, x3816;
x3814 = x3811;
x3815 = x271;
x3816 = x272;
x3813 = x273.forward(x3814,x3815,x3816);

// Dealloc(X209166)
JCudaTensor x3817;
x3817 = x3811;
x3817.free();

// val X209168 = BatchNorm()(X209167,3a2_c_bn_scale,3a2_c_bn_bias)
JCudaTensor x3818;
JCudaTensor x3819, x3820, x3821;
x3819 = x3813;
x3820 = x278;
x3821 = x279;
x3818 = x280.forward_inference(x3819,x3820,x3821);

// Dealloc(X209167)
JCudaTensor x3822;
x3822 = x3813;
x3822.free();

// val X209160 = ReLU()(X209159)
JCudaTensor x3823;
JCudaTensor x3824;
x3824 = x3794;
x3823 = x283.forward(x3824);

// val X209169 = ReLU()(X209168)
JCudaTensor x3825;
JCudaTensor x3826;
x3826 = x3818;
x3825 = x286.forward(x3826);

// val X209170 = (X209160 + X209169)
JCudaTensor x3827;
JCudaTensor x3828, x3829;
x3828 = x3823;
x3829 = x3825;
x3827 = x3828.plus_i(x3829);

// Dealloc(X209169)
JCudaTensor x3830;
x3830 = x3825;
x3830.free();

// val X209171 = ReLU()(X209170)
JCudaTensor x3831;
JCudaTensor x3832;
x3832 = x3827;
x3831 = x283.forward(x3832);

// val X209172 = Convolv(1,0)(X209171,3b_a_cv_W,3b_a_cv_B)
JCudaTensor x3833;
JCudaTensor x3834, x3835, x3836;
x3834 = x3831;
x3835 = x296;
x3836 = x297;
x3833 = x298.forward(x3834,x3835,x3836);

// val X209173 = BatchNorm()(X209172,3b_a_bn_scale,3b_a_bn_bias)
JCudaTensor x3837;
JCudaTensor x3838, x3839, x3840;
x3838 = x3833;
x3839 = x303;
x3840 = x304;
x3837 = x305.forward_inference(x3838,x3839,x3840);

// Dealloc(X209172)
JCudaTensor x3841;
x3841 = x3833;
x3841.free();

// val X209174 = ReLU()(X209173)
JCudaTensor x3842;
JCudaTensor x3843;
x3843 = x3837;
x3842 = x308.forward(x3843);

// val X209175 = Convolv(1,1)(X209174,3b_b_cv_W,3b_b_cv_B)
JCudaTensor x3844;
JCudaTensor x3845, x3846, x3847;
x3845 = x3842;
x3846 = x313;
x3847 = x314;
x3844 = x315.forward(x3845,x3846,x3847);

// Dealloc(X209174)
JCudaTensor x3848;
x3848 = x3842;
x3848.free();

// val X209176 = BatchNorm()(X209175,3b_b_bn_scale,3b_b_bn_bias)
JCudaTensor x3849;
JCudaTensor x3850, x3851, x3852;
x3850 = x3844;
x3851 = x320;
x3852 = x321;
x3849 = x322.forward_inference(x3850,x3851,x3852);

// Dealloc(X209175)
JCudaTensor x3853;
x3853 = x3844;
x3853.free();

// val X209177 = ReLU()(X209176)
JCudaTensor x3854;
JCudaTensor x3855;
x3855 = x3849;
x3854 = x325.forward(x3855);

// val X209178 = Convolv(1,0)(X209177,3b_c_cv_W,3b_c_cv_B)
JCudaTensor x3856;
JCudaTensor x3857, x3858, x3859;
x3857 = x3854;
x3858 = x330;
x3859 = x331;
x3856 = x332.forward(x3857,x3858,x3859);

// Dealloc(X209177)
JCudaTensor x3860;
x3860 = x3854;
x3860.free();

// val X209179 = BatchNorm()(X209178,3b_c_bn_scale,3b_c_bn_bias)
JCudaTensor x3861;
JCudaTensor x3862, x3863, x3864;
x3862 = x3856;
x3863 = x337;
x3864 = x338;
x3861 = x339.forward_inference(x3862,x3863,x3864);

// Dealloc(X209178)
JCudaTensor x3865;
x3865 = x3856;
x3865.free();

// val X209180 = ReLU()(X209179)
JCudaTensor x3866;
JCudaTensor x3867;
x3867 = x3861;
x3866 = x342.forward(x3867);

// val X209181 = (X209180 + X209171)
JCudaTensor x3868;
JCudaTensor x3869, x3870;
x3869 = x3866;
x3870 = x3831;
x3868 = x3869.plus_i(x3870);

// Dealloc(X209171)
JCudaTensor x3871;
x3871 = x3831;
x3871.free();

// val X209182 = ReLU()(X209181)
JCudaTensor x3872;
JCudaTensor x3873;
x3873 = x3868;
x3872 = x342.forward(x3873);

// val X209183 = Convolv(1,0)(X209182,3c_a_cv_W,3c_a_cv_B)
JCudaTensor x3874;
JCudaTensor x3875, x3876, x3877;
x3875 = x3872;
x3876 = x352;
x3877 = x353;
x3874 = x354.forward(x3875,x3876,x3877);

// val X209184 = BatchNorm()(X209183,3c_a_bn_scale,3c_a_bn_bias)
JCudaTensor x3878;
JCudaTensor x3879, x3880, x3881;
x3879 = x3874;
x3880 = x359;
x3881 = x360;
x3878 = x361.forward_inference(x3879,x3880,x3881);

// Dealloc(X209183)
JCudaTensor x3882;
x3882 = x3874;
x3882.free();

// val X209185 = ReLU()(X209184)
JCudaTensor x3883;
JCudaTensor x3884;
x3884 = x3878;
x3883 = x364.forward(x3884);

// val X209186 = Convolv(1,1)(X209185,3c_b_cv_W,3c_b_cv_B)
JCudaTensor x3885;
JCudaTensor x3886, x3887, x3888;
x3886 = x3883;
x3887 = x369;
x3888 = x370;
x3885 = x371.forward(x3886,x3887,x3888);

// Dealloc(X209185)
JCudaTensor x3889;
x3889 = x3883;
x3889.free();

// val X209187 = BatchNorm()(X209186,3c_b_bn_scale,3c_b_bn_bias)
JCudaTensor x3890;
JCudaTensor x3891, x3892, x3893;
x3891 = x3885;
x3892 = x376;
x3893 = x377;
x3890 = x378.forward_inference(x3891,x3892,x3893);

// Dealloc(X209186)
JCudaTensor x3894;
x3894 = x3885;
x3894.free();

// val X209188 = ReLU()(X209187)
JCudaTensor x3895;
JCudaTensor x3896;
x3896 = x3890;
x3895 = x381.forward(x3896);

// val X209189 = Convolv(1,0)(X209188,3c_c_cv_W,3c_c_cv_B)
JCudaTensor x3897;
JCudaTensor x3898, x3899, x3900;
x3898 = x3895;
x3899 = x386;
x3900 = x387;
x3897 = x388.forward(x3898,x3899,x3900);

// Dealloc(X209188)
JCudaTensor x3901;
x3901 = x3895;
x3901.free();

// val X209190 = BatchNorm()(X209189,3c_c_bn_scale,3c_c_bn_bias)
JCudaTensor x3902;
JCudaTensor x3903, x3904, x3905;
x3903 = x3897;
x3904 = x393;
x3905 = x394;
x3902 = x395.forward_inference(x3903,x3904,x3905);

// Dealloc(X209189)
JCudaTensor x3906;
x3906 = x3897;
x3906.free();

// val X209191 = ReLU()(X209190)
JCudaTensor x3907;
JCudaTensor x3908;
x3908 = x3902;
x3907 = x398.forward(x3908);

// val X209192 = (X209191 + X209182)
JCudaTensor x3909;
JCudaTensor x3910, x3911;
x3910 = x3907;
x3911 = x3872;
x3909 = x3910.plus_i(x3911);

// Dealloc(X209182)
JCudaTensor x3912;
x3912 = x3872;
x3912.free();

// val X209193 = ReLU()(X209192)
JCudaTensor x3913;
JCudaTensor x3914;
x3914 = x3909;
x3913 = x398.forward(x3914);

// val X209194 = Convolv(1,0)(X209193,3d_a_cv_W,3d_a_cv_B)
JCudaTensor x3915;
JCudaTensor x3916, x3917, x3918;
x3916 = x3913;
x3917 = x408;
x3918 = x409;
x3915 = x410.forward(x3916,x3917,x3918);

// val X209195 = BatchNorm()(X209194,3d_a_bn_scale,3d_a_bn_bias)
JCudaTensor x3919;
JCudaTensor x3920, x3921, x3922;
x3920 = x3915;
x3921 = x415;
x3922 = x416;
x3919 = x417.forward_inference(x3920,x3921,x3922);

// Dealloc(X209194)
JCudaTensor x3923;
x3923 = x3915;
x3923.free();

// val X209196 = ReLU()(X209195)
JCudaTensor x3924;
JCudaTensor x3925;
x3925 = x3919;
x3924 = x420.forward(x3925);

// val X209197 = Convolv(1,1)(X209196,3d_b_cv_W,3d_b_cv_B)
JCudaTensor x3926;
JCudaTensor x3927, x3928, x3929;
x3927 = x3924;
x3928 = x425;
x3929 = x426;
x3926 = x427.forward(x3927,x3928,x3929);

// Dealloc(X209196)
JCudaTensor x3930;
x3930 = x3924;
x3930.free();

// val X209198 = BatchNorm()(X209197,3d_b_bn_scale,3d_b_bn_bias)
JCudaTensor x3931;
JCudaTensor x3932, x3933, x3934;
x3932 = x3926;
x3933 = x432;
x3934 = x433;
x3931 = x434.forward_inference(x3932,x3933,x3934);

// Dealloc(X209197)
JCudaTensor x3935;
x3935 = x3926;
x3935.free();

// val X209199 = ReLU()(X209198)
JCudaTensor x3936;
JCudaTensor x3937;
x3937 = x3931;
x3936 = x437.forward(x3937);

// val X209200 = Convolv(1,0)(X209199,3d_c_cv_W,3d_c_cv_B)
JCudaTensor x3938;
JCudaTensor x3939, x3940, x3941;
x3939 = x3936;
x3940 = x442;
x3941 = x443;
x3938 = x444.forward(x3939,x3940,x3941);

// Dealloc(X209199)
JCudaTensor x3942;
x3942 = x3936;
x3942.free();

// val X209201 = BatchNorm()(X209200,3d_c_bn_scale,3d_c_bn_bias)
JCudaTensor x3943;
JCudaTensor x3944, x3945, x3946;
x3944 = x3938;
x3945 = x449;
x3946 = x450;
x3943 = x451.forward_inference(x3944,x3945,x3946);

// Dealloc(X209200)
JCudaTensor x3947;
x3947 = x3938;
x3947.free();

// val X209202 = ReLU()(X209201)
JCudaTensor x3948;
JCudaTensor x3949;
x3949 = x3943;
x3948 = x454.forward(x3949);

// val X209203 = (X209202 + X209193)
JCudaTensor x3950;
JCudaTensor x3951, x3952;
x3951 = x3948;
x3952 = x3913;
x3950 = x3951.plus_i(x3952);

// Dealloc(X209193)
JCudaTensor x3953;
x3953 = x3913;
x3953.free();

// val X209204 = ReLU()(X209203)
JCudaTensor x3954;
JCudaTensor x3955;
x3955 = x3950;
x3954 = x454.forward(x3955);

// val X209205 = Convolv(2,0)(X209204,4a1_cv_W,4a1_cv_B)
JCudaTensor x3956;
JCudaTensor x3957, x3958, x3959;
x3957 = x3954;
x3958 = x464;
x3959 = x465;
x3956 = x466.forward(x3957,x3958,x3959);

// val X209208 = Convolv(2,0)(X209204,4a2_a_cv_W,4a2_a_cv_B)
JCudaTensor x3960;
JCudaTensor x3961, x3962, x3963;
x3961 = x3954;
x3962 = x471;
x3963 = x472;
x3960 = x473.forward(x3961,x3962,x3963);

// Dealloc(X209204)
JCudaTensor x3964;
x3964 = x3954;
x3964.free();

// val X209206 = BatchNorm()(X209205,4a1_bn_scale,4a1_bn_bias)
JCudaTensor x3965;
JCudaTensor x3966, x3967, x3968;
x3966 = x3956;
x3967 = x478;
x3968 = x479;
x3965 = x480.forward_inference(x3966,x3967,x3968);

// Dealloc(X209205)
JCudaTensor x3969;
x3969 = x3956;
x3969.free();

// val X209209 = BatchNorm()(X209208,4a2_a_bn_scale,4a2_a_bn_bias)
JCudaTensor x3970;
JCudaTensor x3971, x3972, x3973;
x3971 = x3960;
x3972 = x485;
x3973 = x486;
x3970 = x487.forward_inference(x3971,x3972,x3973);

// Dealloc(X209208)
JCudaTensor x3974;
x3974 = x3960;
x3974.free();

// val X209210 = ReLU()(X209209)
JCudaTensor x3975;
JCudaTensor x3976;
x3976 = x3970;
x3975 = x490.forward(x3976);

// val X209211 = Convolv(1,1)(X209210,4a2_b_cv_W,4a2_b_cv_B)
JCudaTensor x3977;
JCudaTensor x3978, x3979, x3980;
x3978 = x3975;
x3979 = x495;
x3980 = x496;
x3977 = x497.forward(x3978,x3979,x3980);

// Dealloc(X209210)
JCudaTensor x3981;
x3981 = x3975;
x3981.free();

// val X209212 = BatchNorm()(X209211,4a2_b_bn_scale,4a2_b_bn_bias)
JCudaTensor x3982;
JCudaTensor x3983, x3984, x3985;
x3983 = x3977;
x3984 = x502;
x3985 = x503;
x3982 = x504.forward_inference(x3983,x3984,x3985);

// Dealloc(X209211)
JCudaTensor x3986;
x3986 = x3977;
x3986.free();

// val X209213 = ReLU()(X209212)
JCudaTensor x3987;
JCudaTensor x3988;
x3988 = x3982;
x3987 = x507.forward(x3988);

// val X209214 = Convolv(1,0)(X209213,4a2_c_cv_W,4a2_c_cv_B)
JCudaTensor x3989;
JCudaTensor x3990, x3991, x3992;
x3990 = x3987;
x3991 = x512;
x3992 = x513;
x3989 = x514.forward(x3990,x3991,x3992);

// Dealloc(X209213)
JCudaTensor x3993;
x3993 = x3987;
x3993.free();

// val X209215 = BatchNorm()(X209214,4a2_c_bn_scale,4a2_c_bn_bias)
JCudaTensor x3994;
JCudaTensor x3995, x3996, x3997;
x3995 = x3989;
x3996 = x519;
x3997 = x520;
x3994 = x521.forward_inference(x3995,x3996,x3997);

// Dealloc(X209214)
JCudaTensor x3998;
x3998 = x3989;
x3998.free();

// val X209207 = ReLU()(X209206)
JCudaTensor x3999;
JCudaTensor x4000;
x4000 = x3965;
x3999 = x524.forward(x4000);

// val X209216 = ReLU()(X209215)
JCudaTensor x4001;
JCudaTensor x4002;
x4002 = x3994;
x4001 = x527.forward(x4002);

// val X209217 = (X209207 + X209216)
JCudaTensor x4003;
JCudaTensor x4004, x4005;
x4004 = x3999;
x4005 = x4001;
x4003 = x4004.plus_i(x4005);

// Dealloc(X209216)
JCudaTensor x4006;
x4006 = x4001;
x4006.free();

// val X209218 = ReLU()(X209217)
JCudaTensor x4007;
JCudaTensor x4008;
x4008 = x4003;
x4007 = x524.forward(x4008);

// val X209219 = Convolv(1,0)(X209218,4b_a_cv_W,4b_a_cv_B)
JCudaTensor x4009;
JCudaTensor x4010, x4011, x4012;
x4010 = x4007;
x4011 = x537;
x4012 = x538;
x4009 = x539.forward(x4010,x4011,x4012);

// val X209220 = BatchNorm()(X209219,4b_a_bn_scale,4b_a_bn_bias)
JCudaTensor x4013;
JCudaTensor x4014, x4015, x4016;
x4014 = x4009;
x4015 = x544;
x4016 = x545;
x4013 = x546.forward_inference(x4014,x4015,x4016);

// Dealloc(X209219)
JCudaTensor x4017;
x4017 = x4009;
x4017.free();

// val X209221 = ReLU()(X209220)
JCudaTensor x4018;
JCudaTensor x4019;
x4019 = x4013;
x4018 = x549.forward(x4019);

// val X209222 = Convolv(1,1)(X209221,4b_b_cv_W,4b_b_cv_B)
JCudaTensor x4020;
JCudaTensor x4021, x4022, x4023;
x4021 = x4018;
x4022 = x554;
x4023 = x555;
x4020 = x556.forward(x4021,x4022,x4023);

// Dealloc(X209221)
JCudaTensor x4024;
x4024 = x4018;
x4024.free();

// val X209223 = BatchNorm()(X209222,4b_b_bn_scale,4b_b_bn_bias)
JCudaTensor x4025;
JCudaTensor x4026, x4027, x4028;
x4026 = x4020;
x4027 = x561;
x4028 = x562;
x4025 = x563.forward_inference(x4026,x4027,x4028);

// Dealloc(X209222)
JCudaTensor x4029;
x4029 = x4020;
x4029.free();

// val X209224 = ReLU()(X209223)
JCudaTensor x4030;
JCudaTensor x4031;
x4031 = x4025;
x4030 = x566.forward(x4031);

// val X209225 = Convolv(1,0)(X209224,4b_c_cv_W,4b_c_cv_B)
JCudaTensor x4032;
JCudaTensor x4033, x4034, x4035;
x4033 = x4030;
x4034 = x571;
x4035 = x572;
x4032 = x573.forward(x4033,x4034,x4035);

// Dealloc(X209224)
JCudaTensor x4036;
x4036 = x4030;
x4036.free();

// val X209226 = BatchNorm()(X209225,4b_c_bn_scale,4b_c_bn_bias)
JCudaTensor x4037;
JCudaTensor x4038, x4039, x4040;
x4038 = x4032;
x4039 = x578;
x4040 = x579;
x4037 = x580.forward_inference(x4038,x4039,x4040);

// Dealloc(X209225)
JCudaTensor x4041;
x4041 = x4032;
x4041.free();

// val X209227 = ReLU()(X209226)
JCudaTensor x4042;
JCudaTensor x4043;
x4043 = x4037;
x4042 = x583.forward(x4043);

// val X209228 = (X209227 + X209218)
JCudaTensor x4044;
JCudaTensor x4045, x4046;
x4045 = x4042;
x4046 = x4007;
x4044 = x4045.plus_i(x4046);

// Dealloc(X209218)
JCudaTensor x4047;
x4047 = x4007;
x4047.free();

// val X209229 = ReLU()(X209228)
JCudaTensor x4048;
JCudaTensor x4049;
x4049 = x4044;
x4048 = x583.forward(x4049);

// val X209230 = Convolv(1,0)(X209229,4c_a_cv_W,4c_a_cv_B)
JCudaTensor x4050;
JCudaTensor x4051, x4052, x4053;
x4051 = x4048;
x4052 = x593;
x4053 = x594;
x4050 = x595.forward(x4051,x4052,x4053);

// val X209231 = BatchNorm()(X209230,4c_a_bn_scale,4c_a_bn_bias)
JCudaTensor x4054;
JCudaTensor x4055, x4056, x4057;
x4055 = x4050;
x4056 = x600;
x4057 = x601;
x4054 = x602.forward_inference(x4055,x4056,x4057);

// Dealloc(X209230)
JCudaTensor x4058;
x4058 = x4050;
x4058.free();

// val X209232 = ReLU()(X209231)
JCudaTensor x4059;
JCudaTensor x4060;
x4060 = x4054;
x4059 = x605.forward(x4060);

// val X209233 = Convolv(1,1)(X209232,4c_b_cv_W,4c_b_cv_B)
JCudaTensor x4061;
JCudaTensor x4062, x4063, x4064;
x4062 = x4059;
x4063 = x610;
x4064 = x611;
x4061 = x612.forward(x4062,x4063,x4064);

// Dealloc(X209232)
JCudaTensor x4065;
x4065 = x4059;
x4065.free();

// val X209234 = BatchNorm()(X209233,4c_b_bn_scale,4c_b_bn_bias)
JCudaTensor x4066;
JCudaTensor x4067, x4068, x4069;
x4067 = x4061;
x4068 = x617;
x4069 = x618;
x4066 = x619.forward_inference(x4067,x4068,x4069);

// Dealloc(X209233)
JCudaTensor x4070;
x4070 = x4061;
x4070.free();

// val X209235 = ReLU()(X209234)
JCudaTensor x4071;
JCudaTensor x4072;
x4072 = x4066;
x4071 = x622.forward(x4072);

// val X209236 = Convolv(1,0)(X209235,4c_c_cv_W,4c_c_cv_B)
JCudaTensor x4073;
JCudaTensor x4074, x4075, x4076;
x4074 = x4071;
x4075 = x627;
x4076 = x628;
x4073 = x629.forward(x4074,x4075,x4076);

// Dealloc(X209235)
JCudaTensor x4077;
x4077 = x4071;
x4077.free();

// val X209237 = BatchNorm()(X209236,4c_c_bn_scale,4c_c_bn_bias)
JCudaTensor x4078;
JCudaTensor x4079, x4080, x4081;
x4079 = x4073;
x4080 = x634;
x4081 = x635;
x4078 = x636.forward_inference(x4079,x4080,x4081);

// Dealloc(X209236)
JCudaTensor x4082;
x4082 = x4073;
x4082.free();

// val X209238 = ReLU()(X209237)
JCudaTensor x4083;
JCudaTensor x4084;
x4084 = x4078;
x4083 = x639.forward(x4084);

// val X209239 = (X209238 + X209229)
JCudaTensor x4085;
JCudaTensor x4086, x4087;
x4086 = x4083;
x4087 = x4048;
x4085 = x4086.plus_i(x4087);

// Dealloc(X209229)
JCudaTensor x4088;
x4088 = x4048;
x4088.free();

// val X209240 = ReLU()(X209239)
JCudaTensor x4089;
JCudaTensor x4090;
x4090 = x4085;
x4089 = x639.forward(x4090);

// val X209241 = Convolv(1,0)(X209240,4d_a_cv_W,4d_a_cv_B)
JCudaTensor x4091;
JCudaTensor x4092, x4093, x4094;
x4092 = x4089;
x4093 = x649;
x4094 = x650;
x4091 = x651.forward(x4092,x4093,x4094);

// val X209242 = BatchNorm()(X209241,4d_a_bn_scale,4d_a_bn_bias)
JCudaTensor x4095;
JCudaTensor x4096, x4097, x4098;
x4096 = x4091;
x4097 = x656;
x4098 = x657;
x4095 = x658.forward_inference(x4096,x4097,x4098);

// Dealloc(X209241)
JCudaTensor x4099;
x4099 = x4091;
x4099.free();

// val X209243 = ReLU()(X209242)
JCudaTensor x4100;
JCudaTensor x4101;
x4101 = x4095;
x4100 = x661.forward(x4101);

// val X209244 = Convolv(1,1)(X209243,4d_b_cv_W,4d_b_cv_B)
JCudaTensor x4102;
JCudaTensor x4103, x4104, x4105;
x4103 = x4100;
x4104 = x666;
x4105 = x667;
x4102 = x668.forward(x4103,x4104,x4105);

// Dealloc(X209243)
JCudaTensor x4106;
x4106 = x4100;
x4106.free();

// val X209245 = BatchNorm()(X209244,4d_b_bn_scale,4d_b_bn_bias)
JCudaTensor x4107;
JCudaTensor x4108, x4109, x4110;
x4108 = x4102;
x4109 = x673;
x4110 = x674;
x4107 = x675.forward_inference(x4108,x4109,x4110);

// Dealloc(X209244)
JCudaTensor x4111;
x4111 = x4102;
x4111.free();

// val X209246 = ReLU()(X209245)
JCudaTensor x4112;
JCudaTensor x4113;
x4113 = x4107;
x4112 = x678.forward(x4113);

// val X209247 = Convolv(1,0)(X209246,4d_c_cv_W,4d_c_cv_B)
JCudaTensor x4114;
JCudaTensor x4115, x4116, x4117;
x4115 = x4112;
x4116 = x683;
x4117 = x684;
x4114 = x685.forward(x4115,x4116,x4117);

// Dealloc(X209246)
JCudaTensor x4118;
x4118 = x4112;
x4118.free();

// val X209248 = BatchNorm()(X209247,4d_c_bn_scale,4d_c_bn_bias)
JCudaTensor x4119;
JCudaTensor x4120, x4121, x4122;
x4120 = x4114;
x4121 = x690;
x4122 = x691;
x4119 = x692.forward_inference(x4120,x4121,x4122);

// Dealloc(X209247)
JCudaTensor x4123;
x4123 = x4114;
x4123.free();

// val X209249 = ReLU()(X209248)
JCudaTensor x4124;
JCudaTensor x4125;
x4125 = x4119;
x4124 = x695.forward(x4125);

// val X209250 = (X209249 + X209240)
JCudaTensor x4126;
JCudaTensor x4127, x4128;
x4127 = x4124;
x4128 = x4089;
x4126 = x4127.plus_i(x4128);

// Dealloc(X209240)
JCudaTensor x4129;
x4129 = x4089;
x4129.free();

// val X209251 = ReLU()(X209250)
JCudaTensor x4130;
JCudaTensor x4131;
x4131 = x4126;
x4130 = x695.forward(x4131);

// val X209252 = Convolv(1,0)(X209251,4e_a_cv_W,4e_a_cv_B)
JCudaTensor x4132;
JCudaTensor x4133, x4134, x4135;
x4133 = x4130;
x4134 = x705;
x4135 = x706;
x4132 = x707.forward(x4133,x4134,x4135);

// val X209253 = BatchNorm()(X209252,4e_a_bn_scale,4e_a_bn_bias)
JCudaTensor x4136;
JCudaTensor x4137, x4138, x4139;
x4137 = x4132;
x4138 = x712;
x4139 = x713;
x4136 = x714.forward_inference(x4137,x4138,x4139);

// Dealloc(X209252)
JCudaTensor x4140;
x4140 = x4132;
x4140.free();

// val X209254 = ReLU()(X209253)
JCudaTensor x4141;
JCudaTensor x4142;
x4142 = x4136;
x4141 = x717.forward(x4142);

// val X209255 = Convolv(1,1)(X209254,4e_b_cv_W,4e_b_cv_B)
JCudaTensor x4143;
JCudaTensor x4144, x4145, x4146;
x4144 = x4141;
x4145 = x722;
x4146 = x723;
x4143 = x724.forward(x4144,x4145,x4146);

// Dealloc(X209254)
JCudaTensor x4147;
x4147 = x4141;
x4147.free();

// val X209256 = BatchNorm()(X209255,4e_b_bn_scale,4e_b_bn_bias)
JCudaTensor x4148;
JCudaTensor x4149, x4150, x4151;
x4149 = x4143;
x4150 = x729;
x4151 = x730;
x4148 = x731.forward_inference(x4149,x4150,x4151);

// Dealloc(X209255)
JCudaTensor x4152;
x4152 = x4143;
x4152.free();

// val X209257 = ReLU()(X209256)
JCudaTensor x4153;
JCudaTensor x4154;
x4154 = x4148;
x4153 = x734.forward(x4154);

// val X209258 = Convolv(1,0)(X209257,4e_c_cv_W,4e_c_cv_B)
JCudaTensor x4155;
JCudaTensor x4156, x4157, x4158;
x4156 = x4153;
x4157 = x739;
x4158 = x740;
x4155 = x741.forward(x4156,x4157,x4158);

// Dealloc(X209257)
JCudaTensor x4159;
x4159 = x4153;
x4159.free();

// val X209259 = BatchNorm()(X209258,4e_c_bn_scale,4e_c_bn_bias)
JCudaTensor x4160;
JCudaTensor x4161, x4162, x4163;
x4161 = x4155;
x4162 = x746;
x4163 = x747;
x4160 = x748.forward_inference(x4161,x4162,x4163);

// Dealloc(X209258)
JCudaTensor x4164;
x4164 = x4155;
x4164.free();

// val X209260 = ReLU()(X209259)
JCudaTensor x4165;
JCudaTensor x4166;
x4166 = x4160;
x4165 = x751.forward(x4166);

// val X209261 = (X209260 + X209251)
JCudaTensor x4167;
JCudaTensor x4168, x4169;
x4168 = x4165;
x4169 = x4130;
x4167 = x4168.plus_i(x4169);

// Dealloc(X209251)
JCudaTensor x4170;
x4170 = x4130;
x4170.free();

// val X209262 = ReLU()(X209261)
JCudaTensor x4171;
JCudaTensor x4172;
x4172 = x4167;
x4171 = x751.forward(x4172);

// val X209263 = Convolv(1,0)(X209262,4f_a_cv_W,4f_a_cv_B)
JCudaTensor x4173;
JCudaTensor x4174, x4175, x4176;
x4174 = x4171;
x4175 = x761;
x4176 = x762;
x4173 = x763.forward(x4174,x4175,x4176);

// val X209264 = BatchNorm()(X209263,4f_a_bn_scale,4f_a_bn_bias)
JCudaTensor x4177;
JCudaTensor x4178, x4179, x4180;
x4178 = x4173;
x4179 = x768;
x4180 = x769;
x4177 = x770.forward_inference(x4178,x4179,x4180);

// Dealloc(X209263)
JCudaTensor x4181;
x4181 = x4173;
x4181.free();

// val X209265 = ReLU()(X209264)
JCudaTensor x4182;
JCudaTensor x4183;
x4183 = x4177;
x4182 = x773.forward(x4183);

// val X209266 = Convolv(1,1)(X209265,4f_b_cv_W,4f_b_cv_B)
JCudaTensor x4184;
JCudaTensor x4185, x4186, x4187;
x4185 = x4182;
x4186 = x778;
x4187 = x779;
x4184 = x780.forward(x4185,x4186,x4187);

// Dealloc(X209265)
JCudaTensor x4188;
x4188 = x4182;
x4188.free();

// val X209267 = BatchNorm()(X209266,4f_b_bn_scale,4f_b_bn_bias)
JCudaTensor x4189;
JCudaTensor x4190, x4191, x4192;
x4190 = x4184;
x4191 = x785;
x4192 = x786;
x4189 = x787.forward_inference(x4190,x4191,x4192);

// Dealloc(X209266)
JCudaTensor x4193;
x4193 = x4184;
x4193.free();

// val X209268 = ReLU()(X209267)
JCudaTensor x4194;
JCudaTensor x4195;
x4195 = x4189;
x4194 = x790.forward(x4195);

// val X209269 = Convolv(1,0)(X209268,4f_c_cv_W,4f_c_cv_B)
JCudaTensor x4196;
JCudaTensor x4197, x4198, x4199;
x4197 = x4194;
x4198 = x795;
x4199 = x796;
x4196 = x797.forward(x4197,x4198,x4199);

// Dealloc(X209268)
JCudaTensor x4200;
x4200 = x4194;
x4200.free();

// val X209270 = BatchNorm()(X209269,4f_c_bn_scale,4f_c_bn_bias)
JCudaTensor x4201;
JCudaTensor x4202, x4203, x4204;
x4202 = x4196;
x4203 = x802;
x4204 = x803;
x4201 = x804.forward_inference(x4202,x4203,x4204);

// Dealloc(X209269)
JCudaTensor x4205;
x4205 = x4196;
x4205.free();

// val X209271 = ReLU()(X209270)
JCudaTensor x4206;
JCudaTensor x4207;
x4207 = x4201;
x4206 = x807.forward(x4207);

// val X209272 = (X209271 + X209262)
JCudaTensor x4208;
JCudaTensor x4209, x4210;
x4209 = x4206;
x4210 = x4171;
x4208 = x4209.plus_i(x4210);

// Dealloc(X209262)
JCudaTensor x4211;
x4211 = x4171;
x4211.free();

// val X209273 = ReLU()(X209272)
JCudaTensor x4212;
JCudaTensor x4213;
x4213 = x4208;
x4212 = x807.forward(x4213);

// val X209274 = Convolv(2,0)(X209273,5a1_cv_W,5a1_cv_B)
JCudaTensor x4214;
JCudaTensor x4215, x4216, x4217;
x4215 = x4212;
x4216 = x817;
x4217 = x818;
x4214 = x819.forward(x4215,x4216,x4217);

// val X209277 = Convolv(2,0)(X209273,5a2_a_cv_W,5a2_a_cv_B)
JCudaTensor x4218;
JCudaTensor x4219, x4220, x4221;
x4219 = x4212;
x4220 = x824;
x4221 = x825;
x4218 = x826.forward(x4219,x4220,x4221);

// Dealloc(X209273)
JCudaTensor x4222;
x4222 = x4212;
x4222.free();

// val X209275 = BatchNorm()(X209274,5a1_bn_scale,5a1_bn_bias)
JCudaTensor x4223;
JCudaTensor x4224, x4225, x4226;
x4224 = x4214;
x4225 = x838;
x4226 = x839;
x4223 = x840.forward_inference(x4224,x4225,x4226);

// Dealloc(X209274)
JCudaTensor x4227;
x4227 = x4214;
x4227.free();

// val X209278 = BatchNorm()(X209277,5a2_a_bn_scale,5a2_a_bn_bias)
JCudaTensor x4228;
JCudaTensor x4229, x4230, x4231;
x4229 = x4218;
x4230 = x831;
x4231 = x832;
x4228 = x833.forward_inference(x4229,x4230,x4231);

// Dealloc(X209277)
JCudaTensor x4232;
x4232 = x4218;
x4232.free();

// val X209279 = ReLU()(X209278)
JCudaTensor x4233;
JCudaTensor x4234;
x4234 = x4228;
x4233 = x843.forward(x4234);

// val X209280 = Convolv(1,1)(X209279,5a2_b_cv_W,5a2_b_cv_B)
JCudaTensor x4235;
JCudaTensor x4236, x4237, x4238;
x4236 = x4233;
x4237 = x848;
x4238 = x849;
x4235 = x850.forward(x4236,x4237,x4238);

// Dealloc(X209279)
JCudaTensor x4239;
x4239 = x4233;
x4239.free();

// val X209281 = BatchNorm()(X209280,5a2_b_bn_scale,5a2_b_bn_bias)
JCudaTensor x4240;
JCudaTensor x4241, x4242, x4243;
x4241 = x4235;
x4242 = x855;
x4243 = x856;
x4240 = x857.forward_inference(x4241,x4242,x4243);

// Dealloc(X209280)
JCudaTensor x4244;
x4244 = x4235;
x4244.free();

// val X209282 = ReLU()(X209281)
JCudaTensor x4245;
JCudaTensor x4246;
x4246 = x4240;
x4245 = x860.forward(x4246);

// val X209283 = Convolv(1,0)(X209282,5a2_c_cv_W,5a2_c_cv_B)
JCudaTensor x4247;
JCudaTensor x4248, x4249, x4250;
x4248 = x4245;
x4249 = x865;
x4250 = x866;
x4247 = x867.forward(x4248,x4249,x4250);

// Dealloc(X209282)
JCudaTensor x4251;
x4251 = x4245;
x4251.free();

// val X209284 = BatchNorm()(X209283,5a2_c_bn_scale,5a2_c_bn_bias)
JCudaTensor x4252;
JCudaTensor x4253, x4254, x4255;
x4253 = x4247;
x4254 = x872;
x4255 = x873;
x4252 = x874.forward_inference(x4253,x4254,x4255);

// Dealloc(X209283)
JCudaTensor x4256;
x4256 = x4247;
x4256.free();

// val X209276 = ReLU()(X209275)
JCudaTensor x4257;
JCudaTensor x4258;
x4258 = x4223;
x4257 = x877.forward(x4258);

// val X209285 = ReLU()(X209284)
JCudaTensor x4259;
JCudaTensor x4260;
x4260 = x4252;
x4259 = x880.forward(x4260);

// val X209286 = (X209276 + X209285)
JCudaTensor x4261;
JCudaTensor x4262, x4263;
x4262 = x4257;
x4263 = x4259;
x4261 = x4262.plus_i(x4263);

// Dealloc(X209285)
JCudaTensor x4264;
x4264 = x4259;
x4264.free();

// val X209287 = ReLU()(X209286)
JCudaTensor x4265;
JCudaTensor x4266;
x4266 = x4261;
x4265 = x877.forward(x4266);

// val X209288 = Convolv(1,0)(X209287,5b_a_cv_W,5b_a_cv_B)
JCudaTensor x4267;
JCudaTensor x4268, x4269, x4270;
x4268 = x4265;
x4269 = x890;
x4270 = x891;
x4267 = x892.forward(x4268,x4269,x4270);

// val X209289 = BatchNorm()(X209288,5b_a_bn_scale,5b_a_bn_bias)
JCudaTensor x4271;
JCudaTensor x4272, x4273, x4274;
x4272 = x4267;
x4273 = x897;
x4274 = x898;
x4271 = x899.forward_inference(x4272,x4273,x4274);

// Dealloc(X209288)
JCudaTensor x4275;
x4275 = x4267;
x4275.free();

// val X209290 = ReLU()(X209289)
JCudaTensor x4276;
JCudaTensor x4277;
x4277 = x4271;
x4276 = x902.forward(x4277);

// val X209291 = Convolv(1,1)(X209290,5b_b_cv_W,5b_b_cv_B)
JCudaTensor x4278;
JCudaTensor x4279, x4280, x4281;
x4279 = x4276;
x4280 = x907;
x4281 = x908;
x4278 = x909.forward(x4279,x4280,x4281);

// Dealloc(X209290)
JCudaTensor x4282;
x4282 = x4276;
x4282.free();

// val X209292 = BatchNorm()(X209291,5b_b_bn_scale,5b_b_bn_bias)
JCudaTensor x4283;
JCudaTensor x4284, x4285, x4286;
x4284 = x4278;
x4285 = x914;
x4286 = x915;
x4283 = x916.forward_inference(x4284,x4285,x4286);

// Dealloc(X209291)
JCudaTensor x4287;
x4287 = x4278;
x4287.free();

// val X209293 = ReLU()(X209292)
JCudaTensor x4288;
JCudaTensor x4289;
x4289 = x4283;
x4288 = x919.forward(x4289);

// val X209294 = Convolv(1,0)(X209293,5b_c_cv_W,5b_c_cv_B)
JCudaTensor x4290;
JCudaTensor x4291, x4292, x4293;
x4291 = x4288;
x4292 = x924;
x4293 = x925;
x4290 = x926.forward(x4291,x4292,x4293);

// Dealloc(X209293)
JCudaTensor x4294;
x4294 = x4288;
x4294.free();

// val X209295 = BatchNorm()(X209294,5b_c_bn_scale,5b_c_bn_bias)
JCudaTensor x4295;
JCudaTensor x4296, x4297, x4298;
x4296 = x4290;
x4297 = x931;
x4298 = x932;
x4295 = x933.forward_inference(x4296,x4297,x4298);

// Dealloc(X209294)
JCudaTensor x4299;
x4299 = x4290;
x4299.free();

// val X209296 = ReLU()(X209295)
JCudaTensor x4300;
JCudaTensor x4301;
x4301 = x4295;
x4300 = x936.forward(x4301);

// val X209297 = (X209296 + X209287)
JCudaTensor x4302;
JCudaTensor x4303, x4304;
x4303 = x4300;
x4304 = x4265;
x4302 = x4303.plus_i(x4304);

// Dealloc(X209287)
JCudaTensor x4305;
x4305 = x4265;
x4305.free();

// val X209298 = ReLU()(X209297)
JCudaTensor x4306;
JCudaTensor x4307;
x4307 = x4302;
x4306 = x936.forward(x4307);

// val X209299 = Convolv(1,0)(X209298,5c_a_cv_W,5c_a_cv_B)
JCudaTensor x4308;
JCudaTensor x4309, x4310, x4311;
x4309 = x4306;
x4310 = x946;
x4311 = x947;
x4308 = x948.forward(x4309,x4310,x4311);

// val X209300 = BatchNorm()(X209299,5c_a_bn_scale,5c_a_bn_bias)
JCudaTensor x4312;
JCudaTensor x4313, x4314, x4315;
x4313 = x4308;
x4314 = x953;
x4315 = x954;
x4312 = x955.forward_inference(x4313,x4314,x4315);

// Dealloc(X209299)
JCudaTensor x4316;
x4316 = x4308;
x4316.free();

// val X209301 = ReLU()(X209300)
JCudaTensor x4317;
JCudaTensor x4318;
x4318 = x4312;
x4317 = x958.forward(x4318);

// val X209302 = Convolv(1,1)(X209301,5c_b_cv_W,5c_b_cv_B)
JCudaTensor x4319;
JCudaTensor x4320, x4321, x4322;
x4320 = x4317;
x4321 = x963;
x4322 = x964;
x4319 = x965.forward(x4320,x4321,x4322);

// Dealloc(X209301)
JCudaTensor x4323;
x4323 = x4317;
x4323.free();

// val X209303 = BatchNorm()(X209302,5c_b_bn_scale,5c_b_bn_bias)
JCudaTensor x4324;
JCudaTensor x4325, x4326, x4327;
x4325 = x4319;
x4326 = x970;
x4327 = x971;
x4324 = x972.forward_inference(x4325,x4326,x4327);

// Dealloc(X209302)
JCudaTensor x4328;
x4328 = x4319;
x4328.free();

// val X209304 = ReLU()(X209303)
JCudaTensor x4329;
JCudaTensor x4330;
x4330 = x4324;
x4329 = x975.forward(x4330);

// val X209305 = Convolv(1,0)(X209304,5c_c_cv_W,5c_c_cv_B)
JCudaTensor x4331;
JCudaTensor x4332, x4333, x4334;
x4332 = x4329;
x4333 = x980;
x4334 = x981;
x4331 = x982.forward(x4332,x4333,x4334);

// Dealloc(X209304)
JCudaTensor x4335;
x4335 = x4329;
x4335.free();

// val X209306 = BatchNorm()(X209305,5c_c_bn_scale,5c_c_bn_bias)
JCudaTensor x4336;
JCudaTensor x4337, x4338, x4339;
x4337 = x4331;
x4338 = x987;
x4339 = x988;
x4336 = x989.forward_inference(x4337,x4338,x4339);

// Dealloc(X209305)
JCudaTensor x4340;
x4340 = x4331;
x4340.free();

// val X209307 = ReLU()(X209306)
JCudaTensor x4341;
JCudaTensor x4342;
x4342 = x4336;
x4341 = x992.forward(x4342);

// val X209308 = (X209307 + X209298)
JCudaTensor x4343;
JCudaTensor x4344, x4345;
x4344 = x4341;
x4345 = x4306;
x4343 = x4344.plus_i(x4345);

// Dealloc(X209298)
JCudaTensor x4346;
x4346 = x4306;
x4346.free();

// val X209309 = ReLU()(X209308)
JCudaTensor x4347;
JCudaTensor x4348;
x4348 = x4343;
x4347 = x992.forward(x4348);

// val X209310 = Pooling(7,1,0,false)(X209309)
JCudaTensor x4349;
JCudaTensor x4350;
x4350 = x4347;
x4349 = x1000.forward(x4350);

// Dealloc(X209309)
JCudaTensor x4351;
x4351 = x4347;
x4351.free();

// val X209311 = (X209310[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x4352;
JCudaMatrix x4353;
JCudaMatrix x4354;
JCudaTensor x4355;
JCudaTensor x4356;
x4356 = x4349;
x4355 = x4356.flatten(1, new int[]{2048, 1, 1});
x4353 = x4355.asMatrix(1, true);
JCudaTensor x4357;
x4357 = x1007;
x4354 = x4357.asMatrix(1, true);
x4352 = x4353.times(x4354);

// Dealloc(X209310)
JCudaTensor x4358;
x4358 = x4349;
x4358.free();

// val X209313 = (X209311 + (i) => fc_B)
JCudaTensor x4359;
JCudaTensor x4360, x4361;
x4360 = x4352;
x4361 = x1011;
x4359 = x4361.copy(64, x4360);

// val X209314 = Cuda(Indicator(Y, 1000))
JCudaTensor x4362;
JTensorFloat x4363;
x4363 = x4.asIndicator(1000);
x4362 = x4363.asJCudaTensor();

// val X209315 = X209314 .* X209313
JCudaTensor x4364;
JCudaTensor x4365, x4366;
x4365 = x4362;
x4366 = x4359;
x4364 = x4365.times_i(x4366);

// val X209316 = Sum((X209315)(i1783 | @))
JCudaTensor x4367;
JCudaMatrix x4368;
JCudaTensor x4369;
x4369 = x4364;
x4368 = x4369.asMatrix(1, true);
x4367 = x4368.sum();

// Dealloc(X209315)
JCudaTensor x4370;
x4370 = x4364;
x4370.free();

// val X209317 = Max((X209313)(i1783 | @))
JCudaTensor x4371;
JCudaMatrix x4372;
JCudaTensor x4373;
x4373 = x4359;
x4372 = x4373.asMatrix(1, true);
x4371 = x4372.max();

// Dealloc(X209313)
JCudaTensor x4374;
x4374 = x4359;
x4374.free();

// val X209318 = 1{X209316 == X209317}
JCudaTensor x4375;
JCudaTensor x4376, x4377;
x4376 = x4367;
x4377 = x4371;
x4375 = x4376.eq(x4377);

// Dealloc(X209317)
JCudaTensor x4378;
x4378 = x4371;
x4378.free();

// Print((Sum(X209318) / |64|))
float x4379;
float x4380;
float x4381;
JCudaTensor x4382;
x4382 = x4375;
x4380 = x4382.sum();
x4381 = 64;
x4379 = x4380 / x4381;
System.out.println(x5 + " test precision "  + x4379);

// Dealloc(X209318)
JCudaTensor x4383;
x4383 = x4375;
x4383.free();

}
 
}

}