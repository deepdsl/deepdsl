package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.tensor.*;
import deepdsl.util.*;


public class Googlenet {
static{ JCudaTensor.enableMemoryCache();}
// decay_1
static float decay_1 = 0.999995f;
// loss1
static float loss1 = 0.3f;
// loss2
static float loss2 = 0.3f;
// lrn_rate_1
static float lrn_rate_1 = -0.01f;
// lrn_rate_2
static float lrn_rate_2 = -0.02f;
// momentum
static float momentum = 0.9f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/googlenet";
// test_data_path
static String test_data_path = "dataset/imagenet224/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 20;
// train_data_path
static String train_data_path = "dataset/imagenet224/ilsvrc12_train_lmdb";
// train_itr
static int train_itr = 1000;

// (Convolv(1,0),List(128, 128, 4, 4))
static JCudnnConvolution x626 = new JCudnnConvolution(new int[]{128,256,4,4},new int[]{128,256,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(128, 128, 4, 4))
static JCudnnConvolution x301 = new JCudnnConvolution(new int[]{128,256,4,4},new int[]{128,256,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(128, 16, 7, 7))
static JCudnnConvolution x855 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 7, 7))
static JCudnnConvolution x769 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 14, 14))
static JCudnnConvolution x609 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 14, 14))
static JCudnnConvolution x455 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 14, 14))
static JCudnnConvolution x383 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 14, 14))
static JCudnnConvolution x274 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 14, 14))
static JCudnnConvolution x219 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 28, 28))
static JCudnnConvolution x134 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 16, 28, 28))
static JCudnnConvolution x79 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{16,192,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(128, 32, 7, 7))
static JCudnnConvolution x878 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 7, 7))
static JCudnnConvolution x776 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 14, 14))
static JCudnnConvolution x642 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 14, 14))
static JCudnnConvolution x484 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 14, 14))
static JCudnnConvolution x393 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 14, 14))
static JCudnnConvolution x311 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 14, 14))
static JCudnnConvolution x232 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 28, 28))
static JCudnnConvolution x158 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 32, 28, 28))
static JCudnnConvolution x86 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{32,192,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(128, 64, 7, 7))
static JCudnnConvolution x848 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 7, 7))
static JCudnnConvolution x750 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 14, 14))
static JCudnnConvolution x595 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 14, 14))
static JCudnnConvolution x474 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 14, 14))
static JCudnnConvolution x376 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 14, 14))
static JCudnnConvolution x291 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 14, 14))
static JCudnnConvolution x202 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 28, 28))
static JCudnnConvolution x141 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 28, 28))
static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{64,192,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 64, 56, 56))
static JCudnnConvolution x36 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{64,64,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(128, 96, 7, 7))
static JCudnnConvolution x841 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 7, 7))
static JCudnnConvolution x762 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 14, 14))
static JCudnnConvolution x602 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 14, 14))
static JCudnnConvolution x467 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 14, 14))
static JCudnnConvolution x363 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 14, 14))
static JCudnnConvolution x284 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 14, 14))
static JCudnnConvolution x209 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 28, 28))
static JCudnnConvolution x148 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(128, 96, 28, 28))
static JCudnnConvolution x69 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{96,192,1,1},new int[]{96}, 1, 0);
// (Convolv(1,1),List(128, 128, 7, 7))
static JCudnnConvolution x894 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 7, 7))
static JCudnnConvolution x799 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 14, 14))
static JCudnnConvolution x688 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 14, 14))
static JCudnnConvolution x522 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 14, 14))
static JCudnnConvolution x406 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 14, 14))
static JCudnnConvolution x328 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 14, 14))
static JCudnnConvolution x246 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 28, 28))
static JCudnnConvolution x168 = new JCudnnConvolution(new int[]{128,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 128, 28, 28))
static JCudnnConvolution x99 = new JCudnnConvolution(new int[]{128,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(128, 192, 56, 56))
static JCudnnConvolution x46 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{192,64,3,3},new int[]{192}, 1, 1);
// (Convolv(1,2),List(128, 32, 7, 7))
static JCudnnConvolution x906 = new JCudnnConvolution(new int[]{128,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 7, 7))
static JCudnnConvolution x806 = new JCudnnConvolution(new int[]{128,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 14, 14))
static JCudnnConvolution x678 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 14, 14))
static JCudnnConvolution x505 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 14, 14))
static JCudnnConvolution x413 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 14, 14))
static JCudnnConvolution x321 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 14, 14))
static JCudnnConvolution x239 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 28, 28))
static JCudnnConvolution x175 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(128, 32, 28, 28))
static JCudnnConvolution x106 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(2,3),List(128, 64, 112, 112))
static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
// (Dropout(0.4),List(128, 256, 1, 1))
static JCudnnDropout x1009 = new JCudnnDropout(new int[]{128,256,1,1}, 0.4f);
// (Dropout(0.7),List(128, 1024))
static JCudnnDropout x396 = new JCudnnDropout(new int[]{128,1024}, 0.7f);
// (LMDB,false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, 640, new int[]{128, 3, 224, 224});
// (LMDB,true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, 6400, new int[]{128, 3, 224, 224});
// (LRN(5,1.0E-4,0.75),List(128, 192, 56, 56))
static JCudnnLRN x52 = new JCudnnLRN(new int[]{128,192,56,56}, 5, 1.0E-4, 0.75);
// (LRN(5,1.0E-4,0.75),List(128, 64, 56, 56))
static JCudnnLRN x29 = new JCudnnLRN(new int[]{128,64,56,56}, 5, 1.0E-4, 0.75);
// (LogSoftmax(),List(128, 1000))
static JCudnnSoftmax x443 = new JCudnnSoftmax(new int[]{128,1000}, 2);
// (Pooling(3,1,1,true),List(128, 256, 7, 7))
static JCudnnPooling x864 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 256, 7, 7))
static JCudnnPooling x753 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 256, 14, 14))
static JCudnnPooling x581 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 256, 14, 14))
static JCudnnPooling x477 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 256, 14, 14))
static JCudnnPooling x369 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 256, 14, 14))
static JCudnnPooling x277 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 256, 14, 14))
static JCudnnPooling x212 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 256, 28, 28))
static JCudnnPooling x127 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 1, 1, 0);
// (Pooling(3,1,1,true),List(128, 192, 28, 28))
static JCudnnPooling x72 = new JCudnnPooling(new int[]{128,192,28,28}, 3, 1, 1, 0);
// (Pooling(3,2,1,true),List(128, 256, 7, 7))
static JCudnnPooling x741 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 2, 1, 0);
// (Pooling(3,2,1,true),List(128, 256, 14, 14))
static JCudnnPooling x195 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 2, 1, 0);
// (Pooling(3,2,1,true),List(128, 192, 28, 28))
static JCudnnPooling x55 = new JCudnnPooling(new int[]{128,192,56,56}, 3, 2, 1, 0);
// (Pooling(3,2,1,true),List(128, 64, 56, 56))
static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,64,112,112}, 3, 2, 1, 0);
// (Pooling(5,3,0,false),List(128, 256, 4, 4))
static JCudnnPooling x584 = new JCudnnPooling(new int[]{128,256,14,14}, 5, 3, 0, 2);
// (Pooling(5,3,0,false),List(128, 256, 4, 4))
static JCudnnPooling x267 = new JCudnnPooling(new int[]{128,256,14,14}, 5, 3, 0, 2);
// (Pooling(7,1,0,false),List(128, 256, 1, 1))
static JCudnnPooling x981 = new JCudnnPooling(new int[]{128,256,7,7}, 7, 1, 0, 2);
// (ReLU(),List(128, 1024))
static JCudnnActivation x366 = new JCudnnActivation(new int[]{128,1024}, 1);
// (ReLU(),List(128, 128, 7, 7))
static JCudnnActivation x930 = new JCudnnActivation(new int[]{128,128,7,7}, 1);
// (ReLU(),List(128, 128, 7, 7))
static JCudnnActivation x815 = new JCudnnActivation(new int[]{128,128,7,7}, 1);
// (ReLU(),List(128, 128, 4, 4))
static JCudnnActivation x657 = new JCudnnActivation(new int[]{128,128,4,4}, 1);
// (ReLU(),List(128, 128, 14, 14))
static JCudnnActivation x700 = new JCudnnActivation(new int[]{128,128,14,14}, 1);
// (ReLU(),List(128, 128, 14, 14))
static JCudnnActivation x536 = new JCudnnActivation(new int[]{128,128,14,14}, 1);
// (ReLU(),List(128, 128, 14, 14))
static JCudnnActivation x422 = new JCudnnActivation(new int[]{128,128,14,14}, 1);
// (ReLU(),List(128, 128, 4, 4))
static JCudnnActivation x314 = new JCudnnActivation(new int[]{128,128,4,4}, 1);
// (ReLU(),List(128, 128, 14, 14))
static JCudnnActivation x337 = new JCudnnActivation(new int[]{128,128,14,14}, 1);
// (ReLU(),List(128, 128, 14, 14))
static JCudnnActivation x252 = new JCudnnActivation(new int[]{128,128,14,14}, 1);
// (ReLU(),List(128, 128, 28, 28))
static JCudnnActivation x181 = new JCudnnActivation(new int[]{128,128,28,28}, 1);
// (ReLU(),List(128, 128, 28, 28))
static JCudnnActivation x112 = new JCudnnActivation(new int[]{128,128,28,28}, 1);
// (ReLU(),List(128, 16, 7, 7))
static JCudnnActivation x871 = new JCudnnActivation(new int[]{128,16,7,7}, 1);
// (ReLU(),List(128, 16, 7, 7))
static JCudnnActivation x782 = new JCudnnActivation(new int[]{128,16,7,7}, 1);
// (ReLU(),List(128, 16, 14, 14))
static JCudnnActivation x635 = new JCudnnActivation(new int[]{128,16,14,14}, 1);
// (ReLU(),List(128, 16, 14, 14))
static JCudnnActivation x496 = new JCudnnActivation(new int[]{128,16,14,14}, 1);
// (ReLU(),List(128, 16, 14, 14))
static JCudnnActivation x386 = new JCudnnActivation(new int[]{128,16,14,14}, 1);
// (ReLU(),List(128, 16, 14, 14))
static JCudnnActivation x294 = new JCudnnActivation(new int[]{128,16,14,14}, 1);
// (ReLU(),List(128, 16, 14, 14))
static JCudnnActivation x222 = new JCudnnActivation(new int[]{128,16,14,14}, 1);
// (ReLU(),List(128, 16, 28, 28))
static JCudnnActivation x161 = new JCudnnActivation(new int[]{128,16,28,28}, 1);
// (ReLU(),List(128, 16, 28, 28))
static JCudnnActivation x92 = new JCudnnActivation(new int[]{128,16,28,28}, 1);
// (ReLU(),List(128, 192, 56, 56))
static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,192,56,56}, 1);
// (ReLU(),List(128, 32, 7, 7))
static JCudnnActivation x936 = new JCudnnActivation(new int[]{128,32,7,7}, 1);
// (ReLU(),List(128, 32, 7, 7))
static JCudnnActivation x933 = new JCudnnActivation(new int[]{128,32,7,7}, 1);
// (ReLU(),List(128, 32, 7, 7))
static JCudnnActivation x812 = new JCudnnActivation(new int[]{128,32,7,7}, 1);
// (ReLU(),List(128, 32, 7, 7))
static JCudnnActivation x818 = new JCudnnActivation(new int[]{128,32,7,7}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x703 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x697 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x528 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x539 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x425 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x428 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x340 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x331 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x258 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 14, 14))
static JCudnnActivation x255 = new JCudnnActivation(new int[]{128,32,14,14}, 1);
// (ReLU(),List(128, 32, 28, 28))
static JCudnnActivation x187 = new JCudnnActivation(new int[]{128,32,28,28}, 1);
// (ReLU(),List(128, 32, 28, 28))
static JCudnnActivation x184 = new JCudnnActivation(new int[]{128,32,28,28}, 1);
// (ReLU(),List(128, 32, 28, 28))
static JCudnnActivation x118 = new JCudnnActivation(new int[]{128,32,28,28}, 1);
// (ReLU(),List(128, 32, 28, 28))
static JCudnnActivation x115 = new JCudnnActivation(new int[]{128,32,28,28}, 1);
// (ReLU(),List(128, 64, 7, 7))
static JCudnnActivation x927 = new JCudnnActivation(new int[]{128,64,7,7}, 1);
// (ReLU(),List(128, 64, 7, 7))
static JCudnnActivation x821 = new JCudnnActivation(new int[]{128,64,7,7}, 1);
// (ReLU(),List(128, 64, 14, 14))
static JCudnnActivation x723 = new JCudnnActivation(new int[]{128,64,14,14}, 1);
// (ReLU(),List(128, 64, 14, 14))
static JCudnnActivation x533 = new JCudnnActivation(new int[]{128,64,14,14}, 1);
// (ReLU(),List(128, 64, 14, 14))
static JCudnnActivation x431 = new JCudnnActivation(new int[]{128,64,14,14}, 1);
// (ReLU(),List(128, 64, 14, 14))
static JCudnnActivation x334 = new JCudnnActivation(new int[]{128,64,14,14}, 1);
// (ReLU(),List(128, 64, 14, 14))
static JCudnnActivation x249 = new JCudnnActivation(new int[]{128,64,14,14}, 1);
// (ReLU(),List(128, 64, 28, 28))
static JCudnnActivation x178 = new JCudnnActivation(new int[]{128,64,28,28}, 1);
// (ReLU(),List(128, 64, 28, 28))
static JCudnnActivation x109 = new JCudnnActivation(new int[]{128,64,28,28}, 1);
// (ReLU(),List(128, 64, 56, 56))
static JCudnnActivation x39 = new JCudnnActivation(new int[]{128,64,56,56}, 1);
// (ReLU(),List(128, 64, 112, 112))
static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,64,112,112}, 1);
// (ReLU(),List(128, 96, 7, 7))
static JCudnnActivation x884 = new JCudnnActivation(new int[]{128,96,7,7}, 1);
// (ReLU(),List(128, 96, 7, 7))
static JCudnnActivation x779 = new JCudnnActivation(new int[]{128,96,7,7}, 1);
// (ReLU(),List(128, 96, 14, 14))
static JCudnnActivation x619 = new JCudnnActivation(new int[]{128,96,14,14}, 1);
// (ReLU(),List(128, 96, 14, 14))
static JCudnnActivation x493 = new JCudnnActivation(new int[]{128,96,14,14}, 1);
// (ReLU(),List(128, 96, 14, 14))
static JCudnnActivation x399 = new JCudnnActivation(new int[]{128,96,14,14}, 1);
// (ReLU(),List(128, 96, 14, 14))
static JCudnnActivation x304 = new JCudnnActivation(new int[]{128,96,14,14}, 1);
// (ReLU(),List(128, 96, 14, 14))
static JCudnnActivation x225 = new JCudnnActivation(new int[]{128,96,14,14}, 1);
// (ReLU(),List(128, 96, 28, 28))
static JCudnnActivation x151 = new JCudnnActivation(new int[]{128,96,28,28}, 1);
// (ReLU(),List(128, 96, 28, 28))
static JCudnnActivation x89 = new JCudnnActivation(new int[]{128,96,28,28}, 1);
// List(List(128, 64, 14, 14), List(128, 128, 14, 14), List(128, 32, 14, 14), List(128, 32, 14, 14))
static JCudnnConcat x260 = new JCudnnConcat(new int[]{128,64,14,14},new int[]{128,128,14,14},new int[]{128,32,14,14},new int[]{128,32,14,14});
// List(List(128, 64, 28, 28), List(128, 128, 28, 28), List(128, 32, 28, 28), List(128, 32, 28, 28))
static JCudnnConcat x120 = new JCudnnConcat(new int[]{128,64,28,28},new int[]{128,128,28,28},new int[]{128,32,28,28},new int[]{128,32,28,28});
// List(List(128, 64, 7, 7), List(128, 128, 7, 7), List(128, 32, 7, 7), List(128, 32, 7, 7))
static JCudnnConcat x826 = new JCudnnConcat(new int[]{128,64,7,7},new int[]{128,128,7,7},new int[]{128,32,7,7},new int[]{128,32,7,7});
// V_b1cv_B
static JCudaTensor x704 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_b1cv_W
static JCudaTensor x709 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_b1fc1_B
static JCudaTensor x664 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_b1fc1_W
static JCudaTensor x650 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
// V_b1fc2_B
static JCudaTensor x561 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_b1fc2_W
static JCudaTensor x566 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
// V_b2cv_B
static JCudaTensor x1028 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_b2cv_W
static JCudaTensor x1016 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_b2fc1_B
static JCudaTensor x985 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_b2fc1_W
static JCudaTensor x990 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
// V_b2fc2_B
static JCudaTensor x920 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_b2fc2_W
static JCudaTensor x940 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
// V_cv11_B
static JCudaTensor x2443 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv11_W
static JCudaTensor x2409 = JTensor.constFloat(0.0f, 64, 192, 1, 1).asJCudaTensor();
// V_cv12_B
static JCudaTensor x2495 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv12_W
static JCudaTensor x2503 = JTensor.constFloat(0.0f, 96, 192, 1, 1).asJCudaTensor();
// V_cv13_B
static JCudaTensor x2433 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv13_W
static JCudaTensor x2448 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv14_B
static JCudaTensor x2509 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv14_W
static JCudaTensor x2490 = JTensor.constFloat(0.0f, 16, 192, 1, 1).asJCudaTensor();
// V_cv15_B
static JCudaTensor x2421 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv15_W
static JCudaTensor x2437 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv16_B
static JCudaTensor x2414 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv16_W
static JCudaTensor x2425 = JTensor.constFloat(0.0f, 32, 192, 1, 1).asJCudaTensor();
// V_cv1_B
static JCudaTensor x2615 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv1_W
static JCudaTensor x2609 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
// V_cv21_B
static JCudaTensor x2285 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv21_W
static JCudaTensor x2271 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv22_B
static JCudaTensor x2344 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv22_W
static JCudaTensor x2352 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv23_B
static JCudaTensor x2276 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv23_W
static JCudaTensor x2297 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv24_B
static JCudaTensor x2340 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv24_W
static JCudaTensor x2335 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv25_B
static JCudaTensor x2280 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv25_W
static JCudaTensor x2263 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv26_B
static JCudaTensor x2251 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv26_W
static JCudaTensor x2258 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv2_B
static JCudaTensor x2575 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv2_W
static JCudaTensor x2582 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
// V_cv31_B
static JCudaTensor x2114 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv31_W
static JCudaTensor x2095 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv32_B
static JCudaTensor x2192 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv32_W
static JCudaTensor x2183 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv33_B
static JCudaTensor x2105 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv33_W
static JCudaTensor x2109 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv34_B
static JCudaTensor x2188 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv34_W
static JCudaTensor x2174 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv35_B
static JCudaTensor x2122 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv35_W
static JCudaTensor x2100 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv36_B
static JCudaTensor x2133 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv36_W
static JCudaTensor x2090 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv3_B
static JCudaTensor x2552 = JTensor.constFloat(0.0f, 192).asJCudaTensor();
// V_cv3_W
static JCudaTensor x2559 = JTensor.constFloat(0.0f, 192, 64, 3, 3).asJCudaTensor();
// V_cv41_B
static JCudaTensor x1933 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv41_W
static JCudaTensor x1944 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv42_B
static JCudaTensor x2026 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv42_W
static JCudaTensor x2021 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv43_B
static JCudaTensor x1970 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv43_W
static JCudaTensor x1974 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv44_B
static JCudaTensor x2031 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv44_W
static JCudaTensor x2012 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv45_B
static JCudaTensor x1940 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv45_W
static JCudaTensor x1952 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv46_B
static JCudaTensor x1957 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv46_W
static JCudaTensor x1928 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv51_B
static JCudaTensor x1794 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv51_W
static JCudaTensor x1773 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv52_B
static JCudaTensor x1862 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv52_W
static JCudaTensor x1874 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv53_B
static JCudaTensor x1790 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv53_W
static JCudaTensor x1781 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv54_B
static JCudaTensor x1866 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv54_W
static JCudaTensor x1857 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv55_B
static JCudaTensor x1786 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv55_W
static JCudaTensor x1819 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv56_B
static JCudaTensor x1805 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv56_W
static JCudaTensor x1813 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv61_B
static JCudaTensor x1660 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv61_W
static JCudaTensor x1642 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv62_B
static JCudaTensor x1716 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv62_W
static JCudaTensor x1711 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv63_B
static JCudaTensor x1665 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv63_W
static JCudaTensor x1647 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv64_B
static JCudaTensor x1721 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv64_W
static JCudaTensor x1702 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv65_B
static JCudaTensor x1655 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv65_W
static JCudaTensor x1637 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv66_B
static JCudaTensor x1621 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv66_W
static JCudaTensor x1628 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv71_B
static JCudaTensor x1498 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv71_W
static JCudaTensor x1459 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv72_B
static JCudaTensor x1554 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv72_W
static JCudaTensor x1549 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv73_B
static JCudaTensor x1469 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv73_W
static JCudaTensor x1464 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv74_B
static JCudaTensor x1559 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv74_W
static JCudaTensor x1540 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv75_B
static JCudaTensor x1480 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv75_W
static JCudaTensor x1492 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv76_B
static JCudaTensor x1503 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv76_W
static JCudaTensor x1484 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv81_B
static JCudaTensor x1298 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv81_W
static JCudaTensor x1318 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv82_B
static JCudaTensor x1384 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv82_W
static JCudaTensor x1392 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv83_B
static JCudaTensor x1329 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv83_W
static JCudaTensor x1302 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv84_B
static JCudaTensor x1398 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv84_W
static JCudaTensor x1379 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv85_B
static JCudaTensor x1334 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv85_W
static JCudaTensor x1310 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv86_B
static JCudaTensor x1339 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv86_W
static JCudaTensor x1324 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv91_B
static JCudaTensor x1163 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv91_W
static JCudaTensor x1147 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv92_B
static JCudaTensor x1242 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv92_W
static JCudaTensor x1237 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv93_B
static JCudaTensor x1187 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv93_W
static JCudaTensor x1176 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv94_B
static JCudaTensor x1229 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv94_W
static JCudaTensor x1224 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv95_B
static JCudaTensor x1172 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv95_W
static JCudaTensor x1181 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv96_B
static JCudaTensor x1143 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv96_W
static JCudaTensor x1158 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_fc_B
static JCudaTensor x1099 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc_W
static JCudaTensor x1093 = JTensor.constFloat(0.0f, 1000, 256).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// b1cv_B
static JCudaTensor x300 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b1cv_B").asJCudaTensor();
// b1cv_W
static JCudaTensor x299 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b1cv_W").asJCudaTensor();
// b1fc1_B
static JCudaTensor x356 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b1fc1_B").asJCudaTensor();
// b1fc1_W
static JCudaTensor x347 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b1fc1_W").asJCudaTensor();
// b1fc2_B
static JCudaTensor x435 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b1fc2_B").asJCudaTensor();
// b1fc2_W
static JCudaTensor x419 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b1fc2_W").asJCudaTensor();
// b2cv_B
static JCudaTensor x625 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b2cv_B").asJCudaTensor();
// b2cv_W
static JCudaTensor x624 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b2cv_W").asJCudaTensor();
// b2fc1_B
static JCudaTensor x733 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b2fc1_B").asJCudaTensor();
// b2fc1_W
static JCudaTensor x720 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b2fc1_W").asJCudaTensor();
// b2fc2_B
static JCudaTensor x792 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b2fc2_B").asJCudaTensor();
// b2fc2_W
static JCudaTensor x788 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b2fc2_W").asJCudaTensor();
// cv11_B
static JCudaTensor x61 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
// cv11_W
static JCudaTensor x60 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 64, 192, 1, 1).load(network_dir + "/cv11_W").asJCudaTensor();
// cv12_B
static JCudaTensor x68 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv12_B").asJCudaTensor();
// cv12_W
static JCudaTensor x67 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 96, 192, 1, 1).load(network_dir + "/cv12_W").asJCudaTensor();
// cv13_B
static JCudaTensor x98 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv13_B").asJCudaTensor();
// cv13_W
static JCudaTensor x97 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv13_W").asJCudaTensor();
// cv14_B
static JCudaTensor x78 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv14_B").asJCudaTensor();
// cv14_W
static JCudaTensor x77 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 16, 192, 1, 1).load(network_dir + "/cv14_W").asJCudaTensor();
// cv15_B
static JCudaTensor x105 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv15_B").asJCudaTensor();
// cv15_W
static JCudaTensor x104 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv15_W").asJCudaTensor();
// cv16_B
static JCudaTensor x85 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv16_B").asJCudaTensor();
// cv16_W
static JCudaTensor x84 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 32, 192, 1, 1).load(network_dir + "/cv16_W").asJCudaTensor();
// cv1_B
static JCudaTensor x16 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv1_B").asJCudaTensor();
// cv1_W
static JCudaTensor x15 = JTensor.randomFloat(-0.11664237f, 0.11664237f, 64, 3, 7, 7).load(network_dir + "/cv1_W").asJCudaTensor();
// cv21_B
static JCudaTensor x140 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv21_B").asJCudaTensor();
// cv21_W
static JCudaTensor x139 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv21_W").asJCudaTensor();
// cv22_B
static JCudaTensor x147 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv22_B").asJCudaTensor();
// cv22_W
static JCudaTensor x146 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv22_W").asJCudaTensor();
// cv23_B
static JCudaTensor x167 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv23_B").asJCudaTensor();
// cv23_W
static JCudaTensor x166 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv23_W").asJCudaTensor();
// cv24_B
static JCudaTensor x133 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv24_B").asJCudaTensor();
// cv24_W
static JCudaTensor x132 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv24_W").asJCudaTensor();
// cv25_B
static JCudaTensor x174 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv25_B").asJCudaTensor();
// cv25_W
static JCudaTensor x173 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv25_W").asJCudaTensor();
// cv26_B
static JCudaTensor x157 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv26_B").asJCudaTensor();
// cv26_W
static JCudaTensor x156 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv26_W").asJCudaTensor();
// cv2_B
static JCudaTensor x35 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv2_B").asJCudaTensor();
// cv2_W
static JCudaTensor x34 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/cv2_W").asJCudaTensor();
// cv31_B
static JCudaTensor x201 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv31_B").asJCudaTensor();
// cv31_W
static JCudaTensor x200 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv31_W").asJCudaTensor();
// cv32_B
static JCudaTensor x208 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv32_B").asJCudaTensor();
// cv32_W
static JCudaTensor x207 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv32_W").asJCudaTensor();
// cv33_B
static JCudaTensor x245 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv33_B").asJCudaTensor();
// cv33_W
static JCudaTensor x244 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
// cv34_B
static JCudaTensor x218 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv34_B").asJCudaTensor();
// cv34_W
static JCudaTensor x217 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv34_W").asJCudaTensor();
// cv35_B
static JCudaTensor x238 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv35_B").asJCudaTensor();
// cv35_W
static JCudaTensor x237 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv35_W").asJCudaTensor();
// cv36_B
static JCudaTensor x231 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv36_B").asJCudaTensor();
// cv36_W
static JCudaTensor x230 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv36_W").asJCudaTensor();
// cv3_B
static JCudaTensor x45 = JTensor.constFloat(0.2f, 192).load(network_dir + "/cv3_B").asJCudaTensor();
// cv3_W
static JCudaTensor x44 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 192, 64, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
// cv41_B
static JCudaTensor x290 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv41_B").asJCudaTensor();
// cv41_W
static JCudaTensor x289 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv41_W").asJCudaTensor();
// cv42_B
static JCudaTensor x283 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv42_B").asJCudaTensor();
// cv42_W
static JCudaTensor x282 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv42_W").asJCudaTensor();
// cv43_B
static JCudaTensor x327 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv43_B").asJCudaTensor();
// cv43_W
static JCudaTensor x326 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
// cv44_B
static JCudaTensor x273 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv44_B").asJCudaTensor();
// cv44_W
static JCudaTensor x272 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv44_W").asJCudaTensor();
// cv45_B
static JCudaTensor x320 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv45_B").asJCudaTensor();
// cv45_W
static JCudaTensor x319 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv45_W").asJCudaTensor();
// cv46_B
static JCudaTensor x310 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv46_B").asJCudaTensor();
// cv46_W
static JCudaTensor x309 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv46_W").asJCudaTensor();
// cv51_B
static JCudaTensor x375 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv51_B").asJCudaTensor();
// cv51_W
static JCudaTensor x374 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv51_W").asJCudaTensor();
// cv52_B
static JCudaTensor x362 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv52_B").asJCudaTensor();
// cv52_W
static JCudaTensor x361 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv52_W").asJCudaTensor();
// cv53_B
static JCudaTensor x405 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv53_B").asJCudaTensor();
// cv53_W
static JCudaTensor x404 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
// cv54_B
static JCudaTensor x382 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv54_B").asJCudaTensor();
// cv54_W
static JCudaTensor x381 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv54_W").asJCudaTensor();
// cv55_B
static JCudaTensor x412 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv55_B").asJCudaTensor();
// cv55_W
static JCudaTensor x411 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv55_W").asJCudaTensor();
// cv56_B
static JCudaTensor x392 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv56_B").asJCudaTensor();
// cv56_W
static JCudaTensor x391 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv56_W").asJCudaTensor();
// cv61_B
static JCudaTensor x473 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv61_B").asJCudaTensor();
// cv61_W
static JCudaTensor x472 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv61_W").asJCudaTensor();
// cv62_B
static JCudaTensor x466 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv62_B").asJCudaTensor();
// cv62_W
static JCudaTensor x465 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv62_W").asJCudaTensor();
// cv63_B
static JCudaTensor x521 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv63_B").asJCudaTensor();
// cv63_W
static JCudaTensor x520 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv63_W").asJCudaTensor();
// cv64_B
static JCudaTensor x454 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv64_B").asJCudaTensor();
// cv64_W
static JCudaTensor x453 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv64_W").asJCudaTensor();
// cv65_B
static JCudaTensor x504 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv65_B").asJCudaTensor();
// cv65_W
static JCudaTensor x503 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv65_W").asJCudaTensor();
// cv66_B
static JCudaTensor x483 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv66_B").asJCudaTensor();
// cv66_W
static JCudaTensor x482 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv66_W").asJCudaTensor();
// cv71_B
static JCudaTensor x594 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv71_B").asJCudaTensor();
// cv71_W
static JCudaTensor x593 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv71_W").asJCudaTensor();
// cv72_B
static JCudaTensor x601 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv72_B").asJCudaTensor();
// cv72_W
static JCudaTensor x600 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv72_W").asJCudaTensor();
// cv73_B
static JCudaTensor x687 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv73_B").asJCudaTensor();
// cv73_W
static JCudaTensor x686 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv73_W").asJCudaTensor();
// cv74_B
static JCudaTensor x608 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv74_B").asJCudaTensor();
// cv74_W
static JCudaTensor x607 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv74_W").asJCudaTensor();
// cv75_B
static JCudaTensor x677 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv75_B").asJCudaTensor();
// cv75_W
static JCudaTensor x676 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv75_W").asJCudaTensor();
// cv76_B
static JCudaTensor x641 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv76_B").asJCudaTensor();
// cv76_W
static JCudaTensor x640 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv76_W").asJCudaTensor();
// cv81_B
static JCudaTensor x749 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv81_B").asJCudaTensor();
// cv81_W
static JCudaTensor x748 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv81_W").asJCudaTensor();
// cv82_B
static JCudaTensor x761 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv82_B").asJCudaTensor();
// cv82_W
static JCudaTensor x760 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv82_W").asJCudaTensor();
// cv83_B
static JCudaTensor x798 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv83_B").asJCudaTensor();
// cv83_W
static JCudaTensor x797 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv83_W").asJCudaTensor();
// cv84_B
static JCudaTensor x768 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv84_B").asJCudaTensor();
// cv84_W
static JCudaTensor x767 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv84_W").asJCudaTensor();
// cv85_B
static JCudaTensor x805 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv85_B").asJCudaTensor();
// cv85_W
static JCudaTensor x804 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv85_W").asJCudaTensor();
// cv86_B
static JCudaTensor x775 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv86_B").asJCudaTensor();
// cv86_W
static JCudaTensor x774 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv86_W").asJCudaTensor();
// cv91_B
static JCudaTensor x847 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv91_B").asJCudaTensor();
// cv91_W
static JCudaTensor x846 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv91_W").asJCudaTensor();
// cv92_B
static JCudaTensor x840 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv92_B").asJCudaTensor();
// cv92_W
static JCudaTensor x839 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv92_W").asJCudaTensor();
// cv93_B
static JCudaTensor x893 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv93_B").asJCudaTensor();
// cv93_W
static JCudaTensor x892 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv93_W").asJCudaTensor();
// cv94_B
static JCudaTensor x854 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv94_B").asJCudaTensor();
// cv94_W
static JCudaTensor x853 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv94_W").asJCudaTensor();
// cv95_B
static JCudaTensor x905 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv95_B").asJCudaTensor();
// cv95_W
static JCudaTensor x904 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv95_W").asJCudaTensor();
// cv96_B
static JCudaTensor x877 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv96_B").asJCudaTensor();
// cv96_W
static JCudaTensor x876 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv96_W").asJCudaTensor();
// fc_B
static JCudaTensor x1042 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
// fc_W
static JCudaTensor x1027 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1000, 256).load(network_dir + "/fc_W").asJCudaTensor();

public static void main(String[] args){
ArithStats.isStats = false;
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
System.out.println(ArithStats.outStats());
test();
x300.save(network_dir + "/b1cv_B");
x299.save(network_dir + "/b1cv_W");
x356.save(network_dir + "/b1fc1_B");
x347.save(network_dir + "/b1fc1_W");
x435.save(network_dir + "/b1fc2_B");
x419.save(network_dir + "/b1fc2_W");
x625.save(network_dir + "/b2cv_B");
x624.save(network_dir + "/b2cv_W");
x733.save(network_dir + "/b2fc1_B");
x720.save(network_dir + "/b2fc1_W");
x792.save(network_dir + "/b2fc2_B");
x788.save(network_dir + "/b2fc2_W");
x61.save(network_dir + "/cv11_B");
x60.save(network_dir + "/cv11_W");
x68.save(network_dir + "/cv12_B");
x67.save(network_dir + "/cv12_W");
x98.save(network_dir + "/cv13_B");
x97.save(network_dir + "/cv13_W");
x78.save(network_dir + "/cv14_B");
x77.save(network_dir + "/cv14_W");
x105.save(network_dir + "/cv15_B");
x104.save(network_dir + "/cv15_W");
x85.save(network_dir + "/cv16_B");
x84.save(network_dir + "/cv16_W");
x16.save(network_dir + "/cv1_B");
x15.save(network_dir + "/cv1_W");
x140.save(network_dir + "/cv21_B");
x139.save(network_dir + "/cv21_W");
x147.save(network_dir + "/cv22_B");
x146.save(network_dir + "/cv22_W");
x167.save(network_dir + "/cv23_B");
x166.save(network_dir + "/cv23_W");
x133.save(network_dir + "/cv24_B");
x132.save(network_dir + "/cv24_W");
x174.save(network_dir + "/cv25_B");
x173.save(network_dir + "/cv25_W");
x157.save(network_dir + "/cv26_B");
x156.save(network_dir + "/cv26_W");
x35.save(network_dir + "/cv2_B");
x34.save(network_dir + "/cv2_W");
x201.save(network_dir + "/cv31_B");
x200.save(network_dir + "/cv31_W");
x208.save(network_dir + "/cv32_B");
x207.save(network_dir + "/cv32_W");
x245.save(network_dir + "/cv33_B");
x244.save(network_dir + "/cv33_W");
x218.save(network_dir + "/cv34_B");
x217.save(network_dir + "/cv34_W");
x238.save(network_dir + "/cv35_B");
x237.save(network_dir + "/cv35_W");
x231.save(network_dir + "/cv36_B");
x230.save(network_dir + "/cv36_W");
x45.save(network_dir + "/cv3_B");
x44.save(network_dir + "/cv3_W");
x290.save(network_dir + "/cv41_B");
x289.save(network_dir + "/cv41_W");
x283.save(network_dir + "/cv42_B");
x282.save(network_dir + "/cv42_W");
x327.save(network_dir + "/cv43_B");
x326.save(network_dir + "/cv43_W");
x273.save(network_dir + "/cv44_B");
x272.save(network_dir + "/cv44_W");
x320.save(network_dir + "/cv45_B");
x319.save(network_dir + "/cv45_W");
x310.save(network_dir + "/cv46_B");
x309.save(network_dir + "/cv46_W");
x375.save(network_dir + "/cv51_B");
x374.save(network_dir + "/cv51_W");
x362.save(network_dir + "/cv52_B");
x361.save(network_dir + "/cv52_W");
x405.save(network_dir + "/cv53_B");
x404.save(network_dir + "/cv53_W");
x382.save(network_dir + "/cv54_B");
x381.save(network_dir + "/cv54_W");
x412.save(network_dir + "/cv55_B");
x411.save(network_dir + "/cv55_W");
x392.save(network_dir + "/cv56_B");
x391.save(network_dir + "/cv56_W");
x473.save(network_dir + "/cv61_B");
x472.save(network_dir + "/cv61_W");
x466.save(network_dir + "/cv62_B");
x465.save(network_dir + "/cv62_W");
x521.save(network_dir + "/cv63_B");
x520.save(network_dir + "/cv63_W");
x454.save(network_dir + "/cv64_B");
x453.save(network_dir + "/cv64_W");
x504.save(network_dir + "/cv65_B");
x503.save(network_dir + "/cv65_W");
x483.save(network_dir + "/cv66_B");
x482.save(network_dir + "/cv66_W");
x594.save(network_dir + "/cv71_B");
x593.save(network_dir + "/cv71_W");
x601.save(network_dir + "/cv72_B");
x600.save(network_dir + "/cv72_W");
x687.save(network_dir + "/cv73_B");
x686.save(network_dir + "/cv73_W");
x608.save(network_dir + "/cv74_B");
x607.save(network_dir + "/cv74_W");
x677.save(network_dir + "/cv75_B");
x676.save(network_dir + "/cv75_W");
x641.save(network_dir + "/cv76_B");
x640.save(network_dir + "/cv76_W");
x749.save(network_dir + "/cv81_B");
x748.save(network_dir + "/cv81_W");
x761.save(network_dir + "/cv82_B");
x760.save(network_dir + "/cv82_W");
x798.save(network_dir + "/cv83_B");
x797.save(network_dir + "/cv83_W");
x768.save(network_dir + "/cv84_B");
x767.save(network_dir + "/cv84_W");
x805.save(network_dir + "/cv85_B");
x804.save(network_dir + "/cv85_W");
x775.save(network_dir + "/cv86_B");
x774.save(network_dir + "/cv86_W");
x847.save(network_dir + "/cv91_B");
x846.save(network_dir + "/cv91_W");
x840.save(network_dir + "/cv92_B");
x839.save(network_dir + "/cv92_W");
x893.save(network_dir + "/cv93_B");
x892.save(network_dir + "/cv93_W");
x854.save(network_dir + "/cv94_B");
x853.save(network_dir + "/cv94_W");
x905.save(network_dir + "/cv95_B");
x904.save(network_dir + "/cv95_W");
x877.save(network_dir + "/cv96_B");
x876.save(network_dir + "/cv96_W");
x1042.save(network_dir + "/fc_B");
x1027.save(network_dir + "/fc_W");
x381.free();
x61.free();
x1970.free();
x608.free();
x733.free();
x1642.free();
x15.free();
x2095.free();
x1957.free();
x167.free();
x2433.free();
x2552.free();
x767.free();
x2031.free();
x877.free();
x774.free();
x404.free();
x356.free();
x435.free();
x453.free();
x140.free();
x1549.free();
x1392.free();
x1819.free();
x1540.free();
x1974.free();
x847.free();
x594.free();
x1181.free();
x892.free();
x709.free();
x2285.free();
x561.free();
x466.free();
x2021.free();
x97.free();
x2575.free();
x472.free();
x1790.free();
x1324.free();
x34.free();
x1660.free();
x775.free();
x1459.free();
x1484.free();
x84.free();
x299.free();
x798.free();
x1310.free();
x237.free();
x2258.free();
x1224.free();
x720.free();
x664.free();
x405.free();
x2421.free();
x147.free();
x1940.free();
x238.free();
x98.free();
x1469.free();
x2251.free();
x1702.free();
x482.free();
x231.free();
x230.free();
x853.free();
x473.free();
x1172.free();
x1379.free();
x1554.free();
x1016.free();
x67.free();
x319.free();
x624.free();
x704.free();
x2609.free();
x2276.free();
x35.free();
x1334.free();
x1237.free();
x520.free();
x985.free();
x1339.free();
x1866.free();
x375.free();
x2192.free();
x1028.free();
x2271.free();
x1952.free();
x686.free();
x391.free();
x327.free();
x1329.free();
x1480.free();
x2026.free();
x2335.free();
x77.free();
x2414.free();
x1147.free();
x2495.free();
x310.free();
x1163.free();
x839.free();
x1862.free();
x1944.free();
x687.free();
x2114.free();
x761.free();
x201.free();
x85.free();
x200.free();
x854.free();
x904.free();
x804.free();
x483.free();
x290.free();
x893.free();
x166.free();
x45.free();
x272.free();
x60.free();
x768.free();
x2109.free();
x1318.free();
x1093.free();
x2183.free();
x1781.free();
x300.free();
x2174.free();
x78.free();
x2615.free();
x173.free();
x2582.free();
x792.free();
x1655.free();
x2409.free();
x1384.free();
x566.free();
x217.free();
x625.free();
x749.free();
x2012.free();
x2263.free();
x1637.free();
x104.free();
x1711.free();
x362.free();
x382.free();
x640.free();
x788.free();
x1874.free();
x920.free();
x1229.free();
x309.free();
x133.free();
x2133.free();
x326.free();
x677.free();
x521.free();
x1143.free();
x1928.free();
x1933.free();
x1492.free();
x846.free();
x139.free();
x1099.free();
x2352.free();
x600.free();
x2188.free();
x1027.free();
x1176.free();
x1721.free();
x2509.free();
x465.free();
x1187.free();
x1857.free();
x503.free();
x2280.free();
x2122.free();
x2340.free();
x607.free();
x392.free();
x207.free();
x2344.free();
x146.free();
x1794.free();
x2425.free();
x797.free();
x44.free();
x1158.free();
x2090.free();
x419.free();
x1298.free();
x156.free();
x1665.free();
x2503.free();
x840.free();
x1621.free();
x1716.free();
x16.free();
x244.free();
x1398.free();
x454.free();
x2443.free();
x905.free();
x105.free();
x245.free();
x676.free();
x1559.free();
x805.free();
x650.free();
x2105.free();
x601.free();
x157.free();
x411.free();
x374.free();
x593.free();
x2297.free();
x132.free();
x1805.free();
x1302.free();
x68.free();
x208.free();
x289.free();
x990.free();
x282.free();
x347.free();
x273.free();
x412.free();
x2437.free();
x2559.free();
x876.free();
x1628.free();
x2490.free();
x283.free();
x1242.free();
x1773.free();
x1503.free();
x1042.free();
x760.free();
x641.free();
x1498.free();
x1647.free();
x1813.free();
x2100.free();
x174.free();
x748.free();
x504.free();
x218.free();
x940.free();
x1786.free();
x1464.free();
x320.free();
x2448.free();
x361.free();
x871.free();
x753.free();
x252.free();
x187.free();
x178.free();
x428.free();
x855.free();
x321.free();
x812.free();
x493.free();
x376.free();
x337.free();
x906.free();
x776.free();
x184.free();
x232.free();
x474.free();
x127.free();
x703.free();
x933.free();
x741.free();
x602.free();
x219.free();
x769.free();
x39.free();
x841.free();
x134.free();
x657.free();
x678.free();
x369.free();
x334.free();
x331.free();
x175.free();
x212.free();
x396.free();
x239.free();
x422.free();
x29.free();
x894.free();
x750.free();
x393.free();
x106.free();
x723.free();
x1009.free();
x112.free();
x386.free();
x477.free();
x181.free();
x255.free();
x584.free();
x161.free();
x311.free();
x49.free();
x148.free();
x484.free();
x536.free();
x936.free();
x36.free();
x284.free();
x340.free();
x609.free();
x267.free();
x619.free();
x981.free();
x294.free();
x505.free();
x141.free();
x782.free();
x195.free();
x927.free();
x99.free();
x26.free();
x23.free();
x399.free();
x406.free();
x522.free();
x89.free();
x864.free();
x806.free();
x86.free();
x55.free();
x467.free();
x930.free();
x92.free();
x581.free();
x225.free();
x52.free();
x366.free();
x79.free();
x425.free();
x151.free();
x443.free();
x635.free();
x815.free();
x46.free();
x17.free();
x642.free();
x291.free();
x314.free();
x848.free();
x799.free();
x62.free();
x69.free();
x496.free();
x115.free();
x72.free();
x202.free();
x688.free();
x109.free();
x762.free();
x118.free();
x884.free();
x328.free();
x626.free();
x383.free();
x258.free();
x301.free();
x533.free();
x209.free();
x528.free();
x779.free();
x700.free();
x222.free();
x455.free();
x246.free();
x818.free();
x363.free();
x821.free();
x274.free();
x277.free();
x595.free();
x413.free();
x878.free();
x539.free();
x168.free();
x697.free();
x304.free();
x431.free();
x158.free();
x249.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void train() {
 for(int x5=0; x5<train_itr; x5++) {
JTensorFloatTuple x6 =  x1.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X82 = Cuda(X)
JCudaTensor x7;
JTensorFloat x8;
x8 = x3;
x7 = x8.asJCudaTensor();

// val X227 = Cuda(Indicator(Y, 1000))
JCudaTensor x9;
JTensorFloat x10;
x10 = x4.asIndicator(1000);
x9 = x10.asJCudaTensor();

// val X83 = Convolv(2,3)(X82,cv1_W,cv1_B)
JCudaTensor x11;
JCudaTensor x12, x13, x14;
x12 = x7;
x13 = x15;
x14 = x16;
x11 = x17.forward(x12,x13,x14);

// val X336 = - X227.copy
JCudaTensor x18;
JCudaTensor x19;
float x20;
x19 = x9;
x19 = x19.clone();
x20 = -1;
x18 = x19.times_i(x20);

// val X84 = ReLU()(X83)
JCudaTensor x21;
JCudaTensor x22;
x22 = x11;
x21 = x23.forward(x22);

// val X85 = Pooling(3,2,1,true)(X84)
JCudaTensor x24;
JCudaTensor x25;
x25 = x21;
x24 = x26.forward(x25);

// val X86 = LRN(5,1.0E-4,0.75)(X85)
JCudaTensor x27;
JCudaTensor x28;
x28 = x24;
x27 = x29.forward(x28);

// val X87 = Convolv(1,0)(X86,cv2_W,cv2_B)
JCudaTensor x30;
JCudaTensor x31, x32, x33;
x31 = x27;
x32 = x34;
x33 = x35;
x30 = x36.forward(x31,x32,x33);

// val X88 = ReLU()(X87)
JCudaTensor x37;
JCudaTensor x38;
x38 = x30;
x37 = x39.forward(x38);

// val X89 = Convolv(1,1)(X88,cv3_W,cv3_B)
JCudaTensor x40;
JCudaTensor x41, x42, x43;
x41 = x37;
x42 = x44;
x43 = x45;
x40 = x46.forward(x41,x42,x43);

// val X90 = ReLU()(X89)
JCudaTensor x47;
JCudaTensor x48;
x48 = x40;
x47 = x49.forward(x48);

// val X91 = LRN(5,1.0E-4,0.75)(X90)
JCudaTensor x50;
JCudaTensor x51;
x51 = x47;
x50 = x52.forward(x51);

// val X92 = Pooling(3,2,1,true)(X91)
JCudaTensor x53;
JCudaTensor x54;
x54 = x50;
x53 = x55.forward(x54);

// val X93 = Convolv(1,0)(X92,cv11_W,cv11_B)
JCudaTensor x56;
JCudaTensor x57, x58, x59;
x57 = x53;
x58 = x60;
x59 = x61;
x56 = x62.forward(x57,x58,x59);

// val X95 = Convolv(1,0)(X92,cv12_W,cv12_B)
JCudaTensor x63;
JCudaTensor x64, x65, x66;
x64 = x53;
x65 = x67;
x66 = x68;
x63 = x69.forward(x64,x65,x66);

// val X103 = Pooling(3,1,1,true)(X92)
JCudaTensor x70;
JCudaTensor x71;
x71 = x53;
x70 = x72.forward(x71);

// val X99 = Convolv(1,0)(X92,cv14_W,cv14_B)
JCudaTensor x73;
JCudaTensor x74, x75, x76;
x74 = x53;
x75 = x77;
x76 = x78;
x73 = x79.forward(x74,x75,x76);

// val X104 = Convolv(1,0)(X103,cv16_W,cv16_B)
JCudaTensor x80;
JCudaTensor x81, x82, x83;
x81 = x70;
x82 = x84;
x83 = x85;
x80 = x86.forward(x81,x82,x83);

// val X96 = ReLU()(X95)
JCudaTensor x87;
JCudaTensor x88;
x88 = x63;
x87 = x89.forward(x88);

// val X100 = ReLU()(X99)
JCudaTensor x90;
JCudaTensor x91;
x91 = x73;
x90 = x92.forward(x91);

// val X97 = Convolv(1,1)(X96,cv13_W,cv13_B)
JCudaTensor x93;
JCudaTensor x94, x95, x96;
x94 = x87;
x95 = x97;
x96 = x98;
x93 = x99.forward(x94,x95,x96);

// val X101 = Convolv(1,2)(X100,cv15_W,cv15_B)
JCudaTensor x100;
JCudaTensor x101, x102, x103;
x101 = x90;
x102 = x104;
x103 = x105;
x100 = x106.forward(x101,x102,x103);

// val X94 = ReLU()(X93)
JCudaTensor x107;
JCudaTensor x108;
x108 = x56;
x107 = x109.forward(x108);

// val X98 = ReLU()(X97)
JCudaTensor x110;
JCudaTensor x111;
x111 = x93;
x110 = x112.forward(x111);

// val X102 = ReLU()(X101)
JCudaTensor x113;
JCudaTensor x114;
x114 = x100;
x113 = x115.forward(x114);

// val X105 = ReLU()(X104)
JCudaTensor x116;
JCudaTensor x117;
x117 = x80;
x116 = x118.forward(x117);

// val X106 = Concat(X94,X98,X102,X105)
JCudaTensor x119;
JCudaTensor x121, x122, x123, x124;
x121 = x107;
x122 = x110;
x123 = x113;
x124 = x116;
x119 = x120.forward(x121,x122,x123,x124);

// val X117 = Pooling(3,1,1,true)(X106)
JCudaTensor x125;
JCudaTensor x126;
x126 = x119;
x125 = x127.forward(x126);

// val X113 = Convolv(1,0)(X106,cv24_W,cv24_B)
JCudaTensor x128;
JCudaTensor x129, x130, x131;
x129 = x119;
x130 = x132;
x131 = x133;
x128 = x134.forward(x129,x130,x131);

// val X107 = Convolv(1,0)(X106,cv21_W,cv21_B)
JCudaTensor x135;
JCudaTensor x136, x137, x138;
x136 = x119;
x137 = x139;
x138 = x140;
x135 = x141.forward(x136,x137,x138);

// val X109 = Convolv(1,0)(X106,cv22_W,cv22_B)
JCudaTensor x142;
JCudaTensor x143, x144, x145;
x143 = x119;
x144 = x146;
x145 = x147;
x142 = x148.forward(x143,x144,x145);

// val X110 = ReLU()(X109)
JCudaTensor x149;
JCudaTensor x150;
x150 = x142;
x149 = x151.forward(x150);

// val X118 = Convolv(1,0)(X117,cv26_W,cv26_B)
JCudaTensor x152;
JCudaTensor x153, x154, x155;
x153 = x125;
x154 = x156;
x155 = x157;
x152 = x158.forward(x153,x154,x155);

// val X114 = ReLU()(X113)
JCudaTensor x159;
JCudaTensor x160;
x160 = x128;
x159 = x161.forward(x160);

// val X111 = Convolv(1,1)(X110,cv23_W,cv23_B)
JCudaTensor x162;
JCudaTensor x163, x164, x165;
x163 = x149;
x164 = x166;
x165 = x167;
x162 = x168.forward(x163,x164,x165);

// val X115 = Convolv(1,2)(X114,cv25_W,cv25_B)
JCudaTensor x169;
JCudaTensor x170, x171, x172;
x170 = x159;
x171 = x173;
x172 = x174;
x169 = x175.forward(x170,x171,x172);

// val X108 = ReLU()(X107)
JCudaTensor x176;
JCudaTensor x177;
x177 = x135;
x176 = x178.forward(x177);

// val X112 = ReLU()(X111)
JCudaTensor x179;
JCudaTensor x180;
x180 = x162;
x179 = x181.forward(x180);

// val X116 = ReLU()(X115)
JCudaTensor x182;
JCudaTensor x183;
x183 = x169;
x182 = x184.forward(x183);

// val X119 = ReLU()(X118)
JCudaTensor x185;
JCudaTensor x186;
x186 = x152;
x185 = x187.forward(x186);

// val X120 = Concat(X108,X112,X116,X119)
JCudaTensor x188;
JCudaTensor x189, x190, x191, x192;
x189 = x176;
x190 = x179;
x191 = x182;
x192 = x185;
x188 = x120.forward(x189,x190,x191,x192);

// val X121 = Pooling(3,2,1,true)(X120)
JCudaTensor x193;
JCudaTensor x194;
x194 = x188;
x193 = x195.forward(x194);

// val X122 = Convolv(1,0)(X121,cv31_W,cv31_B)
JCudaTensor x196;
JCudaTensor x197, x198, x199;
x197 = x193;
x198 = x200;
x199 = x201;
x196 = x202.forward(x197,x198,x199);

// val X124 = Convolv(1,0)(X121,cv32_W,cv32_B)
JCudaTensor x203;
JCudaTensor x204, x205, x206;
x204 = x193;
x205 = x207;
x206 = x208;
x203 = x209.forward(x204,x205,x206);

// val X132 = Pooling(3,1,1,true)(X121)
JCudaTensor x210;
JCudaTensor x211;
x211 = x193;
x210 = x212.forward(x211);

// val X128 = Convolv(1,0)(X121,cv34_W,cv34_B)
JCudaTensor x213;
JCudaTensor x214, x215, x216;
x214 = x193;
x215 = x217;
x216 = x218;
x213 = x219.forward(x214,x215,x216);

// val X129 = ReLU()(X128)
JCudaTensor x220;
JCudaTensor x221;
x221 = x213;
x220 = x222.forward(x221);

// val X125 = ReLU()(X124)
JCudaTensor x223;
JCudaTensor x224;
x224 = x203;
x223 = x225.forward(x224);

// val X133 = Convolv(1,0)(X132,cv36_W,cv36_B)
JCudaTensor x226;
JCudaTensor x227, x228, x229;
x227 = x210;
x228 = x230;
x229 = x231;
x226 = x232.forward(x227,x228,x229);

// val X130 = Convolv(1,2)(X129,cv35_W,cv35_B)
JCudaTensor x233;
JCudaTensor x234, x235, x236;
x234 = x220;
x235 = x237;
x236 = x238;
x233 = x239.forward(x234,x235,x236);

// val X126 = Convolv(1,1)(X125,cv33_W,cv33_B)
JCudaTensor x240;
JCudaTensor x241, x242, x243;
x241 = x223;
x242 = x244;
x243 = x245;
x240 = x246.forward(x241,x242,x243);

// val X123 = ReLU()(X122)
JCudaTensor x247;
JCudaTensor x248;
x248 = x196;
x247 = x249.forward(x248);

// val X127 = ReLU()(X126)
JCudaTensor x250;
JCudaTensor x251;
x251 = x240;
x250 = x252.forward(x251);

// val X131 = ReLU()(X130)
JCudaTensor x253;
JCudaTensor x254;
x254 = x233;
x253 = x255.forward(x254);

// val X134 = ReLU()(X133)
JCudaTensor x256;
JCudaTensor x257;
x257 = x226;
x256 = x258.forward(x257);

// val X135 = Concat(X123,X127,X131,X134)
JCudaTensor x259;
JCudaTensor x261, x262, x263, x264;
x261 = x247;
x262 = x250;
x263 = x253;
x264 = x256;
x259 = x260.forward(x261,x262,x263,x264);

// val X241 = Pooling(5,3,0,false)(X135)
JCudaTensor x265;
JCudaTensor x266;
x266 = x259;
x265 = x267.forward(x266);

// val X142 = Convolv(1,0)(X135,cv44_W,cv44_B)
JCudaTensor x268;
JCudaTensor x269, x270, x271;
x269 = x259;
x270 = x272;
x271 = x273;
x268 = x274.forward(x269,x270,x271);

// val X146 = Pooling(3,1,1,true)(X135)
JCudaTensor x275;
JCudaTensor x276;
x276 = x259;
x275 = x277.forward(x276);

// val X138 = Convolv(1,0)(X135,cv42_W,cv42_B)
JCudaTensor x278;
JCudaTensor x279, x280, x281;
x279 = x259;
x280 = x282;
x281 = x283;
x278 = x284.forward(x279,x280,x281);

// val X136 = Convolv(1,0)(X135,cv41_W,cv41_B)
JCudaTensor x285;
JCudaTensor x286, x287, x288;
x286 = x259;
x287 = x289;
x288 = x290;
x285 = x291.forward(x286,x287,x288);

// val X143 = ReLU()(X142)
JCudaTensor x292;
JCudaTensor x293;
x293 = x268;
x292 = x294.forward(x293);

// val X242 = Convolv(1,0)(X241,b1cv_W,b1cv_B)
JCudaTensor x295;
JCudaTensor x296, x297, x298;
x296 = x265;
x297 = x299;
x298 = x300;
x295 = x301.forward(x296,x297,x298);

// val X139 = ReLU()(X138)
JCudaTensor x302;
JCudaTensor x303;
x303 = x278;
x302 = x304.forward(x303);

// val X147 = Convolv(1,0)(X146,cv46_W,cv46_B)
JCudaTensor x305;
JCudaTensor x306, x307, x308;
x306 = x275;
x307 = x309;
x308 = x310;
x305 = x311.forward(x306,x307,x308);

// val X243 = ReLU()(X242)
JCudaTensor x312;
JCudaTensor x313;
x313 = x295;
x312 = x314.forward(x313);

// val X144 = Convolv(1,2)(X143,cv45_W,cv45_B)
JCudaTensor x315;
JCudaTensor x316, x317, x318;
x316 = x292;
x317 = x319;
x318 = x320;
x315 = x321.forward(x316,x317,x318);

// val X140 = Convolv(1,1)(X139,cv43_W,cv43_B)
JCudaTensor x322;
JCudaTensor x323, x324, x325;
x323 = x302;
x324 = x326;
x325 = x327;
x322 = x328.forward(x323,x324,x325);

// val X145 = ReLU()(X144)
JCudaTensor x329;
JCudaTensor x330;
x330 = x315;
x329 = x331.forward(x330);

// val X137 = ReLU()(X136)
JCudaTensor x332;
JCudaTensor x333;
x333 = x285;
x332 = x334.forward(x333);

// val X141 = ReLU()(X140)
JCudaTensor x335;
JCudaTensor x336;
x336 = x322;
x335 = x337.forward(x336);

// val X148 = ReLU()(X147)
JCudaTensor x338;
JCudaTensor x339;
x339 = x305;
x338 = x340.forward(x339);

// val X244 = (X243[1><3])(i | @) * (b1fc1_W)(j | @)
JCudaTensor x341;
JCudaMatrix x342;
JCudaMatrix x343;
JCudaTensor x344;
JCudaTensor x345;
x345 = x312;
x344 = x345.flatten(1, new int[]{128, 4, 4});
x342 = x344.asMatrix(1, true);
JCudaTensor x346;
x346 = x347;
x343 = x346.asMatrix(1, true);
x341 = x342.times(x343);

// val X149 = Concat(X137,X141,X145,X148)
JCudaTensor x348;
JCudaTensor x349, x350, x351, x352;
x349 = x332;
x350 = x335;
x351 = x329;
x352 = x338;
x348 = x260.forward(x349,x350,x351,x352);

// val X246 = (X244 + (i) => b1fc1_B)
JCudaTensor x353;
JCudaTensor x354, x355;
x354 = x341;
x355 = x356;
x353 = x355.copy(128, x354);

// val X152 = Convolv(1,0)(X149,cv52_W,cv52_B)
JCudaTensor x357;
JCudaTensor x358, x359, x360;
x358 = x348;
x359 = x361;
x360 = x362;
x357 = x363.forward(x358,x359,x360);

// val X247 = ReLU()(X246)
JCudaTensor x364;
JCudaTensor x365;
x365 = x353;
x364 = x366.forward(x365);

// val X160 = Pooling(3,1,1,true)(X149)
JCudaTensor x367;
JCudaTensor x368;
x368 = x348;
x367 = x369.forward(x368);

// val X150 = Convolv(1,0)(X149,cv51_W,cv51_B)
JCudaTensor x370;
JCudaTensor x371, x372, x373;
x371 = x348;
x372 = x374;
x373 = x375;
x370 = x376.forward(x371,x372,x373);

// val X156 = Convolv(1,0)(X149,cv54_W,cv54_B)
JCudaTensor x377;
JCudaTensor x378, x379, x380;
x378 = x348;
x379 = x381;
x380 = x382;
x377 = x383.forward(x378,x379,x380);

// val X157 = ReLU()(X156)
JCudaTensor x384;
JCudaTensor x385;
x385 = x377;
x384 = x386.forward(x385);

// val X161 = Convolv(1,0)(X160,cv56_W,cv56_B)
JCudaTensor x387;
JCudaTensor x388, x389, x390;
x388 = x367;
x389 = x391;
x390 = x392;
x387 = x393.forward(x388,x389,x390);

// val X248 = Dropout(0.7)(X247)
JCudaTensor x394;
JCudaTensor x395;
x395 = x364;
x394 = x396.forward(x395);

// val X153 = ReLU()(X152)
JCudaTensor x397;
JCudaTensor x398;
x398 = x357;
x397 = x399.forward(x398);

// val X154 = Convolv(1,1)(X153,cv53_W,cv53_B)
JCudaTensor x400;
JCudaTensor x401, x402, x403;
x401 = x397;
x402 = x404;
x403 = x405;
x400 = x406.forward(x401,x402,x403);

// val X158 = Convolv(1,2)(X157,cv55_W,cv55_B)
JCudaTensor x407;
JCudaTensor x408, x409, x410;
x408 = x384;
x409 = x411;
x410 = x412;
x407 = x413.forward(x408,x409,x410);

// val X249 = (X248)(i | @) * (b1fc2_W)(j | @)
JCudaTensor x414;
JCudaMatrix x415;
JCudaMatrix x416;
JCudaTensor x417;
x417 = x394;
x415 = x417.asMatrix(1, true);
JCudaTensor x418;
x418 = x419;
x416 = x418.asMatrix(1, true);
x414 = x415.times(x416);

// val X155 = ReLU()(X154)
JCudaTensor x420;
JCudaTensor x421;
x421 = x400;
x420 = x422.forward(x421);

// val X162 = ReLU()(X161)
JCudaTensor x423;
JCudaTensor x424;
x424 = x387;
x423 = x425.forward(x424);

// val X159 = ReLU()(X158)
JCudaTensor x426;
JCudaTensor x427;
x427 = x407;
x426 = x428.forward(x427);

// val X151 = ReLU()(X150)
JCudaTensor x429;
JCudaTensor x430;
x430 = x370;
x429 = x431.forward(x430);

// val X251 = (X249 + (i) => b1fc2_B)
JCudaTensor x432;
JCudaTensor x433, x434;
x433 = x414;
x434 = x435;
x432 = x434.copy(128, x433);

// val X163 = Concat(X151,X155,X159,X162)
JCudaTensor x436;
JCudaTensor x437, x438, x439, x440;
x437 = x429;
x438 = x420;
x439 = x426;
x440 = x423;
x436 = x260.forward(x437,x438,x439,x440);

// val X252 = LogSoftmax()(X251)
JCudaTensor x441;
JCudaTensor x442;
x442 = x432;
x441 = x443.forward(x442);

// Dealloc(X251)
JCudaTensor x444;
x444 = x432;
x444.free();

// val X337 = (X336 / |128|)
JCudaTensor x445;
JCudaTensor x446;
float x447;
x446 = x18;
float x448;
x448 = 128;
x447 = 1 / x448;
x445 = x446.times_i(x447);

// val X170 = Convolv(1,0)(X163,cv64_W,cv64_B)
JCudaTensor x449;
JCudaTensor x450, x451, x452;
x450 = x436;
x451 = x453;
x452 = x454;
x449 = x455.forward(x450,x451,x452);

// val X339 = X337 * d_LogSoftmax()(X252)/d_X251
JCudaTensor x456;
JCudaTensor x457, x458;
x457 = x445;
x458 = x441;
x456 = x443.backward(x457,x458);

// val m1 = (i2957) => b1fc2_W[@, i2957]
JCudaMatrix x459;
JCudaTensor x460;
x460 = x419;
x459 = x460.asMatrix(1, false);

// val X166 = Convolv(1,0)(X163,cv62_W,cv62_B)
JCudaTensor x461;
JCudaTensor x462, x463, x464;
x462 = x436;
x463 = x465;
x464 = x466;
x461 = x467.forward(x462,x463,x464);

// val X164 = Convolv(1,0)(X163,cv61_W,cv61_B)
JCudaTensor x468;
JCudaTensor x469, x470, x471;
x469 = x436;
x470 = x472;
x471 = x473;
x468 = x474.forward(x469,x470,x471);

// val X174 = Pooling(3,1,1,true)(X163)
JCudaTensor x475;
JCudaTensor x476;
x476 = x436;
x475 = x477.forward(x476);

// val X175 = Convolv(1,0)(X174,cv66_W,cv66_B)
JCudaTensor x478;
JCudaTensor x479, x480, x481;
x479 = x475;
x480 = x482;
x481 = x483;
x478 = x484.forward(x479,x480,x481);

// val m10 = (i7093) => X339[@, i7093]
JCudaMatrix x485;
JCudaTensor x486;
x486 = x456;
x485 = x486.asMatrix(1, false);

// val X348 = (X339)(i2956 | @) * m1
JCudaTensor x487;
JCudaMatrix x488;
JCudaMatrix x489;
JCudaTensor x490;
x490 = x456;
x488 = x490.asMatrix(1, true);
x489 = x459;
x487 = x488.times(x489);

// val X167 = ReLU()(X166)
JCudaTensor x491;
JCudaTensor x492;
x492 = x461;
x491 = x493.forward(x492);

// val X171 = ReLU()(X170)
JCudaTensor x494;
JCudaTensor x495;
x495 = x449;
x494 = x496.forward(x495);

// val m12 = (i8045) => X248[@, i8045]
JCudaMatrix x497;
JCudaTensor x498;
x498 = x394;
x497 = x498.asMatrix(1, false);

// val X172 = Convolv(1,2)(X171,cv65_W,cv65_B)
JCudaTensor x499;
JCudaTensor x500, x501, x502;
x500 = x494;
x501 = x503;
x502 = x504;
x499 = x505.forward(x500,x501,x502);

// val X349 = X348 * d_Dropout(0.7)()/d_X247
JCudaTensor x506;
JCudaTensor x507;
x507 = x487;
x506 = x396.backward(x507);

// Dealloc(X348)
JCudaTensor x508;
x508 = x487;
x508.free();

// val X742 = Sum(m10)
JCudaTensor x509;
JCudaMatrix x510;
x510 = x485;
x509 = x510.sum();

// val X832 = m10 * m12
JCudaTensor x511;
JCudaMatrix x512;
JCudaMatrix x513;
x512 = x485;
x513 = x497;
x511 = x512.times(x513);

// Dealloc(X339)
JCudaTensor x514;
x514 = x456;
x514.free();

// Dealloc(X248)
JCudaTensor x515;
x515 = x394;
x515.free();

// val X168 = Convolv(1,1)(X167,cv63_W,cv63_B)
JCudaTensor x516;
JCudaTensor x517, x518, x519;
x517 = x491;
x518 = x520;
x519 = x521;
x516 = x522.forward(x517,x518,x519);

// val X833 = (X832 * loss1)
JCudaTensor x523;
JCudaTensor x524;
float x525;
x524 = x511;
x525 = loss1;
x523 = x524.times_i(x525);

// val X176 = ReLU()(X175)
JCudaTensor x526;
JCudaTensor x527;
x527 = x478;
x526 = x528.forward(x527);

// val m2 = (i2977) => b1fc1_W[@, i2977]
JCudaMatrix x529;
JCudaTensor x530;
x530 = x347;
x529 = x530.asMatrix(1, false);

// val X165 = ReLU()(X164)
JCudaTensor x531;
JCudaTensor x532;
x532 = x468;
x531 = x533.forward(x532);

// val X169 = ReLU()(X168)
JCudaTensor x534;
JCudaTensor x535;
x535 = x516;
x534 = x536.forward(x535);

// val X173 = ReLU()(X172)
JCudaTensor x537;
JCudaTensor x538;
x538 = x499;
x537 = x539.forward(x538);

// val X743 = (X742 * loss1)
JCudaTensor x540;
JCudaTensor x541;
float x542;
x541 = x509;
x542 = loss1;
x540 = x541.times_i(x542);

// val X351 = X349 * d_ReLU()(X247)/d_X246
JCudaTensor x543;
JCudaTensor x544, x545;
x544 = x506;
x545 = x364;
x543 = x366.backward(x544,x545);

// Dealloc(X247)
JCudaTensor x546;
x546 = x364;
x546.free();

// val m9 = (i6028) => X243[1><3][@, i6028]
JCudaMatrix x547;
JCudaTensor x548;
JCudaTensor x549;
x549 = x312;
x548 = x549.flatten(1, new int[]{128, 4, 4});
x547 = x548.asMatrix(1, false);

// val X352 = (X351)(i2976 | @) * m2
JCudaTensor x550;
JCudaMatrix x551;
JCudaMatrix x552;
JCudaTensor x553;
x553 = x543;
x551 = x553.asMatrix(1, true);
x552 = x529;
x550 = x551.times(x552);

// val m6 = (i5039) => X351[@, i5039]
JCudaMatrix x554;
JCudaTensor x555;
x555 = x543;
x554 = x555.asMatrix(1, false);

// val X177 = Concat(X165,X169,X173,X176)
JCudaTensor x556;
JCudaTensor x557, x558, x559, x560;
x557 = x531;
x558 = x534;
x559 = x537;
x560 = x526;
x556 = x260.forward(x557,x558,x559,x560);

// V_b1fc2_B <~~ X743
float x562, x563;
x562 = lrn_rate_2;
x563 = momentum;
JCudaTensor x564;
x564 = x540;
x561.update(x564, x562, x563);


// Dealloc(X743)
JCudaTensor x565;
x565 = x540;
x565.free();

// V_b1fc2_W <~~ X833
float x567, x568;
x567 = lrn_rate_1;
x568 = momentum;
JCudaTensor x569;
x569 = x523;
x566.update(x569, x567, x568);


// Dealloc(X833)
JCudaTensor x570;
x570 = x523;
x570.free();

// b1fc2_W <~~ V_b1fc2_W
float x571, x572;
x571 = 1;
x572 = decay_1;
JCudaTensor x573;
x573 = x566;
x419.update(x573, x571, x572);


// b1fc2_B <~~ V_b1fc2_B
float x574, x575;
x574 = 1;
x575 = 1;
JCudaTensor x576;
x576 = x561;
x435.update(x576, x574, x575);


// val X555 = Sum(m6)
JCudaTensor x577;
JCudaMatrix x578;
x578 = x554;
x577 = x578.sum();

// val X188 = Pooling(3,1,1,true)(X177)
JCudaTensor x579;
JCudaTensor x580;
x580 = x556;
x579 = x581.forward(x580);

// val X228 = Pooling(5,3,0,false)(X177)
JCudaTensor x582;
JCudaTensor x583;
x583 = x556;
x582 = x584.forward(x583);

// val X354 = X352[1<>3] * d_ReLU()(X243)/d_X242
JCudaTensor x585;
JCudaTensor x586, x587;
JCudaTensor x588;
x588 = x550;
x586 = x588.unflatten(1, new int[]{128, 4, 4});
x587 = x312;
x585 = x314.backward(x586,x587);

// val X178 = Convolv(1,0)(X177,cv71_W,cv71_B)
JCudaTensor x589;
JCudaTensor x590, x591, x592;
x590 = x556;
x591 = x593;
x592 = x594;
x589 = x595.forward(x590,x591,x592);

// val X180 = Convolv(1,0)(X177,cv72_W,cv72_B)
JCudaTensor x596;
JCudaTensor x597, x598, x599;
x597 = x556;
x598 = x600;
x599 = x601;
x596 = x602.forward(x597,x598,x599);

// val X184 = Convolv(1,0)(X177,cv74_W,cv74_B)
JCudaTensor x603;
JCudaTensor x604, x605, x606;
x604 = x556;
x605 = x607;
x606 = x608;
x603 = x609.forward(x604,x605,x606);

// val X652 = m6 * m9
JCudaTensor x610;
JCudaMatrix x611;
JCudaMatrix x612;
x611 = x554;
x612 = x547;
x610 = x611.times(x612);

// Dealloc(X351)
JCudaTensor x613;
x613 = x543;
x613.free();

// Dealloc(X243)
JCudaTensor x614;
x614 = x312;
x614.free();

// val X355 = X354 * d_Convolv(1,0)()/d_b1cv_B
JCudaTensor x615;
JCudaTensor x616;
x616 = x585;
x615 = x301.backward_bias(x616);

// val X181 = ReLU()(X180)
JCudaTensor x617;
JCudaTensor x618;
x618 = x596;
x617 = x619.forward(x618);

// val X229 = Convolv(1,0)(X228,b2cv_W,b2cv_B)
JCudaTensor x620;
JCudaTensor x621, x622, x623;
x621 = x582;
x622 = x624;
x623 = x625;
x620 = x626.forward(x621,x622,x623);

// val X2810 = X354 * d_Convolv(1,0)(b1cv_W)/d_X241
JCudaTensor x627;
JCudaTensor x628, x629;
x628 = x585;
x629 = x299;
x627 = x301.backward_data(x628,x629);

// val X653 = (X652 * loss1)
JCudaTensor x630;
JCudaTensor x631;
float x632;
x631 = x610;
x632 = loss1;
x630 = x631.times_i(x632);

// val X185 = ReLU()(X184)
JCudaTensor x633;
JCudaTensor x634;
x634 = x603;
x633 = x635.forward(x634);

// val X189 = Convolv(1,0)(X188,cv76_W,cv76_B)
JCudaTensor x636;
JCudaTensor x637, x638, x639;
x637 = x579;
x638 = x640;
x639 = x641;
x636 = x642.forward(x637,x638,x639);

// val X458 = X354 * d_Convolv(1,0)(X241)/d_b1cv_W
JCudaTensor x643;
JCudaTensor x644, x645;
x644 = x585;
x645 = x265;
x643 = x301.backward_filter(x644,x645);

// Dealloc(X354)
JCudaTensor x646;
x646 = x585;
x646.free();

// val X556 = (X555 * loss1)
JCudaTensor x647;
JCudaTensor x648;
float x649;
x648 = x577;
x649 = loss1;
x647 = x648.times_i(x649);

// V_b1fc1_W <~~ X653
float x651, x652;
x651 = lrn_rate_1;
x652 = momentum;
JCudaTensor x653;
x653 = x630;
x650.update(x653, x651, x652);


// Dealloc(X653)
JCudaTensor x654;
x654 = x630;
x654.free();

// val X230 = ReLU()(X229)
JCudaTensor x655;
JCudaTensor x656;
x656 = x620;
x655 = x657.forward(x656);

// val X2812 = X2810 * d_Pooling(5,3,0,false)(X241,X135)/d_X135
JCudaTensor x658;
JCudaTensor x659, x660, x661;
x659 = x627;
x660 = x265;
x661 = x259;
x658 = x267.backward(x659,x660,x661);

// Dealloc(X2810)
JCudaTensor x662;
x662 = x627;
x662.free();

// Dealloc(X241)
JCudaTensor x663;
x663 = x265;
x663.free();

// V_b1fc1_B <~~ X556
float x665, x666;
x665 = lrn_rate_2;
x666 = momentum;
JCudaTensor x667;
x667 = x647;
x664.update(x667, x665, x666);


// Dealloc(X556)
JCudaTensor x668;
x668 = x647;
x668.free();

// val X356 = (X355 * loss1)
JCudaTensor x669;
JCudaTensor x670;
float x671;
x670 = x615;
x671 = loss1;
x669 = x670.times_i(x671);

// val X186 = Convolv(1,2)(X185,cv75_W,cv75_B)
JCudaTensor x672;
JCudaTensor x673, x674, x675;
x673 = x633;
x674 = x676;
x675 = x677;
x672 = x678.forward(x673,x674,x675);

// val X459 = (X458 * loss1)
JCudaTensor x679;
JCudaTensor x680;
float x681;
x680 = x643;
x681 = loss1;
x679 = x680.times_i(x681);

// val X182 = Convolv(1,1)(X181,cv73_W,cv73_B)
JCudaTensor x682;
JCudaTensor x683, x684, x685;
x683 = x617;
x684 = x686;
x685 = x687;
x682 = x688.forward(x683,x684,x685);

// b1fc1_B <~~ V_b1fc1_B
float x689, x690;
x689 = 1;
x690 = 1;
JCudaTensor x691;
x691 = x664;
x356.update(x691, x689, x690);


// b1fc1_W <~~ V_b1fc1_W
float x692, x693;
x692 = 1;
x693 = decay_1;
JCudaTensor x694;
x694 = x650;
x347.update(x694, x692, x693);


// val X187 = ReLU()(X186)
JCudaTensor x695;
JCudaTensor x696;
x696 = x672;
x695 = x697.forward(x696);

// val X183 = ReLU()(X182)
JCudaTensor x698;
JCudaTensor x699;
x699 = x682;
x698 = x700.forward(x699);

// val X190 = ReLU()(X189)
JCudaTensor x701;
JCudaTensor x702;
x702 = x636;
x701 = x703.forward(x702);

// V_b1cv_B <~~ X356
float x705, x706;
x705 = lrn_rate_2;
x706 = momentum;
JCudaTensor x707;
x707 = x669;
x704.update(x707, x705, x706);


// Dealloc(X356)
JCudaTensor x708;
x708 = x669;
x708.free();

// V_b1cv_W <~~ X459
float x710, x711;
x710 = lrn_rate_1;
x711 = momentum;
JCudaTensor x712;
x712 = x679;
x709.update(x712, x710, x711);


// Dealloc(X459)
JCudaTensor x713;
x713 = x679;
x713.free();

// val X231 = (X230[1><3])(i | @) * (b2fc1_W)(j | @)
JCudaTensor x714;
JCudaMatrix x715;
JCudaMatrix x716;
JCudaTensor x717;
JCudaTensor x718;
x718 = x655;
x717 = x718.flatten(1, new int[]{128, 4, 4});
x715 = x717.asMatrix(1, true);
JCudaTensor x719;
x719 = x720;
x716 = x719.asMatrix(1, true);
x714 = x715.times(x716);

// val X179 = ReLU()(X178)
JCudaTensor x721;
JCudaTensor x722;
x722 = x589;
x721 = x723.forward(x722);

// b1cv_B <~~ V_b1cv_B
float x724, x725;
x724 = 1;
x725 = 1;
JCudaTensor x726;
x726 = x704;
x300.update(x726, x724, x725);


// b1cv_W <~~ V_b1cv_W
float x727, x728;
x727 = 1;
x728 = decay_1;
JCudaTensor x729;
x729 = x709;
x299.update(x729, x727, x728);


// val X233 = (X231 + (i) => b2fc1_B)
JCudaTensor x730;
JCudaTensor x731, x732;
x731 = x714;
x732 = x733;
x730 = x732.copy(128, x731);

// val X191 = Concat(X179,X183,X187,X190)
JCudaTensor x734;
JCudaTensor x735, x736, x737, x738;
x735 = x721;
x736 = x698;
x737 = x695;
x738 = x701;
x734 = x260.forward(x735,x736,x737,x738);

// val X192 = Pooling(3,2,1,true)(X191)
JCudaTensor x739;
JCudaTensor x740;
x740 = x734;
x739 = x741.forward(x740);

// val X234 = ReLU()(X233)
JCudaTensor x742;
JCudaTensor x743;
x743 = x730;
x742 = x366.forward(x743);

// val X193 = Convolv(1,0)(X192,cv81_W,cv81_B)
JCudaTensor x744;
JCudaTensor x745, x746, x747;
x745 = x739;
x746 = x748;
x747 = x749;
x744 = x750.forward(x745,x746,x747);

// val X203 = Pooling(3,1,1,true)(X192)
JCudaTensor x751;
JCudaTensor x752;
x752 = x739;
x751 = x753.forward(x752);

// val X235 = Dropout(0.7)(X234)
JCudaTensor x754;
JCudaTensor x755;
x755 = x742;
x754 = x396.forward(x755);

// val X195 = Convolv(1,0)(X192,cv82_W,cv82_B)
JCudaTensor x756;
JCudaTensor x757, x758, x759;
x757 = x739;
x758 = x760;
x759 = x761;
x756 = x762.forward(x757,x758,x759);

// val X199 = Convolv(1,0)(X192,cv84_W,cv84_B)
JCudaTensor x763;
JCudaTensor x764, x765, x766;
x764 = x739;
x765 = x767;
x766 = x768;
x763 = x769.forward(x764,x765,x766);

// val X204 = Convolv(1,0)(X203,cv86_W,cv86_B)
JCudaTensor x770;
JCudaTensor x771, x772, x773;
x771 = x751;
x772 = x774;
x773 = x775;
x770 = x776.forward(x771,x772,x773);

// val X196 = ReLU()(X195)
JCudaTensor x777;
JCudaTensor x778;
x778 = x756;
x777 = x779.forward(x778);

// val X200 = ReLU()(X199)
JCudaTensor x780;
JCudaTensor x781;
x781 = x763;
x780 = x782.forward(x781);

// val X236 = (X235)(i | @) * (b2fc2_W)(j | @)
JCudaTensor x783;
JCudaMatrix x784;
JCudaMatrix x785;
JCudaTensor x786;
x786 = x754;
x784 = x786.asMatrix(1, true);
JCudaTensor x787;
x787 = x788;
x785 = x787.asMatrix(1, true);
x783 = x784.times(x785);

// val X238 = (X236 + (i) => b2fc2_B)
JCudaTensor x789;
JCudaTensor x790, x791;
x790 = x783;
x791 = x792;
x789 = x791.copy(128, x790);

// val X197 = Convolv(1,1)(X196,cv83_W,cv83_B)
JCudaTensor x793;
JCudaTensor x794, x795, x796;
x794 = x777;
x795 = x797;
x796 = x798;
x793 = x799.forward(x794,x795,x796);

// val X201 = Convolv(1,2)(X200,cv85_W,cv85_B)
JCudaTensor x800;
JCudaTensor x801, x802, x803;
x801 = x780;
x802 = x804;
x803 = x805;
x800 = x806.forward(x801,x802,x803);

// val X239 = LogSoftmax()(X238)
JCudaTensor x807;
JCudaTensor x808;
x808 = x789;
x807 = x443.forward(x808);

// Dealloc(X238)
JCudaTensor x809;
x809 = x789;
x809.free();

// val X205 = ReLU()(X204)
JCudaTensor x810;
JCudaTensor x811;
x811 = x770;
x810 = x812.forward(x811);

// val X198 = ReLU()(X197)
JCudaTensor x813;
JCudaTensor x814;
x814 = x793;
x813 = x815.forward(x814);

// val X202 = ReLU()(X201)
JCudaTensor x816;
JCudaTensor x817;
x817 = x800;
x816 = x818.forward(x817);

// val X194 = ReLU()(X193)
JCudaTensor x819;
JCudaTensor x820;
x820 = x744;
x819 = x821.forward(x820);

// val X961 = X337 * d_LogSoftmax()(X239)/d_X238
JCudaTensor x822;
JCudaTensor x823, x824;
x823 = x445;
x824 = x807;
x822 = x443.backward(x823,x824);

// val X206 = Concat(X194,X198,X202,X205)
JCudaTensor x825;
JCudaTensor x827, x828, x829, x830;
x827 = x819;
x828 = x813;
x829 = x816;
x830 = x810;
x825 = x826.forward(x827,x828,x829,x830);

// val m13 = (i9005) => b2fc2_W[@, i9005]
JCudaMatrix x831;
JCudaTensor x832;
x832 = x788;
x831 = x832.asMatrix(1, false);

// val m22 = (i15235) => X961[@, i15235]
JCudaMatrix x833;
JCudaTensor x834;
x834 = x822;
x833 = x834.asMatrix(1, false);

// val X209 = Convolv(1,0)(X206,cv92_W,cv92_B)
JCudaTensor x835;
JCudaTensor x836, x837, x838;
x836 = x825;
x837 = x839;
x838 = x840;
x835 = x841.forward(x836,x837,x838);

// val X207 = Convolv(1,0)(X206,cv91_W,cv91_B)
JCudaTensor x842;
JCudaTensor x843, x844, x845;
x843 = x825;
x844 = x846;
x845 = x847;
x842 = x848.forward(x843,x844,x845);

// val X213 = Convolv(1,0)(X206,cv94_W,cv94_B)
JCudaTensor x849;
JCudaTensor x850, x851, x852;
x850 = x825;
x851 = x853;
x852 = x854;
x849 = x855.forward(x850,x851,x852);

// val X970 = (X961)(i9004 | @) * m13
JCudaTensor x856;
JCudaMatrix x857;
JCudaMatrix x858;
JCudaTensor x859;
x859 = x822;
x857 = x859.asMatrix(1, true);
x858 = x831;
x856 = x857.times(x858);

// val m24 = (i16701) => X235[@, i16701]
JCudaMatrix x860;
JCudaTensor x861;
x861 = x754;
x860 = x861.asMatrix(1, false);

// val X217 = Pooling(3,1,1,true)(X206)
JCudaTensor x862;
JCudaTensor x863;
x863 = x825;
x862 = x864.forward(x863);

// val X1664 = m22 * m24
JCudaTensor x865;
JCudaMatrix x866;
JCudaMatrix x867;
x866 = x833;
x867 = x860;
x865 = x866.times(x867);

// Dealloc(X235)
JCudaTensor x868;
x868 = x754;
x868.free();

// val X214 = ReLU()(X213)
JCudaTensor x869;
JCudaTensor x870;
x870 = x849;
x869 = x871.forward(x870);

// val X218 = Convolv(1,0)(X217,cv96_W,cv96_B)
JCudaTensor x872;
JCudaTensor x873, x874, x875;
x873 = x862;
x874 = x876;
x875 = x877;
x872 = x878.forward(x873,x874,x875);

// val X971 = X970 * d_Dropout(0.7)()/d_X234
JCudaTensor x879;
JCudaTensor x880;
x880 = x856;
x879 = x396.backward(x880);

// Dealloc(X970)
JCudaTensor x881;
x881 = x856;
x881.free();

// val X210 = ReLU()(X209)
JCudaTensor x882;
JCudaTensor x883;
x883 = x835;
x882 = x884.forward(x883);

// val X1532 = Sum(m22)
JCudaTensor x885;
JCudaMatrix x886;
x886 = x833;
x885 = x886.sum();

// Dealloc(X961)
JCudaTensor x887;
x887 = x822;
x887.free();

// val X211 = Convolv(1,1)(X210,cv93_W,cv93_B)
JCudaTensor x888;
JCudaTensor x889, x890, x891;
x889 = x882;
x890 = x892;
x891 = x893;
x888 = x894.forward(x889,x890,x891);

// val X1665 = (X1664 * loss2)
JCudaTensor x895;
JCudaTensor x896;
float x897;
x896 = x865;
x897 = loss2;
x895 = x896.times_i(x897);

// val m14 = (i9025) => b2fc1_W[@, i9025]
JCudaMatrix x898;
JCudaTensor x899;
x899 = x720;
x898 = x899.asMatrix(1, false);

// val X215 = Convolv(1,2)(X214,cv95_W,cv95_B)
JCudaTensor x900;
JCudaTensor x901, x902, x903;
x901 = x869;
x902 = x904;
x903 = x905;
x900 = x906.forward(x901,x902,x903);

// val X973 = X971 * d_ReLU()(X234)/d_X233
JCudaTensor x907;
JCudaTensor x908, x909;
x908 = x879;
x909 = x742;
x907 = x366.backward(x908,x909);

// Dealloc(X234)
JCudaTensor x910;
x910 = x742;
x910.free();

// val X1533 = (X1532 * loss2)
JCudaTensor x911;
JCudaTensor x912;
float x913;
x912 = x885;
x913 = loss2;
x911 = x912.times_i(x913);

// val X974 = (X973)(i9024 | @) * m14
JCudaTensor x914;
JCudaMatrix x915;
JCudaMatrix x916;
JCudaTensor x917;
x917 = x907;
x915 = x917.asMatrix(1, true);
x916 = x898;
x914 = x915.times(x916);

// val m18 = (i12111) => X973[@, i12111]
JCudaMatrix x918;
JCudaTensor x919;
x919 = x907;
x918 = x919.asMatrix(1, false);

// V_b2fc2_B <~~ X1533
float x921, x922;
x921 = lrn_rate_2;
x922 = momentum;
JCudaTensor x923;
x923 = x911;
x920.update(x923, x921, x922);


// Dealloc(X1533)
JCudaTensor x924;
x924 = x911;
x924.free();

// val X208 = ReLU()(X207)
JCudaTensor x925;
JCudaTensor x926;
x926 = x842;
x925 = x927.forward(x926);

// val X212 = ReLU()(X211)
JCudaTensor x928;
JCudaTensor x929;
x929 = x888;
x928 = x930.forward(x929);

// val X216 = ReLU()(X215)
JCudaTensor x931;
JCudaTensor x932;
x932 = x900;
x931 = x933.forward(x932);

// val X219 = ReLU()(X218)
JCudaTensor x934;
JCudaTensor x935;
x935 = x872;
x934 = x936.forward(x935);

// val m21 = (i13613) => X230[1><3][@, i13613]
JCudaMatrix x937;
JCudaTensor x938;
JCudaTensor x939;
x939 = x655;
x938 = x939.flatten(1, new int[]{128, 4, 4});
x937 = x938.asMatrix(1, false);

// V_b2fc2_W <~~ X1665
float x941, x942;
x941 = lrn_rate_1;
x942 = momentum;
JCudaTensor x943;
x943 = x895;
x940.update(x943, x941, x942);


// Dealloc(X1665)
JCudaTensor x944;
x944 = x895;
x944.free();

// b2fc2_B <~~ V_b2fc2_B
float x945, x946;
x945 = 1;
x946 = 1;
JCudaTensor x947;
x947 = x920;
x792.update(x947, x945, x946);


// b2fc2_W <~~ V_b2fc2_W
float x948, x949;
x948 = 1;
x949 = decay_1;
JCudaTensor x950;
x950 = x940;
x788.update(x950, x948, x949);


// val X976 = X974[1<>3] * d_ReLU()(X230)/d_X229
JCudaTensor x951;
JCudaTensor x952, x953;
JCudaTensor x954;
x954 = x914;
x952 = x954.unflatten(1, new int[]{128, 4, 4});
x953 = x655;
x951 = x657.backward(x952,x953);

// val X1261 = Sum(m18)
JCudaTensor x955;
JCudaMatrix x956;
x956 = x918;
x955 = x956.sum();

// val X220 = Concat(X208,X212,X216,X219)
JCudaTensor x957;
JCudaTensor x958, x959, x960, x961;
x958 = x925;
x959 = x928;
x960 = x931;
x961 = x934;
x957 = x826.forward(x958,x959,x960,x961);

// val X1400 = m18 * m21
JCudaTensor x962;
JCudaMatrix x963;
JCudaMatrix x964;
x963 = x918;
x964 = x937;
x962 = x963.times(x964);

// Dealloc(X973)
JCudaTensor x965;
x965 = x907;
x965.free();

// Dealloc(X230)
JCudaTensor x966;
x966 = x655;
x966.free();

// val X977 = X976 * d_Convolv(1,0)()/d_b2cv_B
JCudaTensor x967;
JCudaTensor x968;
x968 = x951;
x967 = x626.backward_bias(x968);

// val X1262 = (X1261 * loss2)
JCudaTensor x969;
JCudaTensor x970;
float x971;
x970 = x955;
x971 = loss2;
x969 = x970.times_i(x971);

// val X1122 = X976 * d_Convolv(1,0)(X228)/d_b2cv_W
JCudaTensor x972;
JCudaTensor x973, x974;
x973 = x951;
x974 = x582;
x972 = x626.backward_filter(x973,x974);

// val X2481 = X976 * d_Convolv(1,0)(b2cv_W)/d_X228
JCudaTensor x975;
JCudaTensor x976, x977;
x976 = x951;
x977 = x624;
x975 = x626.backward_data(x976,x977);

// Dealloc(X976)
JCudaTensor x978;
x978 = x951;
x978.free();

// val X221 = Pooling(7,1,0,false)(X220)
JCudaTensor x979;
JCudaTensor x980;
x980 = x957;
x979 = x981.forward(x980);

// val X1401 = (X1400 * loss2)
JCudaTensor x982;
JCudaTensor x983;
float x984;
x983 = x962;
x984 = loss2;
x982 = x983.times_i(x984);

// V_b2fc1_B <~~ X1262
float x986, x987;
x986 = lrn_rate_2;
x987 = momentum;
JCudaTensor x988;
x988 = x969;
x985.update(x988, x986, x987);


// Dealloc(X1262)
JCudaTensor x989;
x989 = x969;
x989.free();

// V_b2fc1_W <~~ X1401
float x991, x992;
x991 = lrn_rate_1;
x992 = momentum;
JCudaTensor x993;
x993 = x982;
x990.update(x993, x991, x992);


// Dealloc(X1401)
JCudaTensor x994;
x994 = x982;
x994.free();

// val X2483 = X2481 * d_Pooling(5,3,0,false)(X228,X177)/d_X177
JCudaTensor x995;
JCudaTensor x996, x997, x998;
x996 = x975;
x997 = x582;
x998 = x556;
x995 = x584.backward(x996,x997,x998);

// Dealloc(X2481)
JCudaTensor x999;
x999 = x975;
x999.free();

// Dealloc(X228)
JCudaTensor x1000;
x1000 = x582;
x1000.free();

// val X978 = (X977 * loss2)
JCudaTensor x1001;
JCudaTensor x1002;
float x1003;
x1002 = x967;
x1003 = loss2;
x1001 = x1002.times_i(x1003);

// val X1123 = (X1122 * loss2)
JCudaTensor x1004;
JCudaTensor x1005;
float x1006;
x1005 = x972;
x1006 = loss2;
x1004 = x1005.times_i(x1006);

// val X222 = Dropout(0.4)(X221)
JCudaTensor x1007;
JCudaTensor x1008;
x1008 = x979;
x1007 = x1009.forward(x1008);

// b2fc1_W <~~ V_b2fc1_W
float x1010, x1011;
x1010 = 1;
x1011 = decay_1;
JCudaTensor x1012;
x1012 = x990;
x720.update(x1012, x1010, x1011);


// b2fc1_B <~~ V_b2fc1_B
float x1013, x1014;
x1013 = 1;
x1014 = 1;
JCudaTensor x1015;
x1015 = x985;
x733.update(x1015, x1013, x1014);


// V_b2cv_W <~~ X1123
float x1017, x1018;
x1017 = lrn_rate_1;
x1018 = momentum;
JCudaTensor x1019;
x1019 = x1004;
x1016.update(x1019, x1017, x1018);


// Dealloc(X1123)
JCudaTensor x1020;
x1020 = x1004;
x1020.free();

// val X223 = (X222[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x1021;
JCudaMatrix x1022;
JCudaMatrix x1023;
JCudaTensor x1024;
JCudaTensor x1025;
x1025 = x1007;
x1024 = x1025.flatten(1, new int[]{256, 1, 1});
x1022 = x1024.asMatrix(1, true);
JCudaTensor x1026;
x1026 = x1027;
x1023 = x1026.asMatrix(1, true);
x1021 = x1022.times(x1023);

// V_b2cv_B <~~ X978
float x1029, x1030;
x1029 = lrn_rate_2;
x1030 = momentum;
JCudaTensor x1031;
x1031 = x1001;
x1028.update(x1031, x1029, x1030);


// Dealloc(X978)
JCudaTensor x1032;
x1032 = x1001;
x1032.free();

// b2cv_W <~~ V_b2cv_W
float x1033, x1034;
x1033 = 1;
x1034 = decay_1;
JCudaTensor x1035;
x1035 = x1016;
x624.update(x1035, x1033, x1034);


// b2cv_B <~~ V_b2cv_B
float x1036, x1037;
x1036 = 1;
x1037 = 1;
JCudaTensor x1038;
x1038 = x1028;
x625.update(x1038, x1036, x1037);


// val X225 = (X223 + (i) => fc_B)
JCudaTensor x1039;
JCudaTensor x1040, x1041;
x1040 = x1021;
x1041 = x1042;
x1039 = x1041.copy(128, x1040);

// val X226 = LogSoftmax()(X225)
JCudaTensor x1043;
JCudaTensor x1044;
x1044 = x1039;
x1043 = x443.forward(x1044);

// Dealloc(X225)
JCudaTensor x1045;
x1045 = x1039;
x1045.free();

// Print(((((0 - (X227 . X226)) / |128|) + (((0 - (X227 . X239)) / |128|) * loss2)) + (((0 - (X227 . X252)) / |128|) * loss1)))
float x1046;
float x1047;
float x1048;
float x1049;
float x1050;
float x1051;
float x1052;
float x1053;
JCudaTensor x1054, x1055;
x1054 = x9;
x1055 = x1043;
x1053 = x1054.dot(x1055);
x1051 = - x1053;
x1052 = 128;
x1049 = x1051 / x1052;
float x1056;
float x1057;
float x1058;
float x1059;
float x1060;
JCudaTensor x1061, x1062;
x1061 = x9;
x1062 = x807;
x1060 = x1061.dot(x1062);
x1058 = - x1060;
x1059 = 128;
x1056 = x1058 / x1059;
x1057 = loss2;
x1050 = x1056 * x1057;
x1047 = x1049 + x1050;
float x1063;
float x1064;
float x1065;
float x1066;
float x1067;
JCudaTensor x1068, x1069;
x1068 = x9;
x1069 = x441;
x1067 = x1068.dot(x1069);
x1065 = - x1067;
x1066 = 128;
x1063 = x1065 / x1066;
x1064 = loss1;
x1048 = x1063 * x1064;
x1046 = x1047 + x1048;
System.out.println(x5 + " " + x1046);
if (Float.isNaN(x1046)) { System.exit(-1); }

// Dealloc(X239)
JCudaTensor x1070;
x1070 = x807;
x1070.free();

// Dealloc(X252)
JCudaTensor x1071;
x1071 = x441;
x1071.free();

// Dealloc(X227)
JCudaTensor x1072;
x1072 = x9;
x1072.free();

// val X2133 = X337 * d_LogSoftmax()(X226)/d_X225
JCudaTensor x1073;
JCudaTensor x1074, x1075;
x1074 = x445;
x1075 = x1043;
x1073 = x443.backward(x1074,x1075);

// Dealloc(X226)
JCudaTensor x1076;
x1076 = x1043;
x1076.free();

// Dealloc(X337)
JCudaTensor x1077;
x1077 = x445;
x1077.free();

// val m25 = (i18177) => fc_W[@, i18177]
JCudaMatrix x1078;
JCudaTensor x1079;
x1079 = x1027;
x1078 = x1079.asMatrix(1, false);

// val m379 = (i1056723) => X2133[@, i1056723]
JCudaMatrix x1080;
JCudaTensor x1081;
x1081 = x1073;
x1080 = x1081.asMatrix(1, false);

// val m381 = (i1058948) => X222[1><3][@, i1058948]
JCudaMatrix x1082;
JCudaTensor x1083;
JCudaTensor x1084;
x1084 = x1007;
x1083 = x1084.flatten(1, new int[]{256, 1, 1});
x1082 = x1083.asMatrix(1, false);

// val X2179 = (X2133)(i18176 | @) * m25
JCudaTensor x1085;
JCudaMatrix x1086;
JCudaMatrix x1087;
JCudaTensor x1088;
x1088 = x1073;
x1086 = x1088.asMatrix(1, true);
x1087 = x1078;
x1085 = x1086.times(x1087);

// val X2180 = X2179[1<>3] * d_Dropout(0.4)()/d_X221
JCudaTensor x1089;
JCudaTensor x1090;
JCudaTensor x1091;
x1091 = x1085;
x1090 = x1091.unflatten(1, new int[]{256, 1, 1});
x1089 = x1009.backward(x1090);

// Dealloc(X2179)
JCudaTensor x1092;
x1092 = x1085;
x1092.free();

// V_fc_W <~~ m379 * m381
float x1094, x1095;
x1094 = lrn_rate_1;
x1095 = momentum;
JCudaMatrix x1096;
JCudaMatrix x1097;
x1096 = x1080;
x1097 = x1082;
x1096.times(x1097, x1093, x1094, x1095);

// Dealloc(X222)
JCudaTensor x1098;
x1098 = x1007;
x1098.free();

// V_fc_B <~~ Sum(m379)
float x1100, x1101;
x1100 = lrn_rate_2;
x1101 = momentum;
JCudaMatrix x1102;
x1102 = x1080;
x1102.sum(x1099, x1100, x1101);

// Dealloc(X2133)
JCudaTensor x1103;
x1103 = x1073;
x1103.free();

// fc_W <~~ V_fc_W
float x1104, x1105;
x1104 = 1;
x1105 = decay_1;
JCudaTensor x1106;
x1106 = x1093;
x1027.update(x1106, x1104, x1105);


// fc_B <~~ V_fc_B
float x1107, x1108;
x1107 = 1;
x1108 = 1;
JCudaTensor x1109;
x1109 = x1099;
x1042.update(x1109, x1107, x1108);


// val X2182 = X2180 * d_Pooling(7,1,0,false)(X221,X220)/d_X220
JCudaTensor x1110;
JCudaTensor x1111, x1112, x1113;
x1111 = x1089;
x1112 = x979;
x1113 = x957;
x1110 = x981.backward(x1111,x1112,x1113);

// Dealloc(X2180)
JCudaTensor x1114;
x1114 = x1089;
x1114.free();

// Dealloc(X221)
JCudaTensor x1115;
x1115 = x979;
x1115.free();

// Dealloc(X220)
JCudaTensor x1116;
x1116 = x957;
x1116.free();

// val X2262 = Proj(X2182, X208,X212,X216,X219, 3)
JCudaTensor x1117;
JCudaTensor x1119;
x1119 = x1110;
JCudaTensor[] x1118 = x826.backward(x1119);
x1117 = x1118[3];

// val X2238 = Proj(X2182, X208,X212,X216,X219, 2)
JCudaTensor x1120;
x1120 = x1118[2];

// val X2214 = Proj(X2182, X208,X212,X216,X219, 1)
JCudaTensor x1121;
x1121 = x1118[1];

// val X2196 = Proj(X2182, X208,X212,X216,X219, 0)
JCudaTensor x1122;
x1122 = x1118[0];

// Dealloc(X2182)
JCudaTensor x1123;
x1123 = x1110;
x1123.free();

// val X2243 = X2238 * d_ReLU()(X216)/d_X215
JCudaTensor x1124;
JCudaTensor x1125, x1126;
x1125 = x1120;
x1126 = x931;
x1124 = x933.backward(x1125,x1126);

// Dealloc(X216)
JCudaTensor x1127;
x1127 = x931;
x1127.free();

// val X2266 = X2262 * d_ReLU()(X219)/d_X218
JCudaTensor x1128;
JCudaTensor x1129, x1130;
x1129 = x1117;
x1130 = x934;
x1128 = x936.backward(x1129,x1130);

// Dealloc(X219)
JCudaTensor x1131;
x1131 = x934;
x1131.free();

// val X2199 = X2196 * d_ReLU()(X208)/d_X207
JCudaTensor x1132;
JCudaTensor x1133, x1134;
x1133 = x1122;
x1134 = x925;
x1132 = x927.backward(x1133,x1134);

// Dealloc(X208)
JCudaTensor x1135;
x1135 = x925;
x1135.free();

// val X2219 = X2214 * d_ReLU()(X212)/d_X211
JCudaTensor x1136;
JCudaTensor x1137, x1138;
x1137 = x1121;
x1138 = x928;
x1136 = x930.backward(x1137,x1138);

// Dealloc(X212)
JCudaTensor x1139;
x1139 = x928;
x1139.free();

// val X2244 = X2243 * d_Convolv(1,2)(cv95_W)/d_X214
JCudaTensor x1140;
JCudaTensor x1141, x1142;
x1141 = x1124;
x1142 = x904;
x1140 = x906.backward_data(x1141,x1142);

// V_cv96_B <~~ X2266 * d_Convolv(1,0)()/d_cv96_B
float x1144, x1145;
x1144 = lrn_rate_2;
x1145 = momentum;
JCudaTensor x1146;
x1146 = x1128;
x878.backward_bias(x1146, x1143, x1144, x1145);

// V_cv91_W <~~ X2199 * d_Convolv(1,0)(X206)/d_cv91_W
float x1148, x1149;
x1148 = lrn_rate_1;
x1149 = momentum;
JCudaTensor x1150, x1151;
x1150 = x1132;
x1151 = x825;
x848.backward_filter(x1150,x1151, x1147, x1148, x1149);

// val X2220 = X2219 * d_Convolv(1,1)(cv93_W)/d_X210
JCudaTensor x1152;
JCudaTensor x1153, x1154;
x1153 = x1136;
x1154 = x892;
x1152 = x894.backward_data(x1153,x1154);

// val X2200 = X2199 * d_Convolv(1,0)(cv91_W)/d_X206
JCudaTensor x1155;
JCudaTensor x1156, x1157;
x1156 = x1132;
x1157 = x846;
x1155 = x848.backward_data(x1156,x1157);

// V_cv96_W <~~ X2266 * d_Convolv(1,0)(X217)/d_cv96_W
float x1159, x1160;
x1159 = lrn_rate_1;
x1160 = momentum;
JCudaTensor x1161, x1162;
x1161 = x1128;
x1162 = x862;
x878.backward_filter(x1161,x1162, x1158, x1159, x1160);

// V_cv91_B <~~ X2199 * d_Convolv(1,0)()/d_cv91_B
float x1164, x1165;
x1164 = lrn_rate_2;
x1165 = momentum;
JCudaTensor x1166;
x1166 = x1132;
x848.backward_bias(x1166, x1163, x1164, x1165);

// Dealloc(X2199)
JCudaTensor x1167;
x1167 = x1132;
x1167.free();

// val X2267 = X2266 * d_Convolv(1,0)(cv96_W)/d_X217
JCudaTensor x1168;
JCudaTensor x1169, x1170;
x1169 = x1128;
x1170 = x876;
x1168 = x878.backward_data(x1169,x1170);

// Dealloc(X2266)
JCudaTensor x1171;
x1171 = x1128;
x1171.free();

// V_cv95_B <~~ X2243 * d_Convolv(1,2)()/d_cv95_B
float x1173, x1174;
x1173 = lrn_rate_2;
x1174 = momentum;
JCudaTensor x1175;
x1175 = x1124;
x906.backward_bias(x1175, x1172, x1173, x1174);

// V_cv93_W <~~ X2219 * d_Convolv(1,1)(X210)/d_cv93_W
float x1177, x1178;
x1177 = lrn_rate_1;
x1178 = momentum;
JCudaTensor x1179, x1180;
x1179 = x1136;
x1180 = x882;
x894.backward_filter(x1179,x1180, x1176, x1177, x1178);

// V_cv95_W <~~ X2243 * d_Convolv(1,2)(X214)/d_cv95_W
float x1182, x1183;
x1182 = lrn_rate_1;
x1183 = momentum;
JCudaTensor x1184, x1185;
x1184 = x1124;
x1185 = x869;
x906.backward_filter(x1184,x1185, x1181, x1182, x1183);

// Dealloc(X2243)
JCudaTensor x1186;
x1186 = x1124;
x1186.free();

// V_cv93_B <~~ X2219 * d_Convolv(1,1)()/d_cv93_B
float x1188, x1189;
x1188 = lrn_rate_2;
x1189 = momentum;
JCudaTensor x1190;
x1190 = x1136;
x894.backward_bias(x1190, x1187, x1188, x1189);

// Dealloc(X2219)
JCudaTensor x1191;
x1191 = x1136;
x1191.free();

// cv93_W <~~ V_cv93_W
float x1192, x1193;
x1192 = 1;
x1193 = decay_1;
JCudaTensor x1194;
x1194 = x1176;
x892.update(x1194, x1192, x1193);


// cv91_W <~~ V_cv91_W
float x1195, x1196;
x1195 = 1;
x1196 = decay_1;
JCudaTensor x1197;
x1197 = x1147;
x846.update(x1197, x1195, x1196);


// cv91_B <~~ V_cv91_B
float x1198, x1199;
x1198 = 1;
x1199 = 1;
JCudaTensor x1200;
x1200 = x1163;
x847.update(x1200, x1198, x1199);


// cv96_W <~~ V_cv96_W
float x1201, x1202;
x1201 = 1;
x1202 = decay_1;
JCudaTensor x1203;
x1203 = x1158;
x876.update(x1203, x1201, x1202);


// cv96_B <~~ V_cv96_B
float x1204, x1205;
x1204 = 1;
x1205 = 1;
JCudaTensor x1206;
x1206 = x1143;
x877.update(x1206, x1204, x1205);


// cv95_B <~~ V_cv95_B
float x1207, x1208;
x1207 = 1;
x1208 = 1;
JCudaTensor x1209;
x1209 = x1172;
x905.update(x1209, x1207, x1208);


// cv95_W <~~ V_cv95_W
float x1210, x1211;
x1210 = 1;
x1211 = decay_1;
JCudaTensor x1212;
x1212 = x1181;
x904.update(x1212, x1210, x1211);


// cv93_B <~~ V_cv93_B
float x1213, x1214;
x1213 = 1;
x1214 = 1;
JCudaTensor x1215;
x1215 = x1187;
x893.update(x1215, x1213, x1214);


// val X2222 = X2220 * d_ReLU()(X210)/d_X209
JCudaTensor x1216;
JCudaTensor x1217, x1218;
x1217 = x1152;
x1218 = x882;
x1216 = x884.backward(x1217,x1218);

// Dealloc(X210)
JCudaTensor x1219;
x1219 = x882;
x1219.free();

// val X2246 = X2244 * d_ReLU()(X214)/d_X213
JCudaTensor x1220;
JCudaTensor x1221, x1222;
x1221 = x1140;
x1222 = x869;
x1220 = x871.backward(x1221,x1222);

// Dealloc(X214)
JCudaTensor x1223;
x1223 = x869;
x1223.free();

// V_cv94_W <~~ X2246 * d_Convolv(1,0)(X206)/d_cv94_W
float x1225, x1226;
x1225 = lrn_rate_1;
x1226 = momentum;
JCudaTensor x1227, x1228;
x1227 = x1220;
x1228 = x825;
x855.backward_filter(x1227,x1228, x1224, x1225, x1226);

// V_cv94_B <~~ X2246 * d_Convolv(1,0)()/d_cv94_B
float x1230, x1231;
x1230 = lrn_rate_2;
x1231 = momentum;
JCudaTensor x1232;
x1232 = x1220;
x855.backward_bias(x1232, x1229, x1230, x1231);

// val X2224 = (X2200 + X2222 * d_Convolv(1,0)(cv92_W)/d_X206)
JCudaTensor x1233;
JCudaTensor x1234;
x1234 = x1155;
JCudaTensor x1235, x1236;
x1235 = x1216;
x1236 = x839;
x1233 = x841.backward_data(x1235,x1236, x1234);

// V_cv92_W <~~ X2222 * d_Convolv(1,0)(X206)/d_cv92_W
float x1238, x1239;
x1238 = lrn_rate_1;
x1239 = momentum;
JCudaTensor x1240, x1241;
x1240 = x1216;
x1241 = x825;
x841.backward_filter(x1240,x1241, x1237, x1238, x1239);

// V_cv92_B <~~ X2222 * d_Convolv(1,0)()/d_cv92_B
float x1243, x1244;
x1243 = lrn_rate_2;
x1244 = momentum;
JCudaTensor x1245;
x1245 = x1216;
x841.backward_bias(x1245, x1242, x1243, x1244);

// Dealloc(X2222)
JCudaTensor x1246;
x1246 = x1216;
x1246.free();

// cv94_B <~~ V_cv94_B
float x1247, x1248;
x1247 = 1;
x1248 = 1;
JCudaTensor x1249;
x1249 = x1229;
x854.update(x1249, x1247, x1248);


// cv92_W <~~ V_cv92_W
float x1250, x1251;
x1250 = 1;
x1251 = decay_1;
JCudaTensor x1252;
x1252 = x1237;
x839.update(x1252, x1250, x1251);


// cv92_B <~~ V_cv92_B
float x1253, x1254;
x1253 = 1;
x1254 = 1;
JCudaTensor x1255;
x1255 = x1242;
x840.update(x1255, x1253, x1254);


// val X2248 = (X2224 + X2246 * d_Convolv(1,0)(cv94_W)/d_X206)
JCudaTensor x1256;
JCudaTensor x1257;
x1257 = x1233;
JCudaTensor x1258, x1259;
x1258 = x1220;
x1259 = x853;
x1256 = x855.backward_data(x1258,x1259, x1257);

// Dealloc(X2246)
JCudaTensor x1260;
x1260 = x1220;
x1260.free();

// cv94_W <~~ V_cv94_W
float x1261, x1262;
x1261 = 1;
x1262 = decay_1;
JCudaTensor x1263;
x1263 = x1224;
x853.update(x1263, x1261, x1262);


// val X2270 = (X2248 + X2267 * d_Pooling(3,1,1,true)(X217,X206)/d_X206)
JCudaTensor x1264;
JCudaTensor x1265;
x1265 = x1256;
JCudaTensor x1266, x1267, x1268;
x1266 = x1168;
x1267 = x862;
x1268 = x825;
x1264 = x864.backward(x1266,x1267,x1268, x1265);

// Dealloc(X2267)
JCudaTensor x1269;
x1269 = x1168;
x1269.free();

// Dealloc(X217)
JCudaTensor x1270;
x1270 = x862;
x1270.free();

// Dealloc(X206)
JCudaTensor x1271;
x1271 = x825;
x1271.free();

// val X2350 = Proj(X2270, X194,X198,X202,X205, 3)
JCudaTensor x1272;
JCudaTensor x1274;
x1274 = x1264;
JCudaTensor[] x1273 = x826.backward(x1274);
x1272 = x1273[3];

// val X2326 = Proj(X2270, X194,X198,X202,X205, 2)
JCudaTensor x1275;
x1275 = x1273[2];

// val X2302 = Proj(X2270, X194,X198,X202,X205, 1)
JCudaTensor x1276;
x1276 = x1273[1];

// val X2284 = Proj(X2270, X194,X198,X202,X205, 0)
JCudaTensor x1277;
x1277 = x1273[0];

// Dealloc(X2270)
JCudaTensor x1278;
x1278 = x1264;
x1278.free();

// val X2287 = X2284 * d_ReLU()(X194)/d_X193
JCudaTensor x1279;
JCudaTensor x1280, x1281;
x1280 = x1277;
x1281 = x819;
x1279 = x821.backward(x1280,x1281);

// Dealloc(X194)
JCudaTensor x1282;
x1282 = x819;
x1282.free();

// val X2307 = X2302 * d_ReLU()(X198)/d_X197
JCudaTensor x1283;
JCudaTensor x1284, x1285;
x1284 = x1276;
x1285 = x813;
x1283 = x815.backward(x1284,x1285);

// Dealloc(X198)
JCudaTensor x1286;
x1286 = x813;
x1286.free();

// val X2331 = X2326 * d_ReLU()(X202)/d_X201
JCudaTensor x1287;
JCudaTensor x1288, x1289;
x1288 = x1275;
x1289 = x816;
x1287 = x818.backward(x1288,x1289);

// Dealloc(X202)
JCudaTensor x1290;
x1290 = x816;
x1290.free();

// val X2354 = X2350 * d_ReLU()(X205)/d_X204
JCudaTensor x1291;
JCudaTensor x1292, x1293;
x1292 = x1272;
x1293 = x810;
x1291 = x812.backward(x1292,x1293);

// Dealloc(X205)
JCudaTensor x1294;
x1294 = x810;
x1294.free();

// val X2288 = X2287 * d_Convolv(1,0)(cv81_W)/d_X192
JCudaTensor x1295;
JCudaTensor x1296, x1297;
x1296 = x1279;
x1297 = x748;
x1295 = x750.backward_data(x1296,x1297);

// V_cv81_B <~~ X2287 * d_Convolv(1,0)()/d_cv81_B
float x1299, x1300;
x1299 = lrn_rate_2;
x1300 = momentum;
JCudaTensor x1301;
x1301 = x1279;
x750.backward_bias(x1301, x1298, x1299, x1300);

// V_cv83_W <~~ X2307 * d_Convolv(1,1)(X196)/d_cv83_W
float x1303, x1304;
x1303 = lrn_rate_1;
x1304 = momentum;
JCudaTensor x1305, x1306;
x1305 = x1283;
x1306 = x777;
x799.backward_filter(x1305,x1306, x1302, x1303, x1304);

// val X2332 = X2331 * d_Convolv(1,2)(cv85_W)/d_X200
JCudaTensor x1307;
JCudaTensor x1308, x1309;
x1308 = x1287;
x1309 = x804;
x1307 = x806.backward_data(x1308,x1309);

// V_cv85_W <~~ X2331 * d_Convolv(1,2)(X200)/d_cv85_W
float x1311, x1312;
x1311 = lrn_rate_1;
x1312 = momentum;
JCudaTensor x1313, x1314;
x1313 = x1287;
x1314 = x780;
x806.backward_filter(x1313,x1314, x1310, x1311, x1312);

// val X2308 = X2307 * d_Convolv(1,1)(cv83_W)/d_X196
JCudaTensor x1315;
JCudaTensor x1316, x1317;
x1316 = x1283;
x1317 = x797;
x1315 = x799.backward_data(x1316,x1317);

// V_cv81_W <~~ X2287 * d_Convolv(1,0)(X192)/d_cv81_W
float x1319, x1320;
x1319 = lrn_rate_1;
x1320 = momentum;
JCudaTensor x1321, x1322;
x1321 = x1279;
x1322 = x739;
x750.backward_filter(x1321,x1322, x1318, x1319, x1320);

// Dealloc(X2287)
JCudaTensor x1323;
x1323 = x1279;
x1323.free();

// V_cv86_W <~~ X2354 * d_Convolv(1,0)(X203)/d_cv86_W
float x1325, x1326;
x1325 = lrn_rate_1;
x1326 = momentum;
JCudaTensor x1327, x1328;
x1327 = x1291;
x1328 = x751;
x776.backward_filter(x1327,x1328, x1324, x1325, x1326);

// V_cv83_B <~~ X2307 * d_Convolv(1,1)()/d_cv83_B
float x1330, x1331;
x1330 = lrn_rate_2;
x1331 = momentum;
JCudaTensor x1332;
x1332 = x1283;
x799.backward_bias(x1332, x1329, x1330, x1331);

// Dealloc(X2307)
JCudaTensor x1333;
x1333 = x1283;
x1333.free();

// V_cv85_B <~~ X2331 * d_Convolv(1,2)()/d_cv85_B
float x1335, x1336;
x1335 = lrn_rate_2;
x1336 = momentum;
JCudaTensor x1337;
x1337 = x1287;
x806.backward_bias(x1337, x1334, x1335, x1336);

// Dealloc(X2331)
JCudaTensor x1338;
x1338 = x1287;
x1338.free();

// V_cv86_B <~~ X2354 * d_Convolv(1,0)()/d_cv86_B
float x1340, x1341;
x1340 = lrn_rate_2;
x1341 = momentum;
JCudaTensor x1342;
x1342 = x1291;
x776.backward_bias(x1342, x1339, x1340, x1341);

// val X2355 = X2354 * d_Convolv(1,0)(cv86_W)/d_X203
JCudaTensor x1343;
JCudaTensor x1344, x1345;
x1344 = x1291;
x1345 = x774;
x1343 = x776.backward_data(x1344,x1345);

// Dealloc(X2354)
JCudaTensor x1346;
x1346 = x1291;
x1346.free();

// cv85_W <~~ V_cv85_W
float x1347, x1348;
x1347 = 1;
x1348 = decay_1;
JCudaTensor x1349;
x1349 = x1310;
x804.update(x1349, x1347, x1348);


// cv86_B <~~ V_cv86_B
float x1350, x1351;
x1350 = 1;
x1351 = 1;
JCudaTensor x1352;
x1352 = x1339;
x775.update(x1352, x1350, x1351);


// cv83_W <~~ V_cv83_W
float x1353, x1354;
x1353 = 1;
x1354 = decay_1;
JCudaTensor x1355;
x1355 = x1302;
x797.update(x1355, x1353, x1354);


// cv86_W <~~ V_cv86_W
float x1356, x1357;
x1356 = 1;
x1357 = decay_1;
JCudaTensor x1358;
x1358 = x1324;
x774.update(x1358, x1356, x1357);


// cv83_B <~~ V_cv83_B
float x1359, x1360;
x1359 = 1;
x1360 = 1;
JCudaTensor x1361;
x1361 = x1329;
x798.update(x1361, x1359, x1360);


// cv85_B <~~ V_cv85_B
float x1362, x1363;
x1362 = 1;
x1363 = 1;
JCudaTensor x1364;
x1364 = x1334;
x805.update(x1364, x1362, x1363);


// cv81_W <~~ V_cv81_W
float x1365, x1366;
x1365 = 1;
x1366 = decay_1;
JCudaTensor x1367;
x1367 = x1318;
x748.update(x1367, x1365, x1366);


// cv81_B <~~ V_cv81_B
float x1368, x1369;
x1368 = 1;
x1369 = 1;
JCudaTensor x1370;
x1370 = x1298;
x749.update(x1370, x1368, x1369);


// val X2334 = X2332 * d_ReLU()(X200)/d_X199
JCudaTensor x1371;
JCudaTensor x1372, x1373;
x1372 = x1307;
x1373 = x780;
x1371 = x782.backward(x1372,x1373);

// Dealloc(X200)
JCudaTensor x1374;
x1374 = x780;
x1374.free();

// val X2310 = X2308 * d_ReLU()(X196)/d_X195
JCudaTensor x1375;
JCudaTensor x1376, x1377;
x1376 = x1315;
x1377 = x777;
x1375 = x779.backward(x1376,x1377);

// Dealloc(X196)
JCudaTensor x1378;
x1378 = x777;
x1378.free();

// V_cv84_W <~~ X2334 * d_Convolv(1,0)(X192)/d_cv84_W
float x1380, x1381;
x1380 = lrn_rate_1;
x1381 = momentum;
JCudaTensor x1382, x1383;
x1382 = x1371;
x1383 = x739;
x769.backward_filter(x1382,x1383, x1379, x1380, x1381);

// V_cv82_B <~~ X2310 * d_Convolv(1,0)()/d_cv82_B
float x1385, x1386;
x1385 = lrn_rate_2;
x1386 = momentum;
JCudaTensor x1387;
x1387 = x1375;
x762.backward_bias(x1387, x1384, x1385, x1386);

// val X2312 = (X2288 + X2310 * d_Convolv(1,0)(cv82_W)/d_X192)
JCudaTensor x1388;
JCudaTensor x1389;
x1389 = x1295;
JCudaTensor x1390, x1391;
x1390 = x1375;
x1391 = x760;
x1388 = x762.backward_data(x1390,x1391, x1389);

// V_cv82_W <~~ X2310 * d_Convolv(1,0)(X192)/d_cv82_W
float x1393, x1394;
x1393 = lrn_rate_1;
x1394 = momentum;
JCudaTensor x1395, x1396;
x1395 = x1375;
x1396 = x739;
x762.backward_filter(x1395,x1396, x1392, x1393, x1394);

// Dealloc(X2310)
JCudaTensor x1397;
x1397 = x1375;
x1397.free();

// V_cv84_B <~~ X2334 * d_Convolv(1,0)()/d_cv84_B
float x1399, x1400;
x1399 = lrn_rate_2;
x1400 = momentum;
JCudaTensor x1401;
x1401 = x1371;
x769.backward_bias(x1401, x1398, x1399, x1400);

// cv82_B <~~ V_cv82_B
float x1402, x1403;
x1402 = 1;
x1403 = 1;
JCudaTensor x1404;
x1404 = x1384;
x761.update(x1404, x1402, x1403);


// cv82_W <~~ V_cv82_W
float x1405, x1406;
x1405 = 1;
x1406 = decay_1;
JCudaTensor x1407;
x1407 = x1392;
x760.update(x1407, x1405, x1406);


// cv84_B <~~ V_cv84_B
float x1408, x1409;
x1408 = 1;
x1409 = 1;
JCudaTensor x1410;
x1410 = x1398;
x768.update(x1410, x1408, x1409);


// val X2336 = (X2312 + X2334 * d_Convolv(1,0)(cv84_W)/d_X192)
JCudaTensor x1411;
JCudaTensor x1412;
x1412 = x1388;
JCudaTensor x1413, x1414;
x1413 = x1371;
x1414 = x767;
x1411 = x769.backward_data(x1413,x1414, x1412);

// Dealloc(X2334)
JCudaTensor x1415;
x1415 = x1371;
x1415.free();

// cv84_W <~~ V_cv84_W
float x1416, x1417;
x1416 = 1;
x1417 = decay_1;
JCudaTensor x1418;
x1418 = x1379;
x767.update(x1418, x1416, x1417);


// val X2358 = (X2336 + X2355 * d_Pooling(3,1,1,true)(X203,X192)/d_X192)
JCudaTensor x1419;
JCudaTensor x1420;
x1420 = x1411;
JCudaTensor x1421, x1422, x1423;
x1421 = x1343;
x1422 = x751;
x1423 = x739;
x1419 = x753.backward(x1421,x1422,x1423, x1420);

// Dealloc(X2355)
JCudaTensor x1424;
x1424 = x1343;
x1424.free();

// Dealloc(X203)
JCudaTensor x1425;
x1425 = x751;
x1425.free();

// val X2360 = X2358 * d_Pooling(3,2,1,true)(X192,X191)/d_X191
JCudaTensor x1426;
JCudaTensor x1427, x1428, x1429;
x1427 = x1419;
x1428 = x739;
x1429 = x734;
x1426 = x741.backward(x1427,x1428,x1429);

// Dealloc(X2358)
JCudaTensor x1430;
x1430 = x1419;
x1430.free();

// Dealloc(X192)
JCudaTensor x1431;
x1431 = x739;
x1431.free();

// Dealloc(X191)
JCudaTensor x1432;
x1432 = x734;
x1432.free();

// val X2392 = Proj(X2360, X179,X183,X187,X190, 1)
JCudaTensor x1433;
JCudaTensor x1435;
x1435 = x1426;
JCudaTensor[] x1434 = x260.backward(x1435);
x1433 = x1434[1];

// val X2416 = Proj(X2360, X179,X183,X187,X190, 2)
JCudaTensor x1436;
x1436 = x1434[2];

// val X2440 = Proj(X2360, X179,X183,X187,X190, 3)
JCudaTensor x1437;
x1437 = x1434[3];

// val X2374 = Proj(X2360, X179,X183,X187,X190, 0)
JCudaTensor x1438;
x1438 = x1434[0];

// Dealloc(X2360)
JCudaTensor x1439;
x1439 = x1426;
x1439.free();

// val X2377 = X2374 * d_ReLU()(X179)/d_X178
JCudaTensor x1440;
JCudaTensor x1441, x1442;
x1441 = x1438;
x1442 = x721;
x1440 = x723.backward(x1441,x1442);

// Dealloc(X179)
JCudaTensor x1443;
x1443 = x721;
x1443.free();

// val X2397 = X2392 * d_ReLU()(X183)/d_X182
JCudaTensor x1444;
JCudaTensor x1445, x1446;
x1445 = x1433;
x1446 = x698;
x1444 = x700.backward(x1445,x1446);

// Dealloc(X183)
JCudaTensor x1447;
x1447 = x698;
x1447.free();

// val X2444 = X2440 * d_ReLU()(X190)/d_X189
JCudaTensor x1448;
JCudaTensor x1449, x1450;
x1449 = x1437;
x1450 = x701;
x1448 = x703.backward(x1449,x1450);

// Dealloc(X190)
JCudaTensor x1451;
x1451 = x701;
x1451.free();

// val X2421 = X2416 * d_ReLU()(X187)/d_X186
JCudaTensor x1452;
JCudaTensor x1453, x1454;
x1453 = x1436;
x1454 = x695;
x1452 = x697.backward(x1453,x1454);

// Dealloc(X187)
JCudaTensor x1455;
x1455 = x695;
x1455.free();

// val X2378 = X2377 * d_Convolv(1,0)(cv71_W)/d_X177
JCudaTensor x1456;
JCudaTensor x1457, x1458;
x1457 = x1440;
x1458 = x593;
x1456 = x595.backward_data(x1457,x1458);

// V_cv71_W <~~ X2377 * d_Convolv(1,0)(X177)/d_cv71_W
float x1460, x1461;
x1460 = lrn_rate_1;
x1461 = momentum;
JCudaTensor x1462, x1463;
x1462 = x1440;
x1463 = x556;
x595.backward_filter(x1462,x1463, x1459, x1460, x1461);

// V_cv73_W <~~ X2397 * d_Convolv(1,1)(X181)/d_cv73_W
float x1465, x1466;
x1465 = lrn_rate_1;
x1466 = momentum;
JCudaTensor x1467, x1468;
x1467 = x1444;
x1468 = x617;
x688.backward_filter(x1467,x1468, x1464, x1465, x1466);

// V_cv73_B <~~ X2397 * d_Convolv(1,1)()/d_cv73_B
float x1470, x1471;
x1470 = lrn_rate_2;
x1471 = momentum;
JCudaTensor x1472;
x1472 = x1444;
x688.backward_bias(x1472, x1469, x1470, x1471);

// val X2398 = X2397 * d_Convolv(1,1)(cv73_W)/d_X181
JCudaTensor x1473;
JCudaTensor x1474, x1475;
x1474 = x1444;
x1475 = x686;
x1473 = x688.backward_data(x1474,x1475);

// Dealloc(X2397)
JCudaTensor x1476;
x1476 = x1444;
x1476.free();

// val X2445 = X2444 * d_Convolv(1,0)(cv76_W)/d_X188
JCudaTensor x1477;
JCudaTensor x1478, x1479;
x1478 = x1448;
x1479 = x640;
x1477 = x642.backward_data(x1478,x1479);

// V_cv75_B <~~ X2421 * d_Convolv(1,2)()/d_cv75_B
float x1481, x1482;
x1481 = lrn_rate_2;
x1482 = momentum;
JCudaTensor x1483;
x1483 = x1452;
x678.backward_bias(x1483, x1480, x1481, x1482);

// V_cv76_W <~~ X2444 * d_Convolv(1,0)(X188)/d_cv76_W
float x1485, x1486;
x1485 = lrn_rate_1;
x1486 = momentum;
JCudaTensor x1487, x1488;
x1487 = x1448;
x1488 = x579;
x642.backward_filter(x1487,x1488, x1484, x1485, x1486);

// val X2422 = X2421 * d_Convolv(1,2)(cv75_W)/d_X185
JCudaTensor x1489;
JCudaTensor x1490, x1491;
x1490 = x1452;
x1491 = x676;
x1489 = x678.backward_data(x1490,x1491);

// V_cv75_W <~~ X2421 * d_Convolv(1,2)(X185)/d_cv75_W
float x1493, x1494;
x1493 = lrn_rate_1;
x1494 = momentum;
JCudaTensor x1495, x1496;
x1495 = x1452;
x1496 = x633;
x678.backward_filter(x1495,x1496, x1492, x1493, x1494);

// Dealloc(X2421)
JCudaTensor x1497;
x1497 = x1452;
x1497.free();

// V_cv71_B <~~ X2377 * d_Convolv(1,0)()/d_cv71_B
float x1499, x1500;
x1499 = lrn_rate_2;
x1500 = momentum;
JCudaTensor x1501;
x1501 = x1440;
x595.backward_bias(x1501, x1498, x1499, x1500);

// Dealloc(X2377)
JCudaTensor x1502;
x1502 = x1440;
x1502.free();

// V_cv76_B <~~ X2444 * d_Convolv(1,0)()/d_cv76_B
float x1504, x1505;
x1504 = lrn_rate_2;
x1505 = momentum;
JCudaTensor x1506;
x1506 = x1448;
x642.backward_bias(x1506, x1503, x1504, x1505);

// Dealloc(X2444)
JCudaTensor x1507;
x1507 = x1448;
x1507.free();

// cv75_W <~~ V_cv75_W
float x1508, x1509;
x1508 = 1;
x1509 = decay_1;
JCudaTensor x1510;
x1510 = x1492;
x676.update(x1510, x1508, x1509);


// cv71_B <~~ V_cv71_B
float x1511, x1512;
x1511 = 1;
x1512 = 1;
JCudaTensor x1513;
x1513 = x1498;
x594.update(x1513, x1511, x1512);


// cv71_W <~~ V_cv71_W
float x1514, x1515;
x1514 = 1;
x1515 = decay_1;
JCudaTensor x1516;
x1516 = x1459;
x593.update(x1516, x1514, x1515);


// cv73_W <~~ V_cv73_W
float x1517, x1518;
x1517 = 1;
x1518 = decay_1;
JCudaTensor x1519;
x1519 = x1464;
x686.update(x1519, x1517, x1518);


// cv76_W <~~ V_cv76_W
float x1520, x1521;
x1520 = 1;
x1521 = decay_1;
JCudaTensor x1522;
x1522 = x1484;
x640.update(x1522, x1520, x1521);


// cv75_B <~~ V_cv75_B
float x1523, x1524;
x1523 = 1;
x1524 = 1;
JCudaTensor x1525;
x1525 = x1480;
x677.update(x1525, x1523, x1524);


// cv73_B <~~ V_cv73_B
float x1526, x1527;
x1526 = 1;
x1527 = 1;
JCudaTensor x1528;
x1528 = x1469;
x687.update(x1528, x1526, x1527);


// cv76_B <~~ V_cv76_B
float x1529, x1530;
x1529 = 1;
x1530 = 1;
JCudaTensor x1531;
x1531 = x1503;
x641.update(x1531, x1529, x1530);


// val X2400 = X2398 * d_ReLU()(X181)/d_X180
JCudaTensor x1532;
JCudaTensor x1533, x1534;
x1533 = x1473;
x1534 = x617;
x1532 = x619.backward(x1533,x1534);

// Dealloc(X181)
JCudaTensor x1535;
x1535 = x617;
x1535.free();

// val X2424 = X2422 * d_ReLU()(X185)/d_X184
JCudaTensor x1536;
JCudaTensor x1537, x1538;
x1537 = x1489;
x1538 = x633;
x1536 = x635.backward(x1537,x1538);

// Dealloc(X185)
JCudaTensor x1539;
x1539 = x633;
x1539.free();

// V_cv74_W <~~ X2424 * d_Convolv(1,0)(X177)/d_cv74_W
float x1541, x1542;
x1541 = lrn_rate_1;
x1542 = momentum;
JCudaTensor x1543, x1544;
x1543 = x1536;
x1544 = x556;
x609.backward_filter(x1543,x1544, x1540, x1541, x1542);

// val X2402 = (X2378 + X2400 * d_Convolv(1,0)(cv72_W)/d_X177)
JCudaTensor x1545;
JCudaTensor x1546;
x1546 = x1456;
JCudaTensor x1547, x1548;
x1547 = x1532;
x1548 = x600;
x1545 = x602.backward_data(x1547,x1548, x1546);

// V_cv72_W <~~ X2400 * d_Convolv(1,0)(X177)/d_cv72_W
float x1550, x1551;
x1550 = lrn_rate_1;
x1551 = momentum;
JCudaTensor x1552, x1553;
x1552 = x1532;
x1553 = x556;
x602.backward_filter(x1552,x1553, x1549, x1550, x1551);

// V_cv72_B <~~ X2400 * d_Convolv(1,0)()/d_cv72_B
float x1555, x1556;
x1555 = lrn_rate_2;
x1556 = momentum;
JCudaTensor x1557;
x1557 = x1532;
x602.backward_bias(x1557, x1554, x1555, x1556);

// Dealloc(X2400)
JCudaTensor x1558;
x1558 = x1532;
x1558.free();

// V_cv74_B <~~ X2424 * d_Convolv(1,0)()/d_cv74_B
float x1560, x1561;
x1560 = lrn_rate_2;
x1561 = momentum;
JCudaTensor x1562;
x1562 = x1536;
x609.backward_bias(x1562, x1559, x1560, x1561);

// cv72_W <~~ V_cv72_W
float x1563, x1564;
x1563 = 1;
x1564 = decay_1;
JCudaTensor x1565;
x1565 = x1549;
x600.update(x1565, x1563, x1564);


// cv72_B <~~ V_cv72_B
float x1566, x1567;
x1566 = 1;
x1567 = 1;
JCudaTensor x1568;
x1568 = x1554;
x601.update(x1568, x1566, x1567);


// cv74_B <~~ V_cv74_B
float x1569, x1570;
x1569 = 1;
x1570 = 1;
JCudaTensor x1571;
x1571 = x1559;
x608.update(x1571, x1569, x1570);


// val X2426 = (X2402 + X2424 * d_Convolv(1,0)(cv74_W)/d_X177)
JCudaTensor x1572;
JCudaTensor x1573;
x1573 = x1545;
JCudaTensor x1574, x1575;
x1574 = x1536;
x1575 = x607;
x1572 = x609.backward_data(x1574,x1575, x1573);

// Dealloc(X2424)
JCudaTensor x1576;
x1576 = x1536;
x1576.free();

// cv74_W <~~ V_cv74_W
float x1577, x1578;
x1577 = 1;
x1578 = decay_1;
JCudaTensor x1579;
x1579 = x1540;
x607.update(x1579, x1577, x1578);


// val X2448 = (X2426 + X2445 * d_Pooling(3,1,1,true)(X188,X177)/d_X177)
JCudaTensor x1580;
JCudaTensor x1581;
x1581 = x1572;
JCudaTensor x1582, x1583, x1584;
x1582 = x1477;
x1583 = x579;
x1584 = x556;
x1580 = x581.backward(x1582,x1583,x1584, x1581);

// Dealloc(X2445)
JCudaTensor x1585;
x1585 = x1477;
x1585.free();

// Dealloc(X188)
JCudaTensor x1586;
x1586 = x579;
x1586.free();

// Dealloc(X177)
JCudaTensor x1587;
x1587 = x556;
x1587.free();

// val X2484 = (X2483 * loss2)
JCudaTensor x1588;
JCudaTensor x1589;
float x1590;
x1589 = x995;
x1590 = loss2;
x1588 = x1589.times_i(x1590);

// val X2485 = (X2448 + X2484)
JCudaTensor x1591;
JCudaTensor x1592, x1593;
x1592 = x1580;
x1593 = x1588;
x1591 = x1592.plus_i(x1593);

// Dealloc(X2484)
JCudaTensor x1594;
x1594 = x1588;
x1594.free();

// val X2527 = Proj(X2485, X165,X169,X173,X176, 0)
JCudaTensor x1595;
JCudaTensor x1597;
x1597 = x1591;
JCudaTensor[] x1596 = x260.backward(x1597);
x1595 = x1596[0];

// val X2569 = Proj(X2485, X165,X169,X173,X176, 2)
JCudaTensor x1598;
x1598 = x1596[2];

// val X2545 = Proj(X2485, X165,X169,X173,X176, 1)
JCudaTensor x1599;
x1599 = x1596[1];

// val X2593 = Proj(X2485, X165,X169,X173,X176, 3)
JCudaTensor x1600;
x1600 = x1596[3];

// Dealloc(X2485)
JCudaTensor x1601;
x1601 = x1591;
x1601.free();

// val X2550 = X2545 * d_ReLU()(X169)/d_X168
JCudaTensor x1602;
JCudaTensor x1603, x1604;
x1603 = x1599;
x1604 = x534;
x1602 = x536.backward(x1603,x1604);

// Dealloc(X169)
JCudaTensor x1605;
x1605 = x534;
x1605.free();

// val X2597 = X2593 * d_ReLU()(X176)/d_X175
JCudaTensor x1606;
JCudaTensor x1607, x1608;
x1607 = x1600;
x1608 = x526;
x1606 = x528.backward(x1607,x1608);

// Dealloc(X176)
JCudaTensor x1609;
x1609 = x526;
x1609.free();

// val X2574 = X2569 * d_ReLU()(X173)/d_X172
JCudaTensor x1610;
JCudaTensor x1611, x1612;
x1611 = x1598;
x1612 = x537;
x1610 = x539.backward(x1611,x1612);

// Dealloc(X173)
JCudaTensor x1613;
x1613 = x537;
x1613.free();

// val X2530 = X2527 * d_ReLU()(X165)/d_X164
JCudaTensor x1614;
JCudaTensor x1615, x1616;
x1615 = x1595;
x1616 = x531;
x1614 = x533.backward(x1615,x1616);

// Dealloc(X165)
JCudaTensor x1617;
x1617 = x531;
x1617.free();

// val X2551 = X2550 * d_Convolv(1,1)(cv63_W)/d_X167
JCudaTensor x1618;
JCudaTensor x1619, x1620;
x1619 = x1602;
x1620 = x520;
x1618 = x522.backward_data(x1619,x1620);

// V_cv66_B <~~ X2597 * d_Convolv(1,0)()/d_cv66_B
float x1622, x1623;
x1622 = lrn_rate_2;
x1623 = momentum;
JCudaTensor x1624;
x1624 = x1606;
x484.backward_bias(x1624, x1621, x1622, x1623);

// val X2598 = X2597 * d_Convolv(1,0)(cv66_W)/d_X174
JCudaTensor x1625;
JCudaTensor x1626, x1627;
x1626 = x1606;
x1627 = x482;
x1625 = x484.backward_data(x1626,x1627);

// V_cv66_W <~~ X2597 * d_Convolv(1,0)(X174)/d_cv66_W
float x1629, x1630;
x1629 = lrn_rate_1;
x1630 = momentum;
JCudaTensor x1631, x1632;
x1631 = x1606;
x1632 = x475;
x484.backward_filter(x1631,x1632, x1628, x1629, x1630);

// Dealloc(X2597)
JCudaTensor x1633;
x1633 = x1606;
x1633.free();

// val X2575 = X2574 * d_Convolv(1,2)(cv65_W)/d_X171
JCudaTensor x1634;
JCudaTensor x1635, x1636;
x1635 = x1610;
x1636 = x503;
x1634 = x505.backward_data(x1635,x1636);

// V_cv65_W <~~ X2574 * d_Convolv(1,2)(X171)/d_cv65_W
float x1638, x1639;
x1638 = lrn_rate_1;
x1639 = momentum;
JCudaTensor x1640, x1641;
x1640 = x1610;
x1641 = x494;
x505.backward_filter(x1640,x1641, x1637, x1638, x1639);

// V_cv61_W <~~ X2530 * d_Convolv(1,0)(X163)/d_cv61_W
float x1643, x1644;
x1643 = lrn_rate_1;
x1644 = momentum;
JCudaTensor x1645, x1646;
x1645 = x1614;
x1646 = x436;
x474.backward_filter(x1645,x1646, x1642, x1643, x1644);

// V_cv63_W <~~ X2550 * d_Convolv(1,1)(X167)/d_cv63_W
float x1648, x1649;
x1648 = lrn_rate_1;
x1649 = momentum;
JCudaTensor x1650, x1651;
x1650 = x1602;
x1651 = x491;
x522.backward_filter(x1650,x1651, x1647, x1648, x1649);

// val X2531 = X2530 * d_Convolv(1,0)(cv61_W)/d_X163
JCudaTensor x1652;
JCudaTensor x1653, x1654;
x1653 = x1614;
x1654 = x472;
x1652 = x474.backward_data(x1653,x1654);

// V_cv65_B <~~ X2574 * d_Convolv(1,2)()/d_cv65_B
float x1656, x1657;
x1656 = lrn_rate_2;
x1657 = momentum;
JCudaTensor x1658;
x1658 = x1610;
x505.backward_bias(x1658, x1655, x1656, x1657);

// Dealloc(X2574)
JCudaTensor x1659;
x1659 = x1610;
x1659.free();

// V_cv61_B <~~ X2530 * d_Convolv(1,0)()/d_cv61_B
float x1661, x1662;
x1661 = lrn_rate_2;
x1662 = momentum;
JCudaTensor x1663;
x1663 = x1614;
x474.backward_bias(x1663, x1660, x1661, x1662);

// Dealloc(X2530)
JCudaTensor x1664;
x1664 = x1614;
x1664.free();

// V_cv63_B <~~ X2550 * d_Convolv(1,1)()/d_cv63_B
float x1666, x1667;
x1666 = lrn_rate_2;
x1667 = momentum;
JCudaTensor x1668;
x1668 = x1602;
x522.backward_bias(x1668, x1665, x1666, x1667);

// Dealloc(X2550)
JCudaTensor x1669;
x1669 = x1602;
x1669.free();

// cv66_W <~~ V_cv66_W
float x1670, x1671;
x1670 = 1;
x1671 = decay_1;
JCudaTensor x1672;
x1672 = x1628;
x482.update(x1672, x1670, x1671);


// cv65_W <~~ V_cv65_W
float x1673, x1674;
x1673 = 1;
x1674 = decay_1;
JCudaTensor x1675;
x1675 = x1637;
x503.update(x1675, x1673, x1674);


// cv61_B <~~ V_cv61_B
float x1676, x1677;
x1676 = 1;
x1677 = 1;
JCudaTensor x1678;
x1678 = x1660;
x473.update(x1678, x1676, x1677);


// cv63_B <~~ V_cv63_B
float x1679, x1680;
x1679 = 1;
x1680 = 1;
JCudaTensor x1681;
x1681 = x1665;
x521.update(x1681, x1679, x1680);


// cv61_W <~~ V_cv61_W
float x1682, x1683;
x1682 = 1;
x1683 = decay_1;
JCudaTensor x1684;
x1684 = x1642;
x472.update(x1684, x1682, x1683);


// cv63_W <~~ V_cv63_W
float x1685, x1686;
x1685 = 1;
x1686 = decay_1;
JCudaTensor x1687;
x1687 = x1647;
x520.update(x1687, x1685, x1686);


// cv65_B <~~ V_cv65_B
float x1688, x1689;
x1688 = 1;
x1689 = 1;
JCudaTensor x1690;
x1690 = x1655;
x504.update(x1690, x1688, x1689);


// cv66_B <~~ V_cv66_B
float x1691, x1692;
x1691 = 1;
x1692 = 1;
JCudaTensor x1693;
x1693 = x1621;
x483.update(x1693, x1691, x1692);


// val X2553 = X2551 * d_ReLU()(X167)/d_X166
JCudaTensor x1694;
JCudaTensor x1695, x1696;
x1695 = x1618;
x1696 = x491;
x1694 = x493.backward(x1695,x1696);

// Dealloc(X167)
JCudaTensor x1697;
x1697 = x491;
x1697.free();

// val X2577 = X2575 * d_ReLU()(X171)/d_X170
JCudaTensor x1698;
JCudaTensor x1699, x1700;
x1699 = x1634;
x1700 = x494;
x1698 = x496.backward(x1699,x1700);

// Dealloc(X171)
JCudaTensor x1701;
x1701 = x494;
x1701.free();

// V_cv64_W <~~ X2577 * d_Convolv(1,0)(X163)/d_cv64_W
float x1703, x1704;
x1703 = lrn_rate_1;
x1704 = momentum;
JCudaTensor x1705, x1706;
x1705 = x1698;
x1706 = x436;
x455.backward_filter(x1705,x1706, x1702, x1703, x1704);

// val X2555 = (X2531 + X2553 * d_Convolv(1,0)(cv62_W)/d_X163)
JCudaTensor x1707;
JCudaTensor x1708;
x1708 = x1652;
JCudaTensor x1709, x1710;
x1709 = x1694;
x1710 = x465;
x1707 = x467.backward_data(x1709,x1710, x1708);

// V_cv62_W <~~ X2553 * d_Convolv(1,0)(X163)/d_cv62_W
float x1712, x1713;
x1712 = lrn_rate_1;
x1713 = momentum;
JCudaTensor x1714, x1715;
x1714 = x1694;
x1715 = x436;
x467.backward_filter(x1714,x1715, x1711, x1712, x1713);

// V_cv62_B <~~ X2553 * d_Convolv(1,0)()/d_cv62_B
float x1717, x1718;
x1717 = lrn_rate_2;
x1718 = momentum;
JCudaTensor x1719;
x1719 = x1694;
x467.backward_bias(x1719, x1716, x1717, x1718);

// Dealloc(X2553)
JCudaTensor x1720;
x1720 = x1694;
x1720.free();

// V_cv64_B <~~ X2577 * d_Convolv(1,0)()/d_cv64_B
float x1722, x1723;
x1722 = lrn_rate_2;
x1723 = momentum;
JCudaTensor x1724;
x1724 = x1698;
x455.backward_bias(x1724, x1721, x1722, x1723);

// cv62_W <~~ V_cv62_W
float x1725, x1726;
x1725 = 1;
x1726 = decay_1;
JCudaTensor x1727;
x1727 = x1711;
x465.update(x1727, x1725, x1726);


// cv62_B <~~ V_cv62_B
float x1728, x1729;
x1728 = 1;
x1729 = 1;
JCudaTensor x1730;
x1730 = x1716;
x466.update(x1730, x1728, x1729);


// cv64_B <~~ V_cv64_B
float x1731, x1732;
x1731 = 1;
x1732 = 1;
JCudaTensor x1733;
x1733 = x1721;
x454.update(x1733, x1731, x1732);


// val X2579 = (X2555 + X2577 * d_Convolv(1,0)(cv64_W)/d_X163)
JCudaTensor x1734;
JCudaTensor x1735;
x1735 = x1707;
JCudaTensor x1736, x1737;
x1736 = x1698;
x1737 = x453;
x1734 = x455.backward_data(x1736,x1737, x1735);

// Dealloc(X2577)
JCudaTensor x1738;
x1738 = x1698;
x1738.free();

// cv64_W <~~ V_cv64_W
float x1739, x1740;
x1739 = 1;
x1740 = decay_1;
JCudaTensor x1741;
x1741 = x1702;
x453.update(x1741, x1739, x1740);


// val X2601 = (X2579 + X2598 * d_Pooling(3,1,1,true)(X174,X163)/d_X163)
JCudaTensor x1742;
JCudaTensor x1743;
x1743 = x1734;
JCudaTensor x1744, x1745, x1746;
x1744 = x1625;
x1745 = x475;
x1746 = x436;
x1742 = x477.backward(x1744,x1745,x1746, x1743);

// Dealloc(X2598)
JCudaTensor x1747;
x1747 = x1625;
x1747.free();

// Dealloc(X174)
JCudaTensor x1748;
x1748 = x475;
x1748.free();

// Dealloc(X163)
JCudaTensor x1749;
x1749 = x436;
x1749.free();

// val X2633 = Proj(X2601, X151,X155,X159,X162, 1)
JCudaTensor x1750;
JCudaTensor x1752;
x1752 = x1742;
JCudaTensor[] x1751 = x260.backward(x1752);
x1750 = x1751[1];

// val X2615 = Proj(X2601, X151,X155,X159,X162, 0)
JCudaTensor x1753;
x1753 = x1751[0];

// val X2681 = Proj(X2601, X151,X155,X159,X162, 3)
JCudaTensor x1754;
x1754 = x1751[3];

// val X2657 = Proj(X2601, X151,X155,X159,X162, 2)
JCudaTensor x1755;
x1755 = x1751[2];

// Dealloc(X2601)
JCudaTensor x1756;
x1756 = x1742;
x1756.free();

// val X2618 = X2615 * d_ReLU()(X151)/d_X150
JCudaTensor x1757;
JCudaTensor x1758, x1759;
x1758 = x1753;
x1759 = x429;
x1757 = x431.backward(x1758,x1759);

// Dealloc(X151)
JCudaTensor x1760;
x1760 = x429;
x1760.free();

// val X2685 = X2681 * d_ReLU()(X162)/d_X161
JCudaTensor x1761;
JCudaTensor x1762, x1763;
x1762 = x1754;
x1763 = x423;
x1761 = x425.backward(x1762,x1763);

// Dealloc(X162)
JCudaTensor x1764;
x1764 = x423;
x1764.free();

// val X2638 = X2633 * d_ReLU()(X155)/d_X154
JCudaTensor x1765;
JCudaTensor x1766, x1767;
x1766 = x1750;
x1767 = x420;
x1765 = x422.backward(x1766,x1767);

// Dealloc(X155)
JCudaTensor x1768;
x1768 = x420;
x1768.free();

// val X2662 = X2657 * d_ReLU()(X159)/d_X158
JCudaTensor x1769;
JCudaTensor x1770, x1771;
x1770 = x1755;
x1771 = x426;
x1769 = x428.backward(x1770,x1771);

// Dealloc(X159)
JCudaTensor x1772;
x1772 = x426;
x1772.free();

// V_cv51_W <~~ X2618 * d_Convolv(1,0)(X149)/d_cv51_W
float x1774, x1775;
x1774 = lrn_rate_1;
x1775 = momentum;
JCudaTensor x1776, x1777;
x1776 = x1757;
x1777 = x348;
x376.backward_filter(x1776,x1777, x1773, x1774, x1775);

// val X2686 = X2685 * d_Convolv(1,0)(cv56_W)/d_X160
JCudaTensor x1778;
JCudaTensor x1779, x1780;
x1779 = x1761;
x1780 = x391;
x1778 = x393.backward_data(x1779,x1780);

// V_cv53_W <~~ X2638 * d_Convolv(1,1)(X153)/d_cv53_W
float x1782, x1783;
x1782 = lrn_rate_1;
x1783 = momentum;
JCudaTensor x1784, x1785;
x1784 = x1765;
x1785 = x397;
x406.backward_filter(x1784,x1785, x1781, x1782, x1783);

// V_cv55_B <~~ X2662 * d_Convolv(1,2)()/d_cv55_B
float x1787, x1788;
x1787 = lrn_rate_2;
x1788 = momentum;
JCudaTensor x1789;
x1789 = x1769;
x413.backward_bias(x1789, x1786, x1787, x1788);

// V_cv53_B <~~ X2638 * d_Convolv(1,1)()/d_cv53_B
float x1791, x1792;
x1791 = lrn_rate_2;
x1792 = momentum;
JCudaTensor x1793;
x1793 = x1765;
x406.backward_bias(x1793, x1790, x1791, x1792);

// V_cv51_B <~~ X2618 * d_Convolv(1,0)()/d_cv51_B
float x1795, x1796;
x1795 = lrn_rate_2;
x1796 = momentum;
JCudaTensor x1797;
x1797 = x1757;
x376.backward_bias(x1797, x1794, x1795, x1796);

// val X2663 = X2662 * d_Convolv(1,2)(cv55_W)/d_X157
JCudaTensor x1798;
JCudaTensor x1799, x1800;
x1799 = x1769;
x1800 = x411;
x1798 = x413.backward_data(x1799,x1800);

// val X2639 = X2638 * d_Convolv(1,1)(cv53_W)/d_X153
JCudaTensor x1801;
JCudaTensor x1802, x1803;
x1802 = x1765;
x1803 = x404;
x1801 = x406.backward_data(x1802,x1803);

// Dealloc(X2638)
JCudaTensor x1804;
x1804 = x1765;
x1804.free();

// V_cv56_B <~~ X2685 * d_Convolv(1,0)()/d_cv56_B
float x1806, x1807;
x1806 = lrn_rate_2;
x1807 = momentum;
JCudaTensor x1808;
x1808 = x1761;
x393.backward_bias(x1808, x1805, x1806, x1807);

// val X2619 = X2618 * d_Convolv(1,0)(cv51_W)/d_X149
JCudaTensor x1809;
JCudaTensor x1810, x1811;
x1810 = x1757;
x1811 = x374;
x1809 = x376.backward_data(x1810,x1811);

// Dealloc(X2618)
JCudaTensor x1812;
x1812 = x1757;
x1812.free();

// V_cv56_W <~~ X2685 * d_Convolv(1,0)(X160)/d_cv56_W
float x1814, x1815;
x1814 = lrn_rate_1;
x1815 = momentum;
JCudaTensor x1816, x1817;
x1816 = x1761;
x1817 = x367;
x393.backward_filter(x1816,x1817, x1813, x1814, x1815);

// Dealloc(X2685)
JCudaTensor x1818;
x1818 = x1761;
x1818.free();

// V_cv55_W <~~ X2662 * d_Convolv(1,2)(X157)/d_cv55_W
float x1820, x1821;
x1820 = lrn_rate_1;
x1821 = momentum;
JCudaTensor x1822, x1823;
x1822 = x1769;
x1823 = x384;
x413.backward_filter(x1822,x1823, x1819, x1820, x1821);

// Dealloc(X2662)
JCudaTensor x1824;
x1824 = x1769;
x1824.free();

// cv56_W <~~ V_cv56_W
float x1825, x1826;
x1825 = 1;
x1826 = decay_1;
JCudaTensor x1827;
x1827 = x1813;
x391.update(x1827, x1825, x1826);


// cv51_W <~~ V_cv51_W
float x1828, x1829;
x1828 = 1;
x1829 = decay_1;
JCudaTensor x1830;
x1830 = x1773;
x374.update(x1830, x1828, x1829);


// cv55_B <~~ V_cv55_B
float x1831, x1832;
x1831 = 1;
x1832 = 1;
JCudaTensor x1833;
x1833 = x1786;
x412.update(x1833, x1831, x1832);


// cv53_B <~~ V_cv53_B
float x1834, x1835;
x1834 = 1;
x1835 = 1;
JCudaTensor x1836;
x1836 = x1790;
x405.update(x1836, x1834, x1835);


// cv51_B <~~ V_cv51_B
float x1837, x1838;
x1837 = 1;
x1838 = 1;
JCudaTensor x1839;
x1839 = x1794;
x375.update(x1839, x1837, x1838);


// cv53_W <~~ V_cv53_W
float x1840, x1841;
x1840 = 1;
x1841 = decay_1;
JCudaTensor x1842;
x1842 = x1781;
x404.update(x1842, x1840, x1841);


// cv55_W <~~ V_cv55_W
float x1843, x1844;
x1843 = 1;
x1844 = decay_1;
JCudaTensor x1845;
x1845 = x1819;
x411.update(x1845, x1843, x1844);


// cv56_B <~~ V_cv56_B
float x1846, x1847;
x1846 = 1;
x1847 = 1;
JCudaTensor x1848;
x1848 = x1805;
x392.update(x1848, x1846, x1847);


// val X2641 = X2639 * d_ReLU()(X153)/d_X152
JCudaTensor x1849;
JCudaTensor x1850, x1851;
x1850 = x1801;
x1851 = x397;
x1849 = x399.backward(x1850,x1851);

// Dealloc(X153)
JCudaTensor x1852;
x1852 = x397;
x1852.free();

// val X2665 = X2663 * d_ReLU()(X157)/d_X156
JCudaTensor x1853;
JCudaTensor x1854, x1855;
x1854 = x1798;
x1855 = x384;
x1853 = x386.backward(x1854,x1855);

// Dealloc(X157)
JCudaTensor x1856;
x1856 = x384;
x1856.free();

// V_cv54_W <~~ X2665 * d_Convolv(1,0)(X149)/d_cv54_W
float x1858, x1859;
x1858 = lrn_rate_1;
x1859 = momentum;
JCudaTensor x1860, x1861;
x1860 = x1853;
x1861 = x348;
x383.backward_filter(x1860,x1861, x1857, x1858, x1859);

// V_cv52_B <~~ X2641 * d_Convolv(1,0)()/d_cv52_B
float x1863, x1864;
x1863 = lrn_rate_2;
x1864 = momentum;
JCudaTensor x1865;
x1865 = x1849;
x363.backward_bias(x1865, x1862, x1863, x1864);

// V_cv54_B <~~ X2665 * d_Convolv(1,0)()/d_cv54_B
float x1867, x1868;
x1867 = lrn_rate_2;
x1868 = momentum;
JCudaTensor x1869;
x1869 = x1853;
x383.backward_bias(x1869, x1866, x1867, x1868);

// val X2643 = (X2619 + X2641 * d_Convolv(1,0)(cv52_W)/d_X149)
JCudaTensor x1870;
JCudaTensor x1871;
x1871 = x1809;
JCudaTensor x1872, x1873;
x1872 = x1849;
x1873 = x361;
x1870 = x363.backward_data(x1872,x1873, x1871);

// V_cv52_W <~~ X2641 * d_Convolv(1,0)(X149)/d_cv52_W
float x1875, x1876;
x1875 = lrn_rate_1;
x1876 = momentum;
JCudaTensor x1877, x1878;
x1877 = x1849;
x1878 = x348;
x363.backward_filter(x1877,x1878, x1874, x1875, x1876);

// Dealloc(X2641)
JCudaTensor x1879;
x1879 = x1849;
x1879.free();

// cv52_B <~~ V_cv52_B
float x1880, x1881;
x1880 = 1;
x1881 = 1;
JCudaTensor x1882;
x1882 = x1862;
x362.update(x1882, x1880, x1881);


// cv54_B <~~ V_cv54_B
float x1883, x1884;
x1883 = 1;
x1884 = 1;
JCudaTensor x1885;
x1885 = x1866;
x382.update(x1885, x1883, x1884);


// cv52_W <~~ V_cv52_W
float x1886, x1887;
x1886 = 1;
x1887 = decay_1;
JCudaTensor x1888;
x1888 = x1874;
x361.update(x1888, x1886, x1887);


// val X2667 = (X2643 + X2665 * d_Convolv(1,0)(cv54_W)/d_X149)
JCudaTensor x1889;
JCudaTensor x1890;
x1890 = x1870;
JCudaTensor x1891, x1892;
x1891 = x1853;
x1892 = x381;
x1889 = x383.backward_data(x1891,x1892, x1890);

// Dealloc(X2665)
JCudaTensor x1893;
x1893 = x1853;
x1893.free();

// cv54_W <~~ V_cv54_W
float x1894, x1895;
x1894 = 1;
x1895 = decay_1;
JCudaTensor x1896;
x1896 = x1857;
x381.update(x1896, x1894, x1895);


// val X2689 = (X2667 + X2686 * d_Pooling(3,1,1,true)(X160,X149)/d_X149)
JCudaTensor x1897;
JCudaTensor x1898;
x1898 = x1889;
JCudaTensor x1899, x1900, x1901;
x1899 = x1778;
x1900 = x367;
x1901 = x348;
x1897 = x369.backward(x1899,x1900,x1901, x1898);

// Dealloc(X2686)
JCudaTensor x1902;
x1902 = x1778;
x1902.free();

// Dealloc(X160)
JCudaTensor x1903;
x1903 = x367;
x1903.free();

// Dealloc(X149)
JCudaTensor x1904;
x1904 = x348;
x1904.free();

// val X2703 = Proj(X2689, X137,X141,X145,X148, 0)
JCudaTensor x1905;
JCudaTensor x1907;
x1907 = x1897;
JCudaTensor[] x1906 = x260.backward(x1907);
x1905 = x1906[0];

// val X2721 = Proj(X2689, X137,X141,X145,X148, 1)
JCudaTensor x1908;
x1908 = x1906[1];

// val X2745 = Proj(X2689, X137,X141,X145,X148, 2)
JCudaTensor x1909;
x1909 = x1906[2];

// val X2769 = Proj(X2689, X137,X141,X145,X148, 3)
JCudaTensor x1910;
x1910 = x1906[3];

// Dealloc(X2689)
JCudaTensor x1911;
x1911 = x1897;
x1911.free();

// val X2773 = X2769 * d_ReLU()(X148)/d_X147
JCudaTensor x1912;
JCudaTensor x1913, x1914;
x1913 = x1910;
x1914 = x338;
x1912 = x340.backward(x1913,x1914);

// Dealloc(X148)
JCudaTensor x1915;
x1915 = x338;
x1915.free();

// val X2706 = X2703 * d_ReLU()(X137)/d_X136
JCudaTensor x1916;
JCudaTensor x1917, x1918;
x1917 = x1905;
x1918 = x332;
x1916 = x334.backward(x1917,x1918);

// Dealloc(X137)
JCudaTensor x1919;
x1919 = x332;
x1919.free();

// val X2750 = X2745 * d_ReLU()(X145)/d_X144
JCudaTensor x1920;
JCudaTensor x1921, x1922;
x1921 = x1909;
x1922 = x329;
x1920 = x331.backward(x1921,x1922);

// Dealloc(X145)
JCudaTensor x1923;
x1923 = x329;
x1923.free();

// val X2726 = X2721 * d_ReLU()(X141)/d_X140
JCudaTensor x1924;
JCudaTensor x1925, x1926;
x1925 = x1908;
x1926 = x335;
x1924 = x337.backward(x1925,x1926);

// Dealloc(X141)
JCudaTensor x1927;
x1927 = x335;
x1927.free();

// V_cv46_W <~~ X2773 * d_Convolv(1,0)(X146)/d_cv46_W
float x1929, x1930;
x1929 = lrn_rate_1;
x1930 = momentum;
JCudaTensor x1931, x1932;
x1931 = x1912;
x1932 = x275;
x311.backward_filter(x1931,x1932, x1928, x1929, x1930);

// V_cv41_B <~~ X2706 * d_Convolv(1,0)()/d_cv41_B
float x1934, x1935;
x1934 = lrn_rate_2;
x1935 = momentum;
JCudaTensor x1936;
x1936 = x1916;
x291.backward_bias(x1936, x1933, x1934, x1935);

// val X2774 = X2773 * d_Convolv(1,0)(cv46_W)/d_X146
JCudaTensor x1937;
JCudaTensor x1938, x1939;
x1938 = x1912;
x1939 = x309;
x1937 = x311.backward_data(x1938,x1939);

// V_cv45_B <~~ X2750 * d_Convolv(1,2)()/d_cv45_B
float x1941, x1942;
x1941 = lrn_rate_2;
x1942 = momentum;
JCudaTensor x1943;
x1943 = x1920;
x321.backward_bias(x1943, x1940, x1941, x1942);

// V_cv41_W <~~ X2706 * d_Convolv(1,0)(X135)/d_cv41_W
float x1945, x1946;
x1945 = lrn_rate_1;
x1946 = momentum;
JCudaTensor x1947, x1948;
x1947 = x1916;
x1948 = x259;
x291.backward_filter(x1947,x1948, x1944, x1945, x1946);

// val X2727 = X2726 * d_Convolv(1,1)(cv43_W)/d_X139
JCudaTensor x1949;
JCudaTensor x1950, x1951;
x1950 = x1924;
x1951 = x326;
x1949 = x328.backward_data(x1950,x1951);

// V_cv45_W <~~ X2750 * d_Convolv(1,2)(X143)/d_cv45_W
float x1953, x1954;
x1953 = lrn_rate_1;
x1954 = momentum;
JCudaTensor x1955, x1956;
x1955 = x1920;
x1956 = x292;
x321.backward_filter(x1955,x1956, x1952, x1953, x1954);

// V_cv46_B <~~ X2773 * d_Convolv(1,0)()/d_cv46_B
float x1958, x1959;
x1958 = lrn_rate_2;
x1959 = momentum;
JCudaTensor x1960;
x1960 = x1912;
x311.backward_bias(x1960, x1957, x1958, x1959);

// Dealloc(X2773)
JCudaTensor x1961;
x1961 = x1912;
x1961.free();

// val X2751 = X2750 * d_Convolv(1,2)(cv45_W)/d_X143
JCudaTensor x1962;
JCudaTensor x1963, x1964;
x1963 = x1920;
x1964 = x319;
x1962 = x321.backward_data(x1963,x1964);

// Dealloc(X2750)
JCudaTensor x1965;
x1965 = x1920;
x1965.free();

// val X2707 = X2706 * d_Convolv(1,0)(cv41_W)/d_X135
JCudaTensor x1966;
JCudaTensor x1967, x1968;
x1967 = x1916;
x1968 = x289;
x1966 = x291.backward_data(x1967,x1968);

// Dealloc(X2706)
JCudaTensor x1969;
x1969 = x1916;
x1969.free();

// V_cv43_B <~~ X2726 * d_Convolv(1,1)()/d_cv43_B
float x1971, x1972;
x1971 = lrn_rate_2;
x1972 = momentum;
JCudaTensor x1973;
x1973 = x1924;
x328.backward_bias(x1973, x1970, x1971, x1972);

// V_cv43_W <~~ X2726 * d_Convolv(1,1)(X139)/d_cv43_W
float x1975, x1976;
x1975 = lrn_rate_1;
x1976 = momentum;
JCudaTensor x1977, x1978;
x1977 = x1924;
x1978 = x302;
x328.backward_filter(x1977,x1978, x1974, x1975, x1976);

// Dealloc(X2726)
JCudaTensor x1979;
x1979 = x1924;
x1979.free();

// cv46_W <~~ V_cv46_W
float x1980, x1981;
x1980 = 1;
x1981 = decay_1;
JCudaTensor x1982;
x1982 = x1928;
x309.update(x1982, x1980, x1981);


// cv43_W <~~ V_cv43_W
float x1983, x1984;
x1983 = 1;
x1984 = decay_1;
JCudaTensor x1985;
x1985 = x1974;
x326.update(x1985, x1983, x1984);


// cv43_B <~~ V_cv43_B
float x1986, x1987;
x1986 = 1;
x1987 = 1;
JCudaTensor x1988;
x1988 = x1970;
x327.update(x1988, x1986, x1987);


// cv41_B <~~ V_cv41_B
float x1989, x1990;
x1989 = 1;
x1990 = 1;
JCudaTensor x1991;
x1991 = x1933;
x290.update(x1991, x1989, x1990);


// cv41_W <~~ V_cv41_W
float x1992, x1993;
x1992 = 1;
x1993 = decay_1;
JCudaTensor x1994;
x1994 = x1944;
x289.update(x1994, x1992, x1993);


// cv45_W <~~ V_cv45_W
float x1995, x1996;
x1995 = 1;
x1996 = decay_1;
JCudaTensor x1997;
x1997 = x1952;
x319.update(x1997, x1995, x1996);


// cv46_B <~~ V_cv46_B
float x1998, x1999;
x1998 = 1;
x1999 = 1;
JCudaTensor x2000;
x2000 = x1957;
x310.update(x2000, x1998, x1999);


// cv45_B <~~ V_cv45_B
float x2001, x2002;
x2001 = 1;
x2002 = 1;
JCudaTensor x2003;
x2003 = x1940;
x320.update(x2003, x2001, x2002);


// val X2753 = X2751 * d_ReLU()(X143)/d_X142
JCudaTensor x2004;
JCudaTensor x2005, x2006;
x2005 = x1962;
x2006 = x292;
x2004 = x294.backward(x2005,x2006);

// Dealloc(X143)
JCudaTensor x2007;
x2007 = x292;
x2007.free();

// val X2729 = X2727 * d_ReLU()(X139)/d_X138
JCudaTensor x2008;
JCudaTensor x2009, x2010;
x2009 = x1949;
x2010 = x302;
x2008 = x304.backward(x2009,x2010);

// Dealloc(X139)
JCudaTensor x2011;
x2011 = x302;
x2011.free();

// V_cv44_W <~~ X2753 * d_Convolv(1,0)(X135)/d_cv44_W
float x2013, x2014;
x2013 = lrn_rate_1;
x2014 = momentum;
JCudaTensor x2015, x2016;
x2015 = x2004;
x2016 = x259;
x274.backward_filter(x2015,x2016, x2012, x2013, x2014);

// val X2731 = (X2707 + X2729 * d_Convolv(1,0)(cv42_W)/d_X135)
JCudaTensor x2017;
JCudaTensor x2018;
x2018 = x1966;
JCudaTensor x2019, x2020;
x2019 = x2008;
x2020 = x282;
x2017 = x284.backward_data(x2019,x2020, x2018);

// V_cv42_W <~~ X2729 * d_Convolv(1,0)(X135)/d_cv42_W
float x2022, x2023;
x2022 = lrn_rate_1;
x2023 = momentum;
JCudaTensor x2024, x2025;
x2024 = x2008;
x2025 = x259;
x284.backward_filter(x2024,x2025, x2021, x2022, x2023);

// V_cv42_B <~~ X2729 * d_Convolv(1,0)()/d_cv42_B
float x2027, x2028;
x2027 = lrn_rate_2;
x2028 = momentum;
JCudaTensor x2029;
x2029 = x2008;
x284.backward_bias(x2029, x2026, x2027, x2028);

// Dealloc(X2729)
JCudaTensor x2030;
x2030 = x2008;
x2030.free();

// V_cv44_B <~~ X2753 * d_Convolv(1,0)()/d_cv44_B
float x2032, x2033;
x2032 = lrn_rate_2;
x2033 = momentum;
JCudaTensor x2034;
x2034 = x2004;
x274.backward_bias(x2034, x2031, x2032, x2033);

// cv42_W <~~ V_cv42_W
float x2035, x2036;
x2035 = 1;
x2036 = decay_1;
JCudaTensor x2037;
x2037 = x2021;
x282.update(x2037, x2035, x2036);


// cv42_B <~~ V_cv42_B
float x2038, x2039;
x2038 = 1;
x2039 = 1;
JCudaTensor x2040;
x2040 = x2026;
x283.update(x2040, x2038, x2039);


// cv44_B <~~ V_cv44_B
float x2041, x2042;
x2041 = 1;
x2042 = 1;
JCudaTensor x2043;
x2043 = x2031;
x273.update(x2043, x2041, x2042);


// val X2755 = (X2731 + X2753 * d_Convolv(1,0)(cv44_W)/d_X135)
JCudaTensor x2044;
JCudaTensor x2045;
x2045 = x2017;
JCudaTensor x2046, x2047;
x2046 = x2004;
x2047 = x272;
x2044 = x274.backward_data(x2046,x2047, x2045);

// Dealloc(X2753)
JCudaTensor x2048;
x2048 = x2004;
x2048.free();

// cv44_W <~~ V_cv44_W
float x2049, x2050;
x2049 = 1;
x2050 = decay_1;
JCudaTensor x2051;
x2051 = x2012;
x272.update(x2051, x2049, x2050);


// val X2777 = (X2755 + X2774 * d_Pooling(3,1,1,true)(X146,X135)/d_X135)
JCudaTensor x2052;
JCudaTensor x2053;
x2053 = x2044;
JCudaTensor x2054, x2055, x2056;
x2054 = x1937;
x2055 = x275;
x2056 = x259;
x2052 = x277.backward(x2054,x2055,x2056, x2053);

// Dealloc(X2774)
JCudaTensor x2057;
x2057 = x1937;
x2057.free();

// Dealloc(X146)
JCudaTensor x2058;
x2058 = x275;
x2058.free();

// Dealloc(X135)
JCudaTensor x2059;
x2059 = x259;
x2059.free();

// val X2813 = (X2812 * loss1)
JCudaTensor x2060;
JCudaTensor x2061;
float x2062;
x2061 = x658;
x2062 = loss1;
x2060 = x2061.times_i(x2062);

// val X2814 = (X2777 + X2813)
JCudaTensor x2063;
JCudaTensor x2064, x2065;
x2064 = x2052;
x2065 = x2060;
x2063 = x2064.plus_i(x2065);

// Dealloc(X2813)
JCudaTensor x2066;
x2066 = x2060;
x2066.free();

// val X2909 = Proj(X2814, X123,X127,X131,X134, 2)
JCudaTensor x2067;
JCudaTensor x2069;
x2069 = x2063;
JCudaTensor[] x2068 = x260.backward(x2069);
x2067 = x2068[2];

// val X2867 = Proj(X2814, X123,X127,X131,X134, 0)
JCudaTensor x2070;
x2070 = x2068[0];

// val X2933 = Proj(X2814, X123,X127,X131,X134, 3)
JCudaTensor x2071;
x2071 = x2068[3];

// val X2885 = Proj(X2814, X123,X127,X131,X134, 1)
JCudaTensor x2072;
x2072 = x2068[1];

// Dealloc(X2814)
JCudaTensor x2073;
x2073 = x2063;
x2073.free();

// val X2937 = X2933 * d_ReLU()(X134)/d_X133
JCudaTensor x2074;
JCudaTensor x2075, x2076;
x2075 = x2071;
x2076 = x256;
x2074 = x258.backward(x2075,x2076);

// Dealloc(X134)
JCudaTensor x2077;
x2077 = x256;
x2077.free();

// val X2870 = X2867 * d_ReLU()(X123)/d_X122
JCudaTensor x2078;
JCudaTensor x2079, x2080;
x2079 = x2070;
x2080 = x247;
x2078 = x249.backward(x2079,x2080);

// Dealloc(X123)
JCudaTensor x2081;
x2081 = x247;
x2081.free();

// val X2914 = X2909 * d_ReLU()(X131)/d_X130
JCudaTensor x2082;
JCudaTensor x2083, x2084;
x2083 = x2067;
x2084 = x253;
x2082 = x255.backward(x2083,x2084);

// Dealloc(X131)
JCudaTensor x2085;
x2085 = x253;
x2085.free();

// val X2890 = X2885 * d_ReLU()(X127)/d_X126
JCudaTensor x2086;
JCudaTensor x2087, x2088;
x2087 = x2072;
x2088 = x250;
x2086 = x252.backward(x2087,x2088);

// Dealloc(X127)
JCudaTensor x2089;
x2089 = x250;
x2089.free();

// V_cv36_W <~~ X2937 * d_Convolv(1,0)(X132)/d_cv36_W
float x2091, x2092;
x2091 = lrn_rate_1;
x2092 = momentum;
JCudaTensor x2093, x2094;
x2093 = x2074;
x2094 = x210;
x232.backward_filter(x2093,x2094, x2090, x2091, x2092);

// V_cv31_W <~~ X2870 * d_Convolv(1,0)(X121)/d_cv31_W
float x2096, x2097;
x2096 = lrn_rate_1;
x2097 = momentum;
JCudaTensor x2098, x2099;
x2098 = x2078;
x2099 = x193;
x202.backward_filter(x2098,x2099, x2095, x2096, x2097);

// V_cv35_W <~~ X2914 * d_Convolv(1,2)(X129)/d_cv35_W
float x2101, x2102;
x2101 = lrn_rate_1;
x2102 = momentum;
JCudaTensor x2103, x2104;
x2103 = x2082;
x2104 = x220;
x239.backward_filter(x2103,x2104, x2100, x2101, x2102);

// V_cv33_B <~~ X2890 * d_Convolv(1,1)()/d_cv33_B
float x2106, x2107;
x2106 = lrn_rate_2;
x2107 = momentum;
JCudaTensor x2108;
x2108 = x2086;
x246.backward_bias(x2108, x2105, x2106, x2107);

// V_cv33_W <~~ X2890 * d_Convolv(1,1)(X125)/d_cv33_W
float x2110, x2111;
x2110 = lrn_rate_1;
x2111 = momentum;
JCudaTensor x2112, x2113;
x2112 = x2086;
x2113 = x223;
x246.backward_filter(x2112,x2113, x2109, x2110, x2111);

// V_cv31_B <~~ X2870 * d_Convolv(1,0)()/d_cv31_B
float x2115, x2116;
x2115 = lrn_rate_2;
x2116 = momentum;
JCudaTensor x2117;
x2117 = x2078;
x202.backward_bias(x2117, x2114, x2115, x2116);

// val X2871 = X2870 * d_Convolv(1,0)(cv31_W)/d_X121
JCudaTensor x2118;
JCudaTensor x2119, x2120;
x2119 = x2078;
x2120 = x200;
x2118 = x202.backward_data(x2119,x2120);

// Dealloc(X2870)
JCudaTensor x2121;
x2121 = x2078;
x2121.free();

// V_cv35_B <~~ X2914 * d_Convolv(1,2)()/d_cv35_B
float x2123, x2124;
x2123 = lrn_rate_2;
x2124 = momentum;
JCudaTensor x2125;
x2125 = x2082;
x239.backward_bias(x2125, x2122, x2123, x2124);

// val X2891 = X2890 * d_Convolv(1,1)(cv33_W)/d_X125
JCudaTensor x2126;
JCudaTensor x2127, x2128;
x2127 = x2086;
x2128 = x244;
x2126 = x246.backward_data(x2127,x2128);

// Dealloc(X2890)
JCudaTensor x2129;
x2129 = x2086;
x2129.free();

// val X2938 = X2937 * d_Convolv(1,0)(cv36_W)/d_X132
JCudaTensor x2130;
JCudaTensor x2131, x2132;
x2131 = x2074;
x2132 = x230;
x2130 = x232.backward_data(x2131,x2132);

// V_cv36_B <~~ X2937 * d_Convolv(1,0)()/d_cv36_B
float x2134, x2135;
x2134 = lrn_rate_2;
x2135 = momentum;
JCudaTensor x2136;
x2136 = x2074;
x232.backward_bias(x2136, x2133, x2134, x2135);

// Dealloc(X2937)
JCudaTensor x2137;
x2137 = x2074;
x2137.free();

// val X2915 = X2914 * d_Convolv(1,2)(cv35_W)/d_X129
JCudaTensor x2138;
JCudaTensor x2139, x2140;
x2139 = x2082;
x2140 = x237;
x2138 = x239.backward_data(x2139,x2140);

// Dealloc(X2914)
JCudaTensor x2141;
x2141 = x2082;
x2141.free();

// cv33_B <~~ V_cv33_B
float x2142, x2143;
x2142 = 1;
x2143 = 1;
JCudaTensor x2144;
x2144 = x2105;
x245.update(x2144, x2142, x2143);


// cv31_B <~~ V_cv31_B
float x2145, x2146;
x2145 = 1;
x2146 = 1;
JCudaTensor x2147;
x2147 = x2114;
x201.update(x2147, x2145, x2146);


// cv35_W <~~ V_cv35_W
float x2148, x2149;
x2148 = 1;
x2149 = decay_1;
JCudaTensor x2150;
x2150 = x2100;
x237.update(x2150, x2148, x2149);


// cv36_B <~~ V_cv36_B
float x2151, x2152;
x2151 = 1;
x2152 = 1;
JCudaTensor x2153;
x2153 = x2133;
x231.update(x2153, x2151, x2152);


// cv33_W <~~ V_cv33_W
float x2154, x2155;
x2154 = 1;
x2155 = decay_1;
JCudaTensor x2156;
x2156 = x2109;
x244.update(x2156, x2154, x2155);


// cv35_B <~~ V_cv35_B
float x2157, x2158;
x2157 = 1;
x2158 = 1;
JCudaTensor x2159;
x2159 = x2122;
x238.update(x2159, x2157, x2158);


// cv31_W <~~ V_cv31_W
float x2160, x2161;
x2160 = 1;
x2161 = decay_1;
JCudaTensor x2162;
x2162 = x2095;
x200.update(x2162, x2160, x2161);


// cv36_W <~~ V_cv36_W
float x2163, x2164;
x2163 = 1;
x2164 = decay_1;
JCudaTensor x2165;
x2165 = x2090;
x230.update(x2165, x2163, x2164);


// val X2893 = X2891 * d_ReLU()(X125)/d_X124
JCudaTensor x2166;
JCudaTensor x2167, x2168;
x2167 = x2126;
x2168 = x223;
x2166 = x225.backward(x2167,x2168);

// Dealloc(X125)
JCudaTensor x2169;
x2169 = x223;
x2169.free();

// val X2917 = X2915 * d_ReLU()(X129)/d_X128
JCudaTensor x2170;
JCudaTensor x2171, x2172;
x2171 = x2138;
x2172 = x220;
x2170 = x222.backward(x2171,x2172);

// Dealloc(X129)
JCudaTensor x2173;
x2173 = x220;
x2173.free();

// V_cv34_W <~~ X2917 * d_Convolv(1,0)(X121)/d_cv34_W
float x2175, x2176;
x2175 = lrn_rate_1;
x2176 = momentum;
JCudaTensor x2177, x2178;
x2177 = x2170;
x2178 = x193;
x219.backward_filter(x2177,x2178, x2174, x2175, x2176);

// val X2895 = (X2871 + X2893 * d_Convolv(1,0)(cv32_W)/d_X121)
JCudaTensor x2179;
JCudaTensor x2180;
x2180 = x2118;
JCudaTensor x2181, x2182;
x2181 = x2166;
x2182 = x207;
x2179 = x209.backward_data(x2181,x2182, x2180);

// V_cv32_W <~~ X2893 * d_Convolv(1,0)(X121)/d_cv32_W
float x2184, x2185;
x2184 = lrn_rate_1;
x2185 = momentum;
JCudaTensor x2186, x2187;
x2186 = x2166;
x2187 = x193;
x209.backward_filter(x2186,x2187, x2183, x2184, x2185);

// V_cv34_B <~~ X2917 * d_Convolv(1,0)()/d_cv34_B
float x2189, x2190;
x2189 = lrn_rate_2;
x2190 = momentum;
JCudaTensor x2191;
x2191 = x2170;
x219.backward_bias(x2191, x2188, x2189, x2190);

// V_cv32_B <~~ X2893 * d_Convolv(1,0)()/d_cv32_B
float x2193, x2194;
x2193 = lrn_rate_2;
x2194 = momentum;
JCudaTensor x2195;
x2195 = x2166;
x209.backward_bias(x2195, x2192, x2193, x2194);

// Dealloc(X2893)
JCudaTensor x2196;
x2196 = x2166;
x2196.free();

// cv32_W <~~ V_cv32_W
float x2197, x2198;
x2197 = 1;
x2198 = decay_1;
JCudaTensor x2199;
x2199 = x2183;
x207.update(x2199, x2197, x2198);


// cv34_B <~~ V_cv34_B
float x2200, x2201;
x2200 = 1;
x2201 = 1;
JCudaTensor x2202;
x2202 = x2188;
x218.update(x2202, x2200, x2201);


// cv32_B <~~ V_cv32_B
float x2203, x2204;
x2203 = 1;
x2204 = 1;
JCudaTensor x2205;
x2205 = x2192;
x208.update(x2205, x2203, x2204);


// val X2919 = (X2895 + X2917 * d_Convolv(1,0)(cv34_W)/d_X121)
JCudaTensor x2206;
JCudaTensor x2207;
x2207 = x2179;
JCudaTensor x2208, x2209;
x2208 = x2170;
x2209 = x217;
x2206 = x219.backward_data(x2208,x2209, x2207);

// Dealloc(X2917)
JCudaTensor x2210;
x2210 = x2170;
x2210.free();

// cv34_W <~~ V_cv34_W
float x2211, x2212;
x2211 = 1;
x2212 = decay_1;
JCudaTensor x2213;
x2213 = x2174;
x217.update(x2213, x2211, x2212);


// val X2941 = (X2919 + X2938 * d_Pooling(3,1,1,true)(X132,X121)/d_X121)
JCudaTensor x2214;
JCudaTensor x2215;
x2215 = x2206;
JCudaTensor x2216, x2217, x2218;
x2216 = x2130;
x2217 = x210;
x2218 = x193;
x2214 = x212.backward(x2216,x2217,x2218, x2215);

// Dealloc(X2938)
JCudaTensor x2219;
x2219 = x2130;
x2219.free();

// Dealloc(X132)
JCudaTensor x2220;
x2220 = x210;
x2220.free();

// val X2943 = X2941 * d_Pooling(3,2,1,true)(X121,X120)/d_X120
JCudaTensor x2221;
JCudaTensor x2222, x2223, x2224;
x2222 = x2214;
x2223 = x193;
x2224 = x188;
x2221 = x195.backward(x2222,x2223,x2224);

// Dealloc(X2941)
JCudaTensor x2225;
x2225 = x2214;
x2225.free();

// Dealloc(X121)
JCudaTensor x2226;
x2226 = x193;
x2226.free();

// Dealloc(X120)
JCudaTensor x2227;
x2227 = x188;
x2227.free();

// val X2957 = Proj(X2943, X108,X112,X116,X119, 0)
JCudaTensor x2228;
JCudaTensor x2230;
x2230 = x2221;
JCudaTensor[] x2229 = x120.backward(x2230);
x2228 = x2229[0];

// val X2999 = Proj(X2943, X108,X112,X116,X119, 2)
JCudaTensor x2231;
x2231 = x2229[2];

// val X2975 = Proj(X2943, X108,X112,X116,X119, 1)
JCudaTensor x2232;
x2232 = x2229[1];

// val X3023 = Proj(X2943, X108,X112,X116,X119, 3)
JCudaTensor x2233;
x2233 = x2229[3];

// Dealloc(X2943)
JCudaTensor x2234;
x2234 = x2221;
x2234.free();

// val X3027 = X3023 * d_ReLU()(X119)/d_X118
JCudaTensor x2235;
JCudaTensor x2236, x2237;
x2236 = x2233;
x2237 = x185;
x2235 = x187.backward(x2236,x2237);

// Dealloc(X119)
JCudaTensor x2238;
x2238 = x185;
x2238.free();

// val X3004 = X2999 * d_ReLU()(X116)/d_X115
JCudaTensor x2239;
JCudaTensor x2240, x2241;
x2240 = x2231;
x2241 = x182;
x2239 = x184.backward(x2240,x2241);

// Dealloc(X116)
JCudaTensor x2242;
x2242 = x182;
x2242.free();

// val X2980 = X2975 * d_ReLU()(X112)/d_X111
JCudaTensor x2243;
JCudaTensor x2244, x2245;
x2244 = x2232;
x2245 = x179;
x2243 = x181.backward(x2244,x2245);

// Dealloc(X112)
JCudaTensor x2246;
x2246 = x179;
x2246.free();

// val X2960 = X2957 * d_ReLU()(X108)/d_X107
JCudaTensor x2247;
JCudaTensor x2248, x2249;
x2248 = x2228;
x2249 = x176;
x2247 = x178.backward(x2248,x2249);

// Dealloc(X108)
JCudaTensor x2250;
x2250 = x176;
x2250.free();

// V_cv26_B <~~ X3027 * d_Convolv(1,0)()/d_cv26_B
float x2252, x2253;
x2252 = lrn_rate_2;
x2253 = momentum;
JCudaTensor x2254;
x2254 = x2235;
x158.backward_bias(x2254, x2251, x2252, x2253);

// val X3005 = X3004 * d_Convolv(1,2)(cv25_W)/d_X114
JCudaTensor x2255;
JCudaTensor x2256, x2257;
x2256 = x2239;
x2257 = x173;
x2255 = x175.backward_data(x2256,x2257);

// V_cv26_W <~~ X3027 * d_Convolv(1,0)(X117)/d_cv26_W
float x2259, x2260;
x2259 = lrn_rate_1;
x2260 = momentum;
JCudaTensor x2261, x2262;
x2261 = x2235;
x2262 = x125;
x158.backward_filter(x2261,x2262, x2258, x2259, x2260);

// V_cv25_W <~~ X3004 * d_Convolv(1,2)(X114)/d_cv25_W
float x2264, x2265;
x2264 = lrn_rate_1;
x2265 = momentum;
JCudaTensor x2266, x2267;
x2266 = x2239;
x2267 = x159;
x175.backward_filter(x2266,x2267, x2263, x2264, x2265);

// val X2981 = X2980 * d_Convolv(1,1)(cv23_W)/d_X110
JCudaTensor x2268;
JCudaTensor x2269, x2270;
x2269 = x2243;
x2270 = x166;
x2268 = x168.backward_data(x2269,x2270);

// V_cv21_W <~~ X2960 * d_Convolv(1,0)(X106)/d_cv21_W
float x2272, x2273;
x2272 = lrn_rate_1;
x2273 = momentum;
JCudaTensor x2274, x2275;
x2274 = x2247;
x2275 = x119;
x141.backward_filter(x2274,x2275, x2271, x2272, x2273);

// V_cv23_B <~~ X2980 * d_Convolv(1,1)()/d_cv23_B
float x2277, x2278;
x2277 = lrn_rate_2;
x2278 = momentum;
JCudaTensor x2279;
x2279 = x2243;
x168.backward_bias(x2279, x2276, x2277, x2278);

// V_cv25_B <~~ X3004 * d_Convolv(1,2)()/d_cv25_B
float x2281, x2282;
x2281 = lrn_rate_2;
x2282 = momentum;
JCudaTensor x2283;
x2283 = x2239;
x175.backward_bias(x2283, x2280, x2281, x2282);

// Dealloc(X3004)
JCudaTensor x2284;
x2284 = x2239;
x2284.free();

// V_cv21_B <~~ X2960 * d_Convolv(1,0)()/d_cv21_B
float x2286, x2287;
x2286 = lrn_rate_2;
x2287 = momentum;
JCudaTensor x2288;
x2288 = x2247;
x141.backward_bias(x2288, x2285, x2286, x2287);

// val X2961 = X2960 * d_Convolv(1,0)(cv21_W)/d_X106
JCudaTensor x2289;
JCudaTensor x2290, x2291;
x2290 = x2247;
x2291 = x139;
x2289 = x141.backward_data(x2290,x2291);

// Dealloc(X2960)
JCudaTensor x2292;
x2292 = x2247;
x2292.free();

// val X3028 = X3027 * d_Convolv(1,0)(cv26_W)/d_X117
JCudaTensor x2293;
JCudaTensor x2294, x2295;
x2294 = x2235;
x2295 = x156;
x2293 = x158.backward_data(x2294,x2295);

// Dealloc(X3027)
JCudaTensor x2296;
x2296 = x2235;
x2296.free();

// V_cv23_W <~~ X2980 * d_Convolv(1,1)(X110)/d_cv23_W
float x2298, x2299;
x2298 = lrn_rate_1;
x2299 = momentum;
JCudaTensor x2300, x2301;
x2300 = x2243;
x2301 = x149;
x168.backward_filter(x2300,x2301, x2297, x2298, x2299);

// Dealloc(X2980)
JCudaTensor x2302;
x2302 = x2243;
x2302.free();

// cv25_W <~~ V_cv25_W
float x2303, x2304;
x2303 = 1;
x2304 = decay_1;
JCudaTensor x2305;
x2305 = x2263;
x173.update(x2305, x2303, x2304);


// cv23_B <~~ V_cv23_B
float x2306, x2307;
x2306 = 1;
x2307 = 1;
JCudaTensor x2308;
x2308 = x2276;
x167.update(x2308, x2306, x2307);


// cv25_B <~~ V_cv25_B
float x2309, x2310;
x2309 = 1;
x2310 = 1;
JCudaTensor x2311;
x2311 = x2280;
x174.update(x2311, x2309, x2310);


// cv26_W <~~ V_cv26_W
float x2312, x2313;
x2312 = 1;
x2313 = decay_1;
JCudaTensor x2314;
x2314 = x2258;
x156.update(x2314, x2312, x2313);


// cv23_W <~~ V_cv23_W
float x2315, x2316;
x2315 = 1;
x2316 = decay_1;
JCudaTensor x2317;
x2317 = x2297;
x166.update(x2317, x2315, x2316);


// cv26_B <~~ V_cv26_B
float x2318, x2319;
x2318 = 1;
x2319 = 1;
JCudaTensor x2320;
x2320 = x2251;
x157.update(x2320, x2318, x2319);


// cv21_B <~~ V_cv21_B
float x2321, x2322;
x2321 = 1;
x2322 = 1;
JCudaTensor x2323;
x2323 = x2285;
x140.update(x2323, x2321, x2322);


// cv21_W <~~ V_cv21_W
float x2324, x2325;
x2324 = 1;
x2325 = decay_1;
JCudaTensor x2326;
x2326 = x2271;
x139.update(x2326, x2324, x2325);


// val X3007 = X3005 * d_ReLU()(X114)/d_X113
JCudaTensor x2327;
JCudaTensor x2328, x2329;
x2328 = x2255;
x2329 = x159;
x2327 = x161.backward(x2328,x2329);

// Dealloc(X114)
JCudaTensor x2330;
x2330 = x159;
x2330.free();

// val X2983 = X2981 * d_ReLU()(X110)/d_X109
JCudaTensor x2331;
JCudaTensor x2332, x2333;
x2332 = x2268;
x2333 = x149;
x2331 = x151.backward(x2332,x2333);

// Dealloc(X110)
JCudaTensor x2334;
x2334 = x149;
x2334.free();

// V_cv24_W <~~ X3007 * d_Convolv(1,0)(X106)/d_cv24_W
float x2336, x2337;
x2336 = lrn_rate_1;
x2337 = momentum;
JCudaTensor x2338, x2339;
x2338 = x2327;
x2339 = x119;
x134.backward_filter(x2338,x2339, x2335, x2336, x2337);

// V_cv24_B <~~ X3007 * d_Convolv(1,0)()/d_cv24_B
float x2341, x2342;
x2341 = lrn_rate_2;
x2342 = momentum;
JCudaTensor x2343;
x2343 = x2327;
x134.backward_bias(x2343, x2340, x2341, x2342);

// V_cv22_B <~~ X2983 * d_Convolv(1,0)()/d_cv22_B
float x2345, x2346;
x2345 = lrn_rate_2;
x2346 = momentum;
JCudaTensor x2347;
x2347 = x2331;
x148.backward_bias(x2347, x2344, x2345, x2346);

// val X2985 = (X2961 + X2983 * d_Convolv(1,0)(cv22_W)/d_X106)
JCudaTensor x2348;
JCudaTensor x2349;
x2349 = x2289;
JCudaTensor x2350, x2351;
x2350 = x2331;
x2351 = x146;
x2348 = x148.backward_data(x2350,x2351, x2349);

// V_cv22_W <~~ X2983 * d_Convolv(1,0)(X106)/d_cv22_W
float x2353, x2354;
x2353 = lrn_rate_1;
x2354 = momentum;
JCudaTensor x2355, x2356;
x2355 = x2331;
x2356 = x119;
x148.backward_filter(x2355,x2356, x2352, x2353, x2354);

// Dealloc(X2983)
JCudaTensor x2357;
x2357 = x2331;
x2357.free();

// cv24_B <~~ V_cv24_B
float x2358, x2359;
x2358 = 1;
x2359 = 1;
JCudaTensor x2360;
x2360 = x2340;
x133.update(x2360, x2358, x2359);


// cv22_B <~~ V_cv22_B
float x2361, x2362;
x2361 = 1;
x2362 = 1;
JCudaTensor x2363;
x2363 = x2344;
x147.update(x2363, x2361, x2362);


// cv22_W <~~ V_cv22_W
float x2364, x2365;
x2364 = 1;
x2365 = decay_1;
JCudaTensor x2366;
x2366 = x2352;
x146.update(x2366, x2364, x2365);


// val X3009 = (X2985 + X3007 * d_Convolv(1,0)(cv24_W)/d_X106)
JCudaTensor x2367;
JCudaTensor x2368;
x2368 = x2348;
JCudaTensor x2369, x2370;
x2369 = x2327;
x2370 = x132;
x2367 = x134.backward_data(x2369,x2370, x2368);

// Dealloc(X3007)
JCudaTensor x2371;
x2371 = x2327;
x2371.free();

// cv24_W <~~ V_cv24_W
float x2372, x2373;
x2372 = 1;
x2373 = decay_1;
JCudaTensor x2374;
x2374 = x2335;
x132.update(x2374, x2372, x2373);


// val X3031 = (X3009 + X3028 * d_Pooling(3,1,1,true)(X117,X106)/d_X106)
JCudaTensor x2375;
JCudaTensor x2376;
x2376 = x2367;
JCudaTensor x2377, x2378, x2379;
x2377 = x2293;
x2378 = x125;
x2379 = x119;
x2375 = x127.backward(x2377,x2378,x2379, x2376);

// Dealloc(X3028)
JCudaTensor x2380;
x2380 = x2293;
x2380.free();

// Dealloc(X117)
JCudaTensor x2381;
x2381 = x125;
x2381.free();

// Dealloc(X106)
JCudaTensor x2382;
x2382 = x119;
x2382.free();

// val X5817 = Proj(X3031, X94,X98,X102,X105, 1)
JCudaTensor x2383;
JCudaTensor x2385;
x2385 = x2375;
JCudaTensor[] x2384 = x120.backward(x2385);
x2383 = x2384[1];

// val X11379 = Proj(X3031, X94,X98,X102,X105, 2)
JCudaTensor x2386;
x2386 = x2384[2];

// val X16938 = Proj(X3031, X94,X98,X102,X105, 3)
JCudaTensor x2387;
x2387 = x2384[3];

// val X3045 = Proj(X3031, X94,X98,X102,X105, 0)
JCudaTensor x2388;
x2388 = x2384[0];

// Dealloc(X3031)
JCudaTensor x2389;
x2389 = x2375;
x2389.free();

// val X5822 = X5817 * d_ReLU()(X98)/d_X97
JCudaTensor x2390;
JCudaTensor x2391, x2392;
x2391 = x2383;
x2392 = x110;
x2390 = x112.backward(x2391,x2392);

// Dealloc(X98)
JCudaTensor x2393;
x2393 = x110;
x2393.free();

// val X3048 = X3045 * d_ReLU()(X94)/d_X93
JCudaTensor x2394;
JCudaTensor x2395, x2396;
x2395 = x2388;
x2396 = x107;
x2394 = x109.backward(x2395,x2396);

// Dealloc(X94)
JCudaTensor x2397;
x2397 = x107;
x2397.free();

// val X16942 = X16938 * d_ReLU()(X105)/d_X104
JCudaTensor x2398;
JCudaTensor x2399, x2400;
x2399 = x2387;
x2400 = x116;
x2398 = x118.backward(x2399,x2400);

// Dealloc(X105)
JCudaTensor x2401;
x2401 = x116;
x2401.free();

// val X11384 = X11379 * d_ReLU()(X102)/d_X101
JCudaTensor x2402;
JCudaTensor x2403, x2404;
x2403 = x2386;
x2404 = x113;
x2402 = x115.backward(x2403,x2404);

// Dealloc(X102)
JCudaTensor x2405;
x2405 = x113;
x2405.free();

// val X5823 = X5822 * d_Convolv(1,1)(cv13_W)/d_X96
JCudaTensor x2406;
JCudaTensor x2407, x2408;
x2407 = x2390;
x2408 = x97;
x2406 = x99.backward_data(x2407,x2408);

// V_cv11_W <~~ X3048 * d_Convolv(1,0)(X92)/d_cv11_W
float x2410, x2411;
x2410 = lrn_rate_1;
x2411 = momentum;
JCudaTensor x2412, x2413;
x2412 = x2394;
x2413 = x53;
x62.backward_filter(x2412,x2413, x2409, x2410, x2411);

// V_cv16_B <~~ X16942 * d_Convolv(1,0)()/d_cv16_B
float x2415, x2416;
x2415 = lrn_rate_2;
x2416 = momentum;
JCudaTensor x2417;
x2417 = x2398;
x86.backward_bias(x2417, x2414, x2415, x2416);

// val X11385 = X11384 * d_Convolv(1,2)(cv15_W)/d_X100
JCudaTensor x2418;
JCudaTensor x2419, x2420;
x2419 = x2402;
x2420 = x104;
x2418 = x106.backward_data(x2419,x2420);

// V_cv15_B <~~ X11384 * d_Convolv(1,2)()/d_cv15_B
float x2422, x2423;
x2422 = lrn_rate_2;
x2423 = momentum;
JCudaTensor x2424;
x2424 = x2402;
x106.backward_bias(x2424, x2421, x2422, x2423);

// V_cv16_W <~~ X16942 * d_Convolv(1,0)(X103)/d_cv16_W
float x2426, x2427;
x2426 = lrn_rate_1;
x2427 = momentum;
JCudaTensor x2428, x2429;
x2428 = x2398;
x2429 = x70;
x86.backward_filter(x2428,x2429, x2425, x2426, x2427);

// val X19749 = X3048 * d_Convolv(1,0)(cv11_W)/d_X92
JCudaTensor x2430;
JCudaTensor x2431, x2432;
x2431 = x2394;
x2432 = x60;
x2430 = x62.backward_data(x2431,x2432);

// V_cv13_B <~~ X5822 * d_Convolv(1,1)()/d_cv13_B
float x2434, x2435;
x2434 = lrn_rate_2;
x2435 = momentum;
JCudaTensor x2436;
x2436 = x2390;
x99.backward_bias(x2436, x2433, x2434, x2435);

// V_cv15_W <~~ X11384 * d_Convolv(1,2)(X100)/d_cv15_W
float x2438, x2439;
x2438 = lrn_rate_1;
x2439 = momentum;
JCudaTensor x2440, x2441;
x2440 = x2402;
x2441 = x90;
x106.backward_filter(x2440,x2441, x2437, x2438, x2439);

// Dealloc(X11384)
JCudaTensor x2442;
x2442 = x2402;
x2442.free();

// V_cv11_B <~~ X3048 * d_Convolv(1,0)()/d_cv11_B
float x2444, x2445;
x2444 = lrn_rate_2;
x2445 = momentum;
JCudaTensor x2446;
x2446 = x2394;
x62.backward_bias(x2446, x2443, x2444, x2445);

// Dealloc(X3048)
JCudaTensor x2447;
x2447 = x2394;
x2447.free();

// V_cv13_W <~~ X5822 * d_Convolv(1,1)(X96)/d_cv13_W
float x2449, x2450;
x2449 = lrn_rate_1;
x2450 = momentum;
JCudaTensor x2451, x2452;
x2451 = x2390;
x2452 = x87;
x99.backward_filter(x2451,x2452, x2448, x2449, x2450);

// Dealloc(X5822)
JCudaTensor x2453;
x2453 = x2390;
x2453.free();

// val X19816 = X16942 * d_Convolv(1,0)(cv16_W)/d_X103
JCudaTensor x2454;
JCudaTensor x2455, x2456;
x2455 = x2398;
x2456 = x84;
x2454 = x86.backward_data(x2455,x2456);

// Dealloc(X16942)
JCudaTensor x2457;
x2457 = x2398;
x2457.free();

// cv15_B <~~ V_cv15_B
float x2458, x2459;
x2458 = 1;
x2459 = 1;
JCudaTensor x2460;
x2460 = x2421;
x105.update(x2460, x2458, x2459);


// cv13_B <~~ V_cv13_B
float x2461, x2462;
x2461 = 1;
x2462 = 1;
JCudaTensor x2463;
x2463 = x2433;
x98.update(x2463, x2461, x2462);


// cv11_W <~~ V_cv11_W
float x2464, x2465;
x2464 = 1;
x2465 = decay_1;
JCudaTensor x2466;
x2466 = x2409;
x60.update(x2466, x2464, x2465);


// cv11_B <~~ V_cv11_B
float x2467, x2468;
x2467 = 1;
x2468 = 1;
JCudaTensor x2469;
x2469 = x2443;
x61.update(x2469, x2467, x2468);


// cv13_W <~~ V_cv13_W
float x2470, x2471;
x2470 = 1;
x2471 = decay_1;
JCudaTensor x2472;
x2472 = x2448;
x97.update(x2472, x2470, x2471);


// cv16_W <~~ V_cv16_W
float x2473, x2474;
x2473 = 1;
x2474 = decay_1;
JCudaTensor x2475;
x2475 = x2425;
x84.update(x2475, x2473, x2474);


// cv16_B <~~ V_cv16_B
float x2476, x2477;
x2476 = 1;
x2477 = 1;
JCudaTensor x2478;
x2478 = x2414;
x85.update(x2478, x2476, x2477);


// cv15_W <~~ V_cv15_W
float x2479, x2480;
x2479 = 1;
x2480 = decay_1;
JCudaTensor x2481;
x2481 = x2437;
x104.update(x2481, x2479, x2480);


// val X11387 = X11385 * d_ReLU()(X100)/d_X99
JCudaTensor x2482;
JCudaTensor x2483, x2484;
x2483 = x2418;
x2484 = x90;
x2482 = x92.backward(x2483,x2484);

// Dealloc(X100)
JCudaTensor x2485;
x2485 = x90;
x2485.free();

// val X5825 = X5823 * d_ReLU()(X96)/d_X95
JCudaTensor x2486;
JCudaTensor x2487, x2488;
x2487 = x2406;
x2488 = x87;
x2486 = x89.backward(x2487,x2488);

// Dealloc(X96)
JCudaTensor x2489;
x2489 = x87;
x2489.free();

// V_cv14_W <~~ X11387 * d_Convolv(1,0)(X92)/d_cv14_W
float x2491, x2492;
x2491 = lrn_rate_1;
x2492 = momentum;
JCudaTensor x2493, x2494;
x2493 = x2482;
x2494 = x53;
x79.backward_filter(x2493,x2494, x2490, x2491, x2492);

// V_cv12_B <~~ X5825 * d_Convolv(1,0)()/d_cv12_B
float x2496, x2497;
x2496 = lrn_rate_2;
x2497 = momentum;
JCudaTensor x2498;
x2498 = x2486;
x69.backward_bias(x2498, x2495, x2496, x2497);

// val X19773 = (X19749 + X5825 * d_Convolv(1,0)(cv12_W)/d_X92)
JCudaTensor x2499;
JCudaTensor x2500;
x2500 = x2430;
JCudaTensor x2501, x2502;
x2501 = x2486;
x2502 = x67;
x2499 = x69.backward_data(x2501,x2502, x2500);

// V_cv12_W <~~ X5825 * d_Convolv(1,0)(X92)/d_cv12_W
float x2504, x2505;
x2504 = lrn_rate_1;
x2505 = momentum;
JCudaTensor x2506, x2507;
x2506 = x2486;
x2507 = x53;
x69.backward_filter(x2506,x2507, x2503, x2504, x2505);

// Dealloc(X5825)
JCudaTensor x2508;
x2508 = x2486;
x2508.free();

// V_cv14_B <~~ X11387 * d_Convolv(1,0)()/d_cv14_B
float x2510, x2511;
x2510 = lrn_rate_2;
x2511 = momentum;
JCudaTensor x2512;
x2512 = x2482;
x79.backward_bias(x2512, x2509, x2510, x2511);

// cv12_B <~~ V_cv12_B
float x2513, x2514;
x2513 = 1;
x2514 = 1;
JCudaTensor x2515;
x2515 = x2495;
x68.update(x2515, x2513, x2514);


// cv12_W <~~ V_cv12_W
float x2516, x2517;
x2516 = 1;
x2517 = decay_1;
JCudaTensor x2518;
x2518 = x2503;
x67.update(x2518, x2516, x2517);


// cv14_B <~~ V_cv14_B
float x2519, x2520;
x2519 = 1;
x2520 = 1;
JCudaTensor x2521;
x2521 = x2509;
x78.update(x2521, x2519, x2520);


// val X19797 = (X19773 + X11387 * d_Convolv(1,0)(cv14_W)/d_X92)
JCudaTensor x2522;
JCudaTensor x2523;
x2523 = x2499;
JCudaTensor x2524, x2525;
x2524 = x2482;
x2525 = x77;
x2522 = x79.backward_data(x2524,x2525, x2523);

// Dealloc(X11387)
JCudaTensor x2526;
x2526 = x2482;
x2526.free();

// cv14_W <~~ V_cv14_W
float x2527, x2528;
x2527 = 1;
x2528 = decay_1;
JCudaTensor x2529;
x2529 = x2490;
x77.update(x2529, x2527, x2528);


// val X19819 = (X19797 + X19816 * d_Pooling(3,1,1,true)(X103,X92)/d_X92)
JCudaTensor x2530;
JCudaTensor x2531;
x2531 = x2522;
JCudaTensor x2532, x2533, x2534;
x2532 = x2454;
x2533 = x70;
x2534 = x53;
x2530 = x72.backward(x2532,x2533,x2534, x2531);

// Dealloc(X19816)
JCudaTensor x2535;
x2535 = x2454;
x2535.free();

// Dealloc(X103)
JCudaTensor x2536;
x2536 = x70;
x2536.free();

// val X19821 = X19819 * d_Pooling(3,2,1,true)(X92,X91)/d_X91
JCudaTensor x2537;
JCudaTensor x2538, x2539, x2540;
x2538 = x2530;
x2539 = x53;
x2540 = x50;
x2537 = x55.backward(x2538,x2539,x2540);

// Dealloc(X19819)
JCudaTensor x2541;
x2541 = x2530;
x2541.free();

// Dealloc(X92)
JCudaTensor x2542;
x2542 = x53;
x2542.free();

// val X19823 = X19821 * d_LRN(5,1.0E-4,0.75)(X91,X90)/d_X90
JCudaTensor x2543;
JCudaTensor x2544, x2545, x2546;
x2544 = x2537;
x2545 = x50;
x2546 = x47;
x2543 = x52.backward(x2544,x2545,x2546);

// Dealloc(X91)
JCudaTensor x2547;
x2547 = x50;
x2547.free();

// val X19825 = X19823 * d_ReLU()(X90)/d_X89
JCudaTensor x2548;
JCudaTensor x2549, x2550;
x2549 = x2543;
x2550 = x47;
x2548 = x49.backward(x2549,x2550);

// Dealloc(X90)
JCudaTensor x2551;
x2551 = x47;
x2551.free();

// V_cv3_B <~~ X19825 * d_Convolv(1,1)()/d_cv3_B
float x2553, x2554;
x2553 = lrn_rate_2;
x2554 = momentum;
JCudaTensor x2555;
x2555 = x2548;
x46.backward_bias(x2555, x2552, x2553, x2554);

// val X19826 = X19825 * d_Convolv(1,1)(cv3_W)/d_X88
JCudaTensor x2556;
JCudaTensor x2557, x2558;
x2557 = x2548;
x2558 = x44;
x2556 = x46.backward_data(x2557,x2558);

// V_cv3_W <~~ X19825 * d_Convolv(1,1)(X88)/d_cv3_W
float x2560, x2561;
x2560 = lrn_rate_1;
x2561 = momentum;
JCudaTensor x2562, x2563;
x2562 = x2548;
x2563 = x37;
x46.backward_filter(x2562,x2563, x2559, x2560, x2561);

// Dealloc(X19825)
JCudaTensor x2564;
x2564 = x2548;
x2564.free();

// cv3_B <~~ V_cv3_B
float x2565, x2566;
x2565 = 1;
x2566 = 1;
JCudaTensor x2567;
x2567 = x2552;
x45.update(x2567, x2565, x2566);


// cv3_W <~~ V_cv3_W
float x2568, x2569;
x2568 = 1;
x2569 = decay_1;
JCudaTensor x2570;
x2570 = x2559;
x44.update(x2570, x2568, x2569);


// val X19828 = X19826 * d_ReLU()(X88)/d_X87
JCudaTensor x2571;
JCudaTensor x2572, x2573;
x2572 = x2556;
x2573 = x37;
x2571 = x39.backward(x2572,x2573);

// Dealloc(X88)
JCudaTensor x2574;
x2574 = x37;
x2574.free();

// V_cv2_B <~~ X19828 * d_Convolv(1,0)()/d_cv2_B
float x2576, x2577;
x2576 = lrn_rate_2;
x2577 = momentum;
JCudaTensor x2578;
x2578 = x2571;
x36.backward_bias(x2578, x2575, x2576, x2577);

// val X19829 = X19828 * d_Convolv(1,0)(cv2_W)/d_X86
JCudaTensor x2579;
JCudaTensor x2580, x2581;
x2580 = x2571;
x2581 = x34;
x2579 = x36.backward_data(x2580,x2581);

// V_cv2_W <~~ X19828 * d_Convolv(1,0)(X86)/d_cv2_W
float x2583, x2584;
x2583 = lrn_rate_1;
x2584 = momentum;
JCudaTensor x2585, x2586;
x2585 = x2571;
x2586 = x27;
x36.backward_filter(x2585,x2586, x2582, x2583, x2584);

// Dealloc(X19828)
JCudaTensor x2587;
x2587 = x2571;
x2587.free();

// cv2_B <~~ V_cv2_B
float x2588, x2589;
x2588 = 1;
x2589 = 1;
JCudaTensor x2590;
x2590 = x2575;
x35.update(x2590, x2588, x2589);


// cv2_W <~~ V_cv2_W
float x2591, x2592;
x2591 = 1;
x2592 = decay_1;
JCudaTensor x2593;
x2593 = x2582;
x34.update(x2593, x2591, x2592);


// val X19831 = X19829 * d_LRN(5,1.0E-4,0.75)(X86,X85)/d_X85
JCudaTensor x2594;
JCudaTensor x2595, x2596, x2597;
x2595 = x2579;
x2596 = x27;
x2597 = x24;
x2594 = x29.backward(x2595,x2596,x2597);

// Dealloc(X86)
JCudaTensor x2598;
x2598 = x27;
x2598.free();

// val X19833 = X19831 * d_Pooling(3,2,1,true)(X85,X84)/d_X84
JCudaTensor x2599;
JCudaTensor x2600, x2601, x2602;
x2600 = x2594;
x2601 = x24;
x2602 = x21;
x2599 = x26.backward(x2600,x2601,x2602);

// Dealloc(X19831)
JCudaTensor x2603;
x2603 = x2594;
x2603.free();

// Dealloc(X85)
JCudaTensor x2604;
x2604 = x24;
x2604.free();

// val X19835 = X19833 * d_ReLU()(X84)/d_X83
JCudaTensor x2605;
JCudaTensor x2606, x2607;
x2606 = x2599;
x2607 = x21;
x2605 = x23.backward(x2606,x2607);

// Dealloc(X84)
JCudaTensor x2608;
x2608 = x21;
x2608.free();

// V_cv1_W <~~ X19835 * d_Convolv(2,3)(X82)/d_cv1_W
float x2610, x2611;
x2610 = lrn_rate_1;
x2611 = momentum;
JCudaTensor x2612, x2613;
x2612 = x2605;
x2613 = x7;
x17.backward_filter(x2612,x2613, x2609, x2610, x2611);

// Dealloc(X82)
JCudaTensor x2614;
x2614 = x7;
x2614.free();

// V_cv1_B <~~ X19835 * d_Convolv(2,3)()/d_cv1_B
float x2616, x2617;
x2616 = lrn_rate_2;
x2617 = momentum;
JCudaTensor x2618;
x2618 = x2605;
x17.backward_bias(x2618, x2615, x2616, x2617);

// Dealloc(X19835)
JCudaTensor x2619;
x2619 = x2605;
x2619.free();

// cv1_W <~~ V_cv1_W
float x2620, x2621;
x2620 = 1;
x2621 = decay_1;
JCudaTensor x2622;
x2622 = x2609;
x15.update(x2622, x2620, x2621);


// cv1_B <~~ V_cv1_B
float x2623, x2624;
x2623 = 1;
x2624 = 1;
JCudaTensor x2625;
x2625 = x2615;
x16.update(x2625, x2623, x2624);


}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X98544 = Cuda(X)
JCudaTensor x2626;
JTensorFloat x2627;
x2627 = x3;
x2626 = x2627.asJCudaTensor();

// val X98545 = Convolv(2,3)(X98544,cv1_W,cv1_B)
JCudaTensor x2628;
JCudaTensor x2629, x2630, x2631;
x2629 = x2626;
x2630 = x15;
x2631 = x16;
x2628 = x17.forward(x2629,x2630,x2631);

// Dealloc(X98544)
JCudaTensor x2632;
x2632 = x2626;
x2632.free();

// val X98546 = ReLU()(X98545)
JCudaTensor x2633;
JCudaTensor x2634;
x2634 = x2628;
x2633 = x23.forward(x2634);

// val X98547 = Pooling(3,2,1,true)(X98546)
JCudaTensor x2635;
JCudaTensor x2636;
x2636 = x2633;
x2635 = x26.forward(x2636);

// Dealloc(X98546)
JCudaTensor x2637;
x2637 = x2633;
x2637.free();

// val X98548 = LRN(5,1.0E-4,0.75)(X98547)
JCudaTensor x2638;
JCudaTensor x2639;
x2639 = x2635;
x2638 = x29.forward(x2639);

// Dealloc(X98547)
JCudaTensor x2640;
x2640 = x2635;
x2640.free();

// val X98549 = Convolv(1,0)(X98548,cv2_W,cv2_B)
JCudaTensor x2641;
JCudaTensor x2642, x2643, x2644;
x2642 = x2638;
x2643 = x34;
x2644 = x35;
x2641 = x36.forward(x2642,x2643,x2644);

// Dealloc(X98548)
JCudaTensor x2645;
x2645 = x2638;
x2645.free();

// val X98550 = ReLU()(X98549)
JCudaTensor x2646;
JCudaTensor x2647;
x2647 = x2641;
x2646 = x39.forward(x2647);

// val X98551 = Convolv(1,1)(X98550,cv3_W,cv3_B)
JCudaTensor x2648;
JCudaTensor x2649, x2650, x2651;
x2649 = x2646;
x2650 = x44;
x2651 = x45;
x2648 = x46.forward(x2649,x2650,x2651);

// Dealloc(X98550)
JCudaTensor x2652;
x2652 = x2646;
x2652.free();

// val X98552 = ReLU()(X98551)
JCudaTensor x2653;
JCudaTensor x2654;
x2654 = x2648;
x2653 = x49.forward(x2654);

// val X98553 = LRN(5,1.0E-4,0.75)(X98552)
JCudaTensor x2655;
JCudaTensor x2656;
x2656 = x2653;
x2655 = x52.forward(x2656);

// Dealloc(X98552)
JCudaTensor x2657;
x2657 = x2653;
x2657.free();

// val X98554 = Pooling(3,2,1,true)(X98553)
JCudaTensor x2658;
JCudaTensor x2659;
x2659 = x2655;
x2658 = x55.forward(x2659);

// Dealloc(X98553)
JCudaTensor x2660;
x2660 = x2655;
x2660.free();

// val X98557 = Convolv(1,0)(X98554,cv12_W,cv12_B)
JCudaTensor x2661;
JCudaTensor x2662, x2663, x2664;
x2662 = x2658;
x2663 = x67;
x2664 = x68;
x2661 = x69.forward(x2662,x2663,x2664);

// val X98561 = Convolv(1,0)(X98554,cv14_W,cv14_B)
JCudaTensor x2665;
JCudaTensor x2666, x2667, x2668;
x2666 = x2658;
x2667 = x77;
x2668 = x78;
x2665 = x79.forward(x2666,x2667,x2668);

// val X98555 = Convolv(1,0)(X98554,cv11_W,cv11_B)
JCudaTensor x2669;
JCudaTensor x2670, x2671, x2672;
x2670 = x2658;
x2671 = x60;
x2672 = x61;
x2669 = x62.forward(x2670,x2671,x2672);

// val X98565 = Pooling(3,1,1,true)(X98554)
JCudaTensor x2673;
JCudaTensor x2674;
x2674 = x2658;
x2673 = x72.forward(x2674);

// Dealloc(X98554)
JCudaTensor x2675;
x2675 = x2658;
x2675.free();

// val X98558 = ReLU()(X98557)
JCudaTensor x2676;
JCudaTensor x2677;
x2677 = x2661;
x2676 = x89.forward(x2677);

// val X98562 = ReLU()(X98561)
JCudaTensor x2678;
JCudaTensor x2679;
x2679 = x2665;
x2678 = x92.forward(x2679);

// val X98566 = Convolv(1,0)(X98565,cv16_W,cv16_B)
JCudaTensor x2680;
JCudaTensor x2681, x2682, x2683;
x2681 = x2673;
x2682 = x84;
x2683 = x85;
x2680 = x86.forward(x2681,x2682,x2683);

// Dealloc(X98565)
JCudaTensor x2684;
x2684 = x2673;
x2684.free();

// val X98559 = Convolv(1,1)(X98558,cv13_W,cv13_B)
JCudaTensor x2685;
JCudaTensor x2686, x2687, x2688;
x2686 = x2676;
x2687 = x97;
x2688 = x98;
x2685 = x99.forward(x2686,x2687,x2688);

// Dealloc(X98558)
JCudaTensor x2689;
x2689 = x2676;
x2689.free();

// val X98563 = Convolv(1,2)(X98562,cv15_W,cv15_B)
JCudaTensor x2690;
JCudaTensor x2691, x2692, x2693;
x2691 = x2678;
x2692 = x104;
x2693 = x105;
x2690 = x106.forward(x2691,x2692,x2693);

// Dealloc(X98562)
JCudaTensor x2694;
x2694 = x2678;
x2694.free();

// val X98556 = ReLU()(X98555)
JCudaTensor x2695;
JCudaTensor x2696;
x2696 = x2669;
x2695 = x109.forward(x2696);

// val X98560 = ReLU()(X98559)
JCudaTensor x2697;
JCudaTensor x2698;
x2698 = x2685;
x2697 = x112.forward(x2698);

// val X98564 = ReLU()(X98563)
JCudaTensor x2699;
JCudaTensor x2700;
x2700 = x2690;
x2699 = x115.forward(x2700);

// val X98567 = ReLU()(X98566)
JCudaTensor x2701;
JCudaTensor x2702;
x2702 = x2680;
x2701 = x118.forward(x2702);

// val X98568 = Concat(X98556,X98560,X98564,X98567)
JCudaTensor x2703;
JCudaTensor x2704, x2705, x2706, x2707;
x2704 = x2695;
x2705 = x2697;
x2706 = x2699;
x2707 = x2701;
x2703 = x120.forward(x2704,x2705,x2706,x2707);

// Dealloc(X98567)
JCudaTensor x2708;
x2708 = x2701;
x2708.free();

// Dealloc(X98564)
JCudaTensor x2709;
x2709 = x2699;
x2709.free();

// Dealloc(X98560)
JCudaTensor x2710;
x2710 = x2697;
x2710.free();

// Dealloc(X98556)
JCudaTensor x2711;
x2711 = x2695;
x2711.free();

// val X98579 = Pooling(3,1,1,true)(X98568)
JCudaTensor x2712;
JCudaTensor x2713;
x2713 = x2703;
x2712 = x127.forward(x2713);

// val X98569 = Convolv(1,0)(X98568,cv21_W,cv21_B)
JCudaTensor x2714;
JCudaTensor x2715, x2716, x2717;
x2715 = x2703;
x2716 = x139;
x2717 = x140;
x2714 = x141.forward(x2715,x2716,x2717);

// val X98575 = Convolv(1,0)(X98568,cv24_W,cv24_B)
JCudaTensor x2718;
JCudaTensor x2719, x2720, x2721;
x2719 = x2703;
x2720 = x132;
x2721 = x133;
x2718 = x134.forward(x2719,x2720,x2721);

// val X98571 = Convolv(1,0)(X98568,cv22_W,cv22_B)
JCudaTensor x2722;
JCudaTensor x2723, x2724, x2725;
x2723 = x2703;
x2724 = x146;
x2725 = x147;
x2722 = x148.forward(x2723,x2724,x2725);

// Dealloc(X98568)
JCudaTensor x2726;
x2726 = x2703;
x2726.free();

// val X98580 = Convolv(1,0)(X98579,cv26_W,cv26_B)
JCudaTensor x2727;
JCudaTensor x2728, x2729, x2730;
x2728 = x2712;
x2729 = x156;
x2730 = x157;
x2727 = x158.forward(x2728,x2729,x2730);

// Dealloc(X98579)
JCudaTensor x2731;
x2731 = x2712;
x2731.free();

// val X98572 = ReLU()(X98571)
JCudaTensor x2732;
JCudaTensor x2733;
x2733 = x2722;
x2732 = x151.forward(x2733);

// val X98576 = ReLU()(X98575)
JCudaTensor x2734;
JCudaTensor x2735;
x2735 = x2718;
x2734 = x161.forward(x2735);

// val X98573 = Convolv(1,1)(X98572,cv23_W,cv23_B)
JCudaTensor x2736;
JCudaTensor x2737, x2738, x2739;
x2737 = x2732;
x2738 = x166;
x2739 = x167;
x2736 = x168.forward(x2737,x2738,x2739);

// Dealloc(X98572)
JCudaTensor x2740;
x2740 = x2732;
x2740.free();

// val X98577 = Convolv(1,2)(X98576,cv25_W,cv25_B)
JCudaTensor x2741;
JCudaTensor x2742, x2743, x2744;
x2742 = x2734;
x2743 = x173;
x2744 = x174;
x2741 = x175.forward(x2742,x2743,x2744);

// Dealloc(X98576)
JCudaTensor x2745;
x2745 = x2734;
x2745.free();

// val X98570 = ReLU()(X98569)
JCudaTensor x2746;
JCudaTensor x2747;
x2747 = x2714;
x2746 = x178.forward(x2747);

// val X98574 = ReLU()(X98573)
JCudaTensor x2748;
JCudaTensor x2749;
x2749 = x2736;
x2748 = x181.forward(x2749);

// val X98578 = ReLU()(X98577)
JCudaTensor x2750;
JCudaTensor x2751;
x2751 = x2741;
x2750 = x184.forward(x2751);

// val X98581 = ReLU()(X98580)
JCudaTensor x2752;
JCudaTensor x2753;
x2753 = x2727;
x2752 = x187.forward(x2753);

// val X98582 = Concat(X98570,X98574,X98578,X98581)
JCudaTensor x2754;
JCudaTensor x2755, x2756, x2757, x2758;
x2755 = x2746;
x2756 = x2748;
x2757 = x2750;
x2758 = x2752;
x2754 = x120.forward(x2755,x2756,x2757,x2758);

// Dealloc(X98581)
JCudaTensor x2759;
x2759 = x2752;
x2759.free();

// Dealloc(X98578)
JCudaTensor x2760;
x2760 = x2750;
x2760.free();

// Dealloc(X98574)
JCudaTensor x2761;
x2761 = x2748;
x2761.free();

// Dealloc(X98570)
JCudaTensor x2762;
x2762 = x2746;
x2762.free();

// val X98583 = Pooling(3,2,1,true)(X98582)
JCudaTensor x2763;
JCudaTensor x2764;
x2764 = x2754;
x2763 = x195.forward(x2764);

// Dealloc(X98582)
JCudaTensor x2765;
x2765 = x2754;
x2765.free();

// val X98590 = Convolv(1,0)(X98583,cv34_W,cv34_B)
JCudaTensor x2766;
JCudaTensor x2767, x2768, x2769;
x2767 = x2763;
x2768 = x217;
x2769 = x218;
x2766 = x219.forward(x2767,x2768,x2769);

// val X98594 = Pooling(3,1,1,true)(X98583)
JCudaTensor x2770;
JCudaTensor x2771;
x2771 = x2763;
x2770 = x212.forward(x2771);

// val X98586 = Convolv(1,0)(X98583,cv32_W,cv32_B)
JCudaTensor x2772;
JCudaTensor x2773, x2774, x2775;
x2773 = x2763;
x2774 = x207;
x2775 = x208;
x2772 = x209.forward(x2773,x2774,x2775);

// val X98584 = Convolv(1,0)(X98583,cv31_W,cv31_B)
JCudaTensor x2776;
JCudaTensor x2777, x2778, x2779;
x2777 = x2763;
x2778 = x200;
x2779 = x201;
x2776 = x202.forward(x2777,x2778,x2779);

// Dealloc(X98583)
JCudaTensor x2780;
x2780 = x2763;
x2780.free();

// val X98591 = ReLU()(X98590)
JCudaTensor x2781;
JCudaTensor x2782;
x2782 = x2766;
x2781 = x222.forward(x2782);

// val X98595 = Convolv(1,0)(X98594,cv36_W,cv36_B)
JCudaTensor x2783;
JCudaTensor x2784, x2785, x2786;
x2784 = x2770;
x2785 = x230;
x2786 = x231;
x2783 = x232.forward(x2784,x2785,x2786);

// Dealloc(X98594)
JCudaTensor x2787;
x2787 = x2770;
x2787.free();

// val X98587 = ReLU()(X98586)
JCudaTensor x2788;
JCudaTensor x2789;
x2789 = x2772;
x2788 = x225.forward(x2789);

// val X98588 = Convolv(1,1)(X98587,cv33_W,cv33_B)
JCudaTensor x2790;
JCudaTensor x2791, x2792, x2793;
x2791 = x2788;
x2792 = x244;
x2793 = x245;
x2790 = x246.forward(x2791,x2792,x2793);

// Dealloc(X98587)
JCudaTensor x2794;
x2794 = x2788;
x2794.free();

// val X98592 = Convolv(1,2)(X98591,cv35_W,cv35_B)
JCudaTensor x2795;
JCudaTensor x2796, x2797, x2798;
x2796 = x2781;
x2797 = x237;
x2798 = x238;
x2795 = x239.forward(x2796,x2797,x2798);

// Dealloc(X98591)
JCudaTensor x2799;
x2799 = x2781;
x2799.free();

// val X98585 = ReLU()(X98584)
JCudaTensor x2800;
JCudaTensor x2801;
x2801 = x2776;
x2800 = x249.forward(x2801);

// val X98589 = ReLU()(X98588)
JCudaTensor x2802;
JCudaTensor x2803;
x2803 = x2790;
x2802 = x252.forward(x2803);

// val X98593 = ReLU()(X98592)
JCudaTensor x2804;
JCudaTensor x2805;
x2805 = x2795;
x2804 = x255.forward(x2805);

// val X98596 = ReLU()(X98595)
JCudaTensor x2806;
JCudaTensor x2807;
x2807 = x2783;
x2806 = x258.forward(x2807);

// val X98597 = Concat(X98585,X98589,X98593,X98596)
JCudaTensor x2808;
JCudaTensor x2809, x2810, x2811, x2812;
x2809 = x2800;
x2810 = x2802;
x2811 = x2804;
x2812 = x2806;
x2808 = x260.forward(x2809,x2810,x2811,x2812);

// Dealloc(X98596)
JCudaTensor x2813;
x2813 = x2806;
x2813.free();

// Dealloc(X98593)
JCudaTensor x2814;
x2814 = x2804;
x2814.free();

// Dealloc(X98589)
JCudaTensor x2815;
x2815 = x2802;
x2815.free();

// Dealloc(X98585)
JCudaTensor x2816;
x2816 = x2800;
x2816.free();

// val X98598 = Convolv(1,0)(X98597,cv41_W,cv41_B)
JCudaTensor x2817;
JCudaTensor x2818, x2819, x2820;
x2818 = x2808;
x2819 = x289;
x2820 = x290;
x2817 = x291.forward(x2818,x2819,x2820);

// val X98608 = Pooling(3,1,1,true)(X98597)
JCudaTensor x2821;
JCudaTensor x2822;
x2822 = x2808;
x2821 = x277.forward(x2822);

// val X98604 = Convolv(1,0)(X98597,cv44_W,cv44_B)
JCudaTensor x2823;
JCudaTensor x2824, x2825, x2826;
x2824 = x2808;
x2825 = x272;
x2826 = x273;
x2823 = x274.forward(x2824,x2825,x2826);

// val X98600 = Convolv(1,0)(X98597,cv42_W,cv42_B)
JCudaTensor x2827;
JCudaTensor x2828, x2829, x2830;
x2828 = x2808;
x2829 = x282;
x2830 = x283;
x2827 = x284.forward(x2828,x2829,x2830);

// Dealloc(X98597)
JCudaTensor x2831;
x2831 = x2808;
x2831.free();

// val X98601 = ReLU()(X98600)
JCudaTensor x2832;
JCudaTensor x2833;
x2833 = x2827;
x2832 = x304.forward(x2833);

// val X98605 = ReLU()(X98604)
JCudaTensor x2834;
JCudaTensor x2835;
x2835 = x2823;
x2834 = x294.forward(x2835);

// val X98609 = Convolv(1,0)(X98608,cv46_W,cv46_B)
JCudaTensor x2836;
JCudaTensor x2837, x2838, x2839;
x2837 = x2821;
x2838 = x309;
x2839 = x310;
x2836 = x311.forward(x2837,x2838,x2839);

// Dealloc(X98608)
JCudaTensor x2840;
x2840 = x2821;
x2840.free();

// val X98606 = Convolv(1,2)(X98605,cv45_W,cv45_B)
JCudaTensor x2841;
JCudaTensor x2842, x2843, x2844;
x2842 = x2834;
x2843 = x319;
x2844 = x320;
x2841 = x321.forward(x2842,x2843,x2844);

// Dealloc(X98605)
JCudaTensor x2845;
x2845 = x2834;
x2845.free();

// val X98602 = Convolv(1,1)(X98601,cv43_W,cv43_B)
JCudaTensor x2846;
JCudaTensor x2847, x2848, x2849;
x2847 = x2832;
x2848 = x326;
x2849 = x327;
x2846 = x328.forward(x2847,x2848,x2849);

// Dealloc(X98601)
JCudaTensor x2850;
x2850 = x2832;
x2850.free();

// val X98599 = ReLU()(X98598)
JCudaTensor x2851;
JCudaTensor x2852;
x2852 = x2817;
x2851 = x334.forward(x2852);

// val X98603 = ReLU()(X98602)
JCudaTensor x2853;
JCudaTensor x2854;
x2854 = x2846;
x2853 = x337.forward(x2854);

// val X98607 = ReLU()(X98606)
JCudaTensor x2855;
JCudaTensor x2856;
x2856 = x2841;
x2855 = x331.forward(x2856);

// val X98610 = ReLU()(X98609)
JCudaTensor x2857;
JCudaTensor x2858;
x2858 = x2836;
x2857 = x340.forward(x2858);

// val X98611 = Concat(X98599,X98603,X98607,X98610)
JCudaTensor x2859;
JCudaTensor x2860, x2861, x2862, x2863;
x2860 = x2851;
x2861 = x2853;
x2862 = x2855;
x2863 = x2857;
x2859 = x260.forward(x2860,x2861,x2862,x2863);

// Dealloc(X98610)
JCudaTensor x2864;
x2864 = x2857;
x2864.free();

// Dealloc(X98607)
JCudaTensor x2865;
x2865 = x2855;
x2865.free();

// Dealloc(X98603)
JCudaTensor x2866;
x2866 = x2853;
x2866.free();

// Dealloc(X98599)
JCudaTensor x2867;
x2867 = x2851;
x2867.free();

// val X98612 = Convolv(1,0)(X98611,cv51_W,cv51_B)
JCudaTensor x2868;
JCudaTensor x2869, x2870, x2871;
x2869 = x2859;
x2870 = x374;
x2871 = x375;
x2868 = x376.forward(x2869,x2870,x2871);

// val X98622 = Pooling(3,1,1,true)(X98611)
JCudaTensor x2872;
JCudaTensor x2873;
x2873 = x2859;
x2872 = x369.forward(x2873);

// val X98614 = Convolv(1,0)(X98611,cv52_W,cv52_B)
JCudaTensor x2874;
JCudaTensor x2875, x2876, x2877;
x2875 = x2859;
x2876 = x361;
x2877 = x362;
x2874 = x363.forward(x2875,x2876,x2877);

// val X98618 = Convolv(1,0)(X98611,cv54_W,cv54_B)
JCudaTensor x2878;
JCudaTensor x2879, x2880, x2881;
x2879 = x2859;
x2880 = x381;
x2881 = x382;
x2878 = x383.forward(x2879,x2880,x2881);

// Dealloc(X98611)
JCudaTensor x2882;
x2882 = x2859;
x2882.free();

// val X98623 = Convolv(1,0)(X98622,cv56_W,cv56_B)
JCudaTensor x2883;
JCudaTensor x2884, x2885, x2886;
x2884 = x2872;
x2885 = x391;
x2886 = x392;
x2883 = x393.forward(x2884,x2885,x2886);

// Dealloc(X98622)
JCudaTensor x2887;
x2887 = x2872;
x2887.free();

// val X98619 = ReLU()(X98618)
JCudaTensor x2888;
JCudaTensor x2889;
x2889 = x2878;
x2888 = x386.forward(x2889);

// val X98615 = ReLU()(X98614)
JCudaTensor x2890;
JCudaTensor x2891;
x2891 = x2874;
x2890 = x399.forward(x2891);

// val X98616 = Convolv(1,1)(X98615,cv53_W,cv53_B)
JCudaTensor x2892;
JCudaTensor x2893, x2894, x2895;
x2893 = x2890;
x2894 = x404;
x2895 = x405;
x2892 = x406.forward(x2893,x2894,x2895);

// Dealloc(X98615)
JCudaTensor x2896;
x2896 = x2890;
x2896.free();

// val X98620 = Convolv(1,2)(X98619,cv55_W,cv55_B)
JCudaTensor x2897;
JCudaTensor x2898, x2899, x2900;
x2898 = x2888;
x2899 = x411;
x2900 = x412;
x2897 = x413.forward(x2898,x2899,x2900);

// Dealloc(X98619)
JCudaTensor x2901;
x2901 = x2888;
x2901.free();

// val X98613 = ReLU()(X98612)
JCudaTensor x2902;
JCudaTensor x2903;
x2903 = x2868;
x2902 = x431.forward(x2903);

// val X98617 = ReLU()(X98616)
JCudaTensor x2904;
JCudaTensor x2905;
x2905 = x2892;
x2904 = x422.forward(x2905);

// val X98621 = ReLU()(X98620)
JCudaTensor x2906;
JCudaTensor x2907;
x2907 = x2897;
x2906 = x428.forward(x2907);

// val X98624 = ReLU()(X98623)
JCudaTensor x2908;
JCudaTensor x2909;
x2909 = x2883;
x2908 = x425.forward(x2909);

// val X98625 = Concat(X98613,X98617,X98621,X98624)
JCudaTensor x2910;
JCudaTensor x2911, x2912, x2913, x2914;
x2911 = x2902;
x2912 = x2904;
x2913 = x2906;
x2914 = x2908;
x2910 = x260.forward(x2911,x2912,x2913,x2914);

// Dealloc(X98624)
JCudaTensor x2915;
x2915 = x2908;
x2915.free();

// Dealloc(X98621)
JCudaTensor x2916;
x2916 = x2906;
x2916.free();

// Dealloc(X98617)
JCudaTensor x2917;
x2917 = x2904;
x2917.free();

// Dealloc(X98613)
JCudaTensor x2918;
x2918 = x2902;
x2918.free();

// val X98626 = Convolv(1,0)(X98625,cv61_W,cv61_B)
JCudaTensor x2919;
JCudaTensor x2920, x2921, x2922;
x2920 = x2910;
x2921 = x472;
x2922 = x473;
x2919 = x474.forward(x2920,x2921,x2922);

// val X98636 = Pooling(3,1,1,true)(X98625)
JCudaTensor x2923;
JCudaTensor x2924;
x2924 = x2910;
x2923 = x477.forward(x2924);

// val X98632 = Convolv(1,0)(X98625,cv64_W,cv64_B)
JCudaTensor x2925;
JCudaTensor x2926, x2927, x2928;
x2926 = x2910;
x2927 = x453;
x2928 = x454;
x2925 = x455.forward(x2926,x2927,x2928);

// val X98628 = Convolv(1,0)(X98625,cv62_W,cv62_B)
JCudaTensor x2929;
JCudaTensor x2930, x2931, x2932;
x2930 = x2910;
x2931 = x465;
x2932 = x466;
x2929 = x467.forward(x2930,x2931,x2932);

// Dealloc(X98625)
JCudaTensor x2933;
x2933 = x2910;
x2933.free();

// val X98633 = ReLU()(X98632)
JCudaTensor x2934;
JCudaTensor x2935;
x2935 = x2925;
x2934 = x496.forward(x2935);

// val X98629 = ReLU()(X98628)
JCudaTensor x2936;
JCudaTensor x2937;
x2937 = x2929;
x2936 = x493.forward(x2937);

// val X98637 = Convolv(1,0)(X98636,cv66_W,cv66_B)
JCudaTensor x2938;
JCudaTensor x2939, x2940, x2941;
x2939 = x2923;
x2940 = x482;
x2941 = x483;
x2938 = x484.forward(x2939,x2940,x2941);

// Dealloc(X98636)
JCudaTensor x2942;
x2942 = x2923;
x2942.free();

// val X98634 = Convolv(1,2)(X98633,cv65_W,cv65_B)
JCudaTensor x2943;
JCudaTensor x2944, x2945, x2946;
x2944 = x2934;
x2945 = x503;
x2946 = x504;
x2943 = x505.forward(x2944,x2945,x2946);

// Dealloc(X98633)
JCudaTensor x2947;
x2947 = x2934;
x2947.free();

// val X98630 = Convolv(1,1)(X98629,cv63_W,cv63_B)
JCudaTensor x2948;
JCudaTensor x2949, x2950, x2951;
x2949 = x2936;
x2950 = x520;
x2951 = x521;
x2948 = x522.forward(x2949,x2950,x2951);

// Dealloc(X98629)
JCudaTensor x2952;
x2952 = x2936;
x2952.free();

// val X98627 = ReLU()(X98626)
JCudaTensor x2953;
JCudaTensor x2954;
x2954 = x2919;
x2953 = x533.forward(x2954);

// val X98631 = ReLU()(X98630)
JCudaTensor x2955;
JCudaTensor x2956;
x2956 = x2948;
x2955 = x536.forward(x2956);

// val X98635 = ReLU()(X98634)
JCudaTensor x2957;
JCudaTensor x2958;
x2958 = x2943;
x2957 = x539.forward(x2958);

// val X98638 = ReLU()(X98637)
JCudaTensor x2959;
JCudaTensor x2960;
x2960 = x2938;
x2959 = x528.forward(x2960);

// val X98639 = Concat(X98627,X98631,X98635,X98638)
JCudaTensor x2961;
JCudaTensor x2962, x2963, x2964, x2965;
x2962 = x2953;
x2963 = x2955;
x2964 = x2957;
x2965 = x2959;
x2961 = x260.forward(x2962,x2963,x2964,x2965);

// Dealloc(X98638)
JCudaTensor x2966;
x2966 = x2959;
x2966.free();

// Dealloc(X98635)
JCudaTensor x2967;
x2967 = x2957;
x2967.free();

// Dealloc(X98631)
JCudaTensor x2968;
x2968 = x2955;
x2968.free();

// Dealloc(X98627)
JCudaTensor x2969;
x2969 = x2953;
x2969.free();

// val X98640 = Convolv(1,0)(X98639,cv71_W,cv71_B)
JCudaTensor x2970;
JCudaTensor x2971, x2972, x2973;
x2971 = x2961;
x2972 = x593;
x2973 = x594;
x2970 = x595.forward(x2971,x2972,x2973);

// val X98646 = Convolv(1,0)(X98639,cv74_W,cv74_B)
JCudaTensor x2974;
JCudaTensor x2975, x2976, x2977;
x2975 = x2961;
x2976 = x607;
x2977 = x608;
x2974 = x609.forward(x2975,x2976,x2977);

// val X98642 = Convolv(1,0)(X98639,cv72_W,cv72_B)
JCudaTensor x2978;
JCudaTensor x2979, x2980, x2981;
x2979 = x2961;
x2980 = x600;
x2981 = x601;
x2978 = x602.forward(x2979,x2980,x2981);

// val X98650 = Pooling(3,1,1,true)(X98639)
JCudaTensor x2982;
JCudaTensor x2983;
x2983 = x2961;
x2982 = x581.forward(x2983);

// Dealloc(X98639)
JCudaTensor x2984;
x2984 = x2961;
x2984.free();

// val X98651 = Convolv(1,0)(X98650,cv76_W,cv76_B)
JCudaTensor x2985;
JCudaTensor x2986, x2987, x2988;
x2986 = x2982;
x2987 = x640;
x2988 = x641;
x2985 = x642.forward(x2986,x2987,x2988);

// Dealloc(X98650)
JCudaTensor x2989;
x2989 = x2982;
x2989.free();

// val X98647 = ReLU()(X98646)
JCudaTensor x2990;
JCudaTensor x2991;
x2991 = x2974;
x2990 = x635.forward(x2991);

// val X98643 = ReLU()(X98642)
JCudaTensor x2992;
JCudaTensor x2993;
x2993 = x2978;
x2992 = x619.forward(x2993);

// val X98648 = Convolv(1,2)(X98647,cv75_W,cv75_B)
JCudaTensor x2994;
JCudaTensor x2995, x2996, x2997;
x2995 = x2990;
x2996 = x676;
x2997 = x677;
x2994 = x678.forward(x2995,x2996,x2997);

// Dealloc(X98647)
JCudaTensor x2998;
x2998 = x2990;
x2998.free();

// val X98644 = Convolv(1,1)(X98643,cv73_W,cv73_B)
JCudaTensor x2999;
JCudaTensor x3000, x3001, x3002;
x3000 = x2992;
x3001 = x686;
x3002 = x687;
x2999 = x688.forward(x3000,x3001,x3002);

// Dealloc(X98643)
JCudaTensor x3003;
x3003 = x2992;
x3003.free();

// val X98641 = ReLU()(X98640)
JCudaTensor x3004;
JCudaTensor x3005;
x3005 = x2970;
x3004 = x723.forward(x3005);

// val X98645 = ReLU()(X98644)
JCudaTensor x3006;
JCudaTensor x3007;
x3007 = x2999;
x3006 = x700.forward(x3007);

// val X98649 = ReLU()(X98648)
JCudaTensor x3008;
JCudaTensor x3009;
x3009 = x2994;
x3008 = x697.forward(x3009);

// val X98652 = ReLU()(X98651)
JCudaTensor x3010;
JCudaTensor x3011;
x3011 = x2985;
x3010 = x703.forward(x3011);

// val X98653 = Concat(X98641,X98645,X98649,X98652)
JCudaTensor x3012;
JCudaTensor x3013, x3014, x3015, x3016;
x3013 = x3004;
x3014 = x3006;
x3015 = x3008;
x3016 = x3010;
x3012 = x260.forward(x3013,x3014,x3015,x3016);

// Dealloc(X98652)
JCudaTensor x3017;
x3017 = x3010;
x3017.free();

// Dealloc(X98649)
JCudaTensor x3018;
x3018 = x3008;
x3018.free();

// Dealloc(X98645)
JCudaTensor x3019;
x3019 = x3006;
x3019.free();

// Dealloc(X98641)
JCudaTensor x3020;
x3020 = x3004;
x3020.free();

// val X98654 = Pooling(3,2,1,true)(X98653)
JCudaTensor x3021;
JCudaTensor x3022;
x3022 = x3012;
x3021 = x741.forward(x3022);

// Dealloc(X98653)
JCudaTensor x3023;
x3023 = x3012;
x3023.free();

// val X98657 = Convolv(1,0)(X98654,cv82_W,cv82_B)
JCudaTensor x3024;
JCudaTensor x3025, x3026, x3027;
x3025 = x3021;
x3026 = x760;
x3027 = x761;
x3024 = x762.forward(x3025,x3026,x3027);

// val X98655 = Convolv(1,0)(X98654,cv81_W,cv81_B)
JCudaTensor x3028;
JCudaTensor x3029, x3030, x3031;
x3029 = x3021;
x3030 = x748;
x3031 = x749;
x3028 = x750.forward(x3029,x3030,x3031);

// val X98661 = Convolv(1,0)(X98654,cv84_W,cv84_B)
JCudaTensor x3032;
JCudaTensor x3033, x3034, x3035;
x3033 = x3021;
x3034 = x767;
x3035 = x768;
x3032 = x769.forward(x3033,x3034,x3035);

// val X98665 = Pooling(3,1,1,true)(X98654)
JCudaTensor x3036;
JCudaTensor x3037;
x3037 = x3021;
x3036 = x753.forward(x3037);

// Dealloc(X98654)
JCudaTensor x3038;
x3038 = x3021;
x3038.free();

// val X98666 = Convolv(1,0)(X98665,cv86_W,cv86_B)
JCudaTensor x3039;
JCudaTensor x3040, x3041, x3042;
x3040 = x3036;
x3041 = x774;
x3042 = x775;
x3039 = x776.forward(x3040,x3041,x3042);

// Dealloc(X98665)
JCudaTensor x3043;
x3043 = x3036;
x3043.free();

// val X98662 = ReLU()(X98661)
JCudaTensor x3044;
JCudaTensor x3045;
x3045 = x3032;
x3044 = x782.forward(x3045);

// val X98658 = ReLU()(X98657)
JCudaTensor x3046;
JCudaTensor x3047;
x3047 = x3024;
x3046 = x779.forward(x3047);

// val X98659 = Convolv(1,1)(X98658,cv83_W,cv83_B)
JCudaTensor x3048;
JCudaTensor x3049, x3050, x3051;
x3049 = x3046;
x3050 = x797;
x3051 = x798;
x3048 = x799.forward(x3049,x3050,x3051);

// Dealloc(X98658)
JCudaTensor x3052;
x3052 = x3046;
x3052.free();

// val X98663 = Convolv(1,2)(X98662,cv85_W,cv85_B)
JCudaTensor x3053;
JCudaTensor x3054, x3055, x3056;
x3054 = x3044;
x3055 = x804;
x3056 = x805;
x3053 = x806.forward(x3054,x3055,x3056);

// Dealloc(X98662)
JCudaTensor x3057;
x3057 = x3044;
x3057.free();

// val X98656 = ReLU()(X98655)
JCudaTensor x3058;
JCudaTensor x3059;
x3059 = x3028;
x3058 = x821.forward(x3059);

// val X98660 = ReLU()(X98659)
JCudaTensor x3060;
JCudaTensor x3061;
x3061 = x3048;
x3060 = x815.forward(x3061);

// val X98664 = ReLU()(X98663)
JCudaTensor x3062;
JCudaTensor x3063;
x3063 = x3053;
x3062 = x818.forward(x3063);

// val X98667 = ReLU()(X98666)
JCudaTensor x3064;
JCudaTensor x3065;
x3065 = x3039;
x3064 = x812.forward(x3065);

// val X98668 = Concat(X98656,X98660,X98664,X98667)
JCudaTensor x3066;
JCudaTensor x3067, x3068, x3069, x3070;
x3067 = x3058;
x3068 = x3060;
x3069 = x3062;
x3070 = x3064;
x3066 = x826.forward(x3067,x3068,x3069,x3070);

// Dealloc(X98667)
JCudaTensor x3071;
x3071 = x3064;
x3071.free();

// Dealloc(X98664)
JCudaTensor x3072;
x3072 = x3062;
x3072.free();

// Dealloc(X98660)
JCudaTensor x3073;
x3073 = x3060;
x3073.free();

// Dealloc(X98656)
JCudaTensor x3074;
x3074 = x3058;
x3074.free();

// val X98675 = Convolv(1,0)(X98668,cv94_W,cv94_B)
JCudaTensor x3075;
JCudaTensor x3076, x3077, x3078;
x3076 = x3066;
x3077 = x853;
x3078 = x854;
x3075 = x855.forward(x3076,x3077,x3078);

// val X98679 = Pooling(3,1,1,true)(X98668)
JCudaTensor x3079;
JCudaTensor x3080;
x3080 = x3066;
x3079 = x864.forward(x3080);

// val X98671 = Convolv(1,0)(X98668,cv92_W,cv92_B)
JCudaTensor x3081;
JCudaTensor x3082, x3083, x3084;
x3082 = x3066;
x3083 = x839;
x3084 = x840;
x3081 = x841.forward(x3082,x3083,x3084);

// val X98669 = Convolv(1,0)(X98668,cv91_W,cv91_B)
JCudaTensor x3085;
JCudaTensor x3086, x3087, x3088;
x3086 = x3066;
x3087 = x846;
x3088 = x847;
x3085 = x848.forward(x3086,x3087,x3088);

// Dealloc(X98668)
JCudaTensor x3089;
x3089 = x3066;
x3089.free();

// val X98680 = Convolv(1,0)(X98679,cv96_W,cv96_B)
JCudaTensor x3090;
JCudaTensor x3091, x3092, x3093;
x3091 = x3079;
x3092 = x876;
x3093 = x877;
x3090 = x878.forward(x3091,x3092,x3093);

// Dealloc(X98679)
JCudaTensor x3094;
x3094 = x3079;
x3094.free();

// val X98672 = ReLU()(X98671)
JCudaTensor x3095;
JCudaTensor x3096;
x3096 = x3081;
x3095 = x884.forward(x3096);

// val X98676 = ReLU()(X98675)
JCudaTensor x3097;
JCudaTensor x3098;
x3098 = x3075;
x3097 = x871.forward(x3098);

// val X98673 = Convolv(1,1)(X98672,cv93_W,cv93_B)
JCudaTensor x3099;
JCudaTensor x3100, x3101, x3102;
x3100 = x3095;
x3101 = x892;
x3102 = x893;
x3099 = x894.forward(x3100,x3101,x3102);

// Dealloc(X98672)
JCudaTensor x3103;
x3103 = x3095;
x3103.free();

// val X98677 = Convolv(1,2)(X98676,cv95_W,cv95_B)
JCudaTensor x3104;
JCudaTensor x3105, x3106, x3107;
x3105 = x3097;
x3106 = x904;
x3107 = x905;
x3104 = x906.forward(x3105,x3106,x3107);

// Dealloc(X98676)
JCudaTensor x3108;
x3108 = x3097;
x3108.free();

// val X98670 = ReLU()(X98669)
JCudaTensor x3109;
JCudaTensor x3110;
x3110 = x3085;
x3109 = x927.forward(x3110);

// val X98674 = ReLU()(X98673)
JCudaTensor x3111;
JCudaTensor x3112;
x3112 = x3099;
x3111 = x930.forward(x3112);

// val X98678 = ReLU()(X98677)
JCudaTensor x3113;
JCudaTensor x3114;
x3114 = x3104;
x3113 = x933.forward(x3114);

// val X98681 = ReLU()(X98680)
JCudaTensor x3115;
JCudaTensor x3116;
x3116 = x3090;
x3115 = x936.forward(x3116);

// val X98682 = Concat(X98670,X98674,X98678,X98681)
JCudaTensor x3117;
JCudaTensor x3118, x3119, x3120, x3121;
x3118 = x3109;
x3119 = x3111;
x3120 = x3113;
x3121 = x3115;
x3117 = x826.forward(x3118,x3119,x3120,x3121);

// Dealloc(X98681)
JCudaTensor x3122;
x3122 = x3115;
x3122.free();

// Dealloc(X98678)
JCudaTensor x3123;
x3123 = x3113;
x3123.free();

// Dealloc(X98674)
JCudaTensor x3124;
x3124 = x3111;
x3124.free();

// Dealloc(X98670)
JCudaTensor x3125;
x3125 = x3109;
x3125.free();

// val X98683 = Pooling(7,1,0,false)(X98682)
JCudaTensor x3126;
JCudaTensor x3127;
x3127 = x3117;
x3126 = x981.forward(x3127);

// Dealloc(X98682)
JCudaTensor x3128;
x3128 = x3117;
x3128.free();

// val X98684 = Dropout(0.4)(X98683)
JCudaTensor x3129;
JCudaTensor x3130;
x3130 = x3126;
x3129 = x1009.forward(x3130);

// Dealloc(X98683)
JCudaTensor x3131;
x3131 = x3126;
x3131.free();

// val X98685 = (X98684[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x3132;
JCudaMatrix x3133;
JCudaMatrix x3134;
JCudaTensor x3135;
JCudaTensor x3136;
x3136 = x3129;
x3135 = x3136.flatten(1, new int[]{256, 1, 1});
x3133 = x3135.asMatrix(1, true);
JCudaTensor x3137;
x3137 = x1027;
x3134 = x3137.asMatrix(1, true);
x3132 = x3133.times(x3134);

// Dealloc(X98684)
JCudaTensor x3138;
x3138 = x3129;
x3138.free();

// val X98687 = (X98685 + (i) => fc_B)
JCudaTensor x3139;
JCudaTensor x3140, x3141;
x3140 = x3132;
x3141 = x1042;
x3139 = x3141.copy(128, x3140);

// val X98688 = Cuda(Indicator(Y, 1000))
JCudaTensor x3142;
JTensorFloat x3143;
x3143 = x4.asIndicator(1000);
x3142 = x3143.asJCudaTensor();

// val X98689 = X98688 .* X98687
JCudaTensor x3144;
JCudaTensor x3145, x3146;
x3145 = x3142;
x3146 = x3139;
x3144 = x3145.times_i(x3146);

// val X98690 = Sum((X98689)(i1317 | @))
JCudaTensor x3147;
JCudaMatrix x3148;
JCudaTensor x3149;
x3149 = x3144;
x3148 = x3149.asMatrix(1, true);
x3147 = x3148.sum();

// Dealloc(X98689)
JCudaTensor x3150;
x3150 = x3144;
x3150.free();

// val X98691 = Max((X98687)(i1317 | @))
JCudaTensor x3151;
JCudaMatrix x3152;
JCudaTensor x3153;
x3153 = x3139;
x3152 = x3153.asMatrix(1, true);
x3151 = x3152.max();

// Dealloc(X98687)
JCudaTensor x3154;
x3154 = x3139;
x3154.free();

// val X98692 = 1{X98690 == X98691}
JCudaTensor x3155;
JCudaTensor x3156, x3157;
x3156 = x3147;
x3157 = x3151;
x3155 = x3156.eq(x3157);

// Dealloc(X98691)
JCudaTensor x3158;
x3158 = x3151;
x3158.free();

// Print((Sum(X98692) / |128|))
float x3159;
float x3160;
float x3161;
JCudaTensor x3162;
x3162 = x3155;
x3160 = x3162.sum();
x3161 = 128;
x3159 = x3160 / x3161;
System.out.println(x5 + " test precision "  + x3159);

// Dealloc(X98692)
JCudaTensor x3163;
x3163 = x3155;
x3163.free();

}
 
}

}