package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;


public class Googlenet {
// comment the line below for memory efficient mode
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

// (Convolv(1,0),List(List(128, 192, 28, 28), List(16, 192, 1, 1), List(16)))
static JCudnnConvolution x79 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{16,192,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 192, 28, 28), List(32, 192, 1, 1), List(32)))
static JCudnnConvolution x86 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{32,192,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 192, 28, 28), List(64, 192, 1, 1), List(64)))
static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{64,192,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 192, 28, 28), List(96, 192, 1, 1), List(96)))
static JCudnnConvolution x69 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{96,192,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(16, 256, 1, 1), List(16)))
static JCudnnConvolution x210 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(32, 256, 1, 1), List(32)))
static JCudnnConvolution x223 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x193 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 14, 14), List(96, 256, 1, 1), List(96)))
static JCudnnConvolution x200 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(16, 256, 1, 1), List(16)))
static JCudnnConvolution x133 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(32, 256, 1, 1), List(32)))
static JCudnnConvolution x156 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x140 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 28, 28), List(96, 256, 1, 1), List(96)))
static JCudnnConvolution x147 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 4, 4), List(128, 256, 1, 1), List(128)))
static JCudnnConvolution x286 = new JCudnnConvolution(new int[]{128,256,4,4},new int[]{128,256,1,1},new int[]{128}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(16, 256, 1, 1), List(16)))
static JCudnnConvolution x704 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(32, 256, 1, 1), List(32)))
static JCudnnConvolution x711 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(64, 256, 1, 1), List(64)))
static JCudnnConvolution x685 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
// (Convolv(1,0),List(List(128, 256, 7, 7), List(96, 256, 1, 1), List(96)))
static JCudnnConvolution x697 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
// (Convolv(1,0),List(List(128, 64, 56, 56), List(64, 64, 1, 1), List(64)))
static JCudnnConvolution x36 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{64,64,1,1},new int[]{64}, 1, 0);
// (Convolv(1,1),List(List(128, 64, 56, 56), List(192, 64, 3, 3), List(192)))
static JCudnnConvolution x46 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{192,64,3,3},new int[]{192}, 1, 1);
// (Convolv(1,1),List(List(128, 96, 14, 14), List(128, 96, 3, 3), List(128)))
static JCudnnConvolution x237 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(128, 96, 28, 28), List(128, 96, 3, 3), List(128)))
static JCudnnConvolution x99 = new JCudnnConvolution(new int[]{128,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(128, 96, 7, 7), List(128, 96, 3, 3), List(128)))
static JCudnnConvolution x734 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
// (Convolv(1,2),List(List(128, 16, 14, 14), List(32, 16, 5, 5), List(32)))
static JCudnnConvolution x230 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(List(128, 16, 28, 28), List(32, 16, 5, 5), List(32)))
static JCudnnConvolution x106 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(1,2),List(List(128, 16, 7, 7), List(32, 16, 5, 5), List(32)))
static JCudnnConvolution x741 = new JCudnnConvolution(new int[]{128,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
// (Convolv(2,3),List(List(128, 3, 224, 224), List(64, 3, 7, 7), List(64)))
static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
// (Dropout(0.4),List(List(128, 256, 1, 1)))
static JCudnnDropout x930 = new JCudnnDropout(new int[]{128,256,1,1}, 0.4f);
// (Dropout(0.7),List(List(128, 1024)))
static JCudnnDropout x361 = new JCudnnDropout(new int[]{128,1024}, 0.7f);
// (LRN(5,1.0E-4,0.75),List(List(128, 192, 56, 56)))
static JCudnnLRN x52 = new JCudnnLRN(new int[]{128,192,56,56}, 5, 1.0E-4, 0.75);
// (LRN(5,1.0E-4,0.75),List(List(128, 64, 56, 56)))
static JCudnnLRN x29 = new JCudnnLRN(new int[]{128,64,56,56}, 5, 1.0E-4, 0.75);
// (Lmdb(1000000,10000,Win32,1000),false)
static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, platform, 1000, true);
// (Lmdb(1000000,10000,Win32,1000),true)
static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, platform, 1000, false);
// (LogSoftmax(),List(List(128, 1000)))
static JCudnnSoftmax x407 = new JCudnnSoftmax(new int[]{128,1000}, SoftmaxAlgorithm.LOG);
// (Pooling(3,1,1,true),List(List(128, 192, 28, 28)))
static JCudnnPooling x72 = new JCudnnPooling(new int[]{128,192,28,28}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,1,1,true),List(List(128, 256, 14, 14)))
static JCudnnPooling x203 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,1,1,true),List(List(128, 256, 28, 28)))
static JCudnnPooling x126 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,1,1,true),List(List(128, 256, 7, 7)))
static JCudnnPooling x688 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 192, 56, 56)))
static JCudnnPooling x55 = new JCudnnPooling(new int[]{128,192,56,56}, 3, 2, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 256, 14, 14)))
static JCudnnPooling x676 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 2, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 256, 28, 28)))
static JCudnnPooling x186 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 2, 1, PoolingType.MAX);
// (Pooling(3,2,1,true),List(List(128, 64, 112, 112)))
static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,64,112,112}, 3, 2, 1, PoolingType.MAX);
// (Pooling(5,3,0,false),List(List(128, 256, 14, 14)))
static JCudnnPooling x257 = new JCudnnPooling(new int[]{128,256,14,14}, 5, 3, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
// (Pooling(7,1,0,false),List(List(128, 256, 7, 7)))
static JCudnnPooling x902 = new JCudnnPooling(new int[]{128,256,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
// (ReLU(),List(List(128, 1024)))
static JCudnnActivation x342 = new JCudnnActivation(new int[]{128,1024}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 14, 14)))
static JCudnnActivation x243 = new JCudnnActivation(new int[]{128,128,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 28, 28)))
static JCudnnActivation x112 = new JCudnnActivation(new int[]{128,128,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 4, 4)))
static JCudnnActivation x303 = new JCudnnActivation(new int[]{128,128,4,4}, ActivationMode.RELU);
// (ReLU(),List(List(128, 128, 7, 7)))
static JCudnnActivation x750 = new JCudnnActivation(new int[]{128,128,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 16, 14, 14)))
static JCudnnActivation x213 = new JCudnnActivation(new int[]{128,16,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 16, 28, 28)))
static JCudnnActivation x92 = new JCudnnActivation(new int[]{128,16,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 16, 7, 7)))
static JCudnnActivation x717 = new JCudnnActivation(new int[]{128,16,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 192, 56, 56)))
static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,192,56,56}, ActivationMode.RELU);
// (ReLU(),List(List(128, 32, 14, 14)))
static JCudnnActivation x246 = new JCudnnActivation(new int[]{128,32,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 32, 28, 28)))
static JCudnnActivation x115 = new JCudnnActivation(new int[]{128,32,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 32, 7, 7)))
static JCudnnActivation x747 = new JCudnnActivation(new int[]{128,32,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 112, 112)))
static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,64,112,112}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 14, 14)))
static JCudnnActivation x240 = new JCudnnActivation(new int[]{128,64,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 28, 28)))
static JCudnnActivation x109 = new JCudnnActivation(new int[]{128,64,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 56, 56)))
static JCudnnActivation x39 = new JCudnnActivation(new int[]{128,64,56,56}, ActivationMode.RELU);
// (ReLU(),List(List(128, 64, 7, 7)))
static JCudnnActivation x755 = new JCudnnActivation(new int[]{128,64,7,7}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 14, 14)))
static JCudnnActivation x216 = new JCudnnActivation(new int[]{128,96,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 28, 28)))
static JCudnnActivation x89 = new JCudnnActivation(new int[]{128,96,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(128, 96, 7, 7)))
static JCudnnActivation x714 = new JCudnnActivation(new int[]{128,96,7,7}, ActivationMode.RELU);
// BatchSum(((Sum(X98692) / |128|) / 10))
static float x3080;
// List(List(128, 64, 14, 14), List(128, 128, 14, 14), List(128, 32, 14, 14), List(128, 32, 14, 14))
static JCudnnConcat x250 = new JCudnnConcat(new int[]{128,64,14,14},new int[]{128,128,14,14},new int[]{128,32,14,14},new int[]{128,32,14,14});
// List(List(128, 64, 28, 28), List(128, 128, 28, 28), List(128, 32, 28, 28), List(128, 32, 28, 28))
static JCudnnConcat x119 = new JCudnnConcat(new int[]{128,64,28,28},new int[]{128,128,28,28},new int[]{128,32,28,28},new int[]{128,32,28,28});
// List(List(128, 64, 7, 7), List(128, 128, 7, 7), List(128, 32, 7, 7), List(128, 32, 7, 7))
static JCudnnConcat x760 = new JCudnnConcat(new int[]{128,64,7,7},new int[]{128,128,7,7},new int[]{128,32,7,7},new int[]{128,32,7,7});
// V_b1cv_B
static JCudaTensor x640 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_b1cv_W
static JCudaTensor x645 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_b1fc1_B
static JCudaTensor x605 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_b1fc1_W
static JCudaTensor x592 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
// V_b1fc2_B
static JCudaTensor x512 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_b1fc2_W
static JCudaTensor x517 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
// V_b2cv_B
static JCudaTensor x949 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_b2cv_W
static JCudaTensor x937 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
// V_b2fc1_B
static JCudaTensor x906 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
// V_b2fc1_W
static JCudaTensor x911 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
// V_b2fc2_B
static JCudaTensor x845 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_b2fc2_W
static JCudaTensor x861 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
// V_cv11_B
static JCudaTensor x2364 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv11_W
static JCudaTensor x2330 = JTensor.constFloat(0.0f, 64, 192, 1, 1).asJCudaTensor();
// V_cv12_B
static JCudaTensor x2416 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv12_W
static JCudaTensor x2424 = JTensor.constFloat(0.0f, 96, 192, 1, 1).asJCudaTensor();
// V_cv13_B
static JCudaTensor x2354 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv13_W
static JCudaTensor x2369 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv14_B
static JCudaTensor x2430 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv14_W
static JCudaTensor x2411 = JTensor.constFloat(0.0f, 16, 192, 1, 1).asJCudaTensor();
// V_cv15_B
static JCudaTensor x2342 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv15_W
static JCudaTensor x2358 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv16_B
static JCudaTensor x2335 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv16_W
static JCudaTensor x2346 = JTensor.constFloat(0.0f, 32, 192, 1, 1).asJCudaTensor();
// V_cv1_B
static JCudaTensor x2536 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv1_W
static JCudaTensor x2530 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
// V_cv21_B
static JCudaTensor x2206 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv21_W
static JCudaTensor x2192 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv22_B
static JCudaTensor x2265 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv22_W
static JCudaTensor x2273 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv23_B
static JCudaTensor x2197 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv23_W
static JCudaTensor x2218 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv24_B
static JCudaTensor x2261 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv24_W
static JCudaTensor x2256 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv25_B
static JCudaTensor x2201 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv25_W
static JCudaTensor x2184 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv26_B
static JCudaTensor x2172 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv26_W
static JCudaTensor x2179 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv2_B
static JCudaTensor x2496 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv2_W
static JCudaTensor x2503 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
// V_cv31_B
static JCudaTensor x2035 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv31_W
static JCudaTensor x2016 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv32_B
static JCudaTensor x2113 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv32_W
static JCudaTensor x2104 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv33_B
static JCudaTensor x2026 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv33_W
static JCudaTensor x2030 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv34_B
static JCudaTensor x2109 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv34_W
static JCudaTensor x2095 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv35_B
static JCudaTensor x2043 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv35_W
static JCudaTensor x2021 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv36_B
static JCudaTensor x2054 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv36_W
static JCudaTensor x2011 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv3_B
static JCudaTensor x2473 = JTensor.constFloat(0.0f, 192).asJCudaTensor();
// V_cv3_W
static JCudaTensor x2480 = JTensor.constFloat(0.0f, 192, 64, 3, 3).asJCudaTensor();
// V_cv41_B
static JCudaTensor x1854 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv41_W
static JCudaTensor x1865 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv42_B
static JCudaTensor x1947 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv42_W
static JCudaTensor x1942 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv43_B
static JCudaTensor x1891 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv43_W
static JCudaTensor x1895 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv44_B
static JCudaTensor x1952 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv44_W
static JCudaTensor x1933 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv45_B
static JCudaTensor x1861 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv45_W
static JCudaTensor x1873 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv46_B
static JCudaTensor x1878 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv46_W
static JCudaTensor x1849 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv51_B
static JCudaTensor x1715 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv51_W
static JCudaTensor x1694 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv52_B
static JCudaTensor x1783 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv52_W
static JCudaTensor x1795 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv53_B
static JCudaTensor x1711 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv53_W
static JCudaTensor x1702 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv54_B
static JCudaTensor x1787 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv54_W
static JCudaTensor x1778 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv55_B
static JCudaTensor x1707 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv55_W
static JCudaTensor x1740 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv56_B
static JCudaTensor x1726 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv56_W
static JCudaTensor x1734 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv61_B
static JCudaTensor x1581 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv61_W
static JCudaTensor x1563 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv62_B
static JCudaTensor x1637 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv62_W
static JCudaTensor x1632 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv63_B
static JCudaTensor x1586 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv63_W
static JCudaTensor x1568 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv64_B
static JCudaTensor x1642 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv64_W
static JCudaTensor x1623 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv65_B
static JCudaTensor x1576 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv65_W
static JCudaTensor x1558 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv66_B
static JCudaTensor x1542 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv66_W
static JCudaTensor x1549 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv71_B
static JCudaTensor x1419 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv71_W
static JCudaTensor x1380 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv72_B
static JCudaTensor x1475 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv72_W
static JCudaTensor x1470 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv73_B
static JCudaTensor x1390 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv73_W
static JCudaTensor x1385 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv74_B
static JCudaTensor x1480 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv74_W
static JCudaTensor x1461 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv75_B
static JCudaTensor x1401 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv75_W
static JCudaTensor x1413 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv76_B
static JCudaTensor x1424 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv76_W
static JCudaTensor x1405 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv81_B
static JCudaTensor x1219 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv81_W
static JCudaTensor x1239 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv82_B
static JCudaTensor x1305 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv82_W
static JCudaTensor x1313 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv83_B
static JCudaTensor x1250 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv83_W
static JCudaTensor x1223 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv84_B
static JCudaTensor x1319 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv84_W
static JCudaTensor x1300 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv85_B
static JCudaTensor x1255 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv85_W
static JCudaTensor x1231 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv86_B
static JCudaTensor x1260 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv86_W
static JCudaTensor x1245 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_cv91_B
static JCudaTensor x1084 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
// V_cv91_W
static JCudaTensor x1068 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
// V_cv92_B
static JCudaTensor x1163 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
// V_cv92_W
static JCudaTensor x1158 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
// V_cv93_B
static JCudaTensor x1108 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
// V_cv93_W
static JCudaTensor x1097 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
// V_cv94_B
static JCudaTensor x1150 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
// V_cv94_W
static JCudaTensor x1145 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
// V_cv95_B
static JCudaTensor x1093 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv95_W
static JCudaTensor x1102 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
// V_cv96_B
static JCudaTensor x1064 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
// V_cv96_W
static JCudaTensor x1079 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
// V_fc_B
static JCudaTensor x1020 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
// V_fc_W
static JCudaTensor x1014 = JTensor.constFloat(0.0f, 1000, 256).asJCudaTensor();
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// b1cv_B
static JCudaTensor x285 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b1cv_B").asJCudaTensor();
// b1cv_W
static JCudaTensor x284 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b1cv_W").asJCudaTensor();
// b1fc1_B
static JCudaTensor x333 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b1fc1_B").asJCudaTensor();
// b1fc1_W
static JCudaTensor x324 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b1fc1_W").asJCudaTensor();
// b1fc2_B
static JCudaTensor x399 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b1fc2_B").asJCudaTensor();
// b1fc2_W
static JCudaTensor x387 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b1fc2_W").asJCudaTensor();
// b2cv_B
static JCudaTensor x570 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b2cv_B").asJCudaTensor();
// b2cv_W
static JCudaTensor x569 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b2cv_W").asJCudaTensor();
// b2fc1_B
static JCudaTensor x668 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b2fc1_B").asJCudaTensor();
// b2fc1_W
static JCudaTensor x656 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b2fc1_W").asJCudaTensor();
// b2fc2_B
static JCudaTensor x727 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b2fc2_B").asJCudaTensor();
// b2fc2_W
static JCudaTensor x723 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b2fc2_W").asJCudaTensor();
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
static JCudaTensor x139 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv21_B").asJCudaTensor();
// cv21_W
static JCudaTensor x138 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv21_W").asJCudaTensor();
// cv22_B
static JCudaTensor x146 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv22_B").asJCudaTensor();
// cv22_W
static JCudaTensor x145 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv22_W").asJCudaTensor();
// cv23_B
static JCudaTensor x164 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv23_B").asJCudaTensor();
// cv23_W
static JCudaTensor x163 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv23_W").asJCudaTensor();
// cv24_B
static JCudaTensor x132 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv24_B").asJCudaTensor();
// cv24_W
static JCudaTensor x131 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv24_W").asJCudaTensor();
// cv25_B
static JCudaTensor x170 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv25_B").asJCudaTensor();
// cv25_W
static JCudaTensor x169 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv25_W").asJCudaTensor();
// cv26_B
static JCudaTensor x155 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv26_B").asJCudaTensor();
// cv26_W
static JCudaTensor x154 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv26_W").asJCudaTensor();
// cv2_B
static JCudaTensor x35 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv2_B").asJCudaTensor();
// cv2_W
static JCudaTensor x34 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/cv2_W").asJCudaTensor();
// cv31_B
static JCudaTensor x192 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv31_B").asJCudaTensor();
// cv31_W
static JCudaTensor x191 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv31_W").asJCudaTensor();
// cv32_B
static JCudaTensor x199 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv32_B").asJCudaTensor();
// cv32_W
static JCudaTensor x198 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv32_W").asJCudaTensor();
// cv33_B
static JCudaTensor x236 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv33_B").asJCudaTensor();
// cv33_W
static JCudaTensor x235 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
// cv34_B
static JCudaTensor x209 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv34_B").asJCudaTensor();
// cv34_W
static JCudaTensor x208 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv34_W").asJCudaTensor();
// cv35_B
static JCudaTensor x229 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv35_B").asJCudaTensor();
// cv35_W
static JCudaTensor x228 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv35_W").asJCudaTensor();
// cv36_B
static JCudaTensor x222 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv36_B").asJCudaTensor();
// cv36_W
static JCudaTensor x221 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv36_W").asJCudaTensor();
// cv3_B
static JCudaTensor x45 = JTensor.constFloat(0.2f, 192).load(network_dir + "/cv3_B").asJCudaTensor();
// cv3_W
static JCudaTensor x44 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 192, 64, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
// cv41_B
static JCudaTensor x277 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv41_B").asJCudaTensor();
// cv41_W
static JCudaTensor x276 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv41_W").asJCudaTensor();
// cv42_B
static JCudaTensor x271 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv42_B").asJCudaTensor();
// cv42_W
static JCudaTensor x270 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv42_W").asJCudaTensor();
// cv43_B
static JCudaTensor x309 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv43_B").asJCudaTensor();
// cv43_W
static JCudaTensor x308 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
// cv44_B
static JCudaTensor x263 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv44_B").asJCudaTensor();
// cv44_W
static JCudaTensor x262 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv44_W").asJCudaTensor();
// cv45_B
static JCudaTensor x300 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv45_B").asJCudaTensor();
// cv45_W
static JCudaTensor x299 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv45_W").asJCudaTensor();
// cv46_B
static JCudaTensor x294 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv46_B").asJCudaTensor();
// cv46_W
static JCudaTensor x293 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv46_W").asJCudaTensor();
// cv51_B
static JCudaTensor x350 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv51_B").asJCudaTensor();
// cv51_W
static JCudaTensor x349 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv51_W").asJCudaTensor();
// cv52_B
static JCudaTensor x339 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv52_B").asJCudaTensor();
// cv52_W
static JCudaTensor x338 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv52_W").asJCudaTensor();
// cv53_B
static JCudaTensor x375 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv53_B").asJCudaTensor();
// cv53_W
static JCudaTensor x374 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
// cv54_B
static JCudaTensor x356 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv54_B").asJCudaTensor();
// cv54_W
static JCudaTensor x355 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv54_W").asJCudaTensor();
// cv55_B
static JCudaTensor x381 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv55_B").asJCudaTensor();
// cv55_W
static JCudaTensor x380 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv55_W").asJCudaTensor();
// cv56_B
static JCudaTensor x367 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv56_B").asJCudaTensor();
// cv56_W
static JCudaTensor x366 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv56_W").asJCudaTensor();
// cv61_B
static JCudaTensor x435 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv61_B").asJCudaTensor();
// cv61_W
static JCudaTensor x434 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv61_W").asJCudaTensor();
// cv62_B
static JCudaTensor x429 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv62_B").asJCudaTensor();
// cv62_W
static JCudaTensor x428 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv62_W").asJCudaTensor();
// cv63_B
static JCudaTensor x474 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv63_B").asJCudaTensor();
// cv63_W
static JCudaTensor x473 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv63_W").asJCudaTensor();
// cv64_B
static JCudaTensor x418 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv64_B").asJCudaTensor();
// cv64_W
static JCudaTensor x417 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv64_W").asJCudaTensor();
// cv65_B
static JCudaTensor x461 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv65_B").asJCudaTensor();
// cv65_W
static JCudaTensor x460 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv65_W").asJCudaTensor();
// cv66_B
static JCudaTensor x443 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv66_B").asJCudaTensor();
// cv66_W
static JCudaTensor x442 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv66_W").asJCudaTensor();
// cv71_B
static JCudaTensor x543 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv71_B").asJCudaTensor();
// cv71_W
static JCudaTensor x542 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv71_W").asJCudaTensor();
// cv72_B
static JCudaTensor x549 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv72_B").asJCudaTensor();
// cv72_W
static JCudaTensor x548 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv72_W").asJCudaTensor();
// cv73_B
static JCudaTensor x627 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv73_B").asJCudaTensor();
// cv73_W
static JCudaTensor x626 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv73_W").asJCudaTensor();
// cv74_B
static JCudaTensor x560 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv74_B").asJCudaTensor();
// cv74_W
static JCudaTensor x559 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv74_W").asJCudaTensor();
// cv75_B
static JCudaTensor x618 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv75_B").asJCudaTensor();
// cv75_W
static JCudaTensor x617 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv75_W").asJCudaTensor();
// cv76_B
static JCudaTensor x584 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv76_B").asJCudaTensor();
// cv76_W
static JCudaTensor x583 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv76_W").asJCudaTensor();
// cv81_B
static JCudaTensor x684 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv81_B").asJCudaTensor();
// cv81_W
static JCudaTensor x683 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv81_W").asJCudaTensor();
// cv82_B
static JCudaTensor x696 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv82_B").asJCudaTensor();
// cv82_W
static JCudaTensor x695 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv82_W").asJCudaTensor();
// cv83_B
static JCudaTensor x733 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv83_B").asJCudaTensor();
// cv83_W
static JCudaTensor x732 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv83_W").asJCudaTensor();
// cv84_B
static JCudaTensor x703 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv84_B").asJCudaTensor();
// cv84_W
static JCudaTensor x702 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv84_W").asJCudaTensor();
// cv85_B
static JCudaTensor x740 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv85_B").asJCudaTensor();
// cv85_W
static JCudaTensor x739 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv85_W").asJCudaTensor();
// cv86_B
static JCudaTensor x710 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv86_B").asJCudaTensor();
// cv86_W
static JCudaTensor x709 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv86_W").asJCudaTensor();
// cv91_B
static JCudaTensor x782 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv91_B").asJCudaTensor();
// cv91_W
static JCudaTensor x781 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv91_W").asJCudaTensor();
// cv92_B
static JCudaTensor x776 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv92_B").asJCudaTensor();
// cv92_W
static JCudaTensor x775 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv92_W").asJCudaTensor();
// cv93_B
static JCudaTensor x820 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv93_B").asJCudaTensor();
// cv93_W
static JCudaTensor x819 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv93_W").asJCudaTensor();
// cv94_B
static JCudaTensor x788 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv94_B").asJCudaTensor();
// cv94_W
static JCudaTensor x787 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv94_W").asJCudaTensor();
// cv95_B
static JCudaTensor x829 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv95_B").asJCudaTensor();
// cv95_W
static JCudaTensor x828 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv95_W").asJCudaTensor();
// cv96_B
static JCudaTensor x802 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv96_B").asJCudaTensor();
// cv96_W
static JCudaTensor x801 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv96_W").asJCudaTensor();
// fc_B
static JCudaTensor x963 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
// fc_W
static JCudaTensor x948 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1000, 256).load(network_dir + "/fc_W").asJCudaTensor();

public static void main(String[] args){
double t = System.nanoTime();
train();
System.out.println((System.nanoTime() - t) / 1.0E9);
test();
x285.save(network_dir + "/b1cv_B");
x284.save(network_dir + "/b1cv_W");
x333.save(network_dir + "/b1fc1_B");
x324.save(network_dir + "/b1fc1_W");
x399.save(network_dir + "/b1fc2_B");
x387.save(network_dir + "/b1fc2_W");
x570.save(network_dir + "/b2cv_B");
x569.save(network_dir + "/b2cv_W");
x668.save(network_dir + "/b2fc1_B");
x656.save(network_dir + "/b2fc1_W");
x727.save(network_dir + "/b2fc2_B");
x723.save(network_dir + "/b2fc2_W");
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
x139.save(network_dir + "/cv21_B");
x138.save(network_dir + "/cv21_W");
x146.save(network_dir + "/cv22_B");
x145.save(network_dir + "/cv22_W");
x164.save(network_dir + "/cv23_B");
x163.save(network_dir + "/cv23_W");
x132.save(network_dir + "/cv24_B");
x131.save(network_dir + "/cv24_W");
x170.save(network_dir + "/cv25_B");
x169.save(network_dir + "/cv25_W");
x155.save(network_dir + "/cv26_B");
x154.save(network_dir + "/cv26_W");
x35.save(network_dir + "/cv2_B");
x34.save(network_dir + "/cv2_W");
x192.save(network_dir + "/cv31_B");
x191.save(network_dir + "/cv31_W");
x199.save(network_dir + "/cv32_B");
x198.save(network_dir + "/cv32_W");
x236.save(network_dir + "/cv33_B");
x235.save(network_dir + "/cv33_W");
x209.save(network_dir + "/cv34_B");
x208.save(network_dir + "/cv34_W");
x229.save(network_dir + "/cv35_B");
x228.save(network_dir + "/cv35_W");
x222.save(network_dir + "/cv36_B");
x221.save(network_dir + "/cv36_W");
x45.save(network_dir + "/cv3_B");
x44.save(network_dir + "/cv3_W");
x277.save(network_dir + "/cv41_B");
x276.save(network_dir + "/cv41_W");
x271.save(network_dir + "/cv42_B");
x270.save(network_dir + "/cv42_W");
x309.save(network_dir + "/cv43_B");
x308.save(network_dir + "/cv43_W");
x263.save(network_dir + "/cv44_B");
x262.save(network_dir + "/cv44_W");
x300.save(network_dir + "/cv45_B");
x299.save(network_dir + "/cv45_W");
x294.save(network_dir + "/cv46_B");
x293.save(network_dir + "/cv46_W");
x350.save(network_dir + "/cv51_B");
x349.save(network_dir + "/cv51_W");
x339.save(network_dir + "/cv52_B");
x338.save(network_dir + "/cv52_W");
x375.save(network_dir + "/cv53_B");
x374.save(network_dir + "/cv53_W");
x356.save(network_dir + "/cv54_B");
x355.save(network_dir + "/cv54_W");
x381.save(network_dir + "/cv55_B");
x380.save(network_dir + "/cv55_W");
x367.save(network_dir + "/cv56_B");
x366.save(network_dir + "/cv56_W");
x435.save(network_dir + "/cv61_B");
x434.save(network_dir + "/cv61_W");
x429.save(network_dir + "/cv62_B");
x428.save(network_dir + "/cv62_W");
x474.save(network_dir + "/cv63_B");
x473.save(network_dir + "/cv63_W");
x418.save(network_dir + "/cv64_B");
x417.save(network_dir + "/cv64_W");
x461.save(network_dir + "/cv65_B");
x460.save(network_dir + "/cv65_W");
x443.save(network_dir + "/cv66_B");
x442.save(network_dir + "/cv66_W");
x543.save(network_dir + "/cv71_B");
x542.save(network_dir + "/cv71_W");
x549.save(network_dir + "/cv72_B");
x548.save(network_dir + "/cv72_W");
x627.save(network_dir + "/cv73_B");
x626.save(network_dir + "/cv73_W");
x560.save(network_dir + "/cv74_B");
x559.save(network_dir + "/cv74_W");
x618.save(network_dir + "/cv75_B");
x617.save(network_dir + "/cv75_W");
x584.save(network_dir + "/cv76_B");
x583.save(network_dir + "/cv76_W");
x684.save(network_dir + "/cv81_B");
x683.save(network_dir + "/cv81_W");
x696.save(network_dir + "/cv82_B");
x695.save(network_dir + "/cv82_W");
x733.save(network_dir + "/cv83_B");
x732.save(network_dir + "/cv83_W");
x703.save(network_dir + "/cv84_B");
x702.save(network_dir + "/cv84_W");
x740.save(network_dir + "/cv85_B");
x739.save(network_dir + "/cv85_W");
x710.save(network_dir + "/cv86_B");
x709.save(network_dir + "/cv86_W");
x782.save(network_dir + "/cv91_B");
x781.save(network_dir + "/cv91_W");
x776.save(network_dir + "/cv92_B");
x775.save(network_dir + "/cv92_W");
x820.save(network_dir + "/cv93_B");
x819.save(network_dir + "/cv93_W");
x788.save(network_dir + "/cv94_B");
x787.save(network_dir + "/cv94_W");
x829.save(network_dir + "/cv95_B");
x828.save(network_dir + "/cv95_W");
x802.save(network_dir + "/cv96_B");
x801.save(network_dir + "/cv96_W");
x963.save(network_dir + "/fc_B");
x948.save(network_dir + "/fc_W");
x381.free();
x1942.free();
x61.free();
x1787.free();
x937.free();
x350.free();
x733.free();
x2369.free();
x1642.free();
x131.free();
x15.free();
x2364.free();
x2095.free();
x683.free();
x2261.free();
x1255.free();
x428.free();
x2330.free();
x781.free();
x1623.free();
x229.free();
x828.free();
x293.free();
x2016.free();
x356.free();
x435.free();
x906.free();
x1581.free();
x324.free();
x776.free();
x1102.free();
x1549.free();
x191.free();
x819.free();
x474.free();
x703.free();
x235.free();
x963.free();
x154.free();
x709.free();
x1895.free();
x164.free();
x2021.free();
x97.free();
x2030.free();
x138.free();
x1795.free();
x1461.free();
x1097.free();
x169.free();
x34.free();
x1726.free();
x775.free();
x732.free();
x2179.free();
x605.free();
x84.free();
x740.free();
x299.free();
x285.free();
x549.free();
x702.free();
x1250.free();
x1014.free();
x1865.free();
x739.free();
x723.free();
x98.free();
x1778.free();
x1239.free();
x1702.free();
x710.free();
x2430.free();
x1223.free();
x1854.free();
x473.free();
x1424.free();
x645.free();
x221.free();
x1563.free();
x584.free();
x67.free();
x1390.free();
x2172.free();
x35.free();
x668.free();
x461.free();
x375.free();
x2104.free();
x1891.free();
x355.free();
x271.free();
x2192.free();
x145.free();
x284.free();
x618.free();
x1952.free();
x333.free();
x656.free();
x801.free();
x163.free();
x1480.free();
x2026.free();
x2335.free();
x2035.free();
x339.free();
x308.free();
x542.free();
x294.free();
x77.free();
x263.free();
x442.free();
x911.free();
x170.free();
x782.free();
x727.free();
x380.free();
x1163.free();
x199.free();
x512.free();
x1020.free();
x399.free();
x338.free();
x2473.free();
x543.free();
x85.free();
x1084.free();
x1145.free();
x460.free();
x367.free();
x2184.free();
x192.free();
x1231.free();
x1401.free();
x787.free();
x1385.free();
x45.free();
x60.free();
x2109.free();
x570.free();
x1068.free();
x1568.free();
x1093.free();
x300.free();
x1150.free();
x78.free();
x2346.free();
x684.free();
x1245.free();
x1108.free();
x1694.free();
x1475.free();
x1419.free();
x349.free();
x1637.free();
x104.free();
x1711.free();
x640.free();
x2411.free();
x788.free();
x2265.free();
x366.free();
x1260.free();
x1707.free();
x309.free();
x1405.free();
x1064.free();
x443.free();
x627.free();
x1558.free();
x1933.free();
x949.free();
x1873.free();
x139.free();
x1313.free();
x861.free();
x276.free();
x1849.free();
x2011.free();
x1470.free();
x1878.free();
x1715.free();
x155.free();
x2218.free();
x2354.free();
x270.free();
x1576.free();
x569.free();
x617.free();
x1586.free();
x1632.free();
x146.free();
x1783.free();
x820.free();
x418.free();
x44.free();
x1947.free();
x1158.free();
x2113.free();
x948.free();
x2503.free();
x1219.free();
x2358.free();
x16.free();
x2256.free();
x695.free();
x626.free();
x2043.free();
x1079.free();
x2206.free();
x105.free();
x209.free();
x417.free();
x1413.free();
x845.free();
x222.free();
x559.free();
x374.free();
x262.free();
x829.free();
x1300.free();
x2197.free();
x132.free();
x429.free();
x1734.free();
x68.free();
x434.free();
x208.free();
x198.free();
x1380.free();
x2424.free();
x1305.free();
x2530.free();
x1861.free();
x583.free();
x236.free();
x592.free();
x517.free();
x2536.free();
x2342.free();
x2480.free();
x2054.free();
x1740.free();
x560.free();
x1542.free();
x2273.free();
x277.free();
x228.free();
x802.free();
x548.free();
x2201.free();
x2496.free();
x2416.free();
x696.free();
x1319.free();
x387.free();
x407.free();
x342.free();
x714.free();
x140.free();
x741.free();
x39.free();
x747.free();
x29.free();
x750.free();
x237.free();
x106.free();
x734.free();
x147.free();
x112.free();
x213.free();
x230.free();
x126.free();
x49.free();
x704.free();
x36.free();
x186.free();
x685.free();
x203.free();
x99.free();
x26.free();
x257.free();
x23.free();
x89.free();
x200.free();
x223.free();
x86.free();
x55.free();
x930.free();
x92.free();
x52.free();
x79.free();
x133.free();
x303.free();
x46.free();
x17.free();
x286.free();
x62.free();
x69.free();
x115.free();
x72.free();
x688.free();
x109.free();
x156.free();
x193.free();
x676.free();
x216.free();
x755.free();
x246.free();
x902.free();
x243.free();
x711.free();
x717.free();
x210.free();
x697.free();
x240.free();
x361.free();
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
x11 = x17.forward(x12, x13, x14);

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
x30 = x36.forward(x31, x32, x33);

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
x40 = x46.forward(x41, x42, x43);

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
x56 = x62.forward(x57, x58, x59);

// val X95 = Convolv(1,0)(X92,cv12_W,cv12_B)
JCudaTensor x63;
JCudaTensor x64, x65, x66;
x64 = x53;
x65 = x67;
x66 = x68;
x63 = x69.forward(x64, x65, x66);

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
x73 = x79.forward(x74, x75, x76);

// val X104 = Convolv(1,0)(X103,cv16_W,cv16_B)
JCudaTensor x80;
JCudaTensor x81, x82, x83;
x81 = x70;
x82 = x84;
x83 = x85;
x80 = x86.forward(x81, x82, x83);

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
x93 = x99.forward(x94, x95, x96);

// val X101 = Convolv(1,2)(X100,cv15_W,cv15_B)
JCudaTensor x100;
JCudaTensor x101, x102, x103;
x101 = x90;
x102 = x104;
x103 = x105;
x100 = x106.forward(x101, x102, x103);

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
x116 = x115.forward(x117);

// val X106 = Concat(X94,X98,X102,X105)
JCudaTensor x118;
JCudaTensor x120, x121, x122, x123;
x120 = x107;
x121 = x110;
x122 = x113;
x123 = x116;
x118 = x119.forward(x120,x121,x122,x123);

// val X117 = Pooling(3,1,1,true)(X106)
JCudaTensor x124;
JCudaTensor x125;
x125 = x118;
x124 = x126.forward(x125);

// val X113 = Convolv(1,0)(X106,cv24_W,cv24_B)
JCudaTensor x127;
JCudaTensor x128, x129, x130;
x128 = x118;
x129 = x131;
x130 = x132;
x127 = x133.forward(x128, x129, x130);

// val X107 = Convolv(1,0)(X106,cv21_W,cv21_B)
JCudaTensor x134;
JCudaTensor x135, x136, x137;
x135 = x118;
x136 = x138;
x137 = x139;
x134 = x140.forward(x135, x136, x137);

// val X109 = Convolv(1,0)(X106,cv22_W,cv22_B)
JCudaTensor x141;
JCudaTensor x142, x143, x144;
x142 = x118;
x143 = x145;
x144 = x146;
x141 = x147.forward(x142, x143, x144);

// val X110 = ReLU()(X109)
JCudaTensor x148;
JCudaTensor x149;
x149 = x141;
x148 = x89.forward(x149);

// val X118 = Convolv(1,0)(X117,cv26_W,cv26_B)
JCudaTensor x150;
JCudaTensor x151, x152, x153;
x151 = x124;
x152 = x154;
x153 = x155;
x150 = x156.forward(x151, x152, x153);

// val X114 = ReLU()(X113)
JCudaTensor x157;
JCudaTensor x158;
x158 = x127;
x157 = x92.forward(x158);

// val X111 = Convolv(1,1)(X110,cv23_W,cv23_B)
JCudaTensor x159;
JCudaTensor x160, x161, x162;
x160 = x148;
x161 = x163;
x162 = x164;
x159 = x99.forward(x160, x161, x162);

// val X115 = Convolv(1,2)(X114,cv25_W,cv25_B)
JCudaTensor x165;
JCudaTensor x166, x167, x168;
x166 = x157;
x167 = x169;
x168 = x170;
x165 = x106.forward(x166, x167, x168);

// val X108 = ReLU()(X107)
JCudaTensor x171;
JCudaTensor x172;
x172 = x134;
x171 = x109.forward(x172);

// val X112 = ReLU()(X111)
JCudaTensor x173;
JCudaTensor x174;
x174 = x159;
x173 = x112.forward(x174);

// val X116 = ReLU()(X115)
JCudaTensor x175;
JCudaTensor x176;
x176 = x165;
x175 = x115.forward(x176);

// val X119 = ReLU()(X118)
JCudaTensor x177;
JCudaTensor x178;
x178 = x150;
x177 = x115.forward(x178);

// val X120 = Concat(X108,X112,X116,X119)
JCudaTensor x179;
JCudaTensor x180, x181, x182, x183;
x180 = x171;
x181 = x173;
x182 = x175;
x183 = x177;
x179 = x119.forward(x180,x181,x182,x183);

// val X121 = Pooling(3,2,1,true)(X120)
JCudaTensor x184;
JCudaTensor x185;
x185 = x179;
x184 = x186.forward(x185);

// val X122 = Convolv(1,0)(X121,cv31_W,cv31_B)
JCudaTensor x187;
JCudaTensor x188, x189, x190;
x188 = x184;
x189 = x191;
x190 = x192;
x187 = x193.forward(x188, x189, x190);

// val X124 = Convolv(1,0)(X121,cv32_W,cv32_B)
JCudaTensor x194;
JCudaTensor x195, x196, x197;
x195 = x184;
x196 = x198;
x197 = x199;
x194 = x200.forward(x195, x196, x197);

// val X132 = Pooling(3,1,1,true)(X121)
JCudaTensor x201;
JCudaTensor x202;
x202 = x184;
x201 = x203.forward(x202);

// val X128 = Convolv(1,0)(X121,cv34_W,cv34_B)
JCudaTensor x204;
JCudaTensor x205, x206, x207;
x205 = x184;
x206 = x208;
x207 = x209;
x204 = x210.forward(x205, x206, x207);

// val X129 = ReLU()(X128)
JCudaTensor x211;
JCudaTensor x212;
x212 = x204;
x211 = x213.forward(x212);

// val X125 = ReLU()(X124)
JCudaTensor x214;
JCudaTensor x215;
x215 = x194;
x214 = x216.forward(x215);

// val X133 = Convolv(1,0)(X132,cv36_W,cv36_B)
JCudaTensor x217;
JCudaTensor x218, x219, x220;
x218 = x201;
x219 = x221;
x220 = x222;
x217 = x223.forward(x218, x219, x220);

// val X130 = Convolv(1,2)(X129,cv35_W,cv35_B)
JCudaTensor x224;
JCudaTensor x225, x226, x227;
x225 = x211;
x226 = x228;
x227 = x229;
x224 = x230.forward(x225, x226, x227);

// val X126 = Convolv(1,1)(X125,cv33_W,cv33_B)
JCudaTensor x231;
JCudaTensor x232, x233, x234;
x232 = x214;
x233 = x235;
x234 = x236;
x231 = x237.forward(x232, x233, x234);

// val X123 = ReLU()(X122)
JCudaTensor x238;
JCudaTensor x239;
x239 = x187;
x238 = x240.forward(x239);

// val X127 = ReLU()(X126)
JCudaTensor x241;
JCudaTensor x242;
x242 = x231;
x241 = x243.forward(x242);

// val X131 = ReLU()(X130)
JCudaTensor x244;
JCudaTensor x245;
x245 = x224;
x244 = x246.forward(x245);

// val X134 = ReLU()(X133)
JCudaTensor x247;
JCudaTensor x248;
x248 = x217;
x247 = x246.forward(x248);

// val X135 = Concat(X123,X127,X131,X134)
JCudaTensor x249;
JCudaTensor x251, x252, x253, x254;
x251 = x238;
x252 = x241;
x253 = x244;
x254 = x247;
x249 = x250.forward(x251,x252,x253,x254);

// val X241 = Pooling(5,3,0,false)(X135)
JCudaTensor x255;
JCudaTensor x256;
x256 = x249;
x255 = x257.forward(x256);

// val X142 = Convolv(1,0)(X135,cv44_W,cv44_B)
JCudaTensor x258;
JCudaTensor x259, x260, x261;
x259 = x249;
x260 = x262;
x261 = x263;
x258 = x210.forward(x259, x260, x261);

// val X146 = Pooling(3,1,1,true)(X135)
JCudaTensor x264;
JCudaTensor x265;
x265 = x249;
x264 = x203.forward(x265);

// val X138 = Convolv(1,0)(X135,cv42_W,cv42_B)
JCudaTensor x266;
JCudaTensor x267, x268, x269;
x267 = x249;
x268 = x270;
x269 = x271;
x266 = x200.forward(x267, x268, x269);

// val X136 = Convolv(1,0)(X135,cv41_W,cv41_B)
JCudaTensor x272;
JCudaTensor x273, x274, x275;
x273 = x249;
x274 = x276;
x275 = x277;
x272 = x193.forward(x273, x274, x275);

// val X143 = ReLU()(X142)
JCudaTensor x278;
JCudaTensor x279;
x279 = x258;
x278 = x213.forward(x279);

// val X242 = Convolv(1,0)(X241,b1cv_W,b1cv_B)
JCudaTensor x280;
JCudaTensor x281, x282, x283;
x281 = x255;
x282 = x284;
x283 = x285;
x280 = x286.forward(x281, x282, x283);

// val X139 = ReLU()(X138)
JCudaTensor x287;
JCudaTensor x288;
x288 = x266;
x287 = x216.forward(x288);

// val X147 = Convolv(1,0)(X146,cv46_W,cv46_B)
JCudaTensor x289;
JCudaTensor x290, x291, x292;
x290 = x264;
x291 = x293;
x292 = x294;
x289 = x223.forward(x290, x291, x292);

// val X144 = Convolv(1,2)(X143,cv45_W,cv45_B)
JCudaTensor x295;
JCudaTensor x296, x297, x298;
x296 = x278;
x297 = x299;
x298 = x300;
x295 = x230.forward(x296, x297, x298);

// val X243 = ReLU()(X242)
JCudaTensor x301;
JCudaTensor x302;
x302 = x280;
x301 = x303.forward(x302);

// val X140 = Convolv(1,1)(X139,cv43_W,cv43_B)
JCudaTensor x304;
JCudaTensor x305, x306, x307;
x305 = x287;
x306 = x308;
x307 = x309;
x304 = x237.forward(x305, x306, x307);

// val X145 = ReLU()(X144)
JCudaTensor x310;
JCudaTensor x311;
x311 = x295;
x310 = x246.forward(x311);

// val X137 = ReLU()(X136)
JCudaTensor x312;
JCudaTensor x313;
x313 = x272;
x312 = x240.forward(x313);

// val X141 = ReLU()(X140)
JCudaTensor x314;
JCudaTensor x315;
x315 = x304;
x314 = x243.forward(x315);

// val X148 = ReLU()(X147)
JCudaTensor x316;
JCudaTensor x317;
x317 = x289;
x316 = x246.forward(x317);

// val X244 = (X243[1><3])(i | @) * (b1fc1_W)(j | @)
JCudaTensor x318;
JCudaMatrix x319;
JCudaMatrix x320;
JCudaTensor x321;
JCudaTensor x322;
x322 = x301;
x321 = x322.flatten(1, new int[]{128, 4, 4});
x319 = x321.asMatrix(1, true);
JCudaTensor x323;
x323 = x324;
x320 = x323.asMatrix(1, true);
x318 = x319.times(x320);

// val X149 = Concat(X137,X141,X145,X148)
JCudaTensor x325;
JCudaTensor x326, x327, x328, x329;
x326 = x312;
x327 = x314;
x328 = x310;
x329 = x316;
x325 = x250.forward(x326,x327,x328,x329);

// val X246 = (X244 + (i) => b1fc1_B)
JCudaTensor x330;
JCudaTensor x331, x332;
x331 = x318;
x332 = x333;
x330 = x332.copy(128, x331);

// val X152 = Convolv(1,0)(X149,cv52_W,cv52_B)
JCudaTensor x334;
JCudaTensor x335, x336, x337;
x335 = x325;
x336 = x338;
x337 = x339;
x334 = x200.forward(x335, x336, x337);

// val X247 = ReLU()(X246)
JCudaTensor x340;
JCudaTensor x341;
x341 = x330;
x340 = x342.forward(x341);

// val X160 = Pooling(3,1,1,true)(X149)
JCudaTensor x343;
JCudaTensor x344;
x344 = x325;
x343 = x203.forward(x344);

// val X150 = Convolv(1,0)(X149,cv51_W,cv51_B)
JCudaTensor x345;
JCudaTensor x346, x347, x348;
x346 = x325;
x347 = x349;
x348 = x350;
x345 = x193.forward(x346, x347, x348);

// val X156 = Convolv(1,0)(X149,cv54_W,cv54_B)
JCudaTensor x351;
JCudaTensor x352, x353, x354;
x352 = x325;
x353 = x355;
x354 = x356;
x351 = x210.forward(x352, x353, x354);

// val X157 = ReLU()(X156)
JCudaTensor x357;
JCudaTensor x358;
x358 = x351;
x357 = x213.forward(x358);

// val X248 = Dropout(0.7)(X247)
JCudaTensor x359;
JCudaTensor x360;
x360 = x340;
x359 = x361.forward(x360);

// val X161 = Convolv(1,0)(X160,cv56_W,cv56_B)
JCudaTensor x362;
JCudaTensor x363, x364, x365;
x363 = x343;
x364 = x366;
x365 = x367;
x362 = x223.forward(x363, x364, x365);

// val X153 = ReLU()(X152)
JCudaTensor x368;
JCudaTensor x369;
x369 = x334;
x368 = x216.forward(x369);

// val X154 = Convolv(1,1)(X153,cv53_W,cv53_B)
JCudaTensor x370;
JCudaTensor x371, x372, x373;
x371 = x368;
x372 = x374;
x373 = x375;
x370 = x237.forward(x371, x372, x373);

// val X158 = Convolv(1,2)(X157,cv55_W,cv55_B)
JCudaTensor x376;
JCudaTensor x377, x378, x379;
x377 = x357;
x378 = x380;
x379 = x381;
x376 = x230.forward(x377, x378, x379);

// val X249 = (X248)(i | @) * (b1fc2_W)(j | @)
JCudaTensor x382;
JCudaMatrix x383;
JCudaMatrix x384;
JCudaTensor x385;
x385 = x359;
x383 = x385.asMatrix(1, true);
JCudaTensor x386;
x386 = x387;
x384 = x386.asMatrix(1, true);
x382 = x383.times(x384);

// val X155 = ReLU()(X154)
JCudaTensor x388;
JCudaTensor x389;
x389 = x370;
x388 = x243.forward(x389);

// val X162 = ReLU()(X161)
JCudaTensor x390;
JCudaTensor x391;
x391 = x362;
x390 = x246.forward(x391);

// val X159 = ReLU()(X158)
JCudaTensor x392;
JCudaTensor x393;
x393 = x376;
x392 = x246.forward(x393);

// val X151 = ReLU()(X150)
JCudaTensor x394;
JCudaTensor x395;
x395 = x345;
x394 = x240.forward(x395);

// val X251 = (X249 + (i) => b1fc2_B)
JCudaTensor x396;
JCudaTensor x397, x398;
x397 = x382;
x398 = x399;
x396 = x398.copy(128, x397);

// val X163 = Concat(X151,X155,X159,X162)
JCudaTensor x400;
JCudaTensor x401, x402, x403, x404;
x401 = x394;
x402 = x388;
x403 = x392;
x404 = x390;
x400 = x250.forward(x401,x402,x403,x404);

// val X252 = LogSoftmax()(X251)
JCudaTensor x405;
JCudaTensor x406;
x406 = x396;
x405 = x407.forward(x406);

// Dealloc(X251)
JCudaTensor x408;
x408 = x396;
x408.free();

// val X337 = (X336 / |128|)
JCudaTensor x409;
JCudaTensor x410;
float x411;
x410 = x18;
float x412;
x412 = 128;
x411 = 1 / x412;
x409 = x410.times_i(x411);

// val X170 = Convolv(1,0)(X163,cv64_W,cv64_B)
JCudaTensor x413;
JCudaTensor x414, x415, x416;
x414 = x400;
x415 = x417;
x416 = x418;
x413 = x210.forward(x414, x415, x416);

// val X339 = X337 * d_LogSoftmax()(X252)/d_X251
JCudaTensor x419;
JCudaTensor x420, x421;
x420 = x409;
x421 = x405;
x419 = x407.backward(x420, x421);

// val m1 = (i23) => b1fc2_W[@, i23]
JCudaMatrix x422;
JCudaTensor x423;
x423 = x387;
x422 = x423.asMatrix(1, false);

// val X166 = Convolv(1,0)(X163,cv62_W,cv62_B)
JCudaTensor x424;
JCudaTensor x425, x426, x427;
x425 = x400;
x426 = x428;
x427 = x429;
x424 = x200.forward(x425, x426, x427);

// val X164 = Convolv(1,0)(X163,cv61_W,cv61_B)
JCudaTensor x430;
JCudaTensor x431, x432, x433;
x431 = x400;
x432 = x434;
x433 = x435;
x430 = x193.forward(x431, x432, x433);

// val X174 = Pooling(3,1,1,true)(X163)
JCudaTensor x436;
JCudaTensor x437;
x437 = x400;
x436 = x203.forward(x437);

// val X175 = Convolv(1,0)(X174,cv66_W,cv66_B)
JCudaTensor x438;
JCudaTensor x439, x440, x441;
x439 = x436;
x440 = x442;
x441 = x443;
x438 = x223.forward(x439, x440, x441);

// val m12 = (i119) => X248[@, i119]
JCudaMatrix x444;
JCudaTensor x445;
x445 = x359;
x444 = x445.asMatrix(1, false);

// val X348 = (X339)(i22 | @) * m1
JCudaTensor x446;
JCudaMatrix x447;
JCudaMatrix x448;
JCudaTensor x449;
x449 = x419;
x447 = x449.asMatrix(1, true);
x448 = x422;
x446 = x447.times(x448);

// val X167 = ReLU()(X166)
JCudaTensor x450;
JCudaTensor x451;
x451 = x424;
x450 = x216.forward(x451);

// val X171 = ReLU()(X170)
JCudaTensor x452;
JCudaTensor x453;
x453 = x413;
x452 = x213.forward(x453);

// val m10 = (i111) => X339[@, i111]
JCudaMatrix x454;
JCudaTensor x455;
x455 = x419;
x454 = x455.asMatrix(1, false);

// val X172 = Convolv(1,2)(X171,cv65_W,cv65_B)
JCudaTensor x456;
JCudaTensor x457, x458, x459;
x457 = x452;
x458 = x460;
x459 = x461;
x456 = x230.forward(x457, x458, x459);

// val X349 = X348 * d_Dropout(0.7)()/d_X247
JCudaTensor x462;
JCudaTensor x463;
x463 = x446;
x462 = x361.backward(x463);

// Dealloc(X348)
JCudaTensor x464;
x464 = x446;
x464.free();

// val X832 = m10 * m12
JCudaTensor x465;
JCudaMatrix x466;
JCudaMatrix x467;
x466 = x454;
x467 = x444;
x465 = x466.times(x467);

// Dealloc(X248)
JCudaTensor x468;
x468 = x359;
x468.free();

// val X168 = Convolv(1,1)(X167,cv63_W,cv63_B)
JCudaTensor x469;
JCudaTensor x470, x471, x472;
x470 = x450;
x471 = x473;
x472 = x474;
x469 = x237.forward(x470, x471, x472);

// val X742 = Sum(m10)
JCudaTensor x475;
JCudaMatrix x476;
x476 = x454;
x475 = x476.sum();

// Dealloc(X339)
JCudaTensor x477;
x477 = x419;
x477.free();

// val X833 = (X832 * loss1)
JCudaTensor x478;
JCudaTensor x479;
float x480;
x479 = x465;
x480 = loss1;
x478 = x479.times_i(x480);

// val X176 = ReLU()(X175)
JCudaTensor x481;
JCudaTensor x482;
x482 = x438;
x481 = x246.forward(x482);

// val X165 = ReLU()(X164)
JCudaTensor x483;
JCudaTensor x484;
x484 = x430;
x483 = x240.forward(x484);

// val m2 = (i27) => b1fc1_W[@, i27]
JCudaMatrix x485;
JCudaTensor x486;
x486 = x324;
x485 = x486.asMatrix(1, false);

// val X169 = ReLU()(X168)
JCudaTensor x487;
JCudaTensor x488;
x488 = x469;
x487 = x243.forward(x488);

// val X173 = ReLU()(X172)
JCudaTensor x489;
JCudaTensor x490;
x490 = x456;
x489 = x246.forward(x490);

// val X743 = (X742 * loss1)
JCudaTensor x491;
JCudaTensor x492;
float x493;
x492 = x475;
x493 = loss1;
x491 = x492.times_i(x493);

// val X351 = X349 * d_ReLU()(X247)/d_X246
JCudaTensor x494;
JCudaTensor x495, x496;
x495 = x462;
x496 = x340;
x494 = x342.backward(x495, x496);

// Dealloc(X247)
JCudaTensor x497;
x497 = x340;
x497.free();

// val m9 = (i92) => X243[1><3][@, i92]
JCudaMatrix x498;
JCudaTensor x499;
JCudaTensor x500;
x500 = x301;
x499 = x500.flatten(1, new int[]{128, 4, 4});
x498 = x499.asMatrix(1, false);

// val m6 = (i74) => X351[@, i74]
JCudaMatrix x501;
JCudaTensor x502;
x502 = x494;
x501 = x502.asMatrix(1, false);

// val X177 = Concat(X165,X169,X173,X176)
JCudaTensor x503;
JCudaTensor x504, x505, x506, x507;
x504 = x483;
x505 = x487;
x506 = x489;
x507 = x481;
x503 = x250.forward(x504,x505,x506,x507);

// val X352 = (X351)(i26 | @) * m2
JCudaTensor x508;
JCudaMatrix x509;
JCudaMatrix x510;
JCudaTensor x511;
x511 = x494;
x509 = x511.asMatrix(1, true);
x510 = x485;
x508 = x509.times(x510);

// V_b1fc2_B <~~ X743
float x513, x514;
x513 = lrn_rate_2;
x514 = momentum;
JCudaTensor x515;
x515 = x491;
x512.update(x515, x513, x514);

// Dealloc(X743)
JCudaTensor x516;
x516 = x491;
x516.free();

// V_b1fc2_W <~~ X833
float x518, x519;
x518 = lrn_rate_1;
x519 = momentum;
JCudaTensor x520;
x520 = x478;
x517.update(x520, x518, x519);

// Dealloc(X833)
JCudaTensor x521;
x521 = x478;
x521.free();

// b1fc2_W <~~ V_b1fc2_W
float x522, x523;
x522 = 1;
x523 = decay_1;
JCudaTensor x524;
x524 = x517;
x387.update(x524, x522, x523);

// b1fc2_B <~~ V_b1fc2_B
float x525, x526;
x525 = 1;
x526 = 1;
JCudaTensor x527;
x527 = x512;
x399.update(x527, x525, x526);

// val X188 = Pooling(3,1,1,true)(X177)
JCudaTensor x528;
JCudaTensor x529;
x529 = x503;
x528 = x203.forward(x529);

// val X228 = Pooling(5,3,0,false)(X177)
JCudaTensor x530;
JCudaTensor x531;
x531 = x503;
x530 = x257.forward(x531);

// val X555 = Sum(m6)
JCudaTensor x532;
JCudaMatrix x533;
x533 = x501;
x532 = x533.sum();

// val X354 = X352[1<>3] * d_ReLU()(X243)/d_X242
JCudaTensor x534;
JCudaTensor x535, x536;
JCudaTensor x537;
x537 = x508;
x535 = x537.unflatten(1, new int[]{128, 4, 4});
x536 = x301;
x534 = x303.backward(x535, x536);

// val X178 = Convolv(1,0)(X177,cv71_W,cv71_B)
JCudaTensor x538;
JCudaTensor x539, x540, x541;
x539 = x503;
x540 = x542;
x541 = x543;
x538 = x193.forward(x539, x540, x541);

// val X180 = Convolv(1,0)(X177,cv72_W,cv72_B)
JCudaTensor x544;
JCudaTensor x545, x546, x547;
x545 = x503;
x546 = x548;
x547 = x549;
x544 = x200.forward(x545, x546, x547);

// val X652 = m6 * m9
JCudaTensor x550;
JCudaMatrix x551;
JCudaMatrix x552;
x551 = x501;
x552 = x498;
x550 = x551.times(x552);

// Dealloc(X351)
JCudaTensor x553;
x553 = x494;
x553.free();

// Dealloc(X243)
JCudaTensor x554;
x554 = x301;
x554.free();

// val X184 = Convolv(1,0)(X177,cv74_W,cv74_B)
JCudaTensor x555;
JCudaTensor x556, x557, x558;
x556 = x503;
x557 = x559;
x558 = x560;
x555 = x210.forward(x556, x557, x558);

// val X355 = X354 * d_Convolv(1,0)()/d_b1cv_B
JCudaTensor x561;
JCudaTensor x562;
x562 = x534;
x561 = x286.backward_bias(x562);

// val X181 = ReLU()(X180)
JCudaTensor x563;
JCudaTensor x564;
x564 = x544;
x563 = x216.forward(x564);

// val X229 = Convolv(1,0)(X228,b2cv_W,b2cv_B)
JCudaTensor x565;
JCudaTensor x566, x567, x568;
x566 = x530;
x567 = x569;
x568 = x570;
x565 = x286.forward(x566, x567, x568);

// val X2810 = X354 * d_Convolv(1,0)(b1cv_W)/d_X241
JCudaTensor x571;
JCudaTensor x572, x573;
x572 = x534;
x573 = x284;
x571 = x286.backward_data(x572, x573);

// val X653 = (X652 * loss1)
JCudaTensor x574;
JCudaTensor x575;
float x576;
x575 = x550;
x576 = loss1;
x574 = x575.times_i(x576);

// val X185 = ReLU()(X184)
JCudaTensor x577;
JCudaTensor x578;
x578 = x555;
x577 = x213.forward(x578);

// val X189 = Convolv(1,0)(X188,cv76_W,cv76_B)
JCudaTensor x579;
JCudaTensor x580, x581, x582;
x580 = x528;
x581 = x583;
x582 = x584;
x579 = x223.forward(x580, x581, x582);

// val X458 = X354 * d_Convolv(1,0)(X241)/d_b1cv_W
JCudaTensor x585;
JCudaTensor x586, x587;
x586 = x534;
x587 = x255;
x585 = x286.backward_filter(x586, x587);

// Dealloc(X354)
JCudaTensor x588;
x588 = x534;
x588.free();

// val X556 = (X555 * loss1)
JCudaTensor x589;
JCudaTensor x590;
float x591;
x590 = x532;
x591 = loss1;
x589 = x590.times_i(x591);

// V_b1fc1_W <~~ X653
float x593, x594;
x593 = lrn_rate_1;
x594 = momentum;
JCudaTensor x595;
x595 = x574;
x592.update(x595, x593, x594);

// Dealloc(X653)
JCudaTensor x596;
x596 = x574;
x596.free();

// val X230 = ReLU()(X229)
JCudaTensor x597;
JCudaTensor x598;
x598 = x565;
x597 = x303.forward(x598);

// val X2812 = X2810 * d_Pooling(5,3,0,false)(X241,X135)/d_X135
JCudaTensor x599;
JCudaTensor x600, x601, x602;
x600 = x571;
x601 = x255;
x602 = x249;
x599 = x257.backward(x600, x601, x602);

// Dealloc(X2810)
JCudaTensor x603;
x603 = x571;
x603.free();

// Dealloc(X241)
JCudaTensor x604;
x604 = x255;
x604.free();

// V_b1fc1_B <~~ X556
float x606, x607;
x606 = lrn_rate_2;
x607 = momentum;
JCudaTensor x608;
x608 = x589;
x605.update(x608, x606, x607);

// Dealloc(X556)
JCudaTensor x609;
x609 = x589;
x609.free();

// val X356 = (X355 * loss1)
JCudaTensor x610;
JCudaTensor x611;
float x612;
x611 = x561;
x612 = loss1;
x610 = x611.times_i(x612);

// val X186 = Convolv(1,2)(X185,cv75_W,cv75_B)
JCudaTensor x613;
JCudaTensor x614, x615, x616;
x614 = x577;
x615 = x617;
x616 = x618;
x613 = x230.forward(x614, x615, x616);

// val X459 = (X458 * loss1)
JCudaTensor x619;
JCudaTensor x620;
float x621;
x620 = x585;
x621 = loss1;
x619 = x620.times_i(x621);

// val X182 = Convolv(1,1)(X181,cv73_W,cv73_B)
JCudaTensor x622;
JCudaTensor x623, x624, x625;
x623 = x563;
x624 = x626;
x625 = x627;
x622 = x237.forward(x623, x624, x625);

// b1fc1_B <~~ V_b1fc1_B
float x628, x629;
x628 = 1;
x629 = 1;
JCudaTensor x630;
x630 = x605;
x333.update(x630, x628, x629);

// b1fc1_W <~~ V_b1fc1_W
float x631, x632;
x631 = 1;
x632 = decay_1;
JCudaTensor x633;
x633 = x592;
x324.update(x633, x631, x632);

// val X187 = ReLU()(X186)
JCudaTensor x634;
JCudaTensor x635;
x635 = x613;
x634 = x246.forward(x635);

// val X183 = ReLU()(X182)
JCudaTensor x636;
JCudaTensor x637;
x637 = x622;
x636 = x243.forward(x637);

// val X190 = ReLU()(X189)
JCudaTensor x638;
JCudaTensor x639;
x639 = x579;
x638 = x246.forward(x639);

// V_b1cv_B <~~ X356
float x641, x642;
x641 = lrn_rate_2;
x642 = momentum;
JCudaTensor x643;
x643 = x610;
x640.update(x643, x641, x642);

// Dealloc(X356)
JCudaTensor x644;
x644 = x610;
x644.free();

// V_b1cv_W <~~ X459
float x646, x647;
x646 = lrn_rate_1;
x647 = momentum;
JCudaTensor x648;
x648 = x619;
x645.update(x648, x646, x647);

// Dealloc(X459)
JCudaTensor x649;
x649 = x619;
x649.free();

// val X231 = (X230[1><3])(i | @) * (b2fc1_W)(j | @)
JCudaTensor x650;
JCudaMatrix x651;
JCudaMatrix x652;
JCudaTensor x653;
JCudaTensor x654;
x654 = x597;
x653 = x654.flatten(1, new int[]{128, 4, 4});
x651 = x653.asMatrix(1, true);
JCudaTensor x655;
x655 = x656;
x652 = x655.asMatrix(1, true);
x650 = x651.times(x652);

// val X179 = ReLU()(X178)
JCudaTensor x657;
JCudaTensor x658;
x658 = x538;
x657 = x240.forward(x658);

// b1cv_B <~~ V_b1cv_B
float x659, x660;
x659 = 1;
x660 = 1;
JCudaTensor x661;
x661 = x640;
x285.update(x661, x659, x660);

// b1cv_W <~~ V_b1cv_W
float x662, x663;
x662 = 1;
x663 = decay_1;
JCudaTensor x664;
x664 = x645;
x284.update(x664, x662, x663);

// val X233 = (X231 + (i) => b2fc1_B)
JCudaTensor x665;
JCudaTensor x666, x667;
x666 = x650;
x667 = x668;
x665 = x667.copy(128, x666);

// val X191 = Concat(X179,X183,X187,X190)
JCudaTensor x669;
JCudaTensor x670, x671, x672, x673;
x670 = x657;
x671 = x636;
x672 = x634;
x673 = x638;
x669 = x250.forward(x670,x671,x672,x673);

// val X192 = Pooling(3,2,1,true)(X191)
JCudaTensor x674;
JCudaTensor x675;
x675 = x669;
x674 = x676.forward(x675);

// val X234 = ReLU()(X233)
JCudaTensor x677;
JCudaTensor x678;
x678 = x665;
x677 = x342.forward(x678);

// val X193 = Convolv(1,0)(X192,cv81_W,cv81_B)
JCudaTensor x679;
JCudaTensor x680, x681, x682;
x680 = x674;
x681 = x683;
x682 = x684;
x679 = x685.forward(x680, x681, x682);

// val X203 = Pooling(3,1,1,true)(X192)
JCudaTensor x686;
JCudaTensor x687;
x687 = x674;
x686 = x688.forward(x687);

// val X235 = Dropout(0.7)(X234)
JCudaTensor x689;
JCudaTensor x690;
x690 = x677;
x689 = x361.forward(x690);

// val X195 = Convolv(1,0)(X192,cv82_W,cv82_B)
JCudaTensor x691;
JCudaTensor x692, x693, x694;
x692 = x674;
x693 = x695;
x694 = x696;
x691 = x697.forward(x692, x693, x694);

// val X199 = Convolv(1,0)(X192,cv84_W,cv84_B)
JCudaTensor x698;
JCudaTensor x699, x700, x701;
x699 = x674;
x700 = x702;
x701 = x703;
x698 = x704.forward(x699, x700, x701);

// val X204 = Convolv(1,0)(X203,cv86_W,cv86_B)
JCudaTensor x705;
JCudaTensor x706, x707, x708;
x706 = x686;
x707 = x709;
x708 = x710;
x705 = x711.forward(x706, x707, x708);

// val X196 = ReLU()(X195)
JCudaTensor x712;
JCudaTensor x713;
x713 = x691;
x712 = x714.forward(x713);

// val X200 = ReLU()(X199)
JCudaTensor x715;
JCudaTensor x716;
x716 = x698;
x715 = x717.forward(x716);

// val X236 = (X235)(i | @) * (b2fc2_W)(j | @)
JCudaTensor x718;
JCudaMatrix x719;
JCudaMatrix x720;
JCudaTensor x721;
x721 = x689;
x719 = x721.asMatrix(1, true);
JCudaTensor x722;
x722 = x723;
x720 = x722.asMatrix(1, true);
x718 = x719.times(x720);

// val X238 = (X236 + (i) => b2fc2_B)
JCudaTensor x724;
JCudaTensor x725, x726;
x725 = x718;
x726 = x727;
x724 = x726.copy(128, x725);

// val X197 = Convolv(1,1)(X196,cv83_W,cv83_B)
JCudaTensor x728;
JCudaTensor x729, x730, x731;
x729 = x712;
x730 = x732;
x731 = x733;
x728 = x734.forward(x729, x730, x731);

// val X201 = Convolv(1,2)(X200,cv85_W,cv85_B)
JCudaTensor x735;
JCudaTensor x736, x737, x738;
x736 = x715;
x737 = x739;
x738 = x740;
x735 = x741.forward(x736, x737, x738);

// val X239 = LogSoftmax()(X238)
JCudaTensor x742;
JCudaTensor x743;
x743 = x724;
x742 = x407.forward(x743);

// Dealloc(X238)
JCudaTensor x744;
x744 = x724;
x744.free();

// val X205 = ReLU()(X204)
JCudaTensor x745;
JCudaTensor x746;
x746 = x705;
x745 = x747.forward(x746);

// val X198 = ReLU()(X197)
JCudaTensor x748;
JCudaTensor x749;
x749 = x728;
x748 = x750.forward(x749);

// val X202 = ReLU()(X201)
JCudaTensor x751;
JCudaTensor x752;
x752 = x735;
x751 = x747.forward(x752);

// val X194 = ReLU()(X193)
JCudaTensor x753;
JCudaTensor x754;
x754 = x679;
x753 = x755.forward(x754);

// val X961 = X337 * d_LogSoftmax()(X239)/d_X238
JCudaTensor x756;
JCudaTensor x757, x758;
x757 = x409;
x758 = x742;
x756 = x407.backward(x757, x758);

// val X206 = Concat(X194,X198,X202,X205)
JCudaTensor x759;
JCudaTensor x761, x762, x763, x764;
x761 = x753;
x762 = x748;
x763 = x751;
x764 = x745;
x759 = x760.forward(x761,x762,x763,x764);

// val m13 = (i133) => b2fc2_W[@, i133]
JCudaMatrix x765;
JCudaTensor x766;
x766 = x723;
x765 = x766.asMatrix(1, false);

// val m22 = (i221) => X961[@, i221]
JCudaMatrix x767;
JCudaTensor x768;
x768 = x756;
x767 = x768.asMatrix(1, false);

// val m24 = (i229) => X235[@, i229]
JCudaMatrix x769;
JCudaTensor x770;
x770 = x689;
x769 = x770.asMatrix(1, false);

// val X209 = Convolv(1,0)(X206,cv92_W,cv92_B)
JCudaTensor x771;
JCudaTensor x772, x773, x774;
x772 = x759;
x773 = x775;
x774 = x776;
x771 = x697.forward(x772, x773, x774);

// val X207 = Convolv(1,0)(X206,cv91_W,cv91_B)
JCudaTensor x777;
JCudaTensor x778, x779, x780;
x778 = x759;
x779 = x781;
x780 = x782;
x777 = x685.forward(x778, x779, x780);

// val X213 = Convolv(1,0)(X206,cv94_W,cv94_B)
JCudaTensor x783;
JCudaTensor x784, x785, x786;
x784 = x759;
x785 = x787;
x786 = x788;
x783 = x704.forward(x784, x785, x786);

// val X970 = (X961)(i132 | @) * m13
JCudaTensor x789;
JCudaMatrix x790;
JCudaMatrix x791;
JCudaTensor x792;
x792 = x756;
x790 = x792.asMatrix(1, true);
x791 = x765;
x789 = x790.times(x791);

// val X217 = Pooling(3,1,1,true)(X206)
JCudaTensor x793;
JCudaTensor x794;
x794 = x759;
x793 = x688.forward(x794);

// val X214 = ReLU()(X213)
JCudaTensor x795;
JCudaTensor x796;
x796 = x783;
x795 = x717.forward(x796);

// val X218 = Convolv(1,0)(X217,cv96_W,cv96_B)
JCudaTensor x797;
JCudaTensor x798, x799, x800;
x798 = x793;
x799 = x801;
x800 = x802;
x797 = x711.forward(x798, x799, x800);

// val X1532 = Sum(m22)
JCudaTensor x803;
JCudaMatrix x804;
x804 = x767;
x803 = x804.sum();

// val X971 = X970 * d_Dropout(0.7)()/d_X234
JCudaTensor x805;
JCudaTensor x806;
x806 = x789;
x805 = x361.backward(x806);

// Dealloc(X970)
JCudaTensor x807;
x807 = x789;
x807.free();

// val X210 = ReLU()(X209)
JCudaTensor x808;
JCudaTensor x809;
x809 = x771;
x808 = x714.forward(x809);

// val X1664 = m22 * m24
JCudaTensor x810;
JCudaMatrix x811;
JCudaMatrix x812;
x811 = x767;
x812 = x769;
x810 = x811.times(x812);

// Dealloc(X961)
JCudaTensor x813;
x813 = x756;
x813.free();

// Dealloc(X235)
JCudaTensor x814;
x814 = x689;
x814.free();

// val X211 = Convolv(1,1)(X210,cv93_W,cv93_B)
JCudaTensor x815;
JCudaTensor x816, x817, x818;
x816 = x808;
x817 = x819;
x818 = x820;
x815 = x734.forward(x816, x817, x818);

// val X1665 = (X1664 * loss2)
JCudaTensor x821;
JCudaTensor x822;
float x823;
x822 = x810;
x823 = loss2;
x821 = x822.times_i(x823);

// val X215 = Convolv(1,2)(X214,cv95_W,cv95_B)
JCudaTensor x824;
JCudaTensor x825, x826, x827;
x825 = x795;
x826 = x828;
x827 = x829;
x824 = x741.forward(x825, x826, x827);

// val m14 = (i137) => b2fc1_W[@, i137]
JCudaMatrix x830;
JCudaTensor x831;
x831 = x656;
x830 = x831.asMatrix(1, false);

// val X973 = X971 * d_ReLU()(X234)/d_X233
JCudaTensor x832;
JCudaTensor x833, x834;
x833 = x805;
x834 = x677;
x832 = x342.backward(x833, x834);

// Dealloc(X234)
JCudaTensor x835;
x835 = x677;
x835.free();

// val X1533 = (X1532 * loss2)
JCudaTensor x836;
JCudaTensor x837;
float x838;
x837 = x803;
x838 = loss2;
x836 = x837.times_i(x838);

// val m18 = (i184) => X973[@, i184]
JCudaMatrix x839;
JCudaTensor x840;
x840 = x832;
x839 = x840.asMatrix(1, false);

// val X974 = (X973)(i136 | @) * m14
JCudaTensor x841;
JCudaMatrix x842;
JCudaMatrix x843;
JCudaTensor x844;
x844 = x832;
x842 = x844.asMatrix(1, true);
x843 = x830;
x841 = x842.times(x843);

// V_b2fc2_B <~~ X1533
float x846, x847;
x846 = lrn_rate_2;
x847 = momentum;
JCudaTensor x848;
x848 = x836;
x845.update(x848, x846, x847);

// Dealloc(X1533)
JCudaTensor x849;
x849 = x836;
x849.free();

// val X208 = ReLU()(X207)
JCudaTensor x850;
JCudaTensor x851;
x851 = x777;
x850 = x755.forward(x851);

// val X212 = ReLU()(X211)
JCudaTensor x852;
JCudaTensor x853;
x853 = x815;
x852 = x750.forward(x853);

// val m21 = (i202) => X230[1><3][@, i202]
JCudaMatrix x854;
JCudaTensor x855;
JCudaTensor x856;
x856 = x597;
x855 = x856.flatten(1, new int[]{128, 4, 4});
x854 = x855.asMatrix(1, false);

// val X216 = ReLU()(X215)
JCudaTensor x857;
JCudaTensor x858;
x858 = x824;
x857 = x747.forward(x858);

// val X219 = ReLU()(X218)
JCudaTensor x859;
JCudaTensor x860;
x860 = x797;
x859 = x747.forward(x860);

// V_b2fc2_W <~~ X1665
float x862, x863;
x862 = lrn_rate_1;
x863 = momentum;
JCudaTensor x864;
x864 = x821;
x861.update(x864, x862, x863);

// Dealloc(X1665)
JCudaTensor x865;
x865 = x821;
x865.free();

// b2fc2_B <~~ V_b2fc2_B
float x866, x867;
x866 = 1;
x867 = 1;
JCudaTensor x868;
x868 = x845;
x727.update(x868, x866, x867);

// b2fc2_W <~~ V_b2fc2_W
float x869, x870;
x869 = 1;
x870 = decay_1;
JCudaTensor x871;
x871 = x861;
x723.update(x871, x869, x870);

// val X976 = X974[1<>3] * d_ReLU()(X230)/d_X229
JCudaTensor x872;
JCudaTensor x873, x874;
JCudaTensor x875;
x875 = x841;
x873 = x875.unflatten(1, new int[]{128, 4, 4});
x874 = x597;
x872 = x303.backward(x873, x874);

// val X1261 = Sum(m18)
JCudaTensor x876;
JCudaMatrix x877;
x877 = x839;
x876 = x877.sum();

// val X220 = Concat(X208,X212,X216,X219)
JCudaTensor x878;
JCudaTensor x879, x880, x881, x882;
x879 = x850;
x880 = x852;
x881 = x857;
x882 = x859;
x878 = x760.forward(x879,x880,x881,x882);

// val X1400 = m18 * m21
JCudaTensor x883;
JCudaMatrix x884;
JCudaMatrix x885;
x884 = x839;
x885 = x854;
x883 = x884.times(x885);

// Dealloc(X973)
JCudaTensor x886;
x886 = x832;
x886.free();

// Dealloc(X230)
JCudaTensor x887;
x887 = x597;
x887.free();

// val X977 = X976 * d_Convolv(1,0)()/d_b2cv_B
JCudaTensor x888;
JCudaTensor x889;
x889 = x872;
x888 = x286.backward_bias(x889);

// val X1262 = (X1261 * loss2)
JCudaTensor x890;
JCudaTensor x891;
float x892;
x891 = x876;
x892 = loss2;
x890 = x891.times_i(x892);

// val X1122 = X976 * d_Convolv(1,0)(X228)/d_b2cv_W
JCudaTensor x893;
JCudaTensor x894, x895;
x894 = x872;
x895 = x530;
x893 = x286.backward_filter(x894, x895);

// val X2481 = X976 * d_Convolv(1,0)(b2cv_W)/d_X228
JCudaTensor x896;
JCudaTensor x897, x898;
x897 = x872;
x898 = x569;
x896 = x286.backward_data(x897, x898);

// Dealloc(X976)
JCudaTensor x899;
x899 = x872;
x899.free();

// val X221 = Pooling(7,1,0,false)(X220)
JCudaTensor x900;
JCudaTensor x901;
x901 = x878;
x900 = x902.forward(x901);

// val X1401 = (X1400 * loss2)
JCudaTensor x903;
JCudaTensor x904;
float x905;
x904 = x883;
x905 = loss2;
x903 = x904.times_i(x905);

// V_b2fc1_B <~~ X1262
float x907, x908;
x907 = lrn_rate_2;
x908 = momentum;
JCudaTensor x909;
x909 = x890;
x906.update(x909, x907, x908);

// Dealloc(X1262)
JCudaTensor x910;
x910 = x890;
x910.free();

// V_b2fc1_W <~~ X1401
float x912, x913;
x912 = lrn_rate_1;
x913 = momentum;
JCudaTensor x914;
x914 = x903;
x911.update(x914, x912, x913);

// Dealloc(X1401)
JCudaTensor x915;
x915 = x903;
x915.free();

// val X2483 = X2481 * d_Pooling(5,3,0,false)(X228,X177)/d_X177
JCudaTensor x916;
JCudaTensor x917, x918, x919;
x917 = x896;
x918 = x530;
x919 = x503;
x916 = x257.backward(x917, x918, x919);

// Dealloc(X2481)
JCudaTensor x920;
x920 = x896;
x920.free();

// Dealloc(X228)
JCudaTensor x921;
x921 = x530;
x921.free();

// val X978 = (X977 * loss2)
JCudaTensor x922;
JCudaTensor x923;
float x924;
x923 = x888;
x924 = loss2;
x922 = x923.times_i(x924);

// val X1123 = (X1122 * loss2)
JCudaTensor x925;
JCudaTensor x926;
float x927;
x926 = x893;
x927 = loss2;
x925 = x926.times_i(x927);

// val X222 = Dropout(0.4)(X221)
JCudaTensor x928;
JCudaTensor x929;
x929 = x900;
x928 = x930.forward(x929);

// b2fc1_W <~~ V_b2fc1_W
float x931, x932;
x931 = 1;
x932 = decay_1;
JCudaTensor x933;
x933 = x911;
x656.update(x933, x931, x932);

// b2fc1_B <~~ V_b2fc1_B
float x934, x935;
x934 = 1;
x935 = 1;
JCudaTensor x936;
x936 = x906;
x668.update(x936, x934, x935);

// V_b2cv_W <~~ X1123
float x938, x939;
x938 = lrn_rate_1;
x939 = momentum;
JCudaTensor x940;
x940 = x925;
x937.update(x940, x938, x939);

// Dealloc(X1123)
JCudaTensor x941;
x941 = x925;
x941.free();

// val X223 = (X222[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x942;
JCudaMatrix x943;
JCudaMatrix x944;
JCudaTensor x945;
JCudaTensor x946;
x946 = x928;
x945 = x946.flatten(1, new int[]{256, 1, 1});
x943 = x945.asMatrix(1, true);
JCudaTensor x947;
x947 = x948;
x944 = x947.asMatrix(1, true);
x942 = x943.times(x944);

// V_b2cv_B <~~ X978
float x950, x951;
x950 = lrn_rate_2;
x951 = momentum;
JCudaTensor x952;
x952 = x922;
x949.update(x952, x950, x951);

// Dealloc(X978)
JCudaTensor x953;
x953 = x922;
x953.free();

// b2cv_W <~~ V_b2cv_W
float x954, x955;
x954 = 1;
x955 = decay_1;
JCudaTensor x956;
x956 = x937;
x569.update(x956, x954, x955);

// b2cv_B <~~ V_b2cv_B
float x957, x958;
x957 = 1;
x958 = 1;
JCudaTensor x959;
x959 = x949;
x570.update(x959, x957, x958);

// val X225 = (X223 + (i) => fc_B)
JCudaTensor x960;
JCudaTensor x961, x962;
x961 = x942;
x962 = x963;
x960 = x962.copy(128, x961);

// val X226 = LogSoftmax()(X225)
JCudaTensor x964;
JCudaTensor x965;
x965 = x960;
x964 = x407.forward(x965);

// Dealloc(X225)
JCudaTensor x966;
x966 = x960;
x966.free();

// Print(((((0 - (X227 . X226)) / |128|) + (((0 - (X227 . X239)) / |128|) * loss2)) + (((0 - (X227 . X252)) / |128|) * loss1)))
float x967;
float x968;
float x969;
float x970;
float x971;
float x972;
float x973;
float x974;
JCudaTensor x975, x976;
x975 = x9;
x976 = x964;
x974 = x975.dot(x976);
x972 = - x974;
x973 = 128;
x970 = x972 / x973;
float x977;
float x978;
float x979;
float x980;
float x981;
JCudaTensor x982, x983;
x982 = x9;
x983 = x742;
x981 = x982.dot(x983);
x979 = - x981;
x980 = 128;
x977 = x979 / x980;
x978 = loss2;
x971 = x977 * x978;
x968 = x970 + x971;
float x984;
float x985;
float x986;
float x987;
float x988;
JCudaTensor x989, x990;
x989 = x9;
x990 = x405;
x988 = x989.dot(x990);
x986 = - x988;
x987 = 128;
x984 = x986 / x987;
x985 = loss1;
x969 = x984 * x985;
x967 = x968 + x969;
System.out.println(x5 + " " + x967);
if (Float.isNaN(x967)) { System.exit(-1); }

// Dealloc(X239)
JCudaTensor x991;
x991 = x742;
x991.free();

// Dealloc(X252)
JCudaTensor x992;
x992 = x405;
x992.free();

// Dealloc(X227)
JCudaTensor x993;
x993 = x9;
x993.free();

// val X2133 = X337 * d_LogSoftmax()(X226)/d_X225
JCudaTensor x994;
JCudaTensor x995, x996;
x995 = x409;
x996 = x964;
x994 = x407.backward(x995, x996);

// Dealloc(X226)
JCudaTensor x997;
x997 = x964;
x997.free();

// Dealloc(X337)
JCudaTensor x998;
x998 = x409;
x998.free();

// val m25 = (i243) => fc_W[@, i243]
JCudaMatrix x999;
JCudaTensor x1000;
x1000 = x948;
x999 = x1000.asMatrix(1, false);

// val m379 = (i31858) => X2133[@, i31858]
JCudaMatrix x1001;
JCudaTensor x1002;
x1002 = x994;
x1001 = x1002.asMatrix(1, false);

// val X2179 = (X2133)(i242 | @) * m25
JCudaTensor x1003;
JCudaMatrix x1004;
JCudaMatrix x1005;
JCudaTensor x1006;
x1006 = x994;
x1004 = x1006.asMatrix(1, true);
x1005 = x999;
x1003 = x1004.times(x1005);

// val m381 = (i31866) => X222[1><3][@, i31866]
JCudaMatrix x1007;
JCudaTensor x1008;
JCudaTensor x1009;
x1009 = x928;
x1008 = x1009.flatten(1, new int[]{256, 1, 1});
x1007 = x1008.asMatrix(1, false);

// val X2180 = X2179[1<>3] * d_Dropout(0.4)()/d_X221
JCudaTensor x1010;
JCudaTensor x1011;
JCudaTensor x1012;
x1012 = x1003;
x1011 = x1012.unflatten(1, new int[]{256, 1, 1});
x1010 = x930.backward(x1011);

// Dealloc(X2179)
JCudaTensor x1013;
x1013 = x1003;
x1013.free();

// V_fc_W <~~ m379 * m381
float x1015, x1016;
x1015 = lrn_rate_1;
x1016 = momentum;
JCudaMatrix x1017;
JCudaMatrix x1018;
x1017 = x1001;
x1018 = x1007;
x1017.times(x1018, x1014, x1015, x1016);

// Dealloc(X222)
JCudaTensor x1019;
x1019 = x928;
x1019.free();

// V_fc_B <~~ Sum(m379)
float x1021, x1022;
x1021 = lrn_rate_2;
x1022 = momentum;
JCudaMatrix x1023;
x1023 = x1001;
x1023.sum(x1020, x1021, x1022);

// Dealloc(X2133)
JCudaTensor x1024;
x1024 = x994;
x1024.free();

// fc_W <~~ V_fc_W
float x1025, x1026;
x1025 = 1;
x1026 = decay_1;
JCudaTensor x1027;
x1027 = x1014;
x948.update(x1027, x1025, x1026);

// fc_B <~~ V_fc_B
float x1028, x1029;
x1028 = 1;
x1029 = 1;
JCudaTensor x1030;
x1030 = x1020;
x963.update(x1030, x1028, x1029);

// val X2182 = X2180 * d_Pooling(7,1,0,false)(X221,X220)/d_X220
JCudaTensor x1031;
JCudaTensor x1032, x1033, x1034;
x1032 = x1010;
x1033 = x900;
x1034 = x878;
x1031 = x902.backward(x1032, x1033, x1034);

// Dealloc(X2180)
JCudaTensor x1035;
x1035 = x1010;
x1035.free();

// Dealloc(X221)
JCudaTensor x1036;
x1036 = x900;
x1036.free();

// Dealloc(X220)
JCudaTensor x1037;
x1037 = x878;
x1037.free();

// val X2262 = Proj(X2182, X208,X212,X216,X219, 3)
JCudaTensor x1038;
JCudaTensor x1040;
x1040 = x1031;
JCudaTensor[] x1039 = x760.backward(x1040);
x1038 = x1039[3];

// val X2238 = Proj(X2182, X208,X212,X216,X219, 2)
JCudaTensor x1041;
x1041 = x1039[2];

// val X2214 = Proj(X2182, X208,X212,X216,X219, 1)
JCudaTensor x1042;
x1042 = x1039[1];

// val X2196 = Proj(X2182, X208,X212,X216,X219, 0)
JCudaTensor x1043;
x1043 = x1039[0];

// Dealloc(X2182)
JCudaTensor x1044;
x1044 = x1031;
x1044.free();

// val X2243 = X2238 * d_ReLU()(X216)/d_X215
JCudaTensor x1045;
JCudaTensor x1046, x1047;
x1046 = x1041;
x1047 = x857;
x1045 = x747.backward(x1046, x1047);

// Dealloc(X216)
JCudaTensor x1048;
x1048 = x857;
x1048.free();

// val X2266 = X2262 * d_ReLU()(X219)/d_X218
JCudaTensor x1049;
JCudaTensor x1050, x1051;
x1050 = x1038;
x1051 = x859;
x1049 = x747.backward(x1050, x1051);

// Dealloc(X219)
JCudaTensor x1052;
x1052 = x859;
x1052.free();

// val X2199 = X2196 * d_ReLU()(X208)/d_X207
JCudaTensor x1053;
JCudaTensor x1054, x1055;
x1054 = x1043;
x1055 = x850;
x1053 = x755.backward(x1054, x1055);

// Dealloc(X208)
JCudaTensor x1056;
x1056 = x850;
x1056.free();

// val X2219 = X2214 * d_ReLU()(X212)/d_X211
JCudaTensor x1057;
JCudaTensor x1058, x1059;
x1058 = x1042;
x1059 = x852;
x1057 = x750.backward(x1058, x1059);

// Dealloc(X212)
JCudaTensor x1060;
x1060 = x852;
x1060.free();

// val X2244 = X2243 * d_Convolv(1,2)(cv95_W)/d_X214
JCudaTensor x1061;
JCudaTensor x1062, x1063;
x1062 = x1045;
x1063 = x828;
x1061 = x741.backward_data(x1062, x1063);

// V_cv96_B <~~ X2266 * d_Convolv(1,0)()/d_cv96_B
float x1065, x1066;
x1065 = lrn_rate_2;
x1066 = momentum;
JCudaTensor x1067;
x1067 = x1049;
x711.backward_bias(x1067, x1064, x1065, x1066);

// V_cv91_W <~~ X2199 * d_Convolv(1,0)(X206)/d_cv91_W
float x1069, x1070;
x1069 = lrn_rate_1;
x1070 = momentum;
JCudaTensor x1071, x1072;
x1071 = x1053;
x1072 = x759;
x685.backward_filter(x1071, x1072, x1068, x1069, x1070);

// val X2220 = X2219 * d_Convolv(1,1)(cv93_W)/d_X210
JCudaTensor x1073;
JCudaTensor x1074, x1075;
x1074 = x1057;
x1075 = x819;
x1073 = x734.backward_data(x1074, x1075);

// val X2200 = X2199 * d_Convolv(1,0)(cv91_W)/d_X206
JCudaTensor x1076;
JCudaTensor x1077, x1078;
x1077 = x1053;
x1078 = x781;
x1076 = x685.backward_data(x1077, x1078);

// V_cv96_W <~~ X2266 * d_Convolv(1,0)(X217)/d_cv96_W
float x1080, x1081;
x1080 = lrn_rate_1;
x1081 = momentum;
JCudaTensor x1082, x1083;
x1082 = x1049;
x1083 = x793;
x711.backward_filter(x1082, x1083, x1079, x1080, x1081);

// V_cv91_B <~~ X2199 * d_Convolv(1,0)()/d_cv91_B
float x1085, x1086;
x1085 = lrn_rate_2;
x1086 = momentum;
JCudaTensor x1087;
x1087 = x1053;
x685.backward_bias(x1087, x1084, x1085, x1086);

// Dealloc(X2199)
JCudaTensor x1088;
x1088 = x1053;
x1088.free();

// val X2267 = X2266 * d_Convolv(1,0)(cv96_W)/d_X217
JCudaTensor x1089;
JCudaTensor x1090, x1091;
x1090 = x1049;
x1091 = x801;
x1089 = x711.backward_data(x1090, x1091);

// Dealloc(X2266)
JCudaTensor x1092;
x1092 = x1049;
x1092.free();

// V_cv95_B <~~ X2243 * d_Convolv(1,2)()/d_cv95_B
float x1094, x1095;
x1094 = lrn_rate_2;
x1095 = momentum;
JCudaTensor x1096;
x1096 = x1045;
x741.backward_bias(x1096, x1093, x1094, x1095);

// V_cv93_W <~~ X2219 * d_Convolv(1,1)(X210)/d_cv93_W
float x1098, x1099;
x1098 = lrn_rate_1;
x1099 = momentum;
JCudaTensor x1100, x1101;
x1100 = x1057;
x1101 = x808;
x734.backward_filter(x1100, x1101, x1097, x1098, x1099);

// V_cv95_W <~~ X2243 * d_Convolv(1,2)(X214)/d_cv95_W
float x1103, x1104;
x1103 = lrn_rate_1;
x1104 = momentum;
JCudaTensor x1105, x1106;
x1105 = x1045;
x1106 = x795;
x741.backward_filter(x1105, x1106, x1102, x1103, x1104);

// Dealloc(X2243)
JCudaTensor x1107;
x1107 = x1045;
x1107.free();

// V_cv93_B <~~ X2219 * d_Convolv(1,1)()/d_cv93_B
float x1109, x1110;
x1109 = lrn_rate_2;
x1110 = momentum;
JCudaTensor x1111;
x1111 = x1057;
x734.backward_bias(x1111, x1108, x1109, x1110);

// Dealloc(X2219)
JCudaTensor x1112;
x1112 = x1057;
x1112.free();

// cv93_W <~~ V_cv93_W
float x1113, x1114;
x1113 = 1;
x1114 = decay_1;
JCudaTensor x1115;
x1115 = x1097;
x819.update(x1115, x1113, x1114);

// cv91_W <~~ V_cv91_W
float x1116, x1117;
x1116 = 1;
x1117 = decay_1;
JCudaTensor x1118;
x1118 = x1068;
x781.update(x1118, x1116, x1117);

// cv91_B <~~ V_cv91_B
float x1119, x1120;
x1119 = 1;
x1120 = 1;
JCudaTensor x1121;
x1121 = x1084;
x782.update(x1121, x1119, x1120);

// cv96_W <~~ V_cv96_W
float x1122, x1123;
x1122 = 1;
x1123 = decay_1;
JCudaTensor x1124;
x1124 = x1079;
x801.update(x1124, x1122, x1123);

// cv96_B <~~ V_cv96_B
float x1125, x1126;
x1125 = 1;
x1126 = 1;
JCudaTensor x1127;
x1127 = x1064;
x802.update(x1127, x1125, x1126);

// cv95_B <~~ V_cv95_B
float x1128, x1129;
x1128 = 1;
x1129 = 1;
JCudaTensor x1130;
x1130 = x1093;
x829.update(x1130, x1128, x1129);

// cv95_W <~~ V_cv95_W
float x1131, x1132;
x1131 = 1;
x1132 = decay_1;
JCudaTensor x1133;
x1133 = x1102;
x828.update(x1133, x1131, x1132);

// cv93_B <~~ V_cv93_B
float x1134, x1135;
x1134 = 1;
x1135 = 1;
JCudaTensor x1136;
x1136 = x1108;
x820.update(x1136, x1134, x1135);

// val X2222 = X2220 * d_ReLU()(X210)/d_X209
JCudaTensor x1137;
JCudaTensor x1138, x1139;
x1138 = x1073;
x1139 = x808;
x1137 = x714.backward(x1138, x1139);

// Dealloc(X210)
JCudaTensor x1140;
x1140 = x808;
x1140.free();

// val X2246 = X2244 * d_ReLU()(X214)/d_X213
JCudaTensor x1141;
JCudaTensor x1142, x1143;
x1142 = x1061;
x1143 = x795;
x1141 = x717.backward(x1142, x1143);

// Dealloc(X214)
JCudaTensor x1144;
x1144 = x795;
x1144.free();

// V_cv94_W <~~ X2246 * d_Convolv(1,0)(X206)/d_cv94_W
float x1146, x1147;
x1146 = lrn_rate_1;
x1147 = momentum;
JCudaTensor x1148, x1149;
x1148 = x1141;
x1149 = x759;
x704.backward_filter(x1148, x1149, x1145, x1146, x1147);

// V_cv94_B <~~ X2246 * d_Convolv(1,0)()/d_cv94_B
float x1151, x1152;
x1151 = lrn_rate_2;
x1152 = momentum;
JCudaTensor x1153;
x1153 = x1141;
x704.backward_bias(x1153, x1150, x1151, x1152);

// val X2224 = (X2200 + X2222 * d_Convolv(1,0)(cv92_W)/d_X206)
JCudaTensor x1154;
JCudaTensor x1155;
x1155 = x1076;
JCudaTensor x1156, x1157;
x1156 = x1137;
x1157 = x775;
x1154 = x697.backward_data(x1156,x1157, x1155);

// V_cv92_W <~~ X2222 * d_Convolv(1,0)(X206)/d_cv92_W
float x1159, x1160;
x1159 = lrn_rate_1;
x1160 = momentum;
JCudaTensor x1161, x1162;
x1161 = x1137;
x1162 = x759;
x697.backward_filter(x1161, x1162, x1158, x1159, x1160);

// V_cv92_B <~~ X2222 * d_Convolv(1,0)()/d_cv92_B
float x1164, x1165;
x1164 = lrn_rate_2;
x1165 = momentum;
JCudaTensor x1166;
x1166 = x1137;
x697.backward_bias(x1166, x1163, x1164, x1165);

// Dealloc(X2222)
JCudaTensor x1167;
x1167 = x1137;
x1167.free();

// cv94_B <~~ V_cv94_B
float x1168, x1169;
x1168 = 1;
x1169 = 1;
JCudaTensor x1170;
x1170 = x1150;
x788.update(x1170, x1168, x1169);

// cv92_W <~~ V_cv92_W
float x1171, x1172;
x1171 = 1;
x1172 = decay_1;
JCudaTensor x1173;
x1173 = x1158;
x775.update(x1173, x1171, x1172);

// cv92_B <~~ V_cv92_B
float x1174, x1175;
x1174 = 1;
x1175 = 1;
JCudaTensor x1176;
x1176 = x1163;
x776.update(x1176, x1174, x1175);

// val X2248 = (X2224 + X2246 * d_Convolv(1,0)(cv94_W)/d_X206)
JCudaTensor x1177;
JCudaTensor x1178;
x1178 = x1154;
JCudaTensor x1179, x1180;
x1179 = x1141;
x1180 = x787;
x1177 = x704.backward_data(x1179,x1180, x1178);

// Dealloc(X2246)
JCudaTensor x1181;
x1181 = x1141;
x1181.free();

// cv94_W <~~ V_cv94_W
float x1182, x1183;
x1182 = 1;
x1183 = decay_1;
JCudaTensor x1184;
x1184 = x1145;
x787.update(x1184, x1182, x1183);

// val X2270 = (X2248 + X2267 * d_Pooling(3,1,1,true)(X217,X206)/d_X206)
JCudaTensor x1185;
JCudaTensor x1186;
x1186 = x1177;
JCudaTensor x1187, x1188, x1189;
x1187 = x1089;
x1188 = x793;
x1189 = x759;
x1185 = x688.backward(x1187,x1188,x1189, x1186);

// Dealloc(X2267)
JCudaTensor x1190;
x1190 = x1089;
x1190.free();

// Dealloc(X217)
JCudaTensor x1191;
x1191 = x793;
x1191.free();

// Dealloc(X206)
JCudaTensor x1192;
x1192 = x759;
x1192.free();

// val X2350 = Proj(X2270, X194,X198,X202,X205, 3)
JCudaTensor x1193;
JCudaTensor x1195;
x1195 = x1185;
JCudaTensor[] x1194 = x760.backward(x1195);
x1193 = x1194[3];

// val X2326 = Proj(X2270, X194,X198,X202,X205, 2)
JCudaTensor x1196;
x1196 = x1194[2];

// val X2302 = Proj(X2270, X194,X198,X202,X205, 1)
JCudaTensor x1197;
x1197 = x1194[1];

// val X2284 = Proj(X2270, X194,X198,X202,X205, 0)
JCudaTensor x1198;
x1198 = x1194[0];

// Dealloc(X2270)
JCudaTensor x1199;
x1199 = x1185;
x1199.free();

// val X2287 = X2284 * d_ReLU()(X194)/d_X193
JCudaTensor x1200;
JCudaTensor x1201, x1202;
x1201 = x1198;
x1202 = x753;
x1200 = x755.backward(x1201, x1202);

// Dealloc(X194)
JCudaTensor x1203;
x1203 = x753;
x1203.free();

// val X2307 = X2302 * d_ReLU()(X198)/d_X197
JCudaTensor x1204;
JCudaTensor x1205, x1206;
x1205 = x1197;
x1206 = x748;
x1204 = x750.backward(x1205, x1206);

// Dealloc(X198)
JCudaTensor x1207;
x1207 = x748;
x1207.free();

// val X2331 = X2326 * d_ReLU()(X202)/d_X201
JCudaTensor x1208;
JCudaTensor x1209, x1210;
x1209 = x1196;
x1210 = x751;
x1208 = x747.backward(x1209, x1210);

// Dealloc(X202)
JCudaTensor x1211;
x1211 = x751;
x1211.free();

// val X2354 = X2350 * d_ReLU()(X205)/d_X204
JCudaTensor x1212;
JCudaTensor x1213, x1214;
x1213 = x1193;
x1214 = x745;
x1212 = x747.backward(x1213, x1214);

// Dealloc(X205)
JCudaTensor x1215;
x1215 = x745;
x1215.free();

// val X2288 = X2287 * d_Convolv(1,0)(cv81_W)/d_X192
JCudaTensor x1216;
JCudaTensor x1217, x1218;
x1217 = x1200;
x1218 = x683;
x1216 = x685.backward_data(x1217, x1218);

// V_cv81_B <~~ X2287 * d_Convolv(1,0)()/d_cv81_B
float x1220, x1221;
x1220 = lrn_rate_2;
x1221 = momentum;
JCudaTensor x1222;
x1222 = x1200;
x685.backward_bias(x1222, x1219, x1220, x1221);

// V_cv83_W <~~ X2307 * d_Convolv(1,1)(X196)/d_cv83_W
float x1224, x1225;
x1224 = lrn_rate_1;
x1225 = momentum;
JCudaTensor x1226, x1227;
x1226 = x1204;
x1227 = x712;
x734.backward_filter(x1226, x1227, x1223, x1224, x1225);

// val X2332 = X2331 * d_Convolv(1,2)(cv85_W)/d_X200
JCudaTensor x1228;
JCudaTensor x1229, x1230;
x1229 = x1208;
x1230 = x739;
x1228 = x741.backward_data(x1229, x1230);

// V_cv85_W <~~ X2331 * d_Convolv(1,2)(X200)/d_cv85_W
float x1232, x1233;
x1232 = lrn_rate_1;
x1233 = momentum;
JCudaTensor x1234, x1235;
x1234 = x1208;
x1235 = x715;
x741.backward_filter(x1234, x1235, x1231, x1232, x1233);

// val X2308 = X2307 * d_Convolv(1,1)(cv83_W)/d_X196
JCudaTensor x1236;
JCudaTensor x1237, x1238;
x1237 = x1204;
x1238 = x732;
x1236 = x734.backward_data(x1237, x1238);

// V_cv81_W <~~ X2287 * d_Convolv(1,0)(X192)/d_cv81_W
float x1240, x1241;
x1240 = lrn_rate_1;
x1241 = momentum;
JCudaTensor x1242, x1243;
x1242 = x1200;
x1243 = x674;
x685.backward_filter(x1242, x1243, x1239, x1240, x1241);

// Dealloc(X2287)
JCudaTensor x1244;
x1244 = x1200;
x1244.free();

// V_cv86_W <~~ X2354 * d_Convolv(1,0)(X203)/d_cv86_W
float x1246, x1247;
x1246 = lrn_rate_1;
x1247 = momentum;
JCudaTensor x1248, x1249;
x1248 = x1212;
x1249 = x686;
x711.backward_filter(x1248, x1249, x1245, x1246, x1247);

// V_cv83_B <~~ X2307 * d_Convolv(1,1)()/d_cv83_B
float x1251, x1252;
x1251 = lrn_rate_2;
x1252 = momentum;
JCudaTensor x1253;
x1253 = x1204;
x734.backward_bias(x1253, x1250, x1251, x1252);

// Dealloc(X2307)
JCudaTensor x1254;
x1254 = x1204;
x1254.free();

// V_cv85_B <~~ X2331 * d_Convolv(1,2)()/d_cv85_B
float x1256, x1257;
x1256 = lrn_rate_2;
x1257 = momentum;
JCudaTensor x1258;
x1258 = x1208;
x741.backward_bias(x1258, x1255, x1256, x1257);

// Dealloc(X2331)
JCudaTensor x1259;
x1259 = x1208;
x1259.free();

// V_cv86_B <~~ X2354 * d_Convolv(1,0)()/d_cv86_B
float x1261, x1262;
x1261 = lrn_rate_2;
x1262 = momentum;
JCudaTensor x1263;
x1263 = x1212;
x711.backward_bias(x1263, x1260, x1261, x1262);

// val X2355 = X2354 * d_Convolv(1,0)(cv86_W)/d_X203
JCudaTensor x1264;
JCudaTensor x1265, x1266;
x1265 = x1212;
x1266 = x709;
x1264 = x711.backward_data(x1265, x1266);

// Dealloc(X2354)
JCudaTensor x1267;
x1267 = x1212;
x1267.free();

// cv85_W <~~ V_cv85_W
float x1268, x1269;
x1268 = 1;
x1269 = decay_1;
JCudaTensor x1270;
x1270 = x1231;
x739.update(x1270, x1268, x1269);

// cv86_B <~~ V_cv86_B
float x1271, x1272;
x1271 = 1;
x1272 = 1;
JCudaTensor x1273;
x1273 = x1260;
x710.update(x1273, x1271, x1272);

// cv83_W <~~ V_cv83_W
float x1274, x1275;
x1274 = 1;
x1275 = decay_1;
JCudaTensor x1276;
x1276 = x1223;
x732.update(x1276, x1274, x1275);

// cv86_W <~~ V_cv86_W
float x1277, x1278;
x1277 = 1;
x1278 = decay_1;
JCudaTensor x1279;
x1279 = x1245;
x709.update(x1279, x1277, x1278);

// cv83_B <~~ V_cv83_B
float x1280, x1281;
x1280 = 1;
x1281 = 1;
JCudaTensor x1282;
x1282 = x1250;
x733.update(x1282, x1280, x1281);

// cv85_B <~~ V_cv85_B
float x1283, x1284;
x1283 = 1;
x1284 = 1;
JCudaTensor x1285;
x1285 = x1255;
x740.update(x1285, x1283, x1284);

// cv81_W <~~ V_cv81_W
float x1286, x1287;
x1286 = 1;
x1287 = decay_1;
JCudaTensor x1288;
x1288 = x1239;
x683.update(x1288, x1286, x1287);

// cv81_B <~~ V_cv81_B
float x1289, x1290;
x1289 = 1;
x1290 = 1;
JCudaTensor x1291;
x1291 = x1219;
x684.update(x1291, x1289, x1290);

// val X2334 = X2332 * d_ReLU()(X200)/d_X199
JCudaTensor x1292;
JCudaTensor x1293, x1294;
x1293 = x1228;
x1294 = x715;
x1292 = x717.backward(x1293, x1294);

// Dealloc(X200)
JCudaTensor x1295;
x1295 = x715;
x1295.free();

// val X2310 = X2308 * d_ReLU()(X196)/d_X195
JCudaTensor x1296;
JCudaTensor x1297, x1298;
x1297 = x1236;
x1298 = x712;
x1296 = x714.backward(x1297, x1298);

// Dealloc(X196)
JCudaTensor x1299;
x1299 = x712;
x1299.free();

// V_cv84_W <~~ X2334 * d_Convolv(1,0)(X192)/d_cv84_W
float x1301, x1302;
x1301 = lrn_rate_1;
x1302 = momentum;
JCudaTensor x1303, x1304;
x1303 = x1292;
x1304 = x674;
x704.backward_filter(x1303, x1304, x1300, x1301, x1302);

// V_cv82_B <~~ X2310 * d_Convolv(1,0)()/d_cv82_B
float x1306, x1307;
x1306 = lrn_rate_2;
x1307 = momentum;
JCudaTensor x1308;
x1308 = x1296;
x697.backward_bias(x1308, x1305, x1306, x1307);

// val X2312 = (X2288 + X2310 * d_Convolv(1,0)(cv82_W)/d_X192)
JCudaTensor x1309;
JCudaTensor x1310;
x1310 = x1216;
JCudaTensor x1311, x1312;
x1311 = x1296;
x1312 = x695;
x1309 = x697.backward_data(x1311,x1312, x1310);

// V_cv82_W <~~ X2310 * d_Convolv(1,0)(X192)/d_cv82_W
float x1314, x1315;
x1314 = lrn_rate_1;
x1315 = momentum;
JCudaTensor x1316, x1317;
x1316 = x1296;
x1317 = x674;
x697.backward_filter(x1316, x1317, x1313, x1314, x1315);

// Dealloc(X2310)
JCudaTensor x1318;
x1318 = x1296;
x1318.free();

// V_cv84_B <~~ X2334 * d_Convolv(1,0)()/d_cv84_B
float x1320, x1321;
x1320 = lrn_rate_2;
x1321 = momentum;
JCudaTensor x1322;
x1322 = x1292;
x704.backward_bias(x1322, x1319, x1320, x1321);

// cv82_B <~~ V_cv82_B
float x1323, x1324;
x1323 = 1;
x1324 = 1;
JCudaTensor x1325;
x1325 = x1305;
x696.update(x1325, x1323, x1324);

// cv82_W <~~ V_cv82_W
float x1326, x1327;
x1326 = 1;
x1327 = decay_1;
JCudaTensor x1328;
x1328 = x1313;
x695.update(x1328, x1326, x1327);

// cv84_B <~~ V_cv84_B
float x1329, x1330;
x1329 = 1;
x1330 = 1;
JCudaTensor x1331;
x1331 = x1319;
x703.update(x1331, x1329, x1330);

// val X2336 = (X2312 + X2334 * d_Convolv(1,0)(cv84_W)/d_X192)
JCudaTensor x1332;
JCudaTensor x1333;
x1333 = x1309;
JCudaTensor x1334, x1335;
x1334 = x1292;
x1335 = x702;
x1332 = x704.backward_data(x1334,x1335, x1333);

// Dealloc(X2334)
JCudaTensor x1336;
x1336 = x1292;
x1336.free();

// cv84_W <~~ V_cv84_W
float x1337, x1338;
x1337 = 1;
x1338 = decay_1;
JCudaTensor x1339;
x1339 = x1300;
x702.update(x1339, x1337, x1338);

// val X2358 = (X2336 + X2355 * d_Pooling(3,1,1,true)(X203,X192)/d_X192)
JCudaTensor x1340;
JCudaTensor x1341;
x1341 = x1332;
JCudaTensor x1342, x1343, x1344;
x1342 = x1264;
x1343 = x686;
x1344 = x674;
x1340 = x688.backward(x1342,x1343,x1344, x1341);

// Dealloc(X2355)
JCudaTensor x1345;
x1345 = x1264;
x1345.free();

// Dealloc(X203)
JCudaTensor x1346;
x1346 = x686;
x1346.free();

// val X2360 = X2358 * d_Pooling(3,2,1,true)(X192,X191)/d_X191
JCudaTensor x1347;
JCudaTensor x1348, x1349, x1350;
x1348 = x1340;
x1349 = x674;
x1350 = x669;
x1347 = x676.backward(x1348, x1349, x1350);

// Dealloc(X2358)
JCudaTensor x1351;
x1351 = x1340;
x1351.free();

// Dealloc(X192)
JCudaTensor x1352;
x1352 = x674;
x1352.free();

// Dealloc(X191)
JCudaTensor x1353;
x1353 = x669;
x1353.free();

// val X2392 = Proj(X2360, X179,X183,X187,X190, 1)
JCudaTensor x1354;
JCudaTensor x1356;
x1356 = x1347;
JCudaTensor[] x1355 = x250.backward(x1356);
x1354 = x1355[1];

// val X2416 = Proj(X2360, X179,X183,X187,X190, 2)
JCudaTensor x1357;
x1357 = x1355[2];

// val X2440 = Proj(X2360, X179,X183,X187,X190, 3)
JCudaTensor x1358;
x1358 = x1355[3];

// val X2374 = Proj(X2360, X179,X183,X187,X190, 0)
JCudaTensor x1359;
x1359 = x1355[0];

// Dealloc(X2360)
JCudaTensor x1360;
x1360 = x1347;
x1360.free();

// val X2377 = X2374 * d_ReLU()(X179)/d_X178
JCudaTensor x1361;
JCudaTensor x1362, x1363;
x1362 = x1359;
x1363 = x657;
x1361 = x240.backward(x1362, x1363);

// Dealloc(X179)
JCudaTensor x1364;
x1364 = x657;
x1364.free();

// val X2397 = X2392 * d_ReLU()(X183)/d_X182
JCudaTensor x1365;
JCudaTensor x1366, x1367;
x1366 = x1354;
x1367 = x636;
x1365 = x243.backward(x1366, x1367);

// Dealloc(X183)
JCudaTensor x1368;
x1368 = x636;
x1368.free();

// val X2444 = X2440 * d_ReLU()(X190)/d_X189
JCudaTensor x1369;
JCudaTensor x1370, x1371;
x1370 = x1358;
x1371 = x638;
x1369 = x246.backward(x1370, x1371);

// Dealloc(X190)
JCudaTensor x1372;
x1372 = x638;
x1372.free();

// val X2421 = X2416 * d_ReLU()(X187)/d_X186
JCudaTensor x1373;
JCudaTensor x1374, x1375;
x1374 = x1357;
x1375 = x634;
x1373 = x246.backward(x1374, x1375);

// Dealloc(X187)
JCudaTensor x1376;
x1376 = x634;
x1376.free();

// val X2378 = X2377 * d_Convolv(1,0)(cv71_W)/d_X177
JCudaTensor x1377;
JCudaTensor x1378, x1379;
x1378 = x1361;
x1379 = x542;
x1377 = x193.backward_data(x1378, x1379);

// V_cv71_W <~~ X2377 * d_Convolv(1,0)(X177)/d_cv71_W
float x1381, x1382;
x1381 = lrn_rate_1;
x1382 = momentum;
JCudaTensor x1383, x1384;
x1383 = x1361;
x1384 = x503;
x193.backward_filter(x1383, x1384, x1380, x1381, x1382);

// V_cv73_W <~~ X2397 * d_Convolv(1,1)(X181)/d_cv73_W
float x1386, x1387;
x1386 = lrn_rate_1;
x1387 = momentum;
JCudaTensor x1388, x1389;
x1388 = x1365;
x1389 = x563;
x237.backward_filter(x1388, x1389, x1385, x1386, x1387);

// V_cv73_B <~~ X2397 * d_Convolv(1,1)()/d_cv73_B
float x1391, x1392;
x1391 = lrn_rate_2;
x1392 = momentum;
JCudaTensor x1393;
x1393 = x1365;
x237.backward_bias(x1393, x1390, x1391, x1392);

// val X2398 = X2397 * d_Convolv(1,1)(cv73_W)/d_X181
JCudaTensor x1394;
JCudaTensor x1395, x1396;
x1395 = x1365;
x1396 = x626;
x1394 = x237.backward_data(x1395, x1396);

// Dealloc(X2397)
JCudaTensor x1397;
x1397 = x1365;
x1397.free();

// val X2445 = X2444 * d_Convolv(1,0)(cv76_W)/d_X188
JCudaTensor x1398;
JCudaTensor x1399, x1400;
x1399 = x1369;
x1400 = x583;
x1398 = x223.backward_data(x1399, x1400);

// V_cv75_B <~~ X2421 * d_Convolv(1,2)()/d_cv75_B
float x1402, x1403;
x1402 = lrn_rate_2;
x1403 = momentum;
JCudaTensor x1404;
x1404 = x1373;
x230.backward_bias(x1404, x1401, x1402, x1403);

// V_cv76_W <~~ X2444 * d_Convolv(1,0)(X188)/d_cv76_W
float x1406, x1407;
x1406 = lrn_rate_1;
x1407 = momentum;
JCudaTensor x1408, x1409;
x1408 = x1369;
x1409 = x528;
x223.backward_filter(x1408, x1409, x1405, x1406, x1407);

// val X2422 = X2421 * d_Convolv(1,2)(cv75_W)/d_X185
JCudaTensor x1410;
JCudaTensor x1411, x1412;
x1411 = x1373;
x1412 = x617;
x1410 = x230.backward_data(x1411, x1412);

// V_cv75_W <~~ X2421 * d_Convolv(1,2)(X185)/d_cv75_W
float x1414, x1415;
x1414 = lrn_rate_1;
x1415 = momentum;
JCudaTensor x1416, x1417;
x1416 = x1373;
x1417 = x577;
x230.backward_filter(x1416, x1417, x1413, x1414, x1415);

// Dealloc(X2421)
JCudaTensor x1418;
x1418 = x1373;
x1418.free();

// V_cv71_B <~~ X2377 * d_Convolv(1,0)()/d_cv71_B
float x1420, x1421;
x1420 = lrn_rate_2;
x1421 = momentum;
JCudaTensor x1422;
x1422 = x1361;
x193.backward_bias(x1422, x1419, x1420, x1421);

// Dealloc(X2377)
JCudaTensor x1423;
x1423 = x1361;
x1423.free();

// V_cv76_B <~~ X2444 * d_Convolv(1,0)()/d_cv76_B
float x1425, x1426;
x1425 = lrn_rate_2;
x1426 = momentum;
JCudaTensor x1427;
x1427 = x1369;
x223.backward_bias(x1427, x1424, x1425, x1426);

// Dealloc(X2444)
JCudaTensor x1428;
x1428 = x1369;
x1428.free();

// cv75_W <~~ V_cv75_W
float x1429, x1430;
x1429 = 1;
x1430 = decay_1;
JCudaTensor x1431;
x1431 = x1413;
x617.update(x1431, x1429, x1430);

// cv71_B <~~ V_cv71_B
float x1432, x1433;
x1432 = 1;
x1433 = 1;
JCudaTensor x1434;
x1434 = x1419;
x543.update(x1434, x1432, x1433);

// cv71_W <~~ V_cv71_W
float x1435, x1436;
x1435 = 1;
x1436 = decay_1;
JCudaTensor x1437;
x1437 = x1380;
x542.update(x1437, x1435, x1436);

// cv73_W <~~ V_cv73_W
float x1438, x1439;
x1438 = 1;
x1439 = decay_1;
JCudaTensor x1440;
x1440 = x1385;
x626.update(x1440, x1438, x1439);

// cv76_W <~~ V_cv76_W
float x1441, x1442;
x1441 = 1;
x1442 = decay_1;
JCudaTensor x1443;
x1443 = x1405;
x583.update(x1443, x1441, x1442);

// cv75_B <~~ V_cv75_B
float x1444, x1445;
x1444 = 1;
x1445 = 1;
JCudaTensor x1446;
x1446 = x1401;
x618.update(x1446, x1444, x1445);

// cv73_B <~~ V_cv73_B
float x1447, x1448;
x1447 = 1;
x1448 = 1;
JCudaTensor x1449;
x1449 = x1390;
x627.update(x1449, x1447, x1448);

// cv76_B <~~ V_cv76_B
float x1450, x1451;
x1450 = 1;
x1451 = 1;
JCudaTensor x1452;
x1452 = x1424;
x584.update(x1452, x1450, x1451);

// val X2400 = X2398 * d_ReLU()(X181)/d_X180
JCudaTensor x1453;
JCudaTensor x1454, x1455;
x1454 = x1394;
x1455 = x563;
x1453 = x216.backward(x1454, x1455);

// Dealloc(X181)
JCudaTensor x1456;
x1456 = x563;
x1456.free();

// val X2424 = X2422 * d_ReLU()(X185)/d_X184
JCudaTensor x1457;
JCudaTensor x1458, x1459;
x1458 = x1410;
x1459 = x577;
x1457 = x213.backward(x1458, x1459);

// Dealloc(X185)
JCudaTensor x1460;
x1460 = x577;
x1460.free();

// V_cv74_W <~~ X2424 * d_Convolv(1,0)(X177)/d_cv74_W
float x1462, x1463;
x1462 = lrn_rate_1;
x1463 = momentum;
JCudaTensor x1464, x1465;
x1464 = x1457;
x1465 = x503;
x210.backward_filter(x1464, x1465, x1461, x1462, x1463);

// val X2402 = (X2378 + X2400 * d_Convolv(1,0)(cv72_W)/d_X177)
JCudaTensor x1466;
JCudaTensor x1467;
x1467 = x1377;
JCudaTensor x1468, x1469;
x1468 = x1453;
x1469 = x548;
x1466 = x200.backward_data(x1468,x1469, x1467);

// V_cv72_W <~~ X2400 * d_Convolv(1,0)(X177)/d_cv72_W
float x1471, x1472;
x1471 = lrn_rate_1;
x1472 = momentum;
JCudaTensor x1473, x1474;
x1473 = x1453;
x1474 = x503;
x200.backward_filter(x1473, x1474, x1470, x1471, x1472);

// V_cv72_B <~~ X2400 * d_Convolv(1,0)()/d_cv72_B
float x1476, x1477;
x1476 = lrn_rate_2;
x1477 = momentum;
JCudaTensor x1478;
x1478 = x1453;
x200.backward_bias(x1478, x1475, x1476, x1477);

// Dealloc(X2400)
JCudaTensor x1479;
x1479 = x1453;
x1479.free();

// V_cv74_B <~~ X2424 * d_Convolv(1,0)()/d_cv74_B
float x1481, x1482;
x1481 = lrn_rate_2;
x1482 = momentum;
JCudaTensor x1483;
x1483 = x1457;
x210.backward_bias(x1483, x1480, x1481, x1482);

// cv72_W <~~ V_cv72_W
float x1484, x1485;
x1484 = 1;
x1485 = decay_1;
JCudaTensor x1486;
x1486 = x1470;
x548.update(x1486, x1484, x1485);

// cv72_B <~~ V_cv72_B
float x1487, x1488;
x1487 = 1;
x1488 = 1;
JCudaTensor x1489;
x1489 = x1475;
x549.update(x1489, x1487, x1488);

// cv74_B <~~ V_cv74_B
float x1490, x1491;
x1490 = 1;
x1491 = 1;
JCudaTensor x1492;
x1492 = x1480;
x560.update(x1492, x1490, x1491);

// val X2426 = (X2402 + X2424 * d_Convolv(1,0)(cv74_W)/d_X177)
JCudaTensor x1493;
JCudaTensor x1494;
x1494 = x1466;
JCudaTensor x1495, x1496;
x1495 = x1457;
x1496 = x559;
x1493 = x210.backward_data(x1495,x1496, x1494);

// Dealloc(X2424)
JCudaTensor x1497;
x1497 = x1457;
x1497.free();

// cv74_W <~~ V_cv74_W
float x1498, x1499;
x1498 = 1;
x1499 = decay_1;
JCudaTensor x1500;
x1500 = x1461;
x559.update(x1500, x1498, x1499);

// val X2448 = (X2426 + X2445 * d_Pooling(3,1,1,true)(X188,X177)/d_X177)
JCudaTensor x1501;
JCudaTensor x1502;
x1502 = x1493;
JCudaTensor x1503, x1504, x1505;
x1503 = x1398;
x1504 = x528;
x1505 = x503;
x1501 = x203.backward(x1503,x1504,x1505, x1502);

// Dealloc(X2445)
JCudaTensor x1506;
x1506 = x1398;
x1506.free();

// Dealloc(X188)
JCudaTensor x1507;
x1507 = x528;
x1507.free();

// Dealloc(X177)
JCudaTensor x1508;
x1508 = x503;
x1508.free();

// val X2484 = (X2483 * loss2)
JCudaTensor x1509;
JCudaTensor x1510;
float x1511;
x1510 = x916;
x1511 = loss2;
x1509 = x1510.times_i(x1511);

// val X2485 = (X2448 + X2484)
JCudaTensor x1512;
JCudaTensor x1513, x1514;
x1513 = x1501;
x1514 = x1509;
x1512 = x1513.plus_i(x1514);

// Dealloc(X2484)
JCudaTensor x1515;
x1515 = x1509;
x1515.free();

// val X2527 = Proj(X2485, X165,X169,X173,X176, 0)
JCudaTensor x1516;
JCudaTensor x1518;
x1518 = x1512;
JCudaTensor[] x1517 = x250.backward(x1518);
x1516 = x1517[0];

// val X2569 = Proj(X2485, X165,X169,X173,X176, 2)
JCudaTensor x1519;
x1519 = x1517[2];

// val X2545 = Proj(X2485, X165,X169,X173,X176, 1)
JCudaTensor x1520;
x1520 = x1517[1];

// val X2593 = Proj(X2485, X165,X169,X173,X176, 3)
JCudaTensor x1521;
x1521 = x1517[3];

// Dealloc(X2485)
JCudaTensor x1522;
x1522 = x1512;
x1522.free();

// val X2550 = X2545 * d_ReLU()(X169)/d_X168
JCudaTensor x1523;
JCudaTensor x1524, x1525;
x1524 = x1520;
x1525 = x487;
x1523 = x243.backward(x1524, x1525);

// Dealloc(X169)
JCudaTensor x1526;
x1526 = x487;
x1526.free();

// val X2597 = X2593 * d_ReLU()(X176)/d_X175
JCudaTensor x1527;
JCudaTensor x1528, x1529;
x1528 = x1521;
x1529 = x481;
x1527 = x246.backward(x1528, x1529);

// Dealloc(X176)
JCudaTensor x1530;
x1530 = x481;
x1530.free();

// val X2574 = X2569 * d_ReLU()(X173)/d_X172
JCudaTensor x1531;
JCudaTensor x1532, x1533;
x1532 = x1519;
x1533 = x489;
x1531 = x246.backward(x1532, x1533);

// Dealloc(X173)
JCudaTensor x1534;
x1534 = x489;
x1534.free();

// val X2530 = X2527 * d_ReLU()(X165)/d_X164
JCudaTensor x1535;
JCudaTensor x1536, x1537;
x1536 = x1516;
x1537 = x483;
x1535 = x240.backward(x1536, x1537);

// Dealloc(X165)
JCudaTensor x1538;
x1538 = x483;
x1538.free();

// val X2551 = X2550 * d_Convolv(1,1)(cv63_W)/d_X167
JCudaTensor x1539;
JCudaTensor x1540, x1541;
x1540 = x1523;
x1541 = x473;
x1539 = x237.backward_data(x1540, x1541);

// V_cv66_B <~~ X2597 * d_Convolv(1,0)()/d_cv66_B
float x1543, x1544;
x1543 = lrn_rate_2;
x1544 = momentum;
JCudaTensor x1545;
x1545 = x1527;
x223.backward_bias(x1545, x1542, x1543, x1544);

// val X2598 = X2597 * d_Convolv(1,0)(cv66_W)/d_X174
JCudaTensor x1546;
JCudaTensor x1547, x1548;
x1547 = x1527;
x1548 = x442;
x1546 = x223.backward_data(x1547, x1548);

// V_cv66_W <~~ X2597 * d_Convolv(1,0)(X174)/d_cv66_W
float x1550, x1551;
x1550 = lrn_rate_1;
x1551 = momentum;
JCudaTensor x1552, x1553;
x1552 = x1527;
x1553 = x436;
x223.backward_filter(x1552, x1553, x1549, x1550, x1551);

// Dealloc(X2597)
JCudaTensor x1554;
x1554 = x1527;
x1554.free();

// val X2575 = X2574 * d_Convolv(1,2)(cv65_W)/d_X171
JCudaTensor x1555;
JCudaTensor x1556, x1557;
x1556 = x1531;
x1557 = x460;
x1555 = x230.backward_data(x1556, x1557);

// V_cv65_W <~~ X2574 * d_Convolv(1,2)(X171)/d_cv65_W
float x1559, x1560;
x1559 = lrn_rate_1;
x1560 = momentum;
JCudaTensor x1561, x1562;
x1561 = x1531;
x1562 = x452;
x230.backward_filter(x1561, x1562, x1558, x1559, x1560);

// V_cv61_W <~~ X2530 * d_Convolv(1,0)(X163)/d_cv61_W
float x1564, x1565;
x1564 = lrn_rate_1;
x1565 = momentum;
JCudaTensor x1566, x1567;
x1566 = x1535;
x1567 = x400;
x193.backward_filter(x1566, x1567, x1563, x1564, x1565);

// V_cv63_W <~~ X2550 * d_Convolv(1,1)(X167)/d_cv63_W
float x1569, x1570;
x1569 = lrn_rate_1;
x1570 = momentum;
JCudaTensor x1571, x1572;
x1571 = x1523;
x1572 = x450;
x237.backward_filter(x1571, x1572, x1568, x1569, x1570);

// val X2531 = X2530 * d_Convolv(1,0)(cv61_W)/d_X163
JCudaTensor x1573;
JCudaTensor x1574, x1575;
x1574 = x1535;
x1575 = x434;
x1573 = x193.backward_data(x1574, x1575);

// V_cv65_B <~~ X2574 * d_Convolv(1,2)()/d_cv65_B
float x1577, x1578;
x1577 = lrn_rate_2;
x1578 = momentum;
JCudaTensor x1579;
x1579 = x1531;
x230.backward_bias(x1579, x1576, x1577, x1578);

// Dealloc(X2574)
JCudaTensor x1580;
x1580 = x1531;
x1580.free();

// V_cv61_B <~~ X2530 * d_Convolv(1,0)()/d_cv61_B
float x1582, x1583;
x1582 = lrn_rate_2;
x1583 = momentum;
JCudaTensor x1584;
x1584 = x1535;
x193.backward_bias(x1584, x1581, x1582, x1583);

// Dealloc(X2530)
JCudaTensor x1585;
x1585 = x1535;
x1585.free();

// V_cv63_B <~~ X2550 * d_Convolv(1,1)()/d_cv63_B
float x1587, x1588;
x1587 = lrn_rate_2;
x1588 = momentum;
JCudaTensor x1589;
x1589 = x1523;
x237.backward_bias(x1589, x1586, x1587, x1588);

// Dealloc(X2550)
JCudaTensor x1590;
x1590 = x1523;
x1590.free();

// cv66_W <~~ V_cv66_W
float x1591, x1592;
x1591 = 1;
x1592 = decay_1;
JCudaTensor x1593;
x1593 = x1549;
x442.update(x1593, x1591, x1592);

// cv65_W <~~ V_cv65_W
float x1594, x1595;
x1594 = 1;
x1595 = decay_1;
JCudaTensor x1596;
x1596 = x1558;
x460.update(x1596, x1594, x1595);

// cv61_B <~~ V_cv61_B
float x1597, x1598;
x1597 = 1;
x1598 = 1;
JCudaTensor x1599;
x1599 = x1581;
x435.update(x1599, x1597, x1598);

// cv63_B <~~ V_cv63_B
float x1600, x1601;
x1600 = 1;
x1601 = 1;
JCudaTensor x1602;
x1602 = x1586;
x474.update(x1602, x1600, x1601);

// cv61_W <~~ V_cv61_W
float x1603, x1604;
x1603 = 1;
x1604 = decay_1;
JCudaTensor x1605;
x1605 = x1563;
x434.update(x1605, x1603, x1604);

// cv63_W <~~ V_cv63_W
float x1606, x1607;
x1606 = 1;
x1607 = decay_1;
JCudaTensor x1608;
x1608 = x1568;
x473.update(x1608, x1606, x1607);

// cv65_B <~~ V_cv65_B
float x1609, x1610;
x1609 = 1;
x1610 = 1;
JCudaTensor x1611;
x1611 = x1576;
x461.update(x1611, x1609, x1610);

// cv66_B <~~ V_cv66_B
float x1612, x1613;
x1612 = 1;
x1613 = 1;
JCudaTensor x1614;
x1614 = x1542;
x443.update(x1614, x1612, x1613);

// val X2553 = X2551 * d_ReLU()(X167)/d_X166
JCudaTensor x1615;
JCudaTensor x1616, x1617;
x1616 = x1539;
x1617 = x450;
x1615 = x216.backward(x1616, x1617);

// Dealloc(X167)
JCudaTensor x1618;
x1618 = x450;
x1618.free();

// val X2577 = X2575 * d_ReLU()(X171)/d_X170
JCudaTensor x1619;
JCudaTensor x1620, x1621;
x1620 = x1555;
x1621 = x452;
x1619 = x213.backward(x1620, x1621);

// Dealloc(X171)
JCudaTensor x1622;
x1622 = x452;
x1622.free();

// V_cv64_W <~~ X2577 * d_Convolv(1,0)(X163)/d_cv64_W
float x1624, x1625;
x1624 = lrn_rate_1;
x1625 = momentum;
JCudaTensor x1626, x1627;
x1626 = x1619;
x1627 = x400;
x210.backward_filter(x1626, x1627, x1623, x1624, x1625);

// val X2555 = (X2531 + X2553 * d_Convolv(1,0)(cv62_W)/d_X163)
JCudaTensor x1628;
JCudaTensor x1629;
x1629 = x1573;
JCudaTensor x1630, x1631;
x1630 = x1615;
x1631 = x428;
x1628 = x200.backward_data(x1630,x1631, x1629);

// V_cv62_W <~~ X2553 * d_Convolv(1,0)(X163)/d_cv62_W
float x1633, x1634;
x1633 = lrn_rate_1;
x1634 = momentum;
JCudaTensor x1635, x1636;
x1635 = x1615;
x1636 = x400;
x200.backward_filter(x1635, x1636, x1632, x1633, x1634);

// V_cv62_B <~~ X2553 * d_Convolv(1,0)()/d_cv62_B
float x1638, x1639;
x1638 = lrn_rate_2;
x1639 = momentum;
JCudaTensor x1640;
x1640 = x1615;
x200.backward_bias(x1640, x1637, x1638, x1639);

// Dealloc(X2553)
JCudaTensor x1641;
x1641 = x1615;
x1641.free();

// V_cv64_B <~~ X2577 * d_Convolv(1,0)()/d_cv64_B
float x1643, x1644;
x1643 = lrn_rate_2;
x1644 = momentum;
JCudaTensor x1645;
x1645 = x1619;
x210.backward_bias(x1645, x1642, x1643, x1644);

// cv62_W <~~ V_cv62_W
float x1646, x1647;
x1646 = 1;
x1647 = decay_1;
JCudaTensor x1648;
x1648 = x1632;
x428.update(x1648, x1646, x1647);

// cv62_B <~~ V_cv62_B
float x1649, x1650;
x1649 = 1;
x1650 = 1;
JCudaTensor x1651;
x1651 = x1637;
x429.update(x1651, x1649, x1650);

// cv64_B <~~ V_cv64_B
float x1652, x1653;
x1652 = 1;
x1653 = 1;
JCudaTensor x1654;
x1654 = x1642;
x418.update(x1654, x1652, x1653);

// val X2579 = (X2555 + X2577 * d_Convolv(1,0)(cv64_W)/d_X163)
JCudaTensor x1655;
JCudaTensor x1656;
x1656 = x1628;
JCudaTensor x1657, x1658;
x1657 = x1619;
x1658 = x417;
x1655 = x210.backward_data(x1657,x1658, x1656);

// Dealloc(X2577)
JCudaTensor x1659;
x1659 = x1619;
x1659.free();

// cv64_W <~~ V_cv64_W
float x1660, x1661;
x1660 = 1;
x1661 = decay_1;
JCudaTensor x1662;
x1662 = x1623;
x417.update(x1662, x1660, x1661);

// val X2601 = (X2579 + X2598 * d_Pooling(3,1,1,true)(X174,X163)/d_X163)
JCudaTensor x1663;
JCudaTensor x1664;
x1664 = x1655;
JCudaTensor x1665, x1666, x1667;
x1665 = x1546;
x1666 = x436;
x1667 = x400;
x1663 = x203.backward(x1665,x1666,x1667, x1664);

// Dealloc(X2598)
JCudaTensor x1668;
x1668 = x1546;
x1668.free();

// Dealloc(X174)
JCudaTensor x1669;
x1669 = x436;
x1669.free();

// Dealloc(X163)
JCudaTensor x1670;
x1670 = x400;
x1670.free();

// val X2633 = Proj(X2601, X151,X155,X159,X162, 1)
JCudaTensor x1671;
JCudaTensor x1673;
x1673 = x1663;
JCudaTensor[] x1672 = x250.backward(x1673);
x1671 = x1672[1];

// val X2615 = Proj(X2601, X151,X155,X159,X162, 0)
JCudaTensor x1674;
x1674 = x1672[0];

// val X2681 = Proj(X2601, X151,X155,X159,X162, 3)
JCudaTensor x1675;
x1675 = x1672[3];

// val X2657 = Proj(X2601, X151,X155,X159,X162, 2)
JCudaTensor x1676;
x1676 = x1672[2];

// Dealloc(X2601)
JCudaTensor x1677;
x1677 = x1663;
x1677.free();

// val X2618 = X2615 * d_ReLU()(X151)/d_X150
JCudaTensor x1678;
JCudaTensor x1679, x1680;
x1679 = x1674;
x1680 = x394;
x1678 = x240.backward(x1679, x1680);

// Dealloc(X151)
JCudaTensor x1681;
x1681 = x394;
x1681.free();

// val X2685 = X2681 * d_ReLU()(X162)/d_X161
JCudaTensor x1682;
JCudaTensor x1683, x1684;
x1683 = x1675;
x1684 = x390;
x1682 = x246.backward(x1683, x1684);

// Dealloc(X162)
JCudaTensor x1685;
x1685 = x390;
x1685.free();

// val X2638 = X2633 * d_ReLU()(X155)/d_X154
JCudaTensor x1686;
JCudaTensor x1687, x1688;
x1687 = x1671;
x1688 = x388;
x1686 = x243.backward(x1687, x1688);

// Dealloc(X155)
JCudaTensor x1689;
x1689 = x388;
x1689.free();

// val X2662 = X2657 * d_ReLU()(X159)/d_X158
JCudaTensor x1690;
JCudaTensor x1691, x1692;
x1691 = x1676;
x1692 = x392;
x1690 = x246.backward(x1691, x1692);

// Dealloc(X159)
JCudaTensor x1693;
x1693 = x392;
x1693.free();

// V_cv51_W <~~ X2618 * d_Convolv(1,0)(X149)/d_cv51_W
float x1695, x1696;
x1695 = lrn_rate_1;
x1696 = momentum;
JCudaTensor x1697, x1698;
x1697 = x1678;
x1698 = x325;
x193.backward_filter(x1697, x1698, x1694, x1695, x1696);

// val X2686 = X2685 * d_Convolv(1,0)(cv56_W)/d_X160
JCudaTensor x1699;
JCudaTensor x1700, x1701;
x1700 = x1682;
x1701 = x366;
x1699 = x223.backward_data(x1700, x1701);

// V_cv53_W <~~ X2638 * d_Convolv(1,1)(X153)/d_cv53_W
float x1703, x1704;
x1703 = lrn_rate_1;
x1704 = momentum;
JCudaTensor x1705, x1706;
x1705 = x1686;
x1706 = x368;
x237.backward_filter(x1705, x1706, x1702, x1703, x1704);

// V_cv55_B <~~ X2662 * d_Convolv(1,2)()/d_cv55_B
float x1708, x1709;
x1708 = lrn_rate_2;
x1709 = momentum;
JCudaTensor x1710;
x1710 = x1690;
x230.backward_bias(x1710, x1707, x1708, x1709);

// V_cv53_B <~~ X2638 * d_Convolv(1,1)()/d_cv53_B
float x1712, x1713;
x1712 = lrn_rate_2;
x1713 = momentum;
JCudaTensor x1714;
x1714 = x1686;
x237.backward_bias(x1714, x1711, x1712, x1713);

// V_cv51_B <~~ X2618 * d_Convolv(1,0)()/d_cv51_B
float x1716, x1717;
x1716 = lrn_rate_2;
x1717 = momentum;
JCudaTensor x1718;
x1718 = x1678;
x193.backward_bias(x1718, x1715, x1716, x1717);

// val X2663 = X2662 * d_Convolv(1,2)(cv55_W)/d_X157
JCudaTensor x1719;
JCudaTensor x1720, x1721;
x1720 = x1690;
x1721 = x380;
x1719 = x230.backward_data(x1720, x1721);

// val X2639 = X2638 * d_Convolv(1,1)(cv53_W)/d_X153
JCudaTensor x1722;
JCudaTensor x1723, x1724;
x1723 = x1686;
x1724 = x374;
x1722 = x237.backward_data(x1723, x1724);

// Dealloc(X2638)
JCudaTensor x1725;
x1725 = x1686;
x1725.free();

// V_cv56_B <~~ X2685 * d_Convolv(1,0)()/d_cv56_B
float x1727, x1728;
x1727 = lrn_rate_2;
x1728 = momentum;
JCudaTensor x1729;
x1729 = x1682;
x223.backward_bias(x1729, x1726, x1727, x1728);

// val X2619 = X2618 * d_Convolv(1,0)(cv51_W)/d_X149
JCudaTensor x1730;
JCudaTensor x1731, x1732;
x1731 = x1678;
x1732 = x349;
x1730 = x193.backward_data(x1731, x1732);

// Dealloc(X2618)
JCudaTensor x1733;
x1733 = x1678;
x1733.free();

// V_cv56_W <~~ X2685 * d_Convolv(1,0)(X160)/d_cv56_W
float x1735, x1736;
x1735 = lrn_rate_1;
x1736 = momentum;
JCudaTensor x1737, x1738;
x1737 = x1682;
x1738 = x343;
x223.backward_filter(x1737, x1738, x1734, x1735, x1736);

// Dealloc(X2685)
JCudaTensor x1739;
x1739 = x1682;
x1739.free();

// V_cv55_W <~~ X2662 * d_Convolv(1,2)(X157)/d_cv55_W
float x1741, x1742;
x1741 = lrn_rate_1;
x1742 = momentum;
JCudaTensor x1743, x1744;
x1743 = x1690;
x1744 = x357;
x230.backward_filter(x1743, x1744, x1740, x1741, x1742);

// Dealloc(X2662)
JCudaTensor x1745;
x1745 = x1690;
x1745.free();

// cv56_W <~~ V_cv56_W
float x1746, x1747;
x1746 = 1;
x1747 = decay_1;
JCudaTensor x1748;
x1748 = x1734;
x366.update(x1748, x1746, x1747);

// cv51_W <~~ V_cv51_W
float x1749, x1750;
x1749 = 1;
x1750 = decay_1;
JCudaTensor x1751;
x1751 = x1694;
x349.update(x1751, x1749, x1750);

// cv55_B <~~ V_cv55_B
float x1752, x1753;
x1752 = 1;
x1753 = 1;
JCudaTensor x1754;
x1754 = x1707;
x381.update(x1754, x1752, x1753);

// cv53_B <~~ V_cv53_B
float x1755, x1756;
x1755 = 1;
x1756 = 1;
JCudaTensor x1757;
x1757 = x1711;
x375.update(x1757, x1755, x1756);

// cv51_B <~~ V_cv51_B
float x1758, x1759;
x1758 = 1;
x1759 = 1;
JCudaTensor x1760;
x1760 = x1715;
x350.update(x1760, x1758, x1759);

// cv53_W <~~ V_cv53_W
float x1761, x1762;
x1761 = 1;
x1762 = decay_1;
JCudaTensor x1763;
x1763 = x1702;
x374.update(x1763, x1761, x1762);

// cv55_W <~~ V_cv55_W
float x1764, x1765;
x1764 = 1;
x1765 = decay_1;
JCudaTensor x1766;
x1766 = x1740;
x380.update(x1766, x1764, x1765);

// cv56_B <~~ V_cv56_B
float x1767, x1768;
x1767 = 1;
x1768 = 1;
JCudaTensor x1769;
x1769 = x1726;
x367.update(x1769, x1767, x1768);

// val X2641 = X2639 * d_ReLU()(X153)/d_X152
JCudaTensor x1770;
JCudaTensor x1771, x1772;
x1771 = x1722;
x1772 = x368;
x1770 = x216.backward(x1771, x1772);

// Dealloc(X153)
JCudaTensor x1773;
x1773 = x368;
x1773.free();

// val X2665 = X2663 * d_ReLU()(X157)/d_X156
JCudaTensor x1774;
JCudaTensor x1775, x1776;
x1775 = x1719;
x1776 = x357;
x1774 = x213.backward(x1775, x1776);

// Dealloc(X157)
JCudaTensor x1777;
x1777 = x357;
x1777.free();

// V_cv54_W <~~ X2665 * d_Convolv(1,0)(X149)/d_cv54_W
float x1779, x1780;
x1779 = lrn_rate_1;
x1780 = momentum;
JCudaTensor x1781, x1782;
x1781 = x1774;
x1782 = x325;
x210.backward_filter(x1781, x1782, x1778, x1779, x1780);

// V_cv52_B <~~ X2641 * d_Convolv(1,0)()/d_cv52_B
float x1784, x1785;
x1784 = lrn_rate_2;
x1785 = momentum;
JCudaTensor x1786;
x1786 = x1770;
x200.backward_bias(x1786, x1783, x1784, x1785);

// V_cv54_B <~~ X2665 * d_Convolv(1,0)()/d_cv54_B
float x1788, x1789;
x1788 = lrn_rate_2;
x1789 = momentum;
JCudaTensor x1790;
x1790 = x1774;
x210.backward_bias(x1790, x1787, x1788, x1789);

// val X2643 = (X2619 + X2641 * d_Convolv(1,0)(cv52_W)/d_X149)
JCudaTensor x1791;
JCudaTensor x1792;
x1792 = x1730;
JCudaTensor x1793, x1794;
x1793 = x1770;
x1794 = x338;
x1791 = x200.backward_data(x1793,x1794, x1792);

// V_cv52_W <~~ X2641 * d_Convolv(1,0)(X149)/d_cv52_W
float x1796, x1797;
x1796 = lrn_rate_1;
x1797 = momentum;
JCudaTensor x1798, x1799;
x1798 = x1770;
x1799 = x325;
x200.backward_filter(x1798, x1799, x1795, x1796, x1797);

// Dealloc(X2641)
JCudaTensor x1800;
x1800 = x1770;
x1800.free();

// cv52_B <~~ V_cv52_B
float x1801, x1802;
x1801 = 1;
x1802 = 1;
JCudaTensor x1803;
x1803 = x1783;
x339.update(x1803, x1801, x1802);

// cv54_B <~~ V_cv54_B
float x1804, x1805;
x1804 = 1;
x1805 = 1;
JCudaTensor x1806;
x1806 = x1787;
x356.update(x1806, x1804, x1805);

// cv52_W <~~ V_cv52_W
float x1807, x1808;
x1807 = 1;
x1808 = decay_1;
JCudaTensor x1809;
x1809 = x1795;
x338.update(x1809, x1807, x1808);

// val X2667 = (X2643 + X2665 * d_Convolv(1,0)(cv54_W)/d_X149)
JCudaTensor x1810;
JCudaTensor x1811;
x1811 = x1791;
JCudaTensor x1812, x1813;
x1812 = x1774;
x1813 = x355;
x1810 = x210.backward_data(x1812,x1813, x1811);

// Dealloc(X2665)
JCudaTensor x1814;
x1814 = x1774;
x1814.free();

// cv54_W <~~ V_cv54_W
float x1815, x1816;
x1815 = 1;
x1816 = decay_1;
JCudaTensor x1817;
x1817 = x1778;
x355.update(x1817, x1815, x1816);

// val X2689 = (X2667 + X2686 * d_Pooling(3,1,1,true)(X160,X149)/d_X149)
JCudaTensor x1818;
JCudaTensor x1819;
x1819 = x1810;
JCudaTensor x1820, x1821, x1822;
x1820 = x1699;
x1821 = x343;
x1822 = x325;
x1818 = x203.backward(x1820,x1821,x1822, x1819);

// Dealloc(X2686)
JCudaTensor x1823;
x1823 = x1699;
x1823.free();

// Dealloc(X160)
JCudaTensor x1824;
x1824 = x343;
x1824.free();

// Dealloc(X149)
JCudaTensor x1825;
x1825 = x325;
x1825.free();

// val X2703 = Proj(X2689, X137,X141,X145,X148, 0)
JCudaTensor x1826;
JCudaTensor x1828;
x1828 = x1818;
JCudaTensor[] x1827 = x250.backward(x1828);
x1826 = x1827[0];

// val X2721 = Proj(X2689, X137,X141,X145,X148, 1)
JCudaTensor x1829;
x1829 = x1827[1];

// val X2745 = Proj(X2689, X137,X141,X145,X148, 2)
JCudaTensor x1830;
x1830 = x1827[2];

// val X2769 = Proj(X2689, X137,X141,X145,X148, 3)
JCudaTensor x1831;
x1831 = x1827[3];

// Dealloc(X2689)
JCudaTensor x1832;
x1832 = x1818;
x1832.free();

// val X2773 = X2769 * d_ReLU()(X148)/d_X147
JCudaTensor x1833;
JCudaTensor x1834, x1835;
x1834 = x1831;
x1835 = x316;
x1833 = x246.backward(x1834, x1835);

// Dealloc(X148)
JCudaTensor x1836;
x1836 = x316;
x1836.free();

// val X2706 = X2703 * d_ReLU()(X137)/d_X136
JCudaTensor x1837;
JCudaTensor x1838, x1839;
x1838 = x1826;
x1839 = x312;
x1837 = x240.backward(x1838, x1839);

// Dealloc(X137)
JCudaTensor x1840;
x1840 = x312;
x1840.free();

// val X2750 = X2745 * d_ReLU()(X145)/d_X144
JCudaTensor x1841;
JCudaTensor x1842, x1843;
x1842 = x1830;
x1843 = x310;
x1841 = x246.backward(x1842, x1843);

// Dealloc(X145)
JCudaTensor x1844;
x1844 = x310;
x1844.free();

// val X2726 = X2721 * d_ReLU()(X141)/d_X140
JCudaTensor x1845;
JCudaTensor x1846, x1847;
x1846 = x1829;
x1847 = x314;
x1845 = x243.backward(x1846, x1847);

// Dealloc(X141)
JCudaTensor x1848;
x1848 = x314;
x1848.free();

// V_cv46_W <~~ X2773 * d_Convolv(1,0)(X146)/d_cv46_W
float x1850, x1851;
x1850 = lrn_rate_1;
x1851 = momentum;
JCudaTensor x1852, x1853;
x1852 = x1833;
x1853 = x264;
x223.backward_filter(x1852, x1853, x1849, x1850, x1851);

// V_cv41_B <~~ X2706 * d_Convolv(1,0)()/d_cv41_B
float x1855, x1856;
x1855 = lrn_rate_2;
x1856 = momentum;
JCudaTensor x1857;
x1857 = x1837;
x193.backward_bias(x1857, x1854, x1855, x1856);

// val X2774 = X2773 * d_Convolv(1,0)(cv46_W)/d_X146
JCudaTensor x1858;
JCudaTensor x1859, x1860;
x1859 = x1833;
x1860 = x293;
x1858 = x223.backward_data(x1859, x1860);

// V_cv45_B <~~ X2750 * d_Convolv(1,2)()/d_cv45_B
float x1862, x1863;
x1862 = lrn_rate_2;
x1863 = momentum;
JCudaTensor x1864;
x1864 = x1841;
x230.backward_bias(x1864, x1861, x1862, x1863);

// V_cv41_W <~~ X2706 * d_Convolv(1,0)(X135)/d_cv41_W
float x1866, x1867;
x1866 = lrn_rate_1;
x1867 = momentum;
JCudaTensor x1868, x1869;
x1868 = x1837;
x1869 = x249;
x193.backward_filter(x1868, x1869, x1865, x1866, x1867);

// val X2727 = X2726 * d_Convolv(1,1)(cv43_W)/d_X139
JCudaTensor x1870;
JCudaTensor x1871, x1872;
x1871 = x1845;
x1872 = x308;
x1870 = x237.backward_data(x1871, x1872);

// V_cv45_W <~~ X2750 * d_Convolv(1,2)(X143)/d_cv45_W
float x1874, x1875;
x1874 = lrn_rate_1;
x1875 = momentum;
JCudaTensor x1876, x1877;
x1876 = x1841;
x1877 = x278;
x230.backward_filter(x1876, x1877, x1873, x1874, x1875);

// V_cv46_B <~~ X2773 * d_Convolv(1,0)()/d_cv46_B
float x1879, x1880;
x1879 = lrn_rate_2;
x1880 = momentum;
JCudaTensor x1881;
x1881 = x1833;
x223.backward_bias(x1881, x1878, x1879, x1880);

// Dealloc(X2773)
JCudaTensor x1882;
x1882 = x1833;
x1882.free();

// val X2751 = X2750 * d_Convolv(1,2)(cv45_W)/d_X143
JCudaTensor x1883;
JCudaTensor x1884, x1885;
x1884 = x1841;
x1885 = x299;
x1883 = x230.backward_data(x1884, x1885);

// Dealloc(X2750)
JCudaTensor x1886;
x1886 = x1841;
x1886.free();

// val X2707 = X2706 * d_Convolv(1,0)(cv41_W)/d_X135
JCudaTensor x1887;
JCudaTensor x1888, x1889;
x1888 = x1837;
x1889 = x276;
x1887 = x193.backward_data(x1888, x1889);

// Dealloc(X2706)
JCudaTensor x1890;
x1890 = x1837;
x1890.free();

// V_cv43_B <~~ X2726 * d_Convolv(1,1)()/d_cv43_B
float x1892, x1893;
x1892 = lrn_rate_2;
x1893 = momentum;
JCudaTensor x1894;
x1894 = x1845;
x237.backward_bias(x1894, x1891, x1892, x1893);

// V_cv43_W <~~ X2726 * d_Convolv(1,1)(X139)/d_cv43_W
float x1896, x1897;
x1896 = lrn_rate_1;
x1897 = momentum;
JCudaTensor x1898, x1899;
x1898 = x1845;
x1899 = x287;
x237.backward_filter(x1898, x1899, x1895, x1896, x1897);

// Dealloc(X2726)
JCudaTensor x1900;
x1900 = x1845;
x1900.free();

// cv46_W <~~ V_cv46_W
float x1901, x1902;
x1901 = 1;
x1902 = decay_1;
JCudaTensor x1903;
x1903 = x1849;
x293.update(x1903, x1901, x1902);

// cv43_W <~~ V_cv43_W
float x1904, x1905;
x1904 = 1;
x1905 = decay_1;
JCudaTensor x1906;
x1906 = x1895;
x308.update(x1906, x1904, x1905);

// cv43_B <~~ V_cv43_B
float x1907, x1908;
x1907 = 1;
x1908 = 1;
JCudaTensor x1909;
x1909 = x1891;
x309.update(x1909, x1907, x1908);

// cv41_B <~~ V_cv41_B
float x1910, x1911;
x1910 = 1;
x1911 = 1;
JCudaTensor x1912;
x1912 = x1854;
x277.update(x1912, x1910, x1911);

// cv41_W <~~ V_cv41_W
float x1913, x1914;
x1913 = 1;
x1914 = decay_1;
JCudaTensor x1915;
x1915 = x1865;
x276.update(x1915, x1913, x1914);

// cv45_W <~~ V_cv45_W
float x1916, x1917;
x1916 = 1;
x1917 = decay_1;
JCudaTensor x1918;
x1918 = x1873;
x299.update(x1918, x1916, x1917);

// cv46_B <~~ V_cv46_B
float x1919, x1920;
x1919 = 1;
x1920 = 1;
JCudaTensor x1921;
x1921 = x1878;
x294.update(x1921, x1919, x1920);

// cv45_B <~~ V_cv45_B
float x1922, x1923;
x1922 = 1;
x1923 = 1;
JCudaTensor x1924;
x1924 = x1861;
x300.update(x1924, x1922, x1923);

// val X2753 = X2751 * d_ReLU()(X143)/d_X142
JCudaTensor x1925;
JCudaTensor x1926, x1927;
x1926 = x1883;
x1927 = x278;
x1925 = x213.backward(x1926, x1927);

// Dealloc(X143)
JCudaTensor x1928;
x1928 = x278;
x1928.free();

// val X2729 = X2727 * d_ReLU()(X139)/d_X138
JCudaTensor x1929;
JCudaTensor x1930, x1931;
x1930 = x1870;
x1931 = x287;
x1929 = x216.backward(x1930, x1931);

// Dealloc(X139)
JCudaTensor x1932;
x1932 = x287;
x1932.free();

// V_cv44_W <~~ X2753 * d_Convolv(1,0)(X135)/d_cv44_W
float x1934, x1935;
x1934 = lrn_rate_1;
x1935 = momentum;
JCudaTensor x1936, x1937;
x1936 = x1925;
x1937 = x249;
x210.backward_filter(x1936, x1937, x1933, x1934, x1935);

// val X2731 = (X2707 + X2729 * d_Convolv(1,0)(cv42_W)/d_X135)
JCudaTensor x1938;
JCudaTensor x1939;
x1939 = x1887;
JCudaTensor x1940, x1941;
x1940 = x1929;
x1941 = x270;
x1938 = x200.backward_data(x1940,x1941, x1939);

// V_cv42_W <~~ X2729 * d_Convolv(1,0)(X135)/d_cv42_W
float x1943, x1944;
x1943 = lrn_rate_1;
x1944 = momentum;
JCudaTensor x1945, x1946;
x1945 = x1929;
x1946 = x249;
x200.backward_filter(x1945, x1946, x1942, x1943, x1944);

// V_cv42_B <~~ X2729 * d_Convolv(1,0)()/d_cv42_B
float x1948, x1949;
x1948 = lrn_rate_2;
x1949 = momentum;
JCudaTensor x1950;
x1950 = x1929;
x200.backward_bias(x1950, x1947, x1948, x1949);

// Dealloc(X2729)
JCudaTensor x1951;
x1951 = x1929;
x1951.free();

// V_cv44_B <~~ X2753 * d_Convolv(1,0)()/d_cv44_B
float x1953, x1954;
x1953 = lrn_rate_2;
x1954 = momentum;
JCudaTensor x1955;
x1955 = x1925;
x210.backward_bias(x1955, x1952, x1953, x1954);

// cv42_W <~~ V_cv42_W
float x1956, x1957;
x1956 = 1;
x1957 = decay_1;
JCudaTensor x1958;
x1958 = x1942;
x270.update(x1958, x1956, x1957);

// cv42_B <~~ V_cv42_B
float x1959, x1960;
x1959 = 1;
x1960 = 1;
JCudaTensor x1961;
x1961 = x1947;
x271.update(x1961, x1959, x1960);

// cv44_B <~~ V_cv44_B
float x1962, x1963;
x1962 = 1;
x1963 = 1;
JCudaTensor x1964;
x1964 = x1952;
x263.update(x1964, x1962, x1963);

// val X2755 = (X2731 + X2753 * d_Convolv(1,0)(cv44_W)/d_X135)
JCudaTensor x1965;
JCudaTensor x1966;
x1966 = x1938;
JCudaTensor x1967, x1968;
x1967 = x1925;
x1968 = x262;
x1965 = x210.backward_data(x1967,x1968, x1966);

// Dealloc(X2753)
JCudaTensor x1969;
x1969 = x1925;
x1969.free();

// cv44_W <~~ V_cv44_W
float x1970, x1971;
x1970 = 1;
x1971 = decay_1;
JCudaTensor x1972;
x1972 = x1933;
x262.update(x1972, x1970, x1971);

// val X2777 = (X2755 + X2774 * d_Pooling(3,1,1,true)(X146,X135)/d_X135)
JCudaTensor x1973;
JCudaTensor x1974;
x1974 = x1965;
JCudaTensor x1975, x1976, x1977;
x1975 = x1858;
x1976 = x264;
x1977 = x249;
x1973 = x203.backward(x1975,x1976,x1977, x1974);

// Dealloc(X2774)
JCudaTensor x1978;
x1978 = x1858;
x1978.free();

// Dealloc(X146)
JCudaTensor x1979;
x1979 = x264;
x1979.free();

// Dealloc(X135)
JCudaTensor x1980;
x1980 = x249;
x1980.free();

// val X2813 = (X2812 * loss1)
JCudaTensor x1981;
JCudaTensor x1982;
float x1983;
x1982 = x599;
x1983 = loss1;
x1981 = x1982.times_i(x1983);

// val X2814 = (X2777 + X2813)
JCudaTensor x1984;
JCudaTensor x1985, x1986;
x1985 = x1973;
x1986 = x1981;
x1984 = x1985.plus_i(x1986);

// Dealloc(X2813)
JCudaTensor x1987;
x1987 = x1981;
x1987.free();

// val X2909 = Proj(X2814, X123,X127,X131,X134, 2)
JCudaTensor x1988;
JCudaTensor x1990;
x1990 = x1984;
JCudaTensor[] x1989 = x250.backward(x1990);
x1988 = x1989[2];

// val X2867 = Proj(X2814, X123,X127,X131,X134, 0)
JCudaTensor x1991;
x1991 = x1989[0];

// val X2933 = Proj(X2814, X123,X127,X131,X134, 3)
JCudaTensor x1992;
x1992 = x1989[3];

// val X2885 = Proj(X2814, X123,X127,X131,X134, 1)
JCudaTensor x1993;
x1993 = x1989[1];

// Dealloc(X2814)
JCudaTensor x1994;
x1994 = x1984;
x1994.free();

// val X2937 = X2933 * d_ReLU()(X134)/d_X133
JCudaTensor x1995;
JCudaTensor x1996, x1997;
x1996 = x1992;
x1997 = x247;
x1995 = x246.backward(x1996, x1997);

// Dealloc(X134)
JCudaTensor x1998;
x1998 = x247;
x1998.free();

// val X2870 = X2867 * d_ReLU()(X123)/d_X122
JCudaTensor x1999;
JCudaTensor x2000, x2001;
x2000 = x1991;
x2001 = x238;
x1999 = x240.backward(x2000, x2001);

// Dealloc(X123)
JCudaTensor x2002;
x2002 = x238;
x2002.free();

// val X2914 = X2909 * d_ReLU()(X131)/d_X130
JCudaTensor x2003;
JCudaTensor x2004, x2005;
x2004 = x1988;
x2005 = x244;
x2003 = x246.backward(x2004, x2005);

// Dealloc(X131)
JCudaTensor x2006;
x2006 = x244;
x2006.free();

// val X2890 = X2885 * d_ReLU()(X127)/d_X126
JCudaTensor x2007;
JCudaTensor x2008, x2009;
x2008 = x1993;
x2009 = x241;
x2007 = x243.backward(x2008, x2009);

// Dealloc(X127)
JCudaTensor x2010;
x2010 = x241;
x2010.free();

// V_cv36_W <~~ X2937 * d_Convolv(1,0)(X132)/d_cv36_W
float x2012, x2013;
x2012 = lrn_rate_1;
x2013 = momentum;
JCudaTensor x2014, x2015;
x2014 = x1995;
x2015 = x201;
x223.backward_filter(x2014, x2015, x2011, x2012, x2013);

// V_cv31_W <~~ X2870 * d_Convolv(1,0)(X121)/d_cv31_W
float x2017, x2018;
x2017 = lrn_rate_1;
x2018 = momentum;
JCudaTensor x2019, x2020;
x2019 = x1999;
x2020 = x184;
x193.backward_filter(x2019, x2020, x2016, x2017, x2018);

// V_cv35_W <~~ X2914 * d_Convolv(1,2)(X129)/d_cv35_W
float x2022, x2023;
x2022 = lrn_rate_1;
x2023 = momentum;
JCudaTensor x2024, x2025;
x2024 = x2003;
x2025 = x211;
x230.backward_filter(x2024, x2025, x2021, x2022, x2023);

// V_cv33_B <~~ X2890 * d_Convolv(1,1)()/d_cv33_B
float x2027, x2028;
x2027 = lrn_rate_2;
x2028 = momentum;
JCudaTensor x2029;
x2029 = x2007;
x237.backward_bias(x2029, x2026, x2027, x2028);

// V_cv33_W <~~ X2890 * d_Convolv(1,1)(X125)/d_cv33_W
float x2031, x2032;
x2031 = lrn_rate_1;
x2032 = momentum;
JCudaTensor x2033, x2034;
x2033 = x2007;
x2034 = x214;
x237.backward_filter(x2033, x2034, x2030, x2031, x2032);

// V_cv31_B <~~ X2870 * d_Convolv(1,0)()/d_cv31_B
float x2036, x2037;
x2036 = lrn_rate_2;
x2037 = momentum;
JCudaTensor x2038;
x2038 = x1999;
x193.backward_bias(x2038, x2035, x2036, x2037);

// val X2871 = X2870 * d_Convolv(1,0)(cv31_W)/d_X121
JCudaTensor x2039;
JCudaTensor x2040, x2041;
x2040 = x1999;
x2041 = x191;
x2039 = x193.backward_data(x2040, x2041);

// Dealloc(X2870)
JCudaTensor x2042;
x2042 = x1999;
x2042.free();

// V_cv35_B <~~ X2914 * d_Convolv(1,2)()/d_cv35_B
float x2044, x2045;
x2044 = lrn_rate_2;
x2045 = momentum;
JCudaTensor x2046;
x2046 = x2003;
x230.backward_bias(x2046, x2043, x2044, x2045);

// val X2891 = X2890 * d_Convolv(1,1)(cv33_W)/d_X125
JCudaTensor x2047;
JCudaTensor x2048, x2049;
x2048 = x2007;
x2049 = x235;
x2047 = x237.backward_data(x2048, x2049);

// Dealloc(X2890)
JCudaTensor x2050;
x2050 = x2007;
x2050.free();

// val X2938 = X2937 * d_Convolv(1,0)(cv36_W)/d_X132
JCudaTensor x2051;
JCudaTensor x2052, x2053;
x2052 = x1995;
x2053 = x221;
x2051 = x223.backward_data(x2052, x2053);

// V_cv36_B <~~ X2937 * d_Convolv(1,0)()/d_cv36_B
float x2055, x2056;
x2055 = lrn_rate_2;
x2056 = momentum;
JCudaTensor x2057;
x2057 = x1995;
x223.backward_bias(x2057, x2054, x2055, x2056);

// Dealloc(X2937)
JCudaTensor x2058;
x2058 = x1995;
x2058.free();

// val X2915 = X2914 * d_Convolv(1,2)(cv35_W)/d_X129
JCudaTensor x2059;
JCudaTensor x2060, x2061;
x2060 = x2003;
x2061 = x228;
x2059 = x230.backward_data(x2060, x2061);

// Dealloc(X2914)
JCudaTensor x2062;
x2062 = x2003;
x2062.free();

// cv33_B <~~ V_cv33_B
float x2063, x2064;
x2063 = 1;
x2064 = 1;
JCudaTensor x2065;
x2065 = x2026;
x236.update(x2065, x2063, x2064);

// cv31_B <~~ V_cv31_B
float x2066, x2067;
x2066 = 1;
x2067 = 1;
JCudaTensor x2068;
x2068 = x2035;
x192.update(x2068, x2066, x2067);

// cv35_W <~~ V_cv35_W
float x2069, x2070;
x2069 = 1;
x2070 = decay_1;
JCudaTensor x2071;
x2071 = x2021;
x228.update(x2071, x2069, x2070);

// cv36_B <~~ V_cv36_B
float x2072, x2073;
x2072 = 1;
x2073 = 1;
JCudaTensor x2074;
x2074 = x2054;
x222.update(x2074, x2072, x2073);

// cv33_W <~~ V_cv33_W
float x2075, x2076;
x2075 = 1;
x2076 = decay_1;
JCudaTensor x2077;
x2077 = x2030;
x235.update(x2077, x2075, x2076);

// cv35_B <~~ V_cv35_B
float x2078, x2079;
x2078 = 1;
x2079 = 1;
JCudaTensor x2080;
x2080 = x2043;
x229.update(x2080, x2078, x2079);

// cv31_W <~~ V_cv31_W
float x2081, x2082;
x2081 = 1;
x2082 = decay_1;
JCudaTensor x2083;
x2083 = x2016;
x191.update(x2083, x2081, x2082);

// cv36_W <~~ V_cv36_W
float x2084, x2085;
x2084 = 1;
x2085 = decay_1;
JCudaTensor x2086;
x2086 = x2011;
x221.update(x2086, x2084, x2085);

// val X2893 = X2891 * d_ReLU()(X125)/d_X124
JCudaTensor x2087;
JCudaTensor x2088, x2089;
x2088 = x2047;
x2089 = x214;
x2087 = x216.backward(x2088, x2089);

// Dealloc(X125)
JCudaTensor x2090;
x2090 = x214;
x2090.free();

// val X2917 = X2915 * d_ReLU()(X129)/d_X128
JCudaTensor x2091;
JCudaTensor x2092, x2093;
x2092 = x2059;
x2093 = x211;
x2091 = x213.backward(x2092, x2093);

// Dealloc(X129)
JCudaTensor x2094;
x2094 = x211;
x2094.free();

// V_cv34_W <~~ X2917 * d_Convolv(1,0)(X121)/d_cv34_W
float x2096, x2097;
x2096 = lrn_rate_1;
x2097 = momentum;
JCudaTensor x2098, x2099;
x2098 = x2091;
x2099 = x184;
x210.backward_filter(x2098, x2099, x2095, x2096, x2097);

// val X2895 = (X2871 + X2893 * d_Convolv(1,0)(cv32_W)/d_X121)
JCudaTensor x2100;
JCudaTensor x2101;
x2101 = x2039;
JCudaTensor x2102, x2103;
x2102 = x2087;
x2103 = x198;
x2100 = x200.backward_data(x2102,x2103, x2101);

// V_cv32_W <~~ X2893 * d_Convolv(1,0)(X121)/d_cv32_W
float x2105, x2106;
x2105 = lrn_rate_1;
x2106 = momentum;
JCudaTensor x2107, x2108;
x2107 = x2087;
x2108 = x184;
x200.backward_filter(x2107, x2108, x2104, x2105, x2106);

// V_cv34_B <~~ X2917 * d_Convolv(1,0)()/d_cv34_B
float x2110, x2111;
x2110 = lrn_rate_2;
x2111 = momentum;
JCudaTensor x2112;
x2112 = x2091;
x210.backward_bias(x2112, x2109, x2110, x2111);

// V_cv32_B <~~ X2893 * d_Convolv(1,0)()/d_cv32_B
float x2114, x2115;
x2114 = lrn_rate_2;
x2115 = momentum;
JCudaTensor x2116;
x2116 = x2087;
x200.backward_bias(x2116, x2113, x2114, x2115);

// Dealloc(X2893)
JCudaTensor x2117;
x2117 = x2087;
x2117.free();

// cv32_W <~~ V_cv32_W
float x2118, x2119;
x2118 = 1;
x2119 = decay_1;
JCudaTensor x2120;
x2120 = x2104;
x198.update(x2120, x2118, x2119);

// cv34_B <~~ V_cv34_B
float x2121, x2122;
x2121 = 1;
x2122 = 1;
JCudaTensor x2123;
x2123 = x2109;
x209.update(x2123, x2121, x2122);

// cv32_B <~~ V_cv32_B
float x2124, x2125;
x2124 = 1;
x2125 = 1;
JCudaTensor x2126;
x2126 = x2113;
x199.update(x2126, x2124, x2125);

// val X2919 = (X2895 + X2917 * d_Convolv(1,0)(cv34_W)/d_X121)
JCudaTensor x2127;
JCudaTensor x2128;
x2128 = x2100;
JCudaTensor x2129, x2130;
x2129 = x2091;
x2130 = x208;
x2127 = x210.backward_data(x2129,x2130, x2128);

// Dealloc(X2917)
JCudaTensor x2131;
x2131 = x2091;
x2131.free();

// cv34_W <~~ V_cv34_W
float x2132, x2133;
x2132 = 1;
x2133 = decay_1;
JCudaTensor x2134;
x2134 = x2095;
x208.update(x2134, x2132, x2133);

// val X2941 = (X2919 + X2938 * d_Pooling(3,1,1,true)(X132,X121)/d_X121)
JCudaTensor x2135;
JCudaTensor x2136;
x2136 = x2127;
JCudaTensor x2137, x2138, x2139;
x2137 = x2051;
x2138 = x201;
x2139 = x184;
x2135 = x203.backward(x2137,x2138,x2139, x2136);

// Dealloc(X2938)
JCudaTensor x2140;
x2140 = x2051;
x2140.free();

// Dealloc(X132)
JCudaTensor x2141;
x2141 = x201;
x2141.free();

// val X2943 = X2941 * d_Pooling(3,2,1,true)(X121,X120)/d_X120
JCudaTensor x2142;
JCudaTensor x2143, x2144, x2145;
x2143 = x2135;
x2144 = x184;
x2145 = x179;
x2142 = x186.backward(x2143, x2144, x2145);

// Dealloc(X2941)
JCudaTensor x2146;
x2146 = x2135;
x2146.free();

// Dealloc(X121)
JCudaTensor x2147;
x2147 = x184;
x2147.free();

// Dealloc(X120)
JCudaTensor x2148;
x2148 = x179;
x2148.free();

// val X2957 = Proj(X2943, X108,X112,X116,X119, 0)
JCudaTensor x2149;
JCudaTensor x2151;
x2151 = x2142;
JCudaTensor[] x2150 = x119.backward(x2151);
x2149 = x2150[0];

// val X2999 = Proj(X2943, X108,X112,X116,X119, 2)
JCudaTensor x2152;
x2152 = x2150[2];

// val X2975 = Proj(X2943, X108,X112,X116,X119, 1)
JCudaTensor x2153;
x2153 = x2150[1];

// val X3023 = Proj(X2943, X108,X112,X116,X119, 3)
JCudaTensor x2154;
x2154 = x2150[3];

// Dealloc(X2943)
JCudaTensor x2155;
x2155 = x2142;
x2155.free();

// val X3027 = X3023 * d_ReLU()(X119)/d_X118
JCudaTensor x2156;
JCudaTensor x2157, x2158;
x2157 = x2154;
x2158 = x177;
x2156 = x115.backward(x2157, x2158);

// Dealloc(X119)
JCudaTensor x2159;
x2159 = x177;
x2159.free();

// val X3004 = X2999 * d_ReLU()(X116)/d_X115
JCudaTensor x2160;
JCudaTensor x2161, x2162;
x2161 = x2152;
x2162 = x175;
x2160 = x115.backward(x2161, x2162);

// Dealloc(X116)
JCudaTensor x2163;
x2163 = x175;
x2163.free();

// val X2980 = X2975 * d_ReLU()(X112)/d_X111
JCudaTensor x2164;
JCudaTensor x2165, x2166;
x2165 = x2153;
x2166 = x173;
x2164 = x112.backward(x2165, x2166);

// Dealloc(X112)
JCudaTensor x2167;
x2167 = x173;
x2167.free();

// val X2960 = X2957 * d_ReLU()(X108)/d_X107
JCudaTensor x2168;
JCudaTensor x2169, x2170;
x2169 = x2149;
x2170 = x171;
x2168 = x109.backward(x2169, x2170);

// Dealloc(X108)
JCudaTensor x2171;
x2171 = x171;
x2171.free();

// V_cv26_B <~~ X3027 * d_Convolv(1,0)()/d_cv26_B
float x2173, x2174;
x2173 = lrn_rate_2;
x2174 = momentum;
JCudaTensor x2175;
x2175 = x2156;
x156.backward_bias(x2175, x2172, x2173, x2174);

// val X3005 = X3004 * d_Convolv(1,2)(cv25_W)/d_X114
JCudaTensor x2176;
JCudaTensor x2177, x2178;
x2177 = x2160;
x2178 = x169;
x2176 = x106.backward_data(x2177, x2178);

// V_cv26_W <~~ X3027 * d_Convolv(1,0)(X117)/d_cv26_W
float x2180, x2181;
x2180 = lrn_rate_1;
x2181 = momentum;
JCudaTensor x2182, x2183;
x2182 = x2156;
x2183 = x124;
x156.backward_filter(x2182, x2183, x2179, x2180, x2181);

// V_cv25_W <~~ X3004 * d_Convolv(1,2)(X114)/d_cv25_W
float x2185, x2186;
x2185 = lrn_rate_1;
x2186 = momentum;
JCudaTensor x2187, x2188;
x2187 = x2160;
x2188 = x157;
x106.backward_filter(x2187, x2188, x2184, x2185, x2186);

// val X2981 = X2980 * d_Convolv(1,1)(cv23_W)/d_X110
JCudaTensor x2189;
JCudaTensor x2190, x2191;
x2190 = x2164;
x2191 = x163;
x2189 = x99.backward_data(x2190, x2191);

// V_cv21_W <~~ X2960 * d_Convolv(1,0)(X106)/d_cv21_W
float x2193, x2194;
x2193 = lrn_rate_1;
x2194 = momentum;
JCudaTensor x2195, x2196;
x2195 = x2168;
x2196 = x118;
x140.backward_filter(x2195, x2196, x2192, x2193, x2194);

// V_cv23_B <~~ X2980 * d_Convolv(1,1)()/d_cv23_B
float x2198, x2199;
x2198 = lrn_rate_2;
x2199 = momentum;
JCudaTensor x2200;
x2200 = x2164;
x99.backward_bias(x2200, x2197, x2198, x2199);

// V_cv25_B <~~ X3004 * d_Convolv(1,2)()/d_cv25_B
float x2202, x2203;
x2202 = lrn_rate_2;
x2203 = momentum;
JCudaTensor x2204;
x2204 = x2160;
x106.backward_bias(x2204, x2201, x2202, x2203);

// Dealloc(X3004)
JCudaTensor x2205;
x2205 = x2160;
x2205.free();

// V_cv21_B <~~ X2960 * d_Convolv(1,0)()/d_cv21_B
float x2207, x2208;
x2207 = lrn_rate_2;
x2208 = momentum;
JCudaTensor x2209;
x2209 = x2168;
x140.backward_bias(x2209, x2206, x2207, x2208);

// val X2961 = X2960 * d_Convolv(1,0)(cv21_W)/d_X106
JCudaTensor x2210;
JCudaTensor x2211, x2212;
x2211 = x2168;
x2212 = x138;
x2210 = x140.backward_data(x2211, x2212);

// Dealloc(X2960)
JCudaTensor x2213;
x2213 = x2168;
x2213.free();

// val X3028 = X3027 * d_Convolv(1,0)(cv26_W)/d_X117
JCudaTensor x2214;
JCudaTensor x2215, x2216;
x2215 = x2156;
x2216 = x154;
x2214 = x156.backward_data(x2215, x2216);

// Dealloc(X3027)
JCudaTensor x2217;
x2217 = x2156;
x2217.free();

// V_cv23_W <~~ X2980 * d_Convolv(1,1)(X110)/d_cv23_W
float x2219, x2220;
x2219 = lrn_rate_1;
x2220 = momentum;
JCudaTensor x2221, x2222;
x2221 = x2164;
x2222 = x148;
x99.backward_filter(x2221, x2222, x2218, x2219, x2220);

// Dealloc(X2980)
JCudaTensor x2223;
x2223 = x2164;
x2223.free();

// cv25_W <~~ V_cv25_W
float x2224, x2225;
x2224 = 1;
x2225 = decay_1;
JCudaTensor x2226;
x2226 = x2184;
x169.update(x2226, x2224, x2225);

// cv23_B <~~ V_cv23_B
float x2227, x2228;
x2227 = 1;
x2228 = 1;
JCudaTensor x2229;
x2229 = x2197;
x164.update(x2229, x2227, x2228);

// cv25_B <~~ V_cv25_B
float x2230, x2231;
x2230 = 1;
x2231 = 1;
JCudaTensor x2232;
x2232 = x2201;
x170.update(x2232, x2230, x2231);

// cv26_W <~~ V_cv26_W
float x2233, x2234;
x2233 = 1;
x2234 = decay_1;
JCudaTensor x2235;
x2235 = x2179;
x154.update(x2235, x2233, x2234);

// cv23_W <~~ V_cv23_W
float x2236, x2237;
x2236 = 1;
x2237 = decay_1;
JCudaTensor x2238;
x2238 = x2218;
x163.update(x2238, x2236, x2237);

// cv26_B <~~ V_cv26_B
float x2239, x2240;
x2239 = 1;
x2240 = 1;
JCudaTensor x2241;
x2241 = x2172;
x155.update(x2241, x2239, x2240);

// cv21_B <~~ V_cv21_B
float x2242, x2243;
x2242 = 1;
x2243 = 1;
JCudaTensor x2244;
x2244 = x2206;
x139.update(x2244, x2242, x2243);

// cv21_W <~~ V_cv21_W
float x2245, x2246;
x2245 = 1;
x2246 = decay_1;
JCudaTensor x2247;
x2247 = x2192;
x138.update(x2247, x2245, x2246);

// val X3007 = X3005 * d_ReLU()(X114)/d_X113
JCudaTensor x2248;
JCudaTensor x2249, x2250;
x2249 = x2176;
x2250 = x157;
x2248 = x92.backward(x2249, x2250);

// Dealloc(X114)
JCudaTensor x2251;
x2251 = x157;
x2251.free();

// val X2983 = X2981 * d_ReLU()(X110)/d_X109
JCudaTensor x2252;
JCudaTensor x2253, x2254;
x2253 = x2189;
x2254 = x148;
x2252 = x89.backward(x2253, x2254);

// Dealloc(X110)
JCudaTensor x2255;
x2255 = x148;
x2255.free();

// V_cv24_W <~~ X3007 * d_Convolv(1,0)(X106)/d_cv24_W
float x2257, x2258;
x2257 = lrn_rate_1;
x2258 = momentum;
JCudaTensor x2259, x2260;
x2259 = x2248;
x2260 = x118;
x133.backward_filter(x2259, x2260, x2256, x2257, x2258);

// V_cv24_B <~~ X3007 * d_Convolv(1,0)()/d_cv24_B
float x2262, x2263;
x2262 = lrn_rate_2;
x2263 = momentum;
JCudaTensor x2264;
x2264 = x2248;
x133.backward_bias(x2264, x2261, x2262, x2263);

// V_cv22_B <~~ X2983 * d_Convolv(1,0)()/d_cv22_B
float x2266, x2267;
x2266 = lrn_rate_2;
x2267 = momentum;
JCudaTensor x2268;
x2268 = x2252;
x147.backward_bias(x2268, x2265, x2266, x2267);

// val X2985 = (X2961 + X2983 * d_Convolv(1,0)(cv22_W)/d_X106)
JCudaTensor x2269;
JCudaTensor x2270;
x2270 = x2210;
JCudaTensor x2271, x2272;
x2271 = x2252;
x2272 = x145;
x2269 = x147.backward_data(x2271,x2272, x2270);

// V_cv22_W <~~ X2983 * d_Convolv(1,0)(X106)/d_cv22_W
float x2274, x2275;
x2274 = lrn_rate_1;
x2275 = momentum;
JCudaTensor x2276, x2277;
x2276 = x2252;
x2277 = x118;
x147.backward_filter(x2276, x2277, x2273, x2274, x2275);

// Dealloc(X2983)
JCudaTensor x2278;
x2278 = x2252;
x2278.free();

// cv24_B <~~ V_cv24_B
float x2279, x2280;
x2279 = 1;
x2280 = 1;
JCudaTensor x2281;
x2281 = x2261;
x132.update(x2281, x2279, x2280);

// cv22_B <~~ V_cv22_B
float x2282, x2283;
x2282 = 1;
x2283 = 1;
JCudaTensor x2284;
x2284 = x2265;
x146.update(x2284, x2282, x2283);

// cv22_W <~~ V_cv22_W
float x2285, x2286;
x2285 = 1;
x2286 = decay_1;
JCudaTensor x2287;
x2287 = x2273;
x145.update(x2287, x2285, x2286);

// val X3009 = (X2985 + X3007 * d_Convolv(1,0)(cv24_W)/d_X106)
JCudaTensor x2288;
JCudaTensor x2289;
x2289 = x2269;
JCudaTensor x2290, x2291;
x2290 = x2248;
x2291 = x131;
x2288 = x133.backward_data(x2290,x2291, x2289);

// Dealloc(X3007)
JCudaTensor x2292;
x2292 = x2248;
x2292.free();

// cv24_W <~~ V_cv24_W
float x2293, x2294;
x2293 = 1;
x2294 = decay_1;
JCudaTensor x2295;
x2295 = x2256;
x131.update(x2295, x2293, x2294);

// val X3031 = (X3009 + X3028 * d_Pooling(3,1,1,true)(X117,X106)/d_X106)
JCudaTensor x2296;
JCudaTensor x2297;
x2297 = x2288;
JCudaTensor x2298, x2299, x2300;
x2298 = x2214;
x2299 = x124;
x2300 = x118;
x2296 = x126.backward(x2298,x2299,x2300, x2297);

// Dealloc(X3028)
JCudaTensor x2301;
x2301 = x2214;
x2301.free();

// Dealloc(X117)
JCudaTensor x2302;
x2302 = x124;
x2302.free();

// Dealloc(X106)
JCudaTensor x2303;
x2303 = x118;
x2303.free();

// val X5817 = Proj(X3031, X94,X98,X102,X105, 1)
JCudaTensor x2304;
JCudaTensor x2306;
x2306 = x2296;
JCudaTensor[] x2305 = x119.backward(x2306);
x2304 = x2305[1];

// val X11379 = Proj(X3031, X94,X98,X102,X105, 2)
JCudaTensor x2307;
x2307 = x2305[2];

// val X16938 = Proj(X3031, X94,X98,X102,X105, 3)
JCudaTensor x2308;
x2308 = x2305[3];

// val X3045 = Proj(X3031, X94,X98,X102,X105, 0)
JCudaTensor x2309;
x2309 = x2305[0];

// Dealloc(X3031)
JCudaTensor x2310;
x2310 = x2296;
x2310.free();

// val X5822 = X5817 * d_ReLU()(X98)/d_X97
JCudaTensor x2311;
JCudaTensor x2312, x2313;
x2312 = x2304;
x2313 = x110;
x2311 = x112.backward(x2312, x2313);

// Dealloc(X98)
JCudaTensor x2314;
x2314 = x110;
x2314.free();

// val X3048 = X3045 * d_ReLU()(X94)/d_X93
JCudaTensor x2315;
JCudaTensor x2316, x2317;
x2316 = x2309;
x2317 = x107;
x2315 = x109.backward(x2316, x2317);

// Dealloc(X94)
JCudaTensor x2318;
x2318 = x107;
x2318.free();

// val X16942 = X16938 * d_ReLU()(X105)/d_X104
JCudaTensor x2319;
JCudaTensor x2320, x2321;
x2320 = x2308;
x2321 = x116;
x2319 = x115.backward(x2320, x2321);

// Dealloc(X105)
JCudaTensor x2322;
x2322 = x116;
x2322.free();

// val X11384 = X11379 * d_ReLU()(X102)/d_X101
JCudaTensor x2323;
JCudaTensor x2324, x2325;
x2324 = x2307;
x2325 = x113;
x2323 = x115.backward(x2324, x2325);

// Dealloc(X102)
JCudaTensor x2326;
x2326 = x113;
x2326.free();

// val X5823 = X5822 * d_Convolv(1,1)(cv13_W)/d_X96
JCudaTensor x2327;
JCudaTensor x2328, x2329;
x2328 = x2311;
x2329 = x97;
x2327 = x99.backward_data(x2328, x2329);

// V_cv11_W <~~ X3048 * d_Convolv(1,0)(X92)/d_cv11_W
float x2331, x2332;
x2331 = lrn_rate_1;
x2332 = momentum;
JCudaTensor x2333, x2334;
x2333 = x2315;
x2334 = x53;
x62.backward_filter(x2333, x2334, x2330, x2331, x2332);

// V_cv16_B <~~ X16942 * d_Convolv(1,0)()/d_cv16_B
float x2336, x2337;
x2336 = lrn_rate_2;
x2337 = momentum;
JCudaTensor x2338;
x2338 = x2319;
x86.backward_bias(x2338, x2335, x2336, x2337);

// val X11385 = X11384 * d_Convolv(1,2)(cv15_W)/d_X100
JCudaTensor x2339;
JCudaTensor x2340, x2341;
x2340 = x2323;
x2341 = x104;
x2339 = x106.backward_data(x2340, x2341);

// V_cv15_B <~~ X11384 * d_Convolv(1,2)()/d_cv15_B
float x2343, x2344;
x2343 = lrn_rate_2;
x2344 = momentum;
JCudaTensor x2345;
x2345 = x2323;
x106.backward_bias(x2345, x2342, x2343, x2344);

// V_cv16_W <~~ X16942 * d_Convolv(1,0)(X103)/d_cv16_W
float x2347, x2348;
x2347 = lrn_rate_1;
x2348 = momentum;
JCudaTensor x2349, x2350;
x2349 = x2319;
x2350 = x70;
x86.backward_filter(x2349, x2350, x2346, x2347, x2348);

// val X19749 = X3048 * d_Convolv(1,0)(cv11_W)/d_X92
JCudaTensor x2351;
JCudaTensor x2352, x2353;
x2352 = x2315;
x2353 = x60;
x2351 = x62.backward_data(x2352, x2353);

// V_cv13_B <~~ X5822 * d_Convolv(1,1)()/d_cv13_B
float x2355, x2356;
x2355 = lrn_rate_2;
x2356 = momentum;
JCudaTensor x2357;
x2357 = x2311;
x99.backward_bias(x2357, x2354, x2355, x2356);

// V_cv15_W <~~ X11384 * d_Convolv(1,2)(X100)/d_cv15_W
float x2359, x2360;
x2359 = lrn_rate_1;
x2360 = momentum;
JCudaTensor x2361, x2362;
x2361 = x2323;
x2362 = x90;
x106.backward_filter(x2361, x2362, x2358, x2359, x2360);

// Dealloc(X11384)
JCudaTensor x2363;
x2363 = x2323;
x2363.free();

// V_cv11_B <~~ X3048 * d_Convolv(1,0)()/d_cv11_B
float x2365, x2366;
x2365 = lrn_rate_2;
x2366 = momentum;
JCudaTensor x2367;
x2367 = x2315;
x62.backward_bias(x2367, x2364, x2365, x2366);

// Dealloc(X3048)
JCudaTensor x2368;
x2368 = x2315;
x2368.free();

// V_cv13_W <~~ X5822 * d_Convolv(1,1)(X96)/d_cv13_W
float x2370, x2371;
x2370 = lrn_rate_1;
x2371 = momentum;
JCudaTensor x2372, x2373;
x2372 = x2311;
x2373 = x87;
x99.backward_filter(x2372, x2373, x2369, x2370, x2371);

// Dealloc(X5822)
JCudaTensor x2374;
x2374 = x2311;
x2374.free();

// val X19816 = X16942 * d_Convolv(1,0)(cv16_W)/d_X103
JCudaTensor x2375;
JCudaTensor x2376, x2377;
x2376 = x2319;
x2377 = x84;
x2375 = x86.backward_data(x2376, x2377);

// Dealloc(X16942)
JCudaTensor x2378;
x2378 = x2319;
x2378.free();

// cv15_B <~~ V_cv15_B
float x2379, x2380;
x2379 = 1;
x2380 = 1;
JCudaTensor x2381;
x2381 = x2342;
x105.update(x2381, x2379, x2380);

// cv13_B <~~ V_cv13_B
float x2382, x2383;
x2382 = 1;
x2383 = 1;
JCudaTensor x2384;
x2384 = x2354;
x98.update(x2384, x2382, x2383);

// cv11_W <~~ V_cv11_W
float x2385, x2386;
x2385 = 1;
x2386 = decay_1;
JCudaTensor x2387;
x2387 = x2330;
x60.update(x2387, x2385, x2386);

// cv11_B <~~ V_cv11_B
float x2388, x2389;
x2388 = 1;
x2389 = 1;
JCudaTensor x2390;
x2390 = x2364;
x61.update(x2390, x2388, x2389);

// cv13_W <~~ V_cv13_W
float x2391, x2392;
x2391 = 1;
x2392 = decay_1;
JCudaTensor x2393;
x2393 = x2369;
x97.update(x2393, x2391, x2392);

// cv16_W <~~ V_cv16_W
float x2394, x2395;
x2394 = 1;
x2395 = decay_1;
JCudaTensor x2396;
x2396 = x2346;
x84.update(x2396, x2394, x2395);

// cv16_B <~~ V_cv16_B
float x2397, x2398;
x2397 = 1;
x2398 = 1;
JCudaTensor x2399;
x2399 = x2335;
x85.update(x2399, x2397, x2398);

// cv15_W <~~ V_cv15_W
float x2400, x2401;
x2400 = 1;
x2401 = decay_1;
JCudaTensor x2402;
x2402 = x2358;
x104.update(x2402, x2400, x2401);

// val X11387 = X11385 * d_ReLU()(X100)/d_X99
JCudaTensor x2403;
JCudaTensor x2404, x2405;
x2404 = x2339;
x2405 = x90;
x2403 = x92.backward(x2404, x2405);

// Dealloc(X100)
JCudaTensor x2406;
x2406 = x90;
x2406.free();

// val X5825 = X5823 * d_ReLU()(X96)/d_X95
JCudaTensor x2407;
JCudaTensor x2408, x2409;
x2408 = x2327;
x2409 = x87;
x2407 = x89.backward(x2408, x2409);

// Dealloc(X96)
JCudaTensor x2410;
x2410 = x87;
x2410.free();

// V_cv14_W <~~ X11387 * d_Convolv(1,0)(X92)/d_cv14_W
float x2412, x2413;
x2412 = lrn_rate_1;
x2413 = momentum;
JCudaTensor x2414, x2415;
x2414 = x2403;
x2415 = x53;
x79.backward_filter(x2414, x2415, x2411, x2412, x2413);

// V_cv12_B <~~ X5825 * d_Convolv(1,0)()/d_cv12_B
float x2417, x2418;
x2417 = lrn_rate_2;
x2418 = momentum;
JCudaTensor x2419;
x2419 = x2407;
x69.backward_bias(x2419, x2416, x2417, x2418);

// val X19773 = (X19749 + X5825 * d_Convolv(1,0)(cv12_W)/d_X92)
JCudaTensor x2420;
JCudaTensor x2421;
x2421 = x2351;
JCudaTensor x2422, x2423;
x2422 = x2407;
x2423 = x67;
x2420 = x69.backward_data(x2422,x2423, x2421);

// V_cv12_W <~~ X5825 * d_Convolv(1,0)(X92)/d_cv12_W
float x2425, x2426;
x2425 = lrn_rate_1;
x2426 = momentum;
JCudaTensor x2427, x2428;
x2427 = x2407;
x2428 = x53;
x69.backward_filter(x2427, x2428, x2424, x2425, x2426);

// Dealloc(X5825)
JCudaTensor x2429;
x2429 = x2407;
x2429.free();

// V_cv14_B <~~ X11387 * d_Convolv(1,0)()/d_cv14_B
float x2431, x2432;
x2431 = lrn_rate_2;
x2432 = momentum;
JCudaTensor x2433;
x2433 = x2403;
x79.backward_bias(x2433, x2430, x2431, x2432);

// cv12_B <~~ V_cv12_B
float x2434, x2435;
x2434 = 1;
x2435 = 1;
JCudaTensor x2436;
x2436 = x2416;
x68.update(x2436, x2434, x2435);

// cv12_W <~~ V_cv12_W
float x2437, x2438;
x2437 = 1;
x2438 = decay_1;
JCudaTensor x2439;
x2439 = x2424;
x67.update(x2439, x2437, x2438);

// cv14_B <~~ V_cv14_B
float x2440, x2441;
x2440 = 1;
x2441 = 1;
JCudaTensor x2442;
x2442 = x2430;
x78.update(x2442, x2440, x2441);

// val X19797 = (X19773 + X11387 * d_Convolv(1,0)(cv14_W)/d_X92)
JCudaTensor x2443;
JCudaTensor x2444;
x2444 = x2420;
JCudaTensor x2445, x2446;
x2445 = x2403;
x2446 = x77;
x2443 = x79.backward_data(x2445,x2446, x2444);

// Dealloc(X11387)
JCudaTensor x2447;
x2447 = x2403;
x2447.free();

// cv14_W <~~ V_cv14_W
float x2448, x2449;
x2448 = 1;
x2449 = decay_1;
JCudaTensor x2450;
x2450 = x2411;
x77.update(x2450, x2448, x2449);

// val X19819 = (X19797 + X19816 * d_Pooling(3,1,1,true)(X103,X92)/d_X92)
JCudaTensor x2451;
JCudaTensor x2452;
x2452 = x2443;
JCudaTensor x2453, x2454, x2455;
x2453 = x2375;
x2454 = x70;
x2455 = x53;
x2451 = x72.backward(x2453,x2454,x2455, x2452);

// Dealloc(X19816)
JCudaTensor x2456;
x2456 = x2375;
x2456.free();

// Dealloc(X103)
JCudaTensor x2457;
x2457 = x70;
x2457.free();

// val X19821 = X19819 * d_Pooling(3,2,1,true)(X92,X91)/d_X91
JCudaTensor x2458;
JCudaTensor x2459, x2460, x2461;
x2459 = x2451;
x2460 = x53;
x2461 = x50;
x2458 = x55.backward(x2459, x2460, x2461);

// Dealloc(X19819)
JCudaTensor x2462;
x2462 = x2451;
x2462.free();

// Dealloc(X92)
JCudaTensor x2463;
x2463 = x53;
x2463.free();

// val X19823 = X19821 * d_LRN(5,1.0E-4,0.75)(X91,X90)/d_X90
JCudaTensor x2464;
JCudaTensor x2465, x2466, x2467;
x2465 = x2458;
x2466 = x50;
x2467 = x47;
x2464 = x52.backward(x2465, x2466, x2467);

// Dealloc(X91)
JCudaTensor x2468;
x2468 = x50;
x2468.free();

// val X19825 = X19823 * d_ReLU()(X90)/d_X89
JCudaTensor x2469;
JCudaTensor x2470, x2471;
x2470 = x2464;
x2471 = x47;
x2469 = x49.backward(x2470, x2471);

// Dealloc(X90)
JCudaTensor x2472;
x2472 = x47;
x2472.free();

// V_cv3_B <~~ X19825 * d_Convolv(1,1)()/d_cv3_B
float x2474, x2475;
x2474 = lrn_rate_2;
x2475 = momentum;
JCudaTensor x2476;
x2476 = x2469;
x46.backward_bias(x2476, x2473, x2474, x2475);

// val X19826 = X19825 * d_Convolv(1,1)(cv3_W)/d_X88
JCudaTensor x2477;
JCudaTensor x2478, x2479;
x2478 = x2469;
x2479 = x44;
x2477 = x46.backward_data(x2478, x2479);

// V_cv3_W <~~ X19825 * d_Convolv(1,1)(X88)/d_cv3_W
float x2481, x2482;
x2481 = lrn_rate_1;
x2482 = momentum;
JCudaTensor x2483, x2484;
x2483 = x2469;
x2484 = x37;
x46.backward_filter(x2483, x2484, x2480, x2481, x2482);

// Dealloc(X19825)
JCudaTensor x2485;
x2485 = x2469;
x2485.free();

// cv3_B <~~ V_cv3_B
float x2486, x2487;
x2486 = 1;
x2487 = 1;
JCudaTensor x2488;
x2488 = x2473;
x45.update(x2488, x2486, x2487);

// cv3_W <~~ V_cv3_W
float x2489, x2490;
x2489 = 1;
x2490 = decay_1;
JCudaTensor x2491;
x2491 = x2480;
x44.update(x2491, x2489, x2490);

// val X19828 = X19826 * d_ReLU()(X88)/d_X87
JCudaTensor x2492;
JCudaTensor x2493, x2494;
x2493 = x2477;
x2494 = x37;
x2492 = x39.backward(x2493, x2494);

// Dealloc(X88)
JCudaTensor x2495;
x2495 = x37;
x2495.free();

// V_cv2_B <~~ X19828 * d_Convolv(1,0)()/d_cv2_B
float x2497, x2498;
x2497 = lrn_rate_2;
x2498 = momentum;
JCudaTensor x2499;
x2499 = x2492;
x36.backward_bias(x2499, x2496, x2497, x2498);

// val X19829 = X19828 * d_Convolv(1,0)(cv2_W)/d_X86
JCudaTensor x2500;
JCudaTensor x2501, x2502;
x2501 = x2492;
x2502 = x34;
x2500 = x36.backward_data(x2501, x2502);

// V_cv2_W <~~ X19828 * d_Convolv(1,0)(X86)/d_cv2_W
float x2504, x2505;
x2504 = lrn_rate_1;
x2505 = momentum;
JCudaTensor x2506, x2507;
x2506 = x2492;
x2507 = x27;
x36.backward_filter(x2506, x2507, x2503, x2504, x2505);

// Dealloc(X19828)
JCudaTensor x2508;
x2508 = x2492;
x2508.free();

// cv2_B <~~ V_cv2_B
float x2509, x2510;
x2509 = 1;
x2510 = 1;
JCudaTensor x2511;
x2511 = x2496;
x35.update(x2511, x2509, x2510);

// cv2_W <~~ V_cv2_W
float x2512, x2513;
x2512 = 1;
x2513 = decay_1;
JCudaTensor x2514;
x2514 = x2503;
x34.update(x2514, x2512, x2513);

// val X19831 = X19829 * d_LRN(5,1.0E-4,0.75)(X86,X85)/d_X85
JCudaTensor x2515;
JCudaTensor x2516, x2517, x2518;
x2516 = x2500;
x2517 = x27;
x2518 = x24;
x2515 = x29.backward(x2516, x2517, x2518);

// Dealloc(X86)
JCudaTensor x2519;
x2519 = x27;
x2519.free();

// val X19833 = X19831 * d_Pooling(3,2,1,true)(X85,X84)/d_X84
JCudaTensor x2520;
JCudaTensor x2521, x2522, x2523;
x2521 = x2515;
x2522 = x24;
x2523 = x21;
x2520 = x26.backward(x2521, x2522, x2523);

// Dealloc(X19831)
JCudaTensor x2524;
x2524 = x2515;
x2524.free();

// Dealloc(X85)
JCudaTensor x2525;
x2525 = x24;
x2525.free();

// val X19835 = X19833 * d_ReLU()(X84)/d_X83
JCudaTensor x2526;
JCudaTensor x2527, x2528;
x2527 = x2520;
x2528 = x21;
x2526 = x23.backward(x2527, x2528);

// Dealloc(X84)
JCudaTensor x2529;
x2529 = x21;
x2529.free();

// V_cv1_W <~~ X19835 * d_Convolv(2,3)(X82)/d_cv1_W
float x2531, x2532;
x2531 = lrn_rate_1;
x2532 = momentum;
JCudaTensor x2533, x2534;
x2533 = x2526;
x2534 = x7;
x17.backward_filter(x2533, x2534, x2530, x2531, x2532);

// Dealloc(X82)
JCudaTensor x2535;
x2535 = x7;
x2535.free();

// V_cv1_B <~~ X19835 * d_Convolv(2,3)()/d_cv1_B
float x2537, x2538;
x2537 = lrn_rate_2;
x2538 = momentum;
JCudaTensor x2539;
x2539 = x2526;
x17.backward_bias(x2539, x2536, x2537, x2538);

// Dealloc(X19835)
JCudaTensor x2540;
x2540 = x2526;
x2540.free();

// cv1_W <~~ V_cv1_W
float x2541, x2542;
x2541 = 1;
x2542 = decay_1;
JCudaTensor x2543;
x2543 = x2530;
x15.update(x2543, x2541, x2542);

// cv1_B <~~ V_cv1_B
float x2544, x2545;
x2544 = 1;
x2545 = 1;
JCudaTensor x2546;
x2546 = x2536;
x16.update(x2546, x2544, x2545);

}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X98544 = Cuda(X)
JCudaTensor x2547;
JTensorFloat x2548;
x2548 = x3;
x2547 = x2548.asJCudaTensor();

// val X98545 = Convolv(2,3)(X98544,cv1_W,cv1_B)
JCudaTensor x2549;
JCudaTensor x2550, x2551, x2552;
x2550 = x2547;
x2551 = x15;
x2552 = x16;
x2549 = x17.forward(x2550, x2551, x2552);

// Dealloc(X98544)
JCudaTensor x2553;
x2553 = x2547;
x2553.free();

// val X98546 = ReLU()(X98545)
JCudaTensor x2554;
JCudaTensor x2555;
x2555 = x2549;
x2554 = x23.forward(x2555);

// val X98547 = Pooling(3,2,1,true)(X98546)
JCudaTensor x2556;
JCudaTensor x2557;
x2557 = x2554;
x2556 = x26.forward(x2557);

// Dealloc(X98546)
JCudaTensor x2558;
x2558 = x2554;
x2558.free();

// val X98548 = LRN(5,1.0E-4,0.75)(X98547)
JCudaTensor x2559;
JCudaTensor x2560;
x2560 = x2556;
x2559 = x29.forward(x2560);

// Dealloc(X98547)
JCudaTensor x2561;
x2561 = x2556;
x2561.free();

// val X98549 = Convolv(1,0)(X98548,cv2_W,cv2_B)
JCudaTensor x2562;
JCudaTensor x2563, x2564, x2565;
x2563 = x2559;
x2564 = x34;
x2565 = x35;
x2562 = x36.forward(x2563, x2564, x2565);

// Dealloc(X98548)
JCudaTensor x2566;
x2566 = x2559;
x2566.free();

// val X98550 = ReLU()(X98549)
JCudaTensor x2567;
JCudaTensor x2568;
x2568 = x2562;
x2567 = x39.forward(x2568);

// val X98551 = Convolv(1,1)(X98550,cv3_W,cv3_B)
JCudaTensor x2569;
JCudaTensor x2570, x2571, x2572;
x2570 = x2567;
x2571 = x44;
x2572 = x45;
x2569 = x46.forward(x2570, x2571, x2572);

// Dealloc(X98550)
JCudaTensor x2573;
x2573 = x2567;
x2573.free();

// val X98552 = ReLU()(X98551)
JCudaTensor x2574;
JCudaTensor x2575;
x2575 = x2569;
x2574 = x49.forward(x2575);

// val X98553 = LRN(5,1.0E-4,0.75)(X98552)
JCudaTensor x2576;
JCudaTensor x2577;
x2577 = x2574;
x2576 = x52.forward(x2577);

// Dealloc(X98552)
JCudaTensor x2578;
x2578 = x2574;
x2578.free();

// val X98554 = Pooling(3,2,1,true)(X98553)
JCudaTensor x2579;
JCudaTensor x2580;
x2580 = x2576;
x2579 = x55.forward(x2580);

// Dealloc(X98553)
JCudaTensor x2581;
x2581 = x2576;
x2581.free();

// val X98557 = Convolv(1,0)(X98554,cv12_W,cv12_B)
JCudaTensor x2582;
JCudaTensor x2583, x2584, x2585;
x2583 = x2579;
x2584 = x67;
x2585 = x68;
x2582 = x69.forward(x2583, x2584, x2585);

// val X98561 = Convolv(1,0)(X98554,cv14_W,cv14_B)
JCudaTensor x2586;
JCudaTensor x2587, x2588, x2589;
x2587 = x2579;
x2588 = x77;
x2589 = x78;
x2586 = x79.forward(x2587, x2588, x2589);

// val X98555 = Convolv(1,0)(X98554,cv11_W,cv11_B)
JCudaTensor x2590;
JCudaTensor x2591, x2592, x2593;
x2591 = x2579;
x2592 = x60;
x2593 = x61;
x2590 = x62.forward(x2591, x2592, x2593);

// val X98565 = Pooling(3,1,1,true)(X98554)
JCudaTensor x2594;
JCudaTensor x2595;
x2595 = x2579;
x2594 = x72.forward(x2595);

// Dealloc(X98554)
JCudaTensor x2596;
x2596 = x2579;
x2596.free();

// val X98558 = ReLU()(X98557)
JCudaTensor x2597;
JCudaTensor x2598;
x2598 = x2582;
x2597 = x89.forward(x2598);

// val X98562 = ReLU()(X98561)
JCudaTensor x2599;
JCudaTensor x2600;
x2600 = x2586;
x2599 = x92.forward(x2600);

// val X98566 = Convolv(1,0)(X98565,cv16_W,cv16_B)
JCudaTensor x2601;
JCudaTensor x2602, x2603, x2604;
x2602 = x2594;
x2603 = x84;
x2604 = x85;
x2601 = x86.forward(x2602, x2603, x2604);

// Dealloc(X98565)
JCudaTensor x2605;
x2605 = x2594;
x2605.free();

// val X98559 = Convolv(1,1)(X98558,cv13_W,cv13_B)
JCudaTensor x2606;
JCudaTensor x2607, x2608, x2609;
x2607 = x2597;
x2608 = x97;
x2609 = x98;
x2606 = x99.forward(x2607, x2608, x2609);

// Dealloc(X98558)
JCudaTensor x2610;
x2610 = x2597;
x2610.free();

// val X98563 = Convolv(1,2)(X98562,cv15_W,cv15_B)
JCudaTensor x2611;
JCudaTensor x2612, x2613, x2614;
x2612 = x2599;
x2613 = x104;
x2614 = x105;
x2611 = x106.forward(x2612, x2613, x2614);

// Dealloc(X98562)
JCudaTensor x2615;
x2615 = x2599;
x2615.free();

// val X98556 = ReLU()(X98555)
JCudaTensor x2616;
JCudaTensor x2617;
x2617 = x2590;
x2616 = x109.forward(x2617);

// val X98560 = ReLU()(X98559)
JCudaTensor x2618;
JCudaTensor x2619;
x2619 = x2606;
x2618 = x112.forward(x2619);

// val X98564 = ReLU()(X98563)
JCudaTensor x2620;
JCudaTensor x2621;
x2621 = x2611;
x2620 = x115.forward(x2621);

// val X98567 = ReLU()(X98566)
JCudaTensor x2622;
JCudaTensor x2623;
x2623 = x2601;
x2622 = x115.forward(x2623);

// val X98568 = Concat(X98556,X98560,X98564,X98567)
JCudaTensor x2624;
JCudaTensor x2625, x2626, x2627, x2628;
x2625 = x2616;
x2626 = x2618;
x2627 = x2620;
x2628 = x2622;
x2624 = x119.forward(x2625,x2626,x2627,x2628);

// Dealloc(X98567)
JCudaTensor x2629;
x2629 = x2622;
x2629.free();

// Dealloc(X98564)
JCudaTensor x2630;
x2630 = x2620;
x2630.free();

// Dealloc(X98560)
JCudaTensor x2631;
x2631 = x2618;
x2631.free();

// Dealloc(X98556)
JCudaTensor x2632;
x2632 = x2616;
x2632.free();

// val X98579 = Pooling(3,1,1,true)(X98568)
JCudaTensor x2633;
JCudaTensor x2634;
x2634 = x2624;
x2633 = x126.forward(x2634);

// val X98569 = Convolv(1,0)(X98568,cv21_W,cv21_B)
JCudaTensor x2635;
JCudaTensor x2636, x2637, x2638;
x2636 = x2624;
x2637 = x138;
x2638 = x139;
x2635 = x140.forward(x2636, x2637, x2638);

// val X98575 = Convolv(1,0)(X98568,cv24_W,cv24_B)
JCudaTensor x2639;
JCudaTensor x2640, x2641, x2642;
x2640 = x2624;
x2641 = x131;
x2642 = x132;
x2639 = x133.forward(x2640, x2641, x2642);

// val X98571 = Convolv(1,0)(X98568,cv22_W,cv22_B)
JCudaTensor x2643;
JCudaTensor x2644, x2645, x2646;
x2644 = x2624;
x2645 = x145;
x2646 = x146;
x2643 = x147.forward(x2644, x2645, x2646);

// Dealloc(X98568)
JCudaTensor x2647;
x2647 = x2624;
x2647.free();

// val X98580 = Convolv(1,0)(X98579,cv26_W,cv26_B)
JCudaTensor x2648;
JCudaTensor x2649, x2650, x2651;
x2649 = x2633;
x2650 = x154;
x2651 = x155;
x2648 = x156.forward(x2649, x2650, x2651);

// Dealloc(X98579)
JCudaTensor x2652;
x2652 = x2633;
x2652.free();

// val X98572 = ReLU()(X98571)
JCudaTensor x2653;
JCudaTensor x2654;
x2654 = x2643;
x2653 = x89.forward(x2654);

// val X98576 = ReLU()(X98575)
JCudaTensor x2655;
JCudaTensor x2656;
x2656 = x2639;
x2655 = x92.forward(x2656);

// val X98573 = Convolv(1,1)(X98572,cv23_W,cv23_B)
JCudaTensor x2657;
JCudaTensor x2658, x2659, x2660;
x2658 = x2653;
x2659 = x163;
x2660 = x164;
x2657 = x99.forward(x2658, x2659, x2660);

// Dealloc(X98572)
JCudaTensor x2661;
x2661 = x2653;
x2661.free();

// val X98577 = Convolv(1,2)(X98576,cv25_W,cv25_B)
JCudaTensor x2662;
JCudaTensor x2663, x2664, x2665;
x2663 = x2655;
x2664 = x169;
x2665 = x170;
x2662 = x106.forward(x2663, x2664, x2665);

// Dealloc(X98576)
JCudaTensor x2666;
x2666 = x2655;
x2666.free();

// val X98570 = ReLU()(X98569)
JCudaTensor x2667;
JCudaTensor x2668;
x2668 = x2635;
x2667 = x109.forward(x2668);

// val X98574 = ReLU()(X98573)
JCudaTensor x2669;
JCudaTensor x2670;
x2670 = x2657;
x2669 = x112.forward(x2670);

// val X98578 = ReLU()(X98577)
JCudaTensor x2671;
JCudaTensor x2672;
x2672 = x2662;
x2671 = x115.forward(x2672);

// val X98581 = ReLU()(X98580)
JCudaTensor x2673;
JCudaTensor x2674;
x2674 = x2648;
x2673 = x115.forward(x2674);

// val X98582 = Concat(X98570,X98574,X98578,X98581)
JCudaTensor x2675;
JCudaTensor x2676, x2677, x2678, x2679;
x2676 = x2667;
x2677 = x2669;
x2678 = x2671;
x2679 = x2673;
x2675 = x119.forward(x2676,x2677,x2678,x2679);

// Dealloc(X98581)
JCudaTensor x2680;
x2680 = x2673;
x2680.free();

// Dealloc(X98578)
JCudaTensor x2681;
x2681 = x2671;
x2681.free();

// Dealloc(X98574)
JCudaTensor x2682;
x2682 = x2669;
x2682.free();

// Dealloc(X98570)
JCudaTensor x2683;
x2683 = x2667;
x2683.free();

// val X98583 = Pooling(3,2,1,true)(X98582)
JCudaTensor x2684;
JCudaTensor x2685;
x2685 = x2675;
x2684 = x186.forward(x2685);

// Dealloc(X98582)
JCudaTensor x2686;
x2686 = x2675;
x2686.free();

// val X98590 = Convolv(1,0)(X98583,cv34_W,cv34_B)
JCudaTensor x2687;
JCudaTensor x2688, x2689, x2690;
x2688 = x2684;
x2689 = x208;
x2690 = x209;
x2687 = x210.forward(x2688, x2689, x2690);

// val X98594 = Pooling(3,1,1,true)(X98583)
JCudaTensor x2691;
JCudaTensor x2692;
x2692 = x2684;
x2691 = x203.forward(x2692);

// val X98586 = Convolv(1,0)(X98583,cv32_W,cv32_B)
JCudaTensor x2693;
JCudaTensor x2694, x2695, x2696;
x2694 = x2684;
x2695 = x198;
x2696 = x199;
x2693 = x200.forward(x2694, x2695, x2696);

// val X98584 = Convolv(1,0)(X98583,cv31_W,cv31_B)
JCudaTensor x2697;
JCudaTensor x2698, x2699, x2700;
x2698 = x2684;
x2699 = x191;
x2700 = x192;
x2697 = x193.forward(x2698, x2699, x2700);

// Dealloc(X98583)
JCudaTensor x2701;
x2701 = x2684;
x2701.free();

// val X98591 = ReLU()(X98590)
JCudaTensor x2702;
JCudaTensor x2703;
x2703 = x2687;
x2702 = x213.forward(x2703);

// val X98595 = Convolv(1,0)(X98594,cv36_W,cv36_B)
JCudaTensor x2704;
JCudaTensor x2705, x2706, x2707;
x2705 = x2691;
x2706 = x221;
x2707 = x222;
x2704 = x223.forward(x2705, x2706, x2707);

// Dealloc(X98594)
JCudaTensor x2708;
x2708 = x2691;
x2708.free();

// val X98587 = ReLU()(X98586)
JCudaTensor x2709;
JCudaTensor x2710;
x2710 = x2693;
x2709 = x216.forward(x2710);

// val X98588 = Convolv(1,1)(X98587,cv33_W,cv33_B)
JCudaTensor x2711;
JCudaTensor x2712, x2713, x2714;
x2712 = x2709;
x2713 = x235;
x2714 = x236;
x2711 = x237.forward(x2712, x2713, x2714);

// Dealloc(X98587)
JCudaTensor x2715;
x2715 = x2709;
x2715.free();

// val X98592 = Convolv(1,2)(X98591,cv35_W,cv35_B)
JCudaTensor x2716;
JCudaTensor x2717, x2718, x2719;
x2717 = x2702;
x2718 = x228;
x2719 = x229;
x2716 = x230.forward(x2717, x2718, x2719);

// Dealloc(X98591)
JCudaTensor x2720;
x2720 = x2702;
x2720.free();

// val X98585 = ReLU()(X98584)
JCudaTensor x2721;
JCudaTensor x2722;
x2722 = x2697;
x2721 = x240.forward(x2722);

// val X98589 = ReLU()(X98588)
JCudaTensor x2723;
JCudaTensor x2724;
x2724 = x2711;
x2723 = x243.forward(x2724);

// val X98593 = ReLU()(X98592)
JCudaTensor x2725;
JCudaTensor x2726;
x2726 = x2716;
x2725 = x246.forward(x2726);

// val X98596 = ReLU()(X98595)
JCudaTensor x2727;
JCudaTensor x2728;
x2728 = x2704;
x2727 = x246.forward(x2728);

// val X98597 = Concat(X98585,X98589,X98593,X98596)
JCudaTensor x2729;
JCudaTensor x2730, x2731, x2732, x2733;
x2730 = x2721;
x2731 = x2723;
x2732 = x2725;
x2733 = x2727;
x2729 = x250.forward(x2730,x2731,x2732,x2733);

// Dealloc(X98596)
JCudaTensor x2734;
x2734 = x2727;
x2734.free();

// Dealloc(X98593)
JCudaTensor x2735;
x2735 = x2725;
x2735.free();

// Dealloc(X98589)
JCudaTensor x2736;
x2736 = x2723;
x2736.free();

// Dealloc(X98585)
JCudaTensor x2737;
x2737 = x2721;
x2737.free();

// val X98598 = Convolv(1,0)(X98597,cv41_W,cv41_B)
JCudaTensor x2738;
JCudaTensor x2739, x2740, x2741;
x2739 = x2729;
x2740 = x276;
x2741 = x277;
x2738 = x193.forward(x2739, x2740, x2741);

// val X98608 = Pooling(3,1,1,true)(X98597)
JCudaTensor x2742;
JCudaTensor x2743;
x2743 = x2729;
x2742 = x203.forward(x2743);

// val X98604 = Convolv(1,0)(X98597,cv44_W,cv44_B)
JCudaTensor x2744;
JCudaTensor x2745, x2746, x2747;
x2745 = x2729;
x2746 = x262;
x2747 = x263;
x2744 = x210.forward(x2745, x2746, x2747);

// val X98600 = Convolv(1,0)(X98597,cv42_W,cv42_B)
JCudaTensor x2748;
JCudaTensor x2749, x2750, x2751;
x2749 = x2729;
x2750 = x270;
x2751 = x271;
x2748 = x200.forward(x2749, x2750, x2751);

// Dealloc(X98597)
JCudaTensor x2752;
x2752 = x2729;
x2752.free();

// val X98601 = ReLU()(X98600)
JCudaTensor x2753;
JCudaTensor x2754;
x2754 = x2748;
x2753 = x216.forward(x2754);

// val X98605 = ReLU()(X98604)
JCudaTensor x2755;
JCudaTensor x2756;
x2756 = x2744;
x2755 = x213.forward(x2756);

// val X98609 = Convolv(1,0)(X98608,cv46_W,cv46_B)
JCudaTensor x2757;
JCudaTensor x2758, x2759, x2760;
x2758 = x2742;
x2759 = x293;
x2760 = x294;
x2757 = x223.forward(x2758, x2759, x2760);

// Dealloc(X98608)
JCudaTensor x2761;
x2761 = x2742;
x2761.free();

// val X98606 = Convolv(1,2)(X98605,cv45_W,cv45_B)
JCudaTensor x2762;
JCudaTensor x2763, x2764, x2765;
x2763 = x2755;
x2764 = x299;
x2765 = x300;
x2762 = x230.forward(x2763, x2764, x2765);

// Dealloc(X98605)
JCudaTensor x2766;
x2766 = x2755;
x2766.free();

// val X98602 = Convolv(1,1)(X98601,cv43_W,cv43_B)
JCudaTensor x2767;
JCudaTensor x2768, x2769, x2770;
x2768 = x2753;
x2769 = x308;
x2770 = x309;
x2767 = x237.forward(x2768, x2769, x2770);

// Dealloc(X98601)
JCudaTensor x2771;
x2771 = x2753;
x2771.free();

// val X98599 = ReLU()(X98598)
JCudaTensor x2772;
JCudaTensor x2773;
x2773 = x2738;
x2772 = x240.forward(x2773);

// val X98603 = ReLU()(X98602)
JCudaTensor x2774;
JCudaTensor x2775;
x2775 = x2767;
x2774 = x243.forward(x2775);

// val X98607 = ReLU()(X98606)
JCudaTensor x2776;
JCudaTensor x2777;
x2777 = x2762;
x2776 = x246.forward(x2777);

// val X98610 = ReLU()(X98609)
JCudaTensor x2778;
JCudaTensor x2779;
x2779 = x2757;
x2778 = x246.forward(x2779);

// val X98611 = Concat(X98599,X98603,X98607,X98610)
JCudaTensor x2780;
JCudaTensor x2781, x2782, x2783, x2784;
x2781 = x2772;
x2782 = x2774;
x2783 = x2776;
x2784 = x2778;
x2780 = x250.forward(x2781,x2782,x2783,x2784);

// Dealloc(X98610)
JCudaTensor x2785;
x2785 = x2778;
x2785.free();

// Dealloc(X98607)
JCudaTensor x2786;
x2786 = x2776;
x2786.free();

// Dealloc(X98603)
JCudaTensor x2787;
x2787 = x2774;
x2787.free();

// Dealloc(X98599)
JCudaTensor x2788;
x2788 = x2772;
x2788.free();

// val X98612 = Convolv(1,0)(X98611,cv51_W,cv51_B)
JCudaTensor x2789;
JCudaTensor x2790, x2791, x2792;
x2790 = x2780;
x2791 = x349;
x2792 = x350;
x2789 = x193.forward(x2790, x2791, x2792);

// val X98622 = Pooling(3,1,1,true)(X98611)
JCudaTensor x2793;
JCudaTensor x2794;
x2794 = x2780;
x2793 = x203.forward(x2794);

// val X98614 = Convolv(1,0)(X98611,cv52_W,cv52_B)
JCudaTensor x2795;
JCudaTensor x2796, x2797, x2798;
x2796 = x2780;
x2797 = x338;
x2798 = x339;
x2795 = x200.forward(x2796, x2797, x2798);

// val X98618 = Convolv(1,0)(X98611,cv54_W,cv54_B)
JCudaTensor x2799;
JCudaTensor x2800, x2801, x2802;
x2800 = x2780;
x2801 = x355;
x2802 = x356;
x2799 = x210.forward(x2800, x2801, x2802);

// Dealloc(X98611)
JCudaTensor x2803;
x2803 = x2780;
x2803.free();

// val X98623 = Convolv(1,0)(X98622,cv56_W,cv56_B)
JCudaTensor x2804;
JCudaTensor x2805, x2806, x2807;
x2805 = x2793;
x2806 = x366;
x2807 = x367;
x2804 = x223.forward(x2805, x2806, x2807);

// Dealloc(X98622)
JCudaTensor x2808;
x2808 = x2793;
x2808.free();

// val X98619 = ReLU()(X98618)
JCudaTensor x2809;
JCudaTensor x2810;
x2810 = x2799;
x2809 = x213.forward(x2810);

// val X98615 = ReLU()(X98614)
JCudaTensor x2811;
JCudaTensor x2812;
x2812 = x2795;
x2811 = x216.forward(x2812);

// val X98616 = Convolv(1,1)(X98615,cv53_W,cv53_B)
JCudaTensor x2813;
JCudaTensor x2814, x2815, x2816;
x2814 = x2811;
x2815 = x374;
x2816 = x375;
x2813 = x237.forward(x2814, x2815, x2816);

// Dealloc(X98615)
JCudaTensor x2817;
x2817 = x2811;
x2817.free();

// val X98620 = Convolv(1,2)(X98619,cv55_W,cv55_B)
JCudaTensor x2818;
JCudaTensor x2819, x2820, x2821;
x2819 = x2809;
x2820 = x380;
x2821 = x381;
x2818 = x230.forward(x2819, x2820, x2821);

// Dealloc(X98619)
JCudaTensor x2822;
x2822 = x2809;
x2822.free();

// val X98613 = ReLU()(X98612)
JCudaTensor x2823;
JCudaTensor x2824;
x2824 = x2789;
x2823 = x240.forward(x2824);

// val X98617 = ReLU()(X98616)
JCudaTensor x2825;
JCudaTensor x2826;
x2826 = x2813;
x2825 = x243.forward(x2826);

// val X98621 = ReLU()(X98620)
JCudaTensor x2827;
JCudaTensor x2828;
x2828 = x2818;
x2827 = x246.forward(x2828);

// val X98624 = ReLU()(X98623)
JCudaTensor x2829;
JCudaTensor x2830;
x2830 = x2804;
x2829 = x246.forward(x2830);

// val X98625 = Concat(X98613,X98617,X98621,X98624)
JCudaTensor x2831;
JCudaTensor x2832, x2833, x2834, x2835;
x2832 = x2823;
x2833 = x2825;
x2834 = x2827;
x2835 = x2829;
x2831 = x250.forward(x2832,x2833,x2834,x2835);

// Dealloc(X98624)
JCudaTensor x2836;
x2836 = x2829;
x2836.free();

// Dealloc(X98621)
JCudaTensor x2837;
x2837 = x2827;
x2837.free();

// Dealloc(X98617)
JCudaTensor x2838;
x2838 = x2825;
x2838.free();

// Dealloc(X98613)
JCudaTensor x2839;
x2839 = x2823;
x2839.free();

// val X98626 = Convolv(1,0)(X98625,cv61_W,cv61_B)
JCudaTensor x2840;
JCudaTensor x2841, x2842, x2843;
x2841 = x2831;
x2842 = x434;
x2843 = x435;
x2840 = x193.forward(x2841, x2842, x2843);

// val X98636 = Pooling(3,1,1,true)(X98625)
JCudaTensor x2844;
JCudaTensor x2845;
x2845 = x2831;
x2844 = x203.forward(x2845);

// val X98632 = Convolv(1,0)(X98625,cv64_W,cv64_B)
JCudaTensor x2846;
JCudaTensor x2847, x2848, x2849;
x2847 = x2831;
x2848 = x417;
x2849 = x418;
x2846 = x210.forward(x2847, x2848, x2849);

// val X98628 = Convolv(1,0)(X98625,cv62_W,cv62_B)
JCudaTensor x2850;
JCudaTensor x2851, x2852, x2853;
x2851 = x2831;
x2852 = x428;
x2853 = x429;
x2850 = x200.forward(x2851, x2852, x2853);

// Dealloc(X98625)
JCudaTensor x2854;
x2854 = x2831;
x2854.free();

// val X98633 = ReLU()(X98632)
JCudaTensor x2855;
JCudaTensor x2856;
x2856 = x2846;
x2855 = x213.forward(x2856);

// val X98629 = ReLU()(X98628)
JCudaTensor x2857;
JCudaTensor x2858;
x2858 = x2850;
x2857 = x216.forward(x2858);

// val X98637 = Convolv(1,0)(X98636,cv66_W,cv66_B)
JCudaTensor x2859;
JCudaTensor x2860, x2861, x2862;
x2860 = x2844;
x2861 = x442;
x2862 = x443;
x2859 = x223.forward(x2860, x2861, x2862);

// Dealloc(X98636)
JCudaTensor x2863;
x2863 = x2844;
x2863.free();

// val X98634 = Convolv(1,2)(X98633,cv65_W,cv65_B)
JCudaTensor x2864;
JCudaTensor x2865, x2866, x2867;
x2865 = x2855;
x2866 = x460;
x2867 = x461;
x2864 = x230.forward(x2865, x2866, x2867);

// Dealloc(X98633)
JCudaTensor x2868;
x2868 = x2855;
x2868.free();

// val X98630 = Convolv(1,1)(X98629,cv63_W,cv63_B)
JCudaTensor x2869;
JCudaTensor x2870, x2871, x2872;
x2870 = x2857;
x2871 = x473;
x2872 = x474;
x2869 = x237.forward(x2870, x2871, x2872);

// Dealloc(X98629)
JCudaTensor x2873;
x2873 = x2857;
x2873.free();

// val X98627 = ReLU()(X98626)
JCudaTensor x2874;
JCudaTensor x2875;
x2875 = x2840;
x2874 = x240.forward(x2875);

// val X98631 = ReLU()(X98630)
JCudaTensor x2876;
JCudaTensor x2877;
x2877 = x2869;
x2876 = x243.forward(x2877);

// val X98635 = ReLU()(X98634)
JCudaTensor x2878;
JCudaTensor x2879;
x2879 = x2864;
x2878 = x246.forward(x2879);

// val X98638 = ReLU()(X98637)
JCudaTensor x2880;
JCudaTensor x2881;
x2881 = x2859;
x2880 = x246.forward(x2881);

// val X98639 = Concat(X98627,X98631,X98635,X98638)
JCudaTensor x2882;
JCudaTensor x2883, x2884, x2885, x2886;
x2883 = x2874;
x2884 = x2876;
x2885 = x2878;
x2886 = x2880;
x2882 = x250.forward(x2883,x2884,x2885,x2886);

// Dealloc(X98638)
JCudaTensor x2887;
x2887 = x2880;
x2887.free();

// Dealloc(X98635)
JCudaTensor x2888;
x2888 = x2878;
x2888.free();

// Dealloc(X98631)
JCudaTensor x2889;
x2889 = x2876;
x2889.free();

// Dealloc(X98627)
JCudaTensor x2890;
x2890 = x2874;
x2890.free();

// val X98640 = Convolv(1,0)(X98639,cv71_W,cv71_B)
JCudaTensor x2891;
JCudaTensor x2892, x2893, x2894;
x2892 = x2882;
x2893 = x542;
x2894 = x543;
x2891 = x193.forward(x2892, x2893, x2894);

// val X98646 = Convolv(1,0)(X98639,cv74_W,cv74_B)
JCudaTensor x2895;
JCudaTensor x2896, x2897, x2898;
x2896 = x2882;
x2897 = x559;
x2898 = x560;
x2895 = x210.forward(x2896, x2897, x2898);

// val X98642 = Convolv(1,0)(X98639,cv72_W,cv72_B)
JCudaTensor x2899;
JCudaTensor x2900, x2901, x2902;
x2900 = x2882;
x2901 = x548;
x2902 = x549;
x2899 = x200.forward(x2900, x2901, x2902);

// val X98650 = Pooling(3,1,1,true)(X98639)
JCudaTensor x2903;
JCudaTensor x2904;
x2904 = x2882;
x2903 = x203.forward(x2904);

// Dealloc(X98639)
JCudaTensor x2905;
x2905 = x2882;
x2905.free();

// val X98651 = Convolv(1,0)(X98650,cv76_W,cv76_B)
JCudaTensor x2906;
JCudaTensor x2907, x2908, x2909;
x2907 = x2903;
x2908 = x583;
x2909 = x584;
x2906 = x223.forward(x2907, x2908, x2909);

// Dealloc(X98650)
JCudaTensor x2910;
x2910 = x2903;
x2910.free();

// val X98647 = ReLU()(X98646)
JCudaTensor x2911;
JCudaTensor x2912;
x2912 = x2895;
x2911 = x213.forward(x2912);

// val X98643 = ReLU()(X98642)
JCudaTensor x2913;
JCudaTensor x2914;
x2914 = x2899;
x2913 = x216.forward(x2914);

// val X98648 = Convolv(1,2)(X98647,cv75_W,cv75_B)
JCudaTensor x2915;
JCudaTensor x2916, x2917, x2918;
x2916 = x2911;
x2917 = x617;
x2918 = x618;
x2915 = x230.forward(x2916, x2917, x2918);

// Dealloc(X98647)
JCudaTensor x2919;
x2919 = x2911;
x2919.free();

// val X98644 = Convolv(1,1)(X98643,cv73_W,cv73_B)
JCudaTensor x2920;
JCudaTensor x2921, x2922, x2923;
x2921 = x2913;
x2922 = x626;
x2923 = x627;
x2920 = x237.forward(x2921, x2922, x2923);

// Dealloc(X98643)
JCudaTensor x2924;
x2924 = x2913;
x2924.free();

// val X98641 = ReLU()(X98640)
JCudaTensor x2925;
JCudaTensor x2926;
x2926 = x2891;
x2925 = x240.forward(x2926);

// val X98645 = ReLU()(X98644)
JCudaTensor x2927;
JCudaTensor x2928;
x2928 = x2920;
x2927 = x243.forward(x2928);

// val X98649 = ReLU()(X98648)
JCudaTensor x2929;
JCudaTensor x2930;
x2930 = x2915;
x2929 = x246.forward(x2930);

// val X98652 = ReLU()(X98651)
JCudaTensor x2931;
JCudaTensor x2932;
x2932 = x2906;
x2931 = x246.forward(x2932);

// val X98653 = Concat(X98641,X98645,X98649,X98652)
JCudaTensor x2933;
JCudaTensor x2934, x2935, x2936, x2937;
x2934 = x2925;
x2935 = x2927;
x2936 = x2929;
x2937 = x2931;
x2933 = x250.forward(x2934,x2935,x2936,x2937);

// Dealloc(X98652)
JCudaTensor x2938;
x2938 = x2931;
x2938.free();

// Dealloc(X98649)
JCudaTensor x2939;
x2939 = x2929;
x2939.free();

// Dealloc(X98645)
JCudaTensor x2940;
x2940 = x2927;
x2940.free();

// Dealloc(X98641)
JCudaTensor x2941;
x2941 = x2925;
x2941.free();

// val X98654 = Pooling(3,2,1,true)(X98653)
JCudaTensor x2942;
JCudaTensor x2943;
x2943 = x2933;
x2942 = x676.forward(x2943);

// Dealloc(X98653)
JCudaTensor x2944;
x2944 = x2933;
x2944.free();

// val X98657 = Convolv(1,0)(X98654,cv82_W,cv82_B)
JCudaTensor x2945;
JCudaTensor x2946, x2947, x2948;
x2946 = x2942;
x2947 = x695;
x2948 = x696;
x2945 = x697.forward(x2946, x2947, x2948);

// val X98655 = Convolv(1,0)(X98654,cv81_W,cv81_B)
JCudaTensor x2949;
JCudaTensor x2950, x2951, x2952;
x2950 = x2942;
x2951 = x683;
x2952 = x684;
x2949 = x685.forward(x2950, x2951, x2952);

// val X98661 = Convolv(1,0)(X98654,cv84_W,cv84_B)
JCudaTensor x2953;
JCudaTensor x2954, x2955, x2956;
x2954 = x2942;
x2955 = x702;
x2956 = x703;
x2953 = x704.forward(x2954, x2955, x2956);

// val X98665 = Pooling(3,1,1,true)(X98654)
JCudaTensor x2957;
JCudaTensor x2958;
x2958 = x2942;
x2957 = x688.forward(x2958);

// Dealloc(X98654)
JCudaTensor x2959;
x2959 = x2942;
x2959.free();

// val X98666 = Convolv(1,0)(X98665,cv86_W,cv86_B)
JCudaTensor x2960;
JCudaTensor x2961, x2962, x2963;
x2961 = x2957;
x2962 = x709;
x2963 = x710;
x2960 = x711.forward(x2961, x2962, x2963);

// Dealloc(X98665)
JCudaTensor x2964;
x2964 = x2957;
x2964.free();

// val X98662 = ReLU()(X98661)
JCudaTensor x2965;
JCudaTensor x2966;
x2966 = x2953;
x2965 = x717.forward(x2966);

// val X98658 = ReLU()(X98657)
JCudaTensor x2967;
JCudaTensor x2968;
x2968 = x2945;
x2967 = x714.forward(x2968);

// val X98659 = Convolv(1,1)(X98658,cv83_W,cv83_B)
JCudaTensor x2969;
JCudaTensor x2970, x2971, x2972;
x2970 = x2967;
x2971 = x732;
x2972 = x733;
x2969 = x734.forward(x2970, x2971, x2972);

// Dealloc(X98658)
JCudaTensor x2973;
x2973 = x2967;
x2973.free();

// val X98663 = Convolv(1,2)(X98662,cv85_W,cv85_B)
JCudaTensor x2974;
JCudaTensor x2975, x2976, x2977;
x2975 = x2965;
x2976 = x739;
x2977 = x740;
x2974 = x741.forward(x2975, x2976, x2977);

// Dealloc(X98662)
JCudaTensor x2978;
x2978 = x2965;
x2978.free();

// val X98656 = ReLU()(X98655)
JCudaTensor x2979;
JCudaTensor x2980;
x2980 = x2949;
x2979 = x755.forward(x2980);

// val X98660 = ReLU()(X98659)
JCudaTensor x2981;
JCudaTensor x2982;
x2982 = x2969;
x2981 = x750.forward(x2982);

// val X98664 = ReLU()(X98663)
JCudaTensor x2983;
JCudaTensor x2984;
x2984 = x2974;
x2983 = x747.forward(x2984);

// val X98667 = ReLU()(X98666)
JCudaTensor x2985;
JCudaTensor x2986;
x2986 = x2960;
x2985 = x747.forward(x2986);

// val X98668 = Concat(X98656,X98660,X98664,X98667)
JCudaTensor x2987;
JCudaTensor x2988, x2989, x2990, x2991;
x2988 = x2979;
x2989 = x2981;
x2990 = x2983;
x2991 = x2985;
x2987 = x760.forward(x2988,x2989,x2990,x2991);

// Dealloc(X98667)
JCudaTensor x2992;
x2992 = x2985;
x2992.free();

// Dealloc(X98664)
JCudaTensor x2993;
x2993 = x2983;
x2993.free();

// Dealloc(X98660)
JCudaTensor x2994;
x2994 = x2981;
x2994.free();

// Dealloc(X98656)
JCudaTensor x2995;
x2995 = x2979;
x2995.free();

// val X98675 = Convolv(1,0)(X98668,cv94_W,cv94_B)
JCudaTensor x2996;
JCudaTensor x2997, x2998, x2999;
x2997 = x2987;
x2998 = x787;
x2999 = x788;
x2996 = x704.forward(x2997, x2998, x2999);

// val X98679 = Pooling(3,1,1,true)(X98668)
JCudaTensor x3000;
JCudaTensor x3001;
x3001 = x2987;
x3000 = x688.forward(x3001);

// val X98671 = Convolv(1,0)(X98668,cv92_W,cv92_B)
JCudaTensor x3002;
JCudaTensor x3003, x3004, x3005;
x3003 = x2987;
x3004 = x775;
x3005 = x776;
x3002 = x697.forward(x3003, x3004, x3005);

// val X98669 = Convolv(1,0)(X98668,cv91_W,cv91_B)
JCudaTensor x3006;
JCudaTensor x3007, x3008, x3009;
x3007 = x2987;
x3008 = x781;
x3009 = x782;
x3006 = x685.forward(x3007, x3008, x3009);

// Dealloc(X98668)
JCudaTensor x3010;
x3010 = x2987;
x3010.free();

// val X98680 = Convolv(1,0)(X98679,cv96_W,cv96_B)
JCudaTensor x3011;
JCudaTensor x3012, x3013, x3014;
x3012 = x3000;
x3013 = x801;
x3014 = x802;
x3011 = x711.forward(x3012, x3013, x3014);

// Dealloc(X98679)
JCudaTensor x3015;
x3015 = x3000;
x3015.free();

// val X98672 = ReLU()(X98671)
JCudaTensor x3016;
JCudaTensor x3017;
x3017 = x3002;
x3016 = x714.forward(x3017);

// val X98676 = ReLU()(X98675)
JCudaTensor x3018;
JCudaTensor x3019;
x3019 = x2996;
x3018 = x717.forward(x3019);

// val X98673 = Convolv(1,1)(X98672,cv93_W,cv93_B)
JCudaTensor x3020;
JCudaTensor x3021, x3022, x3023;
x3021 = x3016;
x3022 = x819;
x3023 = x820;
x3020 = x734.forward(x3021, x3022, x3023);

// Dealloc(X98672)
JCudaTensor x3024;
x3024 = x3016;
x3024.free();

// val X98677 = Convolv(1,2)(X98676,cv95_W,cv95_B)
JCudaTensor x3025;
JCudaTensor x3026, x3027, x3028;
x3026 = x3018;
x3027 = x828;
x3028 = x829;
x3025 = x741.forward(x3026, x3027, x3028);

// Dealloc(X98676)
JCudaTensor x3029;
x3029 = x3018;
x3029.free();

// val X98670 = ReLU()(X98669)
JCudaTensor x3030;
JCudaTensor x3031;
x3031 = x3006;
x3030 = x755.forward(x3031);

// val X98674 = ReLU()(X98673)
JCudaTensor x3032;
JCudaTensor x3033;
x3033 = x3020;
x3032 = x750.forward(x3033);

// val X98678 = ReLU()(X98677)
JCudaTensor x3034;
JCudaTensor x3035;
x3035 = x3025;
x3034 = x747.forward(x3035);

// val X98681 = ReLU()(X98680)
JCudaTensor x3036;
JCudaTensor x3037;
x3037 = x3011;
x3036 = x747.forward(x3037);

// val X98682 = Concat(X98670,X98674,X98678,X98681)
JCudaTensor x3038;
JCudaTensor x3039, x3040, x3041, x3042;
x3039 = x3030;
x3040 = x3032;
x3041 = x3034;
x3042 = x3036;
x3038 = x760.forward(x3039,x3040,x3041,x3042);

// Dealloc(X98681)
JCudaTensor x3043;
x3043 = x3036;
x3043.free();

// Dealloc(X98678)
JCudaTensor x3044;
x3044 = x3034;
x3044.free();

// Dealloc(X98674)
JCudaTensor x3045;
x3045 = x3032;
x3045.free();

// Dealloc(X98670)
JCudaTensor x3046;
x3046 = x3030;
x3046.free();

// val X98683 = Pooling(7,1,0,false)(X98682)
JCudaTensor x3047;
JCudaTensor x3048;
x3048 = x3038;
x3047 = x902.forward(x3048);

// Dealloc(X98682)
JCudaTensor x3049;
x3049 = x3038;
x3049.free();

// val X98684 = Dropout(0.4)(X98683)
JCudaTensor x3050;
JCudaTensor x3051;
x3051 = x3047;
x3050 = x930.forward(x3051);

// Dealloc(X98683)
JCudaTensor x3052;
x3052 = x3047;
x3052.free();

// val X98685 = (X98684[1><3])(i | @) * (fc_W)(j | @)
JCudaTensor x3053;
JCudaMatrix x3054;
JCudaMatrix x3055;
JCudaTensor x3056;
JCudaTensor x3057;
x3057 = x3050;
x3056 = x3057.flatten(1, new int[]{256, 1, 1});
x3054 = x3056.asMatrix(1, true);
JCudaTensor x3058;
x3058 = x948;
x3055 = x3058.asMatrix(1, true);
x3053 = x3054.times(x3055);

// Dealloc(X98684)
JCudaTensor x3059;
x3059 = x3050;
x3059.free();

// val X98687 = (X98685 + (i) => fc_B)
JCudaTensor x3060;
JCudaTensor x3061, x3062;
x3061 = x3053;
x3062 = x963;
x3060 = x3062.copy(128, x3061);

// val X98688 = Cuda(Indicator(Y, 1000))
JCudaTensor x3063;
JTensorFloat x3064;
x3064 = x4.asIndicator(1000);
x3063 = x3064.asJCudaTensor();

// val X98689 = X98688 .* X98687
JCudaTensor x3065;
JCudaTensor x3066, x3067;
x3066 = x3063;
x3067 = x3060;
x3065 = x3066.times_i(x3067);

// val X98690 = Sum((X98689)(i17 | @))
JCudaTensor x3068;
JCudaMatrix x3069;
JCudaTensor x3070;
x3070 = x3065;
x3069 = x3070.asMatrix(1, true);
x3068 = x3069.sum();

// Dealloc(X98689)
JCudaTensor x3071;
x3071 = x3065;
x3071.free();

// val X98691 = Max((X98687)(i17 | @))
JCudaTensor x3072;
JCudaMatrix x3073;
JCudaTensor x3074;
x3074 = x3060;
x3073 = x3074.asMatrix(1, true);
x3072 = x3073.max();

// Dealloc(X98687)
JCudaTensor x3075;
x3075 = x3060;
x3075.free();

// val X98692 = 1{X98690 == X98691}
JCudaTensor x3076;
JCudaTensor x3077, x3078;
x3077 = x3068;
x3078 = x3072;
x3076 = x3077.eq(x3078);

// Dealloc(X98691)
JCudaTensor x3079;
x3079 = x3072;
x3079.free();

// BatchSum(((Sum(X98692) / |128|) / 10))
float x3081;
float x3082;
float x3083;
float x3084;
float x3085;
JCudaTensor x3086;
x3086 = x3076;
x3084 = x3086.sum();
x3085 = 128;
x3082 = x3084 / x3085;
x3083 = 10;
x3081 = x3082 / x3083;
x3080 += x3081;
// Print((Sum(X98692) / |128|))
float x3087;
float x3088;
float x3089;
JCudaTensor x3090;
x3090 = x3076;
x3088 = x3090.sum();
x3089 = 128;
x3087 = x3088 / x3089;
System.out.println(x5 + " test precision "  + x3087);

// Dealloc(X98692)
JCudaTensor x3091;
x3091 = x3076;
x3091.free();

}
System.out.println(x3080); 
}

}