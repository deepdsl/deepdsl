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
	static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{16,192,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(32, 192, 1, 1), List(32)))
	static JCudnnConvolution x86 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{32,192,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(64, 192, 1, 1), List(64)))
	static JCudnnConvolution x69 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{64,192,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(96, 192, 1, 1), List(96)))
	static JCudnnConvolution x79 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{96,192,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x193 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x220 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x200 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x210 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x133 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x154 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x147 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x140 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 4, 4), List(128, 256, 1, 1), List(128)))
	static JCudnnConvolution x292 = new JCudnnConvolution(new int[]{128,256,4,4},new int[]{128,256,1,1},new int[]{128}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x688 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x720 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x704 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
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
	static JCudnnConvolution x741 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,2),List(List(128, 16, 14, 14), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x230 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(1,2),List(List(128, 16, 28, 28), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x106 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(1,2),List(List(128, 16, 7, 7), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x730 = new JCudnnConvolution(new int[]{128,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(2,3),List(List(128, 3, 224, 224), List(64, 3, 7, 7), List(64)))
	static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
	// (Dropout(0.4),List(List(128, 256, 1, 1)))
	static JCudnnDropout x916 = new JCudnnDropout(new int[]{128,256,1,1}, 0.4f);
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
	static JCudnnSoftmax x402 = new JCudnnSoftmax(new int[]{128,1000}, SoftmaxAlgorithm.LOG);
	// (Pooling(3,1,1,true),List(List(128, 192, 28, 28)))
	static JCudnnPooling x72 = new JCudnnPooling(new int[]{128,192,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 14, 14)))
	static JCudnnPooling x203 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 28, 28)))
	static JCudnnPooling x126 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 7, 7)))
	static JCudnnPooling x681 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 192, 56, 56)))
	static JCudnnPooling x55 = new JCudnnPooling(new int[]{128,192,56,56}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 256, 14, 14)))
	static JCudnnPooling x676 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 256, 28, 28)))
	static JCudnnPooling x186 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 64, 112, 112)))
	static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,64,112,112}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(5,3,0,false),List(List(128, 256, 14, 14)))
	static JCudnnPooling x263 = new JCudnnPooling(new int[]{128,256,14,14}, 5, 3, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
	// (Pooling(7,1,0,false),List(List(128, 256, 7, 7)))
	static JCudnnPooling x902 = new JCudnnPooling(new int[]{128,256,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
	// (ReLU(),List(List(128, 1024)))
	static JCudnnActivation x348 = new JCudnnActivation(new int[]{128,1024}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 14, 14)))
	static JCudnnActivation x243 = new JCudnnActivation(new int[]{128,128,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 28, 28)))
	static JCudnnActivation x112 = new JCudnnActivation(new int[]{128,128,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 4, 4)))
	static JCudnnActivation x309 = new JCudnnActivation(new int[]{128,128,4,4}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 7, 7)))
	static JCudnnActivation x753 = new JCudnnActivation(new int[]{128,128,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 14, 14)))
	static JCudnnActivation x223 = new JCudnnActivation(new int[]{128,16,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 28, 28)))
	static JCudnnActivation x92 = new JCudnnActivation(new int[]{128,16,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 7, 7)))
	static JCudnnActivation x713 = new JCudnnActivation(new int[]{128,16,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 192, 56, 56)))
	static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,192,56,56}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 14, 14)))
	static JCudnnActivation x246 = new JCudnnActivation(new int[]{128,32,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 28, 28)))
	static JCudnnActivation x115 = new JCudnnActivation(new int[]{128,32,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 7, 7)))
	static JCudnnActivation x744 = new JCudnnActivation(new int[]{128,32,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 112, 112)))
	static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,64,112,112}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 14, 14)))
	static JCudnnActivation x240 = new JCudnnActivation(new int[]{128,64,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 28, 28)))
	static JCudnnActivation x109 = new JCudnnActivation(new int[]{128,64,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 56, 56)))
	static JCudnnActivation x39 = new JCudnnActivation(new int[]{128,64,56,56}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 7, 7)))
	static JCudnnActivation x747 = new JCudnnActivation(new int[]{128,64,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 14, 14)))
	static JCudnnActivation x213 = new JCudnnActivation(new int[]{128,96,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 28, 28)))
	static JCudnnActivation x89 = new JCudnnActivation(new int[]{128,96,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 7, 7)))
	static JCudnnActivation x723 = new JCudnnActivation(new int[]{128,96,7,7}, ActivationMode.RELU);
	// List(List(128, 64, 14, 14), List(128, 128, 14, 14), List(128, 32, 14, 14), List(128, 32, 14, 14))
	static JCudnnConcat x250 = new JCudnnConcat(new int[]{128,64,14,14},new int[]{128,128,14,14},new int[]{128,32,14,14},new int[]{128,32,14,14});
	// List(List(128, 64, 28, 28), List(128, 128, 28, 28), List(128, 32, 28, 28), List(128, 32, 28, 28))
	static JCudnnConcat x119 = new JCudnnConcat(new int[]{128,64,28,28},new int[]{128,128,28,28},new int[]{128,32,28,28},new int[]{128,32,28,28});
	// List(List(128, 64, 7, 7), List(128, 128, 7, 7), List(128, 32, 7, 7), List(128, 32, 7, 7))
	static JCudnnConcat x757 = new JCudnnConcat(new int[]{128,64,7,7},new int[]{128,128,7,7},new int[]{128,32,7,7},new int[]{128,32,7,7});
	// Precision(Accuracy(1))
	static float x3063;
	// V_b1cv_B
	static JCudaTensor x643 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_b1cv_W
	static JCudaTensor x636 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
	// V_b1fc1_B
	static JCudaTensor x592 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_b1fc1_W
	static JCudaTensor x600 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
	// V_b1fc2_B
	static JCudaTensor x515 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_b1fc2_W
	static JCudaTensor x507 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
	// V_b2cv_B
	static JCudaTensor x942 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_b2cv_W
	static JCudaTensor x937 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
	// V_b2fc1_B
	static JCudaTensor x920 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_b2fc1_W
	static JCudaTensor x906 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
	// V_b2fc2_B
	static JCudaTensor x848 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_b2fc2_W
	static JCudaTensor x859 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
	// V_cv11_B
	static JCudaTensor x2332 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv11_W
	static JCudaTensor x2336 = JTensor.constFloat(0.0f, 64, 192, 1, 1).asJCudaTensor();
	// V_cv12_B
	static JCudaTensor x2416 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv12_W
	static JCudaTensor x2424 = JTensor.constFloat(0.0f, 96, 192, 1, 1).asJCudaTensor();
	// V_cv13_B
	static JCudaTensor x2341 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv13_W
	static JCudaTensor x2363 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv14_B
	static JCudaTensor x2430 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv14_W
	static JCudaTensor x2411 = JTensor.constFloat(0.0f, 16, 192, 1, 1).asJCudaTensor();
	// V_cv15_B
	static JCudaTensor x2369 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv15_W
	static JCudaTensor x2327 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv16_B
	static JCudaTensor x2374 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv16_W
	static JCudaTensor x2355 = JTensor.constFloat(0.0f, 32, 192, 1, 1).asJCudaTensor();
	// V_cv1_B
	static JCudaTensor x2536 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x2530 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
	// V_cv21_B
	static JCudaTensor x2209 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv21_W
	static JCudaTensor x2184 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv22_B
	static JCudaTensor x2265 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv22_W
	static JCudaTensor x2273 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv23_B
	static JCudaTensor x2180 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv23_W
	static JCudaTensor x2193 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv24_B
	static JCudaTensor x2261 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv24_W
	static JCudaTensor x2256 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv25_B
	static JCudaTensor x2189 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv25_W
	static JCudaTensor x2175 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv26_B
	static JCudaTensor x2198 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv26_W
	static JCudaTensor x2214 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x2499 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x2503 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
	// V_cv31_B
	static JCudaTensor x2011 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv31_W
	static JCudaTensor x2032 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv32_B
	static JCudaTensor x2113 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv32_W
	static JCudaTensor x2104 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv33_B
	static JCudaTensor x2058 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv33_W
	static JCudaTensor x2020 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv34_B
	static JCudaTensor x2109 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv34_W
	static JCudaTensor x2095 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv35_B
	static JCudaTensor x2046 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv35_W
	static JCudaTensor x2037 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv36_B
	static JCudaTensor x2025 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv36_W
	static JCudaTensor x2015 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv3_B
	static JCudaTensor x2476 = JTensor.constFloat(0.0f, 192).asJCudaTensor();
	// V_cv3_W
	static JCudaTensor x2480 = JTensor.constFloat(0.0f, 192, 64, 3, 3).asJCudaTensor();
	// V_cv41_B
	static JCudaTensor x1858 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv41_W
	static JCudaTensor x1895 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv42_B
	static JCudaTensor x1947 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv42_W
	static JCudaTensor x1942 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv43_B
	static JCudaTensor x1869 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv43_W
	static JCudaTensor x1853 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv44_B
	static JCudaTensor x1952 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv44_W
	static JCudaTensor x1933 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv45_B
	static JCudaTensor x1849 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv45_W
	static JCudaTensor x1886 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv46_B
	static JCudaTensor x1862 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv46_W
	static JCudaTensor x1874 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv51_B
	static JCudaTensor x1708 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv51_W
	static JCudaTensor x1740 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv52_B
	static JCudaTensor x1783 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv52_W
	static JCudaTensor x1795 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv53_B
	static JCudaTensor x1697 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv53_W
	static JCudaTensor x1729 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv54_B
	static JCudaTensor x1787 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv54_W
	static JCudaTensor x1778 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv55_B
	static JCudaTensor x1735 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv55_W
	static JCudaTensor x1712 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv56_B
	static JCudaTensor x1701 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv56_W
	static JCudaTensor x1723 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv61_B
	static JCudaTensor x1566 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv61_W
	static JCudaTensor x1550 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv62_B
	static JCudaTensor x1637 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv62_W
	static JCudaTensor x1632 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv63_B
	static JCudaTensor x1539 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv63_W
	static JCudaTensor x1555 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv64_B
	static JCudaTensor x1642 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv64_W
	static JCudaTensor x1623 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv65_B
	static JCudaTensor x1570 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv65_W
	static JCudaTensor x1561 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv66_B
	static JCudaTensor x1546 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv66_W
	static JCudaTensor x1581 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv71_B
	static JCudaTensor x1405 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv71_W
	static JCudaTensor x1377 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv72_B
	static JCudaTensor x1475 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv72_W
	static JCudaTensor x1470 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv73_B
	static JCudaTensor x1398 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv73_W
	static JCudaTensor x1414 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv74_B
	static JCudaTensor x1480 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv74_W
	static JCudaTensor x1461 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv75_B
	static JCudaTensor x1388 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv75_W
	static JCudaTensor x1392 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv76_B
	static JCudaTensor x1424 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv76_W
	static JCudaTensor x1409 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv81_B
	static JCudaTensor x1239 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv81_W
	static JCudaTensor x1219 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv82_B
	static JCudaTensor x1309 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv82_W
	static JCudaTensor x1313 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv83_B
	static JCudaTensor x1244 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv83_W
	static JCudaTensor x1228 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv84_B
	static JCudaTensor x1319 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv84_W
	static JCudaTensor x1300 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv85_B
	static JCudaTensor x1263 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv85_W
	static JCudaTensor x1258 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv86_B
	static JCudaTensor x1224 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv86_W
	static JCudaTensor x1248 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv91_B
	static JCudaTensor x1075 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv91_W
	static JCudaTensor x1066 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv92_B
	static JCudaTensor x1163 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv92_W
	static JCudaTensor x1158 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv93_B
	static JCudaTensor x1087 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv93_W
	static JCudaTensor x1061 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv94_B
	static JCudaTensor x1150 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv94_W
	static JCudaTensor x1145 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv95_B
	static JCudaTensor x1071 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv95_W
	static JCudaTensor x1100 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv96_B
	static JCudaTensor x1092 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv96_W
	static JCudaTensor x1079 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_fc_B
	static JCudaTensor x1020 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc_W
	static JCudaTensor x1010 = JTensor.constFloat(0.0f, 1000, 256).asJCudaTensor();
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;
	// b1cv_B
	static JCudaTensor x291 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b1cv_B").asJCudaTensor();
	// b1cv_W
	static JCudaTensor x290 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b1cv_W").asJCudaTensor();
	// b1fc1_B
	static JCudaTensor x333 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b1fc1_B").asJCudaTensor();
	// b1fc1_W
	static JCudaTensor x322 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b1fc1_W").asJCudaTensor();
	// b1fc2_B
	static JCudaTensor x395 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b1fc2_B").asJCudaTensor();
	// b1fc2_W
	static JCudaTensor x381 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b1fc2_W").asJCudaTensor();
	// b2cv_B
	static JCudaTensor x588 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b2cv_B").asJCudaTensor();
	// b2cv_W
	static JCudaTensor x587 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b2cv_W").asJCudaTensor();
	// b2fc1_B
	static JCudaTensor x668 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b2fc1_B").asJCudaTensor();
	// b2fc1_W
	static JCudaTensor x658 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b2fc1_W").asJCudaTensor();
	// b2fc2_B
	static JCudaTensor x734 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b2fc2_B").asJCudaTensor();
	// b2fc2_W
	static JCudaTensor x710 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b2fc2_W").asJCudaTensor();
	// cv11_B
	static JCudaTensor x68 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
	// cv11_W
	static JCudaTensor x67 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 64, 192, 1, 1).load(network_dir + "/cv11_W").asJCudaTensor();
	// cv12_B
	static JCudaTensor x78 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv12_B").asJCudaTensor();
	// cv12_W
	static JCudaTensor x77 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 96, 192, 1, 1).load(network_dir + "/cv12_W").asJCudaTensor();
	// cv13_B
	static JCudaTensor x98 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv13_B").asJCudaTensor();
	// cv13_W
	static JCudaTensor x97 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv13_W").asJCudaTensor();
	// cv14_B
	static JCudaTensor x61 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv14_B").asJCudaTensor();
	// cv14_W
	static JCudaTensor x60 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 16, 192, 1, 1).load(network_dir + "/cv14_W").asJCudaTensor();
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
	static JCudaTensor x146 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv21_B").asJCudaTensor();
	// cv21_W
	static JCudaTensor x145 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv21_W").asJCudaTensor();
	// cv22_B
	static JCudaTensor x139 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv22_B").asJCudaTensor();
	// cv22_W
	static JCudaTensor x138 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv22_W").asJCudaTensor();
	// cv23_B
	static JCudaTensor x170 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv23_B").asJCudaTensor();
	// cv23_W
	static JCudaTensor x169 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv23_W").asJCudaTensor();
	// cv24_B
	static JCudaTensor x132 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv24_B").asJCudaTensor();
	// cv24_W
	static JCudaTensor x131 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv24_W").asJCudaTensor();
	// cv25_B
	static JCudaTensor x164 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv25_B").asJCudaTensor();
	// cv25_W
	static JCudaTensor x163 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv25_W").asJCudaTensor();
	// cv26_B
	static JCudaTensor x153 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv26_B").asJCudaTensor();
	// cv26_W
	static JCudaTensor x152 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv26_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x35 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x34 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/cv2_W").asJCudaTensor();
	// cv31_B
	static JCudaTensor x199 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv31_B").asJCudaTensor();
	// cv31_W
	static JCudaTensor x198 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv31_W").asJCudaTensor();
	// cv32_B
	static JCudaTensor x209 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv32_B").asJCudaTensor();
	// cv32_W
	static JCudaTensor x208 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv32_W").asJCudaTensor();
	// cv33_B
	static JCudaTensor x236 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv33_B").asJCudaTensor();
	// cv33_W
	static JCudaTensor x235 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
	// cv34_B
	static JCudaTensor x192 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv34_B").asJCudaTensor();
	// cv34_W
	static JCudaTensor x191 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv34_W").asJCudaTensor();
	// cv35_B
	static JCudaTensor x229 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv35_B").asJCudaTensor();
	// cv35_W
	static JCudaTensor x228 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv35_W").asJCudaTensor();
	// cv36_B
	static JCudaTensor x219 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv36_B").asJCudaTensor();
	// cv36_W
	static JCudaTensor x218 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv36_W").asJCudaTensor();
	// cv3_B
	static JCudaTensor x45 = JTensor.constFloat(0.2f, 192).load(network_dir + "/cv3_B").asJCudaTensor();
	// cv3_W
	static JCudaTensor x44 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 192, 64, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
	// cv41_B
	static JCudaTensor x271 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv41_B").asJCudaTensor();
	// cv41_W
	static JCudaTensor x270 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv41_W").asJCudaTensor();
	// cv42_B
	static JCudaTensor x260 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv42_B").asJCudaTensor();
	// cv42_W
	static JCudaTensor x259 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv42_W").asJCudaTensor();
	// cv43_B
	static JCudaTensor x306 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv43_B").asJCudaTensor();
	// cv43_W
	static JCudaTensor x305 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
	// cv44_B
	static JCudaTensor x277 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv44_B").asJCudaTensor();
	// cv44_W
	static JCudaTensor x276 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv44_W").asJCudaTensor();
	// cv45_B
	static JCudaTensor x300 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv45_B").asJCudaTensor();
	// cv45_W
	static JCudaTensor x299 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv45_W").asJCudaTensor();
	// cv46_B
	static JCudaTensor x285 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv46_B").asJCudaTensor();
	// cv46_W
	static JCudaTensor x284 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv46_W").asJCudaTensor();
	// cv51_B
	static JCudaTensor x356 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv51_B").asJCudaTensor();
	// cv51_W
	static JCudaTensor x355 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv51_W").asJCudaTensor();
	// cv52_B
	static JCudaTensor x345 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv52_B").asJCudaTensor();
	// cv52_W
	static JCudaTensor x344 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv52_W").asJCudaTensor();
	// cv53_B
	static JCudaTensor x387 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv53_B").asJCudaTensor();
	// cv53_W
	static JCudaTensor x386 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
	// cv54_B
	static JCudaTensor x339 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv54_B").asJCudaTensor();
	// cv54_W
	static JCudaTensor x338 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv54_W").asJCudaTensor();
	// cv55_B
	static JCudaTensor x375 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv55_B").asJCudaTensor();
	// cv55_W
	static JCudaTensor x374 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv55_W").asJCudaTensor();
	// cv56_B
	static JCudaTensor x369 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv56_B").asJCudaTensor();
	// cv56_W
	static JCudaTensor x368 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv56_W").asJCudaTensor();
	// cv61_B
	static JCudaTensor x429 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv61_B").asJCudaTensor();
	// cv61_W
	static JCudaTensor x428 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv61_W").asJCudaTensor();
	// cv62_B
	static JCudaTensor x423 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv62_B").asJCudaTensor();
	// cv62_W
	static JCudaTensor x422 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv62_W").asJCudaTensor();
	// cv63_B
	static JCudaTensor x477 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv63_B").asJCudaTensor();
	// cv63_W
	static JCudaTensor x476 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv63_W").asJCudaTensor();
	// cv64_B
	static JCudaTensor x437 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv64_B").asJCudaTensor();
	// cv64_W
	static JCudaTensor x436 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv64_W").asJCudaTensor();
	// cv65_B
	static JCudaTensor x468 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv65_B").asJCudaTensor();
	// cv65_W
	static JCudaTensor x467 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv65_W").asJCudaTensor();
	// cv66_B
	static JCudaTensor x445 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv66_B").asJCudaTensor();
	// cv66_W
	static JCudaTensor x444 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv66_W").asJCudaTensor();
	// cv71_B
	static JCudaTensor x555 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv71_B").asJCudaTensor();
	// cv71_W
	static JCudaTensor x554 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv71_W").asJCudaTensor();
	// cv72_B
	static JCudaTensor x549 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv72_B").asJCudaTensor();
	// cv72_W
	static JCudaTensor x548 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv72_W").asJCudaTensor();
	// cv73_B
	static JCudaTensor x613 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv73_B").asJCudaTensor();
	// cv73_W
	static JCudaTensor x612 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv73_W").asJCudaTensor();
	// cv74_B
	static JCudaTensor x533 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv74_B").asJCudaTensor();
	// cv74_W
	static JCudaTensor x532 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv74_W").asJCudaTensor();
	// cv75_B
	static JCudaTensor x619 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv75_B").asJCudaTensor();
	// cv75_W
	static JCudaTensor x618 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv75_W").asJCudaTensor();
	// cv76_B
	static JCudaTensor x577 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv76_B").asJCudaTensor();
	// cv76_W
	static JCudaTensor x576 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv76_W").asJCudaTensor();
	// cv81_B
	static JCudaTensor x703 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv81_B").asJCudaTensor();
	// cv81_W
	static JCudaTensor x702 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv81_W").asJCudaTensor();
	// cv82_B
	static JCudaTensor x696 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv82_B").asJCudaTensor();
	// cv82_W
	static JCudaTensor x695 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv82_W").asJCudaTensor();
	// cv83_B
	static JCudaTensor x740 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv83_B").asJCudaTensor();
	// cv83_W
	static JCudaTensor x739 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv83_W").asJCudaTensor();
	// cv84_B
	static JCudaTensor x687 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv84_B").asJCudaTensor();
	// cv84_W
	static JCudaTensor x686 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv84_W").asJCudaTensor();
	// cv85_B
	static JCudaTensor x729 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv85_B").asJCudaTensor();
	// cv85_W
	static JCudaTensor x728 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv85_W").asJCudaTensor();
	// cv86_B
	static JCudaTensor x719 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv86_B").asJCudaTensor();
	// cv86_W
	static JCudaTensor x718 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv86_W").asJCudaTensor();
	// cv91_B
	static JCudaTensor x772 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv91_B").asJCudaTensor();
	// cv91_W
	static JCudaTensor x771 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv91_W").asJCudaTensor();
	// cv92_B
	static JCudaTensor x778 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv92_B").asJCudaTensor();
	// cv92_W
	static JCudaTensor x777 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv92_W").asJCudaTensor();
	// cv93_B
	static JCudaTensor x829 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv93_B").asJCudaTensor();
	// cv93_W
	static JCudaTensor x828 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv93_W").asJCudaTensor();
	// cv94_B
	static JCudaTensor x790 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv94_B").asJCudaTensor();
	// cv94_W
	static JCudaTensor x789 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv94_W").asJCudaTensor();
	// cv95_B
	static JCudaTensor x823 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv95_B").asJCudaTensor();
	// cv95_W
	static JCudaTensor x822 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv95_W").asJCudaTensor();
	// cv96_B
	static JCudaTensor x800 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv96_B").asJCudaTensor();
	// cv96_W
	static JCudaTensor x799 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv96_W").asJCudaTensor();
	// fc_B
	static JCudaTensor x963 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
	// fc_W
	static JCudaTensor x953 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1000, 256).load(network_dir + "/fc_W").asJCudaTensor();

	public static void main(String[] args){
		double t = System.nanoTime();
		train();
		System.out.println((System.nanoTime() - t) / 1.0E9);
		test();
		x291.save(network_dir + "/b1cv_B");
		x290.save(network_dir + "/b1cv_W");
		x333.save(network_dir + "/b1fc1_B");
		x322.save(network_dir + "/b1fc1_W");
		x395.save(network_dir + "/b1fc2_B");
		x381.save(network_dir + "/b1fc2_W");
		x588.save(network_dir + "/b2cv_B");
		x587.save(network_dir + "/b2cv_W");
		x668.save(network_dir + "/b2fc1_B");
		x658.save(network_dir + "/b2fc1_W");
		x734.save(network_dir + "/b2fc2_B");
		x710.save(network_dir + "/b2fc2_W");
		x68.save(network_dir + "/cv11_B");
		x67.save(network_dir + "/cv11_W");
		x78.save(network_dir + "/cv12_B");
		x77.save(network_dir + "/cv12_W");
		x98.save(network_dir + "/cv13_B");
		x97.save(network_dir + "/cv13_W");
		x61.save(network_dir + "/cv14_B");
		x60.save(network_dir + "/cv14_W");
		x105.save(network_dir + "/cv15_B");
		x104.save(network_dir + "/cv15_W");
		x85.save(network_dir + "/cv16_B");
		x84.save(network_dir + "/cv16_W");
		x16.save(network_dir + "/cv1_B");
		x15.save(network_dir + "/cv1_W");
		x146.save(network_dir + "/cv21_B");
		x145.save(network_dir + "/cv21_W");
		x139.save(network_dir + "/cv22_B");
		x138.save(network_dir + "/cv22_W");
		x170.save(network_dir + "/cv23_B");
		x169.save(network_dir + "/cv23_W");
		x132.save(network_dir + "/cv24_B");
		x131.save(network_dir + "/cv24_W");
		x164.save(network_dir + "/cv25_B");
		x163.save(network_dir + "/cv25_W");
		x153.save(network_dir + "/cv26_B");
		x152.save(network_dir + "/cv26_W");
		x35.save(network_dir + "/cv2_B");
		x34.save(network_dir + "/cv2_W");
		x199.save(network_dir + "/cv31_B");
		x198.save(network_dir + "/cv31_W");
		x209.save(network_dir + "/cv32_B");
		x208.save(network_dir + "/cv32_W");
		x236.save(network_dir + "/cv33_B");
		x235.save(network_dir + "/cv33_W");
		x192.save(network_dir + "/cv34_B");
		x191.save(network_dir + "/cv34_W");
		x229.save(network_dir + "/cv35_B");
		x228.save(network_dir + "/cv35_W");
		x219.save(network_dir + "/cv36_B");
		x218.save(network_dir + "/cv36_W");
		x45.save(network_dir + "/cv3_B");
		x44.save(network_dir + "/cv3_W");
		x271.save(network_dir + "/cv41_B");
		x270.save(network_dir + "/cv41_W");
		x260.save(network_dir + "/cv42_B");
		x259.save(network_dir + "/cv42_W");
		x306.save(network_dir + "/cv43_B");
		x305.save(network_dir + "/cv43_W");
		x277.save(network_dir + "/cv44_B");
		x276.save(network_dir + "/cv44_W");
		x300.save(network_dir + "/cv45_B");
		x299.save(network_dir + "/cv45_W");
		x285.save(network_dir + "/cv46_B");
		x284.save(network_dir + "/cv46_W");
		x356.save(network_dir + "/cv51_B");
		x355.save(network_dir + "/cv51_W");
		x345.save(network_dir + "/cv52_B");
		x344.save(network_dir + "/cv52_W");
		x387.save(network_dir + "/cv53_B");
		x386.save(network_dir + "/cv53_W");
		x339.save(network_dir + "/cv54_B");
		x338.save(network_dir + "/cv54_W");
		x375.save(network_dir + "/cv55_B");
		x374.save(network_dir + "/cv55_W");
		x369.save(network_dir + "/cv56_B");
		x368.save(network_dir + "/cv56_W");
		x429.save(network_dir + "/cv61_B");
		x428.save(network_dir + "/cv61_W");
		x423.save(network_dir + "/cv62_B");
		x422.save(network_dir + "/cv62_W");
		x477.save(network_dir + "/cv63_B");
		x476.save(network_dir + "/cv63_W");
		x437.save(network_dir + "/cv64_B");
		x436.save(network_dir + "/cv64_W");
		x468.save(network_dir + "/cv65_B");
		x467.save(network_dir + "/cv65_W");
		x445.save(network_dir + "/cv66_B");
		x444.save(network_dir + "/cv66_W");
		x555.save(network_dir + "/cv71_B");
		x554.save(network_dir + "/cv71_W");
		x549.save(network_dir + "/cv72_B");
		x548.save(network_dir + "/cv72_W");
		x613.save(network_dir + "/cv73_B");
		x612.save(network_dir + "/cv73_W");
		x533.save(network_dir + "/cv74_B");
		x532.save(network_dir + "/cv74_W");
		x619.save(network_dir + "/cv75_B");
		x618.save(network_dir + "/cv75_W");
		x577.save(network_dir + "/cv76_B");
		x576.save(network_dir + "/cv76_W");
		x703.save(network_dir + "/cv81_B");
		x702.save(network_dir + "/cv81_W");
		x696.save(network_dir + "/cv82_B");
		x695.save(network_dir + "/cv82_W");
		x740.save(network_dir + "/cv83_B");
		x739.save(network_dir + "/cv83_W");
		x687.save(network_dir + "/cv84_B");
		x686.save(network_dir + "/cv84_W");
		x729.save(network_dir + "/cv85_B");
		x728.save(network_dir + "/cv85_W");
		x719.save(network_dir + "/cv86_B");
		x718.save(network_dir + "/cv86_W");
		x772.save(network_dir + "/cv91_B");
		x771.save(network_dir + "/cv91_W");
		x778.save(network_dir + "/cv92_B");
		x777.save(network_dir + "/cv92_W");
		x829.save(network_dir + "/cv93_B");
		x828.save(network_dir + "/cv93_W");
		x790.save(network_dir + "/cv94_B");
		x789.save(network_dir + "/cv94_W");
		x823.save(network_dir + "/cv95_B");
		x822.save(network_dir + "/cv95_W");
		x800.save(network_dir + "/cv96_B");
		x799.save(network_dir + "/cv96_W");
		x963.save(network_dir + "/fc_B");
		x953.save(network_dir + "/fc_W");
		x381.free();
		x1942.free();
		x61.free();
		x1787.free();
		x937.free();
		x2369.free();
		x643.free();
		x1642.free();
		x771.free();
		x131.free();
		x15.free();
		x2095.free();
		x2209.free();
		x515.free();
		x2180.free();
		x1555.free();
		x2261.free();
		x428.free();
		x1623.free();
		x229.free();
		x305.free();
		x1723.free();
		x828.free();
		x356.free();
		x906.free();
		x1581.free();
		x859.free();
		x191.free();
		x507.free();
		x1392.free();
		x703.free();
		x260.free();
		x2341.free();
		x1061.free();
		x1388.free();
		x219.free();
		x235.free();
		x963.free();
		x1377.free();
		x1539.free();
		x1895.free();
		x2046.free();
		x1258.free();
		x164.free();
		x97.free();
		x138.free();
		x1795.free();
		x1461.free();
		x169.free();
		x369.free();
		x2020.free();
		x34.free();
		x718.free();
		x1409.free();
		x2025.free();
		x2499.free();
		x790.free();
		x84.free();
		x740.free();
		x587.free();
		x422.free();
		x299.free();
		x285.free();
		x549.free();
		x702.free();
		x1224.free();
		x734.free();
		x739.free();
		x98.free();
		x1778.free();
		x386.free();
		x1735.free();
		x1239.free();
		x477.free();
		x710.free();
		x2430.free();
		x2189.free();
		x368.free();
		x2214.free();
		x554.free();
		x822.free();
		x1424.free();
		x152.free();
		x67.free();
		x2336.free();
		x35.free();
		x668.free();
		x344.free();
		x2198.free();
		x445.free();
		x375.free();
		x2104.free();
		x355.free();
		x271.free();
		x145.free();
		x1309.free();
		x284.free();
		x618.free();
		x1952.free();
		x686.free();
		x1697.free();
		x333.free();
		x1566.free();
		x444.free();
		x532.free();
		x163.free();
		x1010.free();
		x1480.free();
		x619.free();
		x2037.free();
		x339.free();
		x777.free();
		x555.free();
		x2363.free();
		x77.free();
		x2355.free();
		x1100.free();
		x170.free();
		x2332.free();
		x658.free();
		x636.free();
		x1163.free();
		x719.free();
		x1075.free();
		x437.free();
		x1862.free();
		x199.free();
		x687.free();
		x1708.free();
		x1020.free();
		x338.free();
		x576.free();
		x85.free();
		x612.free();
		x1570.free();
		x1145.free();
		x2184.free();
		x789.free();
		x192.free();
		x290.free();
		x2327.free();
		x772.free();
		x45.free();
		x60.free();
		x2175.free();
		x476.free();
		x2109.free();
		x300.free();
		x1150.free();
		x78.free();
		x259.free();
		x729.free();
		x153.free();
		x467.free();
		x1475.free();
		x395.free();
		x1637.free();
		x104.free();
		x2411.free();
		x1874.free();
		x2265.free();
		x920.free();
		x1405.free();
		x1414.free();
		x1933.free();
		x139.free();
		x1550.free();
		x1313.free();
		x468.free();
		x276.free();
		x1849.free();
		x291.free();
		x600.free();
		x577.free();
		x2011.free();
		x306.free();
		x1712.free();
		x848.free();
		x799.free();
		x1470.free();
		x1561.free();
		x436.free();
		x1244.free();
		x270.free();
		x1632.free();
		x146.free();
		x800.free();
		x1853.free();
		x1783.free();
		x1869.free();
		x2058.free();
		x44.free();
		x1071.free();
		x1947.free();
		x1158.free();
		x2113.free();
		x2503.free();
		x1219.free();
		x16.free();
		x2256.free();
		x695.free();
		x1079.free();
		x1398.free();
		x105.free();
		x345.free();
		x533.free();
		x322.free();
		x209.free();
		x1087.free();
		x1546.free();
		x823.free();
		x2015.free();
		x374.free();
		x613.free();
		x829.free();
		x1300.free();
		x132.free();
		x429.free();
		x68.free();
		x208.free();
		x198.free();
		x1701.free();
		x2476.free();
		x1228.free();
		x728.free();
		x1066.free();
		x2424.free();
		x2530.free();
		x236.free();
		x592.free();
		x1263.free();
		x2536.free();
		x2374.free();
		x2480.free();
		x1740.free();
		x1886.free();
		x1858.free();
		x588.free();
		x2273.free();
		x953.free();
		x1092.free();
		x2193.free();
		x277.free();
		x228.free();
		x942.free();
		x218.free();
		x423.free();
		x548.free();
		x778.free();
		x1729.free();
		x2416.free();
		x696.free();
		x1319.free();
		x1248.free();
		x2032.free();
		x387.free();
		x292.free();
		x753.free();
		x140.free();
		x744.free();
		x741.free();
		x154.free();
		x39.free();
		x747.free();
		x29.free();
		x237.free();
		x720.free();
		x106.free();
		x147.free();
		x723.free();
		x112.free();
		x213.free();
		x230.free();
		x126.free();
		x49.free();
		x704.free();
		x36.free();
		x186.free();
		x681.free();
		x263.free();
		x203.free();
		x99.free();
		x26.free();
		x23.free();
		x89.free();
		x200.free();
		x223.free();
		x86.free();
		x55.free();
		x92.free();
		x730.free();
		x402.free();
		x52.free();
		x79.free();
		x309.free();
		x133.free();
		x46.free();
		x17.free();
		x916.free();
		x62.free();
		x69.free();
		x115.free();
		x72.free();
		x688.free();
		x109.free();
		x193.free();
		x220.free();
		x348.free();
		x676.free();
		x246.free();
		x902.free();
		x243.free();
		x713.free();
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

			// val X7542 = Cuda(X)
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x3;
			x7 = x8.asJCudaTensor();

			// val X7687 = Cuda(Indicator(Y, 1000))
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x4.asIndicator(1000);
			x9 = x10.asJCudaTensor();

			// val X7543 = Convolv(2,3)(X7542,cv1_W,cv1_B)
			JCudaTensor x11;
			JCudaTensor x12, x13, x14;
			x12 = x7;
			x13 = x15;
			x14 = x16;
			x11 = x17.forward(x12, x13, x14);

			// val X2084 = - X7687.copy
			JCudaTensor x18;
			JCudaTensor x19;
			float x20;
			x19 = x9;
			x19 = x19.clone();
			x20 = -1;
			x18 = x19.times_i(x20);

			// val X7544 = ReLU()(X7543)
			JCudaTensor x21;
			JCudaTensor x22;
			x22 = x11;
			x21 = x23.forward(x22);

			// val X7545 = Pooling(3,2,1,true)(X7544)
			JCudaTensor x24;
			JCudaTensor x25;
			x25 = x21;
			x24 = x26.forward(x25);

			// val X7546 = LRN(5,1.0E-4,0.75)(X7545)
			JCudaTensor x27;
			JCudaTensor x28;
			x28 = x24;
			x27 = x29.forward(x28);

			// val X7547 = Convolv(1,0)(X7546,cv2_W,cv2_B)
			JCudaTensor x30;
			JCudaTensor x31, x32, x33;
			x31 = x27;
			x32 = x34;
			x33 = x35;
			x30 = x36.forward(x31, x32, x33);

			// val X7548 = ReLU()(X7547)
			JCudaTensor x37;
			JCudaTensor x38;
			x38 = x30;
			x37 = x39.forward(x38);

			// val X7549 = Convolv(1,1)(X7548,cv3_W,cv3_B)
			JCudaTensor x40;
			JCudaTensor x41, x42, x43;
			x41 = x37;
			x42 = x44;
			x43 = x45;
			x40 = x46.forward(x41, x42, x43);

			// val X7550 = ReLU()(X7549)
			JCudaTensor x47;
			JCudaTensor x48;
			x48 = x40;
			x47 = x49.forward(x48);

			// val X7551 = LRN(5,1.0E-4,0.75)(X7550)
			JCudaTensor x50;
			JCudaTensor x51;
			x51 = x47;
			x50 = x52.forward(x51);

			// val X7552 = Pooling(3,2,1,true)(X7551)
			JCudaTensor x53;
			JCudaTensor x54;
			x54 = x50;
			x53 = x55.forward(x54);

			// val X7559 = Convolv(1,0)(X7552,cv14_W,cv14_B)
			JCudaTensor x56;
			JCudaTensor x57, x58, x59;
			x57 = x53;
			x58 = x60;
			x59 = x61;
			x56 = x62.forward(x57, x58, x59);

			// val X7553 = Convolv(1,0)(X7552,cv11_W,cv11_B)
			JCudaTensor x63;
			JCudaTensor x64, x65, x66;
			x64 = x53;
			x65 = x67;
			x66 = x68;
			x63 = x69.forward(x64, x65, x66);

			// val X7563 = Pooling(3,1,1,true)(X7552)
			JCudaTensor x70;
			JCudaTensor x71;
			x71 = x53;
			x70 = x72.forward(x71);

			// val X7555 = Convolv(1,0)(X7552,cv12_W,cv12_B)
			JCudaTensor x73;
			JCudaTensor x74, x75, x76;
			x74 = x53;
			x75 = x77;
			x76 = x78;
			x73 = x79.forward(x74, x75, x76);

			// val X7564 = Convolv(1,0)(X7563,cv16_W,cv16_B)
			JCudaTensor x80;
			JCudaTensor x81, x82, x83;
			x81 = x70;
			x82 = x84;
			x83 = x85;
			x80 = x86.forward(x81, x82, x83);

			// val X7556 = ReLU()(X7555)
			JCudaTensor x87;
			JCudaTensor x88;
			x88 = x73;
			x87 = x89.forward(x88);

			// val X7560 = ReLU()(X7559)
			JCudaTensor x90;
			JCudaTensor x91;
			x91 = x56;
			x90 = x92.forward(x91);

			// val X7557 = Convolv(1,1)(X7556,cv13_W,cv13_B)
			JCudaTensor x93;
			JCudaTensor x94, x95, x96;
			x94 = x87;
			x95 = x97;
			x96 = x98;
			x93 = x99.forward(x94, x95, x96);

			// val X7561 = Convolv(1,2)(X7560,cv15_W,cv15_B)
			JCudaTensor x100;
			JCudaTensor x101, x102, x103;
			x101 = x90;
			x102 = x104;
			x103 = x105;
			x100 = x106.forward(x101, x102, x103);

			// val X7554 = ReLU()(X7553)
			JCudaTensor x107;
			JCudaTensor x108;
			x108 = x63;
			x107 = x109.forward(x108);

			// val X7558 = ReLU()(X7557)
			JCudaTensor x110;
			JCudaTensor x111;
			x111 = x93;
			x110 = x112.forward(x111);

			// val X7562 = ReLU()(X7561)
			JCudaTensor x113;
			JCudaTensor x114;
			x114 = x100;
			x113 = x115.forward(x114);

			// val X7565 = ReLU()(X7564)
			JCudaTensor x116;
			JCudaTensor x117;
			x117 = x80;
			x116 = x115.forward(x117);

			// val X7566 = Concat(X7554,X7558,X7562,X7565)
			JCudaTensor x118;
			JCudaTensor x120, x121, x122, x123;
			x120 = x107;
			x121 = x110;
			x122 = x113;
			x123 = x116;
			x118 = x119.forward(x120,x121,x122,x123);

			// val X7577 = Pooling(3,1,1,true)(X7566)
			JCudaTensor x124;
			JCudaTensor x125;
			x125 = x118;
			x124 = x126.forward(x125);

			// val X7573 = Convolv(1,0)(X7566,cv24_W,cv24_B)
			JCudaTensor x127;
			JCudaTensor x128, x129, x130;
			x128 = x118;
			x129 = x131;
			x130 = x132;
			x127 = x133.forward(x128, x129, x130);

			// val X7569 = Convolv(1,0)(X7566,cv22_W,cv22_B)
			JCudaTensor x134;
			JCudaTensor x135, x136, x137;
			x135 = x118;
			x136 = x138;
			x137 = x139;
			x134 = x140.forward(x135, x136, x137);

			// val X7567 = Convolv(1,0)(X7566,cv21_W,cv21_B)
			JCudaTensor x141;
			JCudaTensor x142, x143, x144;
			x142 = x118;
			x143 = x145;
			x144 = x146;
			x141 = x147.forward(x142, x143, x144);

			// val X7578 = Convolv(1,0)(X7577,cv26_W,cv26_B)
			JCudaTensor x148;
			JCudaTensor x149, x150, x151;
			x149 = x124;
			x150 = x152;
			x151 = x153;
			x148 = x154.forward(x149, x150, x151);

			// val X7570 = ReLU()(X7569)
			JCudaTensor x155;
			JCudaTensor x156;
			x156 = x134;
			x155 = x89.forward(x156);

			// val X7574 = ReLU()(X7573)
			JCudaTensor x157;
			JCudaTensor x158;
			x158 = x127;
			x157 = x92.forward(x158);

			// val X7575 = Convolv(1,2)(X7574,cv25_W,cv25_B)
			JCudaTensor x159;
			JCudaTensor x160, x161, x162;
			x160 = x157;
			x161 = x163;
			x162 = x164;
			x159 = x106.forward(x160, x161, x162);

			// val X7571 = Convolv(1,1)(X7570,cv23_W,cv23_B)
			JCudaTensor x165;
			JCudaTensor x166, x167, x168;
			x166 = x155;
			x167 = x169;
			x168 = x170;
			x165 = x99.forward(x166, x167, x168);

			// val X7568 = ReLU()(X7567)
			JCudaTensor x171;
			JCudaTensor x172;
			x172 = x141;
			x171 = x109.forward(x172);

			// val X7572 = ReLU()(X7571)
			JCudaTensor x173;
			JCudaTensor x174;
			x174 = x165;
			x173 = x112.forward(x174);

			// val X7576 = ReLU()(X7575)
			JCudaTensor x175;
			JCudaTensor x176;
			x176 = x159;
			x175 = x115.forward(x176);

			// val X7579 = ReLU()(X7578)
			JCudaTensor x177;
			JCudaTensor x178;
			x178 = x148;
			x177 = x115.forward(x178);

			// val X7580 = Concat(X7568,X7572,X7576,X7579)
			JCudaTensor x179;
			JCudaTensor x180, x181, x182, x183;
			x180 = x171;
			x181 = x173;
			x182 = x175;
			x183 = x177;
			x179 = x119.forward(x180,x181,x182,x183);

			// val X7581 = Pooling(3,2,1,true)(X7580)
			JCudaTensor x184;
			JCudaTensor x185;
			x185 = x179;
			x184 = x186.forward(x185);

			// val X7588 = Convolv(1,0)(X7581,cv34_W,cv34_B)
			JCudaTensor x187;
			JCudaTensor x188, x189, x190;
			x188 = x184;
			x189 = x191;
			x190 = x192;
			x187 = x193.forward(x188, x189, x190);

			// val X7582 = Convolv(1,0)(X7581,cv31_W,cv31_B)
			JCudaTensor x194;
			JCudaTensor x195, x196, x197;
			x195 = x184;
			x196 = x198;
			x197 = x199;
			x194 = x200.forward(x195, x196, x197);

			// val X7592 = Pooling(3,1,1,true)(X7581)
			JCudaTensor x201;
			JCudaTensor x202;
			x202 = x184;
			x201 = x203.forward(x202);

			// val X7584 = Convolv(1,0)(X7581,cv32_W,cv32_B)
			JCudaTensor x204;
			JCudaTensor x205, x206, x207;
			x205 = x184;
			x206 = x208;
			x207 = x209;
			x204 = x210.forward(x205, x206, x207);

			// val X7585 = ReLU()(X7584)
			JCudaTensor x211;
			JCudaTensor x212;
			x212 = x204;
			x211 = x213.forward(x212);

			// val X7593 = Convolv(1,0)(X7592,cv36_W,cv36_B)
			JCudaTensor x214;
			JCudaTensor x215, x216, x217;
			x215 = x201;
			x216 = x218;
			x217 = x219;
			x214 = x220.forward(x215, x216, x217);

			// val X7589 = ReLU()(X7588)
			JCudaTensor x221;
			JCudaTensor x222;
			x222 = x187;
			x221 = x223.forward(x222);

			// val X7590 = Convolv(1,2)(X7589,cv35_W,cv35_B)
			JCudaTensor x224;
			JCudaTensor x225, x226, x227;
			x225 = x221;
			x226 = x228;
			x227 = x229;
			x224 = x230.forward(x225, x226, x227);

			// val X7586 = Convolv(1,1)(X7585,cv33_W,cv33_B)
			JCudaTensor x231;
			JCudaTensor x232, x233, x234;
			x232 = x211;
			x233 = x235;
			x234 = x236;
			x231 = x237.forward(x232, x233, x234);

			// val X7583 = ReLU()(X7582)
			JCudaTensor x238;
			JCudaTensor x239;
			x239 = x194;
			x238 = x240.forward(x239);

			// val X7587 = ReLU()(X7586)
			JCudaTensor x241;
			JCudaTensor x242;
			x242 = x231;
			x241 = x243.forward(x242);

			// val X7591 = ReLU()(X7590)
			JCudaTensor x244;
			JCudaTensor x245;
			x245 = x224;
			x244 = x246.forward(x245);

			// val X7594 = ReLU()(X7593)
			JCudaTensor x247;
			JCudaTensor x248;
			x248 = x214;
			x247 = x246.forward(x248);

			// val X7595 = Concat(X7583,X7587,X7591,X7594)
			JCudaTensor x249;
			JCudaTensor x251, x252, x253, x254;
			x251 = x238;
			x252 = x241;
			x253 = x244;
			x254 = x247;
			x249 = x250.forward(x251,x252,x253,x254);

			// val X7598 = Convolv(1,0)(X7595,cv42_W,cv42_B)
			JCudaTensor x255;
			JCudaTensor x256, x257, x258;
			x256 = x249;
			x257 = x259;
			x258 = x260;
			x255 = x210.forward(x256, x257, x258);

			// val X7701 = Pooling(5,3,0,false)(X7595)
			JCudaTensor x261;
			JCudaTensor x262;
			x262 = x249;
			x261 = x263.forward(x262);

			// val X7606 = Pooling(3,1,1,true)(X7595)
			JCudaTensor x264;
			JCudaTensor x265;
			x265 = x249;
			x264 = x203.forward(x265);

			// val X7596 = Convolv(1,0)(X7595,cv41_W,cv41_B)
			JCudaTensor x266;
			JCudaTensor x267, x268, x269;
			x267 = x249;
			x268 = x270;
			x269 = x271;
			x266 = x200.forward(x267, x268, x269);

			// val X7602 = Convolv(1,0)(X7595,cv44_W,cv44_B)
			JCudaTensor x272;
			JCudaTensor x273, x274, x275;
			x273 = x249;
			x274 = x276;
			x275 = x277;
			x272 = x193.forward(x273, x274, x275);

			// val X7603 = ReLU()(X7602)
			JCudaTensor x278;
			JCudaTensor x279;
			x279 = x272;
			x278 = x223.forward(x279);

			// val X7607 = Convolv(1,0)(X7606,cv46_W,cv46_B)
			JCudaTensor x280;
			JCudaTensor x281, x282, x283;
			x281 = x264;
			x282 = x284;
			x283 = x285;
			x280 = x220.forward(x281, x282, x283);

			// val X7702 = Convolv(1,0)(X7701,b1cv_W,b1cv_B)
			JCudaTensor x286;
			JCudaTensor x287, x288, x289;
			x287 = x261;
			x288 = x290;
			x289 = x291;
			x286 = x292.forward(x287, x288, x289);

			// val X7599 = ReLU()(X7598)
			JCudaTensor x293;
			JCudaTensor x294;
			x294 = x255;
			x293 = x213.forward(x294);

			// val X7604 = Convolv(1,2)(X7603,cv45_W,cv45_B)
			JCudaTensor x295;
			JCudaTensor x296, x297, x298;
			x296 = x278;
			x297 = x299;
			x298 = x300;
			x295 = x230.forward(x296, x297, x298);

			// val X7600 = Convolv(1,1)(X7599,cv43_W,cv43_B)
			JCudaTensor x301;
			JCudaTensor x302, x303, x304;
			x302 = x293;
			x303 = x305;
			x304 = x306;
			x301 = x237.forward(x302, x303, x304);

			// val X7703 = ReLU()(X7702)
			JCudaTensor x307;
			JCudaTensor x308;
			x308 = x286;
			x307 = x309.forward(x308);

			// val X7605 = ReLU()(X7604)
			JCudaTensor x310;
			JCudaTensor x311;
			x311 = x295;
			x310 = x246.forward(x311);

			// val X7601 = ReLU()(X7600)
			JCudaTensor x312;
			JCudaTensor x313;
			x313 = x301;
			x312 = x243.forward(x313);

			// val X7608 = ReLU()(X7607)
			JCudaTensor x314;
			JCudaTensor x315;
			x315 = x280;
			x314 = x246.forward(x315);

			// val X7704 = (X7703[1><3])(i | @) * (b1fc1_W)(j | @)
			JCudaTensor x316;
			JCudaMatrix x317;
			JCudaMatrix x318;
			JCudaTensor x319;
			JCudaTensor x320;
			x320 = x307;
			x319 = x320.flatten(1, new int[]{128, 4, 4});
			x317 = x319.asMatrix(1, true);
			JCudaTensor x321;
			x321 = x322;
			x318 = x321.asMatrix(1, true);
			x316 = x317.times(x318);

			// val X7597 = ReLU()(X7596)
			JCudaTensor x323;
			JCudaTensor x324;
			x324 = x266;
			x323 = x240.forward(x324);

			// val X7609 = Concat(X7597,X7601,X7605,X7608)
			JCudaTensor x325;
			JCudaTensor x326, x327, x328, x329;
			x326 = x323;
			x327 = x312;
			x328 = x310;
			x329 = x314;
			x325 = x250.forward(x326,x327,x328,x329);

			// val X7706 = (X7704 + (i) => b1fc1_B)
			JCudaTensor x330;
			JCudaTensor x331, x332;
			x331 = x316;
			x332 = x333;
			x330 = x332.copy(128, x331);

			// val X7616 = Convolv(1,0)(X7609,cv54_W,cv54_B)
			JCudaTensor x334;
			JCudaTensor x335, x336, x337;
			x335 = x325;
			x336 = x338;
			x337 = x339;
			x334 = x193.forward(x335, x336, x337);

			// val X7612 = Convolv(1,0)(X7609,cv52_W,cv52_B)
			JCudaTensor x340;
			JCudaTensor x341, x342, x343;
			x341 = x325;
			x342 = x344;
			x343 = x345;
			x340 = x210.forward(x341, x342, x343);

			// val X7707 = ReLU()(X7706)
			JCudaTensor x346;
			JCudaTensor x347;
			x347 = x330;
			x346 = x348.forward(x347);

			// val X7620 = Pooling(3,1,1,true)(X7609)
			JCudaTensor x349;
			JCudaTensor x350;
			x350 = x325;
			x349 = x203.forward(x350);

			// val X7610 = Convolv(1,0)(X7609,cv51_W,cv51_B)
			JCudaTensor x351;
			JCudaTensor x352, x353, x354;
			x352 = x325;
			x353 = x355;
			x354 = x356;
			x351 = x200.forward(x352, x353, x354);

			// val X7617 = ReLU()(X7616)
			JCudaTensor x357;
			JCudaTensor x358;
			x358 = x334;
			x357 = x223.forward(x358);

			// val X7708 = Dropout(0.7)(X7707)
			JCudaTensor x359;
			JCudaTensor x360;
			x360 = x346;
			x359 = x361.forward(x360);

			// val X7613 = ReLU()(X7612)
			JCudaTensor x362;
			JCudaTensor x363;
			x363 = x340;
			x362 = x213.forward(x363);

			// val X7621 = Convolv(1,0)(X7620,cv56_W,cv56_B)
			JCudaTensor x364;
			JCudaTensor x365, x366, x367;
			x365 = x349;
			x366 = x368;
			x367 = x369;
			x364 = x220.forward(x365, x366, x367);

			// val X7618 = Convolv(1,2)(X7617,cv55_W,cv55_B)
			JCudaTensor x370;
			JCudaTensor x371, x372, x373;
			x371 = x357;
			x372 = x374;
			x373 = x375;
			x370 = x230.forward(x371, x372, x373);

			// val X7709 = (X7708)(i | @) * (b1fc2_W)(j | @)
			JCudaTensor x376;
			JCudaMatrix x377;
			JCudaMatrix x378;
			JCudaTensor x379;
			x379 = x359;
			x377 = x379.asMatrix(1, true);
			JCudaTensor x380;
			x380 = x381;
			x378 = x380.asMatrix(1, true);
			x376 = x377.times(x378);

			// val X7614 = Convolv(1,1)(X7613,cv53_W,cv53_B)
			JCudaTensor x382;
			JCudaTensor x383, x384, x385;
			x383 = x362;
			x384 = x386;
			x385 = x387;
			x382 = x237.forward(x383, x384, x385);

			// val X7611 = ReLU()(X7610)
			JCudaTensor x388;
			JCudaTensor x389;
			x389 = x351;
			x388 = x240.forward(x389);

			// val X7619 = ReLU()(X7618)
			JCudaTensor x390;
			JCudaTensor x391;
			x391 = x370;
			x390 = x246.forward(x391);

			// val X7711 = (X7709 + (i) => b1fc2_B)
			JCudaTensor x392;
			JCudaTensor x393, x394;
			x393 = x376;
			x394 = x395;
			x392 = x394.copy(128, x393);

			// val X7622 = ReLU()(X7621)
			JCudaTensor x396;
			JCudaTensor x397;
			x397 = x364;
			x396 = x246.forward(x397);

			// val X7615 = ReLU()(X7614)
			JCudaTensor x398;
			JCudaTensor x399;
			x399 = x382;
			x398 = x243.forward(x399);

			// val X7712 = LogSoftmax()(X7711)
			JCudaTensor x400;
			JCudaTensor x401;
			x401 = x392;
			x400 = x402.forward(x401);

			// Dealloc(X7711)
			JCudaTensor x403;
			x403 = x392;
			x403.free();

			// val X2085 = (X2084 / |128|)
			JCudaTensor x404;
			JCudaTensor x405;
			float x406;
			x405 = x18;
			float x407;
			x407 = 128;
			x406 = 1 / x407;
			x404 = x405.times_i(x406);

			// val X7623 = Concat(X7611,X7615,X7619,X7622)
			JCudaTensor x408;
			JCudaTensor x409, x410, x411, x412;
			x409 = x388;
			x410 = x398;
			x411 = x390;
			x412 = x396;
			x408 = x250.forward(x409,x410,x411,x412);

			// val X2748 = X2085 * d_LogSoftmax()(X7712)/d_X7711
			JCudaTensor x413;
			JCudaTensor x414, x415;
			x414 = x404;
			x415 = x400;
			x413 = x402.backward(x414, x415);

			// val X7634 = Pooling(3,1,1,true)(X7623)
			JCudaTensor x416;
			JCudaTensor x417;
			x417 = x408;
			x416 = x203.forward(x417);

			// val X7626 = Convolv(1,0)(X7623,cv62_W,cv62_B)
			JCudaTensor x418;
			JCudaTensor x419, x420, x421;
			x419 = x408;
			x420 = x422;
			x421 = x423;
			x418 = x210.forward(x419, x420, x421);

			// val X7624 = Convolv(1,0)(X7623,cv61_W,cv61_B)
			JCudaTensor x424;
			JCudaTensor x425, x426, x427;
			x425 = x408;
			x426 = x428;
			x427 = x429;
			x424 = x200.forward(x425, x426, x427);

			// val m4 = (i3347) => b1fc2_W[@, i3347]
			JCudaMatrix x430;
			JCudaTensor x431;
			x431 = x381;
			x430 = x431.asMatrix(1, false);

			// val X7630 = Convolv(1,0)(X7623,cv64_W,cv64_B)
			JCudaTensor x432;
			JCudaTensor x433, x434, x435;
			x433 = x408;
			x434 = x436;
			x435 = x437;
			x432 = x193.forward(x433, x434, x435);

			// val X7627 = ReLU()(X7626)
			JCudaTensor x438;
			JCudaTensor x439;
			x439 = x418;
			x438 = x213.forward(x439);

			// val X7635 = Convolv(1,0)(X7634,cv66_W,cv66_B)
			JCudaTensor x440;
			JCudaTensor x441, x442, x443;
			x441 = x416;
			x442 = x444;
			x443 = x445;
			x440 = x220.forward(x441, x442, x443);

			// val X2757 = (X2748)(i3346 | @) * m4
			JCudaTensor x446;
			JCudaMatrix x447;
			JCudaMatrix x448;
			JCudaTensor x449;
			x449 = x413;
			x447 = x449.asMatrix(1, true);
			x448 = x430;
			x446 = x447.times(x448);

			// val m23 = (i1272) => X7708[@, i1272]
			JCudaMatrix x450;
			JCudaTensor x451;
			x451 = x359;
			x450 = x451.asMatrix(1, false);

			// val X7631 = ReLU()(X7630)
			JCudaTensor x452;
			JCudaTensor x453;
			x453 = x432;
			x452 = x223.forward(x453);

			// val m21 = (i1268) => X2748[@, i1268]
			JCudaMatrix x454;
			JCudaTensor x455;
			x455 = x413;
			x454 = x455.asMatrix(1, false);

			// val X5304 = Sum(m21)
			JCudaTensor x456;
			JCudaMatrix x457;
			x457 = x454;
			x456 = x457.sum();

			// val X5306 = m21 * m23
			JCudaTensor x458;
			JCudaMatrix x459;
			JCudaMatrix x460;
			x459 = x454;
			x460 = x450;
			x458 = x459.times(x460);

			// Dealloc(X2748)
			JCudaTensor x461;
			x461 = x413;
			x461.free();

			// Dealloc(X7708)
			JCudaTensor x462;
			x462 = x359;
			x462.free();

			// val X7632 = Convolv(1,2)(X7631,cv65_W,cv65_B)
			JCudaTensor x463;
			JCudaTensor x464, x465, x466;
			x464 = x452;
			x465 = x467;
			x466 = x468;
			x463 = x230.forward(x464, x465, x466);

			// val X2758 = X2757 * d_Dropout(0.7)()/d_X7707
			JCudaTensor x469;
			JCudaTensor x470;
			x470 = x446;
			x469 = x361.backward(x470);

			// Dealloc(X2757)
			JCudaTensor x471;
			x471 = x446;
			x471.free();

			// val X7628 = Convolv(1,1)(X7627,cv63_W,cv63_B)
			JCudaTensor x472;
			JCudaTensor x473, x474, x475;
			x473 = x438;
			x474 = x476;
			x475 = x477;
			x472 = x237.forward(x473, x474, x475);

			// val X5307 = (X5306 * loss1)
			JCudaTensor x478;
			JCudaTensor x479;
			float x480;
			x479 = x458;
			x480 = loss1;
			x478 = x479.times_i(x480);

			// val X2760 = X2758 * d_ReLU()(X7707)/d_X7706
			JCudaTensor x481;
			JCudaTensor x482, x483;
			x482 = x469;
			x483 = x346;
			x481 = x348.backward(x482, x483);

			// Dealloc(X7707)
			JCudaTensor x484;
			x484 = x346;
			x484.free();

			// val X7625 = ReLU()(X7624)
			JCudaTensor x485;
			JCudaTensor x486;
			x486 = x424;
			x485 = x240.forward(x486);

			// val m5 = (i3351) => b1fc1_W[@, i3351]
			JCudaMatrix x487;
			JCudaTensor x488;
			x488 = x322;
			x487 = x488.asMatrix(1, false);

			// val X7629 = ReLU()(X7628)
			JCudaTensor x489;
			JCudaTensor x490;
			x490 = x472;
			x489 = x243.forward(x490);

			// val X5305 = (X5304 * loss1)
			JCudaTensor x491;
			JCudaTensor x492;
			float x493;
			x492 = x456;
			x493 = loss1;
			x491 = x492.times_i(x493);

			// val X7636 = ReLU()(X7635)
			JCudaTensor x494;
			JCudaTensor x495;
			x495 = x440;
			x494 = x246.forward(x495);

			// val X7633 = ReLU()(X7632)
			JCudaTensor x496;
			JCudaTensor x497;
			x497 = x463;
			x496 = x246.forward(x497);

			// val X2761 = (X2760)(i3350 | @) * m5
			JCudaTensor x498;
			JCudaMatrix x499;
			JCudaMatrix x500;
			JCudaTensor x501;
			x501 = x481;
			x499 = x501.asMatrix(1, true);
			x500 = x487;
			x498 = x499.times(x500);

			// val X7637 = Concat(X7625,X7629,X7633,X7636)
			JCudaTensor x502;
			JCudaTensor x503, x504, x505, x506;
			x503 = x485;
			x504 = x489;
			x505 = x496;
			x506 = x494;
			x502 = x250.forward(x503,x504,x505,x506);

			// V_b1fc2_W <~~ X5307
			float x508, x509;
			x508 = lrn_rate_1;
			x509 = momentum;
			JCudaTensor x510;
			x510 = x478;
			x507.update(x510, x508, x509);

			// Dealloc(X5307)
			JCudaTensor x511;
			x511 = x478;
			x511.free();

			// val m20 = (i1285) => X7703[1><3][@, i1285]
			JCudaMatrix x512;
			JCudaTensor x513;
			JCudaTensor x514;
			x514 = x307;
			x513 = x514.flatten(1, new int[]{128, 4, 4});
			x512 = x513.asMatrix(1, false);

			// V_b1fc2_B <~~ X5305
			float x516, x517;
			x516 = lrn_rate_2;
			x517 = momentum;
			JCudaTensor x518;
			x518 = x491;
			x515.update(x518, x516, x517);

			// Dealloc(X5305)
			JCudaTensor x519;
			x519 = x491;
			x519.free();

			// val m18 = (i1281) => X2760[@, i1281]
			JCudaMatrix x520;
			JCudaTensor x521;
			x521 = x481;
			x520 = x521.asMatrix(1, false);

			// b1fc2_W <~~ V_b1fc2_W
			float x522, x523;
			x522 = 1;
			x523 = decay_1;
			JCudaTensor x524;
			x524 = x507;
			x381.update(x524, x522, x523);

			// b1fc2_B <~~ V_b1fc2_B
			float x525, x526;
			x525 = 1;
			x526 = 1;
			JCudaTensor x527;
			x527 = x515;
			x395.update(x527, x525, x526);

			// val X7644 = Convolv(1,0)(X7637,cv74_W,cv74_B)
			JCudaTensor x528;
			JCudaTensor x529, x530, x531;
			x529 = x502;
			x530 = x532;
			x531 = x533;
			x528 = x193.forward(x529, x530, x531);

			// val X5300 = Sum(m18)
			JCudaTensor x534;
			JCudaMatrix x535;
			x535 = x520;
			x534 = x535.sum();

			// val X7688 = Pooling(5,3,0,false)(X7637)
			JCudaTensor x536;
			JCudaTensor x537;
			x537 = x502;
			x536 = x263.forward(x537);

			// val X7648 = Pooling(3,1,1,true)(X7637)
			JCudaTensor x538;
			JCudaTensor x539;
			x539 = x502;
			x538 = x203.forward(x539);

			// val X5302 = m18 * m20
			JCudaTensor x540;
			JCudaMatrix x541;
			JCudaMatrix x542;
			x541 = x520;
			x542 = x512;
			x540 = x541.times(x542);

			// Dealloc(X2760)
			JCudaTensor x543;
			x543 = x481;
			x543.free();

			// val X7640 = Convolv(1,0)(X7637,cv72_W,cv72_B)
			JCudaTensor x544;
			JCudaTensor x545, x546, x547;
			x545 = x502;
			x546 = x548;
			x547 = x549;
			x544 = x210.forward(x545, x546, x547);

			// val X7638 = Convolv(1,0)(X7637,cv71_W,cv71_B)
			JCudaTensor x550;
			JCudaTensor x551, x552, x553;
			x551 = x502;
			x552 = x554;
			x553 = x555;
			x550 = x200.forward(x551, x552, x553);

			// val X2763 = X2761[1<>3] * d_ReLU()(X7703)/d_X7702
			JCudaTensor x556;
			JCudaTensor x557, x558;
			JCudaTensor x559;
			x559 = x498;
			x557 = x559.unflatten(1, new int[]{128, 4, 4});
			x558 = x307;
			x556 = x309.backward(x557, x558);

			// Dealloc(X7703)
			JCudaTensor x560;
			x560 = x307;
			x560.free();

			// val X5301 = (X5300 * loss1)
			JCudaTensor x561;
			JCudaTensor x562;
			float x563;
			x562 = x534;
			x563 = loss1;
			x561 = x562.times_i(x563);

			// val X7645 = ReLU()(X7644)
			JCudaTensor x564;
			JCudaTensor x565;
			x565 = x528;
			x564 = x223.forward(x565);

			// val X2764 = X2763 * d_Convolv(1,0)(b1cv_W)/d_X7701
			JCudaTensor x566;
			JCudaTensor x567, x568;
			x567 = x556;
			x568 = x290;
			x566 = x292.backward_data(x567, x568);

			// val X5298 = X2763 * d_Convolv(1,0)(X7701)/d_b1cv_W
			JCudaTensor x569;
			JCudaTensor x570, x571;
			x570 = x556;
			x571 = x261;
			x569 = x292.backward_filter(x570, x571);

			// val X7649 = Convolv(1,0)(X7648,cv76_W,cv76_B)
			JCudaTensor x572;
			JCudaTensor x573, x574, x575;
			x573 = x538;
			x574 = x576;
			x575 = x577;
			x572 = x220.forward(x573, x574, x575);

			// val X7641 = ReLU()(X7640)
			JCudaTensor x578;
			JCudaTensor x579;
			x579 = x544;
			x578 = x213.forward(x579);

			// val X5296 = X2763 * d_Convolv(1,0)()/d_b1cv_B
			JCudaTensor x580;
			JCudaTensor x581;
			x581 = x556;
			x580 = x292.backward_bias(x581);

			// Dealloc(X2763)
			JCudaTensor x582;
			x582 = x556;
			x582.free();

			// val X7689 = Convolv(1,0)(X7688,b2cv_W,b2cv_B)
			JCudaTensor x583;
			JCudaTensor x584, x585, x586;
			x584 = x536;
			x585 = x587;
			x586 = x588;
			x583 = x292.forward(x584, x585, x586);

			// val X5303 = (X5302 * loss1)
			JCudaTensor x589;
			JCudaTensor x590;
			float x591;
			x590 = x540;
			x591 = loss1;
			x589 = x590.times_i(x591);

			// V_b1fc1_B <~~ X5301
			float x593, x594;
			x593 = lrn_rate_2;
			x594 = momentum;
			JCudaTensor x595;
			x595 = x561;
			x592.update(x595, x593, x594);

			// Dealloc(X5301)
			JCudaTensor x596;
			x596 = x561;
			x596.free();

			// val X5297 = (X5296 * loss1)
			JCudaTensor x597;
			JCudaTensor x598;
			float x599;
			x598 = x580;
			x599 = loss1;
			x597 = x598.times_i(x599);

			// V_b1fc1_W <~~ X5303
			float x601, x602;
			x601 = lrn_rate_1;
			x602 = momentum;
			JCudaTensor x603;
			x603 = x589;
			x600.update(x603, x601, x602);

			// Dealloc(X5303)
			JCudaTensor x604;
			x604 = x589;
			x604.free();

			// val X5299 = (X5298 * loss1)
			JCudaTensor x605;
			JCudaTensor x606;
			float x607;
			x606 = x569;
			x607 = loss1;
			x605 = x606.times_i(x607);

			// val X7642 = Convolv(1,1)(X7641,cv73_W,cv73_B)
			JCudaTensor x608;
			JCudaTensor x609, x610, x611;
			x609 = x578;
			x610 = x612;
			x611 = x613;
			x608 = x237.forward(x609, x610, x611);

			// val X7646 = Convolv(1,2)(X7645,cv75_W,cv75_B)
			JCudaTensor x614;
			JCudaTensor x615, x616, x617;
			x615 = x564;
			x616 = x618;
			x617 = x619;
			x614 = x230.forward(x615, x616, x617);

			// val X2766 = X2764 * d_Pooling(5,3,0,false)(X7701,X7595)/d_X7595
			JCudaTensor x620;
			JCudaTensor x621, x622, x623;
			x621 = x566;
			x622 = x261;
			x623 = x249;
			x620 = x263.backward(x621, x622, x623);

			// Dealloc(X2764)
			JCudaTensor x624;
			x624 = x566;
			x624.free();

			// Dealloc(X7701)
			JCudaTensor x625;
			x625 = x261;
			x625.free();

			// val X7690 = ReLU()(X7689)
			JCudaTensor x626;
			JCudaTensor x627;
			x627 = x583;
			x626 = x309.forward(x627);

			// b1fc1_B <~~ V_b1fc1_B
			float x628, x629;
			x628 = 1;
			x629 = 1;
			JCudaTensor x630;
			x630 = x592;
			x333.update(x630, x628, x629);

			// b1fc1_W <~~ V_b1fc1_W
			float x631, x632;
			x631 = 1;
			x632 = decay_1;
			JCudaTensor x633;
			x633 = x600;
			x322.update(x633, x631, x632);

			// val X7639 = ReLU()(X7638)
			JCudaTensor x634;
			JCudaTensor x635;
			x635 = x550;
			x634 = x240.forward(x635);

			// V_b1cv_W <~~ X5299
			float x637, x638;
			x637 = lrn_rate_1;
			x638 = momentum;
			JCudaTensor x639;
			x639 = x605;
			x636.update(x639, x637, x638);

			// Dealloc(X5299)
			JCudaTensor x640;
			x640 = x605;
			x640.free();

			// val X7643 = ReLU()(X7642)
			JCudaTensor x641;
			JCudaTensor x642;
			x642 = x608;
			x641 = x243.forward(x642);

			// V_b1cv_B <~~ X5297
			float x644, x645;
			x644 = lrn_rate_2;
			x645 = momentum;
			JCudaTensor x646;
			x646 = x597;
			x643.update(x646, x644, x645);

			// Dealloc(X5297)
			JCudaTensor x647;
			x647 = x597;
			x647.free();

			// val X7650 = ReLU()(X7649)
			JCudaTensor x648;
			JCudaTensor x649;
			x649 = x572;
			x648 = x246.forward(x649);

			// val X7647 = ReLU()(X7646)
			JCudaTensor x650;
			JCudaTensor x651;
			x651 = x614;
			x650 = x246.forward(x651);

			// val X7691 = (X7690[1><3])(i | @) * (b2fc1_W)(j | @)
			JCudaTensor x652;
			JCudaMatrix x653;
			JCudaMatrix x654;
			JCudaTensor x655;
			JCudaTensor x656;
			x656 = x626;
			x655 = x656.flatten(1, new int[]{128, 4, 4});
			x653 = x655.asMatrix(1, true);
			JCudaTensor x657;
			x657 = x658;
			x654 = x657.asMatrix(1, true);
			x652 = x653.times(x654);

			// b1cv_B <~~ V_b1cv_B
			float x659, x660;
			x659 = 1;
			x660 = 1;
			JCudaTensor x661;
			x661 = x643;
			x291.update(x661, x659, x660);

			// b1cv_W <~~ V_b1cv_W
			float x662, x663;
			x662 = 1;
			x663 = decay_1;
			JCudaTensor x664;
			x664 = x636;
			x290.update(x664, x662, x663);

			// val X7693 = (X7691 + (i) => b2fc1_B)
			JCudaTensor x665;
			JCudaTensor x666, x667;
			x666 = x652;
			x667 = x668;
			x665 = x667.copy(128, x666);

			// val X7651 = Concat(X7639,X7643,X7647,X7650)
			JCudaTensor x669;
			JCudaTensor x670, x671, x672, x673;
			x670 = x634;
			x671 = x641;
			x672 = x650;
			x673 = x648;
			x669 = x250.forward(x670,x671,x672,x673);

			// val X7652 = Pooling(3,2,1,true)(X7651)
			JCudaTensor x674;
			JCudaTensor x675;
			x675 = x669;
			x674 = x676.forward(x675);

			// val X7694 = ReLU()(X7693)
			JCudaTensor x677;
			JCudaTensor x678;
			x678 = x665;
			x677 = x348.forward(x678);

			// val X7663 = Pooling(3,1,1,true)(X7652)
			JCudaTensor x679;
			JCudaTensor x680;
			x680 = x674;
			x679 = x681.forward(x680);

			// val X7659 = Convolv(1,0)(X7652,cv84_W,cv84_B)
			JCudaTensor x682;
			JCudaTensor x683, x684, x685;
			x683 = x674;
			x684 = x686;
			x685 = x687;
			x682 = x688.forward(x683, x684, x685);

			// val X7695 = Dropout(0.7)(X7694)
			JCudaTensor x689;
			JCudaTensor x690;
			x690 = x677;
			x689 = x361.forward(x690);

			// val X7655 = Convolv(1,0)(X7652,cv82_W,cv82_B)
			JCudaTensor x691;
			JCudaTensor x692, x693, x694;
			x692 = x674;
			x693 = x695;
			x694 = x696;
			x691 = x697.forward(x692, x693, x694);

			// val X7653 = Convolv(1,0)(X7652,cv81_W,cv81_B)
			JCudaTensor x698;
			JCudaTensor x699, x700, x701;
			x699 = x674;
			x700 = x702;
			x701 = x703;
			x698 = x704.forward(x699, x700, x701);

			// val X7696 = (X7695)(i | @) * (b2fc2_W)(j | @)
			JCudaTensor x705;
			JCudaMatrix x706;
			JCudaMatrix x707;
			JCudaTensor x708;
			x708 = x689;
			x706 = x708.asMatrix(1, true);
			JCudaTensor x709;
			x709 = x710;
			x707 = x709.asMatrix(1, true);
			x705 = x706.times(x707);

			// val X7660 = ReLU()(X7659)
			JCudaTensor x711;
			JCudaTensor x712;
			x712 = x682;
			x711 = x713.forward(x712);

			// val X7664 = Convolv(1,0)(X7663,cv86_W,cv86_B)
			JCudaTensor x714;
			JCudaTensor x715, x716, x717;
			x715 = x679;
			x716 = x718;
			x717 = x719;
			x714 = x720.forward(x715, x716, x717);

			// val X7656 = ReLU()(X7655)
			JCudaTensor x721;
			JCudaTensor x722;
			x722 = x691;
			x721 = x723.forward(x722);

			// val X7661 = Convolv(1,2)(X7660,cv85_W,cv85_B)
			JCudaTensor x724;
			JCudaTensor x725, x726, x727;
			x725 = x711;
			x726 = x728;
			x727 = x729;
			x724 = x730.forward(x725, x726, x727);

			// val X7698 = (X7696 + (i) => b2fc2_B)
			JCudaTensor x731;
			JCudaTensor x732, x733;
			x732 = x705;
			x733 = x734;
			x731 = x733.copy(128, x732);

			// val X7657 = Convolv(1,1)(X7656,cv83_W,cv83_B)
			JCudaTensor x735;
			JCudaTensor x736, x737, x738;
			x736 = x721;
			x737 = x739;
			x738 = x740;
			x735 = x741.forward(x736, x737, x738);

			// val X7662 = ReLU()(X7661)
			JCudaTensor x742;
			JCudaTensor x743;
			x743 = x724;
			x742 = x744.forward(x743);

			// val X7654 = ReLU()(X7653)
			JCudaTensor x745;
			JCudaTensor x746;
			x746 = x698;
			x745 = x747.forward(x746);

			// val X7699 = LogSoftmax()(X7698)
			JCudaTensor x748;
			JCudaTensor x749;
			x749 = x731;
			x748 = x402.forward(x749);

			// Dealloc(X7698)
			JCudaTensor x750;
			x750 = x731;
			x750.free();

			// val X7658 = ReLU()(X7657)
			JCudaTensor x751;
			JCudaTensor x752;
			x752 = x735;
			x751 = x753.forward(x752);

			// val X7665 = ReLU()(X7664)
			JCudaTensor x754;
			JCudaTensor x755;
			x755 = x714;
			x754 = x744.forward(x755);

			// val X7666 = Concat(X7654,X7658,X7662,X7665)
			JCudaTensor x756;
			JCudaTensor x758, x759, x760, x761;
			x758 = x745;
			x759 = x751;
			x760 = x742;
			x761 = x754;
			x756 = x757.forward(x758,x759,x760,x761);

			// val X2419 = X2085 * d_LogSoftmax()(X7699)/d_X7698
			JCudaTensor x762;
			JCudaTensor x763, x764;
			x763 = x404;
			x764 = x748;
			x762 = x402.backward(x763, x764);

			// val m2 = (i3163) => b2fc2_W[@, i3163]
			JCudaMatrix x765;
			JCudaTensor x766;
			x766 = x710;
			x765 = x766.asMatrix(1, false);

			// val X7667 = Convolv(1,0)(X7666,cv91_W,cv91_B)
			JCudaTensor x767;
			JCudaTensor x768, x769, x770;
			x768 = x756;
			x769 = x771;
			x770 = x772;
			x767 = x704.forward(x768, x769, x770);

			// val X7669 = Convolv(1,0)(X7666,cv92_W,cv92_B)
			JCudaTensor x773;
			JCudaTensor x774, x775, x776;
			x774 = x756;
			x775 = x777;
			x776 = x778;
			x773 = x697.forward(x774, x775, x776);

			// val X7677 = Pooling(3,1,1,true)(X7666)
			JCudaTensor x779;
			JCudaTensor x780;
			x780 = x756;
			x779 = x681.forward(x780);

			// val m27 = (i492) => X2419[@, i492]
			JCudaMatrix x781;
			JCudaTensor x782;
			x782 = x762;
			x781 = x782.asMatrix(1, false);

			// val m29 = (i496) => X7695[@, i496]
			JCudaMatrix x783;
			JCudaTensor x784;
			x784 = x689;
			x783 = x784.asMatrix(1, false);

			// val X7673 = Convolv(1,0)(X7666,cv94_W,cv94_B)
			JCudaTensor x785;
			JCudaTensor x786, x787, x788;
			x786 = x756;
			x787 = x789;
			x788 = x790;
			x785 = x688.forward(x786, x787, x788);

			// val X2428 = (X2419)(i3162 | @) * m2
			JCudaTensor x791;
			JCudaMatrix x792;
			JCudaMatrix x793;
			JCudaTensor x794;
			x794 = x762;
			x792 = x794.asMatrix(1, true);
			x793 = x765;
			x791 = x792.times(x793);

			// val X7678 = Convolv(1,0)(X7677,cv96_W,cv96_B)
			JCudaTensor x795;
			JCudaTensor x796, x797, x798;
			x796 = x779;
			x797 = x799;
			x798 = x800;
			x795 = x720.forward(x796, x797, x798);

			// val X2429 = X2428 * d_Dropout(0.7)()/d_X7694
			JCudaTensor x801;
			JCudaTensor x802;
			x802 = x791;
			x801 = x361.backward(x802);

			// Dealloc(X2428)
			JCudaTensor x803;
			x803 = x791;
			x803.free();

			// val X7670 = ReLU()(X7669)
			JCudaTensor x804;
			JCudaTensor x805;
			x805 = x773;
			x804 = x723.forward(x805);

			// val X7674 = ReLU()(X7673)
			JCudaTensor x806;
			JCudaTensor x807;
			x807 = x785;
			x806 = x713.forward(x807);

			// val X5316 = Sum(m27)
			JCudaTensor x808;
			JCudaMatrix x809;
			x809 = x781;
			x808 = x809.sum();

			// val X5318 = m27 * m29
			JCudaTensor x810;
			JCudaMatrix x811;
			JCudaMatrix x812;
			x811 = x781;
			x812 = x783;
			x810 = x811.times(x812);

			// Dealloc(X2419)
			JCudaTensor x813;
			x813 = x762;
			x813.free();

			// Dealloc(X7695)
			JCudaTensor x814;
			x814 = x689;
			x814.free();

			// val X5319 = (X5318 * loss2)
			JCudaTensor x815;
			JCudaTensor x816;
			float x817;
			x816 = x810;
			x817 = loss2;
			x815 = x816.times_i(x817);

			// val X7675 = Convolv(1,2)(X7674,cv95_W,cv95_B)
			JCudaTensor x818;
			JCudaTensor x819, x820, x821;
			x819 = x806;
			x820 = x822;
			x821 = x823;
			x818 = x730.forward(x819, x820, x821);

			// val X7671 = Convolv(1,1)(X7670,cv93_W,cv93_B)
			JCudaTensor x824;
			JCudaTensor x825, x826, x827;
			x825 = x804;
			x826 = x828;
			x827 = x829;
			x824 = x741.forward(x825, x826, x827);

			// val X5317 = (X5316 * loss2)
			JCudaTensor x830;
			JCudaTensor x831;
			float x832;
			x831 = x808;
			x832 = loss2;
			x830 = x831.times_i(x832);

			// val X2431 = X2429 * d_ReLU()(X7694)/d_X7693
			JCudaTensor x833;
			JCudaTensor x834, x835;
			x834 = x801;
			x835 = x677;
			x833 = x348.backward(x834, x835);

			// Dealloc(X7694)
			JCudaTensor x836;
			x836 = x677;
			x836.free();

			// val m3 = (i3167) => b2fc1_W[@, i3167]
			JCudaMatrix x837;
			JCudaTensor x838;
			x838 = x658;
			x837 = x838.asMatrix(1, false);

			// val m26 = (i509) => X7690[1><3][@, i509]
			JCudaMatrix x839;
			JCudaTensor x840;
			JCudaTensor x841;
			x841 = x626;
			x840 = x841.flatten(1, new int[]{128, 4, 4});
			x839 = x840.asMatrix(1, false);

			// val X7668 = ReLU()(X7667)
			JCudaTensor x842;
			JCudaTensor x843;
			x843 = x767;
			x842 = x747.forward(x843);

			// val X2432 = (X2431)(i3166 | @) * m3
			JCudaTensor x844;
			JCudaMatrix x845;
			JCudaMatrix x846;
			JCudaTensor x847;
			x847 = x833;
			x845 = x847.asMatrix(1, true);
			x846 = x837;
			x844 = x845.times(x846);

			// V_b2fc2_B <~~ X5317
			float x849, x850;
			x849 = lrn_rate_2;
			x850 = momentum;
			JCudaTensor x851;
			x851 = x830;
			x848.update(x851, x849, x850);

			// Dealloc(X5317)
			JCudaTensor x852;
			x852 = x830;
			x852.free();

			// val m24 = (i505) => X2431[@, i505]
			JCudaMatrix x853;
			JCudaTensor x854;
			x854 = x833;
			x853 = x854.asMatrix(1, false);

			// val X7676 = ReLU()(X7675)
			JCudaTensor x855;
			JCudaTensor x856;
			x856 = x818;
			x855 = x744.forward(x856);

			// val X7672 = ReLU()(X7671)
			JCudaTensor x857;
			JCudaTensor x858;
			x858 = x824;
			x857 = x753.forward(x858);

			// V_b2fc2_W <~~ X5319
			float x860, x861;
			x860 = lrn_rate_1;
			x861 = momentum;
			JCudaTensor x862;
			x862 = x815;
			x859.update(x862, x860, x861);

			// Dealloc(X5319)
			JCudaTensor x863;
			x863 = x815;
			x863.free();

			// val X7679 = ReLU()(X7678)
			JCudaTensor x864;
			JCudaTensor x865;
			x865 = x795;
			x864 = x744.forward(x865);

			// b2fc2_B <~~ V_b2fc2_B
			float x866, x867;
			x866 = 1;
			x867 = 1;
			JCudaTensor x868;
			x868 = x848;
			x734.update(x868, x866, x867);

			// b2fc2_W <~~ V_b2fc2_W
			float x869, x870;
			x869 = 1;
			x870 = decay_1;
			JCudaTensor x871;
			x871 = x859;
			x710.update(x871, x869, x870);

			// val X2434 = X2432[1<>3] * d_ReLU()(X7690)/d_X7689
			JCudaTensor x872;
			JCudaTensor x873, x874;
			JCudaTensor x875;
			x875 = x844;
			x873 = x875.unflatten(1, new int[]{128, 4, 4});
			x874 = x626;
			x872 = x309.backward(x873, x874);

			// val X5314 = m24 * m26
			JCudaTensor x876;
			JCudaMatrix x877;
			JCudaMatrix x878;
			x877 = x853;
			x878 = x839;
			x876 = x877.times(x878);

			// Dealloc(X7690)
			JCudaTensor x879;
			x879 = x626;
			x879.free();

			// val X5312 = Sum(m24)
			JCudaTensor x880;
			JCudaMatrix x881;
			x881 = x853;
			x880 = x881.sum();

			// Dealloc(X2431)
			JCudaTensor x882;
			x882 = x833;
			x882.free();

			// val X7680 = Concat(X7668,X7672,X7676,X7679)
			JCudaTensor x883;
			JCudaTensor x884, x885, x886, x887;
			x884 = x842;
			x885 = x857;
			x886 = x855;
			x887 = x864;
			x883 = x757.forward(x884,x885,x886,x887);

			// val X5310 = X2434 * d_Convolv(1,0)(X7688)/d_b2cv_W
			JCudaTensor x888;
			JCudaTensor x889, x890;
			x889 = x872;
			x890 = x536;
			x888 = x292.backward_filter(x889, x890);

			// val X5315 = (X5314 * loss2)
			JCudaTensor x891;
			JCudaTensor x892;
			float x893;
			x892 = x876;
			x893 = loss2;
			x891 = x892.times_i(x893);

			// val X5313 = (X5312 * loss2)
			JCudaTensor x894;
			JCudaTensor x895;
			float x896;
			x895 = x880;
			x896 = loss2;
			x894 = x895.times_i(x896);

			// val X2435 = X2434 * d_Convolv(1,0)(b2cv_W)/d_X7688
			JCudaTensor x897;
			JCudaTensor x898, x899;
			x898 = x872;
			x899 = x587;
			x897 = x292.backward_data(x898, x899);

			// val X7681 = Pooling(7,1,0,false)(X7680)
			JCudaTensor x900;
			JCudaTensor x901;
			x901 = x883;
			x900 = x902.forward(x901);

			// val X5308 = X2434 * d_Convolv(1,0)()/d_b2cv_B
			JCudaTensor x903;
			JCudaTensor x904;
			x904 = x872;
			x903 = x292.backward_bias(x904);

			// Dealloc(X2434)
			JCudaTensor x905;
			x905 = x872;
			x905.free();

			// V_b2fc1_W <~~ X5315
			float x907, x908;
			x907 = lrn_rate_1;
			x908 = momentum;
			JCudaTensor x909;
			x909 = x891;
			x906.update(x909, x907, x908);

			// Dealloc(X5315)
			JCudaTensor x910;
			x910 = x891;
			x910.free();

			// val X5309 = (X5308 * loss2)
			JCudaTensor x911;
			JCudaTensor x912;
			float x913;
			x912 = x903;
			x913 = loss2;
			x911 = x912.times_i(x913);

			// val X7682 = Dropout(0.4)(X7681)
			JCudaTensor x914;
			JCudaTensor x915;
			x915 = x900;
			x914 = x916.forward(x915);

			// val X5311 = (X5310 * loss2)
			JCudaTensor x917;
			JCudaTensor x918;
			float x919;
			x918 = x888;
			x919 = loss2;
			x917 = x918.times_i(x919);

			// V_b2fc1_B <~~ X5313
			float x921, x922;
			x921 = lrn_rate_2;
			x922 = momentum;
			JCudaTensor x923;
			x923 = x894;
			x920.update(x923, x921, x922);

			// Dealloc(X5313)
			JCudaTensor x924;
			x924 = x894;
			x924.free();

			// val X2437 = X2435 * d_Pooling(5,3,0,false)(X7688,X7637)/d_X7637
			JCudaTensor x925;
			JCudaTensor x926, x927, x928;
			x926 = x897;
			x927 = x536;
			x928 = x502;
			x925 = x263.backward(x926, x927, x928);

			// Dealloc(X2435)
			JCudaTensor x929;
			x929 = x897;
			x929.free();

			// Dealloc(X7688)
			JCudaTensor x930;
			x930 = x536;
			x930.free();

			// b2fc1_W <~~ V_b2fc1_W
			float x931, x932;
			x931 = 1;
			x932 = decay_1;
			JCudaTensor x933;
			x933 = x906;
			x658.update(x933, x931, x932);

			// b2fc1_B <~~ V_b2fc1_B
			float x934, x935;
			x934 = 1;
			x935 = 1;
			JCudaTensor x936;
			x936 = x920;
			x668.update(x936, x934, x935);

			// V_b2cv_W <~~ X5311
			float x938, x939;
			x938 = lrn_rate_1;
			x939 = momentum;
			JCudaTensor x940;
			x940 = x917;
			x937.update(x940, x938, x939);

			// Dealloc(X5311)
			JCudaTensor x941;
			x941 = x917;
			x941.free();

			// V_b2cv_B <~~ X5309
			float x943, x944;
			x943 = lrn_rate_2;
			x944 = momentum;
			JCudaTensor x945;
			x945 = x911;
			x942.update(x945, x943, x944);

			// Dealloc(X5309)
			JCudaTensor x946;
			x946 = x911;
			x946.free();

			// val X7683 = (X7682[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x947;
			JCudaMatrix x948;
			JCudaMatrix x949;
			JCudaTensor x950;
			JCudaTensor x951;
			x951 = x914;
			x950 = x951.flatten(1, new int[]{256, 1, 1});
			x948 = x950.asMatrix(1, true);
			JCudaTensor x952;
			x952 = x953;
			x949 = x952.asMatrix(1, true);
			x947 = x948.times(x949);

			// b2cv_W <~~ V_b2cv_W
			float x954, x955;
			x954 = 1;
			x955 = decay_1;
			JCudaTensor x956;
			x956 = x937;
			x587.update(x956, x954, x955);

			// b2cv_B <~~ V_b2cv_B
			float x957, x958;
			x957 = 1;
			x958 = 1;
			JCudaTensor x959;
			x959 = x942;
			x588.update(x959, x957, x958);

			// val X7685 = (X7683 + (i) => fc_B)
			JCudaTensor x960;
			JCudaTensor x961, x962;
			x961 = x947;
			x962 = x963;
			x960 = x962.copy(128, x961);

			// val X7686 = LogSoftmax()(X7685)
			JCudaTensor x964;
			JCudaTensor x965;
			x965 = x960;
			x964 = x402.forward(x965);

			// Dealloc(X7685)
			JCudaTensor x966;
			x966 = x960;
			x966.free();

			// Cost(((((0 - (X7687 . X7686)) / |128|) + (((0 - (X7687 . X7699)) / |128|) * loss2)) + (((0 - (X7687 . X7712)) / |128|) * loss1)))
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
			x983 = x748;
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
			x990 = x400;
			x988 = x989.dot(x990);
			x986 = - x988;
			x987 = 128;
			x984 = x986 / x987;
			x985 = loss1;
			x969 = x984 * x985;
			x967 = x968 + x969;
			System.out.println(x5 + " " + x967);
			if (Float.isNaN(x967)) { System.exit(-1); }

			// Dealloc(X7699)
			JCudaTensor x991;
			x991 = x748;
			x991.free();

			// Dealloc(X7712)
			JCudaTensor x992;
			x992 = x400;
			x992.free();

			// Dealloc(X7687)
			JCudaTensor x993;
			x993 = x9;
			x993.free();

			// val X2087 = X2085 * d_LogSoftmax()(X7686)/d_X7685
			JCudaTensor x994;
			JCudaTensor x995, x996;
			x995 = x404;
			x996 = x964;
			x994 = x402.backward(x995, x996);

			// Dealloc(X7686)
			JCudaTensor x997;
			x997 = x964;
			x997.free();

			// Dealloc(X2085)
			JCudaTensor x998;
			x998 = x404;
			x998.free();

			// val m1 = (i2995) => fc_W[@, i2995]
			JCudaMatrix x999;
			JCudaTensor x1000;
			x1000 = x953;
			x999 = x1000.asMatrix(1, false);

			// val X2133 = (X2087)(i2994 | @) * m1
			JCudaTensor x1001;
			JCudaMatrix x1002;
			JCudaMatrix x1003;
			JCudaTensor x1004;
			x1004 = x994;
			x1002 = x1004.asMatrix(1, true);
			x1003 = x999;
			x1001 = x1002.times(x1003);

			// val m30 = (i21) => X2087[@, i21]
			JCudaMatrix x1005;
			JCudaTensor x1006;
			x1006 = x994;
			x1005 = x1006.asMatrix(1, false);

			// val m32 = (i25) => X7682[1><3][@, i25]
			JCudaMatrix x1007;
			JCudaTensor x1008;
			JCudaTensor x1009;
			x1009 = x914;
			x1008 = x1009.flatten(1, new int[]{256, 1, 1});
			x1007 = x1008.asMatrix(1, false);

			// V_fc_W <~~ m30 * m32
			float x1011, x1012;
			x1011 = lrn_rate_1;
			x1012 = momentum;
			JCudaMatrix x1013;
			JCudaMatrix x1014;
			x1013 = x1005;
			x1014 = x1007;
			x1013.times(x1014, x1010, x1011, x1012);

			// Dealloc(X7682)
			JCudaTensor x1015;
			x1015 = x914;
			x1015.free();

			// val X2134 = X2133[1<>3] * d_Dropout(0.4)()/d_X7681
			JCudaTensor x1016;
			JCudaTensor x1017;
			JCudaTensor x1018;
			x1018 = x1001;
			x1017 = x1018.unflatten(1, new int[]{256, 1, 1});
			x1016 = x916.backward(x1017);

			// Dealloc(X2133)
			JCudaTensor x1019;
			x1019 = x1001;
			x1019.free();

			// V_fc_B <~~ Sum(m30)
			float x1021, x1022;
			x1021 = lrn_rate_2;
			x1022 = momentum;
			JCudaMatrix x1023;
			x1023 = x1005;
			x1023.sum(x1020, x1021, x1022);

			// Dealloc(X2087)
			JCudaTensor x1024;
			x1024 = x994;
			x1024.free();

			// fc_W <~~ V_fc_W
			float x1025, x1026;
			x1025 = 1;
			x1026 = decay_1;
			JCudaTensor x1027;
			x1027 = x1010;
			x953.update(x1027, x1025, x1026);

			// fc_B <~~ V_fc_B
			float x1028, x1029;
			x1028 = 1;
			x1029 = 1;
			JCudaTensor x1030;
			x1030 = x1020;
			x963.update(x1030, x1028, x1029);

			// val X2136 = X2134 * d_Pooling(7,1,0,false)(X7681,X7680)/d_X7680
			JCudaTensor x1031;
			JCudaTensor x1032, x1033, x1034;
			x1032 = x1016;
			x1033 = x900;
			x1034 = x883;
			x1031 = x902.backward(x1032, x1033, x1034);

			// Dealloc(X2134)
			JCudaTensor x1035;
			x1035 = x1016;
			x1035.free();

			// Dealloc(X7681)
			JCudaTensor x1036;
			x1036 = x900;
			x1036.free();

			// Dealloc(X7680)
			JCudaTensor x1037;
			x1037 = x883;
			x1037.free();

			// val X2192 = Proj(X2136, X7668,X7672,X7676,X7679, 2)
			JCudaTensor x1038;
			JCudaTensor x1040;
			x1040 = x1031;
			JCudaTensor[] x1039 = x757.backward(x1040);
			x1038 = x1039[2];

			// val X2150 = Proj(X2136, X7668,X7672,X7676,X7679, 0)
			JCudaTensor x1041;
			x1041 = x1039[0];

			// val X2168 = Proj(X2136, X7668,X7672,X7676,X7679, 1)
			JCudaTensor x1042;
			x1042 = x1039[1];

			// val X2216 = Proj(X2136, X7668,X7672,X7676,X7679, 3)
			JCudaTensor x1043;
			x1043 = x1039[3];

			// Dealloc(X2136)
			JCudaTensor x1044;
			x1044 = x1031;
			x1044.free();

			// val X2173 = X2168 * d_ReLU()(X7672)/d_X7671
			JCudaTensor x1045;
			JCudaTensor x1046, x1047;
			x1046 = x1042;
			x1047 = x857;
			x1045 = x753.backward(x1046, x1047);

			// Dealloc(X7672)
			JCudaTensor x1048;
			x1048 = x857;
			x1048.free();

			// val X2153 = X2150 * d_ReLU()(X7668)/d_X7667
			JCudaTensor x1049;
			JCudaTensor x1050, x1051;
			x1050 = x1041;
			x1051 = x842;
			x1049 = x747.backward(x1050, x1051);

			// Dealloc(X7668)
			JCudaTensor x1052;
			x1052 = x842;
			x1052.free();

			// val X2197 = X2192 * d_ReLU()(X7676)/d_X7675
			JCudaTensor x1053;
			JCudaTensor x1054, x1055;
			x1054 = x1038;
			x1055 = x855;
			x1053 = x744.backward(x1054, x1055);

			// Dealloc(X7676)
			JCudaTensor x1056;
			x1056 = x855;
			x1056.free();

			// val X2220 = X2216 * d_ReLU()(X7679)/d_X7678
			JCudaTensor x1057;
			JCudaTensor x1058, x1059;
			x1058 = x1043;
			x1059 = x864;
			x1057 = x744.backward(x1058, x1059);

			// Dealloc(X7679)
			JCudaTensor x1060;
			x1060 = x864;
			x1060.free();

			// V_cv93_W <~~ X2173 * d_Convolv(1,1)(X7670)/d_cv93_W
			float x1062, x1063;
			x1062 = lrn_rate_1;
			x1063 = momentum;
			JCudaTensor x1064, x1065;
			x1064 = x1045;
			x1065 = x804;
			x741.backward_filter(x1064, x1065, x1061, x1062, x1063);

			// V_cv91_W <~~ X2153 * d_Convolv(1,0)(X7666)/d_cv91_W
			float x1067, x1068;
			x1067 = lrn_rate_1;
			x1068 = momentum;
			JCudaTensor x1069, x1070;
			x1069 = x1049;
			x1070 = x756;
			x704.backward_filter(x1069, x1070, x1066, x1067, x1068);

			// V_cv95_B <~~ X2197 * d_Convolv(1,2)()/d_cv95_B
			float x1072, x1073;
			x1072 = lrn_rate_2;
			x1073 = momentum;
			JCudaTensor x1074;
			x1074 = x1053;
			x730.backward_bias(x1074, x1071, x1072, x1073);

			// V_cv91_B <~~ X2153 * d_Convolv(1,0)()/d_cv91_B
			float x1076, x1077;
			x1076 = lrn_rate_2;
			x1077 = momentum;
			JCudaTensor x1078;
			x1078 = x1049;
			x704.backward_bias(x1078, x1075, x1076, x1077);

			// V_cv96_W <~~ X2220 * d_Convolv(1,0)(X7677)/d_cv96_W
			float x1080, x1081;
			x1080 = lrn_rate_1;
			x1081 = momentum;
			JCudaTensor x1082, x1083;
			x1082 = x1057;
			x1083 = x779;
			x720.backward_filter(x1082, x1083, x1079, x1080, x1081);

			// val X2174 = X2173 * d_Convolv(1,1)(cv93_W)/d_X7670
			JCudaTensor x1084;
			JCudaTensor x1085, x1086;
			x1085 = x1045;
			x1086 = x828;
			x1084 = x741.backward_data(x1085, x1086);

			// V_cv93_B <~~ X2173 * d_Convolv(1,1)()/d_cv93_B
			float x1088, x1089;
			x1088 = lrn_rate_2;
			x1089 = momentum;
			JCudaTensor x1090;
			x1090 = x1045;
			x741.backward_bias(x1090, x1087, x1088, x1089);

			// Dealloc(X2173)
			JCudaTensor x1091;
			x1091 = x1045;
			x1091.free();

			// V_cv96_B <~~ X2220 * d_Convolv(1,0)()/d_cv96_B
			float x1093, x1094;
			x1093 = lrn_rate_2;
			x1094 = momentum;
			JCudaTensor x1095;
			x1095 = x1057;
			x720.backward_bias(x1095, x1092, x1093, x1094);

			// val X2154 = X2153 * d_Convolv(1,0)(cv91_W)/d_X7666
			JCudaTensor x1096;
			JCudaTensor x1097, x1098;
			x1097 = x1049;
			x1098 = x771;
			x1096 = x704.backward_data(x1097, x1098);

			// Dealloc(X2153)
			JCudaTensor x1099;
			x1099 = x1049;
			x1099.free();

			// V_cv95_W <~~ X2197 * d_Convolv(1,2)(X7674)/d_cv95_W
			float x1101, x1102;
			x1101 = lrn_rate_1;
			x1102 = momentum;
			JCudaTensor x1103, x1104;
			x1103 = x1053;
			x1104 = x806;
			x730.backward_filter(x1103, x1104, x1100, x1101, x1102);

			// val X2198 = X2197 * d_Convolv(1,2)(cv95_W)/d_X7674
			JCudaTensor x1105;
			JCudaTensor x1106, x1107;
			x1106 = x1053;
			x1107 = x822;
			x1105 = x730.backward_data(x1106, x1107);

			// Dealloc(X2197)
			JCudaTensor x1108;
			x1108 = x1053;
			x1108.free();

			// val X2221 = X2220 * d_Convolv(1,0)(cv96_W)/d_X7677
			JCudaTensor x1109;
			JCudaTensor x1110, x1111;
			x1110 = x1057;
			x1111 = x799;
			x1109 = x720.backward_data(x1110, x1111);

			// Dealloc(X2220)
			JCudaTensor x1112;
			x1112 = x1057;
			x1112.free();

			// cv93_W <~~ V_cv93_W
			float x1113, x1114;
			x1113 = 1;
			x1114 = decay_1;
			JCudaTensor x1115;
			x1115 = x1061;
			x828.update(x1115, x1113, x1114);

			// cv91_W <~~ V_cv91_W
			float x1116, x1117;
			x1116 = 1;
			x1117 = decay_1;
			JCudaTensor x1118;
			x1118 = x1066;
			x771.update(x1118, x1116, x1117);

			// cv91_B <~~ V_cv91_B
			float x1119, x1120;
			x1119 = 1;
			x1120 = 1;
			JCudaTensor x1121;
			x1121 = x1075;
			x772.update(x1121, x1119, x1120);

			// cv96_W <~~ V_cv96_W
			float x1122, x1123;
			x1122 = 1;
			x1123 = decay_1;
			JCudaTensor x1124;
			x1124 = x1079;
			x799.update(x1124, x1122, x1123);

			// cv96_B <~~ V_cv96_B
			float x1125, x1126;
			x1125 = 1;
			x1126 = 1;
			JCudaTensor x1127;
			x1127 = x1092;
			x800.update(x1127, x1125, x1126);

			// cv95_B <~~ V_cv95_B
			float x1128, x1129;
			x1128 = 1;
			x1129 = 1;
			JCudaTensor x1130;
			x1130 = x1071;
			x823.update(x1130, x1128, x1129);

			// cv95_W <~~ V_cv95_W
			float x1131, x1132;
			x1131 = 1;
			x1132 = decay_1;
			JCudaTensor x1133;
			x1133 = x1100;
			x822.update(x1133, x1131, x1132);

			// cv93_B <~~ V_cv93_B
			float x1134, x1135;
			x1134 = 1;
			x1135 = 1;
			JCudaTensor x1136;
			x1136 = x1087;
			x829.update(x1136, x1134, x1135);

			// val X2176 = X2174 * d_ReLU()(X7670)/d_X7669
			JCudaTensor x1137;
			JCudaTensor x1138, x1139;
			x1138 = x1084;
			x1139 = x804;
			x1137 = x723.backward(x1138, x1139);

			// Dealloc(X7670)
			JCudaTensor x1140;
			x1140 = x804;
			x1140.free();

			// val X2200 = X2198 * d_ReLU()(X7674)/d_X7673
			JCudaTensor x1141;
			JCudaTensor x1142, x1143;
			x1142 = x1105;
			x1143 = x806;
			x1141 = x713.backward(x1142, x1143);

			// Dealloc(X7674)
			JCudaTensor x1144;
			x1144 = x806;
			x1144.free();

			// V_cv94_W <~~ X2200 * d_Convolv(1,0)(X7666)/d_cv94_W
			float x1146, x1147;
			x1146 = lrn_rate_1;
			x1147 = momentum;
			JCudaTensor x1148, x1149;
			x1148 = x1141;
			x1149 = x756;
			x688.backward_filter(x1148, x1149, x1145, x1146, x1147);

			// V_cv94_B <~~ X2200 * d_Convolv(1,0)()/d_cv94_B
			float x1151, x1152;
			x1151 = lrn_rate_2;
			x1152 = momentum;
			JCudaTensor x1153;
			x1153 = x1141;
			x688.backward_bias(x1153, x1150, x1151, x1152);

			// val X2178 = (X2154 + X2176 * d_Convolv(1,0)(cv92_W)/d_X7666)
			JCudaTensor x1154;
			JCudaTensor x1155;
			x1155 = x1096;
			JCudaTensor x1156, x1157;
			x1156 = x1137;
			x1157 = x777;
			x1154 = x697.backward_data(x1156,x1157, x1155);

			// V_cv92_W <~~ X2176 * d_Convolv(1,0)(X7666)/d_cv92_W
			float x1159, x1160;
			x1159 = lrn_rate_1;
			x1160 = momentum;
			JCudaTensor x1161, x1162;
			x1161 = x1137;
			x1162 = x756;
			x697.backward_filter(x1161, x1162, x1158, x1159, x1160);

			// V_cv92_B <~~ X2176 * d_Convolv(1,0)()/d_cv92_B
			float x1164, x1165;
			x1164 = lrn_rate_2;
			x1165 = momentum;
			JCudaTensor x1166;
			x1166 = x1137;
			x697.backward_bias(x1166, x1163, x1164, x1165);

			// Dealloc(X2176)
			JCudaTensor x1167;
			x1167 = x1137;
			x1167.free();

			// cv94_B <~~ V_cv94_B
			float x1168, x1169;
			x1168 = 1;
			x1169 = 1;
			JCudaTensor x1170;
			x1170 = x1150;
			x790.update(x1170, x1168, x1169);

			// cv92_W <~~ V_cv92_W
			float x1171, x1172;
			x1171 = 1;
			x1172 = decay_1;
			JCudaTensor x1173;
			x1173 = x1158;
			x777.update(x1173, x1171, x1172);

			// cv92_B <~~ V_cv92_B
			float x1174, x1175;
			x1174 = 1;
			x1175 = 1;
			JCudaTensor x1176;
			x1176 = x1163;
			x778.update(x1176, x1174, x1175);

			// val X2202 = (X2178 + X2200 * d_Convolv(1,0)(cv94_W)/d_X7666)
			JCudaTensor x1177;
			JCudaTensor x1178;
			x1178 = x1154;
			JCudaTensor x1179, x1180;
			x1179 = x1141;
			x1180 = x789;
			x1177 = x688.backward_data(x1179,x1180, x1178);

			// Dealloc(X2200)
			JCudaTensor x1181;
			x1181 = x1141;
			x1181.free();

			// cv94_W <~~ V_cv94_W
			float x1182, x1183;
			x1182 = 1;
			x1183 = decay_1;
			JCudaTensor x1184;
			x1184 = x1145;
			x789.update(x1184, x1182, x1183);

			// val X2224 = (X2202 + X2221 * d_Pooling(3,1,1,true)(X7677,X7666)/d_X7666)
			JCudaTensor x1185;
			JCudaTensor x1186;
			x1186 = x1177;
			JCudaTensor x1187, x1188, x1189;
			x1187 = x1109;
			x1188 = x779;
			x1189 = x756;
			x1185 = x681.backward(x1187,x1188,x1189, x1186);

			// Dealloc(X2221)
			JCudaTensor x1190;
			x1190 = x1109;
			x1190.free();

			// Dealloc(X7677)
			JCudaTensor x1191;
			x1191 = x779;
			x1191.free();

			// Dealloc(X7666)
			JCudaTensor x1192;
			x1192 = x756;
			x1192.free();

			// val X2256 = Proj(X2224, X7654,X7658,X7662,X7665, 1)
			JCudaTensor x1193;
			JCudaTensor x1195;
			x1195 = x1185;
			JCudaTensor[] x1194 = x757.backward(x1195);
			x1193 = x1194[1];

			// val X2280 = Proj(X2224, X7654,X7658,X7662,X7665, 2)
			JCudaTensor x1196;
			x1196 = x1194[2];

			// val X2304 = Proj(X2224, X7654,X7658,X7662,X7665, 3)
			JCudaTensor x1197;
			x1197 = x1194[3];

			// val X2238 = Proj(X2224, X7654,X7658,X7662,X7665, 0)
			JCudaTensor x1198;
			x1198 = x1194[0];

			// Dealloc(X2224)
			JCudaTensor x1199;
			x1199 = x1185;
			x1199.free();

			// val X2285 = X2280 * d_ReLU()(X7662)/d_X7661
			JCudaTensor x1200;
			JCudaTensor x1201, x1202;
			x1201 = x1196;
			x1202 = x742;
			x1200 = x744.backward(x1201, x1202);

			// Dealloc(X7662)
			JCudaTensor x1203;
			x1203 = x742;
			x1203.free();

			// val X2241 = X2238 * d_ReLU()(X7654)/d_X7653
			JCudaTensor x1204;
			JCudaTensor x1205, x1206;
			x1205 = x1198;
			x1206 = x745;
			x1204 = x747.backward(x1205, x1206);

			// Dealloc(X7654)
			JCudaTensor x1207;
			x1207 = x745;
			x1207.free();

			// val X2308 = X2304 * d_ReLU()(X7665)/d_X7664
			JCudaTensor x1208;
			JCudaTensor x1209, x1210;
			x1209 = x1197;
			x1210 = x754;
			x1208 = x744.backward(x1209, x1210);

			// Dealloc(X7665)
			JCudaTensor x1211;
			x1211 = x754;
			x1211.free();

			// val X2261 = X2256 * d_ReLU()(X7658)/d_X7657
			JCudaTensor x1212;
			JCudaTensor x1213, x1214;
			x1213 = x1193;
			x1214 = x751;
			x1212 = x753.backward(x1213, x1214);

			// Dealloc(X7658)
			JCudaTensor x1215;
			x1215 = x751;
			x1215.free();

			// val X2286 = X2285 * d_Convolv(1,2)(cv85_W)/d_X7660
			JCudaTensor x1216;
			JCudaTensor x1217, x1218;
			x1217 = x1200;
			x1218 = x728;
			x1216 = x730.backward_data(x1217, x1218);

			// V_cv81_W <~~ X2241 * d_Convolv(1,0)(X7652)/d_cv81_W
			float x1220, x1221;
			x1220 = lrn_rate_1;
			x1221 = momentum;
			JCudaTensor x1222, x1223;
			x1222 = x1204;
			x1223 = x674;
			x704.backward_filter(x1222, x1223, x1219, x1220, x1221);

			// V_cv86_B <~~ X2308 * d_Convolv(1,0)()/d_cv86_B
			float x1225, x1226;
			x1225 = lrn_rate_2;
			x1226 = momentum;
			JCudaTensor x1227;
			x1227 = x1208;
			x720.backward_bias(x1227, x1224, x1225, x1226);

			// V_cv83_W <~~ X2261 * d_Convolv(1,1)(X7656)/d_cv83_W
			float x1229, x1230;
			x1229 = lrn_rate_1;
			x1230 = momentum;
			JCudaTensor x1231, x1232;
			x1231 = x1212;
			x1232 = x721;
			x741.backward_filter(x1231, x1232, x1228, x1229, x1230);

			// val X2309 = X2308 * d_Convolv(1,0)(cv86_W)/d_X7663
			JCudaTensor x1233;
			JCudaTensor x1234, x1235;
			x1234 = x1208;
			x1235 = x718;
			x1233 = x720.backward_data(x1234, x1235);

			// val X2242 = X2241 * d_Convolv(1,0)(cv81_W)/d_X7652
			JCudaTensor x1236;
			JCudaTensor x1237, x1238;
			x1237 = x1204;
			x1238 = x702;
			x1236 = x704.backward_data(x1237, x1238);

			// V_cv81_B <~~ X2241 * d_Convolv(1,0)()/d_cv81_B
			float x1240, x1241;
			x1240 = lrn_rate_2;
			x1241 = momentum;
			JCudaTensor x1242;
			x1242 = x1204;
			x704.backward_bias(x1242, x1239, x1240, x1241);

			// Dealloc(X2241)
			JCudaTensor x1243;
			x1243 = x1204;
			x1243.free();

			// V_cv83_B <~~ X2261 * d_Convolv(1,1)()/d_cv83_B
			float x1245, x1246;
			x1245 = lrn_rate_2;
			x1246 = momentum;
			JCudaTensor x1247;
			x1247 = x1212;
			x741.backward_bias(x1247, x1244, x1245, x1246);

			// V_cv86_W <~~ X2308 * d_Convolv(1,0)(X7663)/d_cv86_W
			float x1249, x1250;
			x1249 = lrn_rate_1;
			x1250 = momentum;
			JCudaTensor x1251, x1252;
			x1251 = x1208;
			x1252 = x679;
			x720.backward_filter(x1251, x1252, x1248, x1249, x1250);

			// Dealloc(X2308)
			JCudaTensor x1253;
			x1253 = x1208;
			x1253.free();

			// val X2262 = X2261 * d_Convolv(1,1)(cv83_W)/d_X7656
			JCudaTensor x1254;
			JCudaTensor x1255, x1256;
			x1255 = x1212;
			x1256 = x739;
			x1254 = x741.backward_data(x1255, x1256);

			// Dealloc(X2261)
			JCudaTensor x1257;
			x1257 = x1212;
			x1257.free();

			// V_cv85_W <~~ X2285 * d_Convolv(1,2)(X7660)/d_cv85_W
			float x1259, x1260;
			x1259 = lrn_rate_1;
			x1260 = momentum;
			JCudaTensor x1261, x1262;
			x1261 = x1200;
			x1262 = x711;
			x730.backward_filter(x1261, x1262, x1258, x1259, x1260);

			// V_cv85_B <~~ X2285 * d_Convolv(1,2)()/d_cv85_B
			float x1264, x1265;
			x1264 = lrn_rate_2;
			x1265 = momentum;
			JCudaTensor x1266;
			x1266 = x1200;
			x730.backward_bias(x1266, x1263, x1264, x1265);

			// Dealloc(X2285)
			JCudaTensor x1267;
			x1267 = x1200;
			x1267.free();

			// cv85_W <~~ V_cv85_W
			float x1268, x1269;
			x1268 = 1;
			x1269 = decay_1;
			JCudaTensor x1270;
			x1270 = x1258;
			x728.update(x1270, x1268, x1269);

			// cv86_B <~~ V_cv86_B
			float x1271, x1272;
			x1271 = 1;
			x1272 = 1;
			JCudaTensor x1273;
			x1273 = x1224;
			x719.update(x1273, x1271, x1272);

			// cv83_W <~~ V_cv83_W
			float x1274, x1275;
			x1274 = 1;
			x1275 = decay_1;
			JCudaTensor x1276;
			x1276 = x1228;
			x739.update(x1276, x1274, x1275);

			// cv86_W <~~ V_cv86_W
			float x1277, x1278;
			x1277 = 1;
			x1278 = decay_1;
			JCudaTensor x1279;
			x1279 = x1248;
			x718.update(x1279, x1277, x1278);

			// cv83_B <~~ V_cv83_B
			float x1280, x1281;
			x1280 = 1;
			x1281 = 1;
			JCudaTensor x1282;
			x1282 = x1244;
			x740.update(x1282, x1280, x1281);

			// cv85_B <~~ V_cv85_B
			float x1283, x1284;
			x1283 = 1;
			x1284 = 1;
			JCudaTensor x1285;
			x1285 = x1263;
			x729.update(x1285, x1283, x1284);

			// cv81_W <~~ V_cv81_W
			float x1286, x1287;
			x1286 = 1;
			x1287 = decay_1;
			JCudaTensor x1288;
			x1288 = x1219;
			x702.update(x1288, x1286, x1287);

			// cv81_B <~~ V_cv81_B
			float x1289, x1290;
			x1289 = 1;
			x1290 = 1;
			JCudaTensor x1291;
			x1291 = x1239;
			x703.update(x1291, x1289, x1290);

			// val X2288 = X2286 * d_ReLU()(X7660)/d_X7659
			JCudaTensor x1292;
			JCudaTensor x1293, x1294;
			x1293 = x1216;
			x1294 = x711;
			x1292 = x713.backward(x1293, x1294);

			// Dealloc(X7660)
			JCudaTensor x1295;
			x1295 = x711;
			x1295.free();

			// val X2264 = X2262 * d_ReLU()(X7656)/d_X7655
			JCudaTensor x1296;
			JCudaTensor x1297, x1298;
			x1297 = x1254;
			x1298 = x721;
			x1296 = x723.backward(x1297, x1298);

			// Dealloc(X7656)
			JCudaTensor x1299;
			x1299 = x721;
			x1299.free();

			// V_cv84_W <~~ X2288 * d_Convolv(1,0)(X7652)/d_cv84_W
			float x1301, x1302;
			x1301 = lrn_rate_1;
			x1302 = momentum;
			JCudaTensor x1303, x1304;
			x1303 = x1292;
			x1304 = x674;
			x688.backward_filter(x1303, x1304, x1300, x1301, x1302);

			// val X2266 = (X2242 + X2264 * d_Convolv(1,0)(cv82_W)/d_X7652)
			JCudaTensor x1305;
			JCudaTensor x1306;
			x1306 = x1236;
			JCudaTensor x1307, x1308;
			x1307 = x1296;
			x1308 = x695;
			x1305 = x697.backward_data(x1307,x1308, x1306);

			// V_cv82_B <~~ X2264 * d_Convolv(1,0)()/d_cv82_B
			float x1310, x1311;
			x1310 = lrn_rate_2;
			x1311 = momentum;
			JCudaTensor x1312;
			x1312 = x1296;
			x697.backward_bias(x1312, x1309, x1310, x1311);

			// V_cv82_W <~~ X2264 * d_Convolv(1,0)(X7652)/d_cv82_W
			float x1314, x1315;
			x1314 = lrn_rate_1;
			x1315 = momentum;
			JCudaTensor x1316, x1317;
			x1316 = x1296;
			x1317 = x674;
			x697.backward_filter(x1316, x1317, x1313, x1314, x1315);

			// Dealloc(X2264)
			JCudaTensor x1318;
			x1318 = x1296;
			x1318.free();

			// V_cv84_B <~~ X2288 * d_Convolv(1,0)()/d_cv84_B
			float x1320, x1321;
			x1320 = lrn_rate_2;
			x1321 = momentum;
			JCudaTensor x1322;
			x1322 = x1292;
			x688.backward_bias(x1322, x1319, x1320, x1321);

			// cv82_B <~~ V_cv82_B
			float x1323, x1324;
			x1323 = 1;
			x1324 = 1;
			JCudaTensor x1325;
			x1325 = x1309;
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
			x687.update(x1331, x1329, x1330);

			// val X2290 = (X2266 + X2288 * d_Convolv(1,0)(cv84_W)/d_X7652)
			JCudaTensor x1332;
			JCudaTensor x1333;
			x1333 = x1305;
			JCudaTensor x1334, x1335;
			x1334 = x1292;
			x1335 = x686;
			x1332 = x688.backward_data(x1334,x1335, x1333);

			// Dealloc(X2288)
			JCudaTensor x1336;
			x1336 = x1292;
			x1336.free();

			// cv84_W <~~ V_cv84_W
			float x1337, x1338;
			x1337 = 1;
			x1338 = decay_1;
			JCudaTensor x1339;
			x1339 = x1300;
			x686.update(x1339, x1337, x1338);

			// val X2312 = (X2290 + X2309 * d_Pooling(3,1,1,true)(X7663,X7652)/d_X7652)
			JCudaTensor x1340;
			JCudaTensor x1341;
			x1341 = x1332;
			JCudaTensor x1342, x1343, x1344;
			x1342 = x1233;
			x1343 = x679;
			x1344 = x674;
			x1340 = x681.backward(x1342,x1343,x1344, x1341);

			// Dealloc(X2309)
			JCudaTensor x1345;
			x1345 = x1233;
			x1345.free();

			// Dealloc(X7663)
			JCudaTensor x1346;
			x1346 = x679;
			x1346.free();

			// val X2314 = X2312 * d_Pooling(3,2,1,true)(X7652,X7651)/d_X7651
			JCudaTensor x1347;
			JCudaTensor x1348, x1349, x1350;
			x1348 = x1340;
			x1349 = x674;
			x1350 = x669;
			x1347 = x676.backward(x1348, x1349, x1350);

			// Dealloc(X2312)
			JCudaTensor x1351;
			x1351 = x1340;
			x1351.free();

			// Dealloc(X7652)
			JCudaTensor x1352;
			x1352 = x674;
			x1352.free();

			// Dealloc(X7651)
			JCudaTensor x1353;
			x1353 = x669;
			x1353.free();

			// val X2346 = Proj(X2314, X7639,X7643,X7647,X7650, 1)
			JCudaTensor x1354;
			JCudaTensor x1356;
			x1356 = x1347;
			JCudaTensor[] x1355 = x250.backward(x1356);
			x1354 = x1355[1];

			// val X2328 = Proj(X2314, X7639,X7643,X7647,X7650, 0)
			JCudaTensor x1357;
			x1357 = x1355[0];

			// val X2394 = Proj(X2314, X7639,X7643,X7647,X7650, 3)
			JCudaTensor x1358;
			x1358 = x1355[3];

			// val X2370 = Proj(X2314, X7639,X7643,X7647,X7650, 2)
			JCudaTensor x1359;
			x1359 = x1355[2];

			// Dealloc(X2314)
			JCudaTensor x1360;
			x1360 = x1347;
			x1360.free();

			// val X2331 = X2328 * d_ReLU()(X7639)/d_X7638
			JCudaTensor x1361;
			JCudaTensor x1362, x1363;
			x1362 = x1357;
			x1363 = x634;
			x1361 = x240.backward(x1362, x1363);

			// Dealloc(X7639)
			JCudaTensor x1364;
			x1364 = x634;
			x1364.free();

			// val X2375 = X2370 * d_ReLU()(X7647)/d_X7646
			JCudaTensor x1365;
			JCudaTensor x1366, x1367;
			x1366 = x1359;
			x1367 = x650;
			x1365 = x246.backward(x1366, x1367);

			// Dealloc(X7647)
			JCudaTensor x1368;
			x1368 = x650;
			x1368.free();

			// val X2351 = X2346 * d_ReLU()(X7643)/d_X7642
			JCudaTensor x1369;
			JCudaTensor x1370, x1371;
			x1370 = x1354;
			x1371 = x641;
			x1369 = x243.backward(x1370, x1371);

			// Dealloc(X7643)
			JCudaTensor x1372;
			x1372 = x641;
			x1372.free();

			// val X2398 = X2394 * d_ReLU()(X7650)/d_X7649
			JCudaTensor x1373;
			JCudaTensor x1374, x1375;
			x1374 = x1358;
			x1375 = x648;
			x1373 = x246.backward(x1374, x1375);

			// Dealloc(X7650)
			JCudaTensor x1376;
			x1376 = x648;
			x1376.free();

			// V_cv71_W <~~ X2331 * d_Convolv(1,0)(X7637)/d_cv71_W
			float x1378, x1379;
			x1378 = lrn_rate_1;
			x1379 = momentum;
			JCudaTensor x1380, x1381;
			x1380 = x1361;
			x1381 = x502;
			x200.backward_filter(x1380, x1381, x1377, x1378, x1379);

			// val X2376 = X2375 * d_Convolv(1,2)(cv75_W)/d_X7645
			JCudaTensor x1382;
			JCudaTensor x1383, x1384;
			x1383 = x1365;
			x1384 = x618;
			x1382 = x230.backward_data(x1383, x1384);

			// val X2352 = X2351 * d_Convolv(1,1)(cv73_W)/d_X7641
			JCudaTensor x1385;
			JCudaTensor x1386, x1387;
			x1386 = x1369;
			x1387 = x612;
			x1385 = x237.backward_data(x1386, x1387);

			// V_cv75_B <~~ X2375 * d_Convolv(1,2)()/d_cv75_B
			float x1389, x1390;
			x1389 = lrn_rate_2;
			x1390 = momentum;
			JCudaTensor x1391;
			x1391 = x1365;
			x230.backward_bias(x1391, x1388, x1389, x1390);

			// V_cv75_W <~~ X2375 * d_Convolv(1,2)(X7645)/d_cv75_W
			float x1393, x1394;
			x1393 = lrn_rate_1;
			x1394 = momentum;
			JCudaTensor x1395, x1396;
			x1395 = x1365;
			x1396 = x564;
			x230.backward_filter(x1395, x1396, x1392, x1393, x1394);

			// Dealloc(X2375)
			JCudaTensor x1397;
			x1397 = x1365;
			x1397.free();

			// V_cv73_B <~~ X2351 * d_Convolv(1,1)()/d_cv73_B
			float x1399, x1400;
			x1399 = lrn_rate_2;
			x1400 = momentum;
			JCudaTensor x1401;
			x1401 = x1369;
			x237.backward_bias(x1401, x1398, x1399, x1400);

			// val X2399 = X2398 * d_Convolv(1,0)(cv76_W)/d_X7648
			JCudaTensor x1402;
			JCudaTensor x1403, x1404;
			x1403 = x1373;
			x1404 = x576;
			x1402 = x220.backward_data(x1403, x1404);

			// V_cv71_B <~~ X2331 * d_Convolv(1,0)()/d_cv71_B
			float x1406, x1407;
			x1406 = lrn_rate_2;
			x1407 = momentum;
			JCudaTensor x1408;
			x1408 = x1361;
			x200.backward_bias(x1408, x1405, x1406, x1407);

			// V_cv76_W <~~ X2398 * d_Convolv(1,0)(X7648)/d_cv76_W
			float x1410, x1411;
			x1410 = lrn_rate_1;
			x1411 = momentum;
			JCudaTensor x1412, x1413;
			x1412 = x1373;
			x1413 = x538;
			x220.backward_filter(x1412, x1413, x1409, x1410, x1411);

			// V_cv73_W <~~ X2351 * d_Convolv(1,1)(X7641)/d_cv73_W
			float x1415, x1416;
			x1415 = lrn_rate_1;
			x1416 = momentum;
			JCudaTensor x1417, x1418;
			x1417 = x1369;
			x1418 = x578;
			x237.backward_filter(x1417, x1418, x1414, x1415, x1416);

			// Dealloc(X2351)
			JCudaTensor x1419;
			x1419 = x1369;
			x1419.free();

			// val X2332 = X2331 * d_Convolv(1,0)(cv71_W)/d_X7637
			JCudaTensor x1420;
			JCudaTensor x1421, x1422;
			x1421 = x1361;
			x1422 = x554;
			x1420 = x200.backward_data(x1421, x1422);

			// Dealloc(X2331)
			JCudaTensor x1423;
			x1423 = x1361;
			x1423.free();

			// V_cv76_B <~~ X2398 * d_Convolv(1,0)()/d_cv76_B
			float x1425, x1426;
			x1425 = lrn_rate_2;
			x1426 = momentum;
			JCudaTensor x1427;
			x1427 = x1373;
			x220.backward_bias(x1427, x1424, x1425, x1426);

			// Dealloc(X2398)
			JCudaTensor x1428;
			x1428 = x1373;
			x1428.free();

			// cv75_W <~~ V_cv75_W
			float x1429, x1430;
			x1429 = 1;
			x1430 = decay_1;
			JCudaTensor x1431;
			x1431 = x1392;
			x618.update(x1431, x1429, x1430);

			// cv71_B <~~ V_cv71_B
			float x1432, x1433;
			x1432 = 1;
			x1433 = 1;
			JCudaTensor x1434;
			x1434 = x1405;
			x555.update(x1434, x1432, x1433);

			// cv71_W <~~ V_cv71_W
			float x1435, x1436;
			x1435 = 1;
			x1436 = decay_1;
			JCudaTensor x1437;
			x1437 = x1377;
			x554.update(x1437, x1435, x1436);

			// cv73_W <~~ V_cv73_W
			float x1438, x1439;
			x1438 = 1;
			x1439 = decay_1;
			JCudaTensor x1440;
			x1440 = x1414;
			x612.update(x1440, x1438, x1439);

			// cv76_W <~~ V_cv76_W
			float x1441, x1442;
			x1441 = 1;
			x1442 = decay_1;
			JCudaTensor x1443;
			x1443 = x1409;
			x576.update(x1443, x1441, x1442);

			// cv75_B <~~ V_cv75_B
			float x1444, x1445;
			x1444 = 1;
			x1445 = 1;
			JCudaTensor x1446;
			x1446 = x1388;
			x619.update(x1446, x1444, x1445);

			// cv73_B <~~ V_cv73_B
			float x1447, x1448;
			x1447 = 1;
			x1448 = 1;
			JCudaTensor x1449;
			x1449 = x1398;
			x613.update(x1449, x1447, x1448);

			// cv76_B <~~ V_cv76_B
			float x1450, x1451;
			x1450 = 1;
			x1451 = 1;
			JCudaTensor x1452;
			x1452 = x1424;
			x577.update(x1452, x1450, x1451);

			// val X2378 = X2376 * d_ReLU()(X7645)/d_X7644
			JCudaTensor x1453;
			JCudaTensor x1454, x1455;
			x1454 = x1382;
			x1455 = x564;
			x1453 = x223.backward(x1454, x1455);

			// Dealloc(X7645)
			JCudaTensor x1456;
			x1456 = x564;
			x1456.free();

			// val X2354 = X2352 * d_ReLU()(X7641)/d_X7640
			JCudaTensor x1457;
			JCudaTensor x1458, x1459;
			x1458 = x1385;
			x1459 = x578;
			x1457 = x213.backward(x1458, x1459);

			// Dealloc(X7641)
			JCudaTensor x1460;
			x1460 = x578;
			x1460.free();

			// V_cv74_W <~~ X2378 * d_Convolv(1,0)(X7637)/d_cv74_W
			float x1462, x1463;
			x1462 = lrn_rate_1;
			x1463 = momentum;
			JCudaTensor x1464, x1465;
			x1464 = x1453;
			x1465 = x502;
			x193.backward_filter(x1464, x1465, x1461, x1462, x1463);

			// val X2356 = (X2332 + X2354 * d_Convolv(1,0)(cv72_W)/d_X7637)
			JCudaTensor x1466;
			JCudaTensor x1467;
			x1467 = x1420;
			JCudaTensor x1468, x1469;
			x1468 = x1457;
			x1469 = x548;
			x1466 = x210.backward_data(x1468,x1469, x1467);

			// V_cv72_W <~~ X2354 * d_Convolv(1,0)(X7637)/d_cv72_W
			float x1471, x1472;
			x1471 = lrn_rate_1;
			x1472 = momentum;
			JCudaTensor x1473, x1474;
			x1473 = x1457;
			x1474 = x502;
			x210.backward_filter(x1473, x1474, x1470, x1471, x1472);

			// V_cv72_B <~~ X2354 * d_Convolv(1,0)()/d_cv72_B
			float x1476, x1477;
			x1476 = lrn_rate_2;
			x1477 = momentum;
			JCudaTensor x1478;
			x1478 = x1457;
			x210.backward_bias(x1478, x1475, x1476, x1477);

			// Dealloc(X2354)
			JCudaTensor x1479;
			x1479 = x1457;
			x1479.free();

			// V_cv74_B <~~ X2378 * d_Convolv(1,0)()/d_cv74_B
			float x1481, x1482;
			x1481 = lrn_rate_2;
			x1482 = momentum;
			JCudaTensor x1483;
			x1483 = x1453;
			x193.backward_bias(x1483, x1480, x1481, x1482);

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
			x533.update(x1492, x1490, x1491);

			// val X2380 = (X2356 + X2378 * d_Convolv(1,0)(cv74_W)/d_X7637)
			JCudaTensor x1493;
			JCudaTensor x1494;
			x1494 = x1466;
			JCudaTensor x1495, x1496;
			x1495 = x1453;
			x1496 = x532;
			x1493 = x193.backward_data(x1495,x1496, x1494);

			// Dealloc(X2378)
			JCudaTensor x1497;
			x1497 = x1453;
			x1497.free();

			// cv74_W <~~ V_cv74_W
			float x1498, x1499;
			x1498 = 1;
			x1499 = decay_1;
			JCudaTensor x1500;
			x1500 = x1461;
			x532.update(x1500, x1498, x1499);

			// val X2402 = (X2380 + X2399 * d_Pooling(3,1,1,true)(X7648,X7637)/d_X7637)
			JCudaTensor x1501;
			JCudaTensor x1502;
			x1502 = x1493;
			JCudaTensor x1503, x1504, x1505;
			x1503 = x1402;
			x1504 = x538;
			x1505 = x502;
			x1501 = x203.backward(x1503,x1504,x1505, x1502);

			// Dealloc(X2399)
			JCudaTensor x1506;
			x1506 = x1402;
			x1506.free();

			// Dealloc(X7648)
			JCudaTensor x1507;
			x1507 = x538;
			x1507.free();

			// Dealloc(X7637)
			JCudaTensor x1508;
			x1508 = x502;
			x1508.free();

			// val X2438 = (X2437 * loss2)
			JCudaTensor x1509;
			JCudaTensor x1510;
			float x1511;
			x1510 = x925;
			x1511 = loss2;
			x1509 = x1510.times_i(x1511);

			// val X2439 = (X2402 + X2438)
			JCudaTensor x1512;
			JCudaTensor x1513, x1514;
			x1513 = x1501;
			x1514 = x1509;
			x1512 = x1513.plus_i(x1514);

			// Dealloc(X2438)
			JCudaTensor x1515;
			x1515 = x1509;
			x1515.free();

			// val X2523 = Proj(X2439, X7625,X7629,X7633,X7636, 2)
			JCudaTensor x1516;
			JCudaTensor x1518;
			x1518 = x1512;
			JCudaTensor[] x1517 = x250.backward(x1518);
			x1516 = x1517[2];

			// val X2547 = Proj(X2439, X7625,X7629,X7633,X7636, 3)
			JCudaTensor x1519;
			x1519 = x1517[3];

			// val X2481 = Proj(X2439, X7625,X7629,X7633,X7636, 0)
			JCudaTensor x1520;
			x1520 = x1517[0];

			// val X2499 = Proj(X2439, X7625,X7629,X7633,X7636, 1)
			JCudaTensor x1521;
			x1521 = x1517[1];

			// Dealloc(X2439)
			JCudaTensor x1522;
			x1522 = x1512;
			x1522.free();

			// val X2504 = X2499 * d_ReLU()(X7629)/d_X7628
			JCudaTensor x1523;
			JCudaTensor x1524, x1525;
			x1524 = x1521;
			x1525 = x489;
			x1523 = x243.backward(x1524, x1525);

			// Dealloc(X7629)
			JCudaTensor x1526;
			x1526 = x489;
			x1526.free();

			// val X2551 = X2547 * d_ReLU()(X7636)/d_X7635
			JCudaTensor x1527;
			JCudaTensor x1528, x1529;
			x1528 = x1519;
			x1529 = x494;
			x1527 = x246.backward(x1528, x1529);

			// Dealloc(X7636)
			JCudaTensor x1530;
			x1530 = x494;
			x1530.free();

			// val X2484 = X2481 * d_ReLU()(X7625)/d_X7624
			JCudaTensor x1531;
			JCudaTensor x1532, x1533;
			x1532 = x1520;
			x1533 = x485;
			x1531 = x240.backward(x1532, x1533);

			// Dealloc(X7625)
			JCudaTensor x1534;
			x1534 = x485;
			x1534.free();

			// val X2528 = X2523 * d_ReLU()(X7633)/d_X7632
			JCudaTensor x1535;
			JCudaTensor x1536, x1537;
			x1536 = x1516;
			x1537 = x496;
			x1535 = x246.backward(x1536, x1537);

			// Dealloc(X7633)
			JCudaTensor x1538;
			x1538 = x496;
			x1538.free();

			// V_cv63_B <~~ X2504 * d_Convolv(1,1)()/d_cv63_B
			float x1540, x1541;
			x1540 = lrn_rate_2;
			x1541 = momentum;
			JCudaTensor x1542;
			x1542 = x1523;
			x237.backward_bias(x1542, x1539, x1540, x1541);

			// val X2505 = X2504 * d_Convolv(1,1)(cv63_W)/d_X7627
			JCudaTensor x1543;
			JCudaTensor x1544, x1545;
			x1544 = x1523;
			x1545 = x476;
			x1543 = x237.backward_data(x1544, x1545);

			// V_cv66_B <~~ X2551 * d_Convolv(1,0)()/d_cv66_B
			float x1547, x1548;
			x1547 = lrn_rate_2;
			x1548 = momentum;
			JCudaTensor x1549;
			x1549 = x1527;
			x220.backward_bias(x1549, x1546, x1547, x1548);

			// V_cv61_W <~~ X2484 * d_Convolv(1,0)(X7623)/d_cv61_W
			float x1551, x1552;
			x1551 = lrn_rate_1;
			x1552 = momentum;
			JCudaTensor x1553, x1554;
			x1553 = x1531;
			x1554 = x408;
			x200.backward_filter(x1553, x1554, x1550, x1551, x1552);

			// V_cv63_W <~~ X2504 * d_Convolv(1,1)(X7627)/d_cv63_W
			float x1556, x1557;
			x1556 = lrn_rate_1;
			x1557 = momentum;
			JCudaTensor x1558, x1559;
			x1558 = x1523;
			x1559 = x438;
			x237.backward_filter(x1558, x1559, x1555, x1556, x1557);

			// Dealloc(X2504)
			JCudaTensor x1560;
			x1560 = x1523;
			x1560.free();

			// V_cv65_W <~~ X2528 * d_Convolv(1,2)(X7631)/d_cv65_W
			float x1562, x1563;
			x1562 = lrn_rate_1;
			x1563 = momentum;
			JCudaTensor x1564, x1565;
			x1564 = x1535;
			x1565 = x452;
			x230.backward_filter(x1564, x1565, x1561, x1562, x1563);

			// V_cv61_B <~~ X2484 * d_Convolv(1,0)()/d_cv61_B
			float x1567, x1568;
			x1567 = lrn_rate_2;
			x1568 = momentum;
			JCudaTensor x1569;
			x1569 = x1531;
			x200.backward_bias(x1569, x1566, x1567, x1568);

			// V_cv65_B <~~ X2528 * d_Convolv(1,2)()/d_cv65_B
			float x1571, x1572;
			x1571 = lrn_rate_2;
			x1572 = momentum;
			JCudaTensor x1573;
			x1573 = x1535;
			x230.backward_bias(x1573, x1570, x1571, x1572);

			// val X2485 = X2484 * d_Convolv(1,0)(cv61_W)/d_X7623
			JCudaTensor x1574;
			JCudaTensor x1575, x1576;
			x1575 = x1531;
			x1576 = x428;
			x1574 = x200.backward_data(x1575, x1576);

			// Dealloc(X2484)
			JCudaTensor x1577;
			x1577 = x1531;
			x1577.free();

			// val X2552 = X2551 * d_Convolv(1,0)(cv66_W)/d_X7634
			JCudaTensor x1578;
			JCudaTensor x1579, x1580;
			x1579 = x1527;
			x1580 = x444;
			x1578 = x220.backward_data(x1579, x1580);

			// V_cv66_W <~~ X2551 * d_Convolv(1,0)(X7634)/d_cv66_W
			float x1582, x1583;
			x1582 = lrn_rate_1;
			x1583 = momentum;
			JCudaTensor x1584, x1585;
			x1584 = x1527;
			x1585 = x416;
			x220.backward_filter(x1584, x1585, x1581, x1582, x1583);

			// Dealloc(X2551)
			JCudaTensor x1586;
			x1586 = x1527;
			x1586.free();

			// val X2529 = X2528 * d_Convolv(1,2)(cv65_W)/d_X7631
			JCudaTensor x1587;
			JCudaTensor x1588, x1589;
			x1588 = x1535;
			x1589 = x467;
			x1587 = x230.backward_data(x1588, x1589);

			// Dealloc(X2528)
			JCudaTensor x1590;
			x1590 = x1535;
			x1590.free();

			// cv66_W <~~ V_cv66_W
			float x1591, x1592;
			x1591 = 1;
			x1592 = decay_1;
			JCudaTensor x1593;
			x1593 = x1581;
			x444.update(x1593, x1591, x1592);

			// cv65_W <~~ V_cv65_W
			float x1594, x1595;
			x1594 = 1;
			x1595 = decay_1;
			JCudaTensor x1596;
			x1596 = x1561;
			x467.update(x1596, x1594, x1595);

			// cv61_B <~~ V_cv61_B
			float x1597, x1598;
			x1597 = 1;
			x1598 = 1;
			JCudaTensor x1599;
			x1599 = x1566;
			x429.update(x1599, x1597, x1598);

			// cv63_B <~~ V_cv63_B
			float x1600, x1601;
			x1600 = 1;
			x1601 = 1;
			JCudaTensor x1602;
			x1602 = x1539;
			x477.update(x1602, x1600, x1601);

			// cv61_W <~~ V_cv61_W
			float x1603, x1604;
			x1603 = 1;
			x1604 = decay_1;
			JCudaTensor x1605;
			x1605 = x1550;
			x428.update(x1605, x1603, x1604);

			// cv63_W <~~ V_cv63_W
			float x1606, x1607;
			x1606 = 1;
			x1607 = decay_1;
			JCudaTensor x1608;
			x1608 = x1555;
			x476.update(x1608, x1606, x1607);

			// cv65_B <~~ V_cv65_B
			float x1609, x1610;
			x1609 = 1;
			x1610 = 1;
			JCudaTensor x1611;
			x1611 = x1570;
			x468.update(x1611, x1609, x1610);

			// cv66_B <~~ V_cv66_B
			float x1612, x1613;
			x1612 = 1;
			x1613 = 1;
			JCudaTensor x1614;
			x1614 = x1546;
			x445.update(x1614, x1612, x1613);

			// val X2507 = X2505 * d_ReLU()(X7627)/d_X7626
			JCudaTensor x1615;
			JCudaTensor x1616, x1617;
			x1616 = x1543;
			x1617 = x438;
			x1615 = x213.backward(x1616, x1617);

			// Dealloc(X7627)
			JCudaTensor x1618;
			x1618 = x438;
			x1618.free();

			// val X2531 = X2529 * d_ReLU()(X7631)/d_X7630
			JCudaTensor x1619;
			JCudaTensor x1620, x1621;
			x1620 = x1587;
			x1621 = x452;
			x1619 = x223.backward(x1620, x1621);

			// Dealloc(X7631)
			JCudaTensor x1622;
			x1622 = x452;
			x1622.free();

			// V_cv64_W <~~ X2531 * d_Convolv(1,0)(X7623)/d_cv64_W
			float x1624, x1625;
			x1624 = lrn_rate_1;
			x1625 = momentum;
			JCudaTensor x1626, x1627;
			x1626 = x1619;
			x1627 = x408;
			x193.backward_filter(x1626, x1627, x1623, x1624, x1625);

			// val X2509 = (X2485 + X2507 * d_Convolv(1,0)(cv62_W)/d_X7623)
			JCudaTensor x1628;
			JCudaTensor x1629;
			x1629 = x1574;
			JCudaTensor x1630, x1631;
			x1630 = x1615;
			x1631 = x422;
			x1628 = x210.backward_data(x1630,x1631, x1629);

			// V_cv62_W <~~ X2507 * d_Convolv(1,0)(X7623)/d_cv62_W
			float x1633, x1634;
			x1633 = lrn_rate_1;
			x1634 = momentum;
			JCudaTensor x1635, x1636;
			x1635 = x1615;
			x1636 = x408;
			x210.backward_filter(x1635, x1636, x1632, x1633, x1634);

			// V_cv62_B <~~ X2507 * d_Convolv(1,0)()/d_cv62_B
			float x1638, x1639;
			x1638 = lrn_rate_2;
			x1639 = momentum;
			JCudaTensor x1640;
			x1640 = x1615;
			x210.backward_bias(x1640, x1637, x1638, x1639);

			// Dealloc(X2507)
			JCudaTensor x1641;
			x1641 = x1615;
			x1641.free();

			// V_cv64_B <~~ X2531 * d_Convolv(1,0)()/d_cv64_B
			float x1643, x1644;
			x1643 = lrn_rate_2;
			x1644 = momentum;
			JCudaTensor x1645;
			x1645 = x1619;
			x193.backward_bias(x1645, x1642, x1643, x1644);

			// cv62_W <~~ V_cv62_W
			float x1646, x1647;
			x1646 = 1;
			x1647 = decay_1;
			JCudaTensor x1648;
			x1648 = x1632;
			x422.update(x1648, x1646, x1647);

			// cv62_B <~~ V_cv62_B
			float x1649, x1650;
			x1649 = 1;
			x1650 = 1;
			JCudaTensor x1651;
			x1651 = x1637;
			x423.update(x1651, x1649, x1650);

			// cv64_B <~~ V_cv64_B
			float x1652, x1653;
			x1652 = 1;
			x1653 = 1;
			JCudaTensor x1654;
			x1654 = x1642;
			x437.update(x1654, x1652, x1653);

			// val X2533 = (X2509 + X2531 * d_Convolv(1,0)(cv64_W)/d_X7623)
			JCudaTensor x1655;
			JCudaTensor x1656;
			x1656 = x1628;
			JCudaTensor x1657, x1658;
			x1657 = x1619;
			x1658 = x436;
			x1655 = x193.backward_data(x1657,x1658, x1656);

			// Dealloc(X2531)
			JCudaTensor x1659;
			x1659 = x1619;
			x1659.free();

			// cv64_W <~~ V_cv64_W
			float x1660, x1661;
			x1660 = 1;
			x1661 = decay_1;
			JCudaTensor x1662;
			x1662 = x1623;
			x436.update(x1662, x1660, x1661);

			// val X2555 = (X2533 + X2552 * d_Pooling(3,1,1,true)(X7634,X7623)/d_X7623)
			JCudaTensor x1663;
			JCudaTensor x1664;
			x1664 = x1655;
			JCudaTensor x1665, x1666, x1667;
			x1665 = x1578;
			x1666 = x416;
			x1667 = x408;
			x1663 = x203.backward(x1665,x1666,x1667, x1664);

			// Dealloc(X2552)
			JCudaTensor x1668;
			x1668 = x1578;
			x1668.free();

			// Dealloc(X7634)
			JCudaTensor x1669;
			x1669 = x416;
			x1669.free();

			// Dealloc(X7623)
			JCudaTensor x1670;
			x1670 = x408;
			x1670.free();

			// val X2635 = Proj(X2555, X7611,X7615,X7619,X7622, 3)
			JCudaTensor x1671;
			JCudaTensor x1673;
			x1673 = x1663;
			JCudaTensor[] x1672 = x250.backward(x1673);
			x1671 = x1672[3];

			// val X2587 = Proj(X2555, X7611,X7615,X7619,X7622, 1)
			JCudaTensor x1674;
			x1674 = x1672[1];

			// val X2611 = Proj(X2555, X7611,X7615,X7619,X7622, 2)
			JCudaTensor x1675;
			x1675 = x1672[2];

			// val X2569 = Proj(X2555, X7611,X7615,X7619,X7622, 0)
			JCudaTensor x1676;
			x1676 = x1672[0];

			// Dealloc(X2555)
			JCudaTensor x1677;
			x1677 = x1663;
			x1677.free();

			// val X2639 = X2635 * d_ReLU()(X7622)/d_X7621
			JCudaTensor x1678;
			JCudaTensor x1679, x1680;
			x1679 = x1671;
			x1680 = x396;
			x1678 = x246.backward(x1679, x1680);

			// Dealloc(X7622)
			JCudaTensor x1681;
			x1681 = x396;
			x1681.free();

			// val X2592 = X2587 * d_ReLU()(X7615)/d_X7614
			JCudaTensor x1682;
			JCudaTensor x1683, x1684;
			x1683 = x1674;
			x1684 = x398;
			x1682 = x243.backward(x1683, x1684);

			// Dealloc(X7615)
			JCudaTensor x1685;
			x1685 = x398;
			x1685.free();

			// val X2616 = X2611 * d_ReLU()(X7619)/d_X7618
			JCudaTensor x1686;
			JCudaTensor x1687, x1688;
			x1687 = x1675;
			x1688 = x390;
			x1686 = x246.backward(x1687, x1688);

			// Dealloc(X7619)
			JCudaTensor x1689;
			x1689 = x390;
			x1689.free();

			// val X2572 = X2569 * d_ReLU()(X7611)/d_X7610
			JCudaTensor x1690;
			JCudaTensor x1691, x1692;
			x1691 = x1676;
			x1692 = x388;
			x1690 = x240.backward(x1691, x1692);

			// Dealloc(X7611)
			JCudaTensor x1693;
			x1693 = x388;
			x1693.free();

			// val X2640 = X2639 * d_Convolv(1,0)(cv56_W)/d_X7620
			JCudaTensor x1694;
			JCudaTensor x1695, x1696;
			x1695 = x1678;
			x1696 = x368;
			x1694 = x220.backward_data(x1695, x1696);

			// V_cv53_B <~~ X2592 * d_Convolv(1,1)()/d_cv53_B
			float x1698, x1699;
			x1698 = lrn_rate_2;
			x1699 = momentum;
			JCudaTensor x1700;
			x1700 = x1682;
			x237.backward_bias(x1700, x1697, x1698, x1699);

			// V_cv56_B <~~ X2639 * d_Convolv(1,0)()/d_cv56_B
			float x1702, x1703;
			x1702 = lrn_rate_2;
			x1703 = momentum;
			JCudaTensor x1704;
			x1704 = x1678;
			x220.backward_bias(x1704, x1701, x1702, x1703);

			// val X2617 = X2616 * d_Convolv(1,2)(cv55_W)/d_X7617
			JCudaTensor x1705;
			JCudaTensor x1706, x1707;
			x1706 = x1686;
			x1707 = x374;
			x1705 = x230.backward_data(x1706, x1707);

			// V_cv51_B <~~ X2572 * d_Convolv(1,0)()/d_cv51_B
			float x1709, x1710;
			x1709 = lrn_rate_2;
			x1710 = momentum;
			JCudaTensor x1711;
			x1711 = x1690;
			x200.backward_bias(x1711, x1708, x1709, x1710);

			// V_cv55_W <~~ X2616 * d_Convolv(1,2)(X7617)/d_cv55_W
			float x1713, x1714;
			x1713 = lrn_rate_1;
			x1714 = momentum;
			JCudaTensor x1715, x1716;
			x1715 = x1686;
			x1716 = x357;
			x230.backward_filter(x1715, x1716, x1712, x1713, x1714);

			// val X2573 = X2572 * d_Convolv(1,0)(cv51_W)/d_X7609
			JCudaTensor x1717;
			JCudaTensor x1718, x1719;
			x1718 = x1690;
			x1719 = x355;
			x1717 = x200.backward_data(x1718, x1719);

			// val X2593 = X2592 * d_Convolv(1,1)(cv53_W)/d_X7613
			JCudaTensor x1720;
			JCudaTensor x1721, x1722;
			x1721 = x1682;
			x1722 = x386;
			x1720 = x237.backward_data(x1721, x1722);

			// V_cv56_W <~~ X2639 * d_Convolv(1,0)(X7620)/d_cv56_W
			float x1724, x1725;
			x1724 = lrn_rate_1;
			x1725 = momentum;
			JCudaTensor x1726, x1727;
			x1726 = x1678;
			x1727 = x349;
			x220.backward_filter(x1726, x1727, x1723, x1724, x1725);

			// Dealloc(X2639)
			JCudaTensor x1728;
			x1728 = x1678;
			x1728.free();

			// V_cv53_W <~~ X2592 * d_Convolv(1,1)(X7613)/d_cv53_W
			float x1730, x1731;
			x1730 = lrn_rate_1;
			x1731 = momentum;
			JCudaTensor x1732, x1733;
			x1732 = x1682;
			x1733 = x362;
			x237.backward_filter(x1732, x1733, x1729, x1730, x1731);

			// Dealloc(X2592)
			JCudaTensor x1734;
			x1734 = x1682;
			x1734.free();

			// V_cv55_B <~~ X2616 * d_Convolv(1,2)()/d_cv55_B
			float x1736, x1737;
			x1736 = lrn_rate_2;
			x1737 = momentum;
			JCudaTensor x1738;
			x1738 = x1686;
			x230.backward_bias(x1738, x1735, x1736, x1737);

			// Dealloc(X2616)
			JCudaTensor x1739;
			x1739 = x1686;
			x1739.free();

			// V_cv51_W <~~ X2572 * d_Convolv(1,0)(X7609)/d_cv51_W
			float x1741, x1742;
			x1741 = lrn_rate_1;
			x1742 = momentum;
			JCudaTensor x1743, x1744;
			x1743 = x1690;
			x1744 = x325;
			x200.backward_filter(x1743, x1744, x1740, x1741, x1742);

			// Dealloc(X2572)
			JCudaTensor x1745;
			x1745 = x1690;
			x1745.free();

			// cv56_W <~~ V_cv56_W
			float x1746, x1747;
			x1746 = 1;
			x1747 = decay_1;
			JCudaTensor x1748;
			x1748 = x1723;
			x368.update(x1748, x1746, x1747);

			// cv51_W <~~ V_cv51_W
			float x1749, x1750;
			x1749 = 1;
			x1750 = decay_1;
			JCudaTensor x1751;
			x1751 = x1740;
			x355.update(x1751, x1749, x1750);

			// cv55_B <~~ V_cv55_B
			float x1752, x1753;
			x1752 = 1;
			x1753 = 1;
			JCudaTensor x1754;
			x1754 = x1735;
			x375.update(x1754, x1752, x1753);

			// cv53_B <~~ V_cv53_B
			float x1755, x1756;
			x1755 = 1;
			x1756 = 1;
			JCudaTensor x1757;
			x1757 = x1697;
			x387.update(x1757, x1755, x1756);

			// cv51_B <~~ V_cv51_B
			float x1758, x1759;
			x1758 = 1;
			x1759 = 1;
			JCudaTensor x1760;
			x1760 = x1708;
			x356.update(x1760, x1758, x1759);

			// cv53_W <~~ V_cv53_W
			float x1761, x1762;
			x1761 = 1;
			x1762 = decay_1;
			JCudaTensor x1763;
			x1763 = x1729;
			x386.update(x1763, x1761, x1762);

			// cv55_W <~~ V_cv55_W
			float x1764, x1765;
			x1764 = 1;
			x1765 = decay_1;
			JCudaTensor x1766;
			x1766 = x1712;
			x374.update(x1766, x1764, x1765);

			// cv56_B <~~ V_cv56_B
			float x1767, x1768;
			x1767 = 1;
			x1768 = 1;
			JCudaTensor x1769;
			x1769 = x1701;
			x369.update(x1769, x1767, x1768);

			// val X2595 = X2593 * d_ReLU()(X7613)/d_X7612
			JCudaTensor x1770;
			JCudaTensor x1771, x1772;
			x1771 = x1720;
			x1772 = x362;
			x1770 = x213.backward(x1771, x1772);

			// Dealloc(X7613)
			JCudaTensor x1773;
			x1773 = x362;
			x1773.free();

			// val X2619 = X2617 * d_ReLU()(X7617)/d_X7616
			JCudaTensor x1774;
			JCudaTensor x1775, x1776;
			x1775 = x1705;
			x1776 = x357;
			x1774 = x223.backward(x1775, x1776);

			// Dealloc(X7617)
			JCudaTensor x1777;
			x1777 = x357;
			x1777.free();

			// V_cv54_W <~~ X2619 * d_Convolv(1,0)(X7609)/d_cv54_W
			float x1779, x1780;
			x1779 = lrn_rate_1;
			x1780 = momentum;
			JCudaTensor x1781, x1782;
			x1781 = x1774;
			x1782 = x325;
			x193.backward_filter(x1781, x1782, x1778, x1779, x1780);

			// V_cv52_B <~~ X2595 * d_Convolv(1,0)()/d_cv52_B
			float x1784, x1785;
			x1784 = lrn_rate_2;
			x1785 = momentum;
			JCudaTensor x1786;
			x1786 = x1770;
			x210.backward_bias(x1786, x1783, x1784, x1785);

			// V_cv54_B <~~ X2619 * d_Convolv(1,0)()/d_cv54_B
			float x1788, x1789;
			x1788 = lrn_rate_2;
			x1789 = momentum;
			JCudaTensor x1790;
			x1790 = x1774;
			x193.backward_bias(x1790, x1787, x1788, x1789);

			// val X2597 = (X2573 + X2595 * d_Convolv(1,0)(cv52_W)/d_X7609)
			JCudaTensor x1791;
			JCudaTensor x1792;
			x1792 = x1717;
			JCudaTensor x1793, x1794;
			x1793 = x1770;
			x1794 = x344;
			x1791 = x210.backward_data(x1793,x1794, x1792);

			// V_cv52_W <~~ X2595 * d_Convolv(1,0)(X7609)/d_cv52_W
			float x1796, x1797;
			x1796 = lrn_rate_1;
			x1797 = momentum;
			JCudaTensor x1798, x1799;
			x1798 = x1770;
			x1799 = x325;
			x210.backward_filter(x1798, x1799, x1795, x1796, x1797);

			// Dealloc(X2595)
			JCudaTensor x1800;
			x1800 = x1770;
			x1800.free();

			// cv52_B <~~ V_cv52_B
			float x1801, x1802;
			x1801 = 1;
			x1802 = 1;
			JCudaTensor x1803;
			x1803 = x1783;
			x345.update(x1803, x1801, x1802);

			// cv54_B <~~ V_cv54_B
			float x1804, x1805;
			x1804 = 1;
			x1805 = 1;
			JCudaTensor x1806;
			x1806 = x1787;
			x339.update(x1806, x1804, x1805);

			// cv52_W <~~ V_cv52_W
			float x1807, x1808;
			x1807 = 1;
			x1808 = decay_1;
			JCudaTensor x1809;
			x1809 = x1795;
			x344.update(x1809, x1807, x1808);

			// val X2621 = (X2597 + X2619 * d_Convolv(1,0)(cv54_W)/d_X7609)
			JCudaTensor x1810;
			JCudaTensor x1811;
			x1811 = x1791;
			JCudaTensor x1812, x1813;
			x1812 = x1774;
			x1813 = x338;
			x1810 = x193.backward_data(x1812,x1813, x1811);

			// Dealloc(X2619)
			JCudaTensor x1814;
			x1814 = x1774;
			x1814.free();

			// cv54_W <~~ V_cv54_W
			float x1815, x1816;
			x1815 = 1;
			x1816 = decay_1;
			JCudaTensor x1817;
			x1817 = x1778;
			x338.update(x1817, x1815, x1816);

			// val X2643 = (X2621 + X2640 * d_Pooling(3,1,1,true)(X7620,X7609)/d_X7609)
			JCudaTensor x1818;
			JCudaTensor x1819;
			x1819 = x1810;
			JCudaTensor x1820, x1821, x1822;
			x1820 = x1694;
			x1821 = x349;
			x1822 = x325;
			x1818 = x203.backward(x1820,x1821,x1822, x1819);

			// Dealloc(X2640)
			JCudaTensor x1823;
			x1823 = x1694;
			x1823.free();

			// Dealloc(X7620)
			JCudaTensor x1824;
			x1824 = x349;
			x1824.free();

			// Dealloc(X7609)
			JCudaTensor x1825;
			x1825 = x325;
			x1825.free();

			// val X2675 = Proj(X2643, X7597,X7601,X7605,X7608, 1)
			JCudaTensor x1826;
			JCudaTensor x1828;
			x1828 = x1818;
			JCudaTensor[] x1827 = x250.backward(x1828);
			x1826 = x1827[1];

			// val X2699 = Proj(X2643, X7597,X7601,X7605,X7608, 2)
			JCudaTensor x1829;
			x1829 = x1827[2];

			// val X2723 = Proj(X2643, X7597,X7601,X7605,X7608, 3)
			JCudaTensor x1830;
			x1830 = x1827[3];

			// val X2657 = Proj(X2643, X7597,X7601,X7605,X7608, 0)
			JCudaTensor x1831;
			x1831 = x1827[0];

			// Dealloc(X2643)
			JCudaTensor x1832;
			x1832 = x1818;
			x1832.free();

			// val X2704 = X2699 * d_ReLU()(X7605)/d_X7604
			JCudaTensor x1833;
			JCudaTensor x1834, x1835;
			x1834 = x1829;
			x1835 = x310;
			x1833 = x246.backward(x1834, x1835);

			// Dealloc(X7605)
			JCudaTensor x1836;
			x1836 = x310;
			x1836.free();

			// val X2680 = X2675 * d_ReLU()(X7601)/d_X7600
			JCudaTensor x1837;
			JCudaTensor x1838, x1839;
			x1838 = x1826;
			x1839 = x312;
			x1837 = x243.backward(x1838, x1839);

			// Dealloc(X7601)
			JCudaTensor x1840;
			x1840 = x312;
			x1840.free();

			// val X2660 = X2657 * d_ReLU()(X7597)/d_X7596
			JCudaTensor x1841;
			JCudaTensor x1842, x1843;
			x1842 = x1831;
			x1843 = x323;
			x1841 = x240.backward(x1842, x1843);

			// Dealloc(X7597)
			JCudaTensor x1844;
			x1844 = x323;
			x1844.free();

			// val X2727 = X2723 * d_ReLU()(X7608)/d_X7607
			JCudaTensor x1845;
			JCudaTensor x1846, x1847;
			x1846 = x1830;
			x1847 = x314;
			x1845 = x246.backward(x1846, x1847);

			// Dealloc(X7608)
			JCudaTensor x1848;
			x1848 = x314;
			x1848.free();

			// V_cv45_B <~~ X2704 * d_Convolv(1,2)()/d_cv45_B
			float x1850, x1851;
			x1850 = lrn_rate_2;
			x1851 = momentum;
			JCudaTensor x1852;
			x1852 = x1833;
			x230.backward_bias(x1852, x1849, x1850, x1851);

			// V_cv43_W <~~ X2680 * d_Convolv(1,1)(X7599)/d_cv43_W
			float x1854, x1855;
			x1854 = lrn_rate_1;
			x1855 = momentum;
			JCudaTensor x1856, x1857;
			x1856 = x1837;
			x1857 = x293;
			x237.backward_filter(x1856, x1857, x1853, x1854, x1855);

			// V_cv41_B <~~ X2660 * d_Convolv(1,0)()/d_cv41_B
			float x1859, x1860;
			x1859 = lrn_rate_2;
			x1860 = momentum;
			JCudaTensor x1861;
			x1861 = x1841;
			x200.backward_bias(x1861, x1858, x1859, x1860);

			// V_cv46_B <~~ X2727 * d_Convolv(1,0)()/d_cv46_B
			float x1863, x1864;
			x1863 = lrn_rate_2;
			x1864 = momentum;
			JCudaTensor x1865;
			x1865 = x1845;
			x220.backward_bias(x1865, x1862, x1863, x1864);

			// val X2681 = X2680 * d_Convolv(1,1)(cv43_W)/d_X7599
			JCudaTensor x1866;
			JCudaTensor x1867, x1868;
			x1867 = x1837;
			x1868 = x305;
			x1866 = x237.backward_data(x1867, x1868);

			// V_cv43_B <~~ X2680 * d_Convolv(1,1)()/d_cv43_B
			float x1870, x1871;
			x1870 = lrn_rate_2;
			x1871 = momentum;
			JCudaTensor x1872;
			x1872 = x1837;
			x237.backward_bias(x1872, x1869, x1870, x1871);

			// Dealloc(X2680)
			JCudaTensor x1873;
			x1873 = x1837;
			x1873.free();

			// V_cv46_W <~~ X2727 * d_Convolv(1,0)(X7606)/d_cv46_W
			float x1875, x1876;
			x1875 = lrn_rate_1;
			x1876 = momentum;
			JCudaTensor x1877, x1878;
			x1877 = x1845;
			x1878 = x264;
			x220.backward_filter(x1877, x1878, x1874, x1875, x1876);

			// val X2728 = X2727 * d_Convolv(1,0)(cv46_W)/d_X7606
			JCudaTensor x1879;
			JCudaTensor x1880, x1881;
			x1880 = x1845;
			x1881 = x284;
			x1879 = x220.backward_data(x1880, x1881);

			// Dealloc(X2727)
			JCudaTensor x1882;
			x1882 = x1845;
			x1882.free();

			// val X2661 = X2660 * d_Convolv(1,0)(cv41_W)/d_X7595
			JCudaTensor x1883;
			JCudaTensor x1884, x1885;
			x1884 = x1841;
			x1885 = x270;
			x1883 = x200.backward_data(x1884, x1885);

			// V_cv45_W <~~ X2704 * d_Convolv(1,2)(X7603)/d_cv45_W
			float x1887, x1888;
			x1887 = lrn_rate_1;
			x1888 = momentum;
			JCudaTensor x1889, x1890;
			x1889 = x1833;
			x1890 = x278;
			x230.backward_filter(x1889, x1890, x1886, x1887, x1888);

			// val X2705 = X2704 * d_Convolv(1,2)(cv45_W)/d_X7603
			JCudaTensor x1891;
			JCudaTensor x1892, x1893;
			x1892 = x1833;
			x1893 = x299;
			x1891 = x230.backward_data(x1892, x1893);

			// Dealloc(X2704)
			JCudaTensor x1894;
			x1894 = x1833;
			x1894.free();

			// V_cv41_W <~~ X2660 * d_Convolv(1,0)(X7595)/d_cv41_W
			float x1896, x1897;
			x1896 = lrn_rate_1;
			x1897 = momentum;
			JCudaTensor x1898, x1899;
			x1898 = x1841;
			x1899 = x249;
			x200.backward_filter(x1898, x1899, x1895, x1896, x1897);

			// Dealloc(X2660)
			JCudaTensor x1900;
			x1900 = x1841;
			x1900.free();

			// cv46_W <~~ V_cv46_W
			float x1901, x1902;
			x1901 = 1;
			x1902 = decay_1;
			JCudaTensor x1903;
			x1903 = x1874;
			x284.update(x1903, x1901, x1902);

			// cv43_W <~~ V_cv43_W
			float x1904, x1905;
			x1904 = 1;
			x1905 = decay_1;
			JCudaTensor x1906;
			x1906 = x1853;
			x305.update(x1906, x1904, x1905);

			// cv43_B <~~ V_cv43_B
			float x1907, x1908;
			x1907 = 1;
			x1908 = 1;
			JCudaTensor x1909;
			x1909 = x1869;
			x306.update(x1909, x1907, x1908);

			// cv41_B <~~ V_cv41_B
			float x1910, x1911;
			x1910 = 1;
			x1911 = 1;
			JCudaTensor x1912;
			x1912 = x1858;
			x271.update(x1912, x1910, x1911);

			// cv41_W <~~ V_cv41_W
			float x1913, x1914;
			x1913 = 1;
			x1914 = decay_1;
			JCudaTensor x1915;
			x1915 = x1895;
			x270.update(x1915, x1913, x1914);

			// cv45_W <~~ V_cv45_W
			float x1916, x1917;
			x1916 = 1;
			x1917 = decay_1;
			JCudaTensor x1918;
			x1918 = x1886;
			x299.update(x1918, x1916, x1917);

			// cv46_B <~~ V_cv46_B
			float x1919, x1920;
			x1919 = 1;
			x1920 = 1;
			JCudaTensor x1921;
			x1921 = x1862;
			x285.update(x1921, x1919, x1920);

			// cv45_B <~~ V_cv45_B
			float x1922, x1923;
			x1922 = 1;
			x1923 = 1;
			JCudaTensor x1924;
			x1924 = x1849;
			x300.update(x1924, x1922, x1923);

			// val X2683 = X2681 * d_ReLU()(X7599)/d_X7598
			JCudaTensor x1925;
			JCudaTensor x1926, x1927;
			x1926 = x1866;
			x1927 = x293;
			x1925 = x213.backward(x1926, x1927);

			// Dealloc(X7599)
			JCudaTensor x1928;
			x1928 = x293;
			x1928.free();

			// val X2707 = X2705 * d_ReLU()(X7603)/d_X7602
			JCudaTensor x1929;
			JCudaTensor x1930, x1931;
			x1930 = x1891;
			x1931 = x278;
			x1929 = x223.backward(x1930, x1931);

			// Dealloc(X7603)
			JCudaTensor x1932;
			x1932 = x278;
			x1932.free();

			// V_cv44_W <~~ X2707 * d_Convolv(1,0)(X7595)/d_cv44_W
			float x1934, x1935;
			x1934 = lrn_rate_1;
			x1935 = momentum;
			JCudaTensor x1936, x1937;
			x1936 = x1929;
			x1937 = x249;
			x193.backward_filter(x1936, x1937, x1933, x1934, x1935);

			// val X2685 = (X2661 + X2683 * d_Convolv(1,0)(cv42_W)/d_X7595)
			JCudaTensor x1938;
			JCudaTensor x1939;
			x1939 = x1883;
			JCudaTensor x1940, x1941;
			x1940 = x1925;
			x1941 = x259;
			x1938 = x210.backward_data(x1940,x1941, x1939);

			// V_cv42_W <~~ X2683 * d_Convolv(1,0)(X7595)/d_cv42_W
			float x1943, x1944;
			x1943 = lrn_rate_1;
			x1944 = momentum;
			JCudaTensor x1945, x1946;
			x1945 = x1925;
			x1946 = x249;
			x210.backward_filter(x1945, x1946, x1942, x1943, x1944);

			// V_cv42_B <~~ X2683 * d_Convolv(1,0)()/d_cv42_B
			float x1948, x1949;
			x1948 = lrn_rate_2;
			x1949 = momentum;
			JCudaTensor x1950;
			x1950 = x1925;
			x210.backward_bias(x1950, x1947, x1948, x1949);

			// Dealloc(X2683)
			JCudaTensor x1951;
			x1951 = x1925;
			x1951.free();

			// V_cv44_B <~~ X2707 * d_Convolv(1,0)()/d_cv44_B
			float x1953, x1954;
			x1953 = lrn_rate_2;
			x1954 = momentum;
			JCudaTensor x1955;
			x1955 = x1929;
			x193.backward_bias(x1955, x1952, x1953, x1954);

			// cv42_W <~~ V_cv42_W
			float x1956, x1957;
			x1956 = 1;
			x1957 = decay_1;
			JCudaTensor x1958;
			x1958 = x1942;
			x259.update(x1958, x1956, x1957);

			// cv42_B <~~ V_cv42_B
			float x1959, x1960;
			x1959 = 1;
			x1960 = 1;
			JCudaTensor x1961;
			x1961 = x1947;
			x260.update(x1961, x1959, x1960);

			// cv44_B <~~ V_cv44_B
			float x1962, x1963;
			x1962 = 1;
			x1963 = 1;
			JCudaTensor x1964;
			x1964 = x1952;
			x277.update(x1964, x1962, x1963);

			// val X2709 = (X2685 + X2707 * d_Convolv(1,0)(cv44_W)/d_X7595)
			JCudaTensor x1965;
			JCudaTensor x1966;
			x1966 = x1938;
			JCudaTensor x1967, x1968;
			x1967 = x1929;
			x1968 = x276;
			x1965 = x193.backward_data(x1967,x1968, x1966);

			// Dealloc(X2707)
			JCudaTensor x1969;
			x1969 = x1929;
			x1969.free();

			// cv44_W <~~ V_cv44_W
			float x1970, x1971;
			x1970 = 1;
			x1971 = decay_1;
			JCudaTensor x1972;
			x1972 = x1933;
			x276.update(x1972, x1970, x1971);

			// val X2731 = (X2709 + X2728 * d_Pooling(3,1,1,true)(X7606,X7595)/d_X7595)
			JCudaTensor x1973;
			JCudaTensor x1974;
			x1974 = x1965;
			JCudaTensor x1975, x1976, x1977;
			x1975 = x1879;
			x1976 = x264;
			x1977 = x249;
			x1973 = x203.backward(x1975,x1976,x1977, x1974);

			// Dealloc(X2728)
			JCudaTensor x1978;
			x1978 = x1879;
			x1978.free();

			// Dealloc(X7606)
			JCudaTensor x1979;
			x1979 = x264;
			x1979.free();

			// Dealloc(X7595)
			JCudaTensor x1980;
			x1980 = x249;
			x1980.free();

			// val X2767 = (X2766 * loss1)
			JCudaTensor x1981;
			JCudaTensor x1982;
			float x1983;
			x1982 = x620;
			x1983 = loss1;
			x1981 = x1982.times_i(x1983);

			// val X2768 = (X2731 + X2767)
			JCudaTensor x1984;
			JCudaTensor x1985, x1986;
			x1985 = x1973;
			x1986 = x1981;
			x1984 = x1985.plus_i(x1986);

			// Dealloc(X2767)
			JCudaTensor x1987;
			x1987 = x1981;
			x1987.free();

			// val X4158 = Proj(X2768, X7583,X7587,X7591,X7594, 3)
			JCudaTensor x1988;
			JCudaTensor x1990;
			x1990 = x1984;
			JCudaTensor[] x1989 = x250.backward(x1990);
			x1988 = x1989[3];

			// val X4110 = Proj(X2768, X7583,X7587,X7591,X7594, 1)
			JCudaTensor x1991;
			x1991 = x1989[1];

			// val X4092 = Proj(X2768, X7583,X7587,X7591,X7594, 0)
			JCudaTensor x1992;
			x1992 = x1989[0];

			// val X4134 = Proj(X2768, X7583,X7587,X7591,X7594, 2)
			JCudaTensor x1993;
			x1993 = x1989[2];

			// Dealloc(X2768)
			JCudaTensor x1994;
			x1994 = x1984;
			x1994.free();

			// val X4095 = X4092 * d_ReLU()(X7583)/d_X7582
			JCudaTensor x1995;
			JCudaTensor x1996, x1997;
			x1996 = x1992;
			x1997 = x238;
			x1995 = x240.backward(x1996, x1997);

			// Dealloc(X7583)
			JCudaTensor x1998;
			x1998 = x238;
			x1998.free();

			// val X4162 = X4158 * d_ReLU()(X7594)/d_X7593
			JCudaTensor x1999;
			JCudaTensor x2000, x2001;
			x2000 = x1988;
			x2001 = x247;
			x1999 = x246.backward(x2000, x2001);

			// Dealloc(X7594)
			JCudaTensor x2002;
			x2002 = x247;
			x2002.free();

			// val X4115 = X4110 * d_ReLU()(X7587)/d_X7586
			JCudaTensor x2003;
			JCudaTensor x2004, x2005;
			x2004 = x1991;
			x2005 = x241;
			x2003 = x243.backward(x2004, x2005);

			// Dealloc(X7587)
			JCudaTensor x2006;
			x2006 = x241;
			x2006.free();

			// val X4139 = X4134 * d_ReLU()(X7591)/d_X7590
			JCudaTensor x2007;
			JCudaTensor x2008, x2009;
			x2008 = x1993;
			x2009 = x244;
			x2007 = x246.backward(x2008, x2009);

			// Dealloc(X7591)
			JCudaTensor x2010;
			x2010 = x244;
			x2010.free();

			// V_cv31_B <~~ X4095 * d_Convolv(1,0)()/d_cv31_B
			float x2012, x2013;
			x2012 = lrn_rate_2;
			x2013 = momentum;
			JCudaTensor x2014;
			x2014 = x1995;
			x200.backward_bias(x2014, x2011, x2012, x2013);

			// V_cv36_W <~~ X4162 * d_Convolv(1,0)(X7592)/d_cv36_W
			float x2016, x2017;
			x2016 = lrn_rate_1;
			x2017 = momentum;
			JCudaTensor x2018, x2019;
			x2018 = x1999;
			x2019 = x201;
			x220.backward_filter(x2018, x2019, x2015, x2016, x2017);

			// V_cv33_W <~~ X4115 * d_Convolv(1,1)(X7585)/d_cv33_W
			float x2021, x2022;
			x2021 = lrn_rate_1;
			x2022 = momentum;
			JCudaTensor x2023, x2024;
			x2023 = x2003;
			x2024 = x211;
			x237.backward_filter(x2023, x2024, x2020, x2021, x2022);

			// V_cv36_B <~~ X4162 * d_Convolv(1,0)()/d_cv36_B
			float x2026, x2027;
			x2026 = lrn_rate_2;
			x2027 = momentum;
			JCudaTensor x2028;
			x2028 = x1999;
			x220.backward_bias(x2028, x2025, x2026, x2027);

			// val X4140 = X4139 * d_Convolv(1,2)(cv35_W)/d_X7589
			JCudaTensor x2029;
			JCudaTensor x2030, x2031;
			x2030 = x2007;
			x2031 = x228;
			x2029 = x230.backward_data(x2030, x2031);

			// V_cv31_W <~~ X4095 * d_Convolv(1,0)(X7581)/d_cv31_W
			float x2033, x2034;
			x2033 = lrn_rate_1;
			x2034 = momentum;
			JCudaTensor x2035, x2036;
			x2035 = x1995;
			x2036 = x184;
			x200.backward_filter(x2035, x2036, x2032, x2033, x2034);

			// V_cv35_W <~~ X4139 * d_Convolv(1,2)(X7589)/d_cv35_W
			float x2038, x2039;
			x2038 = lrn_rate_1;
			x2039 = momentum;
			JCudaTensor x2040, x2041;
			x2040 = x2007;
			x2041 = x221;
			x230.backward_filter(x2040, x2041, x2037, x2038, x2039);

			// val X4163 = X4162 * d_Convolv(1,0)(cv36_W)/d_X7592
			JCudaTensor x2042;
			JCudaTensor x2043, x2044;
			x2043 = x1999;
			x2044 = x218;
			x2042 = x220.backward_data(x2043, x2044);

			// Dealloc(X4162)
			JCudaTensor x2045;
			x2045 = x1999;
			x2045.free();

			// V_cv35_B <~~ X4139 * d_Convolv(1,2)()/d_cv35_B
			float x2047, x2048;
			x2047 = lrn_rate_2;
			x2048 = momentum;
			JCudaTensor x2049;
			x2049 = x2007;
			x230.backward_bias(x2049, x2046, x2047, x2048);

			// Dealloc(X4139)
			JCudaTensor x2050;
			x2050 = x2007;
			x2050.free();

			// val X4116 = X4115 * d_Convolv(1,1)(cv33_W)/d_X7585
			JCudaTensor x2051;
			JCudaTensor x2052, x2053;
			x2052 = x2003;
			x2053 = x235;
			x2051 = x237.backward_data(x2052, x2053);

			// val X4096 = X4095 * d_Convolv(1,0)(cv31_W)/d_X7581
			JCudaTensor x2054;
			JCudaTensor x2055, x2056;
			x2055 = x1995;
			x2056 = x198;
			x2054 = x200.backward_data(x2055, x2056);

			// Dealloc(X4095)
			JCudaTensor x2057;
			x2057 = x1995;
			x2057.free();

			// V_cv33_B <~~ X4115 * d_Convolv(1,1)()/d_cv33_B
			float x2059, x2060;
			x2059 = lrn_rate_2;
			x2060 = momentum;
			JCudaTensor x2061;
			x2061 = x2003;
			x237.backward_bias(x2061, x2058, x2059, x2060);

			// Dealloc(X4115)
			JCudaTensor x2062;
			x2062 = x2003;
			x2062.free();

			// cv33_B <~~ V_cv33_B
			float x2063, x2064;
			x2063 = 1;
			x2064 = 1;
			JCudaTensor x2065;
			x2065 = x2058;
			x236.update(x2065, x2063, x2064);

			// cv31_B <~~ V_cv31_B
			float x2066, x2067;
			x2066 = 1;
			x2067 = 1;
			JCudaTensor x2068;
			x2068 = x2011;
			x199.update(x2068, x2066, x2067);

			// cv35_W <~~ V_cv35_W
			float x2069, x2070;
			x2069 = 1;
			x2070 = decay_1;
			JCudaTensor x2071;
			x2071 = x2037;
			x228.update(x2071, x2069, x2070);

			// cv36_B <~~ V_cv36_B
			float x2072, x2073;
			x2072 = 1;
			x2073 = 1;
			JCudaTensor x2074;
			x2074 = x2025;
			x219.update(x2074, x2072, x2073);

			// cv33_W <~~ V_cv33_W
			float x2075, x2076;
			x2075 = 1;
			x2076 = decay_1;
			JCudaTensor x2077;
			x2077 = x2020;
			x235.update(x2077, x2075, x2076);

			// cv35_B <~~ V_cv35_B
			float x2078, x2079;
			x2078 = 1;
			x2079 = 1;
			JCudaTensor x2080;
			x2080 = x2046;
			x229.update(x2080, x2078, x2079);

			// cv31_W <~~ V_cv31_W
			float x2081, x2082;
			x2081 = 1;
			x2082 = decay_1;
			JCudaTensor x2083;
			x2083 = x2032;
			x198.update(x2083, x2081, x2082);

			// cv36_W <~~ V_cv36_W
			float x2084, x2085;
			x2084 = 1;
			x2085 = decay_1;
			JCudaTensor x2086;
			x2086 = x2015;
			x218.update(x2086, x2084, x2085);

			// val X4118 = X4116 * d_ReLU()(X7585)/d_X7584
			JCudaTensor x2087;
			JCudaTensor x2088, x2089;
			x2088 = x2051;
			x2089 = x211;
			x2087 = x213.backward(x2088, x2089);

			// Dealloc(X7585)
			JCudaTensor x2090;
			x2090 = x211;
			x2090.free();

			// val X4142 = X4140 * d_ReLU()(X7589)/d_X7588
			JCudaTensor x2091;
			JCudaTensor x2092, x2093;
			x2092 = x2029;
			x2093 = x221;
			x2091 = x223.backward(x2092, x2093);

			// Dealloc(X7589)
			JCudaTensor x2094;
			x2094 = x221;
			x2094.free();

			// V_cv34_W <~~ X4142 * d_Convolv(1,0)(X7581)/d_cv34_W
			float x2096, x2097;
			x2096 = lrn_rate_1;
			x2097 = momentum;
			JCudaTensor x2098, x2099;
			x2098 = x2091;
			x2099 = x184;
			x193.backward_filter(x2098, x2099, x2095, x2096, x2097);

			// val X4120 = (X4096 + X4118 * d_Convolv(1,0)(cv32_W)/d_X7581)
			JCudaTensor x2100;
			JCudaTensor x2101;
			x2101 = x2054;
			JCudaTensor x2102, x2103;
			x2102 = x2087;
			x2103 = x208;
			x2100 = x210.backward_data(x2102,x2103, x2101);

			// V_cv32_W <~~ X4118 * d_Convolv(1,0)(X7581)/d_cv32_W
			float x2105, x2106;
			x2105 = lrn_rate_1;
			x2106 = momentum;
			JCudaTensor x2107, x2108;
			x2107 = x2087;
			x2108 = x184;
			x210.backward_filter(x2107, x2108, x2104, x2105, x2106);

			// V_cv34_B <~~ X4142 * d_Convolv(1,0)()/d_cv34_B
			float x2110, x2111;
			x2110 = lrn_rate_2;
			x2111 = momentum;
			JCudaTensor x2112;
			x2112 = x2091;
			x193.backward_bias(x2112, x2109, x2110, x2111);

			// V_cv32_B <~~ X4118 * d_Convolv(1,0)()/d_cv32_B
			float x2114, x2115;
			x2114 = lrn_rate_2;
			x2115 = momentum;
			JCudaTensor x2116;
			x2116 = x2087;
			x210.backward_bias(x2116, x2113, x2114, x2115);

			// Dealloc(X4118)
			JCudaTensor x2117;
			x2117 = x2087;
			x2117.free();

			// cv32_W <~~ V_cv32_W
			float x2118, x2119;
			x2118 = 1;
			x2119 = decay_1;
			JCudaTensor x2120;
			x2120 = x2104;
			x208.update(x2120, x2118, x2119);

			// cv34_B <~~ V_cv34_B
			float x2121, x2122;
			x2121 = 1;
			x2122 = 1;
			JCudaTensor x2123;
			x2123 = x2109;
			x192.update(x2123, x2121, x2122);

			// cv32_B <~~ V_cv32_B
			float x2124, x2125;
			x2124 = 1;
			x2125 = 1;
			JCudaTensor x2126;
			x2126 = x2113;
			x209.update(x2126, x2124, x2125);

			// val X4144 = (X4120 + X4142 * d_Convolv(1,0)(cv34_W)/d_X7581)
			JCudaTensor x2127;
			JCudaTensor x2128;
			x2128 = x2100;
			JCudaTensor x2129, x2130;
			x2129 = x2091;
			x2130 = x191;
			x2127 = x193.backward_data(x2129,x2130, x2128);

			// Dealloc(X4142)
			JCudaTensor x2131;
			x2131 = x2091;
			x2131.free();

			// cv34_W <~~ V_cv34_W
			float x2132, x2133;
			x2132 = 1;
			x2133 = decay_1;
			JCudaTensor x2134;
			x2134 = x2095;
			x191.update(x2134, x2132, x2133);

			// val X4166 = (X4144 + X4163 * d_Pooling(3,1,1,true)(X7592,X7581)/d_X7581)
			JCudaTensor x2135;
			JCudaTensor x2136;
			x2136 = x2127;
			JCudaTensor x2137, x2138, x2139;
			x2137 = x2042;
			x2138 = x201;
			x2139 = x184;
			x2135 = x203.backward(x2137,x2138,x2139, x2136);

			// Dealloc(X4163)
			JCudaTensor x2140;
			x2140 = x2042;
			x2140.free();

			// Dealloc(X7592)
			JCudaTensor x2141;
			x2141 = x201;
			x2141.free();

			// val X4168 = X4166 * d_Pooling(3,2,1,true)(X7581,X7580)/d_X7580
			JCudaTensor x2142;
			JCudaTensor x2143, x2144, x2145;
			x2143 = x2135;
			x2144 = x184;
			x2145 = x179;
			x2142 = x186.backward(x2143, x2144, x2145);

			// Dealloc(X4166)
			JCudaTensor x2146;
			x2146 = x2135;
			x2146.free();

			// Dealloc(X7581)
			JCudaTensor x2147;
			x2147 = x184;
			x2147.free();

			// Dealloc(X7580)
			JCudaTensor x2148;
			x2148 = x179;
			x2148.free();

			// val X4182 = Proj(X4168, X7568,X7572,X7576,X7579, 0)
			JCudaTensor x2149;
			JCudaTensor x2151;
			x2151 = x2142;
			JCudaTensor[] x2150 = x119.backward(x2151);
			x2149 = x2150[0];

			// val X4200 = Proj(X4168, X7568,X7572,X7576,X7579, 1)
			JCudaTensor x2152;
			x2152 = x2150[1];

			// val X4224 = Proj(X4168, X7568,X7572,X7576,X7579, 2)
			JCudaTensor x2153;
			x2153 = x2150[2];

			// val X4248 = Proj(X4168, X7568,X7572,X7576,X7579, 3)
			JCudaTensor x2154;
			x2154 = x2150[3];

			// Dealloc(X4168)
			JCudaTensor x2155;
			x2155 = x2142;
			x2155.free();

			// val X4252 = X4248 * d_ReLU()(X7579)/d_X7578
			JCudaTensor x2156;
			JCudaTensor x2157, x2158;
			x2157 = x2154;
			x2158 = x177;
			x2156 = x115.backward(x2157, x2158);

			// Dealloc(X7579)
			JCudaTensor x2159;
			x2159 = x177;
			x2159.free();

			// val X4229 = X4224 * d_ReLU()(X7576)/d_X7575
			JCudaTensor x2160;
			JCudaTensor x2161, x2162;
			x2161 = x2153;
			x2162 = x175;
			x2160 = x115.backward(x2161, x2162);

			// Dealloc(X7576)
			JCudaTensor x2163;
			x2163 = x175;
			x2163.free();

			// val X4205 = X4200 * d_ReLU()(X7572)/d_X7571
			JCudaTensor x2164;
			JCudaTensor x2165, x2166;
			x2165 = x2152;
			x2166 = x173;
			x2164 = x112.backward(x2165, x2166);

			// Dealloc(X7572)
			JCudaTensor x2167;
			x2167 = x173;
			x2167.free();

			// val X4185 = X4182 * d_ReLU()(X7568)/d_X7567
			JCudaTensor x2168;
			JCudaTensor x2169, x2170;
			x2169 = x2149;
			x2170 = x171;
			x2168 = x109.backward(x2169, x2170);

			// Dealloc(X7568)
			JCudaTensor x2171;
			x2171 = x171;
			x2171.free();

			// val X4253 = X4252 * d_Convolv(1,0)(cv26_W)/d_X7577
			JCudaTensor x2172;
			JCudaTensor x2173, x2174;
			x2173 = x2156;
			x2174 = x152;
			x2172 = x154.backward_data(x2173, x2174);

			// V_cv25_W <~~ X4229 * d_Convolv(1,2)(X7574)/d_cv25_W
			float x2176, x2177;
			x2176 = lrn_rate_1;
			x2177 = momentum;
			JCudaTensor x2178, x2179;
			x2178 = x2160;
			x2179 = x157;
			x106.backward_filter(x2178, x2179, x2175, x2176, x2177);

			// V_cv23_B <~~ X4205 * d_Convolv(1,1)()/d_cv23_B
			float x2181, x2182;
			x2181 = lrn_rate_2;
			x2182 = momentum;
			JCudaTensor x2183;
			x2183 = x2164;
			x99.backward_bias(x2183, x2180, x2181, x2182);

			// V_cv21_W <~~ X4185 * d_Convolv(1,0)(X7566)/d_cv21_W
			float x2185, x2186;
			x2185 = lrn_rate_1;
			x2186 = momentum;
			JCudaTensor x2187, x2188;
			x2187 = x2168;
			x2188 = x118;
			x147.backward_filter(x2187, x2188, x2184, x2185, x2186);

			// V_cv25_B <~~ X4229 * d_Convolv(1,2)()/d_cv25_B
			float x2190, x2191;
			x2190 = lrn_rate_2;
			x2191 = momentum;
			JCudaTensor x2192;
			x2192 = x2160;
			x106.backward_bias(x2192, x2189, x2190, x2191);

			// V_cv23_W <~~ X4205 * d_Convolv(1,1)(X7570)/d_cv23_W
			float x2194, x2195;
			x2194 = lrn_rate_1;
			x2195 = momentum;
			JCudaTensor x2196, x2197;
			x2196 = x2164;
			x2197 = x155;
			x99.backward_filter(x2196, x2197, x2193, x2194, x2195);

			// V_cv26_B <~~ X4252 * d_Convolv(1,0)()/d_cv26_B
			float x2199, x2200;
			x2199 = lrn_rate_2;
			x2200 = momentum;
			JCudaTensor x2201;
			x2201 = x2156;
			x154.backward_bias(x2201, x2198, x2199, x2200);

			// val X4186 = X4185 * d_Convolv(1,0)(cv21_W)/d_X7566
			JCudaTensor x2202;
			JCudaTensor x2203, x2204;
			x2203 = x2168;
			x2204 = x145;
			x2202 = x147.backward_data(x2203, x2204);

			// val X4206 = X4205 * d_Convolv(1,1)(cv23_W)/d_X7570
			JCudaTensor x2205;
			JCudaTensor x2206, x2207;
			x2206 = x2164;
			x2207 = x169;
			x2205 = x99.backward_data(x2206, x2207);

			// Dealloc(X4205)
			JCudaTensor x2208;
			x2208 = x2164;
			x2208.free();

			// V_cv21_B <~~ X4185 * d_Convolv(1,0)()/d_cv21_B
			float x2210, x2211;
			x2210 = lrn_rate_2;
			x2211 = momentum;
			JCudaTensor x2212;
			x2212 = x2168;
			x147.backward_bias(x2212, x2209, x2210, x2211);

			// Dealloc(X4185)
			JCudaTensor x2213;
			x2213 = x2168;
			x2213.free();

			// V_cv26_W <~~ X4252 * d_Convolv(1,0)(X7577)/d_cv26_W
			float x2215, x2216;
			x2215 = lrn_rate_1;
			x2216 = momentum;
			JCudaTensor x2217, x2218;
			x2217 = x2156;
			x2218 = x124;
			x154.backward_filter(x2217, x2218, x2214, x2215, x2216);

			// Dealloc(X4252)
			JCudaTensor x2219;
			x2219 = x2156;
			x2219.free();

			// val X4230 = X4229 * d_Convolv(1,2)(cv25_W)/d_X7574
			JCudaTensor x2220;
			JCudaTensor x2221, x2222;
			x2221 = x2160;
			x2222 = x163;
			x2220 = x106.backward_data(x2221, x2222);

			// Dealloc(X4229)
			JCudaTensor x2223;
			x2223 = x2160;
			x2223.free();

			// cv25_W <~~ V_cv25_W
			float x2224, x2225;
			x2224 = 1;
			x2225 = decay_1;
			JCudaTensor x2226;
			x2226 = x2175;
			x163.update(x2226, x2224, x2225);

			// cv23_B <~~ V_cv23_B
			float x2227, x2228;
			x2227 = 1;
			x2228 = 1;
			JCudaTensor x2229;
			x2229 = x2180;
			x170.update(x2229, x2227, x2228);

			// cv25_B <~~ V_cv25_B
			float x2230, x2231;
			x2230 = 1;
			x2231 = 1;
			JCudaTensor x2232;
			x2232 = x2189;
			x164.update(x2232, x2230, x2231);

			// cv26_W <~~ V_cv26_W
			float x2233, x2234;
			x2233 = 1;
			x2234 = decay_1;
			JCudaTensor x2235;
			x2235 = x2214;
			x152.update(x2235, x2233, x2234);

			// cv23_W <~~ V_cv23_W
			float x2236, x2237;
			x2236 = 1;
			x2237 = decay_1;
			JCudaTensor x2238;
			x2238 = x2193;
			x169.update(x2238, x2236, x2237);

			// cv26_B <~~ V_cv26_B
			float x2239, x2240;
			x2239 = 1;
			x2240 = 1;
			JCudaTensor x2241;
			x2241 = x2198;
			x153.update(x2241, x2239, x2240);

			// cv21_B <~~ V_cv21_B
			float x2242, x2243;
			x2242 = 1;
			x2243 = 1;
			JCudaTensor x2244;
			x2244 = x2209;
			x146.update(x2244, x2242, x2243);

			// cv21_W <~~ V_cv21_W
			float x2245, x2246;
			x2245 = 1;
			x2246 = decay_1;
			JCudaTensor x2247;
			x2247 = x2184;
			x145.update(x2247, x2245, x2246);

			// val X4232 = X4230 * d_ReLU()(X7574)/d_X7573
			JCudaTensor x2248;
			JCudaTensor x2249, x2250;
			x2249 = x2220;
			x2250 = x157;
			x2248 = x92.backward(x2249, x2250);

			// Dealloc(X7574)
			JCudaTensor x2251;
			x2251 = x157;
			x2251.free();

			// val X4208 = X4206 * d_ReLU()(X7570)/d_X7569
			JCudaTensor x2252;
			JCudaTensor x2253, x2254;
			x2253 = x2205;
			x2254 = x155;
			x2252 = x89.backward(x2253, x2254);

			// Dealloc(X7570)
			JCudaTensor x2255;
			x2255 = x155;
			x2255.free();

			// V_cv24_W <~~ X4232 * d_Convolv(1,0)(X7566)/d_cv24_W
			float x2257, x2258;
			x2257 = lrn_rate_1;
			x2258 = momentum;
			JCudaTensor x2259, x2260;
			x2259 = x2248;
			x2260 = x118;
			x133.backward_filter(x2259, x2260, x2256, x2257, x2258);

			// V_cv24_B <~~ X4232 * d_Convolv(1,0)()/d_cv24_B
			float x2262, x2263;
			x2262 = lrn_rate_2;
			x2263 = momentum;
			JCudaTensor x2264;
			x2264 = x2248;
			x133.backward_bias(x2264, x2261, x2262, x2263);

			// V_cv22_B <~~ X4208 * d_Convolv(1,0)()/d_cv22_B
			float x2266, x2267;
			x2266 = lrn_rate_2;
			x2267 = momentum;
			JCudaTensor x2268;
			x2268 = x2252;
			x140.backward_bias(x2268, x2265, x2266, x2267);

			// val X4210 = (X4186 + X4208 * d_Convolv(1,0)(cv22_W)/d_X7566)
			JCudaTensor x2269;
			JCudaTensor x2270;
			x2270 = x2202;
			JCudaTensor x2271, x2272;
			x2271 = x2252;
			x2272 = x138;
			x2269 = x140.backward_data(x2271,x2272, x2270);

			// V_cv22_W <~~ X4208 * d_Convolv(1,0)(X7566)/d_cv22_W
			float x2274, x2275;
			x2274 = lrn_rate_1;
			x2275 = momentum;
			JCudaTensor x2276, x2277;
			x2276 = x2252;
			x2277 = x118;
			x140.backward_filter(x2276, x2277, x2273, x2274, x2275);

			// Dealloc(X4208)
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
			x139.update(x2284, x2282, x2283);

			// cv22_W <~~ V_cv22_W
			float x2285, x2286;
			x2285 = 1;
			x2286 = decay_1;
			JCudaTensor x2287;
			x2287 = x2273;
			x138.update(x2287, x2285, x2286);

			// val X4234 = (X4210 + X4232 * d_Convolv(1,0)(cv24_W)/d_X7566)
			JCudaTensor x2288;
			JCudaTensor x2289;
			x2289 = x2269;
			JCudaTensor x2290, x2291;
			x2290 = x2248;
			x2291 = x131;
			x2288 = x133.backward_data(x2290,x2291, x2289);

			// Dealloc(X4232)
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

			// val X4256 = (X4234 + X4253 * d_Pooling(3,1,1,true)(X7577,X7566)/d_X7566)
			JCudaTensor x2296;
			JCudaTensor x2297;
			x2297 = x2288;
			JCudaTensor x2298, x2299, x2300;
			x2298 = x2172;
			x2299 = x124;
			x2300 = x118;
			x2296 = x126.backward(x2298,x2299,x2300, x2297);

			// Dealloc(X4253)
			JCudaTensor x2301;
			x2301 = x2172;
			x2301.free();

			// Dealloc(X7577)
			JCudaTensor x2302;
			x2302 = x124;
			x2302.free();

			// Dealloc(X7566)
			JCudaTensor x2303;
			x2303 = x118;
			x2303.free();

			// val X4270 = Proj(X4256, X7554,X7558,X7562,X7565, 0)
			JCudaTensor x2304;
			JCudaTensor x2306;
			x2306 = x2296;
			JCudaTensor[] x2305 = x119.backward(x2306);
			x2304 = x2305[0];

			// val X4288 = Proj(X4256, X7554,X7558,X7562,X7565, 1)
			JCudaTensor x2307;
			x2307 = x2305[1];

			// val X4312 = Proj(X4256, X7554,X7558,X7562,X7565, 2)
			JCudaTensor x2308;
			x2308 = x2305[2];

			// val X4336 = Proj(X4256, X7554,X7558,X7562,X7565, 3)
			JCudaTensor x2309;
			x2309 = x2305[3];

			// Dealloc(X4256)
			JCudaTensor x2310;
			x2310 = x2296;
			x2310.free();

			// val X4317 = X4312 * d_ReLU()(X7562)/d_X7561
			JCudaTensor x2311;
			JCudaTensor x2312, x2313;
			x2312 = x2308;
			x2313 = x113;
			x2311 = x115.backward(x2312, x2313);

			// Dealloc(X7562)
			JCudaTensor x2314;
			x2314 = x113;
			x2314.free();

			// val X4273 = X4270 * d_ReLU()(X7554)/d_X7553
			JCudaTensor x2315;
			JCudaTensor x2316, x2317;
			x2316 = x2304;
			x2317 = x107;
			x2315 = x109.backward(x2316, x2317);

			// Dealloc(X7554)
			JCudaTensor x2318;
			x2318 = x107;
			x2318.free();

			// val X4293 = X4288 * d_ReLU()(X7558)/d_X7557
			JCudaTensor x2319;
			JCudaTensor x2320, x2321;
			x2320 = x2307;
			x2321 = x110;
			x2319 = x112.backward(x2320, x2321);

			// Dealloc(X7558)
			JCudaTensor x2322;
			x2322 = x110;
			x2322.free();

			// val X4340 = X4336 * d_ReLU()(X7565)/d_X7564
			JCudaTensor x2323;
			JCudaTensor x2324, x2325;
			x2324 = x2309;
			x2325 = x116;
			x2323 = x115.backward(x2324, x2325);

			// Dealloc(X7565)
			JCudaTensor x2326;
			x2326 = x116;
			x2326.free();

			// V_cv15_W <~~ X4317 * d_Convolv(1,2)(X7560)/d_cv15_W
			float x2328, x2329;
			x2328 = lrn_rate_1;
			x2329 = momentum;
			JCudaTensor x2330, x2331;
			x2330 = x2311;
			x2331 = x90;
			x106.backward_filter(x2330, x2331, x2327, x2328, x2329);

			// V_cv11_B <~~ X4273 * d_Convolv(1,0)()/d_cv11_B
			float x2333, x2334;
			x2333 = lrn_rate_2;
			x2334 = momentum;
			JCudaTensor x2335;
			x2335 = x2315;
			x69.backward_bias(x2335, x2332, x2333, x2334);

			// V_cv11_W <~~ X4273 * d_Convolv(1,0)(X7552)/d_cv11_W
			float x2337, x2338;
			x2337 = lrn_rate_1;
			x2338 = momentum;
			JCudaTensor x2339, x2340;
			x2339 = x2315;
			x2340 = x53;
			x69.backward_filter(x2339, x2340, x2336, x2337, x2338);

			// V_cv13_B <~~ X4293 * d_Convolv(1,1)()/d_cv13_B
			float x2342, x2343;
			x2342 = lrn_rate_2;
			x2343 = momentum;
			JCudaTensor x2344;
			x2344 = x2319;
			x99.backward_bias(x2344, x2341, x2342, x2343);

			// val X4341 = X4340 * d_Convolv(1,0)(cv16_W)/d_X7563
			JCudaTensor x2345;
			JCudaTensor x2346, x2347;
			x2346 = x2323;
			x2347 = x84;
			x2345 = x86.backward_data(x2346, x2347);

			// val X4274 = X4273 * d_Convolv(1,0)(cv11_W)/d_X7552
			JCudaTensor x2348;
			JCudaTensor x2349, x2350;
			x2349 = x2315;
			x2350 = x67;
			x2348 = x69.backward_data(x2349, x2350);

			// Dealloc(X4273)
			JCudaTensor x2351;
			x2351 = x2315;
			x2351.free();

			// val X4318 = X4317 * d_Convolv(1,2)(cv15_W)/d_X7560
			JCudaTensor x2352;
			JCudaTensor x2353, x2354;
			x2353 = x2311;
			x2354 = x104;
			x2352 = x106.backward_data(x2353, x2354);

			// V_cv16_W <~~ X4340 * d_Convolv(1,0)(X7563)/d_cv16_W
			float x2356, x2357;
			x2356 = lrn_rate_1;
			x2357 = momentum;
			JCudaTensor x2358, x2359;
			x2358 = x2323;
			x2359 = x70;
			x86.backward_filter(x2358, x2359, x2355, x2356, x2357);

			// val X4294 = X4293 * d_Convolv(1,1)(cv13_W)/d_X7556
			JCudaTensor x2360;
			JCudaTensor x2361, x2362;
			x2361 = x2319;
			x2362 = x97;
			x2360 = x99.backward_data(x2361, x2362);

			// V_cv13_W <~~ X4293 * d_Convolv(1,1)(X7556)/d_cv13_W
			float x2364, x2365;
			x2364 = lrn_rate_1;
			x2365 = momentum;
			JCudaTensor x2366, x2367;
			x2366 = x2319;
			x2367 = x87;
			x99.backward_filter(x2366, x2367, x2363, x2364, x2365);

			// Dealloc(X4293)
			JCudaTensor x2368;
			x2368 = x2319;
			x2368.free();

			// V_cv15_B <~~ X4317 * d_Convolv(1,2)()/d_cv15_B
			float x2370, x2371;
			x2370 = lrn_rate_2;
			x2371 = momentum;
			JCudaTensor x2372;
			x2372 = x2311;
			x106.backward_bias(x2372, x2369, x2370, x2371);

			// Dealloc(X4317)
			JCudaTensor x2373;
			x2373 = x2311;
			x2373.free();

			// V_cv16_B <~~ X4340 * d_Convolv(1,0)()/d_cv16_B
			float x2375, x2376;
			x2375 = lrn_rate_2;
			x2376 = momentum;
			JCudaTensor x2377;
			x2377 = x2323;
			x86.backward_bias(x2377, x2374, x2375, x2376);

			// Dealloc(X4340)
			JCudaTensor x2378;
			x2378 = x2323;
			x2378.free();

			// cv15_B <~~ V_cv15_B
			float x2379, x2380;
			x2379 = 1;
			x2380 = 1;
			JCudaTensor x2381;
			x2381 = x2369;
			x105.update(x2381, x2379, x2380);

			// cv13_B <~~ V_cv13_B
			float x2382, x2383;
			x2382 = 1;
			x2383 = 1;
			JCudaTensor x2384;
			x2384 = x2341;
			x98.update(x2384, x2382, x2383);

			// cv11_W <~~ V_cv11_W
			float x2385, x2386;
			x2385 = 1;
			x2386 = decay_1;
			JCudaTensor x2387;
			x2387 = x2336;
			x67.update(x2387, x2385, x2386);

			// cv11_B <~~ V_cv11_B
			float x2388, x2389;
			x2388 = 1;
			x2389 = 1;
			JCudaTensor x2390;
			x2390 = x2332;
			x68.update(x2390, x2388, x2389);

			// cv13_W <~~ V_cv13_W
			float x2391, x2392;
			x2391 = 1;
			x2392 = decay_1;
			JCudaTensor x2393;
			x2393 = x2363;
			x97.update(x2393, x2391, x2392);

			// cv16_W <~~ V_cv16_W
			float x2394, x2395;
			x2394 = 1;
			x2395 = decay_1;
			JCudaTensor x2396;
			x2396 = x2355;
			x84.update(x2396, x2394, x2395);

			// cv16_B <~~ V_cv16_B
			float x2397, x2398;
			x2397 = 1;
			x2398 = 1;
			JCudaTensor x2399;
			x2399 = x2374;
			x85.update(x2399, x2397, x2398);

			// cv15_W <~~ V_cv15_W
			float x2400, x2401;
			x2400 = 1;
			x2401 = decay_1;
			JCudaTensor x2402;
			x2402 = x2327;
			x104.update(x2402, x2400, x2401);

			// val X4320 = X4318 * d_ReLU()(X7560)/d_X7559
			JCudaTensor x2403;
			JCudaTensor x2404, x2405;
			x2404 = x2352;
			x2405 = x90;
			x2403 = x92.backward(x2404, x2405);

			// Dealloc(X7560)
			JCudaTensor x2406;
			x2406 = x90;
			x2406.free();

			// val X4296 = X4294 * d_ReLU()(X7556)/d_X7555
			JCudaTensor x2407;
			JCudaTensor x2408, x2409;
			x2408 = x2360;
			x2409 = x87;
			x2407 = x89.backward(x2408, x2409);

			// Dealloc(X7556)
			JCudaTensor x2410;
			x2410 = x87;
			x2410.free();

			// V_cv14_W <~~ X4320 * d_Convolv(1,0)(X7552)/d_cv14_W
			float x2412, x2413;
			x2412 = lrn_rate_1;
			x2413 = momentum;
			JCudaTensor x2414, x2415;
			x2414 = x2403;
			x2415 = x53;
			x62.backward_filter(x2414, x2415, x2411, x2412, x2413);

			// V_cv12_B <~~ X4296 * d_Convolv(1,0)()/d_cv12_B
			float x2417, x2418;
			x2417 = lrn_rate_2;
			x2418 = momentum;
			JCudaTensor x2419;
			x2419 = x2407;
			x79.backward_bias(x2419, x2416, x2417, x2418);

			// val X4298 = (X4274 + X4296 * d_Convolv(1,0)(cv12_W)/d_X7552)
			JCudaTensor x2420;
			JCudaTensor x2421;
			x2421 = x2348;
			JCudaTensor x2422, x2423;
			x2422 = x2407;
			x2423 = x77;
			x2420 = x79.backward_data(x2422,x2423, x2421);

			// V_cv12_W <~~ X4296 * d_Convolv(1,0)(X7552)/d_cv12_W
			float x2425, x2426;
			x2425 = lrn_rate_1;
			x2426 = momentum;
			JCudaTensor x2427, x2428;
			x2427 = x2407;
			x2428 = x53;
			x79.backward_filter(x2427, x2428, x2424, x2425, x2426);

			// Dealloc(X4296)
			JCudaTensor x2429;
			x2429 = x2407;
			x2429.free();

			// V_cv14_B <~~ X4320 * d_Convolv(1,0)()/d_cv14_B
			float x2431, x2432;
			x2431 = lrn_rate_2;
			x2432 = momentum;
			JCudaTensor x2433;
			x2433 = x2403;
			x62.backward_bias(x2433, x2430, x2431, x2432);

			// cv12_B <~~ V_cv12_B
			float x2434, x2435;
			x2434 = 1;
			x2435 = 1;
			JCudaTensor x2436;
			x2436 = x2416;
			x78.update(x2436, x2434, x2435);

			// cv12_W <~~ V_cv12_W
			float x2437, x2438;
			x2437 = 1;
			x2438 = decay_1;
			JCudaTensor x2439;
			x2439 = x2424;
			x77.update(x2439, x2437, x2438);

			// cv14_B <~~ V_cv14_B
			float x2440, x2441;
			x2440 = 1;
			x2441 = 1;
			JCudaTensor x2442;
			x2442 = x2430;
			x61.update(x2442, x2440, x2441);

			// val X4322 = (X4298 + X4320 * d_Convolv(1,0)(cv14_W)/d_X7552)
			JCudaTensor x2443;
			JCudaTensor x2444;
			x2444 = x2420;
			JCudaTensor x2445, x2446;
			x2445 = x2403;
			x2446 = x60;
			x2443 = x62.backward_data(x2445,x2446, x2444);

			// Dealloc(X4320)
			JCudaTensor x2447;
			x2447 = x2403;
			x2447.free();

			// cv14_W <~~ V_cv14_W
			float x2448, x2449;
			x2448 = 1;
			x2449 = decay_1;
			JCudaTensor x2450;
			x2450 = x2411;
			x60.update(x2450, x2448, x2449);

			// val X4344 = (X4322 + X4341 * d_Pooling(3,1,1,true)(X7563,X7552)/d_X7552)
			JCudaTensor x2451;
			JCudaTensor x2452;
			x2452 = x2443;
			JCudaTensor x2453, x2454, x2455;
			x2453 = x2345;
			x2454 = x70;
			x2455 = x53;
			x2451 = x72.backward(x2453,x2454,x2455, x2452);

			// Dealloc(X4341)
			JCudaTensor x2456;
			x2456 = x2345;
			x2456.free();

			// Dealloc(X7563)
			JCudaTensor x2457;
			x2457 = x70;
			x2457.free();

			// val X4346 = X4344 * d_Pooling(3,2,1,true)(X7552,X7551)/d_X7551
			JCudaTensor x2458;
			JCudaTensor x2459, x2460, x2461;
			x2459 = x2451;
			x2460 = x53;
			x2461 = x50;
			x2458 = x55.backward(x2459, x2460, x2461);

			// Dealloc(X4344)
			JCudaTensor x2462;
			x2462 = x2451;
			x2462.free();

			// Dealloc(X7552)
			JCudaTensor x2463;
			x2463 = x53;
			x2463.free();

			// val X4348 = X4346 * d_LRN(5,1.0E-4,0.75)(X7551,X7550)/d_X7550
			JCudaTensor x2464;
			JCudaTensor x2465, x2466, x2467;
			x2465 = x2458;
			x2466 = x50;
			x2467 = x47;
			x2464 = x52.backward(x2465, x2466, x2467);

			// Dealloc(X7551)
			JCudaTensor x2468;
			x2468 = x50;
			x2468.free();

			// val X4350 = X4348 * d_ReLU()(X7550)/d_X7549
			JCudaTensor x2469;
			JCudaTensor x2470, x2471;
			x2470 = x2464;
			x2471 = x47;
			x2469 = x49.backward(x2470, x2471);

			// Dealloc(X7550)
			JCudaTensor x2472;
			x2472 = x47;
			x2472.free();

			// val X4351 = X4350 * d_Convolv(1,1)(cv3_W)/d_X7548
			JCudaTensor x2473;
			JCudaTensor x2474, x2475;
			x2474 = x2469;
			x2475 = x44;
			x2473 = x46.backward_data(x2474, x2475);

			// V_cv3_B <~~ X4350 * d_Convolv(1,1)()/d_cv3_B
			float x2477, x2478;
			x2477 = lrn_rate_2;
			x2478 = momentum;
			JCudaTensor x2479;
			x2479 = x2469;
			x46.backward_bias(x2479, x2476, x2477, x2478);

			// V_cv3_W <~~ X4350 * d_Convolv(1,1)(X7548)/d_cv3_W
			float x2481, x2482;
			x2481 = lrn_rate_1;
			x2482 = momentum;
			JCudaTensor x2483, x2484;
			x2483 = x2469;
			x2484 = x37;
			x46.backward_filter(x2483, x2484, x2480, x2481, x2482);

			// Dealloc(X4350)
			JCudaTensor x2485;
			x2485 = x2469;
			x2485.free();

			// cv3_B <~~ V_cv3_B
			float x2486, x2487;
			x2486 = 1;
			x2487 = 1;
			JCudaTensor x2488;
			x2488 = x2476;
			x45.update(x2488, x2486, x2487);

			// cv3_W <~~ V_cv3_W
			float x2489, x2490;
			x2489 = 1;
			x2490 = decay_1;
			JCudaTensor x2491;
			x2491 = x2480;
			x44.update(x2491, x2489, x2490);

			// val X4353 = X4351 * d_ReLU()(X7548)/d_X7547
			JCudaTensor x2492;
			JCudaTensor x2493, x2494;
			x2493 = x2473;
			x2494 = x37;
			x2492 = x39.backward(x2493, x2494);

			// Dealloc(X7548)
			JCudaTensor x2495;
			x2495 = x37;
			x2495.free();

			// val X4354 = X4353 * d_Convolv(1,0)(cv2_W)/d_X7546
			JCudaTensor x2496;
			JCudaTensor x2497, x2498;
			x2497 = x2492;
			x2498 = x34;
			x2496 = x36.backward_data(x2497, x2498);

			// V_cv2_B <~~ X4353 * d_Convolv(1,0)()/d_cv2_B
			float x2500, x2501;
			x2500 = lrn_rate_2;
			x2501 = momentum;
			JCudaTensor x2502;
			x2502 = x2492;
			x36.backward_bias(x2502, x2499, x2500, x2501);

			// V_cv2_W <~~ X4353 * d_Convolv(1,0)(X7546)/d_cv2_W
			float x2504, x2505;
			x2504 = lrn_rate_1;
			x2505 = momentum;
			JCudaTensor x2506, x2507;
			x2506 = x2492;
			x2507 = x27;
			x36.backward_filter(x2506, x2507, x2503, x2504, x2505);

			// Dealloc(X4353)
			JCudaTensor x2508;
			x2508 = x2492;
			x2508.free();

			// cv2_B <~~ V_cv2_B
			float x2509, x2510;
			x2509 = 1;
			x2510 = 1;
			JCudaTensor x2511;
			x2511 = x2499;
			x35.update(x2511, x2509, x2510);

			// cv2_W <~~ V_cv2_W
			float x2512, x2513;
			x2512 = 1;
			x2513 = decay_1;
			JCudaTensor x2514;
			x2514 = x2503;
			x34.update(x2514, x2512, x2513);

			// val X4356 = X4354 * d_LRN(5,1.0E-4,0.75)(X7546,X7545)/d_X7545
			JCudaTensor x2515;
			JCudaTensor x2516, x2517, x2518;
			x2516 = x2496;
			x2517 = x27;
			x2518 = x24;
			x2515 = x29.backward(x2516, x2517, x2518);

			// Dealloc(X7546)
			JCudaTensor x2519;
			x2519 = x27;
			x2519.free();

			// val X4358 = X4356 * d_Pooling(3,2,1,true)(X7545,X7544)/d_X7544
			JCudaTensor x2520;
			JCudaTensor x2521, x2522, x2523;
			x2521 = x2515;
			x2522 = x24;
			x2523 = x21;
			x2520 = x26.backward(x2521, x2522, x2523);

			// Dealloc(X4356)
			JCudaTensor x2524;
			x2524 = x2515;
			x2524.free();

			// Dealloc(X7545)
			JCudaTensor x2525;
			x2525 = x24;
			x2525.free();

			// val X4360 = X4358 * d_ReLU()(X7544)/d_X7543
			JCudaTensor x2526;
			JCudaTensor x2527, x2528;
			x2527 = x2520;
			x2528 = x21;
			x2526 = x23.backward(x2527, x2528);

			// Dealloc(X7544)
			JCudaTensor x2529;
			x2529 = x21;
			x2529.free();

			// V_cv1_W <~~ X4360 * d_Convolv(2,3)(X7542)/d_cv1_W
			float x2531, x2532;
			x2531 = lrn_rate_1;
			x2532 = momentum;
			JCudaTensor x2533, x2534;
			x2533 = x2526;
			x2534 = x7;
			x17.backward_filter(x2533, x2534, x2530, x2531, x2532);

			// Dealloc(X7542)
			JCudaTensor x2535;
			x2535 = x7;
			x2535.free();

			// V_cv1_B <~~ X4360 * d_Convolv(2,3)()/d_cv1_B
			float x2537, x2538;
			x2537 = lrn_rate_2;
			x2538 = momentum;
			JCudaTensor x2539;
			x2539 = x2526;
			x17.backward_bias(x2539, x2536, x2537, x2538);

			// Dealloc(X4360)
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

			// val X7714 = Cuda(X)
			JCudaTensor x2547;
			JTensorFloat x2548;
			x2548 = x3;
			x2547 = x2548.asJCudaTensor();

			// val X7715 = Convolv(2,3)(X7714,cv1_W,cv1_B)
			JCudaTensor x2549;
			JCudaTensor x2550, x2551, x2552;
			x2550 = x2547;
			x2551 = x15;
			x2552 = x16;
			x2549 = x17.forward(x2550, x2551, x2552);

			// Dealloc(X7714)
			JCudaTensor x2553;
			x2553 = x2547;
			x2553.free();

			// val X7716 = ReLU()(X7715)
			JCudaTensor x2554;
			JCudaTensor x2555;
			x2555 = x2549;
			x2554 = x23.forward(x2555);

			// val X7717 = Pooling(3,2,1,true)(X7716)
			JCudaTensor x2556;
			JCudaTensor x2557;
			x2557 = x2554;
			x2556 = x26.forward(x2557);

			// Dealloc(X7716)
			JCudaTensor x2558;
			x2558 = x2554;
			x2558.free();

			// val X7718 = LRN(5,1.0E-4,0.75)(X7717)
			JCudaTensor x2559;
			JCudaTensor x2560;
			x2560 = x2556;
			x2559 = x29.forward(x2560);

			// Dealloc(X7717)
			JCudaTensor x2561;
			x2561 = x2556;
			x2561.free();

			// val X7719 = Convolv(1,0)(X7718,cv2_W,cv2_B)
			JCudaTensor x2562;
			JCudaTensor x2563, x2564, x2565;
			x2563 = x2559;
			x2564 = x34;
			x2565 = x35;
			x2562 = x36.forward(x2563, x2564, x2565);

			// Dealloc(X7718)
			JCudaTensor x2566;
			x2566 = x2559;
			x2566.free();

			// val X7720 = ReLU()(X7719)
			JCudaTensor x2567;
			JCudaTensor x2568;
			x2568 = x2562;
			x2567 = x39.forward(x2568);

			// val X7721 = Convolv(1,1)(X7720,cv3_W,cv3_B)
			JCudaTensor x2569;
			JCudaTensor x2570, x2571, x2572;
			x2570 = x2567;
			x2571 = x44;
			x2572 = x45;
			x2569 = x46.forward(x2570, x2571, x2572);

			// Dealloc(X7720)
			JCudaTensor x2573;
			x2573 = x2567;
			x2573.free();

			// val X7722 = ReLU()(X7721)
			JCudaTensor x2574;
			JCudaTensor x2575;
			x2575 = x2569;
			x2574 = x49.forward(x2575);

			// val X7723 = LRN(5,1.0E-4,0.75)(X7722)
			JCudaTensor x2576;
			JCudaTensor x2577;
			x2577 = x2574;
			x2576 = x52.forward(x2577);

			// Dealloc(X7722)
			JCudaTensor x2578;
			x2578 = x2574;
			x2578.free();

			// val X7724 = Pooling(3,2,1,true)(X7723)
			JCudaTensor x2579;
			JCudaTensor x2580;
			x2580 = x2576;
			x2579 = x55.forward(x2580);

			// Dealloc(X7723)
			JCudaTensor x2581;
			x2581 = x2576;
			x2581.free();

			// val X7735 = Pooling(3,1,1,true)(X7724)
			JCudaTensor x2582;
			JCudaTensor x2583;
			x2583 = x2579;
			x2582 = x72.forward(x2583);

			// val X7725 = Convolv(1,0)(X7724,cv11_W,cv11_B)
			JCudaTensor x2584;
			JCudaTensor x2585, x2586, x2587;
			x2585 = x2579;
			x2586 = x67;
			x2587 = x68;
			x2584 = x69.forward(x2585, x2586, x2587);

			// val X7727 = Convolv(1,0)(X7724,cv12_W,cv12_B)
			JCudaTensor x2588;
			JCudaTensor x2589, x2590, x2591;
			x2589 = x2579;
			x2590 = x77;
			x2591 = x78;
			x2588 = x79.forward(x2589, x2590, x2591);

			// val X7731 = Convolv(1,0)(X7724,cv14_W,cv14_B)
			JCudaTensor x2592;
			JCudaTensor x2593, x2594, x2595;
			x2593 = x2579;
			x2594 = x60;
			x2595 = x61;
			x2592 = x62.forward(x2593, x2594, x2595);

			// Dealloc(X7724)
			JCudaTensor x2596;
			x2596 = x2579;
			x2596.free();

			// val X7732 = ReLU()(X7731)
			JCudaTensor x2597;
			JCudaTensor x2598;
			x2598 = x2592;
			x2597 = x92.forward(x2598);

			// val X7736 = Convolv(1,0)(X7735,cv16_W,cv16_B)
			JCudaTensor x2599;
			JCudaTensor x2600, x2601, x2602;
			x2600 = x2582;
			x2601 = x84;
			x2602 = x85;
			x2599 = x86.forward(x2600, x2601, x2602);

			// Dealloc(X7735)
			JCudaTensor x2603;
			x2603 = x2582;
			x2603.free();

			// val X7728 = ReLU()(X7727)
			JCudaTensor x2604;
			JCudaTensor x2605;
			x2605 = x2588;
			x2604 = x89.forward(x2605);

			// val X7729 = Convolv(1,1)(X7728,cv13_W,cv13_B)
			JCudaTensor x2606;
			JCudaTensor x2607, x2608, x2609;
			x2607 = x2604;
			x2608 = x97;
			x2609 = x98;
			x2606 = x99.forward(x2607, x2608, x2609);

			// Dealloc(X7728)
			JCudaTensor x2610;
			x2610 = x2604;
			x2610.free();

			// val X7733 = Convolv(1,2)(X7732,cv15_W,cv15_B)
			JCudaTensor x2611;
			JCudaTensor x2612, x2613, x2614;
			x2612 = x2597;
			x2613 = x104;
			x2614 = x105;
			x2611 = x106.forward(x2612, x2613, x2614);

			// Dealloc(X7732)
			JCudaTensor x2615;
			x2615 = x2597;
			x2615.free();

			// val X7726 = ReLU()(X7725)
			JCudaTensor x2616;
			JCudaTensor x2617;
			x2617 = x2584;
			x2616 = x109.forward(x2617);

			// val X7730 = ReLU()(X7729)
			JCudaTensor x2618;
			JCudaTensor x2619;
			x2619 = x2606;
			x2618 = x112.forward(x2619);

			// val X7734 = ReLU()(X7733)
			JCudaTensor x2620;
			JCudaTensor x2621;
			x2621 = x2611;
			x2620 = x115.forward(x2621);

			// val X7737 = ReLU()(X7736)
			JCudaTensor x2622;
			JCudaTensor x2623;
			x2623 = x2599;
			x2622 = x115.forward(x2623);

			// val X7738 = Concat(X7726,X7730,X7734,X7737)
			JCudaTensor x2624;
			JCudaTensor x2625, x2626, x2627, x2628;
			x2625 = x2616;
			x2626 = x2618;
			x2627 = x2620;
			x2628 = x2622;
			x2624 = x119.forward(x2625,x2626,x2627,x2628);

			// Dealloc(X7737)
			JCudaTensor x2629;
			x2629 = x2622;
			x2629.free();

			// Dealloc(X7734)
			JCudaTensor x2630;
			x2630 = x2620;
			x2630.free();

			// Dealloc(X7730)
			JCudaTensor x2631;
			x2631 = x2618;
			x2631.free();

			// Dealloc(X7726)
			JCudaTensor x2632;
			x2632 = x2616;
			x2632.free();

			// val X7739 = Convolv(1,0)(X7738,cv21_W,cv21_B)
			JCudaTensor x2633;
			JCudaTensor x2634, x2635, x2636;
			x2634 = x2624;
			x2635 = x145;
			x2636 = x146;
			x2633 = x147.forward(x2634, x2635, x2636);

			// val X7745 = Convolv(1,0)(X7738,cv24_W,cv24_B)
			JCudaTensor x2637;
			JCudaTensor x2638, x2639, x2640;
			x2638 = x2624;
			x2639 = x131;
			x2640 = x132;
			x2637 = x133.forward(x2638, x2639, x2640);

			// val X7749 = Pooling(3,1,1,true)(X7738)
			JCudaTensor x2641;
			JCudaTensor x2642;
			x2642 = x2624;
			x2641 = x126.forward(x2642);

			// val X7741 = Convolv(1,0)(X7738,cv22_W,cv22_B)
			JCudaTensor x2643;
			JCudaTensor x2644, x2645, x2646;
			x2644 = x2624;
			x2645 = x138;
			x2646 = x139;
			x2643 = x140.forward(x2644, x2645, x2646);

			// Dealloc(X7738)
			JCudaTensor x2647;
			x2647 = x2624;
			x2647.free();

			// val X7742 = ReLU()(X7741)
			JCudaTensor x2648;
			JCudaTensor x2649;
			x2649 = x2643;
			x2648 = x89.forward(x2649);

			// val X7746 = ReLU()(X7745)
			JCudaTensor x2650;
			JCudaTensor x2651;
			x2651 = x2637;
			x2650 = x92.forward(x2651);

			// val X7750 = Convolv(1,0)(X7749,cv26_W,cv26_B)
			JCudaTensor x2652;
			JCudaTensor x2653, x2654, x2655;
			x2653 = x2641;
			x2654 = x152;
			x2655 = x153;
			x2652 = x154.forward(x2653, x2654, x2655);

			// Dealloc(X7749)
			JCudaTensor x2656;
			x2656 = x2641;
			x2656.free();

			// val X7743 = Convolv(1,1)(X7742,cv23_W,cv23_B)
			JCudaTensor x2657;
			JCudaTensor x2658, x2659, x2660;
			x2658 = x2648;
			x2659 = x169;
			x2660 = x170;
			x2657 = x99.forward(x2658, x2659, x2660);

			// Dealloc(X7742)
			JCudaTensor x2661;
			x2661 = x2648;
			x2661.free();

			// val X7747 = Convolv(1,2)(X7746,cv25_W,cv25_B)
			JCudaTensor x2662;
			JCudaTensor x2663, x2664, x2665;
			x2663 = x2650;
			x2664 = x163;
			x2665 = x164;
			x2662 = x106.forward(x2663, x2664, x2665);

			// Dealloc(X7746)
			JCudaTensor x2666;
			x2666 = x2650;
			x2666.free();

			// val X7740 = ReLU()(X7739)
			JCudaTensor x2667;
			JCudaTensor x2668;
			x2668 = x2633;
			x2667 = x109.forward(x2668);

			// val X7744 = ReLU()(X7743)
			JCudaTensor x2669;
			JCudaTensor x2670;
			x2670 = x2657;
			x2669 = x112.forward(x2670);

			// val X7748 = ReLU()(X7747)
			JCudaTensor x2671;
			JCudaTensor x2672;
			x2672 = x2662;
			x2671 = x115.forward(x2672);

			// val X7751 = ReLU()(X7750)
			JCudaTensor x2673;
			JCudaTensor x2674;
			x2674 = x2652;
			x2673 = x115.forward(x2674);

			// val X7752 = Concat(X7740,X7744,X7748,X7751)
			JCudaTensor x2675;
			JCudaTensor x2676, x2677, x2678, x2679;
			x2676 = x2667;
			x2677 = x2669;
			x2678 = x2671;
			x2679 = x2673;
			x2675 = x119.forward(x2676,x2677,x2678,x2679);

			// Dealloc(X7751)
			JCudaTensor x2680;
			x2680 = x2673;
			x2680.free();

			// Dealloc(X7748)
			JCudaTensor x2681;
			x2681 = x2671;
			x2681.free();

			// Dealloc(X7744)
			JCudaTensor x2682;
			x2682 = x2669;
			x2682.free();

			// Dealloc(X7740)
			JCudaTensor x2683;
			x2683 = x2667;
			x2683.free();

			// val X7753 = Pooling(3,2,1,true)(X7752)
			JCudaTensor x2684;
			JCudaTensor x2685;
			x2685 = x2675;
			x2684 = x186.forward(x2685);

			// Dealloc(X7752)
			JCudaTensor x2686;
			x2686 = x2675;
			x2686.free();

			// val X7764 = Pooling(3,1,1,true)(X7753)
			JCudaTensor x2687;
			JCudaTensor x2688;
			x2688 = x2684;
			x2687 = x203.forward(x2688);

			// val X7756 = Convolv(1,0)(X7753,cv32_W,cv32_B)
			JCudaTensor x2689;
			JCudaTensor x2690, x2691, x2692;
			x2690 = x2684;
			x2691 = x208;
			x2692 = x209;
			x2689 = x210.forward(x2690, x2691, x2692);

			// val X7760 = Convolv(1,0)(X7753,cv34_W,cv34_B)
			JCudaTensor x2693;
			JCudaTensor x2694, x2695, x2696;
			x2694 = x2684;
			x2695 = x191;
			x2696 = x192;
			x2693 = x193.forward(x2694, x2695, x2696);

			// val X7754 = Convolv(1,0)(X7753,cv31_W,cv31_B)
			JCudaTensor x2697;
			JCudaTensor x2698, x2699, x2700;
			x2698 = x2684;
			x2699 = x198;
			x2700 = x199;
			x2697 = x200.forward(x2698, x2699, x2700);

			// Dealloc(X7753)
			JCudaTensor x2701;
			x2701 = x2684;
			x2701.free();

			// val X7757 = ReLU()(X7756)
			JCudaTensor x2702;
			JCudaTensor x2703;
			x2703 = x2689;
			x2702 = x213.forward(x2703);

			// val X7765 = Convolv(1,0)(X7764,cv36_W,cv36_B)
			JCudaTensor x2704;
			JCudaTensor x2705, x2706, x2707;
			x2705 = x2687;
			x2706 = x218;
			x2707 = x219;
			x2704 = x220.forward(x2705, x2706, x2707);

			// Dealloc(X7764)
			JCudaTensor x2708;
			x2708 = x2687;
			x2708.free();

			// val X7761 = ReLU()(X7760)
			JCudaTensor x2709;
			JCudaTensor x2710;
			x2710 = x2693;
			x2709 = x223.forward(x2710);

			// val X7762 = Convolv(1,2)(X7761,cv35_W,cv35_B)
			JCudaTensor x2711;
			JCudaTensor x2712, x2713, x2714;
			x2712 = x2709;
			x2713 = x228;
			x2714 = x229;
			x2711 = x230.forward(x2712, x2713, x2714);

			// Dealloc(X7761)
			JCudaTensor x2715;
			x2715 = x2709;
			x2715.free();

			// val X7758 = Convolv(1,1)(X7757,cv33_W,cv33_B)
			JCudaTensor x2716;
			JCudaTensor x2717, x2718, x2719;
			x2717 = x2702;
			x2718 = x235;
			x2719 = x236;
			x2716 = x237.forward(x2717, x2718, x2719);

			// Dealloc(X7757)
			JCudaTensor x2720;
			x2720 = x2702;
			x2720.free();

			// val X7755 = ReLU()(X7754)
			JCudaTensor x2721;
			JCudaTensor x2722;
			x2722 = x2697;
			x2721 = x240.forward(x2722);

			// val X7759 = ReLU()(X7758)
			JCudaTensor x2723;
			JCudaTensor x2724;
			x2724 = x2716;
			x2723 = x243.forward(x2724);

			// val X7763 = ReLU()(X7762)
			JCudaTensor x2725;
			JCudaTensor x2726;
			x2726 = x2711;
			x2725 = x246.forward(x2726);

			// val X7766 = ReLU()(X7765)
			JCudaTensor x2727;
			JCudaTensor x2728;
			x2728 = x2704;
			x2727 = x246.forward(x2728);

			// val X7767 = Concat(X7755,X7759,X7763,X7766)
			JCudaTensor x2729;
			JCudaTensor x2730, x2731, x2732, x2733;
			x2730 = x2721;
			x2731 = x2723;
			x2732 = x2725;
			x2733 = x2727;
			x2729 = x250.forward(x2730,x2731,x2732,x2733);

			// Dealloc(X7766)
			JCudaTensor x2734;
			x2734 = x2727;
			x2734.free();

			// Dealloc(X7763)
			JCudaTensor x2735;
			x2735 = x2725;
			x2735.free();

			// Dealloc(X7759)
			JCudaTensor x2736;
			x2736 = x2723;
			x2736.free();

			// Dealloc(X7755)
			JCudaTensor x2737;
			x2737 = x2721;
			x2737.free();

			// val X7778 = Pooling(3,1,1,true)(X7767)
			JCudaTensor x2738;
			JCudaTensor x2739;
			x2739 = x2729;
			x2738 = x203.forward(x2739);

			// val X7770 = Convolv(1,0)(X7767,cv42_W,cv42_B)
			JCudaTensor x2740;
			JCudaTensor x2741, x2742, x2743;
			x2741 = x2729;
			x2742 = x259;
			x2743 = x260;
			x2740 = x210.forward(x2741, x2742, x2743);

			// val X7774 = Convolv(1,0)(X7767,cv44_W,cv44_B)
			JCudaTensor x2744;
			JCudaTensor x2745, x2746, x2747;
			x2745 = x2729;
			x2746 = x276;
			x2747 = x277;
			x2744 = x193.forward(x2745, x2746, x2747);

			// val X7768 = Convolv(1,0)(X7767,cv41_W,cv41_B)
			JCudaTensor x2748;
			JCudaTensor x2749, x2750, x2751;
			x2749 = x2729;
			x2750 = x270;
			x2751 = x271;
			x2748 = x200.forward(x2749, x2750, x2751);

			// Dealloc(X7767)
			JCudaTensor x2752;
			x2752 = x2729;
			x2752.free();

			// val X7775 = ReLU()(X7774)
			JCudaTensor x2753;
			JCudaTensor x2754;
			x2754 = x2744;
			x2753 = x223.forward(x2754);

			// val X7779 = Convolv(1,0)(X7778,cv46_W,cv46_B)
			JCudaTensor x2755;
			JCudaTensor x2756, x2757, x2758;
			x2756 = x2738;
			x2757 = x284;
			x2758 = x285;
			x2755 = x220.forward(x2756, x2757, x2758);

			// Dealloc(X7778)
			JCudaTensor x2759;
			x2759 = x2738;
			x2759.free();

			// val X7771 = ReLU()(X7770)
			JCudaTensor x2760;
			JCudaTensor x2761;
			x2761 = x2740;
			x2760 = x213.forward(x2761);

			// val X7776 = Convolv(1,2)(X7775,cv45_W,cv45_B)
			JCudaTensor x2762;
			JCudaTensor x2763, x2764, x2765;
			x2763 = x2753;
			x2764 = x299;
			x2765 = x300;
			x2762 = x230.forward(x2763, x2764, x2765);

			// Dealloc(X7775)
			JCudaTensor x2766;
			x2766 = x2753;
			x2766.free();

			// val X7772 = Convolv(1,1)(X7771,cv43_W,cv43_B)
			JCudaTensor x2767;
			JCudaTensor x2768, x2769, x2770;
			x2768 = x2760;
			x2769 = x305;
			x2770 = x306;
			x2767 = x237.forward(x2768, x2769, x2770);

			// Dealloc(X7771)
			JCudaTensor x2771;
			x2771 = x2760;
			x2771.free();

			// val X7769 = ReLU()(X7768)
			JCudaTensor x2772;
			JCudaTensor x2773;
			x2773 = x2748;
			x2772 = x240.forward(x2773);

			// val X7773 = ReLU()(X7772)
			JCudaTensor x2774;
			JCudaTensor x2775;
			x2775 = x2767;
			x2774 = x243.forward(x2775);

			// val X7777 = ReLU()(X7776)
			JCudaTensor x2776;
			JCudaTensor x2777;
			x2777 = x2762;
			x2776 = x246.forward(x2777);

			// val X7780 = ReLU()(X7779)
			JCudaTensor x2778;
			JCudaTensor x2779;
			x2779 = x2755;
			x2778 = x246.forward(x2779);

			// val X7781 = Concat(X7769,X7773,X7777,X7780)
			JCudaTensor x2780;
			JCudaTensor x2781, x2782, x2783, x2784;
			x2781 = x2772;
			x2782 = x2774;
			x2783 = x2776;
			x2784 = x2778;
			x2780 = x250.forward(x2781,x2782,x2783,x2784);

			// Dealloc(X7780)
			JCudaTensor x2785;
			x2785 = x2778;
			x2785.free();

			// Dealloc(X7777)
			JCudaTensor x2786;
			x2786 = x2776;
			x2786.free();

			// Dealloc(X7773)
			JCudaTensor x2787;
			x2787 = x2774;
			x2787.free();

			// Dealloc(X7769)
			JCudaTensor x2788;
			x2788 = x2772;
			x2788.free();

			// val X7788 = Convolv(1,0)(X7781,cv54_W,cv54_B)
			JCudaTensor x2789;
			JCudaTensor x2790, x2791, x2792;
			x2790 = x2780;
			x2791 = x338;
			x2792 = x339;
			x2789 = x193.forward(x2790, x2791, x2792);

			// val X7792 = Pooling(3,1,1,true)(X7781)
			JCudaTensor x2793;
			JCudaTensor x2794;
			x2794 = x2780;
			x2793 = x203.forward(x2794);

			// val X7784 = Convolv(1,0)(X7781,cv52_W,cv52_B)
			JCudaTensor x2795;
			JCudaTensor x2796, x2797, x2798;
			x2796 = x2780;
			x2797 = x344;
			x2798 = x345;
			x2795 = x210.forward(x2796, x2797, x2798);

			// val X7782 = Convolv(1,0)(X7781,cv51_W,cv51_B)
			JCudaTensor x2799;
			JCudaTensor x2800, x2801, x2802;
			x2800 = x2780;
			x2801 = x355;
			x2802 = x356;
			x2799 = x200.forward(x2800, x2801, x2802);

			// Dealloc(X7781)
			JCudaTensor x2803;
			x2803 = x2780;
			x2803.free();

			// val X7789 = ReLU()(X7788)
			JCudaTensor x2804;
			JCudaTensor x2805;
			x2805 = x2789;
			x2804 = x223.forward(x2805);

			// val X7793 = Convolv(1,0)(X7792,cv56_W,cv56_B)
			JCudaTensor x2806;
			JCudaTensor x2807, x2808, x2809;
			x2807 = x2793;
			x2808 = x368;
			x2809 = x369;
			x2806 = x220.forward(x2807, x2808, x2809);

			// Dealloc(X7792)
			JCudaTensor x2810;
			x2810 = x2793;
			x2810.free();

			// val X7785 = ReLU()(X7784)
			JCudaTensor x2811;
			JCudaTensor x2812;
			x2812 = x2795;
			x2811 = x213.forward(x2812);

			// val X7790 = Convolv(1,2)(X7789,cv55_W,cv55_B)
			JCudaTensor x2813;
			JCudaTensor x2814, x2815, x2816;
			x2814 = x2804;
			x2815 = x374;
			x2816 = x375;
			x2813 = x230.forward(x2814, x2815, x2816);

			// Dealloc(X7789)
			JCudaTensor x2817;
			x2817 = x2804;
			x2817.free();

			// val X7786 = Convolv(1,1)(X7785,cv53_W,cv53_B)
			JCudaTensor x2818;
			JCudaTensor x2819, x2820, x2821;
			x2819 = x2811;
			x2820 = x386;
			x2821 = x387;
			x2818 = x237.forward(x2819, x2820, x2821);

			// Dealloc(X7785)
			JCudaTensor x2822;
			x2822 = x2811;
			x2822.free();

			// val X7783 = ReLU()(X7782)
			JCudaTensor x2823;
			JCudaTensor x2824;
			x2824 = x2799;
			x2823 = x240.forward(x2824);

			// val X7787 = ReLU()(X7786)
			JCudaTensor x2825;
			JCudaTensor x2826;
			x2826 = x2818;
			x2825 = x243.forward(x2826);

			// val X7791 = ReLU()(X7790)
			JCudaTensor x2827;
			JCudaTensor x2828;
			x2828 = x2813;
			x2827 = x246.forward(x2828);

			// val X7794 = ReLU()(X7793)
			JCudaTensor x2829;
			JCudaTensor x2830;
			x2830 = x2806;
			x2829 = x246.forward(x2830);

			// val X7795 = Concat(X7783,X7787,X7791,X7794)
			JCudaTensor x2831;
			JCudaTensor x2832, x2833, x2834, x2835;
			x2832 = x2823;
			x2833 = x2825;
			x2834 = x2827;
			x2835 = x2829;
			x2831 = x250.forward(x2832,x2833,x2834,x2835);

			// Dealloc(X7794)
			JCudaTensor x2836;
			x2836 = x2829;
			x2836.free();

			// Dealloc(X7791)
			JCudaTensor x2837;
			x2837 = x2827;
			x2837.free();

			// Dealloc(X7787)
			JCudaTensor x2838;
			x2838 = x2825;
			x2838.free();

			// Dealloc(X7783)
			JCudaTensor x2839;
			x2839 = x2823;
			x2839.free();

			// val X7806 = Pooling(3,1,1,true)(X7795)
			JCudaTensor x2840;
			JCudaTensor x2841;
			x2841 = x2831;
			x2840 = x203.forward(x2841);

			// val X7802 = Convolv(1,0)(X7795,cv64_W,cv64_B)
			JCudaTensor x2842;
			JCudaTensor x2843, x2844, x2845;
			x2843 = x2831;
			x2844 = x436;
			x2845 = x437;
			x2842 = x193.forward(x2843, x2844, x2845);

			// val X7796 = Convolv(1,0)(X7795,cv61_W,cv61_B)
			JCudaTensor x2846;
			JCudaTensor x2847, x2848, x2849;
			x2847 = x2831;
			x2848 = x428;
			x2849 = x429;
			x2846 = x200.forward(x2847, x2848, x2849);

			// val X7798 = Convolv(1,0)(X7795,cv62_W,cv62_B)
			JCudaTensor x2850;
			JCudaTensor x2851, x2852, x2853;
			x2851 = x2831;
			x2852 = x422;
			x2853 = x423;
			x2850 = x210.forward(x2851, x2852, x2853);

			// Dealloc(X7795)
			JCudaTensor x2854;
			x2854 = x2831;
			x2854.free();

			// val X7799 = ReLU()(X7798)
			JCudaTensor x2855;
			JCudaTensor x2856;
			x2856 = x2850;
			x2855 = x213.forward(x2856);

			// val X7803 = ReLU()(X7802)
			JCudaTensor x2857;
			JCudaTensor x2858;
			x2858 = x2842;
			x2857 = x223.forward(x2858);

			// val X7807 = Convolv(1,0)(X7806,cv66_W,cv66_B)
			JCudaTensor x2859;
			JCudaTensor x2860, x2861, x2862;
			x2860 = x2840;
			x2861 = x444;
			x2862 = x445;
			x2859 = x220.forward(x2860, x2861, x2862);

			// Dealloc(X7806)
			JCudaTensor x2863;
			x2863 = x2840;
			x2863.free();

			// val X7800 = Convolv(1,1)(X7799,cv63_W,cv63_B)
			JCudaTensor x2864;
			JCudaTensor x2865, x2866, x2867;
			x2865 = x2855;
			x2866 = x476;
			x2867 = x477;
			x2864 = x237.forward(x2865, x2866, x2867);

			// Dealloc(X7799)
			JCudaTensor x2868;
			x2868 = x2855;
			x2868.free();

			// val X7804 = Convolv(1,2)(X7803,cv65_W,cv65_B)
			JCudaTensor x2869;
			JCudaTensor x2870, x2871, x2872;
			x2870 = x2857;
			x2871 = x467;
			x2872 = x468;
			x2869 = x230.forward(x2870, x2871, x2872);

			// Dealloc(X7803)
			JCudaTensor x2873;
			x2873 = x2857;
			x2873.free();

			// val X7797 = ReLU()(X7796)
			JCudaTensor x2874;
			JCudaTensor x2875;
			x2875 = x2846;
			x2874 = x240.forward(x2875);

			// val X7801 = ReLU()(X7800)
			JCudaTensor x2876;
			JCudaTensor x2877;
			x2877 = x2864;
			x2876 = x243.forward(x2877);

			// val X7805 = ReLU()(X7804)
			JCudaTensor x2878;
			JCudaTensor x2879;
			x2879 = x2869;
			x2878 = x246.forward(x2879);

			// val X7808 = ReLU()(X7807)
			JCudaTensor x2880;
			JCudaTensor x2881;
			x2881 = x2859;
			x2880 = x246.forward(x2881);

			// val X7809 = Concat(X7797,X7801,X7805,X7808)
			JCudaTensor x2882;
			JCudaTensor x2883, x2884, x2885, x2886;
			x2883 = x2874;
			x2884 = x2876;
			x2885 = x2878;
			x2886 = x2880;
			x2882 = x250.forward(x2883,x2884,x2885,x2886);

			// Dealloc(X7808)
			JCudaTensor x2887;
			x2887 = x2880;
			x2887.free();

			// Dealloc(X7805)
			JCudaTensor x2888;
			x2888 = x2878;
			x2888.free();

			// Dealloc(X7801)
			JCudaTensor x2889;
			x2889 = x2876;
			x2889.free();

			// Dealloc(X7797)
			JCudaTensor x2890;
			x2890 = x2874;
			x2890.free();

			// val X7820 = Pooling(3,1,1,true)(X7809)
			JCudaTensor x2891;
			JCudaTensor x2892;
			x2892 = x2882;
			x2891 = x203.forward(x2892);

			// val X7816 = Convolv(1,0)(X7809,cv74_W,cv74_B)
			JCudaTensor x2893;
			JCudaTensor x2894, x2895, x2896;
			x2894 = x2882;
			x2895 = x532;
			x2896 = x533;
			x2893 = x193.forward(x2894, x2895, x2896);

			// val X7810 = Convolv(1,0)(X7809,cv71_W,cv71_B)
			JCudaTensor x2897;
			JCudaTensor x2898, x2899, x2900;
			x2898 = x2882;
			x2899 = x554;
			x2900 = x555;
			x2897 = x200.forward(x2898, x2899, x2900);

			// val X7812 = Convolv(1,0)(X7809,cv72_W,cv72_B)
			JCudaTensor x2901;
			JCudaTensor x2902, x2903, x2904;
			x2902 = x2882;
			x2903 = x548;
			x2904 = x549;
			x2901 = x210.forward(x2902, x2903, x2904);

			// Dealloc(X7809)
			JCudaTensor x2905;
			x2905 = x2882;
			x2905.free();

			// val X7821 = Convolv(1,0)(X7820,cv76_W,cv76_B)
			JCudaTensor x2906;
			JCudaTensor x2907, x2908, x2909;
			x2907 = x2891;
			x2908 = x576;
			x2909 = x577;
			x2906 = x220.forward(x2907, x2908, x2909);

			// Dealloc(X7820)
			JCudaTensor x2910;
			x2910 = x2891;
			x2910.free();

			// val X7813 = ReLU()(X7812)
			JCudaTensor x2911;
			JCudaTensor x2912;
			x2912 = x2901;
			x2911 = x213.forward(x2912);

			// val X7817 = ReLU()(X7816)
			JCudaTensor x2913;
			JCudaTensor x2914;
			x2914 = x2893;
			x2913 = x223.forward(x2914);

			// val X7814 = Convolv(1,1)(X7813,cv73_W,cv73_B)
			JCudaTensor x2915;
			JCudaTensor x2916, x2917, x2918;
			x2916 = x2911;
			x2917 = x612;
			x2918 = x613;
			x2915 = x237.forward(x2916, x2917, x2918);

			// Dealloc(X7813)
			JCudaTensor x2919;
			x2919 = x2911;
			x2919.free();

			// val X7818 = Convolv(1,2)(X7817,cv75_W,cv75_B)
			JCudaTensor x2920;
			JCudaTensor x2921, x2922, x2923;
			x2921 = x2913;
			x2922 = x618;
			x2923 = x619;
			x2920 = x230.forward(x2921, x2922, x2923);

			// Dealloc(X7817)
			JCudaTensor x2924;
			x2924 = x2913;
			x2924.free();

			// val X7811 = ReLU()(X7810)
			JCudaTensor x2925;
			JCudaTensor x2926;
			x2926 = x2897;
			x2925 = x240.forward(x2926);

			// val X7815 = ReLU()(X7814)
			JCudaTensor x2927;
			JCudaTensor x2928;
			x2928 = x2915;
			x2927 = x243.forward(x2928);

			// val X7819 = ReLU()(X7818)
			JCudaTensor x2929;
			JCudaTensor x2930;
			x2930 = x2920;
			x2929 = x246.forward(x2930);

			// val X7822 = ReLU()(X7821)
			JCudaTensor x2931;
			JCudaTensor x2932;
			x2932 = x2906;
			x2931 = x246.forward(x2932);

			// val X7823 = Concat(X7811,X7815,X7819,X7822)
			JCudaTensor x2933;
			JCudaTensor x2934, x2935, x2936, x2937;
			x2934 = x2925;
			x2935 = x2927;
			x2936 = x2929;
			x2937 = x2931;
			x2933 = x250.forward(x2934,x2935,x2936,x2937);

			// Dealloc(X7822)
			JCudaTensor x2938;
			x2938 = x2931;
			x2938.free();

			// Dealloc(X7819)
			JCudaTensor x2939;
			x2939 = x2929;
			x2939.free();

			// Dealloc(X7815)
			JCudaTensor x2940;
			x2940 = x2927;
			x2940.free();

			// Dealloc(X7811)
			JCudaTensor x2941;
			x2941 = x2925;
			x2941.free();

			// val X7824 = Pooling(3,2,1,true)(X7823)
			JCudaTensor x2942;
			JCudaTensor x2943;
			x2943 = x2933;
			x2942 = x676.forward(x2943);

			// Dealloc(X7823)
			JCudaTensor x2944;
			x2944 = x2933;
			x2944.free();

			// val X7835 = Pooling(3,1,1,true)(X7824)
			JCudaTensor x2945;
			JCudaTensor x2946;
			x2946 = x2942;
			x2945 = x681.forward(x2946);

			// val X7831 = Convolv(1,0)(X7824,cv84_W,cv84_B)
			JCudaTensor x2947;
			JCudaTensor x2948, x2949, x2950;
			x2948 = x2942;
			x2949 = x686;
			x2950 = x687;
			x2947 = x688.forward(x2948, x2949, x2950);

			// val X7827 = Convolv(1,0)(X7824,cv82_W,cv82_B)
			JCudaTensor x2951;
			JCudaTensor x2952, x2953, x2954;
			x2952 = x2942;
			x2953 = x695;
			x2954 = x696;
			x2951 = x697.forward(x2952, x2953, x2954);

			// val X7825 = Convolv(1,0)(X7824,cv81_W,cv81_B)
			JCudaTensor x2955;
			JCudaTensor x2956, x2957, x2958;
			x2956 = x2942;
			x2957 = x702;
			x2958 = x703;
			x2955 = x704.forward(x2956, x2957, x2958);

			// Dealloc(X7824)
			JCudaTensor x2959;
			x2959 = x2942;
			x2959.free();

			// val X7828 = ReLU()(X7827)
			JCudaTensor x2960;
			JCudaTensor x2961;
			x2961 = x2951;
			x2960 = x723.forward(x2961);

			// val X7832 = ReLU()(X7831)
			JCudaTensor x2962;
			JCudaTensor x2963;
			x2963 = x2947;
			x2962 = x713.forward(x2963);

			// val X7836 = Convolv(1,0)(X7835,cv86_W,cv86_B)
			JCudaTensor x2964;
			JCudaTensor x2965, x2966, x2967;
			x2965 = x2945;
			x2966 = x718;
			x2967 = x719;
			x2964 = x720.forward(x2965, x2966, x2967);

			// Dealloc(X7835)
			JCudaTensor x2968;
			x2968 = x2945;
			x2968.free();

			// val X7833 = Convolv(1,2)(X7832,cv85_W,cv85_B)
			JCudaTensor x2969;
			JCudaTensor x2970, x2971, x2972;
			x2970 = x2962;
			x2971 = x728;
			x2972 = x729;
			x2969 = x730.forward(x2970, x2971, x2972);

			// Dealloc(X7832)
			JCudaTensor x2973;
			x2973 = x2962;
			x2973.free();

			// val X7829 = Convolv(1,1)(X7828,cv83_W,cv83_B)
			JCudaTensor x2974;
			JCudaTensor x2975, x2976, x2977;
			x2975 = x2960;
			x2976 = x739;
			x2977 = x740;
			x2974 = x741.forward(x2975, x2976, x2977);

			// Dealloc(X7828)
			JCudaTensor x2978;
			x2978 = x2960;
			x2978.free();

			// val X7826 = ReLU()(X7825)
			JCudaTensor x2979;
			JCudaTensor x2980;
			x2980 = x2955;
			x2979 = x747.forward(x2980);

			// val X7830 = ReLU()(X7829)
			JCudaTensor x2981;
			JCudaTensor x2982;
			x2982 = x2974;
			x2981 = x753.forward(x2982);

			// val X7834 = ReLU()(X7833)
			JCudaTensor x2983;
			JCudaTensor x2984;
			x2984 = x2969;
			x2983 = x744.forward(x2984);

			// val X7837 = ReLU()(X7836)
			JCudaTensor x2985;
			JCudaTensor x2986;
			x2986 = x2964;
			x2985 = x744.forward(x2986);

			// val X7838 = Concat(X7826,X7830,X7834,X7837)
			JCudaTensor x2987;
			JCudaTensor x2988, x2989, x2990, x2991;
			x2988 = x2979;
			x2989 = x2981;
			x2990 = x2983;
			x2991 = x2985;
			x2987 = x757.forward(x2988,x2989,x2990,x2991);

			// Dealloc(X7837)
			JCudaTensor x2992;
			x2992 = x2985;
			x2992.free();

			// Dealloc(X7834)
			JCudaTensor x2993;
			x2993 = x2983;
			x2993.free();

			// Dealloc(X7830)
			JCudaTensor x2994;
			x2994 = x2981;
			x2994.free();

			// Dealloc(X7826)
			JCudaTensor x2995;
			x2995 = x2979;
			x2995.free();

			// val X7845 = Convolv(1,0)(X7838,cv94_W,cv94_B)
			JCudaTensor x2996;
			JCudaTensor x2997, x2998, x2999;
			x2997 = x2987;
			x2998 = x789;
			x2999 = x790;
			x2996 = x688.forward(x2997, x2998, x2999);

			// val X7849 = Pooling(3,1,1,true)(X7838)
			JCudaTensor x3000;
			JCudaTensor x3001;
			x3001 = x2987;
			x3000 = x681.forward(x3001);

			// val X7841 = Convolv(1,0)(X7838,cv92_W,cv92_B)
			JCudaTensor x3002;
			JCudaTensor x3003, x3004, x3005;
			x3003 = x2987;
			x3004 = x777;
			x3005 = x778;
			x3002 = x697.forward(x3003, x3004, x3005);

			// val X7839 = Convolv(1,0)(X7838,cv91_W,cv91_B)
			JCudaTensor x3006;
			JCudaTensor x3007, x3008, x3009;
			x3007 = x2987;
			x3008 = x771;
			x3009 = x772;
			x3006 = x704.forward(x3007, x3008, x3009);

			// Dealloc(X7838)
			JCudaTensor x3010;
			x3010 = x2987;
			x3010.free();

			// val X7850 = Convolv(1,0)(X7849,cv96_W,cv96_B)
			JCudaTensor x3011;
			JCudaTensor x3012, x3013, x3014;
			x3012 = x3000;
			x3013 = x799;
			x3014 = x800;
			x3011 = x720.forward(x3012, x3013, x3014);

			// Dealloc(X7849)
			JCudaTensor x3015;
			x3015 = x3000;
			x3015.free();

			// val X7842 = ReLU()(X7841)
			JCudaTensor x3016;
			JCudaTensor x3017;
			x3017 = x3002;
			x3016 = x723.forward(x3017);

			// val X7846 = ReLU()(X7845)
			JCudaTensor x3018;
			JCudaTensor x3019;
			x3019 = x2996;
			x3018 = x713.forward(x3019);

			// val X7847 = Convolv(1,2)(X7846,cv95_W,cv95_B)
			JCudaTensor x3020;
			JCudaTensor x3021, x3022, x3023;
			x3021 = x3018;
			x3022 = x822;
			x3023 = x823;
			x3020 = x730.forward(x3021, x3022, x3023);

			// Dealloc(X7846)
			JCudaTensor x3024;
			x3024 = x3018;
			x3024.free();

			// val X7843 = Convolv(1,1)(X7842,cv93_W,cv93_B)
			JCudaTensor x3025;
			JCudaTensor x3026, x3027, x3028;
			x3026 = x3016;
			x3027 = x828;
			x3028 = x829;
			x3025 = x741.forward(x3026, x3027, x3028);

			// Dealloc(X7842)
			JCudaTensor x3029;
			x3029 = x3016;
			x3029.free();

			// val X7840 = ReLU()(X7839)
			JCudaTensor x3030;
			JCudaTensor x3031;
			x3031 = x3006;
			x3030 = x747.forward(x3031);

			// val X7844 = ReLU()(X7843)
			JCudaTensor x3032;
			JCudaTensor x3033;
			x3033 = x3025;
			x3032 = x753.forward(x3033);

			// val X7848 = ReLU()(X7847)
			JCudaTensor x3034;
			JCudaTensor x3035;
			x3035 = x3020;
			x3034 = x744.forward(x3035);

			// val X7851 = ReLU()(X7850)
			JCudaTensor x3036;
			JCudaTensor x3037;
			x3037 = x3011;
			x3036 = x744.forward(x3037);

			// val X7852 = Concat(X7840,X7844,X7848,X7851)
			JCudaTensor x3038;
			JCudaTensor x3039, x3040, x3041, x3042;
			x3039 = x3030;
			x3040 = x3032;
			x3041 = x3034;
			x3042 = x3036;
			x3038 = x757.forward(x3039,x3040,x3041,x3042);

			// Dealloc(X7851)
			JCudaTensor x3043;
			x3043 = x3036;
			x3043.free();

			// Dealloc(X7848)
			JCudaTensor x3044;
			x3044 = x3034;
			x3044.free();

			// Dealloc(X7844)
			JCudaTensor x3045;
			x3045 = x3032;
			x3045.free();

			// Dealloc(X7840)
			JCudaTensor x3046;
			x3046 = x3030;
			x3046.free();

			// val X7853 = Pooling(7,1,0,false)(X7852)
			JCudaTensor x3047;
			JCudaTensor x3048;
			x3048 = x3038;
			x3047 = x902.forward(x3048);

			// Dealloc(X7852)
			JCudaTensor x3049;
			x3049 = x3038;
			x3049.free();

			// val X7854 = Dropout(0.4)(X7853)
			JCudaTensor x3050;
			JCudaTensor x3051;
			x3051 = x3047;
			x3050 = x916.forward(x3051);

			// Dealloc(X7853)
			JCudaTensor x3052;
			x3052 = x3047;
			x3052.free();

			// val X7855 = (X7854[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x3053;
			JCudaMatrix x3054;
			JCudaMatrix x3055;
			JCudaTensor x3056;
			JCudaTensor x3057;
			x3057 = x3050;
			x3056 = x3057.flatten(1, new int[]{256, 1, 1});
			x3054 = x3056.asMatrix(1, true);
			JCudaTensor x3058;
			x3058 = x953;
			x3055 = x3058.asMatrix(1, true);
			x3053 = x3054.times(x3055);

			// Dealloc(X7854)
			JCudaTensor x3059;
			x3059 = x3050;
			x3059.free();

			// val X7857 = (X7855 + (i) => fc_B)
			JCudaTensor x3060;
			JCudaTensor x3061, x3062;
			x3061 = x3053;
			x3062 = x963;
			x3060 = x3062.copy(128, x3061);

			// Precision(Accuracy(1))
			float x3064;
			JCudaTensor x3065;
			JTensorFloat x3066;
			x3065 = x3060;
			x3066 = x4;
			x3064 = x3065.accuracy(x3066, 1);
			System.out.println(x5 + " test precision "  + x3064);
			x3063 += x3064;

			// Dealloc(X7857)
			JCudaTensor x3067;
			x3067 = x3060;
			x3067.free();

		}
		System.out.println();
		System.out.println("average precision: " + x3063/10);
		System.out.println(); 
	}

}