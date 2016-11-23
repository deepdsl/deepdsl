package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Googlenet {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// decay
	static float decay = 5.0E-4f;
	// loss1
	static float loss1 = 0.3f;
	// loss2
	static float loss2 = 0.3f;
	// lrn_rate
	static float lrn_rate = -0.01f;
	// momentum
	static float momentum = 0.9f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/googlenet";
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
	static JCudnnConvolution x724 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x756 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x740 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x733 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 64, 56, 56), List(64, 64, 1, 1), List(64)))
	static JCudnnConvolution x36 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{64,64,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,1),List(List(128, 64, 56, 56), List(192, 64, 3, 3), List(192)))
	static JCudnnConvolution x46 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{192,64,3,3},new int[]{192}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 14, 14), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x237 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 28, 28), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x99 = new JCudnnConvolution(new int[]{128,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 7, 7), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x777 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,2),List(List(128, 16, 14, 14), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x230 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(1,2),List(List(128, 16, 28, 28), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x106 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(1,2),List(List(128, 16, 7, 7), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x766 = new JCudnnConvolution(new int[]{128,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(2,3),List(List(128, 3, 224, 224), List(64, 3, 7, 7), List(64)))
	static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
	// (Dropout(0.4),List(List(128, 256, 1, 1)))
	static JCudnnDropout x966 = new JCudnnDropout(new int[]{128,256,1,1}, 0.4f);
	// (Dropout(0.7),List(List(128, 1024)))
	static JCudnnDropout x361 = new JCudnnDropout(new int[]{128,1024}, 0.7f);
	// (LRN(5,1.0E-4,0.75),List(List(128, 192, 56, 56)))
	static JCudnnLRN x52 = new JCudnnLRN(new int[]{128,192,56,56}, 5, 1.0E-4, 0.75);
	// (LRN(5,1.0E-4,0.75),List(List(128, 64, 56, 56)))
	static JCudnnLRN x29 = new JCudnnLRN(new int[]{128,64,56,56}, 5, 1.0E-4, 0.75);
	// (Lmdb(1000000,10000,1000),false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, 1000, true);
	// (Lmdb(1000000,10000,1000),true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, 1000, false);
	// (LogSoftmax(),List(List(128, 1000)))
	static JCudnnSoftmax x402 = new JCudnnSoftmax(new int[]{128,1000}, SoftmaxAlgorithm.LOG);
	// (Pooling(3,1,1,true),List(List(128, 192, 28, 28)))
	static JCudnnPooling x72 = new JCudnnPooling(new int[]{128,192,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 14, 14)))
	static JCudnnPooling x203 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 28, 28)))
	static JCudnnPooling x126 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 7, 7)))
	static JCudnnPooling x717 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 192, 56, 56)))
	static JCudnnPooling x55 = new JCudnnPooling(new int[]{128,192,56,56}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 256, 14, 14)))
	static JCudnnPooling x712 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 256, 28, 28)))
	static JCudnnPooling x186 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 64, 112, 112)))
	static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,64,112,112}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(5,3,0,false),List(List(128, 256, 14, 14)))
	static JCudnnPooling x263 = new JCudnnPooling(new int[]{128,256,14,14}, 5, 3, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
	// (Pooling(7,1,0,false),List(List(128, 256, 7, 7)))
	static JCudnnPooling x950 = new JCudnnPooling(new int[]{128,256,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
	// (ReLU(),List(List(128, 1024)))
	static JCudnnActivation x348 = new JCudnnActivation(new int[]{128,1024}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 14, 14)))
	static JCudnnActivation x243 = new JCudnnActivation(new int[]{128,128,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 28, 28)))
	static JCudnnActivation x112 = new JCudnnActivation(new int[]{128,128,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 4, 4)))
	static JCudnnActivation x309 = new JCudnnActivation(new int[]{128,128,4,4}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 7, 7)))
	static JCudnnActivation x789 = new JCudnnActivation(new int[]{128,128,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 14, 14)))
	static JCudnnActivation x223 = new JCudnnActivation(new int[]{128,16,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 28, 28)))
	static JCudnnActivation x92 = new JCudnnActivation(new int[]{128,16,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 7, 7)))
	static JCudnnActivation x749 = new JCudnnActivation(new int[]{128,16,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 192, 56, 56)))
	static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,192,56,56}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 14, 14)))
	static JCudnnActivation x246 = new JCudnnActivation(new int[]{128,32,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 28, 28)))
	static JCudnnActivation x115 = new JCudnnActivation(new int[]{128,32,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 7, 7)))
	static JCudnnActivation x780 = new JCudnnActivation(new int[]{128,32,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 112, 112)))
	static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,64,112,112}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 14, 14)))
	static JCudnnActivation x240 = new JCudnnActivation(new int[]{128,64,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 28, 28)))
	static JCudnnActivation x109 = new JCudnnActivation(new int[]{128,64,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 56, 56)))
	static JCudnnActivation x39 = new JCudnnActivation(new int[]{128,64,56,56}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 7, 7)))
	static JCudnnActivation x783 = new JCudnnActivation(new int[]{128,64,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 14, 14)))
	static JCudnnActivation x213 = new JCudnnActivation(new int[]{128,96,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 28, 28)))
	static JCudnnActivation x89 = new JCudnnActivation(new int[]{128,96,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 7, 7)))
	static JCudnnActivation x759 = new JCudnnActivation(new int[]{128,96,7,7}, ActivationMode.RELU);
	// List(List(128, 64, 14, 14), List(128, 128, 14, 14), List(128, 32, 14, 14), List(128, 32, 14, 14))
	static JCudnnConcat x250 = new JCudnnConcat(new int[]{128,64,14,14},new int[]{128,128,14,14},new int[]{128,32,14,14},new int[]{128,32,14,14});
	// List(List(128, 64, 28, 28), List(128, 128, 28, 28), List(128, 32, 28, 28), List(128, 32, 28, 28))
	static JCudnnConcat x119 = new JCudnnConcat(new int[]{128,64,28,28},new int[]{128,128,28,28},new int[]{128,32,28,28},new int[]{128,32,28,28});
	// List(List(128, 64, 7, 7), List(128, 128, 7, 7), List(128, 32, 7, 7), List(128, 32, 7, 7))
	static JCudnnConcat x793 = new JCudnnConcat(new int[]{128,64,7,7},new int[]{128,128,7,7},new int[]{128,32,7,7},new int[]{128,32,7,7});
	// Precision(Accuracy(X7857, Y, 1))
	static float x3831;
	// V_b1cv_B
	static JCudaTensor x658 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_b1cv_W
	static JCudaTensor x667 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
	// V_b1fc1_B
	static JCudaTensor x623 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_b1fc1_W
	static JCudaTensor x604 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
	// V_b1fc2_B
	static JCudaTensor x507 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_b1fc2_W
	static JCudaTensor x517 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
	// V_b2cv_B
	static JCudaTensor x1004 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_b2cv_W
	static JCudaTensor x997 = JTensor.constFloat(0.0f, 128, 256, 1, 1).asJCudaTensor();
	// V_b2fc1_B
	static JCudaTensor x954 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_b2fc1_W
	static JCudaTensor x976 = JTensor.constFloat(0.0f, 1024, 2048).asJCudaTensor();
	// V_b2fc2_B
	static JCudaTensor x897 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_b2fc2_W
	static JCudaTensor x888 = JTensor.constFloat(0.0f, 1000, 1024).asJCudaTensor();
	// V_cv11_B
	static JCudaTensor x3007 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv11_W
	static JCudaTensor x2993 = JTensor.constFloat(0.0f, 64, 192, 1, 1).asJCudaTensor();
	// V_cv12_B
	static JCudaTensor x3126 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv12_W
	static JCudaTensor x3142 = JTensor.constFloat(0.0f, 96, 192, 1, 1).asJCudaTensor();
	// V_cv13_B
	static JCudaTensor x3032 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv13_W
	static JCudaTensor x3000 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv14_B
	static JCudaTensor x3136 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv14_W
	static JCudaTensor x3119 = JTensor.constFloat(0.0f, 16, 192, 1, 1).asJCudaTensor();
	// V_cv15_B
	static JCudaTensor x3026 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv15_W
	static JCudaTensor x3039 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv16_B
	static JCudaTensor x2987 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv16_W
	static JCudaTensor x3047 = JTensor.constFloat(0.0f, 32, 192, 1, 1).asJCudaTensor();
	// V_cv1_B
	static JCudaTensor x3294 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x3286 = JTensor.constFloat(0.0f, 64, 3, 7, 7).asJCudaTensor();
	// V_cv21_B
	static JCudaTensor x2770 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv21_W
	static JCudaTensor x2782 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv22_B
	static JCudaTensor x2905 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv22_W
	static JCudaTensor x2915 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv23_B
	static JCudaTensor x2789 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv23_W
	static JCudaTensor x2760 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv24_B
	static JCudaTensor x2899 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv24_W
	static JCudaTensor x2892 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv25_B
	static JCudaTensor x2803 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv25_W
	static JCudaTensor x2809 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv26_B
	static JCudaTensor x2776 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv26_W
	static JCudaTensor x2795 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x3243 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x3249 = JTensor.constFloat(0.0f, 64, 64, 1, 1).asJCudaTensor();
	// V_cv31_B
	static JCudaTensor x2588 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv31_W
	static JCudaTensor x2567 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv32_B
	static JCudaTensor x2676 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv32_W
	static JCudaTensor x2682 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv33_B
	static JCudaTensor x2546 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv33_W
	static JCudaTensor x2577 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv34_B
	static JCudaTensor x2670 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv34_W
	static JCudaTensor x2659 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv35_B
	static JCudaTensor x2534 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv35_W
	static JCudaTensor x2527 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv36_B
	static JCudaTensor x2540 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv36_W
	static JCudaTensor x2556 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv3_B
	static JCudaTensor x3208 = JTensor.constFloat(0.0f, 192).asJCudaTensor();
	// V_cv3_W
	static JCudaTensor x3214 = JTensor.constFloat(0.0f, 192, 64, 3, 3).asJCudaTensor();
	// V_cv41_B
	static JCudaTensor x2346 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv41_W
	static JCudaTensor x2322 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv42_B
	static JCudaTensor x2436 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv42_W
	static JCudaTensor x2442 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv43_B
	static JCudaTensor x2296 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv43_W
	static JCudaTensor x2302 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv44_B
	static JCudaTensor x2450 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv44_W
	static JCudaTensor x2425 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv45_B
	static JCudaTensor x2337 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv45_W
	static JCudaTensor x2353 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv46_B
	static JCudaTensor x2316 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv46_W
	static JCudaTensor x2329 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv51_B
	static JCudaTensor x2097 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv51_W
	static JCudaTensor x2090 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv52_B
	static JCudaTensor x2205 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv52_W
	static JCudaTensor x2221 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv53_B
	static JCudaTensor x2078 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv53_W
	static JCudaTensor x2104 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv54_B
	static JCudaTensor x2211 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv54_W
	static JCudaTensor x2198 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv55_B
	static JCudaTensor x2069 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv55_W
	static JCudaTensor x2118 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv56_B
	static JCudaTensor x2112 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv56_W
	static JCudaTensor x2126 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv61_B
	static JCudaTensor x1896 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv61_W
	static JCudaTensor x1839 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv62_B
	static JCudaTensor x1989 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv62_W
	static JCudaTensor x1982 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv63_B
	static JCudaTensor x1855 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv63_W
	static JCudaTensor x1875 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv64_B
	static JCudaTensor x1996 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv64_W
	static JCudaTensor x1971 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv65_B
	static JCudaTensor x1849 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv65_W
	static JCudaTensor x1868 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv66_B
	static JCudaTensor x1883 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv66_W
	static JCudaTensor x1861 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv71_B
	static JCudaTensor x1641 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv71_W
	static JCudaTensor x1662 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv72_B
	static JCudaTensor x1755 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv72_W
	static JCudaTensor x1748 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv73_B
	static JCudaTensor x1625 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv73_W
	static JCudaTensor x1605 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv74_B
	static JCudaTensor x1762 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv74_W
	static JCudaTensor x1737 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv75_B
	static JCudaTensor x1632 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv75_W
	static JCudaTensor x1654 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv76_B
	static JCudaTensor x1647 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv76_W
	static JCudaTensor x1612 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv81_B
	static JCudaTensor x1372 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv81_W
	static JCudaTensor x1400 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv82_B
	static JCudaTensor x1515 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv82_W
	static JCudaTensor x1521 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv83_B
	static JCudaTensor x1408 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv83_W
	static JCudaTensor x1381 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv84_B
	static JCudaTensor x1529 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv84_W
	static JCudaTensor x1504 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv85_B
	static JCudaTensor x1391 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv85_W
	static JCudaTensor x1424 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv86_B
	static JCudaTensor x1418 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv86_W
	static JCudaTensor x1432 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_cv91_B
	static JCudaTensor x1177 = JTensor.constFloat(0.0f, 64).asJCudaTensor();
	// V_cv91_W
	static JCudaTensor x1205 = JTensor.constFloat(0.0f, 64, 256, 1, 1).asJCudaTensor();
	// V_cv92_B
	static JCudaTensor x1301 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv92_W
	static JCudaTensor x1294 = JTensor.constFloat(0.0f, 96, 256, 1, 1).asJCudaTensor();
	// V_cv93_B
	static JCudaTensor x1145 = JTensor.constFloat(0.0f, 128).asJCudaTensor();
	// V_cv93_W
	static JCudaTensor x1183 = JTensor.constFloat(0.0f, 128, 96, 3, 3).asJCudaTensor();
	// V_cv94_B
	static JCudaTensor x1284 = JTensor.constFloat(0.0f, 16).asJCudaTensor();
	// V_cv94_W
	static JCudaTensor x1277 = JTensor.constFloat(0.0f, 16, 256, 1, 1).asJCudaTensor();
	// V_cv95_B
	static JCudaTensor x1168 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv95_W
	static JCudaTensor x1158 = JTensor.constFloat(0.0f, 32, 16, 5, 5).asJCudaTensor();
	// V_cv96_B
	static JCudaTensor x1191 = JTensor.constFloat(0.0f, 32).asJCudaTensor();
	// V_cv96_W
	static JCudaTensor x1151 = JTensor.constFloat(0.0f, 32, 256, 1, 1).asJCudaTensor();
	// V_fc_B
	static JCudaTensor x1094 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc_W
	static JCudaTensor x1086 = JTensor.constFloat(0.0f, 1000, 256).asJCudaTensor();
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
	static JCudaTensor x600 = JTensor.constFloat(0.2f, 128).load(network_dir + "/b2cv_B").asJCudaTensor();
	// b2cv_W
	static JCudaTensor x599 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 128, 256, 1, 1).load(network_dir + "/b2cv_W").asJCudaTensor();
	// b2fc1_B
	static JCudaTensor x704 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/b2fc1_B").asJCudaTensor();
	// b2fc1_W
	static JCudaTensor x686 = JTensor.randomFloat(-0.03125f, 0.03125f, 1024, 2048).load(network_dir + "/b2fc1_W").asJCudaTensor();
	// b2fc2_B
	static JCudaTensor x770 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/b2fc2_B").asJCudaTensor();
	// b2fc2_W
	static JCudaTensor x746 = JTensor.randomFloat(-0.044194173f, 0.044194173f, 1000, 1024).load(network_dir + "/b2fc2_W").asJCudaTensor();
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
	static JCudaTensor x567 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv71_B").asJCudaTensor();
	// cv71_W
	static JCudaTensor x566 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv71_W").asJCudaTensor();
	// cv72_B
	static JCudaTensor x561 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv72_B").asJCudaTensor();
	// cv72_W
	static JCudaTensor x560 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv72_W").asJCudaTensor();
	// cv73_B
	static JCudaTensor x622 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv73_B").asJCudaTensor();
	// cv73_W
	static JCudaTensor x621 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv73_W").asJCudaTensor();
	// cv74_B
	static JCudaTensor x545 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv74_B").asJCudaTensor();
	// cv74_W
	static JCudaTensor x544 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv74_W").asJCudaTensor();
	// cv75_B
	static JCudaTensor x635 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv75_B").asJCudaTensor();
	// cv75_W
	static JCudaTensor x634 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv75_W").asJCudaTensor();
	// cv76_B
	static JCudaTensor x589 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv76_B").asJCudaTensor();
	// cv76_W
	static JCudaTensor x588 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv76_W").asJCudaTensor();
	// cv81_B
	static JCudaTensor x739 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv81_B").asJCudaTensor();
	// cv81_W
	static JCudaTensor x738 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv81_W").asJCudaTensor();
	// cv82_B
	static JCudaTensor x732 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv82_B").asJCudaTensor();
	// cv82_W
	static JCudaTensor x731 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv82_W").asJCudaTensor();
	// cv83_B
	static JCudaTensor x776 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv83_B").asJCudaTensor();
	// cv83_W
	static JCudaTensor x775 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv83_W").asJCudaTensor();
	// cv84_B
	static JCudaTensor x723 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv84_B").asJCudaTensor();
	// cv84_W
	static JCudaTensor x722 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv84_W").asJCudaTensor();
	// cv85_B
	static JCudaTensor x765 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv85_B").asJCudaTensor();
	// cv85_W
	static JCudaTensor x764 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv85_W").asJCudaTensor();
	// cv86_B
	static JCudaTensor x755 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv86_B").asJCudaTensor();
	// cv86_W
	static JCudaTensor x754 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv86_W").asJCudaTensor();
	// cv91_B
	static JCudaTensor x808 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv91_B").asJCudaTensor();
	// cv91_W
	static JCudaTensor x807 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv91_W").asJCudaTensor();
	// cv92_B
	static JCudaTensor x814 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv92_B").asJCudaTensor();
	// cv92_W
	static JCudaTensor x813 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv92_W").asJCudaTensor();
	// cv93_B
	static JCudaTensor x865 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv93_B").asJCudaTensor();
	// cv93_W
	static JCudaTensor x864 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv93_W").asJCudaTensor();
	// cv94_B
	static JCudaTensor x826 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv94_B").asJCudaTensor();
	// cv94_W
	static JCudaTensor x825 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv94_W").asJCudaTensor();
	// cv95_B
	static JCudaTensor x859 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv95_B").asJCudaTensor();
	// cv95_W
	static JCudaTensor x858 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv95_W").asJCudaTensor();
	// cv96_B
	static JCudaTensor x836 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv96_B").asJCudaTensor();
	// cv96_W
	static JCudaTensor x835 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv96_W").asJCudaTensor();
	// fc_B
	static JCudaTensor x1035 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
	// fc_W
	static JCudaTensor x1017 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1000, 256).load(network_dir + "/fc_W").asJCudaTensor();

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
		x600.save(network_dir + "/b2cv_B");
		x599.save(network_dir + "/b2cv_W");
		x704.save(network_dir + "/b2fc1_B");
		x686.save(network_dir + "/b2fc1_W");
		x770.save(network_dir + "/b2fc2_B");
		x746.save(network_dir + "/b2fc2_W");
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
		x567.save(network_dir + "/cv71_B");
		x566.save(network_dir + "/cv71_W");
		x561.save(network_dir + "/cv72_B");
		x560.save(network_dir + "/cv72_W");
		x622.save(network_dir + "/cv73_B");
		x621.save(network_dir + "/cv73_W");
		x545.save(network_dir + "/cv74_B");
		x544.save(network_dir + "/cv74_W");
		x635.save(network_dir + "/cv75_B");
		x634.save(network_dir + "/cv75_W");
		x589.save(network_dir + "/cv76_B");
		x588.save(network_dir + "/cv76_W");
		x739.save(network_dir + "/cv81_B");
		x738.save(network_dir + "/cv81_W");
		x732.save(network_dir + "/cv82_B");
		x731.save(network_dir + "/cv82_W");
		x776.save(network_dir + "/cv83_B");
		x775.save(network_dir + "/cv83_W");
		x723.save(network_dir + "/cv84_B");
		x722.save(network_dir + "/cv84_W");
		x765.save(network_dir + "/cv85_B");
		x764.save(network_dir + "/cv85_W");
		x755.save(network_dir + "/cv86_B");
		x754.save(network_dir + "/cv86_W");
		x808.save(network_dir + "/cv91_B");
		x807.save(network_dir + "/cv91_W");
		x814.save(network_dir + "/cv92_B");
		x813.save(network_dir + "/cv92_W");
		x865.save(network_dir + "/cv93_B");
		x864.save(network_dir + "/cv93_W");
		x826.save(network_dir + "/cv94_B");
		x825.save(network_dir + "/cv94_W");
		x859.save(network_dir + "/cv95_B");
		x858.save(network_dir + "/cv95_W");
		x836.save(network_dir + "/cv96_B");
		x835.save(network_dir + "/cv96_W");
		x1035.save(network_dir + "/fc_B");
		x1017.save(network_dir + "/fc_W");
		x381.free();
		x61.free();
		x1284.free();
		x1625.free();
		x1086.free();
		x1504.free();
		x1017.free();
		x131.free();
		x1971.free();
		x15.free();
		x1605.free();
		x3126.free();
		x2776.free();
		x1839.free();
		x770.free();
		x731.free();
		x1996.free();
		x825.free();
		x428.free();
		x1737.free();
		x229.free();
		x305.free();
		x356.free();
		x1391.free();
		x1168.free();
		x859.free();
		x776.free();
		x191.free();
		x507.free();
		x2567.free();
		x260.free();
		x1883.free();
		x219.free();
		x235.free();
		x599.free();
		x561.free();
		x164.free();
		x2676.free();
		x97.free();
		x138.free();
		x2436.free();
		x667.free();
		x1896.free();
		x765.free();
		x169.free();
		x369.free();
		x1372.free();
		x3007.free();
		x2588.free();
		x34.free();
		x775.free();
		x732.free();
		x1191.free();
		x3047.free();
		x84.free();
		x1183.free();
		x422.free();
		x299.free();
		x3294.free();
		x285.free();
		x3249.free();
		x1151.free();
		x1381.free();
		x2442.free();
		x1432.free();
		x2221.free();
		x739.free();
		x723.free();
		x2905.free();
		x98.free();
		x386.free();
		x477.free();
		x3136.free();
		x2760.free();
		x2782.free();
		x2809.free();
		x368.free();
		x622.free();
		x1424.free();
		x1035.free();
		x152.free();
		x1408.free();
		x1762.free();
		x836.free();
		x67.free();
		x704.free();
		x35.free();
		x634.free();
		x344.free();
		x2198.free();
		x445.free();
		x375.free();
		x2104.free();
		x2546.free();
		x355.free();
		x271.free();
		x2126.free();
		x145.free();
		x284.free();
		x858.free();
		x1641.free();
		x686.free();
		x333.free();
		x444.free();
		x163.free();
		x339.free();
		x3142.free();
		x3286.free();
		x2527.free();
		x77.free();
		x722.free();
		x1400.free();
		x2078.free();
		x2329.free();
		x170.free();
		x658.free();
		x1529.free();
		x2337.free();
		x437.free();
		x199.free();
		x1855.free();
		x338.free();
		x1418.free();
		x85.free();
		x1145.free();
		x2795.free();
		x545.free();
		x2296.free();
		x3243.free();
		x192.free();
		x290.free();
		x1982.free();
		x621.free();
		x45.free();
		x835.free();
		x60.free();
		x3208.free();
		x476.free();
		x864.free();
		x300.free();
		x1612.free();
		x1301.free();
		x2302.free();
		x544.free();
		x78.free();
		x2346.free();
		x259.free();
		x2205.free();
		x808.free();
		x3039.free();
		x3000.free();
		x153.free();
		x3026.free();
		x566.free();
		x1515.free();
		x467.free();
		x2112.free();
		x395.free();
		x104.free();
		x1989.free();
		x865.free();
		x1748.free();
		x2577.free();
		x635.free();
		x2987.free();
		x2915.free();
		x139.free();
		x1868.free();
		x468.free();
		x746.free();
		x276.free();
		x1849.free();
		x1294.free();
		x291.free();
		x600.free();
		x306.free();
		x2322.free();
		x436.free();
		x270.free();
		x1654.free();
		x1632.free();
		x2534.free();
		x146.free();
		x2425.free();
		x2097.free();
		x997.free();
		x44.free();
		x623.free();
		x1158.free();
		x2090.free();
		x764.free();
		x813.free();
		x1875.free();
		x3214.free();
		x16.free();
		x567.free();
		x954.free();
		x897.free();
		x589.free();
		x105.free();
		x345.free();
		x3032.free();
		x2353.free();
		x322.free();
		x209.free();
		x1094.free();
		x604.free();
		x2803.free();
		x2899.free();
		x826.free();
		x1521.free();
		x374.free();
		x132.free();
		x429.free();
		x2069.free();
		x2659.free();
		x755.free();
		x68.free();
		x1755.free();
		x2211.free();
		x2450.free();
		x208.free();
		x198.free();
		x888.free();
		x738.free();
		x1662.free();
		x1861.free();
		x2670.free();
		x236.free();
		x517.free();
		x2770.free();
		x2540.free();
		x588.free();
		x976.free();
		x560.free();
		x1647.free();
		x807.free();
		x814.free();
		x277.free();
		x1205.free();
		x2316.free();
		x228.free();
		x2892.free();
		x2993.free();
		x2118.free();
		x218.free();
		x754.free();
		x2789.free();
		x2682.free();
		x2556.free();
		x423.free();
		x1004.free();
		x1177.free();
		x3119.free();
		x1277.free();
		x387.free();
		x292.free();
		x733.free();
		x140.free();
		x780.free();
		x154.free();
		x39.free();
		x740.free();
		x29.free();
		x237.free();
		x106.free();
		x147.free();
		x112.free();
		x213.free();
		x230.free();
		x126.free();
		x49.free();
		x36.free();
		x950.free();
		x186.free();
		x759.free();
		x777.free();
		x263.free();
		x966.free();
		x203.free();
		x99.free();
		x712.free();
		x26.free();
		x23.free();
		x89.free();
		x724.free();
		x200.free();
		x789.free();
		x223.free();
		x86.free();
		x55.free();
		x749.free();
		x92.free();
		x402.free();
		x52.free();
		x79.free();
		x309.free();
		x133.free();
		x756.free();
		x46.free();
		x17.free();
		x62.free();
		x69.free();
		x115.free();
		x72.free();
		x109.free();
		x766.free();
		x193.free();
		x220.free();
		x348.free();
		x246.free();
		x243.free();
		x783.free();
		x717.free();
		x210.free();
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

			// V_b1fc2_B <~~ X5305
			float x508, x509;
			float x510;
			float x511;
			x510 = 2;
			x511 = lrn_rate;
			x508 = x510 * x511;
			x509 = momentum;
			JCudaTensor x512;
			x512 = x491;
			x507.update(x512, x508, x509);

			// Dealloc(X5305)
			JCudaTensor x513;
			x513 = x491;
			x513.free();

			// val m20 = (i1285) => X7703[1><3][@, i1285]
			JCudaMatrix x514;
			JCudaTensor x515;
			JCudaTensor x516;
			x516 = x307;
			x515 = x516.flatten(1, new int[]{128, 4, 4});
			x514 = x515.asMatrix(1, false);

			// V_b1fc2_W <~~ X5307
			float x518, x519;
			float x520;
			float x521;
			x520 = 1;
			x521 = lrn_rate;
			x518 = x520 * x521;
			x519 = momentum;
			JCudaTensor x522;
			x522 = x478;
			x517.update(x522, x518, x519);

			// Dealloc(X5307)
			JCudaTensor x523;
			x523 = x478;
			x523.free();

			// val m18 = (i1281) => X2760[@, i1281]
			JCudaMatrix x524;
			JCudaTensor x525;
			x525 = x481;
			x524 = x525.asMatrix(1, false);

			// b1fc2_W <~~ V_b1fc2_W
			float x526, x527;
			x526 = 1;
			float x528;
			float x529;
			x528 = 1;
			float x530;
			float x531;
			float x532;
			float x533;
			x532 = 1;
			x533 = decay;
			x530 = x532 * x533;
			float x534;
			float x535;
			x534 = 1;
			x535 = lrn_rate;
			x531 = x534 * x535;
			x529 = x530 * x531;
			x527 = x528 + x529;
			JCudaTensor x536;
			x536 = x517;
			x381.update(x536, x526, x527);

			// b1fc2_B <~~ V_b1fc2_B
			float x537, x538;
			x537 = 1;
			x538 = 1;
			JCudaTensor x539;
			x539 = x507;
			x395.update(x539, x537, x538);

			// val X7644 = Convolv(1,0)(X7637,cv74_W,cv74_B)
			JCudaTensor x540;
			JCudaTensor x541, x542, x543;
			x541 = x502;
			x542 = x544;
			x543 = x545;
			x540 = x193.forward(x541, x542, x543);

			// val X5300 = Sum(m18)
			JCudaTensor x546;
			JCudaMatrix x547;
			x547 = x524;
			x546 = x547.sum();

			// val X7688 = Pooling(5,3,0,false)(X7637)
			JCudaTensor x548;
			JCudaTensor x549;
			x549 = x502;
			x548 = x263.forward(x549);

			// val X7648 = Pooling(3,1,1,true)(X7637)
			JCudaTensor x550;
			JCudaTensor x551;
			x551 = x502;
			x550 = x203.forward(x551);

			// val X5302 = m18 * m20
			JCudaTensor x552;
			JCudaMatrix x553;
			JCudaMatrix x554;
			x553 = x524;
			x554 = x514;
			x552 = x553.times(x554);

			// Dealloc(X2760)
			JCudaTensor x555;
			x555 = x481;
			x555.free();

			// val X7640 = Convolv(1,0)(X7637,cv72_W,cv72_B)
			JCudaTensor x556;
			JCudaTensor x557, x558, x559;
			x557 = x502;
			x558 = x560;
			x559 = x561;
			x556 = x210.forward(x557, x558, x559);

			// val X7638 = Convolv(1,0)(X7637,cv71_W,cv71_B)
			JCudaTensor x562;
			JCudaTensor x563, x564, x565;
			x563 = x502;
			x564 = x566;
			x565 = x567;
			x562 = x200.forward(x563, x564, x565);

			// val X2763 = X2761[1<>3] * d_ReLU()(X7703)/d_X7702
			JCudaTensor x568;
			JCudaTensor x569, x570;
			JCudaTensor x571;
			x571 = x498;
			x569 = x571.unflatten(1, new int[]{128, 4, 4});
			x570 = x307;
			x568 = x309.backward(x569, x570);

			// Dealloc(X7703)
			JCudaTensor x572;
			x572 = x307;
			x572.free();

			// val X5301 = (X5300 * loss1)
			JCudaTensor x573;
			JCudaTensor x574;
			float x575;
			x574 = x546;
			x575 = loss1;
			x573 = x574.times_i(x575);

			// val X7645 = ReLU()(X7644)
			JCudaTensor x576;
			JCudaTensor x577;
			x577 = x540;
			x576 = x223.forward(x577);

			// val X2764 = X2763 * d_Convolv(1,0)(b1cv_W)/d_X7701
			JCudaTensor x578;
			JCudaTensor x579, x580;
			x579 = x568;
			x580 = x290;
			x578 = x292.backward_data(x579, x580);

			// val X5298 = X2763 * d_Convolv(1,0)(X7701)/d_b1cv_W
			JCudaTensor x581;
			JCudaTensor x582, x583;
			x582 = x568;
			x583 = x261;
			x581 = x292.backward_filter(x582, x583);

			// val X7649 = Convolv(1,0)(X7648,cv76_W,cv76_B)
			JCudaTensor x584;
			JCudaTensor x585, x586, x587;
			x585 = x550;
			x586 = x588;
			x587 = x589;
			x584 = x220.forward(x585, x586, x587);

			// val X7641 = ReLU()(X7640)
			JCudaTensor x590;
			JCudaTensor x591;
			x591 = x556;
			x590 = x213.forward(x591);

			// val X5296 = X2763 * d_Convolv(1,0)()/d_b1cv_B
			JCudaTensor x592;
			JCudaTensor x593;
			x593 = x568;
			x592 = x292.backward_bias(x593);

			// Dealloc(X2763)
			JCudaTensor x594;
			x594 = x568;
			x594.free();

			// val X7689 = Convolv(1,0)(X7688,b2cv_W,b2cv_B)
			JCudaTensor x595;
			JCudaTensor x596, x597, x598;
			x596 = x548;
			x597 = x599;
			x598 = x600;
			x595 = x292.forward(x596, x597, x598);

			// val X5303 = (X5302 * loss1)
			JCudaTensor x601;
			JCudaTensor x602;
			float x603;
			x602 = x552;
			x603 = loss1;
			x601 = x602.times_i(x603);

			// V_b1fc1_W <~~ X5303
			float x605, x606;
			float x607;
			float x608;
			x607 = 1;
			x608 = lrn_rate;
			x605 = x607 * x608;
			x606 = momentum;
			JCudaTensor x609;
			x609 = x601;
			x604.update(x609, x605, x606);

			// Dealloc(X5303)
			JCudaTensor x610;
			x610 = x601;
			x610.free();

			// val X5297 = (X5296 * loss1)
			JCudaTensor x611;
			JCudaTensor x612;
			float x613;
			x612 = x592;
			x613 = loss1;
			x611 = x612.times_i(x613);

			// val X5299 = (X5298 * loss1)
			JCudaTensor x614;
			JCudaTensor x615;
			float x616;
			x615 = x581;
			x616 = loss1;
			x614 = x615.times_i(x616);

			// val X7642 = Convolv(1,1)(X7641,cv73_W,cv73_B)
			JCudaTensor x617;
			JCudaTensor x618, x619, x620;
			x618 = x590;
			x619 = x621;
			x620 = x622;
			x617 = x237.forward(x618, x619, x620);

			// V_b1fc1_B <~~ X5301
			float x624, x625;
			float x626;
			float x627;
			x626 = 2;
			x627 = lrn_rate;
			x624 = x626 * x627;
			x625 = momentum;
			JCudaTensor x628;
			x628 = x573;
			x623.update(x628, x624, x625);

			// Dealloc(X5301)
			JCudaTensor x629;
			x629 = x573;
			x629.free();

			// val X7646 = Convolv(1,2)(X7645,cv75_W,cv75_B)
			JCudaTensor x630;
			JCudaTensor x631, x632, x633;
			x631 = x576;
			x632 = x634;
			x633 = x635;
			x630 = x230.forward(x631, x632, x633);

			// val X2766 = X2764 * d_Pooling(5,3,0,false)(X7701,X7595)/d_X7595
			JCudaTensor x636;
			JCudaTensor x637, x638, x639;
			x637 = x578;
			x638 = x261;
			x639 = x249;
			x636 = x263.backward(x637, x638, x639);

			// Dealloc(X2764)
			JCudaTensor x640;
			x640 = x578;
			x640.free();

			// Dealloc(X7701)
			JCudaTensor x641;
			x641 = x261;
			x641.free();

			// val X7690 = ReLU()(X7689)
			JCudaTensor x642;
			JCudaTensor x643;
			x643 = x595;
			x642 = x309.forward(x643);

			// b1fc1_B <~~ V_b1fc1_B
			float x644, x645;
			x644 = 1;
			x645 = 1;
			JCudaTensor x646;
			x646 = x623;
			x333.update(x646, x644, x645);

			// b1fc1_W <~~ V_b1fc1_W
			float x647, x648;
			x647 = 1;
			float x649;
			float x650;
			x649 = 1;
			float x651;
			float x652;
			float x653;
			float x654;
			x653 = 1;
			x654 = decay;
			x651 = x653 * x654;
			float x655;
			float x656;
			x655 = 1;
			x656 = lrn_rate;
			x652 = x655 * x656;
			x650 = x651 * x652;
			x648 = x649 + x650;
			JCudaTensor x657;
			x657 = x604;
			x322.update(x657, x647, x648);

			// V_b1cv_B <~~ X5297
			float x659, x660;
			float x661;
			float x662;
			x661 = 2;
			x662 = lrn_rate;
			x659 = x661 * x662;
			x660 = momentum;
			JCudaTensor x663;
			x663 = x611;
			x658.update(x663, x659, x660);

			// Dealloc(X5297)
			JCudaTensor x664;
			x664 = x611;
			x664.free();

			// val X7639 = ReLU()(X7638)
			JCudaTensor x665;
			JCudaTensor x666;
			x666 = x562;
			x665 = x240.forward(x666);

			// V_b1cv_W <~~ X5299
			float x668, x669;
			float x670;
			float x671;
			x670 = 1;
			x671 = lrn_rate;
			x668 = x670 * x671;
			x669 = momentum;
			JCudaTensor x672;
			x672 = x614;
			x667.update(x672, x668, x669);

			// Dealloc(X5299)
			JCudaTensor x673;
			x673 = x614;
			x673.free();

			// val X7643 = ReLU()(X7642)
			JCudaTensor x674;
			JCudaTensor x675;
			x675 = x617;
			x674 = x243.forward(x675);

			// val X7650 = ReLU()(X7649)
			JCudaTensor x676;
			JCudaTensor x677;
			x677 = x584;
			x676 = x246.forward(x677);

			// val X7647 = ReLU()(X7646)
			JCudaTensor x678;
			JCudaTensor x679;
			x679 = x630;
			x678 = x246.forward(x679);

			// val X7691 = (X7690[1><3])(i | @) * (b2fc1_W)(j | @)
			JCudaTensor x680;
			JCudaMatrix x681;
			JCudaMatrix x682;
			JCudaTensor x683;
			JCudaTensor x684;
			x684 = x642;
			x683 = x684.flatten(1, new int[]{128, 4, 4});
			x681 = x683.asMatrix(1, true);
			JCudaTensor x685;
			x685 = x686;
			x682 = x685.asMatrix(1, true);
			x680 = x681.times(x682);

			// b1cv_W <~~ V_b1cv_W
			float x687, x688;
			x687 = 1;
			float x689;
			float x690;
			x689 = 1;
			float x691;
			float x692;
			float x693;
			float x694;
			x693 = 1;
			x694 = decay;
			x691 = x693 * x694;
			float x695;
			float x696;
			x695 = 1;
			x696 = lrn_rate;
			x692 = x695 * x696;
			x690 = x691 * x692;
			x688 = x689 + x690;
			JCudaTensor x697;
			x697 = x667;
			x290.update(x697, x687, x688);

			// b1cv_B <~~ V_b1cv_B
			float x698, x699;
			x698 = 1;
			x699 = 1;
			JCudaTensor x700;
			x700 = x658;
			x291.update(x700, x698, x699);

			// val X7693 = (X7691 + (i) => b2fc1_B)
			JCudaTensor x701;
			JCudaTensor x702, x703;
			x702 = x680;
			x703 = x704;
			x701 = x703.copy(128, x702);

			// val X7651 = Concat(X7639,X7643,X7647,X7650)
			JCudaTensor x705;
			JCudaTensor x706, x707, x708, x709;
			x706 = x665;
			x707 = x674;
			x708 = x678;
			x709 = x676;
			x705 = x250.forward(x706,x707,x708,x709);

			// val X7652 = Pooling(3,2,1,true)(X7651)
			JCudaTensor x710;
			JCudaTensor x711;
			x711 = x705;
			x710 = x712.forward(x711);

			// val X7694 = ReLU()(X7693)
			JCudaTensor x713;
			JCudaTensor x714;
			x714 = x701;
			x713 = x348.forward(x714);

			// val X7663 = Pooling(3,1,1,true)(X7652)
			JCudaTensor x715;
			JCudaTensor x716;
			x716 = x710;
			x715 = x717.forward(x716);

			// val X7659 = Convolv(1,0)(X7652,cv84_W,cv84_B)
			JCudaTensor x718;
			JCudaTensor x719, x720, x721;
			x719 = x710;
			x720 = x722;
			x721 = x723;
			x718 = x724.forward(x719, x720, x721);

			// val X7695 = Dropout(0.7)(X7694)
			JCudaTensor x725;
			JCudaTensor x726;
			x726 = x713;
			x725 = x361.forward(x726);

			// val X7655 = Convolv(1,0)(X7652,cv82_W,cv82_B)
			JCudaTensor x727;
			JCudaTensor x728, x729, x730;
			x728 = x710;
			x729 = x731;
			x730 = x732;
			x727 = x733.forward(x728, x729, x730);

			// val X7653 = Convolv(1,0)(X7652,cv81_W,cv81_B)
			JCudaTensor x734;
			JCudaTensor x735, x736, x737;
			x735 = x710;
			x736 = x738;
			x737 = x739;
			x734 = x740.forward(x735, x736, x737);

			// val X7696 = (X7695)(i | @) * (b2fc2_W)(j | @)
			JCudaTensor x741;
			JCudaMatrix x742;
			JCudaMatrix x743;
			JCudaTensor x744;
			x744 = x725;
			x742 = x744.asMatrix(1, true);
			JCudaTensor x745;
			x745 = x746;
			x743 = x745.asMatrix(1, true);
			x741 = x742.times(x743);

			// val X7660 = ReLU()(X7659)
			JCudaTensor x747;
			JCudaTensor x748;
			x748 = x718;
			x747 = x749.forward(x748);

			// val X7664 = Convolv(1,0)(X7663,cv86_W,cv86_B)
			JCudaTensor x750;
			JCudaTensor x751, x752, x753;
			x751 = x715;
			x752 = x754;
			x753 = x755;
			x750 = x756.forward(x751, x752, x753);

			// val X7656 = ReLU()(X7655)
			JCudaTensor x757;
			JCudaTensor x758;
			x758 = x727;
			x757 = x759.forward(x758);

			// val X7661 = Convolv(1,2)(X7660,cv85_W,cv85_B)
			JCudaTensor x760;
			JCudaTensor x761, x762, x763;
			x761 = x747;
			x762 = x764;
			x763 = x765;
			x760 = x766.forward(x761, x762, x763);

			// val X7698 = (X7696 + (i) => b2fc2_B)
			JCudaTensor x767;
			JCudaTensor x768, x769;
			x768 = x741;
			x769 = x770;
			x767 = x769.copy(128, x768);

			// val X7657 = Convolv(1,1)(X7656,cv83_W,cv83_B)
			JCudaTensor x771;
			JCudaTensor x772, x773, x774;
			x772 = x757;
			x773 = x775;
			x774 = x776;
			x771 = x777.forward(x772, x773, x774);

			// val X7662 = ReLU()(X7661)
			JCudaTensor x778;
			JCudaTensor x779;
			x779 = x760;
			x778 = x780.forward(x779);

			// val X7654 = ReLU()(X7653)
			JCudaTensor x781;
			JCudaTensor x782;
			x782 = x734;
			x781 = x783.forward(x782);

			// val X7699 = LogSoftmax()(X7698)
			JCudaTensor x784;
			JCudaTensor x785;
			x785 = x767;
			x784 = x402.forward(x785);

			// Dealloc(X7698)
			JCudaTensor x786;
			x786 = x767;
			x786.free();

			// val X7658 = ReLU()(X7657)
			JCudaTensor x787;
			JCudaTensor x788;
			x788 = x771;
			x787 = x789.forward(x788);

			// val X7665 = ReLU()(X7664)
			JCudaTensor x790;
			JCudaTensor x791;
			x791 = x750;
			x790 = x780.forward(x791);

			// val X7666 = Concat(X7654,X7658,X7662,X7665)
			JCudaTensor x792;
			JCudaTensor x794, x795, x796, x797;
			x794 = x781;
			x795 = x787;
			x796 = x778;
			x797 = x790;
			x792 = x793.forward(x794,x795,x796,x797);

			// val X2419 = X2085 * d_LogSoftmax()(X7699)/d_X7698
			JCudaTensor x798;
			JCudaTensor x799, x800;
			x799 = x404;
			x800 = x784;
			x798 = x402.backward(x799, x800);

			// val m2 = (i3163) => b2fc2_W[@, i3163]
			JCudaMatrix x801;
			JCudaTensor x802;
			x802 = x746;
			x801 = x802.asMatrix(1, false);

			// val X7667 = Convolv(1,0)(X7666,cv91_W,cv91_B)
			JCudaTensor x803;
			JCudaTensor x804, x805, x806;
			x804 = x792;
			x805 = x807;
			x806 = x808;
			x803 = x740.forward(x804, x805, x806);

			// val X7669 = Convolv(1,0)(X7666,cv92_W,cv92_B)
			JCudaTensor x809;
			JCudaTensor x810, x811, x812;
			x810 = x792;
			x811 = x813;
			x812 = x814;
			x809 = x733.forward(x810, x811, x812);

			// val X7677 = Pooling(3,1,1,true)(X7666)
			JCudaTensor x815;
			JCudaTensor x816;
			x816 = x792;
			x815 = x717.forward(x816);

			// val m27 = (i492) => X2419[@, i492]
			JCudaMatrix x817;
			JCudaTensor x818;
			x818 = x798;
			x817 = x818.asMatrix(1, false);

			// val m29 = (i496) => X7695[@, i496]
			JCudaMatrix x819;
			JCudaTensor x820;
			x820 = x725;
			x819 = x820.asMatrix(1, false);

			// val X7673 = Convolv(1,0)(X7666,cv94_W,cv94_B)
			JCudaTensor x821;
			JCudaTensor x822, x823, x824;
			x822 = x792;
			x823 = x825;
			x824 = x826;
			x821 = x724.forward(x822, x823, x824);

			// val X2428 = (X2419)(i3162 | @) * m2
			JCudaTensor x827;
			JCudaMatrix x828;
			JCudaMatrix x829;
			JCudaTensor x830;
			x830 = x798;
			x828 = x830.asMatrix(1, true);
			x829 = x801;
			x827 = x828.times(x829);

			// val X7678 = Convolv(1,0)(X7677,cv96_W,cv96_B)
			JCudaTensor x831;
			JCudaTensor x832, x833, x834;
			x832 = x815;
			x833 = x835;
			x834 = x836;
			x831 = x756.forward(x832, x833, x834);

			// val X2429 = X2428 * d_Dropout(0.7)()/d_X7694
			JCudaTensor x837;
			JCudaTensor x838;
			x838 = x827;
			x837 = x361.backward(x838);

			// Dealloc(X2428)
			JCudaTensor x839;
			x839 = x827;
			x839.free();

			// val X7670 = ReLU()(X7669)
			JCudaTensor x840;
			JCudaTensor x841;
			x841 = x809;
			x840 = x759.forward(x841);

			// val X7674 = ReLU()(X7673)
			JCudaTensor x842;
			JCudaTensor x843;
			x843 = x821;
			x842 = x749.forward(x843);

			// val X5316 = Sum(m27)
			JCudaTensor x844;
			JCudaMatrix x845;
			x845 = x817;
			x844 = x845.sum();

			// val X5318 = m27 * m29
			JCudaTensor x846;
			JCudaMatrix x847;
			JCudaMatrix x848;
			x847 = x817;
			x848 = x819;
			x846 = x847.times(x848);

			// Dealloc(X2419)
			JCudaTensor x849;
			x849 = x798;
			x849.free();

			// Dealloc(X7695)
			JCudaTensor x850;
			x850 = x725;
			x850.free();

			// val X5319 = (X5318 * loss2)
			JCudaTensor x851;
			JCudaTensor x852;
			float x853;
			x852 = x846;
			x853 = loss2;
			x851 = x852.times_i(x853);

			// val X7675 = Convolv(1,2)(X7674,cv95_W,cv95_B)
			JCudaTensor x854;
			JCudaTensor x855, x856, x857;
			x855 = x842;
			x856 = x858;
			x857 = x859;
			x854 = x766.forward(x855, x856, x857);

			// val X7671 = Convolv(1,1)(X7670,cv93_W,cv93_B)
			JCudaTensor x860;
			JCudaTensor x861, x862, x863;
			x861 = x840;
			x862 = x864;
			x863 = x865;
			x860 = x777.forward(x861, x862, x863);

			// val X5317 = (X5316 * loss2)
			JCudaTensor x866;
			JCudaTensor x867;
			float x868;
			x867 = x844;
			x868 = loss2;
			x866 = x867.times_i(x868);

			// val X2431 = X2429 * d_ReLU()(X7694)/d_X7693
			JCudaTensor x869;
			JCudaTensor x870, x871;
			x870 = x837;
			x871 = x713;
			x869 = x348.backward(x870, x871);

			// Dealloc(X7694)
			JCudaTensor x872;
			x872 = x713;
			x872.free();

			// val m3 = (i3167) => b2fc1_W[@, i3167]
			JCudaMatrix x873;
			JCudaTensor x874;
			x874 = x686;
			x873 = x874.asMatrix(1, false);

			// val m26 = (i509) => X7690[1><3][@, i509]
			JCudaMatrix x875;
			JCudaTensor x876;
			JCudaTensor x877;
			x877 = x642;
			x876 = x877.flatten(1, new int[]{128, 4, 4});
			x875 = x876.asMatrix(1, false);

			// val X7668 = ReLU()(X7667)
			JCudaTensor x878;
			JCudaTensor x879;
			x879 = x803;
			x878 = x783.forward(x879);

			// val X2432 = (X2431)(i3166 | @) * m3
			JCudaTensor x880;
			JCudaMatrix x881;
			JCudaMatrix x882;
			JCudaTensor x883;
			x883 = x869;
			x881 = x883.asMatrix(1, true);
			x882 = x873;
			x880 = x881.times(x882);

			// val m24 = (i505) => X2431[@, i505]
			JCudaMatrix x884;
			JCudaTensor x885;
			x885 = x869;
			x884 = x885.asMatrix(1, false);

			// val X7676 = ReLU()(X7675)
			JCudaTensor x886;
			JCudaTensor x887;
			x887 = x854;
			x886 = x780.forward(x887);

			// V_b2fc2_W <~~ X5319
			float x889, x890;
			float x891;
			float x892;
			x891 = 1;
			x892 = lrn_rate;
			x889 = x891 * x892;
			x890 = momentum;
			JCudaTensor x893;
			x893 = x851;
			x888.update(x893, x889, x890);

			// Dealloc(X5319)
			JCudaTensor x894;
			x894 = x851;
			x894.free();

			// val X7672 = ReLU()(X7671)
			JCudaTensor x895;
			JCudaTensor x896;
			x896 = x860;
			x895 = x789.forward(x896);

			// V_b2fc2_B <~~ X5317
			float x898, x899;
			float x900;
			float x901;
			x900 = 2;
			x901 = lrn_rate;
			x898 = x900 * x901;
			x899 = momentum;
			JCudaTensor x902;
			x902 = x866;
			x897.update(x902, x898, x899);

			// Dealloc(X5317)
			JCudaTensor x903;
			x903 = x866;
			x903.free();

			// val X7679 = ReLU()(X7678)
			JCudaTensor x904;
			JCudaTensor x905;
			x905 = x831;
			x904 = x780.forward(x905);

			// b2fc2_B <~~ V_b2fc2_B
			float x906, x907;
			x906 = 1;
			x907 = 1;
			JCudaTensor x908;
			x908 = x897;
			x770.update(x908, x906, x907);

			// b2fc2_W <~~ V_b2fc2_W
			float x909, x910;
			x909 = 1;
			float x911;
			float x912;
			x911 = 1;
			float x913;
			float x914;
			float x915;
			float x916;
			x915 = 1;
			x916 = decay;
			x913 = x915 * x916;
			float x917;
			float x918;
			x917 = 1;
			x918 = lrn_rate;
			x914 = x917 * x918;
			x912 = x913 * x914;
			x910 = x911 + x912;
			JCudaTensor x919;
			x919 = x888;
			x746.update(x919, x909, x910);

			// val X2434 = X2432[1<>3] * d_ReLU()(X7690)/d_X7689
			JCudaTensor x920;
			JCudaTensor x921, x922;
			JCudaTensor x923;
			x923 = x880;
			x921 = x923.unflatten(1, new int[]{128, 4, 4});
			x922 = x642;
			x920 = x309.backward(x921, x922);

			// val X5314 = m24 * m26
			JCudaTensor x924;
			JCudaMatrix x925;
			JCudaMatrix x926;
			x925 = x884;
			x926 = x875;
			x924 = x925.times(x926);

			// Dealloc(X7690)
			JCudaTensor x927;
			x927 = x642;
			x927.free();

			// val X5312 = Sum(m24)
			JCudaTensor x928;
			JCudaMatrix x929;
			x929 = x884;
			x928 = x929.sum();

			// Dealloc(X2431)
			JCudaTensor x930;
			x930 = x869;
			x930.free();

			// val X7680 = Concat(X7668,X7672,X7676,X7679)
			JCudaTensor x931;
			JCudaTensor x932, x933, x934, x935;
			x932 = x878;
			x933 = x895;
			x934 = x886;
			x935 = x904;
			x931 = x793.forward(x932,x933,x934,x935);

			// val X5310 = X2434 * d_Convolv(1,0)(X7688)/d_b2cv_W
			JCudaTensor x936;
			JCudaTensor x937, x938;
			x937 = x920;
			x938 = x548;
			x936 = x292.backward_filter(x937, x938);

			// val X5315 = (X5314 * loss2)
			JCudaTensor x939;
			JCudaTensor x940;
			float x941;
			x940 = x924;
			x941 = loss2;
			x939 = x940.times_i(x941);

			// val X5313 = (X5312 * loss2)
			JCudaTensor x942;
			JCudaTensor x943;
			float x944;
			x943 = x928;
			x944 = loss2;
			x942 = x943.times_i(x944);

			// val X2435 = X2434 * d_Convolv(1,0)(b2cv_W)/d_X7688
			JCudaTensor x945;
			JCudaTensor x946, x947;
			x946 = x920;
			x947 = x599;
			x945 = x292.backward_data(x946, x947);

			// val X7681 = Pooling(7,1,0,false)(X7680)
			JCudaTensor x948;
			JCudaTensor x949;
			x949 = x931;
			x948 = x950.forward(x949);

			// val X5308 = X2434 * d_Convolv(1,0)()/d_b2cv_B
			JCudaTensor x951;
			JCudaTensor x952;
			x952 = x920;
			x951 = x292.backward_bias(x952);

			// Dealloc(X2434)
			JCudaTensor x953;
			x953 = x920;
			x953.free();

			// V_b2fc1_B <~~ X5313
			float x955, x956;
			float x957;
			float x958;
			x957 = 2;
			x958 = lrn_rate;
			x955 = x957 * x958;
			x956 = momentum;
			JCudaTensor x959;
			x959 = x942;
			x954.update(x959, x955, x956);

			// Dealloc(X5313)
			JCudaTensor x960;
			x960 = x942;
			x960.free();

			// val X5309 = (X5308 * loss2)
			JCudaTensor x961;
			JCudaTensor x962;
			float x963;
			x962 = x951;
			x963 = loss2;
			x961 = x962.times_i(x963);

			// val X7682 = Dropout(0.4)(X7681)
			JCudaTensor x964;
			JCudaTensor x965;
			x965 = x948;
			x964 = x966.forward(x965);

			// val X5311 = (X5310 * loss2)
			JCudaTensor x967;
			JCudaTensor x968;
			float x969;
			x968 = x936;
			x969 = loss2;
			x967 = x968.times_i(x969);

			// val X2437 = X2435 * d_Pooling(5,3,0,false)(X7688,X7637)/d_X7637
			JCudaTensor x970;
			JCudaTensor x971, x972, x973;
			x971 = x945;
			x972 = x548;
			x973 = x502;
			x970 = x263.backward(x971, x972, x973);

			// Dealloc(X2435)
			JCudaTensor x974;
			x974 = x945;
			x974.free();

			// Dealloc(X7688)
			JCudaTensor x975;
			x975 = x548;
			x975.free();

			// V_b2fc1_W <~~ X5315
			float x977, x978;
			float x979;
			float x980;
			x979 = 1;
			x980 = lrn_rate;
			x977 = x979 * x980;
			x978 = momentum;
			JCudaTensor x981;
			x981 = x939;
			x976.update(x981, x977, x978);

			// Dealloc(X5315)
			JCudaTensor x982;
			x982 = x939;
			x982.free();

			// b2fc1_B <~~ V_b2fc1_B
			float x983, x984;
			x983 = 1;
			x984 = 1;
			JCudaTensor x985;
			x985 = x954;
			x704.update(x985, x983, x984);

			// b2fc1_W <~~ V_b2fc1_W
			float x986, x987;
			x986 = 1;
			float x988;
			float x989;
			x988 = 1;
			float x990;
			float x991;
			float x992;
			float x993;
			x992 = 1;
			x993 = decay;
			x990 = x992 * x993;
			float x994;
			float x995;
			x994 = 1;
			x995 = lrn_rate;
			x991 = x994 * x995;
			x989 = x990 * x991;
			x987 = x988 + x989;
			JCudaTensor x996;
			x996 = x976;
			x686.update(x996, x986, x987);

			// V_b2cv_W <~~ X5311
			float x998, x999;
			float x1000;
			float x1001;
			x1000 = 1;
			x1001 = lrn_rate;
			x998 = x1000 * x1001;
			x999 = momentum;
			JCudaTensor x1002;
			x1002 = x967;
			x997.update(x1002, x998, x999);

			// Dealloc(X5311)
			JCudaTensor x1003;
			x1003 = x967;
			x1003.free();

			// V_b2cv_B <~~ X5309
			float x1005, x1006;
			float x1007;
			float x1008;
			x1007 = 2;
			x1008 = lrn_rate;
			x1005 = x1007 * x1008;
			x1006 = momentum;
			JCudaTensor x1009;
			x1009 = x961;
			x1004.update(x1009, x1005, x1006);

			// Dealloc(X5309)
			JCudaTensor x1010;
			x1010 = x961;
			x1010.free();

			// val X7683 = (X7682[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x1011;
			JCudaMatrix x1012;
			JCudaMatrix x1013;
			JCudaTensor x1014;
			JCudaTensor x1015;
			x1015 = x964;
			x1014 = x1015.flatten(1, new int[]{256, 1, 1});
			x1012 = x1014.asMatrix(1, true);
			JCudaTensor x1016;
			x1016 = x1017;
			x1013 = x1016.asMatrix(1, true);
			x1011 = x1012.times(x1013);

			// b2cv_W <~~ V_b2cv_W
			float x1018, x1019;
			x1018 = 1;
			float x1020;
			float x1021;
			x1020 = 1;
			float x1022;
			float x1023;
			float x1024;
			float x1025;
			x1024 = 1;
			x1025 = decay;
			x1022 = x1024 * x1025;
			float x1026;
			float x1027;
			x1026 = 1;
			x1027 = lrn_rate;
			x1023 = x1026 * x1027;
			x1021 = x1022 * x1023;
			x1019 = x1020 + x1021;
			JCudaTensor x1028;
			x1028 = x997;
			x599.update(x1028, x1018, x1019);

			// b2cv_B <~~ V_b2cv_B
			float x1029, x1030;
			x1029 = 1;
			x1030 = 1;
			JCudaTensor x1031;
			x1031 = x1004;
			x600.update(x1031, x1029, x1030);

			// val X7685 = (X7683 + (i) => fc_B)
			JCudaTensor x1032;
			JCudaTensor x1033, x1034;
			x1033 = x1011;
			x1034 = x1035;
			x1032 = x1034.copy(128, x1033);

			// val X7686 = LogSoftmax()(X7685)
			JCudaTensor x1036;
			JCudaTensor x1037;
			x1037 = x1032;
			x1036 = x402.forward(x1037);

			// Dealloc(X7685)
			JCudaTensor x1038;
			x1038 = x1032;
			x1038.free();

			// Cost(((((0 - (X7687 . X7686)) / |128|) + (((0 - (X7687 . X7699)) / |128|) * loss2)) + (((0 - (X7687 . X7712)) / |128|) * loss1)))
			float x1039;
			float x1040;
			float x1041;
			float x1042;
			float x1043;
			float x1044;
			float x1045;
			float x1046;
			JCudaTensor x1047, x1048;
			x1047 = x9;
			x1048 = x1036;
			x1046 = x1047.dot(x1048);
			x1044 = - x1046;
			x1045 = 128;
			x1042 = x1044 / x1045;
			float x1049;
			float x1050;
			float x1051;
			float x1052;
			float x1053;
			JCudaTensor x1054, x1055;
			x1054 = x9;
			x1055 = x784;
			x1053 = x1054.dot(x1055);
			x1051 = - x1053;
			x1052 = 128;
			x1049 = x1051 / x1052;
			x1050 = loss2;
			x1043 = x1049 * x1050;
			x1040 = x1042 + x1043;
			float x1056;
			float x1057;
			float x1058;
			float x1059;
			float x1060;
			JCudaTensor x1061, x1062;
			x1061 = x9;
			x1062 = x400;
			x1060 = x1061.dot(x1062);
			x1058 = - x1060;
			x1059 = 128;
			x1056 = x1058 / x1059;
			x1057 = loss1;
			x1041 = x1056 * x1057;
			x1039 = x1040 + x1041;
			System.out.println(x5 + " " + x1039);
			if (Float.isNaN(x1039)) { System.exit(-1); }

			// Dealloc(X7699)
			JCudaTensor x1063;
			x1063 = x784;
			x1063.free();

			// Dealloc(X7712)
			JCudaTensor x1064;
			x1064 = x400;
			x1064.free();

			// Dealloc(X7687)
			JCudaTensor x1065;
			x1065 = x9;
			x1065.free();

			// val X2087 = X2085 * d_LogSoftmax()(X7686)/d_X7685
			JCudaTensor x1066;
			JCudaTensor x1067, x1068;
			x1067 = x404;
			x1068 = x1036;
			x1066 = x402.backward(x1067, x1068);

			// Dealloc(X7686)
			JCudaTensor x1069;
			x1069 = x1036;
			x1069.free();

			// Dealloc(X2085)
			JCudaTensor x1070;
			x1070 = x404;
			x1070.free();

			// val m1 = (i2995) => fc_W[@, i2995]
			JCudaMatrix x1071;
			JCudaTensor x1072;
			x1072 = x1017;
			x1071 = x1072.asMatrix(1, false);

			// val X2133 = (X2087)(i2994 | @) * m1
			JCudaTensor x1073;
			JCudaMatrix x1074;
			JCudaMatrix x1075;
			JCudaTensor x1076;
			x1076 = x1066;
			x1074 = x1076.asMatrix(1, true);
			x1075 = x1071;
			x1073 = x1074.times(x1075);

			// val m30 = (i21) => X2087[@, i21]
			JCudaMatrix x1077;
			JCudaTensor x1078;
			x1078 = x1066;
			x1077 = x1078.asMatrix(1, false);

			// val m32 = (i25) => X7682[1><3][@, i25]
			JCudaMatrix x1079;
			JCudaTensor x1080;
			JCudaTensor x1081;
			x1081 = x964;
			x1080 = x1081.flatten(1, new int[]{256, 1, 1});
			x1079 = x1080.asMatrix(1, false);

			// val X2134 = X2133[1<>3] * d_Dropout(0.4)()/d_X7681
			JCudaTensor x1082;
			JCudaTensor x1083;
			JCudaTensor x1084;
			x1084 = x1073;
			x1083 = x1084.unflatten(1, new int[]{256, 1, 1});
			x1082 = x966.backward(x1083);

			// Dealloc(X2133)
			JCudaTensor x1085;
			x1085 = x1073;
			x1085.free();

			// V_fc_W <~~ m30 * m32
			float x1087, x1088;
			float x1089;
			float x1090;
			x1089 = 1;
			x1090 = lrn_rate;
			x1087 = x1089 * x1090;
			x1088 = momentum;
			JCudaMatrix x1091;
			JCudaMatrix x1092;
			x1091 = x1077;
			x1092 = x1079;
			x1091.times(x1092, x1086, x1087, x1088);

			// Dealloc(X7682)
			JCudaTensor x1093;
			x1093 = x964;
			x1093.free();

			// V_fc_B <~~ Sum(m30)
			float x1095, x1096;
			float x1097;
			float x1098;
			x1097 = 2;
			x1098 = lrn_rate;
			x1095 = x1097 * x1098;
			x1096 = momentum;
			JCudaMatrix x1099;
			x1099 = x1077;
			x1099.sum(x1094, x1095, x1096);

			// Dealloc(X2087)
			JCudaTensor x1100;
			x1100 = x1066;
			x1100.free();

			// fc_W <~~ V_fc_W
			float x1101, x1102;
			x1101 = 1;
			float x1103;
			float x1104;
			x1103 = 1;
			float x1105;
			float x1106;
			float x1107;
			float x1108;
			x1107 = 1;
			x1108 = decay;
			x1105 = x1107 * x1108;
			float x1109;
			float x1110;
			x1109 = 1;
			x1110 = lrn_rate;
			x1106 = x1109 * x1110;
			x1104 = x1105 * x1106;
			x1102 = x1103 + x1104;
			JCudaTensor x1111;
			x1111 = x1086;
			x1017.update(x1111, x1101, x1102);

			// fc_B <~~ V_fc_B
			float x1112, x1113;
			x1112 = 1;
			x1113 = 1;
			JCudaTensor x1114;
			x1114 = x1094;
			x1035.update(x1114, x1112, x1113);

			// val X2136 = X2134 * d_Pooling(7,1,0,false)(X7681,X7680)/d_X7680
			JCudaTensor x1115;
			JCudaTensor x1116, x1117, x1118;
			x1116 = x1082;
			x1117 = x948;
			x1118 = x931;
			x1115 = x950.backward(x1116, x1117, x1118);

			// Dealloc(X2134)
			JCudaTensor x1119;
			x1119 = x1082;
			x1119.free();

			// Dealloc(X7681)
			JCudaTensor x1120;
			x1120 = x948;
			x1120.free();

			// Dealloc(X7680)
			JCudaTensor x1121;
			x1121 = x931;
			x1121.free();

			// val X2192 = Proj(X2136, X7668,X7672,X7676,X7679, 2)
			JCudaTensor x1122;
			JCudaTensor x1124;
			x1124 = x1115;
			JCudaTensor[] x1123 = x793.backward(x1124);
			x1122 = x1123[2];

			// val X2150 = Proj(X2136, X7668,X7672,X7676,X7679, 0)
			JCudaTensor x1125;
			x1125 = x1123[0];

			// val X2168 = Proj(X2136, X7668,X7672,X7676,X7679, 1)
			JCudaTensor x1126;
			x1126 = x1123[1];

			// val X2216 = Proj(X2136, X7668,X7672,X7676,X7679, 3)
			JCudaTensor x1127;
			x1127 = x1123[3];

			// Dealloc(X2136)
			JCudaTensor x1128;
			x1128 = x1115;
			x1128.free();

			// val X2173 = X2168 * d_ReLU()(X7672)/d_X7671
			JCudaTensor x1129;
			JCudaTensor x1130, x1131;
			x1130 = x1126;
			x1131 = x895;
			x1129 = x789.backward(x1130, x1131);

			// Dealloc(X7672)
			JCudaTensor x1132;
			x1132 = x895;
			x1132.free();

			// val X2220 = X2216 * d_ReLU()(X7679)/d_X7678
			JCudaTensor x1133;
			JCudaTensor x1134, x1135;
			x1134 = x1127;
			x1135 = x904;
			x1133 = x780.backward(x1134, x1135);

			// Dealloc(X7679)
			JCudaTensor x1136;
			x1136 = x904;
			x1136.free();

			// val X2197 = X2192 * d_ReLU()(X7676)/d_X7675
			JCudaTensor x1137;
			JCudaTensor x1138, x1139;
			x1138 = x1122;
			x1139 = x886;
			x1137 = x780.backward(x1138, x1139);

			// Dealloc(X7676)
			JCudaTensor x1140;
			x1140 = x886;
			x1140.free();

			// val X2153 = X2150 * d_ReLU()(X7668)/d_X7667
			JCudaTensor x1141;
			JCudaTensor x1142, x1143;
			x1142 = x1125;
			x1143 = x878;
			x1141 = x783.backward(x1142, x1143);

			// Dealloc(X7668)
			JCudaTensor x1144;
			x1144 = x878;
			x1144.free();

			// V_cv93_B <~~ X2173 * d_Convolv(1,1)()/d_cv93_B
			float x1146, x1147;
			float x1148;
			float x1149;
			x1148 = 2;
			x1149 = lrn_rate;
			x1146 = x1148 * x1149;
			x1147 = momentum;
			JCudaTensor x1150;
			x1150 = x1129;
			x777.backward_bias(x1150, x1145, x1146, x1147);

			// V_cv96_W <~~ X2220 * d_Convolv(1,0)(X7677)/d_cv96_W
			float x1152, x1153;
			float x1154;
			float x1155;
			x1154 = 1;
			x1155 = lrn_rate;
			x1152 = x1154 * x1155;
			x1153 = momentum;
			JCudaTensor x1156, x1157;
			x1156 = x1133;
			x1157 = x815;
			x756.backward_filter(x1156, x1157, x1151, x1152, x1153);

			// V_cv95_W <~~ X2197 * d_Convolv(1,2)(X7674)/d_cv95_W
			float x1159, x1160;
			float x1161;
			float x1162;
			x1161 = 1;
			x1162 = lrn_rate;
			x1159 = x1161 * x1162;
			x1160 = momentum;
			JCudaTensor x1163, x1164;
			x1163 = x1137;
			x1164 = x842;
			x766.backward_filter(x1163, x1164, x1158, x1159, x1160);

			// val X2174 = X2173 * d_Convolv(1,1)(cv93_W)/d_X7670
			JCudaTensor x1165;
			JCudaTensor x1166, x1167;
			x1166 = x1129;
			x1167 = x864;
			x1165 = x777.backward_data(x1166, x1167);

			// V_cv95_B <~~ X2197 * d_Convolv(1,2)()/d_cv95_B
			float x1169, x1170;
			float x1171;
			float x1172;
			x1171 = 2;
			x1172 = lrn_rate;
			x1169 = x1171 * x1172;
			x1170 = momentum;
			JCudaTensor x1173;
			x1173 = x1137;
			x766.backward_bias(x1173, x1168, x1169, x1170);

			// val X2154 = X2153 * d_Convolv(1,0)(cv91_W)/d_X7666
			JCudaTensor x1174;
			JCudaTensor x1175, x1176;
			x1175 = x1141;
			x1176 = x807;
			x1174 = x740.backward_data(x1175, x1176);

			// V_cv91_B <~~ X2153 * d_Convolv(1,0)()/d_cv91_B
			float x1178, x1179;
			float x1180;
			float x1181;
			x1180 = 2;
			x1181 = lrn_rate;
			x1178 = x1180 * x1181;
			x1179 = momentum;
			JCudaTensor x1182;
			x1182 = x1141;
			x740.backward_bias(x1182, x1177, x1178, x1179);

			// V_cv93_W <~~ X2173 * d_Convolv(1,1)(X7670)/d_cv93_W
			float x1184, x1185;
			float x1186;
			float x1187;
			x1186 = 1;
			x1187 = lrn_rate;
			x1184 = x1186 * x1187;
			x1185 = momentum;
			JCudaTensor x1188, x1189;
			x1188 = x1129;
			x1189 = x840;
			x777.backward_filter(x1188, x1189, x1183, x1184, x1185);

			// Dealloc(X2173)
			JCudaTensor x1190;
			x1190 = x1129;
			x1190.free();

			// V_cv96_B <~~ X2220 * d_Convolv(1,0)()/d_cv96_B
			float x1192, x1193;
			float x1194;
			float x1195;
			x1194 = 2;
			x1195 = lrn_rate;
			x1192 = x1194 * x1195;
			x1193 = momentum;
			JCudaTensor x1196;
			x1196 = x1133;
			x756.backward_bias(x1196, x1191, x1192, x1193);

			// val X2198 = X2197 * d_Convolv(1,2)(cv95_W)/d_X7674
			JCudaTensor x1197;
			JCudaTensor x1198, x1199;
			x1198 = x1137;
			x1199 = x858;
			x1197 = x766.backward_data(x1198, x1199);

			// Dealloc(X2197)
			JCudaTensor x1200;
			x1200 = x1137;
			x1200.free();

			// val X2221 = X2220 * d_Convolv(1,0)(cv96_W)/d_X7677
			JCudaTensor x1201;
			JCudaTensor x1202, x1203;
			x1202 = x1133;
			x1203 = x835;
			x1201 = x756.backward_data(x1202, x1203);

			// Dealloc(X2220)
			JCudaTensor x1204;
			x1204 = x1133;
			x1204.free();

			// V_cv91_W <~~ X2153 * d_Convolv(1,0)(X7666)/d_cv91_W
			float x1206, x1207;
			float x1208;
			float x1209;
			x1208 = 1;
			x1209 = lrn_rate;
			x1206 = x1208 * x1209;
			x1207 = momentum;
			JCudaTensor x1210, x1211;
			x1210 = x1141;
			x1211 = x792;
			x740.backward_filter(x1210, x1211, x1205, x1206, x1207);

			// Dealloc(X2153)
			JCudaTensor x1212;
			x1212 = x1141;
			x1212.free();

			// cv96_W <~~ V_cv96_W
			float x1213, x1214;
			x1213 = 1;
			float x1215;
			float x1216;
			x1215 = 1;
			float x1217;
			float x1218;
			float x1219;
			float x1220;
			x1219 = 1;
			x1220 = decay;
			x1217 = x1219 * x1220;
			float x1221;
			float x1222;
			x1221 = 1;
			x1222 = lrn_rate;
			x1218 = x1221 * x1222;
			x1216 = x1217 * x1218;
			x1214 = x1215 + x1216;
			JCudaTensor x1223;
			x1223 = x1151;
			x835.update(x1223, x1213, x1214);

			// cv91_B <~~ V_cv91_B
			float x1224, x1225;
			x1224 = 1;
			x1225 = 1;
			JCudaTensor x1226;
			x1226 = x1177;
			x808.update(x1226, x1224, x1225);

			// cv95_W <~~ V_cv95_W
			float x1227, x1228;
			x1227 = 1;
			float x1229;
			float x1230;
			x1229 = 1;
			float x1231;
			float x1232;
			float x1233;
			float x1234;
			x1233 = 1;
			x1234 = decay;
			x1231 = x1233 * x1234;
			float x1235;
			float x1236;
			x1235 = 1;
			x1236 = lrn_rate;
			x1232 = x1235 * x1236;
			x1230 = x1231 * x1232;
			x1228 = x1229 + x1230;
			JCudaTensor x1237;
			x1237 = x1158;
			x858.update(x1237, x1227, x1228);

			// cv91_W <~~ V_cv91_W
			float x1238, x1239;
			x1238 = 1;
			float x1240;
			float x1241;
			x1240 = 1;
			float x1242;
			float x1243;
			float x1244;
			float x1245;
			x1244 = 1;
			x1245 = decay;
			x1242 = x1244 * x1245;
			float x1246;
			float x1247;
			x1246 = 1;
			x1247 = lrn_rate;
			x1243 = x1246 * x1247;
			x1241 = x1242 * x1243;
			x1239 = x1240 + x1241;
			JCudaTensor x1248;
			x1248 = x1205;
			x807.update(x1248, x1238, x1239);

			// cv96_B <~~ V_cv96_B
			float x1249, x1250;
			x1249 = 1;
			x1250 = 1;
			JCudaTensor x1251;
			x1251 = x1191;
			x836.update(x1251, x1249, x1250);

			// cv95_B <~~ V_cv95_B
			float x1252, x1253;
			x1252 = 1;
			x1253 = 1;
			JCudaTensor x1254;
			x1254 = x1168;
			x859.update(x1254, x1252, x1253);

			// cv93_B <~~ V_cv93_B
			float x1255, x1256;
			x1255 = 1;
			x1256 = 1;
			JCudaTensor x1257;
			x1257 = x1145;
			x865.update(x1257, x1255, x1256);

			// cv93_W <~~ V_cv93_W
			float x1258, x1259;
			x1258 = 1;
			float x1260;
			float x1261;
			x1260 = 1;
			float x1262;
			float x1263;
			float x1264;
			float x1265;
			x1264 = 1;
			x1265 = decay;
			x1262 = x1264 * x1265;
			float x1266;
			float x1267;
			x1266 = 1;
			x1267 = lrn_rate;
			x1263 = x1266 * x1267;
			x1261 = x1262 * x1263;
			x1259 = x1260 + x1261;
			JCudaTensor x1268;
			x1268 = x1183;
			x864.update(x1268, x1258, x1259);

			// val X2176 = X2174 * d_ReLU()(X7670)/d_X7669
			JCudaTensor x1269;
			JCudaTensor x1270, x1271;
			x1270 = x1165;
			x1271 = x840;
			x1269 = x759.backward(x1270, x1271);

			// Dealloc(X7670)
			JCudaTensor x1272;
			x1272 = x840;
			x1272.free();

			// val X2200 = X2198 * d_ReLU()(X7674)/d_X7673
			JCudaTensor x1273;
			JCudaTensor x1274, x1275;
			x1274 = x1197;
			x1275 = x842;
			x1273 = x749.backward(x1274, x1275);

			// Dealloc(X7674)
			JCudaTensor x1276;
			x1276 = x842;
			x1276.free();

			// V_cv94_W <~~ X2200 * d_Convolv(1,0)(X7666)/d_cv94_W
			float x1278, x1279;
			float x1280;
			float x1281;
			x1280 = 1;
			x1281 = lrn_rate;
			x1278 = x1280 * x1281;
			x1279 = momentum;
			JCudaTensor x1282, x1283;
			x1282 = x1273;
			x1283 = x792;
			x724.backward_filter(x1282, x1283, x1277, x1278, x1279);

			// V_cv94_B <~~ X2200 * d_Convolv(1,0)()/d_cv94_B
			float x1285, x1286;
			float x1287;
			float x1288;
			x1287 = 2;
			x1288 = lrn_rate;
			x1285 = x1287 * x1288;
			x1286 = momentum;
			JCudaTensor x1289;
			x1289 = x1273;
			x724.backward_bias(x1289, x1284, x1285, x1286);

			// val X2178 = (X2154 + X2176 * d_Convolv(1,0)(cv92_W)/d_X7666)
			JCudaTensor x1290;
			JCudaTensor x1291;
			x1291 = x1174;
			JCudaTensor x1292, x1293;
			x1292 = x1269;
			x1293 = x813;
			x1290 = x733.backward_data(x1292,x1293, x1291);

			// V_cv92_W <~~ X2176 * d_Convolv(1,0)(X7666)/d_cv92_W
			float x1295, x1296;
			float x1297;
			float x1298;
			x1297 = 1;
			x1298 = lrn_rate;
			x1295 = x1297 * x1298;
			x1296 = momentum;
			JCudaTensor x1299, x1300;
			x1299 = x1269;
			x1300 = x792;
			x733.backward_filter(x1299, x1300, x1294, x1295, x1296);

			// V_cv92_B <~~ X2176 * d_Convolv(1,0)()/d_cv92_B
			float x1302, x1303;
			float x1304;
			float x1305;
			x1304 = 2;
			x1305 = lrn_rate;
			x1302 = x1304 * x1305;
			x1303 = momentum;
			JCudaTensor x1306;
			x1306 = x1269;
			x733.backward_bias(x1306, x1301, x1302, x1303);

			// Dealloc(X2176)
			JCudaTensor x1307;
			x1307 = x1269;
			x1307.free();

			// cv94_B <~~ V_cv94_B
			float x1308, x1309;
			x1308 = 1;
			x1309 = 1;
			JCudaTensor x1310;
			x1310 = x1284;
			x826.update(x1310, x1308, x1309);

			// cv92_W <~~ V_cv92_W
			float x1311, x1312;
			x1311 = 1;
			float x1313;
			float x1314;
			x1313 = 1;
			float x1315;
			float x1316;
			float x1317;
			float x1318;
			x1317 = 1;
			x1318 = decay;
			x1315 = x1317 * x1318;
			float x1319;
			float x1320;
			x1319 = 1;
			x1320 = lrn_rate;
			x1316 = x1319 * x1320;
			x1314 = x1315 * x1316;
			x1312 = x1313 + x1314;
			JCudaTensor x1321;
			x1321 = x1294;
			x813.update(x1321, x1311, x1312);

			// cv92_B <~~ V_cv92_B
			float x1322, x1323;
			x1322 = 1;
			x1323 = 1;
			JCudaTensor x1324;
			x1324 = x1301;
			x814.update(x1324, x1322, x1323);

			// val X2202 = (X2178 + X2200 * d_Convolv(1,0)(cv94_W)/d_X7666)
			JCudaTensor x1325;
			JCudaTensor x1326;
			x1326 = x1290;
			JCudaTensor x1327, x1328;
			x1327 = x1273;
			x1328 = x825;
			x1325 = x724.backward_data(x1327,x1328, x1326);

			// Dealloc(X2200)
			JCudaTensor x1329;
			x1329 = x1273;
			x1329.free();

			// cv94_W <~~ V_cv94_W
			float x1330, x1331;
			x1330 = 1;
			float x1332;
			float x1333;
			x1332 = 1;
			float x1334;
			float x1335;
			float x1336;
			float x1337;
			x1336 = 1;
			x1337 = decay;
			x1334 = x1336 * x1337;
			float x1338;
			float x1339;
			x1338 = 1;
			x1339 = lrn_rate;
			x1335 = x1338 * x1339;
			x1333 = x1334 * x1335;
			x1331 = x1332 + x1333;
			JCudaTensor x1340;
			x1340 = x1277;
			x825.update(x1340, x1330, x1331);

			// val X2224 = (X2202 + X2221 * d_Pooling(3,1,1,true)(X7677,X7666)/d_X7666)
			JCudaTensor x1341;
			JCudaTensor x1342;
			x1342 = x1325;
			JCudaTensor x1343, x1344, x1345;
			x1343 = x1201;
			x1344 = x815;
			x1345 = x792;
			x1341 = x717.backward(x1343,x1344,x1345, x1342);

			// Dealloc(X2221)
			JCudaTensor x1346;
			x1346 = x1201;
			x1346.free();

			// Dealloc(X7677)
			JCudaTensor x1347;
			x1347 = x815;
			x1347.free();

			// Dealloc(X7666)
			JCudaTensor x1348;
			x1348 = x792;
			x1348.free();

			// val X2256 = Proj(X2224, X7654,X7658,X7662,X7665, 1)
			JCudaTensor x1349;
			JCudaTensor x1351;
			x1351 = x1341;
			JCudaTensor[] x1350 = x793.backward(x1351);
			x1349 = x1350[1];

			// val X2280 = Proj(X2224, X7654,X7658,X7662,X7665, 2)
			JCudaTensor x1352;
			x1352 = x1350[2];

			// val X2304 = Proj(X2224, X7654,X7658,X7662,X7665, 3)
			JCudaTensor x1353;
			x1353 = x1350[3];

			// val X2238 = Proj(X2224, X7654,X7658,X7662,X7665, 0)
			JCudaTensor x1354;
			x1354 = x1350[0];

			// Dealloc(X2224)
			JCudaTensor x1355;
			x1355 = x1341;
			x1355.free();

			// val X2241 = X2238 * d_ReLU()(X7654)/d_X7653
			JCudaTensor x1356;
			JCudaTensor x1357, x1358;
			x1357 = x1354;
			x1358 = x781;
			x1356 = x783.backward(x1357, x1358);

			// Dealloc(X7654)
			JCudaTensor x1359;
			x1359 = x781;
			x1359.free();

			// val X2285 = X2280 * d_ReLU()(X7662)/d_X7661
			JCudaTensor x1360;
			JCudaTensor x1361, x1362;
			x1361 = x1352;
			x1362 = x778;
			x1360 = x780.backward(x1361, x1362);

			// Dealloc(X7662)
			JCudaTensor x1363;
			x1363 = x778;
			x1363.free();

			// val X2261 = X2256 * d_ReLU()(X7658)/d_X7657
			JCudaTensor x1364;
			JCudaTensor x1365, x1366;
			x1365 = x1349;
			x1366 = x787;
			x1364 = x789.backward(x1365, x1366);

			// Dealloc(X7658)
			JCudaTensor x1367;
			x1367 = x787;
			x1367.free();

			// val X2308 = X2304 * d_ReLU()(X7665)/d_X7664
			JCudaTensor x1368;
			JCudaTensor x1369, x1370;
			x1369 = x1353;
			x1370 = x790;
			x1368 = x780.backward(x1369, x1370);

			// Dealloc(X7665)
			JCudaTensor x1371;
			x1371 = x790;
			x1371.free();

			// V_cv81_B <~~ X2241 * d_Convolv(1,0)()/d_cv81_B
			float x1373, x1374;
			float x1375;
			float x1376;
			x1375 = 2;
			x1376 = lrn_rate;
			x1373 = x1375 * x1376;
			x1374 = momentum;
			JCudaTensor x1377;
			x1377 = x1356;
			x740.backward_bias(x1377, x1372, x1373, x1374);

			// val X2286 = X2285 * d_Convolv(1,2)(cv85_W)/d_X7660
			JCudaTensor x1378;
			JCudaTensor x1379, x1380;
			x1379 = x1360;
			x1380 = x764;
			x1378 = x766.backward_data(x1379, x1380);

			// V_cv83_W <~~ X2261 * d_Convolv(1,1)(X7656)/d_cv83_W
			float x1382, x1383;
			float x1384;
			float x1385;
			x1384 = 1;
			x1385 = lrn_rate;
			x1382 = x1384 * x1385;
			x1383 = momentum;
			JCudaTensor x1386, x1387;
			x1386 = x1364;
			x1387 = x757;
			x777.backward_filter(x1386, x1387, x1381, x1382, x1383);

			// val X2309 = X2308 * d_Convolv(1,0)(cv86_W)/d_X7663
			JCudaTensor x1388;
			JCudaTensor x1389, x1390;
			x1389 = x1368;
			x1390 = x754;
			x1388 = x756.backward_data(x1389, x1390);

			// V_cv85_B <~~ X2285 * d_Convolv(1,2)()/d_cv85_B
			float x1392, x1393;
			float x1394;
			float x1395;
			x1394 = 2;
			x1395 = lrn_rate;
			x1392 = x1394 * x1395;
			x1393 = momentum;
			JCudaTensor x1396;
			x1396 = x1360;
			x766.backward_bias(x1396, x1391, x1392, x1393);

			// val X2242 = X2241 * d_Convolv(1,0)(cv81_W)/d_X7652
			JCudaTensor x1397;
			JCudaTensor x1398, x1399;
			x1398 = x1356;
			x1399 = x738;
			x1397 = x740.backward_data(x1398, x1399);

			// V_cv81_W <~~ X2241 * d_Convolv(1,0)(X7652)/d_cv81_W
			float x1401, x1402;
			float x1403;
			float x1404;
			x1403 = 1;
			x1404 = lrn_rate;
			x1401 = x1403 * x1404;
			x1402 = momentum;
			JCudaTensor x1405, x1406;
			x1405 = x1356;
			x1406 = x710;
			x740.backward_filter(x1405, x1406, x1400, x1401, x1402);

			// Dealloc(X2241)
			JCudaTensor x1407;
			x1407 = x1356;
			x1407.free();

			// V_cv83_B <~~ X2261 * d_Convolv(1,1)()/d_cv83_B
			float x1409, x1410;
			float x1411;
			float x1412;
			x1411 = 2;
			x1412 = lrn_rate;
			x1409 = x1411 * x1412;
			x1410 = momentum;
			JCudaTensor x1413;
			x1413 = x1364;
			x777.backward_bias(x1413, x1408, x1409, x1410);

			// val X2262 = X2261 * d_Convolv(1,1)(cv83_W)/d_X7656
			JCudaTensor x1414;
			JCudaTensor x1415, x1416;
			x1415 = x1364;
			x1416 = x775;
			x1414 = x777.backward_data(x1415, x1416);

			// Dealloc(X2261)
			JCudaTensor x1417;
			x1417 = x1364;
			x1417.free();

			// V_cv86_B <~~ X2308 * d_Convolv(1,0)()/d_cv86_B
			float x1419, x1420;
			float x1421;
			float x1422;
			x1421 = 2;
			x1422 = lrn_rate;
			x1419 = x1421 * x1422;
			x1420 = momentum;
			JCudaTensor x1423;
			x1423 = x1368;
			x756.backward_bias(x1423, x1418, x1419, x1420);

			// V_cv85_W <~~ X2285 * d_Convolv(1,2)(X7660)/d_cv85_W
			float x1425, x1426;
			float x1427;
			float x1428;
			x1427 = 1;
			x1428 = lrn_rate;
			x1425 = x1427 * x1428;
			x1426 = momentum;
			JCudaTensor x1429, x1430;
			x1429 = x1360;
			x1430 = x747;
			x766.backward_filter(x1429, x1430, x1424, x1425, x1426);

			// Dealloc(X2285)
			JCudaTensor x1431;
			x1431 = x1360;
			x1431.free();

			// V_cv86_W <~~ X2308 * d_Convolv(1,0)(X7663)/d_cv86_W
			float x1433, x1434;
			float x1435;
			float x1436;
			x1435 = 1;
			x1436 = lrn_rate;
			x1433 = x1435 * x1436;
			x1434 = momentum;
			JCudaTensor x1437, x1438;
			x1437 = x1368;
			x1438 = x715;
			x756.backward_filter(x1437, x1438, x1432, x1433, x1434);

			// Dealloc(X2308)
			JCudaTensor x1439;
			x1439 = x1368;
			x1439.free();

			// cv81_W <~~ V_cv81_W
			float x1440, x1441;
			x1440 = 1;
			float x1442;
			float x1443;
			x1442 = 1;
			float x1444;
			float x1445;
			float x1446;
			float x1447;
			x1446 = 1;
			x1447 = decay;
			x1444 = x1446 * x1447;
			float x1448;
			float x1449;
			x1448 = 1;
			x1449 = lrn_rate;
			x1445 = x1448 * x1449;
			x1443 = x1444 * x1445;
			x1441 = x1442 + x1443;
			JCudaTensor x1450;
			x1450 = x1400;
			x738.update(x1450, x1440, x1441);

			// cv86_B <~~ V_cv86_B
			float x1451, x1452;
			x1451 = 1;
			x1452 = 1;
			JCudaTensor x1453;
			x1453 = x1418;
			x755.update(x1453, x1451, x1452);

			// cv83_W <~~ V_cv83_W
			float x1454, x1455;
			x1454 = 1;
			float x1456;
			float x1457;
			x1456 = 1;
			float x1458;
			float x1459;
			float x1460;
			float x1461;
			x1460 = 1;
			x1461 = decay;
			x1458 = x1460 * x1461;
			float x1462;
			float x1463;
			x1462 = 1;
			x1463 = lrn_rate;
			x1459 = x1462 * x1463;
			x1457 = x1458 * x1459;
			x1455 = x1456 + x1457;
			JCudaTensor x1464;
			x1464 = x1381;
			x775.update(x1464, x1454, x1455);

			// cv85_W <~~ V_cv85_W
			float x1465, x1466;
			x1465 = 1;
			float x1467;
			float x1468;
			x1467 = 1;
			float x1469;
			float x1470;
			float x1471;
			float x1472;
			x1471 = 1;
			x1472 = decay;
			x1469 = x1471 * x1472;
			float x1473;
			float x1474;
			x1473 = 1;
			x1474 = lrn_rate;
			x1470 = x1473 * x1474;
			x1468 = x1469 * x1470;
			x1466 = x1467 + x1468;
			JCudaTensor x1475;
			x1475 = x1424;
			x764.update(x1475, x1465, x1466);

			// cv83_B <~~ V_cv83_B
			float x1476, x1477;
			x1476 = 1;
			x1477 = 1;
			JCudaTensor x1478;
			x1478 = x1408;
			x776.update(x1478, x1476, x1477);

			// cv85_B <~~ V_cv85_B
			float x1479, x1480;
			x1479 = 1;
			x1480 = 1;
			JCudaTensor x1481;
			x1481 = x1391;
			x765.update(x1481, x1479, x1480);

			// cv86_W <~~ V_cv86_W
			float x1482, x1483;
			x1482 = 1;
			float x1484;
			float x1485;
			x1484 = 1;
			float x1486;
			float x1487;
			float x1488;
			float x1489;
			x1488 = 1;
			x1489 = decay;
			x1486 = x1488 * x1489;
			float x1490;
			float x1491;
			x1490 = 1;
			x1491 = lrn_rate;
			x1487 = x1490 * x1491;
			x1485 = x1486 * x1487;
			x1483 = x1484 + x1485;
			JCudaTensor x1492;
			x1492 = x1432;
			x754.update(x1492, x1482, x1483);

			// cv81_B <~~ V_cv81_B
			float x1493, x1494;
			x1493 = 1;
			x1494 = 1;
			JCudaTensor x1495;
			x1495 = x1372;
			x739.update(x1495, x1493, x1494);

			// val X2264 = X2262 * d_ReLU()(X7656)/d_X7655
			JCudaTensor x1496;
			JCudaTensor x1497, x1498;
			x1497 = x1414;
			x1498 = x757;
			x1496 = x759.backward(x1497, x1498);

			// Dealloc(X7656)
			JCudaTensor x1499;
			x1499 = x757;
			x1499.free();

			// val X2288 = X2286 * d_ReLU()(X7660)/d_X7659
			JCudaTensor x1500;
			JCudaTensor x1501, x1502;
			x1501 = x1378;
			x1502 = x747;
			x1500 = x749.backward(x1501, x1502);

			// Dealloc(X7660)
			JCudaTensor x1503;
			x1503 = x747;
			x1503.free();

			// V_cv84_W <~~ X2288 * d_Convolv(1,0)(X7652)/d_cv84_W
			float x1505, x1506;
			float x1507;
			float x1508;
			x1507 = 1;
			x1508 = lrn_rate;
			x1505 = x1507 * x1508;
			x1506 = momentum;
			JCudaTensor x1509, x1510;
			x1509 = x1500;
			x1510 = x710;
			x724.backward_filter(x1509, x1510, x1504, x1505, x1506);

			// val X2266 = (X2242 + X2264 * d_Convolv(1,0)(cv82_W)/d_X7652)
			JCudaTensor x1511;
			JCudaTensor x1512;
			x1512 = x1397;
			JCudaTensor x1513, x1514;
			x1513 = x1496;
			x1514 = x731;
			x1511 = x733.backward_data(x1513,x1514, x1512);

			// V_cv82_B <~~ X2264 * d_Convolv(1,0)()/d_cv82_B
			float x1516, x1517;
			float x1518;
			float x1519;
			x1518 = 2;
			x1519 = lrn_rate;
			x1516 = x1518 * x1519;
			x1517 = momentum;
			JCudaTensor x1520;
			x1520 = x1496;
			x733.backward_bias(x1520, x1515, x1516, x1517);

			// V_cv82_W <~~ X2264 * d_Convolv(1,0)(X7652)/d_cv82_W
			float x1522, x1523;
			float x1524;
			float x1525;
			x1524 = 1;
			x1525 = lrn_rate;
			x1522 = x1524 * x1525;
			x1523 = momentum;
			JCudaTensor x1526, x1527;
			x1526 = x1496;
			x1527 = x710;
			x733.backward_filter(x1526, x1527, x1521, x1522, x1523);

			// Dealloc(X2264)
			JCudaTensor x1528;
			x1528 = x1496;
			x1528.free();

			// V_cv84_B <~~ X2288 * d_Convolv(1,0)()/d_cv84_B
			float x1530, x1531;
			float x1532;
			float x1533;
			x1532 = 2;
			x1533 = lrn_rate;
			x1530 = x1532 * x1533;
			x1531 = momentum;
			JCudaTensor x1534;
			x1534 = x1500;
			x724.backward_bias(x1534, x1529, x1530, x1531);

			// cv82_B <~~ V_cv82_B
			float x1535, x1536;
			x1535 = 1;
			x1536 = 1;
			JCudaTensor x1537;
			x1537 = x1515;
			x732.update(x1537, x1535, x1536);

			// cv82_W <~~ V_cv82_W
			float x1538, x1539;
			x1538 = 1;
			float x1540;
			float x1541;
			x1540 = 1;
			float x1542;
			float x1543;
			float x1544;
			float x1545;
			x1544 = 1;
			x1545 = decay;
			x1542 = x1544 * x1545;
			float x1546;
			float x1547;
			x1546 = 1;
			x1547 = lrn_rate;
			x1543 = x1546 * x1547;
			x1541 = x1542 * x1543;
			x1539 = x1540 + x1541;
			JCudaTensor x1548;
			x1548 = x1521;
			x731.update(x1548, x1538, x1539);

			// cv84_B <~~ V_cv84_B
			float x1549, x1550;
			x1549 = 1;
			x1550 = 1;
			JCudaTensor x1551;
			x1551 = x1529;
			x723.update(x1551, x1549, x1550);

			// val X2290 = (X2266 + X2288 * d_Convolv(1,0)(cv84_W)/d_X7652)
			JCudaTensor x1552;
			JCudaTensor x1553;
			x1553 = x1511;
			JCudaTensor x1554, x1555;
			x1554 = x1500;
			x1555 = x722;
			x1552 = x724.backward_data(x1554,x1555, x1553);

			// Dealloc(X2288)
			JCudaTensor x1556;
			x1556 = x1500;
			x1556.free();

			// cv84_W <~~ V_cv84_W
			float x1557, x1558;
			x1557 = 1;
			float x1559;
			float x1560;
			x1559 = 1;
			float x1561;
			float x1562;
			float x1563;
			float x1564;
			x1563 = 1;
			x1564 = decay;
			x1561 = x1563 * x1564;
			float x1565;
			float x1566;
			x1565 = 1;
			x1566 = lrn_rate;
			x1562 = x1565 * x1566;
			x1560 = x1561 * x1562;
			x1558 = x1559 + x1560;
			JCudaTensor x1567;
			x1567 = x1504;
			x722.update(x1567, x1557, x1558);

			// val X2312 = (X2290 + X2309 * d_Pooling(3,1,1,true)(X7663,X7652)/d_X7652)
			JCudaTensor x1568;
			JCudaTensor x1569;
			x1569 = x1552;
			JCudaTensor x1570, x1571, x1572;
			x1570 = x1388;
			x1571 = x715;
			x1572 = x710;
			x1568 = x717.backward(x1570,x1571,x1572, x1569);

			// Dealloc(X2309)
			JCudaTensor x1573;
			x1573 = x1388;
			x1573.free();

			// Dealloc(X7663)
			JCudaTensor x1574;
			x1574 = x715;
			x1574.free();

			// val X2314 = X2312 * d_Pooling(3,2,1,true)(X7652,X7651)/d_X7651
			JCudaTensor x1575;
			JCudaTensor x1576, x1577, x1578;
			x1576 = x1568;
			x1577 = x710;
			x1578 = x705;
			x1575 = x712.backward(x1576, x1577, x1578);

			// Dealloc(X2312)
			JCudaTensor x1579;
			x1579 = x1568;
			x1579.free();

			// Dealloc(X7652)
			JCudaTensor x1580;
			x1580 = x710;
			x1580.free();

			// Dealloc(X7651)
			JCudaTensor x1581;
			x1581 = x705;
			x1581.free();

			// val X2346 = Proj(X2314, X7639,X7643,X7647,X7650, 1)
			JCudaTensor x1582;
			JCudaTensor x1584;
			x1584 = x1575;
			JCudaTensor[] x1583 = x250.backward(x1584);
			x1582 = x1583[1];

			// val X2328 = Proj(X2314, X7639,X7643,X7647,X7650, 0)
			JCudaTensor x1585;
			x1585 = x1583[0];

			// val X2394 = Proj(X2314, X7639,X7643,X7647,X7650, 3)
			JCudaTensor x1586;
			x1586 = x1583[3];

			// val X2370 = Proj(X2314, X7639,X7643,X7647,X7650, 2)
			JCudaTensor x1587;
			x1587 = x1583[2];

			// Dealloc(X2314)
			JCudaTensor x1588;
			x1588 = x1575;
			x1588.free();

			// val X2351 = X2346 * d_ReLU()(X7643)/d_X7642
			JCudaTensor x1589;
			JCudaTensor x1590, x1591;
			x1590 = x1582;
			x1591 = x674;
			x1589 = x243.backward(x1590, x1591);

			// Dealloc(X7643)
			JCudaTensor x1592;
			x1592 = x674;
			x1592.free();

			// val X2398 = X2394 * d_ReLU()(X7650)/d_X7649
			JCudaTensor x1593;
			JCudaTensor x1594, x1595;
			x1594 = x1586;
			x1595 = x676;
			x1593 = x246.backward(x1594, x1595);

			// Dealloc(X7650)
			JCudaTensor x1596;
			x1596 = x676;
			x1596.free();

			// val X2375 = X2370 * d_ReLU()(X7647)/d_X7646
			JCudaTensor x1597;
			JCudaTensor x1598, x1599;
			x1598 = x1587;
			x1599 = x678;
			x1597 = x246.backward(x1598, x1599);

			// Dealloc(X7647)
			JCudaTensor x1600;
			x1600 = x678;
			x1600.free();

			// val X2331 = X2328 * d_ReLU()(X7639)/d_X7638
			JCudaTensor x1601;
			JCudaTensor x1602, x1603;
			x1602 = x1585;
			x1603 = x665;
			x1601 = x240.backward(x1602, x1603);

			// Dealloc(X7639)
			JCudaTensor x1604;
			x1604 = x665;
			x1604.free();

			// V_cv73_W <~~ X2351 * d_Convolv(1,1)(X7641)/d_cv73_W
			float x1606, x1607;
			float x1608;
			float x1609;
			x1608 = 1;
			x1609 = lrn_rate;
			x1606 = x1608 * x1609;
			x1607 = momentum;
			JCudaTensor x1610, x1611;
			x1610 = x1589;
			x1611 = x590;
			x237.backward_filter(x1610, x1611, x1605, x1606, x1607);

			// V_cv76_W <~~ X2398 * d_Convolv(1,0)(X7648)/d_cv76_W
			float x1613, x1614;
			float x1615;
			float x1616;
			x1615 = 1;
			x1616 = lrn_rate;
			x1613 = x1615 * x1616;
			x1614 = momentum;
			JCudaTensor x1617, x1618;
			x1617 = x1593;
			x1618 = x550;
			x220.backward_filter(x1617, x1618, x1612, x1613, x1614);

			// val X2376 = X2375 * d_Convolv(1,2)(cv75_W)/d_X7645
			JCudaTensor x1619;
			JCudaTensor x1620, x1621;
			x1620 = x1597;
			x1621 = x634;
			x1619 = x230.backward_data(x1620, x1621);

			// val X2352 = X2351 * d_Convolv(1,1)(cv73_W)/d_X7641
			JCudaTensor x1622;
			JCudaTensor x1623, x1624;
			x1623 = x1589;
			x1624 = x621;
			x1622 = x237.backward_data(x1623, x1624);

			// V_cv73_B <~~ X2351 * d_Convolv(1,1)()/d_cv73_B
			float x1626, x1627;
			float x1628;
			float x1629;
			x1628 = 2;
			x1629 = lrn_rate;
			x1626 = x1628 * x1629;
			x1627 = momentum;
			JCudaTensor x1630;
			x1630 = x1589;
			x237.backward_bias(x1630, x1625, x1626, x1627);

			// Dealloc(X2351)
			JCudaTensor x1631;
			x1631 = x1589;
			x1631.free();

			// V_cv75_B <~~ X2375 * d_Convolv(1,2)()/d_cv75_B
			float x1633, x1634;
			float x1635;
			float x1636;
			x1635 = 2;
			x1636 = lrn_rate;
			x1633 = x1635 * x1636;
			x1634 = momentum;
			JCudaTensor x1637;
			x1637 = x1597;
			x230.backward_bias(x1637, x1632, x1633, x1634);

			// val X2399 = X2398 * d_Convolv(1,0)(cv76_W)/d_X7648
			JCudaTensor x1638;
			JCudaTensor x1639, x1640;
			x1639 = x1593;
			x1640 = x588;
			x1638 = x220.backward_data(x1639, x1640);

			// V_cv71_B <~~ X2331 * d_Convolv(1,0)()/d_cv71_B
			float x1642, x1643;
			float x1644;
			float x1645;
			x1644 = 2;
			x1645 = lrn_rate;
			x1642 = x1644 * x1645;
			x1643 = momentum;
			JCudaTensor x1646;
			x1646 = x1601;
			x200.backward_bias(x1646, x1641, x1642, x1643);

			// V_cv76_B <~~ X2398 * d_Convolv(1,0)()/d_cv76_B
			float x1648, x1649;
			float x1650;
			float x1651;
			x1650 = 2;
			x1651 = lrn_rate;
			x1648 = x1650 * x1651;
			x1649 = momentum;
			JCudaTensor x1652;
			x1652 = x1593;
			x220.backward_bias(x1652, x1647, x1648, x1649);

			// Dealloc(X2398)
			JCudaTensor x1653;
			x1653 = x1593;
			x1653.free();

			// V_cv75_W <~~ X2375 * d_Convolv(1,2)(X7645)/d_cv75_W
			float x1655, x1656;
			float x1657;
			float x1658;
			x1657 = 1;
			x1658 = lrn_rate;
			x1655 = x1657 * x1658;
			x1656 = momentum;
			JCudaTensor x1659, x1660;
			x1659 = x1597;
			x1660 = x576;
			x230.backward_filter(x1659, x1660, x1654, x1655, x1656);

			// Dealloc(X2375)
			JCudaTensor x1661;
			x1661 = x1597;
			x1661.free();

			// V_cv71_W <~~ X2331 * d_Convolv(1,0)(X7637)/d_cv71_W
			float x1663, x1664;
			float x1665;
			float x1666;
			x1665 = 1;
			x1666 = lrn_rate;
			x1663 = x1665 * x1666;
			x1664 = momentum;
			JCudaTensor x1667, x1668;
			x1667 = x1601;
			x1668 = x502;
			x200.backward_filter(x1667, x1668, x1662, x1663, x1664);

			// val X2332 = X2331 * d_Convolv(1,0)(cv71_W)/d_X7637
			JCudaTensor x1669;
			JCudaTensor x1670, x1671;
			x1670 = x1601;
			x1671 = x566;
			x1669 = x200.backward_data(x1670, x1671);

			// Dealloc(X2331)
			JCudaTensor x1672;
			x1672 = x1601;
			x1672.free();

			// cv71_B <~~ V_cv71_B
			float x1673, x1674;
			x1673 = 1;
			x1674 = 1;
			JCudaTensor x1675;
			x1675 = x1641;
			x567.update(x1675, x1673, x1674);

			// cv75_B <~~ V_cv75_B
			float x1676, x1677;
			x1676 = 1;
			x1677 = 1;
			JCudaTensor x1678;
			x1678 = x1632;
			x635.update(x1678, x1676, x1677);

			// cv73_B <~~ V_cv73_B
			float x1679, x1680;
			x1679 = 1;
			x1680 = 1;
			JCudaTensor x1681;
			x1681 = x1625;
			x622.update(x1681, x1679, x1680);

			// cv76_W <~~ V_cv76_W
			float x1682, x1683;
			x1682 = 1;
			float x1684;
			float x1685;
			x1684 = 1;
			float x1686;
			float x1687;
			float x1688;
			float x1689;
			x1688 = 1;
			x1689 = decay;
			x1686 = x1688 * x1689;
			float x1690;
			float x1691;
			x1690 = 1;
			x1691 = lrn_rate;
			x1687 = x1690 * x1691;
			x1685 = x1686 * x1687;
			x1683 = x1684 + x1685;
			JCudaTensor x1692;
			x1692 = x1612;
			x588.update(x1692, x1682, x1683);

			// cv73_W <~~ V_cv73_W
			float x1693, x1694;
			x1693 = 1;
			float x1695;
			float x1696;
			x1695 = 1;
			float x1697;
			float x1698;
			float x1699;
			float x1700;
			x1699 = 1;
			x1700 = decay;
			x1697 = x1699 * x1700;
			float x1701;
			float x1702;
			x1701 = 1;
			x1702 = lrn_rate;
			x1698 = x1701 * x1702;
			x1696 = x1697 * x1698;
			x1694 = x1695 + x1696;
			JCudaTensor x1703;
			x1703 = x1605;
			x621.update(x1703, x1693, x1694);

			// cv71_W <~~ V_cv71_W
			float x1704, x1705;
			x1704 = 1;
			float x1706;
			float x1707;
			x1706 = 1;
			float x1708;
			float x1709;
			float x1710;
			float x1711;
			x1710 = 1;
			x1711 = decay;
			x1708 = x1710 * x1711;
			float x1712;
			float x1713;
			x1712 = 1;
			x1713 = lrn_rate;
			x1709 = x1712 * x1713;
			x1707 = x1708 * x1709;
			x1705 = x1706 + x1707;
			JCudaTensor x1714;
			x1714 = x1662;
			x566.update(x1714, x1704, x1705);

			// cv76_B <~~ V_cv76_B
			float x1715, x1716;
			x1715 = 1;
			x1716 = 1;
			JCudaTensor x1717;
			x1717 = x1647;
			x589.update(x1717, x1715, x1716);

			// cv75_W <~~ V_cv75_W
			float x1718, x1719;
			x1718 = 1;
			float x1720;
			float x1721;
			x1720 = 1;
			float x1722;
			float x1723;
			float x1724;
			float x1725;
			x1724 = 1;
			x1725 = decay;
			x1722 = x1724 * x1725;
			float x1726;
			float x1727;
			x1726 = 1;
			x1727 = lrn_rate;
			x1723 = x1726 * x1727;
			x1721 = x1722 * x1723;
			x1719 = x1720 + x1721;
			JCudaTensor x1728;
			x1728 = x1654;
			x634.update(x1728, x1718, x1719);

			// val X2378 = X2376 * d_ReLU()(X7645)/d_X7644
			JCudaTensor x1729;
			JCudaTensor x1730, x1731;
			x1730 = x1619;
			x1731 = x576;
			x1729 = x223.backward(x1730, x1731);

			// Dealloc(X7645)
			JCudaTensor x1732;
			x1732 = x576;
			x1732.free();

			// val X2354 = X2352 * d_ReLU()(X7641)/d_X7640
			JCudaTensor x1733;
			JCudaTensor x1734, x1735;
			x1734 = x1622;
			x1735 = x590;
			x1733 = x213.backward(x1734, x1735);

			// Dealloc(X7641)
			JCudaTensor x1736;
			x1736 = x590;
			x1736.free();

			// V_cv74_W <~~ X2378 * d_Convolv(1,0)(X7637)/d_cv74_W
			float x1738, x1739;
			float x1740;
			float x1741;
			x1740 = 1;
			x1741 = lrn_rate;
			x1738 = x1740 * x1741;
			x1739 = momentum;
			JCudaTensor x1742, x1743;
			x1742 = x1729;
			x1743 = x502;
			x193.backward_filter(x1742, x1743, x1737, x1738, x1739);

			// val X2356 = (X2332 + X2354 * d_Convolv(1,0)(cv72_W)/d_X7637)
			JCudaTensor x1744;
			JCudaTensor x1745;
			x1745 = x1669;
			JCudaTensor x1746, x1747;
			x1746 = x1733;
			x1747 = x560;
			x1744 = x210.backward_data(x1746,x1747, x1745);

			// V_cv72_W <~~ X2354 * d_Convolv(1,0)(X7637)/d_cv72_W
			float x1749, x1750;
			float x1751;
			float x1752;
			x1751 = 1;
			x1752 = lrn_rate;
			x1749 = x1751 * x1752;
			x1750 = momentum;
			JCudaTensor x1753, x1754;
			x1753 = x1733;
			x1754 = x502;
			x210.backward_filter(x1753, x1754, x1748, x1749, x1750);

			// V_cv72_B <~~ X2354 * d_Convolv(1,0)()/d_cv72_B
			float x1756, x1757;
			float x1758;
			float x1759;
			x1758 = 2;
			x1759 = lrn_rate;
			x1756 = x1758 * x1759;
			x1757 = momentum;
			JCudaTensor x1760;
			x1760 = x1733;
			x210.backward_bias(x1760, x1755, x1756, x1757);

			// Dealloc(X2354)
			JCudaTensor x1761;
			x1761 = x1733;
			x1761.free();

			// V_cv74_B <~~ X2378 * d_Convolv(1,0)()/d_cv74_B
			float x1763, x1764;
			float x1765;
			float x1766;
			x1765 = 2;
			x1766 = lrn_rate;
			x1763 = x1765 * x1766;
			x1764 = momentum;
			JCudaTensor x1767;
			x1767 = x1729;
			x193.backward_bias(x1767, x1762, x1763, x1764);

			// cv72_W <~~ V_cv72_W
			float x1768, x1769;
			x1768 = 1;
			float x1770;
			float x1771;
			x1770 = 1;
			float x1772;
			float x1773;
			float x1774;
			float x1775;
			x1774 = 1;
			x1775 = decay;
			x1772 = x1774 * x1775;
			float x1776;
			float x1777;
			x1776 = 1;
			x1777 = lrn_rate;
			x1773 = x1776 * x1777;
			x1771 = x1772 * x1773;
			x1769 = x1770 + x1771;
			JCudaTensor x1778;
			x1778 = x1748;
			x560.update(x1778, x1768, x1769);

			// cv72_B <~~ V_cv72_B
			float x1779, x1780;
			x1779 = 1;
			x1780 = 1;
			JCudaTensor x1781;
			x1781 = x1755;
			x561.update(x1781, x1779, x1780);

			// cv74_B <~~ V_cv74_B
			float x1782, x1783;
			x1782 = 1;
			x1783 = 1;
			JCudaTensor x1784;
			x1784 = x1762;
			x545.update(x1784, x1782, x1783);

			// val X2380 = (X2356 + X2378 * d_Convolv(1,0)(cv74_W)/d_X7637)
			JCudaTensor x1785;
			JCudaTensor x1786;
			x1786 = x1744;
			JCudaTensor x1787, x1788;
			x1787 = x1729;
			x1788 = x544;
			x1785 = x193.backward_data(x1787,x1788, x1786);

			// Dealloc(X2378)
			JCudaTensor x1789;
			x1789 = x1729;
			x1789.free();

			// cv74_W <~~ V_cv74_W
			float x1790, x1791;
			x1790 = 1;
			float x1792;
			float x1793;
			x1792 = 1;
			float x1794;
			float x1795;
			float x1796;
			float x1797;
			x1796 = 1;
			x1797 = decay;
			x1794 = x1796 * x1797;
			float x1798;
			float x1799;
			x1798 = 1;
			x1799 = lrn_rate;
			x1795 = x1798 * x1799;
			x1793 = x1794 * x1795;
			x1791 = x1792 + x1793;
			JCudaTensor x1800;
			x1800 = x1737;
			x544.update(x1800, x1790, x1791);

			// val X2402 = (X2380 + X2399 * d_Pooling(3,1,1,true)(X7648,X7637)/d_X7637)
			JCudaTensor x1801;
			JCudaTensor x1802;
			x1802 = x1785;
			JCudaTensor x1803, x1804, x1805;
			x1803 = x1638;
			x1804 = x550;
			x1805 = x502;
			x1801 = x203.backward(x1803,x1804,x1805, x1802);

			// Dealloc(X2399)
			JCudaTensor x1806;
			x1806 = x1638;
			x1806.free();

			// Dealloc(X7648)
			JCudaTensor x1807;
			x1807 = x550;
			x1807.free();

			// Dealloc(X7637)
			JCudaTensor x1808;
			x1808 = x502;
			x1808.free();

			// val X2438 = (X2437 * loss2)
			JCudaTensor x1809;
			JCudaTensor x1810;
			float x1811;
			x1810 = x970;
			x1811 = loss2;
			x1809 = x1810.times_i(x1811);

			// val X2439 = (X2402 + X2438)
			JCudaTensor x1812;
			JCudaTensor x1813, x1814;
			x1813 = x1801;
			x1814 = x1809;
			x1812 = x1813.plus_i(x1814);

			// Dealloc(X2438)
			JCudaTensor x1815;
			x1815 = x1809;
			x1815.free();

			// val X2523 = Proj(X2439, X7625,X7629,X7633,X7636, 2)
			JCudaTensor x1816;
			JCudaTensor x1818;
			x1818 = x1812;
			JCudaTensor[] x1817 = x250.backward(x1818);
			x1816 = x1817[2];

			// val X2547 = Proj(X2439, X7625,X7629,X7633,X7636, 3)
			JCudaTensor x1819;
			x1819 = x1817[3];

			// val X2481 = Proj(X2439, X7625,X7629,X7633,X7636, 0)
			JCudaTensor x1820;
			x1820 = x1817[0];

			// val X2499 = Proj(X2439, X7625,X7629,X7633,X7636, 1)
			JCudaTensor x1821;
			x1821 = x1817[1];

			// Dealloc(X2439)
			JCudaTensor x1822;
			x1822 = x1812;
			x1822.free();

			// val X2484 = X2481 * d_ReLU()(X7625)/d_X7624
			JCudaTensor x1823;
			JCudaTensor x1824, x1825;
			x1824 = x1820;
			x1825 = x485;
			x1823 = x240.backward(x1824, x1825);

			// Dealloc(X7625)
			JCudaTensor x1826;
			x1826 = x485;
			x1826.free();

			// val X2504 = X2499 * d_ReLU()(X7629)/d_X7628
			JCudaTensor x1827;
			JCudaTensor x1828, x1829;
			x1828 = x1821;
			x1829 = x489;
			x1827 = x243.backward(x1828, x1829);

			// Dealloc(X7629)
			JCudaTensor x1830;
			x1830 = x489;
			x1830.free();

			// val X2528 = X2523 * d_ReLU()(X7633)/d_X7632
			JCudaTensor x1831;
			JCudaTensor x1832, x1833;
			x1832 = x1816;
			x1833 = x496;
			x1831 = x246.backward(x1832, x1833);

			// Dealloc(X7633)
			JCudaTensor x1834;
			x1834 = x496;
			x1834.free();

			// val X2551 = X2547 * d_ReLU()(X7636)/d_X7635
			JCudaTensor x1835;
			JCudaTensor x1836, x1837;
			x1836 = x1819;
			x1837 = x494;
			x1835 = x246.backward(x1836, x1837);

			// Dealloc(X7636)
			JCudaTensor x1838;
			x1838 = x494;
			x1838.free();

			// V_cv61_W <~~ X2484 * d_Convolv(1,0)(X7623)/d_cv61_W
			float x1840, x1841;
			float x1842;
			float x1843;
			x1842 = 1;
			x1843 = lrn_rate;
			x1840 = x1842 * x1843;
			x1841 = momentum;
			JCudaTensor x1844, x1845;
			x1844 = x1823;
			x1845 = x408;
			x200.backward_filter(x1844, x1845, x1839, x1840, x1841);

			// val X2505 = X2504 * d_Convolv(1,1)(cv63_W)/d_X7627
			JCudaTensor x1846;
			JCudaTensor x1847, x1848;
			x1847 = x1827;
			x1848 = x476;
			x1846 = x237.backward_data(x1847, x1848);

			// V_cv65_B <~~ X2528 * d_Convolv(1,2)()/d_cv65_B
			float x1850, x1851;
			float x1852;
			float x1853;
			x1852 = 2;
			x1853 = lrn_rate;
			x1850 = x1852 * x1853;
			x1851 = momentum;
			JCudaTensor x1854;
			x1854 = x1831;
			x230.backward_bias(x1854, x1849, x1850, x1851);

			// V_cv63_B <~~ X2504 * d_Convolv(1,1)()/d_cv63_B
			float x1856, x1857;
			float x1858;
			float x1859;
			x1858 = 2;
			x1859 = lrn_rate;
			x1856 = x1858 * x1859;
			x1857 = momentum;
			JCudaTensor x1860;
			x1860 = x1827;
			x237.backward_bias(x1860, x1855, x1856, x1857);

			// V_cv66_W <~~ X2551 * d_Convolv(1,0)(X7634)/d_cv66_W
			float x1862, x1863;
			float x1864;
			float x1865;
			x1864 = 1;
			x1865 = lrn_rate;
			x1862 = x1864 * x1865;
			x1863 = momentum;
			JCudaTensor x1866, x1867;
			x1866 = x1835;
			x1867 = x416;
			x220.backward_filter(x1866, x1867, x1861, x1862, x1863);

			// V_cv65_W <~~ X2528 * d_Convolv(1,2)(X7631)/d_cv65_W
			float x1869, x1870;
			float x1871;
			float x1872;
			x1871 = 1;
			x1872 = lrn_rate;
			x1869 = x1871 * x1872;
			x1870 = momentum;
			JCudaTensor x1873, x1874;
			x1873 = x1831;
			x1874 = x452;
			x230.backward_filter(x1873, x1874, x1868, x1869, x1870);

			// V_cv63_W <~~ X2504 * d_Convolv(1,1)(X7627)/d_cv63_W
			float x1876, x1877;
			float x1878;
			float x1879;
			x1878 = 1;
			x1879 = lrn_rate;
			x1876 = x1878 * x1879;
			x1877 = momentum;
			JCudaTensor x1880, x1881;
			x1880 = x1827;
			x1881 = x438;
			x237.backward_filter(x1880, x1881, x1875, x1876, x1877);

			// Dealloc(X2504)
			JCudaTensor x1882;
			x1882 = x1827;
			x1882.free();

			// V_cv66_B <~~ X2551 * d_Convolv(1,0)()/d_cv66_B
			float x1884, x1885;
			float x1886;
			float x1887;
			x1886 = 2;
			x1887 = lrn_rate;
			x1884 = x1886 * x1887;
			x1885 = momentum;
			JCudaTensor x1888;
			x1888 = x1835;
			x220.backward_bias(x1888, x1883, x1884, x1885);

			// val X2485 = X2484 * d_Convolv(1,0)(cv61_W)/d_X7623
			JCudaTensor x1889;
			JCudaTensor x1890, x1891;
			x1890 = x1823;
			x1891 = x428;
			x1889 = x200.backward_data(x1890, x1891);

			// val X2552 = X2551 * d_Convolv(1,0)(cv66_W)/d_X7634
			JCudaTensor x1892;
			JCudaTensor x1893, x1894;
			x1893 = x1835;
			x1894 = x444;
			x1892 = x220.backward_data(x1893, x1894);

			// Dealloc(X2551)
			JCudaTensor x1895;
			x1895 = x1835;
			x1895.free();

			// V_cv61_B <~~ X2484 * d_Convolv(1,0)()/d_cv61_B
			float x1897, x1898;
			float x1899;
			float x1900;
			x1899 = 2;
			x1900 = lrn_rate;
			x1897 = x1899 * x1900;
			x1898 = momentum;
			JCudaTensor x1901;
			x1901 = x1823;
			x200.backward_bias(x1901, x1896, x1897, x1898);

			// Dealloc(X2484)
			JCudaTensor x1902;
			x1902 = x1823;
			x1902.free();

			// val X2529 = X2528 * d_Convolv(1,2)(cv65_W)/d_X7631
			JCudaTensor x1903;
			JCudaTensor x1904, x1905;
			x1904 = x1831;
			x1905 = x467;
			x1903 = x230.backward_data(x1904, x1905);

			// Dealloc(X2528)
			JCudaTensor x1906;
			x1906 = x1831;
			x1906.free();

			// cv61_W <~~ V_cv61_W
			float x1907, x1908;
			x1907 = 1;
			float x1909;
			float x1910;
			x1909 = 1;
			float x1911;
			float x1912;
			float x1913;
			float x1914;
			x1913 = 1;
			x1914 = decay;
			x1911 = x1913 * x1914;
			float x1915;
			float x1916;
			x1915 = 1;
			x1916 = lrn_rate;
			x1912 = x1915 * x1916;
			x1910 = x1911 * x1912;
			x1908 = x1909 + x1910;
			JCudaTensor x1917;
			x1917 = x1839;
			x428.update(x1917, x1907, x1908);

			// cv61_B <~~ V_cv61_B
			float x1918, x1919;
			x1918 = 1;
			x1919 = 1;
			JCudaTensor x1920;
			x1920 = x1896;
			x429.update(x1920, x1918, x1919);

			// cv65_W <~~ V_cv65_W
			float x1921, x1922;
			x1921 = 1;
			float x1923;
			float x1924;
			x1923 = 1;
			float x1925;
			float x1926;
			float x1927;
			float x1928;
			x1927 = 1;
			x1928 = decay;
			x1925 = x1927 * x1928;
			float x1929;
			float x1930;
			x1929 = 1;
			x1930 = lrn_rate;
			x1926 = x1929 * x1930;
			x1924 = x1925 * x1926;
			x1922 = x1923 + x1924;
			JCudaTensor x1931;
			x1931 = x1868;
			x467.update(x1931, x1921, x1922);

			// cv63_B <~~ V_cv63_B
			float x1932, x1933;
			x1932 = 1;
			x1933 = 1;
			JCudaTensor x1934;
			x1934 = x1855;
			x477.update(x1934, x1932, x1933);

			// cv63_W <~~ V_cv63_W
			float x1935, x1936;
			x1935 = 1;
			float x1937;
			float x1938;
			x1937 = 1;
			float x1939;
			float x1940;
			float x1941;
			float x1942;
			x1941 = 1;
			x1942 = decay;
			x1939 = x1941 * x1942;
			float x1943;
			float x1944;
			x1943 = 1;
			x1944 = lrn_rate;
			x1940 = x1943 * x1944;
			x1938 = x1939 * x1940;
			x1936 = x1937 + x1938;
			JCudaTensor x1945;
			x1945 = x1875;
			x476.update(x1945, x1935, x1936);

			// cv66_W <~~ V_cv66_W
			float x1946, x1947;
			x1946 = 1;
			float x1948;
			float x1949;
			x1948 = 1;
			float x1950;
			float x1951;
			float x1952;
			float x1953;
			x1952 = 1;
			x1953 = decay;
			x1950 = x1952 * x1953;
			float x1954;
			float x1955;
			x1954 = 1;
			x1955 = lrn_rate;
			x1951 = x1954 * x1955;
			x1949 = x1950 * x1951;
			x1947 = x1948 + x1949;
			JCudaTensor x1956;
			x1956 = x1861;
			x444.update(x1956, x1946, x1947);

			// cv65_B <~~ V_cv65_B
			float x1957, x1958;
			x1957 = 1;
			x1958 = 1;
			JCudaTensor x1959;
			x1959 = x1849;
			x468.update(x1959, x1957, x1958);

			// cv66_B <~~ V_cv66_B
			float x1960, x1961;
			x1960 = 1;
			x1961 = 1;
			JCudaTensor x1962;
			x1962 = x1883;
			x445.update(x1962, x1960, x1961);

			// val X2507 = X2505 * d_ReLU()(X7627)/d_X7626
			JCudaTensor x1963;
			JCudaTensor x1964, x1965;
			x1964 = x1846;
			x1965 = x438;
			x1963 = x213.backward(x1964, x1965);

			// Dealloc(X7627)
			JCudaTensor x1966;
			x1966 = x438;
			x1966.free();

			// val X2531 = X2529 * d_ReLU()(X7631)/d_X7630
			JCudaTensor x1967;
			JCudaTensor x1968, x1969;
			x1968 = x1903;
			x1969 = x452;
			x1967 = x223.backward(x1968, x1969);

			// Dealloc(X7631)
			JCudaTensor x1970;
			x1970 = x452;
			x1970.free();

			// V_cv64_W <~~ X2531 * d_Convolv(1,0)(X7623)/d_cv64_W
			float x1972, x1973;
			float x1974;
			float x1975;
			x1974 = 1;
			x1975 = lrn_rate;
			x1972 = x1974 * x1975;
			x1973 = momentum;
			JCudaTensor x1976, x1977;
			x1976 = x1967;
			x1977 = x408;
			x193.backward_filter(x1976, x1977, x1971, x1972, x1973);

			// val X2509 = (X2485 + X2507 * d_Convolv(1,0)(cv62_W)/d_X7623)
			JCudaTensor x1978;
			JCudaTensor x1979;
			x1979 = x1889;
			JCudaTensor x1980, x1981;
			x1980 = x1963;
			x1981 = x422;
			x1978 = x210.backward_data(x1980,x1981, x1979);

			// V_cv62_W <~~ X2507 * d_Convolv(1,0)(X7623)/d_cv62_W
			float x1983, x1984;
			float x1985;
			float x1986;
			x1985 = 1;
			x1986 = lrn_rate;
			x1983 = x1985 * x1986;
			x1984 = momentum;
			JCudaTensor x1987, x1988;
			x1987 = x1963;
			x1988 = x408;
			x210.backward_filter(x1987, x1988, x1982, x1983, x1984);

			// V_cv62_B <~~ X2507 * d_Convolv(1,0)()/d_cv62_B
			float x1990, x1991;
			float x1992;
			float x1993;
			x1992 = 2;
			x1993 = lrn_rate;
			x1990 = x1992 * x1993;
			x1991 = momentum;
			JCudaTensor x1994;
			x1994 = x1963;
			x210.backward_bias(x1994, x1989, x1990, x1991);

			// Dealloc(X2507)
			JCudaTensor x1995;
			x1995 = x1963;
			x1995.free();

			// V_cv64_B <~~ X2531 * d_Convolv(1,0)()/d_cv64_B
			float x1997, x1998;
			float x1999;
			float x2000;
			x1999 = 2;
			x2000 = lrn_rate;
			x1997 = x1999 * x2000;
			x1998 = momentum;
			JCudaTensor x2001;
			x2001 = x1967;
			x193.backward_bias(x2001, x1996, x1997, x1998);

			// cv62_W <~~ V_cv62_W
			float x2002, x2003;
			x2002 = 1;
			float x2004;
			float x2005;
			x2004 = 1;
			float x2006;
			float x2007;
			float x2008;
			float x2009;
			x2008 = 1;
			x2009 = decay;
			x2006 = x2008 * x2009;
			float x2010;
			float x2011;
			x2010 = 1;
			x2011 = lrn_rate;
			x2007 = x2010 * x2011;
			x2005 = x2006 * x2007;
			x2003 = x2004 + x2005;
			JCudaTensor x2012;
			x2012 = x1982;
			x422.update(x2012, x2002, x2003);

			// cv62_B <~~ V_cv62_B
			float x2013, x2014;
			x2013 = 1;
			x2014 = 1;
			JCudaTensor x2015;
			x2015 = x1989;
			x423.update(x2015, x2013, x2014);

			// cv64_B <~~ V_cv64_B
			float x2016, x2017;
			x2016 = 1;
			x2017 = 1;
			JCudaTensor x2018;
			x2018 = x1996;
			x437.update(x2018, x2016, x2017);

			// val X2533 = (X2509 + X2531 * d_Convolv(1,0)(cv64_W)/d_X7623)
			JCudaTensor x2019;
			JCudaTensor x2020;
			x2020 = x1978;
			JCudaTensor x2021, x2022;
			x2021 = x1967;
			x2022 = x436;
			x2019 = x193.backward_data(x2021,x2022, x2020);

			// Dealloc(X2531)
			JCudaTensor x2023;
			x2023 = x1967;
			x2023.free();

			// cv64_W <~~ V_cv64_W
			float x2024, x2025;
			x2024 = 1;
			float x2026;
			float x2027;
			x2026 = 1;
			float x2028;
			float x2029;
			float x2030;
			float x2031;
			x2030 = 1;
			x2031 = decay;
			x2028 = x2030 * x2031;
			float x2032;
			float x2033;
			x2032 = 1;
			x2033 = lrn_rate;
			x2029 = x2032 * x2033;
			x2027 = x2028 * x2029;
			x2025 = x2026 + x2027;
			JCudaTensor x2034;
			x2034 = x1971;
			x436.update(x2034, x2024, x2025);

			// val X2555 = (X2533 + X2552 * d_Pooling(3,1,1,true)(X7634,X7623)/d_X7623)
			JCudaTensor x2035;
			JCudaTensor x2036;
			x2036 = x2019;
			JCudaTensor x2037, x2038, x2039;
			x2037 = x1892;
			x2038 = x416;
			x2039 = x408;
			x2035 = x203.backward(x2037,x2038,x2039, x2036);

			// Dealloc(X2552)
			JCudaTensor x2040;
			x2040 = x1892;
			x2040.free();

			// Dealloc(X7634)
			JCudaTensor x2041;
			x2041 = x416;
			x2041.free();

			// Dealloc(X7623)
			JCudaTensor x2042;
			x2042 = x408;
			x2042.free();

			// val X2635 = Proj(X2555, X7611,X7615,X7619,X7622, 3)
			JCudaTensor x2043;
			JCudaTensor x2045;
			x2045 = x2035;
			JCudaTensor[] x2044 = x250.backward(x2045);
			x2043 = x2044[3];

			// val X2587 = Proj(X2555, X7611,X7615,X7619,X7622, 1)
			JCudaTensor x2046;
			x2046 = x2044[1];

			// val X2611 = Proj(X2555, X7611,X7615,X7619,X7622, 2)
			JCudaTensor x2047;
			x2047 = x2044[2];

			// val X2569 = Proj(X2555, X7611,X7615,X7619,X7622, 0)
			JCudaTensor x2048;
			x2048 = x2044[0];

			// Dealloc(X2555)
			JCudaTensor x2049;
			x2049 = x2035;
			x2049.free();

			// val X2639 = X2635 * d_ReLU()(X7622)/d_X7621
			JCudaTensor x2050;
			JCudaTensor x2051, x2052;
			x2051 = x2043;
			x2052 = x396;
			x2050 = x246.backward(x2051, x2052);

			// Dealloc(X7622)
			JCudaTensor x2053;
			x2053 = x396;
			x2053.free();

			// val X2616 = X2611 * d_ReLU()(X7619)/d_X7618
			JCudaTensor x2054;
			JCudaTensor x2055, x2056;
			x2055 = x2047;
			x2056 = x390;
			x2054 = x246.backward(x2055, x2056);

			// Dealloc(X7619)
			JCudaTensor x2057;
			x2057 = x390;
			x2057.free();

			// val X2592 = X2587 * d_ReLU()(X7615)/d_X7614
			JCudaTensor x2058;
			JCudaTensor x2059, x2060;
			x2059 = x2046;
			x2060 = x398;
			x2058 = x243.backward(x2059, x2060);

			// Dealloc(X7615)
			JCudaTensor x2061;
			x2061 = x398;
			x2061.free();

			// val X2572 = X2569 * d_ReLU()(X7611)/d_X7610
			JCudaTensor x2062;
			JCudaTensor x2063, x2064;
			x2063 = x2048;
			x2064 = x388;
			x2062 = x240.backward(x2063, x2064);

			// Dealloc(X7611)
			JCudaTensor x2065;
			x2065 = x388;
			x2065.free();

			// val X2640 = X2639 * d_Convolv(1,0)(cv56_W)/d_X7620
			JCudaTensor x2066;
			JCudaTensor x2067, x2068;
			x2067 = x2050;
			x2068 = x368;
			x2066 = x220.backward_data(x2067, x2068);

			// V_cv55_B <~~ X2616 * d_Convolv(1,2)()/d_cv55_B
			float x2070, x2071;
			float x2072;
			float x2073;
			x2072 = 2;
			x2073 = lrn_rate;
			x2070 = x2072 * x2073;
			x2071 = momentum;
			JCudaTensor x2074;
			x2074 = x2054;
			x230.backward_bias(x2074, x2069, x2070, x2071);

			// val X2617 = X2616 * d_Convolv(1,2)(cv55_W)/d_X7617
			JCudaTensor x2075;
			JCudaTensor x2076, x2077;
			x2076 = x2054;
			x2077 = x374;
			x2075 = x230.backward_data(x2076, x2077);

			// V_cv53_B <~~ X2592 * d_Convolv(1,1)()/d_cv53_B
			float x2079, x2080;
			float x2081;
			float x2082;
			x2081 = 2;
			x2082 = lrn_rate;
			x2079 = x2081 * x2082;
			x2080 = momentum;
			JCudaTensor x2083;
			x2083 = x2058;
			x237.backward_bias(x2083, x2078, x2079, x2080);

			// val X2573 = X2572 * d_Convolv(1,0)(cv51_W)/d_X7609
			JCudaTensor x2084;
			JCudaTensor x2085, x2086;
			x2085 = x2062;
			x2086 = x355;
			x2084 = x200.backward_data(x2085, x2086);

			// val X2593 = X2592 * d_Convolv(1,1)(cv53_W)/d_X7613
			JCudaTensor x2087;
			JCudaTensor x2088, x2089;
			x2088 = x2058;
			x2089 = x386;
			x2087 = x237.backward_data(x2088, x2089);

			// V_cv51_W <~~ X2572 * d_Convolv(1,0)(X7609)/d_cv51_W
			float x2091, x2092;
			float x2093;
			float x2094;
			x2093 = 1;
			x2094 = lrn_rate;
			x2091 = x2093 * x2094;
			x2092 = momentum;
			JCudaTensor x2095, x2096;
			x2095 = x2062;
			x2096 = x325;
			x200.backward_filter(x2095, x2096, x2090, x2091, x2092);

			// V_cv51_B <~~ X2572 * d_Convolv(1,0)()/d_cv51_B
			float x2098, x2099;
			float x2100;
			float x2101;
			x2100 = 2;
			x2101 = lrn_rate;
			x2098 = x2100 * x2101;
			x2099 = momentum;
			JCudaTensor x2102;
			x2102 = x2062;
			x200.backward_bias(x2102, x2097, x2098, x2099);

			// Dealloc(X2572)
			JCudaTensor x2103;
			x2103 = x2062;
			x2103.free();

			// V_cv53_W <~~ X2592 * d_Convolv(1,1)(X7613)/d_cv53_W
			float x2105, x2106;
			float x2107;
			float x2108;
			x2107 = 1;
			x2108 = lrn_rate;
			x2105 = x2107 * x2108;
			x2106 = momentum;
			JCudaTensor x2109, x2110;
			x2109 = x2058;
			x2110 = x362;
			x237.backward_filter(x2109, x2110, x2104, x2105, x2106);

			// Dealloc(X2592)
			JCudaTensor x2111;
			x2111 = x2058;
			x2111.free();

			// V_cv56_B <~~ X2639 * d_Convolv(1,0)()/d_cv56_B
			float x2113, x2114;
			float x2115;
			float x2116;
			x2115 = 2;
			x2116 = lrn_rate;
			x2113 = x2115 * x2116;
			x2114 = momentum;
			JCudaTensor x2117;
			x2117 = x2050;
			x220.backward_bias(x2117, x2112, x2113, x2114);

			// V_cv55_W <~~ X2616 * d_Convolv(1,2)(X7617)/d_cv55_W
			float x2119, x2120;
			float x2121;
			float x2122;
			x2121 = 1;
			x2122 = lrn_rate;
			x2119 = x2121 * x2122;
			x2120 = momentum;
			JCudaTensor x2123, x2124;
			x2123 = x2054;
			x2124 = x357;
			x230.backward_filter(x2123, x2124, x2118, x2119, x2120);

			// Dealloc(X2616)
			JCudaTensor x2125;
			x2125 = x2054;
			x2125.free();

			// V_cv56_W <~~ X2639 * d_Convolv(1,0)(X7620)/d_cv56_W
			float x2127, x2128;
			float x2129;
			float x2130;
			x2129 = 1;
			x2130 = lrn_rate;
			x2127 = x2129 * x2130;
			x2128 = momentum;
			JCudaTensor x2131, x2132;
			x2131 = x2050;
			x2132 = x349;
			x220.backward_filter(x2131, x2132, x2126, x2127, x2128);

			// Dealloc(X2639)
			JCudaTensor x2133;
			x2133 = x2050;
			x2133.free();

			// cv56_W <~~ V_cv56_W
			float x2134, x2135;
			x2134 = 1;
			float x2136;
			float x2137;
			x2136 = 1;
			float x2138;
			float x2139;
			float x2140;
			float x2141;
			x2140 = 1;
			x2141 = decay;
			x2138 = x2140 * x2141;
			float x2142;
			float x2143;
			x2142 = 1;
			x2143 = lrn_rate;
			x2139 = x2142 * x2143;
			x2137 = x2138 * x2139;
			x2135 = x2136 + x2137;
			JCudaTensor x2144;
			x2144 = x2126;
			x368.update(x2144, x2134, x2135);

			// cv55_B <~~ V_cv55_B
			float x2145, x2146;
			x2145 = 1;
			x2146 = 1;
			JCudaTensor x2147;
			x2147 = x2069;
			x375.update(x2147, x2145, x2146);

			// cv53_W <~~ V_cv53_W
			float x2148, x2149;
			x2148 = 1;
			float x2150;
			float x2151;
			x2150 = 1;
			float x2152;
			float x2153;
			float x2154;
			float x2155;
			x2154 = 1;
			x2155 = decay;
			x2152 = x2154 * x2155;
			float x2156;
			float x2157;
			x2156 = 1;
			x2157 = lrn_rate;
			x2153 = x2156 * x2157;
			x2151 = x2152 * x2153;
			x2149 = x2150 + x2151;
			JCudaTensor x2158;
			x2158 = x2104;
			x386.update(x2158, x2148, x2149);

			// cv53_B <~~ V_cv53_B
			float x2159, x2160;
			x2159 = 1;
			x2160 = 1;
			JCudaTensor x2161;
			x2161 = x2078;
			x387.update(x2161, x2159, x2160);

			// cv51_B <~~ V_cv51_B
			float x2162, x2163;
			x2162 = 1;
			x2163 = 1;
			JCudaTensor x2164;
			x2164 = x2097;
			x356.update(x2164, x2162, x2163);

			// cv51_W <~~ V_cv51_W
			float x2165, x2166;
			x2165 = 1;
			float x2167;
			float x2168;
			x2167 = 1;
			float x2169;
			float x2170;
			float x2171;
			float x2172;
			x2171 = 1;
			x2172 = decay;
			x2169 = x2171 * x2172;
			float x2173;
			float x2174;
			x2173 = 1;
			x2174 = lrn_rate;
			x2170 = x2173 * x2174;
			x2168 = x2169 * x2170;
			x2166 = x2167 + x2168;
			JCudaTensor x2175;
			x2175 = x2090;
			x355.update(x2175, x2165, x2166);

			// cv55_W <~~ V_cv55_W
			float x2176, x2177;
			x2176 = 1;
			float x2178;
			float x2179;
			x2178 = 1;
			float x2180;
			float x2181;
			float x2182;
			float x2183;
			x2182 = 1;
			x2183 = decay;
			x2180 = x2182 * x2183;
			float x2184;
			float x2185;
			x2184 = 1;
			x2185 = lrn_rate;
			x2181 = x2184 * x2185;
			x2179 = x2180 * x2181;
			x2177 = x2178 + x2179;
			JCudaTensor x2186;
			x2186 = x2118;
			x374.update(x2186, x2176, x2177);

			// cv56_B <~~ V_cv56_B
			float x2187, x2188;
			x2187 = 1;
			x2188 = 1;
			JCudaTensor x2189;
			x2189 = x2112;
			x369.update(x2189, x2187, x2188);

			// val X2595 = X2593 * d_ReLU()(X7613)/d_X7612
			JCudaTensor x2190;
			JCudaTensor x2191, x2192;
			x2191 = x2087;
			x2192 = x362;
			x2190 = x213.backward(x2191, x2192);

			// Dealloc(X7613)
			JCudaTensor x2193;
			x2193 = x362;
			x2193.free();

			// val X2619 = X2617 * d_ReLU()(X7617)/d_X7616
			JCudaTensor x2194;
			JCudaTensor x2195, x2196;
			x2195 = x2075;
			x2196 = x357;
			x2194 = x223.backward(x2195, x2196);

			// Dealloc(X7617)
			JCudaTensor x2197;
			x2197 = x357;
			x2197.free();

			// V_cv54_W <~~ X2619 * d_Convolv(1,0)(X7609)/d_cv54_W
			float x2199, x2200;
			float x2201;
			float x2202;
			x2201 = 1;
			x2202 = lrn_rate;
			x2199 = x2201 * x2202;
			x2200 = momentum;
			JCudaTensor x2203, x2204;
			x2203 = x2194;
			x2204 = x325;
			x193.backward_filter(x2203, x2204, x2198, x2199, x2200);

			// V_cv52_B <~~ X2595 * d_Convolv(1,0)()/d_cv52_B
			float x2206, x2207;
			float x2208;
			float x2209;
			x2208 = 2;
			x2209 = lrn_rate;
			x2206 = x2208 * x2209;
			x2207 = momentum;
			JCudaTensor x2210;
			x2210 = x2190;
			x210.backward_bias(x2210, x2205, x2206, x2207);

			// V_cv54_B <~~ X2619 * d_Convolv(1,0)()/d_cv54_B
			float x2212, x2213;
			float x2214;
			float x2215;
			x2214 = 2;
			x2215 = lrn_rate;
			x2212 = x2214 * x2215;
			x2213 = momentum;
			JCudaTensor x2216;
			x2216 = x2194;
			x193.backward_bias(x2216, x2211, x2212, x2213);

			// val X2597 = (X2573 + X2595 * d_Convolv(1,0)(cv52_W)/d_X7609)
			JCudaTensor x2217;
			JCudaTensor x2218;
			x2218 = x2084;
			JCudaTensor x2219, x2220;
			x2219 = x2190;
			x2220 = x344;
			x2217 = x210.backward_data(x2219,x2220, x2218);

			// V_cv52_W <~~ X2595 * d_Convolv(1,0)(X7609)/d_cv52_W
			float x2222, x2223;
			float x2224;
			float x2225;
			x2224 = 1;
			x2225 = lrn_rate;
			x2222 = x2224 * x2225;
			x2223 = momentum;
			JCudaTensor x2226, x2227;
			x2226 = x2190;
			x2227 = x325;
			x210.backward_filter(x2226, x2227, x2221, x2222, x2223);

			// Dealloc(X2595)
			JCudaTensor x2228;
			x2228 = x2190;
			x2228.free();

			// cv52_B <~~ V_cv52_B
			float x2229, x2230;
			x2229 = 1;
			x2230 = 1;
			JCudaTensor x2231;
			x2231 = x2205;
			x345.update(x2231, x2229, x2230);

			// cv54_B <~~ V_cv54_B
			float x2232, x2233;
			x2232 = 1;
			x2233 = 1;
			JCudaTensor x2234;
			x2234 = x2211;
			x339.update(x2234, x2232, x2233);

			// cv52_W <~~ V_cv52_W
			float x2235, x2236;
			x2235 = 1;
			float x2237;
			float x2238;
			x2237 = 1;
			float x2239;
			float x2240;
			float x2241;
			float x2242;
			x2241 = 1;
			x2242 = decay;
			x2239 = x2241 * x2242;
			float x2243;
			float x2244;
			x2243 = 1;
			x2244 = lrn_rate;
			x2240 = x2243 * x2244;
			x2238 = x2239 * x2240;
			x2236 = x2237 + x2238;
			JCudaTensor x2245;
			x2245 = x2221;
			x344.update(x2245, x2235, x2236);

			// val X2621 = (X2597 + X2619 * d_Convolv(1,0)(cv54_W)/d_X7609)
			JCudaTensor x2246;
			JCudaTensor x2247;
			x2247 = x2217;
			JCudaTensor x2248, x2249;
			x2248 = x2194;
			x2249 = x338;
			x2246 = x193.backward_data(x2248,x2249, x2247);

			// Dealloc(X2619)
			JCudaTensor x2250;
			x2250 = x2194;
			x2250.free();

			// cv54_W <~~ V_cv54_W
			float x2251, x2252;
			x2251 = 1;
			float x2253;
			float x2254;
			x2253 = 1;
			float x2255;
			float x2256;
			float x2257;
			float x2258;
			x2257 = 1;
			x2258 = decay;
			x2255 = x2257 * x2258;
			float x2259;
			float x2260;
			x2259 = 1;
			x2260 = lrn_rate;
			x2256 = x2259 * x2260;
			x2254 = x2255 * x2256;
			x2252 = x2253 + x2254;
			JCudaTensor x2261;
			x2261 = x2198;
			x338.update(x2261, x2251, x2252);

			// val X2643 = (X2621 + X2640 * d_Pooling(3,1,1,true)(X7620,X7609)/d_X7609)
			JCudaTensor x2262;
			JCudaTensor x2263;
			x2263 = x2246;
			JCudaTensor x2264, x2265, x2266;
			x2264 = x2066;
			x2265 = x349;
			x2266 = x325;
			x2262 = x203.backward(x2264,x2265,x2266, x2263);

			// Dealloc(X2640)
			JCudaTensor x2267;
			x2267 = x2066;
			x2267.free();

			// Dealloc(X7620)
			JCudaTensor x2268;
			x2268 = x349;
			x2268.free();

			// Dealloc(X7609)
			JCudaTensor x2269;
			x2269 = x325;
			x2269.free();

			// val X2675 = Proj(X2643, X7597,X7601,X7605,X7608, 1)
			JCudaTensor x2270;
			JCudaTensor x2272;
			x2272 = x2262;
			JCudaTensor[] x2271 = x250.backward(x2272);
			x2270 = x2271[1];

			// val X2699 = Proj(X2643, X7597,X7601,X7605,X7608, 2)
			JCudaTensor x2273;
			x2273 = x2271[2];

			// val X2723 = Proj(X2643, X7597,X7601,X7605,X7608, 3)
			JCudaTensor x2274;
			x2274 = x2271[3];

			// val X2657 = Proj(X2643, X7597,X7601,X7605,X7608, 0)
			JCudaTensor x2275;
			x2275 = x2271[0];

			// Dealloc(X2643)
			JCudaTensor x2276;
			x2276 = x2262;
			x2276.free();

			// val X2680 = X2675 * d_ReLU()(X7601)/d_X7600
			JCudaTensor x2277;
			JCudaTensor x2278, x2279;
			x2278 = x2270;
			x2279 = x312;
			x2277 = x243.backward(x2278, x2279);

			// Dealloc(X7601)
			JCudaTensor x2280;
			x2280 = x312;
			x2280.free();

			// val X2727 = X2723 * d_ReLU()(X7608)/d_X7607
			JCudaTensor x2281;
			JCudaTensor x2282, x2283;
			x2282 = x2274;
			x2283 = x314;
			x2281 = x246.backward(x2282, x2283);

			// Dealloc(X7608)
			JCudaTensor x2284;
			x2284 = x314;
			x2284.free();

			// val X2660 = X2657 * d_ReLU()(X7597)/d_X7596
			JCudaTensor x2285;
			JCudaTensor x2286, x2287;
			x2286 = x2275;
			x2287 = x323;
			x2285 = x240.backward(x2286, x2287);

			// Dealloc(X7597)
			JCudaTensor x2288;
			x2288 = x323;
			x2288.free();

			// val X2704 = X2699 * d_ReLU()(X7605)/d_X7604
			JCudaTensor x2289;
			JCudaTensor x2290, x2291;
			x2290 = x2273;
			x2291 = x310;
			x2289 = x246.backward(x2290, x2291);

			// Dealloc(X7605)
			JCudaTensor x2292;
			x2292 = x310;
			x2292.free();

			// val X2681 = X2680 * d_Convolv(1,1)(cv43_W)/d_X7599
			JCudaTensor x2293;
			JCudaTensor x2294, x2295;
			x2294 = x2277;
			x2295 = x305;
			x2293 = x237.backward_data(x2294, x2295);

			// V_cv43_B <~~ X2680 * d_Convolv(1,1)()/d_cv43_B
			float x2297, x2298;
			float x2299;
			float x2300;
			x2299 = 2;
			x2300 = lrn_rate;
			x2297 = x2299 * x2300;
			x2298 = momentum;
			JCudaTensor x2301;
			x2301 = x2277;
			x237.backward_bias(x2301, x2296, x2297, x2298);

			// V_cv43_W <~~ X2680 * d_Convolv(1,1)(X7599)/d_cv43_W
			float x2303, x2304;
			float x2305;
			float x2306;
			x2305 = 1;
			x2306 = lrn_rate;
			x2303 = x2305 * x2306;
			x2304 = momentum;
			JCudaTensor x2307, x2308;
			x2307 = x2277;
			x2308 = x293;
			x237.backward_filter(x2307, x2308, x2302, x2303, x2304);

			// Dealloc(X2680)
			JCudaTensor x2309;
			x2309 = x2277;
			x2309.free();

			// val X2728 = X2727 * d_Convolv(1,0)(cv46_W)/d_X7606
			JCudaTensor x2310;
			JCudaTensor x2311, x2312;
			x2311 = x2281;
			x2312 = x284;
			x2310 = x220.backward_data(x2311, x2312);

			// val X2661 = X2660 * d_Convolv(1,0)(cv41_W)/d_X7595
			JCudaTensor x2313;
			JCudaTensor x2314, x2315;
			x2314 = x2285;
			x2315 = x270;
			x2313 = x200.backward_data(x2314, x2315);

			// V_cv46_B <~~ X2727 * d_Convolv(1,0)()/d_cv46_B
			float x2317, x2318;
			float x2319;
			float x2320;
			x2319 = 2;
			x2320 = lrn_rate;
			x2317 = x2319 * x2320;
			x2318 = momentum;
			JCudaTensor x2321;
			x2321 = x2281;
			x220.backward_bias(x2321, x2316, x2317, x2318);

			// V_cv41_W <~~ X2660 * d_Convolv(1,0)(X7595)/d_cv41_W
			float x2323, x2324;
			float x2325;
			float x2326;
			x2325 = 1;
			x2326 = lrn_rate;
			x2323 = x2325 * x2326;
			x2324 = momentum;
			JCudaTensor x2327, x2328;
			x2327 = x2285;
			x2328 = x249;
			x200.backward_filter(x2327, x2328, x2322, x2323, x2324);

			// V_cv46_W <~~ X2727 * d_Convolv(1,0)(X7606)/d_cv46_W
			float x2330, x2331;
			float x2332;
			float x2333;
			x2332 = 1;
			x2333 = lrn_rate;
			x2330 = x2332 * x2333;
			x2331 = momentum;
			JCudaTensor x2334, x2335;
			x2334 = x2281;
			x2335 = x264;
			x220.backward_filter(x2334, x2335, x2329, x2330, x2331);

			// Dealloc(X2727)
			JCudaTensor x2336;
			x2336 = x2281;
			x2336.free();

			// V_cv45_B <~~ X2704 * d_Convolv(1,2)()/d_cv45_B
			float x2338, x2339;
			float x2340;
			float x2341;
			x2340 = 2;
			x2341 = lrn_rate;
			x2338 = x2340 * x2341;
			x2339 = momentum;
			JCudaTensor x2342;
			x2342 = x2289;
			x230.backward_bias(x2342, x2337, x2338, x2339);

			// val X2705 = X2704 * d_Convolv(1,2)(cv45_W)/d_X7603
			JCudaTensor x2343;
			JCudaTensor x2344, x2345;
			x2344 = x2289;
			x2345 = x299;
			x2343 = x230.backward_data(x2344, x2345);

			// V_cv41_B <~~ X2660 * d_Convolv(1,0)()/d_cv41_B
			float x2347, x2348;
			float x2349;
			float x2350;
			x2349 = 2;
			x2350 = lrn_rate;
			x2347 = x2349 * x2350;
			x2348 = momentum;
			JCudaTensor x2351;
			x2351 = x2285;
			x200.backward_bias(x2351, x2346, x2347, x2348);

			// Dealloc(X2660)
			JCudaTensor x2352;
			x2352 = x2285;
			x2352.free();

			// V_cv45_W <~~ X2704 * d_Convolv(1,2)(X7603)/d_cv45_W
			float x2354, x2355;
			float x2356;
			float x2357;
			x2356 = 1;
			x2357 = lrn_rate;
			x2354 = x2356 * x2357;
			x2355 = momentum;
			JCudaTensor x2358, x2359;
			x2358 = x2289;
			x2359 = x278;
			x230.backward_filter(x2358, x2359, x2353, x2354, x2355);

			// Dealloc(X2704)
			JCudaTensor x2360;
			x2360 = x2289;
			x2360.free();

			// cv41_W <~~ V_cv41_W
			float x2361, x2362;
			x2361 = 1;
			float x2363;
			float x2364;
			x2363 = 1;
			float x2365;
			float x2366;
			float x2367;
			float x2368;
			x2367 = 1;
			x2368 = decay;
			x2365 = x2367 * x2368;
			float x2369;
			float x2370;
			x2369 = 1;
			x2370 = lrn_rate;
			x2366 = x2369 * x2370;
			x2364 = x2365 * x2366;
			x2362 = x2363 + x2364;
			JCudaTensor x2371;
			x2371 = x2322;
			x270.update(x2371, x2361, x2362);

			// cv43_B <~~ V_cv43_B
			float x2372, x2373;
			x2372 = 1;
			x2373 = 1;
			JCudaTensor x2374;
			x2374 = x2296;
			x306.update(x2374, x2372, x2373);

			// cv41_B <~~ V_cv41_B
			float x2375, x2376;
			x2375 = 1;
			x2376 = 1;
			JCudaTensor x2377;
			x2377 = x2346;
			x271.update(x2377, x2375, x2376);

			// cv45_W <~~ V_cv45_W
			float x2378, x2379;
			x2378 = 1;
			float x2380;
			float x2381;
			x2380 = 1;
			float x2382;
			float x2383;
			float x2384;
			float x2385;
			x2384 = 1;
			x2385 = decay;
			x2382 = x2384 * x2385;
			float x2386;
			float x2387;
			x2386 = 1;
			x2387 = lrn_rate;
			x2383 = x2386 * x2387;
			x2381 = x2382 * x2383;
			x2379 = x2380 + x2381;
			JCudaTensor x2388;
			x2388 = x2353;
			x299.update(x2388, x2378, x2379);

			// cv46_B <~~ V_cv46_B
			float x2389, x2390;
			x2389 = 1;
			x2390 = 1;
			JCudaTensor x2391;
			x2391 = x2316;
			x285.update(x2391, x2389, x2390);

			// cv43_W <~~ V_cv43_W
			float x2392, x2393;
			x2392 = 1;
			float x2394;
			float x2395;
			x2394 = 1;
			float x2396;
			float x2397;
			float x2398;
			float x2399;
			x2398 = 1;
			x2399 = decay;
			x2396 = x2398 * x2399;
			float x2400;
			float x2401;
			x2400 = 1;
			x2401 = lrn_rate;
			x2397 = x2400 * x2401;
			x2395 = x2396 * x2397;
			x2393 = x2394 + x2395;
			JCudaTensor x2402;
			x2402 = x2302;
			x305.update(x2402, x2392, x2393);

			// cv45_B <~~ V_cv45_B
			float x2403, x2404;
			x2403 = 1;
			x2404 = 1;
			JCudaTensor x2405;
			x2405 = x2337;
			x300.update(x2405, x2403, x2404);

			// cv46_W <~~ V_cv46_W
			float x2406, x2407;
			x2406 = 1;
			float x2408;
			float x2409;
			x2408 = 1;
			float x2410;
			float x2411;
			float x2412;
			float x2413;
			x2412 = 1;
			x2413 = decay;
			x2410 = x2412 * x2413;
			float x2414;
			float x2415;
			x2414 = 1;
			x2415 = lrn_rate;
			x2411 = x2414 * x2415;
			x2409 = x2410 * x2411;
			x2407 = x2408 + x2409;
			JCudaTensor x2416;
			x2416 = x2329;
			x284.update(x2416, x2406, x2407);

			// val X2683 = X2681 * d_ReLU()(X7599)/d_X7598
			JCudaTensor x2417;
			JCudaTensor x2418, x2419;
			x2418 = x2293;
			x2419 = x293;
			x2417 = x213.backward(x2418, x2419);

			// Dealloc(X7599)
			JCudaTensor x2420;
			x2420 = x293;
			x2420.free();

			// val X2707 = X2705 * d_ReLU()(X7603)/d_X7602
			JCudaTensor x2421;
			JCudaTensor x2422, x2423;
			x2422 = x2343;
			x2423 = x278;
			x2421 = x223.backward(x2422, x2423);

			// Dealloc(X7603)
			JCudaTensor x2424;
			x2424 = x278;
			x2424.free();

			// V_cv44_W <~~ X2707 * d_Convolv(1,0)(X7595)/d_cv44_W
			float x2426, x2427;
			float x2428;
			float x2429;
			x2428 = 1;
			x2429 = lrn_rate;
			x2426 = x2428 * x2429;
			x2427 = momentum;
			JCudaTensor x2430, x2431;
			x2430 = x2421;
			x2431 = x249;
			x193.backward_filter(x2430, x2431, x2425, x2426, x2427);

			// val X2685 = (X2661 + X2683 * d_Convolv(1,0)(cv42_W)/d_X7595)
			JCudaTensor x2432;
			JCudaTensor x2433;
			x2433 = x2313;
			JCudaTensor x2434, x2435;
			x2434 = x2417;
			x2435 = x259;
			x2432 = x210.backward_data(x2434,x2435, x2433);

			// V_cv42_B <~~ X2683 * d_Convolv(1,0)()/d_cv42_B
			float x2437, x2438;
			float x2439;
			float x2440;
			x2439 = 2;
			x2440 = lrn_rate;
			x2437 = x2439 * x2440;
			x2438 = momentum;
			JCudaTensor x2441;
			x2441 = x2417;
			x210.backward_bias(x2441, x2436, x2437, x2438);

			// V_cv42_W <~~ X2683 * d_Convolv(1,0)(X7595)/d_cv42_W
			float x2443, x2444;
			float x2445;
			float x2446;
			x2445 = 1;
			x2446 = lrn_rate;
			x2443 = x2445 * x2446;
			x2444 = momentum;
			JCudaTensor x2447, x2448;
			x2447 = x2417;
			x2448 = x249;
			x210.backward_filter(x2447, x2448, x2442, x2443, x2444);

			// Dealloc(X2683)
			JCudaTensor x2449;
			x2449 = x2417;
			x2449.free();

			// V_cv44_B <~~ X2707 * d_Convolv(1,0)()/d_cv44_B
			float x2451, x2452;
			float x2453;
			float x2454;
			x2453 = 2;
			x2454 = lrn_rate;
			x2451 = x2453 * x2454;
			x2452 = momentum;
			JCudaTensor x2455;
			x2455 = x2421;
			x193.backward_bias(x2455, x2450, x2451, x2452);

			// cv42_B <~~ V_cv42_B
			float x2456, x2457;
			x2456 = 1;
			x2457 = 1;
			JCudaTensor x2458;
			x2458 = x2436;
			x260.update(x2458, x2456, x2457);

			// cv42_W <~~ V_cv42_W
			float x2459, x2460;
			x2459 = 1;
			float x2461;
			float x2462;
			x2461 = 1;
			float x2463;
			float x2464;
			float x2465;
			float x2466;
			x2465 = 1;
			x2466 = decay;
			x2463 = x2465 * x2466;
			float x2467;
			float x2468;
			x2467 = 1;
			x2468 = lrn_rate;
			x2464 = x2467 * x2468;
			x2462 = x2463 * x2464;
			x2460 = x2461 + x2462;
			JCudaTensor x2469;
			x2469 = x2442;
			x259.update(x2469, x2459, x2460);

			// cv44_B <~~ V_cv44_B
			float x2470, x2471;
			x2470 = 1;
			x2471 = 1;
			JCudaTensor x2472;
			x2472 = x2450;
			x277.update(x2472, x2470, x2471);

			// val X2709 = (X2685 + X2707 * d_Convolv(1,0)(cv44_W)/d_X7595)
			JCudaTensor x2473;
			JCudaTensor x2474;
			x2474 = x2432;
			JCudaTensor x2475, x2476;
			x2475 = x2421;
			x2476 = x276;
			x2473 = x193.backward_data(x2475,x2476, x2474);

			// Dealloc(X2707)
			JCudaTensor x2477;
			x2477 = x2421;
			x2477.free();

			// cv44_W <~~ V_cv44_W
			float x2478, x2479;
			x2478 = 1;
			float x2480;
			float x2481;
			x2480 = 1;
			float x2482;
			float x2483;
			float x2484;
			float x2485;
			x2484 = 1;
			x2485 = decay;
			x2482 = x2484 * x2485;
			float x2486;
			float x2487;
			x2486 = 1;
			x2487 = lrn_rate;
			x2483 = x2486 * x2487;
			x2481 = x2482 * x2483;
			x2479 = x2480 + x2481;
			JCudaTensor x2488;
			x2488 = x2425;
			x276.update(x2488, x2478, x2479);

			// val X2731 = (X2709 + X2728 * d_Pooling(3,1,1,true)(X7606,X7595)/d_X7595)
			JCudaTensor x2489;
			JCudaTensor x2490;
			x2490 = x2473;
			JCudaTensor x2491, x2492, x2493;
			x2491 = x2310;
			x2492 = x264;
			x2493 = x249;
			x2489 = x203.backward(x2491,x2492,x2493, x2490);

			// Dealloc(X2728)
			JCudaTensor x2494;
			x2494 = x2310;
			x2494.free();

			// Dealloc(X7606)
			JCudaTensor x2495;
			x2495 = x264;
			x2495.free();

			// Dealloc(X7595)
			JCudaTensor x2496;
			x2496 = x249;
			x2496.free();

			// val X2767 = (X2766 * loss1)
			JCudaTensor x2497;
			JCudaTensor x2498;
			float x2499;
			x2498 = x636;
			x2499 = loss1;
			x2497 = x2498.times_i(x2499);

			// val X2768 = (X2731 + X2767)
			JCudaTensor x2500;
			JCudaTensor x2501, x2502;
			x2501 = x2489;
			x2502 = x2497;
			x2500 = x2501.plus_i(x2502);

			// Dealloc(X2767)
			JCudaTensor x2503;
			x2503 = x2497;
			x2503.free();

			// val X4158 = Proj(X2768, X7583,X7587,X7591,X7594, 3)
			JCudaTensor x2504;
			JCudaTensor x2506;
			x2506 = x2500;
			JCudaTensor[] x2505 = x250.backward(x2506);
			x2504 = x2505[3];

			// val X4110 = Proj(X2768, X7583,X7587,X7591,X7594, 1)
			JCudaTensor x2507;
			x2507 = x2505[1];

			// val X4092 = Proj(X2768, X7583,X7587,X7591,X7594, 0)
			JCudaTensor x2508;
			x2508 = x2505[0];

			// val X4134 = Proj(X2768, X7583,X7587,X7591,X7594, 2)
			JCudaTensor x2509;
			x2509 = x2505[2];

			// Dealloc(X2768)
			JCudaTensor x2510;
			x2510 = x2500;
			x2510.free();

			// val X4139 = X4134 * d_ReLU()(X7591)/d_X7590
			JCudaTensor x2511;
			JCudaTensor x2512, x2513;
			x2512 = x2509;
			x2513 = x244;
			x2511 = x246.backward(x2512, x2513);

			// Dealloc(X7591)
			JCudaTensor x2514;
			x2514 = x244;
			x2514.free();

			// val X4162 = X4158 * d_ReLU()(X7594)/d_X7593
			JCudaTensor x2515;
			JCudaTensor x2516, x2517;
			x2516 = x2504;
			x2517 = x247;
			x2515 = x246.backward(x2516, x2517);

			// Dealloc(X7594)
			JCudaTensor x2518;
			x2518 = x247;
			x2518.free();

			// val X4115 = X4110 * d_ReLU()(X7587)/d_X7586
			JCudaTensor x2519;
			JCudaTensor x2520, x2521;
			x2520 = x2507;
			x2521 = x241;
			x2519 = x243.backward(x2520, x2521);

			// Dealloc(X7587)
			JCudaTensor x2522;
			x2522 = x241;
			x2522.free();

			// val X4095 = X4092 * d_ReLU()(X7583)/d_X7582
			JCudaTensor x2523;
			JCudaTensor x2524, x2525;
			x2524 = x2508;
			x2525 = x238;
			x2523 = x240.backward(x2524, x2525);

			// Dealloc(X7583)
			JCudaTensor x2526;
			x2526 = x238;
			x2526.free();

			// V_cv35_W <~~ X4139 * d_Convolv(1,2)(X7589)/d_cv35_W
			float x2528, x2529;
			float x2530;
			float x2531;
			x2530 = 1;
			x2531 = lrn_rate;
			x2528 = x2530 * x2531;
			x2529 = momentum;
			JCudaTensor x2532, x2533;
			x2532 = x2511;
			x2533 = x221;
			x230.backward_filter(x2532, x2533, x2527, x2528, x2529);

			// V_cv35_B <~~ X4139 * d_Convolv(1,2)()/d_cv35_B
			float x2535, x2536;
			float x2537;
			float x2538;
			x2537 = 2;
			x2538 = lrn_rate;
			x2535 = x2537 * x2538;
			x2536 = momentum;
			JCudaTensor x2539;
			x2539 = x2511;
			x230.backward_bias(x2539, x2534, x2535, x2536);

			// V_cv36_B <~~ X4162 * d_Convolv(1,0)()/d_cv36_B
			float x2541, x2542;
			float x2543;
			float x2544;
			x2543 = 2;
			x2544 = lrn_rate;
			x2541 = x2543 * x2544;
			x2542 = momentum;
			JCudaTensor x2545;
			x2545 = x2515;
			x220.backward_bias(x2545, x2540, x2541, x2542);

			// V_cv33_B <~~ X4115 * d_Convolv(1,1)()/d_cv33_B
			float x2547, x2548;
			float x2549;
			float x2550;
			x2549 = 2;
			x2550 = lrn_rate;
			x2547 = x2549 * x2550;
			x2548 = momentum;
			JCudaTensor x2551;
			x2551 = x2519;
			x237.backward_bias(x2551, x2546, x2547, x2548);

			// val X4140 = X4139 * d_Convolv(1,2)(cv35_W)/d_X7589
			JCudaTensor x2552;
			JCudaTensor x2553, x2554;
			x2553 = x2511;
			x2554 = x228;
			x2552 = x230.backward_data(x2553, x2554);

			// Dealloc(X4139)
			JCudaTensor x2555;
			x2555 = x2511;
			x2555.free();

			// V_cv36_W <~~ X4162 * d_Convolv(1,0)(X7592)/d_cv36_W
			float x2557, x2558;
			float x2559;
			float x2560;
			x2559 = 1;
			x2560 = lrn_rate;
			x2557 = x2559 * x2560;
			x2558 = momentum;
			JCudaTensor x2561, x2562;
			x2561 = x2515;
			x2562 = x201;
			x220.backward_filter(x2561, x2562, x2556, x2557, x2558);

			// val X4163 = X4162 * d_Convolv(1,0)(cv36_W)/d_X7592
			JCudaTensor x2563;
			JCudaTensor x2564, x2565;
			x2564 = x2515;
			x2565 = x218;
			x2563 = x220.backward_data(x2564, x2565);

			// Dealloc(X4162)
			JCudaTensor x2566;
			x2566 = x2515;
			x2566.free();

			// V_cv31_W <~~ X4095 * d_Convolv(1,0)(X7581)/d_cv31_W
			float x2568, x2569;
			float x2570;
			float x2571;
			x2570 = 1;
			x2571 = lrn_rate;
			x2568 = x2570 * x2571;
			x2569 = momentum;
			JCudaTensor x2572, x2573;
			x2572 = x2523;
			x2573 = x184;
			x200.backward_filter(x2572, x2573, x2567, x2568, x2569);

			// val X4116 = X4115 * d_Convolv(1,1)(cv33_W)/d_X7585
			JCudaTensor x2574;
			JCudaTensor x2575, x2576;
			x2575 = x2519;
			x2576 = x235;
			x2574 = x237.backward_data(x2575, x2576);

			// V_cv33_W <~~ X4115 * d_Convolv(1,1)(X7585)/d_cv33_W
			float x2578, x2579;
			float x2580;
			float x2581;
			x2580 = 1;
			x2581 = lrn_rate;
			x2578 = x2580 * x2581;
			x2579 = momentum;
			JCudaTensor x2582, x2583;
			x2582 = x2519;
			x2583 = x211;
			x237.backward_filter(x2582, x2583, x2577, x2578, x2579);

			// Dealloc(X4115)
			JCudaTensor x2584;
			x2584 = x2519;
			x2584.free();

			// val X4096 = X4095 * d_Convolv(1,0)(cv31_W)/d_X7581
			JCudaTensor x2585;
			JCudaTensor x2586, x2587;
			x2586 = x2523;
			x2587 = x198;
			x2585 = x200.backward_data(x2586, x2587);

			// V_cv31_B <~~ X4095 * d_Convolv(1,0)()/d_cv31_B
			float x2589, x2590;
			float x2591;
			float x2592;
			x2591 = 2;
			x2592 = lrn_rate;
			x2589 = x2591 * x2592;
			x2590 = momentum;
			JCudaTensor x2593;
			x2593 = x2523;
			x200.backward_bias(x2593, x2588, x2589, x2590);

			// Dealloc(X4095)
			JCudaTensor x2594;
			x2594 = x2523;
			x2594.free();

			// cv33_B <~~ V_cv33_B
			float x2595, x2596;
			x2595 = 1;
			x2596 = 1;
			JCudaTensor x2597;
			x2597 = x2546;
			x236.update(x2597, x2595, x2596);

			// cv33_W <~~ V_cv33_W
			float x2598, x2599;
			x2598 = 1;
			float x2600;
			float x2601;
			x2600 = 1;
			float x2602;
			float x2603;
			float x2604;
			float x2605;
			x2604 = 1;
			x2605 = decay;
			x2602 = x2604 * x2605;
			float x2606;
			float x2607;
			x2606 = 1;
			x2607 = lrn_rate;
			x2603 = x2606 * x2607;
			x2601 = x2602 * x2603;
			x2599 = x2600 + x2601;
			JCudaTensor x2608;
			x2608 = x2577;
			x235.update(x2608, x2598, x2599);

			// cv31_B <~~ V_cv31_B
			float x2609, x2610;
			x2609 = 1;
			x2610 = 1;
			JCudaTensor x2611;
			x2611 = x2588;
			x199.update(x2611, x2609, x2610);

			// cv31_W <~~ V_cv31_W
			float x2612, x2613;
			x2612 = 1;
			float x2614;
			float x2615;
			x2614 = 1;
			float x2616;
			float x2617;
			float x2618;
			float x2619;
			x2618 = 1;
			x2619 = decay;
			x2616 = x2618 * x2619;
			float x2620;
			float x2621;
			x2620 = 1;
			x2621 = lrn_rate;
			x2617 = x2620 * x2621;
			x2615 = x2616 * x2617;
			x2613 = x2614 + x2615;
			JCudaTensor x2622;
			x2622 = x2567;
			x198.update(x2622, x2612, x2613);

			// cv36_B <~~ V_cv36_B
			float x2623, x2624;
			x2623 = 1;
			x2624 = 1;
			JCudaTensor x2625;
			x2625 = x2540;
			x219.update(x2625, x2623, x2624);

			// cv35_B <~~ V_cv35_B
			float x2626, x2627;
			x2626 = 1;
			x2627 = 1;
			JCudaTensor x2628;
			x2628 = x2534;
			x229.update(x2628, x2626, x2627);

			// cv35_W <~~ V_cv35_W
			float x2629, x2630;
			x2629 = 1;
			float x2631;
			float x2632;
			x2631 = 1;
			float x2633;
			float x2634;
			float x2635;
			float x2636;
			x2635 = 1;
			x2636 = decay;
			x2633 = x2635 * x2636;
			float x2637;
			float x2638;
			x2637 = 1;
			x2638 = lrn_rate;
			x2634 = x2637 * x2638;
			x2632 = x2633 * x2634;
			x2630 = x2631 + x2632;
			JCudaTensor x2639;
			x2639 = x2527;
			x228.update(x2639, x2629, x2630);

			// cv36_W <~~ V_cv36_W
			float x2640, x2641;
			x2640 = 1;
			float x2642;
			float x2643;
			x2642 = 1;
			float x2644;
			float x2645;
			float x2646;
			float x2647;
			x2646 = 1;
			x2647 = decay;
			x2644 = x2646 * x2647;
			float x2648;
			float x2649;
			x2648 = 1;
			x2649 = lrn_rate;
			x2645 = x2648 * x2649;
			x2643 = x2644 * x2645;
			x2641 = x2642 + x2643;
			JCudaTensor x2650;
			x2650 = x2556;
			x218.update(x2650, x2640, x2641);

			// val X4118 = X4116 * d_ReLU()(X7585)/d_X7584
			JCudaTensor x2651;
			JCudaTensor x2652, x2653;
			x2652 = x2574;
			x2653 = x211;
			x2651 = x213.backward(x2652, x2653);

			// Dealloc(X7585)
			JCudaTensor x2654;
			x2654 = x211;
			x2654.free();

			// val X4142 = X4140 * d_ReLU()(X7589)/d_X7588
			JCudaTensor x2655;
			JCudaTensor x2656, x2657;
			x2656 = x2552;
			x2657 = x221;
			x2655 = x223.backward(x2656, x2657);

			// Dealloc(X7589)
			JCudaTensor x2658;
			x2658 = x221;
			x2658.free();

			// V_cv34_W <~~ X4142 * d_Convolv(1,0)(X7581)/d_cv34_W
			float x2660, x2661;
			float x2662;
			float x2663;
			x2662 = 1;
			x2663 = lrn_rate;
			x2660 = x2662 * x2663;
			x2661 = momentum;
			JCudaTensor x2664, x2665;
			x2664 = x2655;
			x2665 = x184;
			x193.backward_filter(x2664, x2665, x2659, x2660, x2661);

			// val X4120 = (X4096 + X4118 * d_Convolv(1,0)(cv32_W)/d_X7581)
			JCudaTensor x2666;
			JCudaTensor x2667;
			x2667 = x2585;
			JCudaTensor x2668, x2669;
			x2668 = x2651;
			x2669 = x208;
			x2666 = x210.backward_data(x2668,x2669, x2667);

			// V_cv34_B <~~ X4142 * d_Convolv(1,0)()/d_cv34_B
			float x2671, x2672;
			float x2673;
			float x2674;
			x2673 = 2;
			x2674 = lrn_rate;
			x2671 = x2673 * x2674;
			x2672 = momentum;
			JCudaTensor x2675;
			x2675 = x2655;
			x193.backward_bias(x2675, x2670, x2671, x2672);

			// V_cv32_B <~~ X4118 * d_Convolv(1,0)()/d_cv32_B
			float x2677, x2678;
			float x2679;
			float x2680;
			x2679 = 2;
			x2680 = lrn_rate;
			x2677 = x2679 * x2680;
			x2678 = momentum;
			JCudaTensor x2681;
			x2681 = x2651;
			x210.backward_bias(x2681, x2676, x2677, x2678);

			// V_cv32_W <~~ X4118 * d_Convolv(1,0)(X7581)/d_cv32_W
			float x2683, x2684;
			float x2685;
			float x2686;
			x2685 = 1;
			x2686 = lrn_rate;
			x2683 = x2685 * x2686;
			x2684 = momentum;
			JCudaTensor x2687, x2688;
			x2687 = x2651;
			x2688 = x184;
			x210.backward_filter(x2687, x2688, x2682, x2683, x2684);

			// Dealloc(X4118)
			JCudaTensor x2689;
			x2689 = x2651;
			x2689.free();

			// cv34_B <~~ V_cv34_B
			float x2690, x2691;
			x2690 = 1;
			x2691 = 1;
			JCudaTensor x2692;
			x2692 = x2670;
			x192.update(x2692, x2690, x2691);

			// cv32_B <~~ V_cv32_B
			float x2693, x2694;
			x2693 = 1;
			x2694 = 1;
			JCudaTensor x2695;
			x2695 = x2676;
			x209.update(x2695, x2693, x2694);

			// cv32_W <~~ V_cv32_W
			float x2696, x2697;
			x2696 = 1;
			float x2698;
			float x2699;
			x2698 = 1;
			float x2700;
			float x2701;
			float x2702;
			float x2703;
			x2702 = 1;
			x2703 = decay;
			x2700 = x2702 * x2703;
			float x2704;
			float x2705;
			x2704 = 1;
			x2705 = lrn_rate;
			x2701 = x2704 * x2705;
			x2699 = x2700 * x2701;
			x2697 = x2698 + x2699;
			JCudaTensor x2706;
			x2706 = x2682;
			x208.update(x2706, x2696, x2697);

			// val X4144 = (X4120 + X4142 * d_Convolv(1,0)(cv34_W)/d_X7581)
			JCudaTensor x2707;
			JCudaTensor x2708;
			x2708 = x2666;
			JCudaTensor x2709, x2710;
			x2709 = x2655;
			x2710 = x191;
			x2707 = x193.backward_data(x2709,x2710, x2708);

			// Dealloc(X4142)
			JCudaTensor x2711;
			x2711 = x2655;
			x2711.free();

			// cv34_W <~~ V_cv34_W
			float x2712, x2713;
			x2712 = 1;
			float x2714;
			float x2715;
			x2714 = 1;
			float x2716;
			float x2717;
			float x2718;
			float x2719;
			x2718 = 1;
			x2719 = decay;
			x2716 = x2718 * x2719;
			float x2720;
			float x2721;
			x2720 = 1;
			x2721 = lrn_rate;
			x2717 = x2720 * x2721;
			x2715 = x2716 * x2717;
			x2713 = x2714 + x2715;
			JCudaTensor x2722;
			x2722 = x2659;
			x191.update(x2722, x2712, x2713);

			// val X4166 = (X4144 + X4163 * d_Pooling(3,1,1,true)(X7592,X7581)/d_X7581)
			JCudaTensor x2723;
			JCudaTensor x2724;
			x2724 = x2707;
			JCudaTensor x2725, x2726, x2727;
			x2725 = x2563;
			x2726 = x201;
			x2727 = x184;
			x2723 = x203.backward(x2725,x2726,x2727, x2724);

			// Dealloc(X4163)
			JCudaTensor x2728;
			x2728 = x2563;
			x2728.free();

			// Dealloc(X7592)
			JCudaTensor x2729;
			x2729 = x201;
			x2729.free();

			// val X4168 = X4166 * d_Pooling(3,2,1,true)(X7581,X7580)/d_X7580
			JCudaTensor x2730;
			JCudaTensor x2731, x2732, x2733;
			x2731 = x2723;
			x2732 = x184;
			x2733 = x179;
			x2730 = x186.backward(x2731, x2732, x2733);

			// Dealloc(X4166)
			JCudaTensor x2734;
			x2734 = x2723;
			x2734.free();

			// Dealloc(X7581)
			JCudaTensor x2735;
			x2735 = x184;
			x2735.free();

			// Dealloc(X7580)
			JCudaTensor x2736;
			x2736 = x179;
			x2736.free();

			// val X4182 = Proj(X4168, X7568,X7572,X7576,X7579, 0)
			JCudaTensor x2737;
			JCudaTensor x2739;
			x2739 = x2730;
			JCudaTensor[] x2738 = x119.backward(x2739);
			x2737 = x2738[0];

			// val X4200 = Proj(X4168, X7568,X7572,X7576,X7579, 1)
			JCudaTensor x2740;
			x2740 = x2738[1];

			// val X4224 = Proj(X4168, X7568,X7572,X7576,X7579, 2)
			JCudaTensor x2741;
			x2741 = x2738[2];

			// val X4248 = Proj(X4168, X7568,X7572,X7576,X7579, 3)
			JCudaTensor x2742;
			x2742 = x2738[3];

			// Dealloc(X4168)
			JCudaTensor x2743;
			x2743 = x2730;
			x2743.free();

			// val X4205 = X4200 * d_ReLU()(X7572)/d_X7571
			JCudaTensor x2744;
			JCudaTensor x2745, x2746;
			x2745 = x2740;
			x2746 = x173;
			x2744 = x112.backward(x2745, x2746);

			// Dealloc(X7572)
			JCudaTensor x2747;
			x2747 = x173;
			x2747.free();

			// val X4252 = X4248 * d_ReLU()(X7579)/d_X7578
			JCudaTensor x2748;
			JCudaTensor x2749, x2750;
			x2749 = x2742;
			x2750 = x177;
			x2748 = x115.backward(x2749, x2750);

			// Dealloc(X7579)
			JCudaTensor x2751;
			x2751 = x177;
			x2751.free();

			// val X4185 = X4182 * d_ReLU()(X7568)/d_X7567
			JCudaTensor x2752;
			JCudaTensor x2753, x2754;
			x2753 = x2737;
			x2754 = x171;
			x2752 = x109.backward(x2753, x2754);

			// Dealloc(X7568)
			JCudaTensor x2755;
			x2755 = x171;
			x2755.free();

			// val X4229 = X4224 * d_ReLU()(X7576)/d_X7575
			JCudaTensor x2756;
			JCudaTensor x2757, x2758;
			x2757 = x2741;
			x2758 = x175;
			x2756 = x115.backward(x2757, x2758);

			// Dealloc(X7576)
			JCudaTensor x2759;
			x2759 = x175;
			x2759.free();

			// V_cv23_W <~~ X4205 * d_Convolv(1,1)(X7570)/d_cv23_W
			float x2761, x2762;
			float x2763;
			float x2764;
			x2763 = 1;
			x2764 = lrn_rate;
			x2761 = x2763 * x2764;
			x2762 = momentum;
			JCudaTensor x2765, x2766;
			x2765 = x2744;
			x2766 = x155;
			x99.backward_filter(x2765, x2766, x2760, x2761, x2762);

			// val X4253 = X4252 * d_Convolv(1,0)(cv26_W)/d_X7577
			JCudaTensor x2767;
			JCudaTensor x2768, x2769;
			x2768 = x2748;
			x2769 = x152;
			x2767 = x154.backward_data(x2768, x2769);

			// V_cv21_B <~~ X4185 * d_Convolv(1,0)()/d_cv21_B
			float x2771, x2772;
			float x2773;
			float x2774;
			x2773 = 2;
			x2774 = lrn_rate;
			x2771 = x2773 * x2774;
			x2772 = momentum;
			JCudaTensor x2775;
			x2775 = x2752;
			x147.backward_bias(x2775, x2770, x2771, x2772);

			// V_cv26_B <~~ X4252 * d_Convolv(1,0)()/d_cv26_B
			float x2777, x2778;
			float x2779;
			float x2780;
			x2779 = 2;
			x2780 = lrn_rate;
			x2777 = x2779 * x2780;
			x2778 = momentum;
			JCudaTensor x2781;
			x2781 = x2748;
			x154.backward_bias(x2781, x2776, x2777, x2778);

			// V_cv21_W <~~ X4185 * d_Convolv(1,0)(X7566)/d_cv21_W
			float x2783, x2784;
			float x2785;
			float x2786;
			x2785 = 1;
			x2786 = lrn_rate;
			x2783 = x2785 * x2786;
			x2784 = momentum;
			JCudaTensor x2787, x2788;
			x2787 = x2752;
			x2788 = x118;
			x147.backward_filter(x2787, x2788, x2782, x2783, x2784);

			// V_cv23_B <~~ X4205 * d_Convolv(1,1)()/d_cv23_B
			float x2790, x2791;
			float x2792;
			float x2793;
			x2792 = 2;
			x2793 = lrn_rate;
			x2790 = x2792 * x2793;
			x2791 = momentum;
			JCudaTensor x2794;
			x2794 = x2744;
			x99.backward_bias(x2794, x2789, x2790, x2791);

			// V_cv26_W <~~ X4252 * d_Convolv(1,0)(X7577)/d_cv26_W
			float x2796, x2797;
			float x2798;
			float x2799;
			x2798 = 1;
			x2799 = lrn_rate;
			x2796 = x2798 * x2799;
			x2797 = momentum;
			JCudaTensor x2800, x2801;
			x2800 = x2748;
			x2801 = x124;
			x154.backward_filter(x2800, x2801, x2795, x2796, x2797);

			// Dealloc(X4252)
			JCudaTensor x2802;
			x2802 = x2748;
			x2802.free();

			// V_cv25_B <~~ X4229 * d_Convolv(1,2)()/d_cv25_B
			float x2804, x2805;
			float x2806;
			float x2807;
			x2806 = 2;
			x2807 = lrn_rate;
			x2804 = x2806 * x2807;
			x2805 = momentum;
			JCudaTensor x2808;
			x2808 = x2756;
			x106.backward_bias(x2808, x2803, x2804, x2805);

			// V_cv25_W <~~ X4229 * d_Convolv(1,2)(X7574)/d_cv25_W
			float x2810, x2811;
			float x2812;
			float x2813;
			x2812 = 1;
			x2813 = lrn_rate;
			x2810 = x2812 * x2813;
			x2811 = momentum;
			JCudaTensor x2814, x2815;
			x2814 = x2756;
			x2815 = x157;
			x106.backward_filter(x2814, x2815, x2809, x2810, x2811);

			// val X4186 = X4185 * d_Convolv(1,0)(cv21_W)/d_X7566
			JCudaTensor x2816;
			JCudaTensor x2817, x2818;
			x2817 = x2752;
			x2818 = x145;
			x2816 = x147.backward_data(x2817, x2818);

			// Dealloc(X4185)
			JCudaTensor x2819;
			x2819 = x2752;
			x2819.free();

			// val X4206 = X4205 * d_Convolv(1,1)(cv23_W)/d_X7570
			JCudaTensor x2820;
			JCudaTensor x2821, x2822;
			x2821 = x2744;
			x2822 = x169;
			x2820 = x99.backward_data(x2821, x2822);

			// Dealloc(X4205)
			JCudaTensor x2823;
			x2823 = x2744;
			x2823.free();

			// val X4230 = X4229 * d_Convolv(1,2)(cv25_W)/d_X7574
			JCudaTensor x2824;
			JCudaTensor x2825, x2826;
			x2825 = x2756;
			x2826 = x163;
			x2824 = x106.backward_data(x2825, x2826);

			// Dealloc(X4229)
			JCudaTensor x2827;
			x2827 = x2756;
			x2827.free();

			// cv23_W <~~ V_cv23_W
			float x2828, x2829;
			x2828 = 1;
			float x2830;
			float x2831;
			x2830 = 1;
			float x2832;
			float x2833;
			float x2834;
			float x2835;
			x2834 = 1;
			x2835 = decay;
			x2832 = x2834 * x2835;
			float x2836;
			float x2837;
			x2836 = 1;
			x2837 = lrn_rate;
			x2833 = x2836 * x2837;
			x2831 = x2832 * x2833;
			x2829 = x2830 + x2831;
			JCudaTensor x2838;
			x2838 = x2760;
			x169.update(x2838, x2828, x2829);

			// cv25_W <~~ V_cv25_W
			float x2839, x2840;
			x2839 = 1;
			float x2841;
			float x2842;
			x2841 = 1;
			float x2843;
			float x2844;
			float x2845;
			float x2846;
			x2845 = 1;
			x2846 = decay;
			x2843 = x2845 * x2846;
			float x2847;
			float x2848;
			x2847 = 1;
			x2848 = lrn_rate;
			x2844 = x2847 * x2848;
			x2842 = x2843 * x2844;
			x2840 = x2841 + x2842;
			JCudaTensor x2849;
			x2849 = x2809;
			x163.update(x2849, x2839, x2840);

			// cv23_B <~~ V_cv23_B
			float x2850, x2851;
			x2850 = 1;
			x2851 = 1;
			JCudaTensor x2852;
			x2852 = x2789;
			x170.update(x2852, x2850, x2851);

			// cv25_B <~~ V_cv25_B
			float x2853, x2854;
			x2853 = 1;
			x2854 = 1;
			JCudaTensor x2855;
			x2855 = x2803;
			x164.update(x2855, x2853, x2854);

			// cv26_W <~~ V_cv26_W
			float x2856, x2857;
			x2856 = 1;
			float x2858;
			float x2859;
			x2858 = 1;
			float x2860;
			float x2861;
			float x2862;
			float x2863;
			x2862 = 1;
			x2863 = decay;
			x2860 = x2862 * x2863;
			float x2864;
			float x2865;
			x2864 = 1;
			x2865 = lrn_rate;
			x2861 = x2864 * x2865;
			x2859 = x2860 * x2861;
			x2857 = x2858 + x2859;
			JCudaTensor x2866;
			x2866 = x2795;
			x152.update(x2866, x2856, x2857);

			// cv26_B <~~ V_cv26_B
			float x2867, x2868;
			x2867 = 1;
			x2868 = 1;
			JCudaTensor x2869;
			x2869 = x2776;
			x153.update(x2869, x2867, x2868);

			// cv21_W <~~ V_cv21_W
			float x2870, x2871;
			x2870 = 1;
			float x2872;
			float x2873;
			x2872 = 1;
			float x2874;
			float x2875;
			float x2876;
			float x2877;
			x2876 = 1;
			x2877 = decay;
			x2874 = x2876 * x2877;
			float x2878;
			float x2879;
			x2878 = 1;
			x2879 = lrn_rate;
			x2875 = x2878 * x2879;
			x2873 = x2874 * x2875;
			x2871 = x2872 + x2873;
			JCudaTensor x2880;
			x2880 = x2782;
			x145.update(x2880, x2870, x2871);

			// cv21_B <~~ V_cv21_B
			float x2881, x2882;
			x2881 = 1;
			x2882 = 1;
			JCudaTensor x2883;
			x2883 = x2770;
			x146.update(x2883, x2881, x2882);

			// val X4232 = X4230 * d_ReLU()(X7574)/d_X7573
			JCudaTensor x2884;
			JCudaTensor x2885, x2886;
			x2885 = x2824;
			x2886 = x157;
			x2884 = x92.backward(x2885, x2886);

			// Dealloc(X7574)
			JCudaTensor x2887;
			x2887 = x157;
			x2887.free();

			// val X4208 = X4206 * d_ReLU()(X7570)/d_X7569
			JCudaTensor x2888;
			JCudaTensor x2889, x2890;
			x2889 = x2820;
			x2890 = x155;
			x2888 = x89.backward(x2889, x2890);

			// Dealloc(X7570)
			JCudaTensor x2891;
			x2891 = x155;
			x2891.free();

			// V_cv24_W <~~ X4232 * d_Convolv(1,0)(X7566)/d_cv24_W
			float x2893, x2894;
			float x2895;
			float x2896;
			x2895 = 1;
			x2896 = lrn_rate;
			x2893 = x2895 * x2896;
			x2894 = momentum;
			JCudaTensor x2897, x2898;
			x2897 = x2884;
			x2898 = x118;
			x133.backward_filter(x2897, x2898, x2892, x2893, x2894);

			// V_cv24_B <~~ X4232 * d_Convolv(1,0)()/d_cv24_B
			float x2900, x2901;
			float x2902;
			float x2903;
			x2902 = 2;
			x2903 = lrn_rate;
			x2900 = x2902 * x2903;
			x2901 = momentum;
			JCudaTensor x2904;
			x2904 = x2884;
			x133.backward_bias(x2904, x2899, x2900, x2901);

			// V_cv22_B <~~ X4208 * d_Convolv(1,0)()/d_cv22_B
			float x2906, x2907;
			float x2908;
			float x2909;
			x2908 = 2;
			x2909 = lrn_rate;
			x2906 = x2908 * x2909;
			x2907 = momentum;
			JCudaTensor x2910;
			x2910 = x2888;
			x140.backward_bias(x2910, x2905, x2906, x2907);

			// val X4210 = (X4186 + X4208 * d_Convolv(1,0)(cv22_W)/d_X7566)
			JCudaTensor x2911;
			JCudaTensor x2912;
			x2912 = x2816;
			JCudaTensor x2913, x2914;
			x2913 = x2888;
			x2914 = x138;
			x2911 = x140.backward_data(x2913,x2914, x2912);

			// V_cv22_W <~~ X4208 * d_Convolv(1,0)(X7566)/d_cv22_W
			float x2916, x2917;
			float x2918;
			float x2919;
			x2918 = 1;
			x2919 = lrn_rate;
			x2916 = x2918 * x2919;
			x2917 = momentum;
			JCudaTensor x2920, x2921;
			x2920 = x2888;
			x2921 = x118;
			x140.backward_filter(x2920, x2921, x2915, x2916, x2917);

			// Dealloc(X4208)
			JCudaTensor x2922;
			x2922 = x2888;
			x2922.free();

			// cv24_B <~~ V_cv24_B
			float x2923, x2924;
			x2923 = 1;
			x2924 = 1;
			JCudaTensor x2925;
			x2925 = x2899;
			x132.update(x2925, x2923, x2924);

			// cv22_B <~~ V_cv22_B
			float x2926, x2927;
			x2926 = 1;
			x2927 = 1;
			JCudaTensor x2928;
			x2928 = x2905;
			x139.update(x2928, x2926, x2927);

			// cv22_W <~~ V_cv22_W
			float x2929, x2930;
			x2929 = 1;
			float x2931;
			float x2932;
			x2931 = 1;
			float x2933;
			float x2934;
			float x2935;
			float x2936;
			x2935 = 1;
			x2936 = decay;
			x2933 = x2935 * x2936;
			float x2937;
			float x2938;
			x2937 = 1;
			x2938 = lrn_rate;
			x2934 = x2937 * x2938;
			x2932 = x2933 * x2934;
			x2930 = x2931 + x2932;
			JCudaTensor x2939;
			x2939 = x2915;
			x138.update(x2939, x2929, x2930);

			// val X4234 = (X4210 + X4232 * d_Convolv(1,0)(cv24_W)/d_X7566)
			JCudaTensor x2940;
			JCudaTensor x2941;
			x2941 = x2911;
			JCudaTensor x2942, x2943;
			x2942 = x2884;
			x2943 = x131;
			x2940 = x133.backward_data(x2942,x2943, x2941);

			// Dealloc(X4232)
			JCudaTensor x2944;
			x2944 = x2884;
			x2944.free();

			// cv24_W <~~ V_cv24_W
			float x2945, x2946;
			x2945 = 1;
			float x2947;
			float x2948;
			x2947 = 1;
			float x2949;
			float x2950;
			float x2951;
			float x2952;
			x2951 = 1;
			x2952 = decay;
			x2949 = x2951 * x2952;
			float x2953;
			float x2954;
			x2953 = 1;
			x2954 = lrn_rate;
			x2950 = x2953 * x2954;
			x2948 = x2949 * x2950;
			x2946 = x2947 + x2948;
			JCudaTensor x2955;
			x2955 = x2892;
			x131.update(x2955, x2945, x2946);

			// val X4256 = (X4234 + X4253 * d_Pooling(3,1,1,true)(X7577,X7566)/d_X7566)
			JCudaTensor x2956;
			JCudaTensor x2957;
			x2957 = x2940;
			JCudaTensor x2958, x2959, x2960;
			x2958 = x2767;
			x2959 = x124;
			x2960 = x118;
			x2956 = x126.backward(x2958,x2959,x2960, x2957);

			// Dealloc(X4253)
			JCudaTensor x2961;
			x2961 = x2767;
			x2961.free();

			// Dealloc(X7577)
			JCudaTensor x2962;
			x2962 = x124;
			x2962.free();

			// Dealloc(X7566)
			JCudaTensor x2963;
			x2963 = x118;
			x2963.free();

			// val X4270 = Proj(X4256, X7554,X7558,X7562,X7565, 0)
			JCudaTensor x2964;
			JCudaTensor x2966;
			x2966 = x2956;
			JCudaTensor[] x2965 = x119.backward(x2966);
			x2964 = x2965[0];

			// val X4288 = Proj(X4256, X7554,X7558,X7562,X7565, 1)
			JCudaTensor x2967;
			x2967 = x2965[1];

			// val X4312 = Proj(X4256, X7554,X7558,X7562,X7565, 2)
			JCudaTensor x2968;
			x2968 = x2965[2];

			// val X4336 = Proj(X4256, X7554,X7558,X7562,X7565, 3)
			JCudaTensor x2969;
			x2969 = x2965[3];

			// Dealloc(X4256)
			JCudaTensor x2970;
			x2970 = x2956;
			x2970.free();

			// val X4340 = X4336 * d_ReLU()(X7565)/d_X7564
			JCudaTensor x2971;
			JCudaTensor x2972, x2973;
			x2972 = x2969;
			x2973 = x116;
			x2971 = x115.backward(x2972, x2973);

			// Dealloc(X7565)
			JCudaTensor x2974;
			x2974 = x116;
			x2974.free();

			// val X4273 = X4270 * d_ReLU()(X7554)/d_X7553
			JCudaTensor x2975;
			JCudaTensor x2976, x2977;
			x2976 = x2964;
			x2977 = x107;
			x2975 = x109.backward(x2976, x2977);

			// Dealloc(X7554)
			JCudaTensor x2978;
			x2978 = x107;
			x2978.free();

			// val X4293 = X4288 * d_ReLU()(X7558)/d_X7557
			JCudaTensor x2979;
			JCudaTensor x2980, x2981;
			x2980 = x2967;
			x2981 = x110;
			x2979 = x112.backward(x2980, x2981);

			// Dealloc(X7558)
			JCudaTensor x2982;
			x2982 = x110;
			x2982.free();

			// val X4317 = X4312 * d_ReLU()(X7562)/d_X7561
			JCudaTensor x2983;
			JCudaTensor x2984, x2985;
			x2984 = x2968;
			x2985 = x113;
			x2983 = x115.backward(x2984, x2985);

			// Dealloc(X7562)
			JCudaTensor x2986;
			x2986 = x113;
			x2986.free();

			// V_cv16_B <~~ X4340 * d_Convolv(1,0)()/d_cv16_B
			float x2988, x2989;
			float x2990;
			float x2991;
			x2990 = 2;
			x2991 = lrn_rate;
			x2988 = x2990 * x2991;
			x2989 = momentum;
			JCudaTensor x2992;
			x2992 = x2971;
			x86.backward_bias(x2992, x2987, x2988, x2989);

			// V_cv11_W <~~ X4273 * d_Convolv(1,0)(X7552)/d_cv11_W
			float x2994, x2995;
			float x2996;
			float x2997;
			x2996 = 1;
			x2997 = lrn_rate;
			x2994 = x2996 * x2997;
			x2995 = momentum;
			JCudaTensor x2998, x2999;
			x2998 = x2975;
			x2999 = x53;
			x69.backward_filter(x2998, x2999, x2993, x2994, x2995);

			// V_cv13_W <~~ X4293 * d_Convolv(1,1)(X7556)/d_cv13_W
			float x3001, x3002;
			float x3003;
			float x3004;
			x3003 = 1;
			x3004 = lrn_rate;
			x3001 = x3003 * x3004;
			x3002 = momentum;
			JCudaTensor x3005, x3006;
			x3005 = x2979;
			x3006 = x87;
			x99.backward_filter(x3005, x3006, x3000, x3001, x3002);

			// V_cv11_B <~~ X4273 * d_Convolv(1,0)()/d_cv11_B
			float x3008, x3009;
			float x3010;
			float x3011;
			x3010 = 2;
			x3011 = lrn_rate;
			x3008 = x3010 * x3011;
			x3009 = momentum;
			JCudaTensor x3012;
			x3012 = x2975;
			x69.backward_bias(x3012, x3007, x3008, x3009);

			// val X4341 = X4340 * d_Convolv(1,0)(cv16_W)/d_X7563
			JCudaTensor x3013;
			JCudaTensor x3014, x3015;
			x3014 = x2971;
			x3015 = x84;
			x3013 = x86.backward_data(x3014, x3015);

			// val X4274 = X4273 * d_Convolv(1,0)(cv11_W)/d_X7552
			JCudaTensor x3016;
			JCudaTensor x3017, x3018;
			x3017 = x2975;
			x3018 = x67;
			x3016 = x69.backward_data(x3017, x3018);

			// Dealloc(X4273)
			JCudaTensor x3019;
			x3019 = x2975;
			x3019.free();

			// val X4318 = X4317 * d_Convolv(1,2)(cv15_W)/d_X7560
			JCudaTensor x3020;
			JCudaTensor x3021, x3022;
			x3021 = x2983;
			x3022 = x104;
			x3020 = x106.backward_data(x3021, x3022);

			// val X4294 = X4293 * d_Convolv(1,1)(cv13_W)/d_X7556
			JCudaTensor x3023;
			JCudaTensor x3024, x3025;
			x3024 = x2979;
			x3025 = x97;
			x3023 = x99.backward_data(x3024, x3025);

			// V_cv15_B <~~ X4317 * d_Convolv(1,2)()/d_cv15_B
			float x3027, x3028;
			float x3029;
			float x3030;
			x3029 = 2;
			x3030 = lrn_rate;
			x3027 = x3029 * x3030;
			x3028 = momentum;
			JCudaTensor x3031;
			x3031 = x2983;
			x106.backward_bias(x3031, x3026, x3027, x3028);

			// V_cv13_B <~~ X4293 * d_Convolv(1,1)()/d_cv13_B
			float x3033, x3034;
			float x3035;
			float x3036;
			x3035 = 2;
			x3036 = lrn_rate;
			x3033 = x3035 * x3036;
			x3034 = momentum;
			JCudaTensor x3037;
			x3037 = x2979;
			x99.backward_bias(x3037, x3032, x3033, x3034);

			// Dealloc(X4293)
			JCudaTensor x3038;
			x3038 = x2979;
			x3038.free();

			// V_cv15_W <~~ X4317 * d_Convolv(1,2)(X7560)/d_cv15_W
			float x3040, x3041;
			float x3042;
			float x3043;
			x3042 = 1;
			x3043 = lrn_rate;
			x3040 = x3042 * x3043;
			x3041 = momentum;
			JCudaTensor x3044, x3045;
			x3044 = x2983;
			x3045 = x90;
			x106.backward_filter(x3044, x3045, x3039, x3040, x3041);

			// Dealloc(X4317)
			JCudaTensor x3046;
			x3046 = x2983;
			x3046.free();

			// V_cv16_W <~~ X4340 * d_Convolv(1,0)(X7563)/d_cv16_W
			float x3048, x3049;
			float x3050;
			float x3051;
			x3050 = 1;
			x3051 = lrn_rate;
			x3048 = x3050 * x3051;
			x3049 = momentum;
			JCudaTensor x3052, x3053;
			x3052 = x2971;
			x3053 = x70;
			x86.backward_filter(x3052, x3053, x3047, x3048, x3049);

			// Dealloc(X4340)
			JCudaTensor x3054;
			x3054 = x2971;
			x3054.free();

			// cv11_W <~~ V_cv11_W
			float x3055, x3056;
			x3055 = 1;
			float x3057;
			float x3058;
			x3057 = 1;
			float x3059;
			float x3060;
			float x3061;
			float x3062;
			x3061 = 1;
			x3062 = decay;
			x3059 = x3061 * x3062;
			float x3063;
			float x3064;
			x3063 = 1;
			x3064 = lrn_rate;
			x3060 = x3063 * x3064;
			x3058 = x3059 * x3060;
			x3056 = x3057 + x3058;
			JCudaTensor x3065;
			x3065 = x2993;
			x67.update(x3065, x3055, x3056);

			// cv15_B <~~ V_cv15_B
			float x3066, x3067;
			x3066 = 1;
			x3067 = 1;
			JCudaTensor x3068;
			x3068 = x3026;
			x105.update(x3068, x3066, x3067);

			// cv13_B <~~ V_cv13_B
			float x3069, x3070;
			x3069 = 1;
			x3070 = 1;
			JCudaTensor x3071;
			x3071 = x3032;
			x98.update(x3071, x3069, x3070);

			// cv11_B <~~ V_cv11_B
			float x3072, x3073;
			x3072 = 1;
			x3073 = 1;
			JCudaTensor x3074;
			x3074 = x3007;
			x68.update(x3074, x3072, x3073);

			// cv13_W <~~ V_cv13_W
			float x3075, x3076;
			x3075 = 1;
			float x3077;
			float x3078;
			x3077 = 1;
			float x3079;
			float x3080;
			float x3081;
			float x3082;
			x3081 = 1;
			x3082 = decay;
			x3079 = x3081 * x3082;
			float x3083;
			float x3084;
			x3083 = 1;
			x3084 = lrn_rate;
			x3080 = x3083 * x3084;
			x3078 = x3079 * x3080;
			x3076 = x3077 + x3078;
			JCudaTensor x3085;
			x3085 = x3000;
			x97.update(x3085, x3075, x3076);

			// cv16_B <~~ V_cv16_B
			float x3086, x3087;
			x3086 = 1;
			x3087 = 1;
			JCudaTensor x3088;
			x3088 = x2987;
			x85.update(x3088, x3086, x3087);

			// cv15_W <~~ V_cv15_W
			float x3089, x3090;
			x3089 = 1;
			float x3091;
			float x3092;
			x3091 = 1;
			float x3093;
			float x3094;
			float x3095;
			float x3096;
			x3095 = 1;
			x3096 = decay;
			x3093 = x3095 * x3096;
			float x3097;
			float x3098;
			x3097 = 1;
			x3098 = lrn_rate;
			x3094 = x3097 * x3098;
			x3092 = x3093 * x3094;
			x3090 = x3091 + x3092;
			JCudaTensor x3099;
			x3099 = x3039;
			x104.update(x3099, x3089, x3090);

			// cv16_W <~~ V_cv16_W
			float x3100, x3101;
			x3100 = 1;
			float x3102;
			float x3103;
			x3102 = 1;
			float x3104;
			float x3105;
			float x3106;
			float x3107;
			x3106 = 1;
			x3107 = decay;
			x3104 = x3106 * x3107;
			float x3108;
			float x3109;
			x3108 = 1;
			x3109 = lrn_rate;
			x3105 = x3108 * x3109;
			x3103 = x3104 * x3105;
			x3101 = x3102 + x3103;
			JCudaTensor x3110;
			x3110 = x3047;
			x84.update(x3110, x3100, x3101);

			// val X4296 = X4294 * d_ReLU()(X7556)/d_X7555
			JCudaTensor x3111;
			JCudaTensor x3112, x3113;
			x3112 = x3023;
			x3113 = x87;
			x3111 = x89.backward(x3112, x3113);

			// Dealloc(X7556)
			JCudaTensor x3114;
			x3114 = x87;
			x3114.free();

			// val X4320 = X4318 * d_ReLU()(X7560)/d_X7559
			JCudaTensor x3115;
			JCudaTensor x3116, x3117;
			x3116 = x3020;
			x3117 = x90;
			x3115 = x92.backward(x3116, x3117);

			// Dealloc(X7560)
			JCudaTensor x3118;
			x3118 = x90;
			x3118.free();

			// V_cv14_W <~~ X4320 * d_Convolv(1,0)(X7552)/d_cv14_W
			float x3120, x3121;
			float x3122;
			float x3123;
			x3122 = 1;
			x3123 = lrn_rate;
			x3120 = x3122 * x3123;
			x3121 = momentum;
			JCudaTensor x3124, x3125;
			x3124 = x3115;
			x3125 = x53;
			x62.backward_filter(x3124, x3125, x3119, x3120, x3121);

			// V_cv12_B <~~ X4296 * d_Convolv(1,0)()/d_cv12_B
			float x3127, x3128;
			float x3129;
			float x3130;
			x3129 = 2;
			x3130 = lrn_rate;
			x3127 = x3129 * x3130;
			x3128 = momentum;
			JCudaTensor x3131;
			x3131 = x3111;
			x79.backward_bias(x3131, x3126, x3127, x3128);

			// val X4298 = (X4274 + X4296 * d_Convolv(1,0)(cv12_W)/d_X7552)
			JCudaTensor x3132;
			JCudaTensor x3133;
			x3133 = x3016;
			JCudaTensor x3134, x3135;
			x3134 = x3111;
			x3135 = x77;
			x3132 = x79.backward_data(x3134,x3135, x3133);

			// V_cv14_B <~~ X4320 * d_Convolv(1,0)()/d_cv14_B
			float x3137, x3138;
			float x3139;
			float x3140;
			x3139 = 2;
			x3140 = lrn_rate;
			x3137 = x3139 * x3140;
			x3138 = momentum;
			JCudaTensor x3141;
			x3141 = x3115;
			x62.backward_bias(x3141, x3136, x3137, x3138);

			// V_cv12_W <~~ X4296 * d_Convolv(1,0)(X7552)/d_cv12_W
			float x3143, x3144;
			float x3145;
			float x3146;
			x3145 = 1;
			x3146 = lrn_rate;
			x3143 = x3145 * x3146;
			x3144 = momentum;
			JCudaTensor x3147, x3148;
			x3147 = x3111;
			x3148 = x53;
			x79.backward_filter(x3147, x3148, x3142, x3143, x3144);

			// Dealloc(X4296)
			JCudaTensor x3149;
			x3149 = x3111;
			x3149.free();

			// cv12_B <~~ V_cv12_B
			float x3150, x3151;
			x3150 = 1;
			x3151 = 1;
			JCudaTensor x3152;
			x3152 = x3126;
			x78.update(x3152, x3150, x3151);

			// cv14_B <~~ V_cv14_B
			float x3153, x3154;
			x3153 = 1;
			x3154 = 1;
			JCudaTensor x3155;
			x3155 = x3136;
			x61.update(x3155, x3153, x3154);

			// cv12_W <~~ V_cv12_W
			float x3156, x3157;
			x3156 = 1;
			float x3158;
			float x3159;
			x3158 = 1;
			float x3160;
			float x3161;
			float x3162;
			float x3163;
			x3162 = 1;
			x3163 = decay;
			x3160 = x3162 * x3163;
			float x3164;
			float x3165;
			x3164 = 1;
			x3165 = lrn_rate;
			x3161 = x3164 * x3165;
			x3159 = x3160 * x3161;
			x3157 = x3158 + x3159;
			JCudaTensor x3166;
			x3166 = x3142;
			x77.update(x3166, x3156, x3157);

			// val X4322 = (X4298 + X4320 * d_Convolv(1,0)(cv14_W)/d_X7552)
			JCudaTensor x3167;
			JCudaTensor x3168;
			x3168 = x3132;
			JCudaTensor x3169, x3170;
			x3169 = x3115;
			x3170 = x60;
			x3167 = x62.backward_data(x3169,x3170, x3168);

			// Dealloc(X4320)
			JCudaTensor x3171;
			x3171 = x3115;
			x3171.free();

			// cv14_W <~~ V_cv14_W
			float x3172, x3173;
			x3172 = 1;
			float x3174;
			float x3175;
			x3174 = 1;
			float x3176;
			float x3177;
			float x3178;
			float x3179;
			x3178 = 1;
			x3179 = decay;
			x3176 = x3178 * x3179;
			float x3180;
			float x3181;
			x3180 = 1;
			x3181 = lrn_rate;
			x3177 = x3180 * x3181;
			x3175 = x3176 * x3177;
			x3173 = x3174 + x3175;
			JCudaTensor x3182;
			x3182 = x3119;
			x60.update(x3182, x3172, x3173);

			// val X4344 = (X4322 + X4341 * d_Pooling(3,1,1,true)(X7563,X7552)/d_X7552)
			JCudaTensor x3183;
			JCudaTensor x3184;
			x3184 = x3167;
			JCudaTensor x3185, x3186, x3187;
			x3185 = x3013;
			x3186 = x70;
			x3187 = x53;
			x3183 = x72.backward(x3185,x3186,x3187, x3184);

			// Dealloc(X4341)
			JCudaTensor x3188;
			x3188 = x3013;
			x3188.free();

			// Dealloc(X7563)
			JCudaTensor x3189;
			x3189 = x70;
			x3189.free();

			// val X4346 = X4344 * d_Pooling(3,2,1,true)(X7552,X7551)/d_X7551
			JCudaTensor x3190;
			JCudaTensor x3191, x3192, x3193;
			x3191 = x3183;
			x3192 = x53;
			x3193 = x50;
			x3190 = x55.backward(x3191, x3192, x3193);

			// Dealloc(X4344)
			JCudaTensor x3194;
			x3194 = x3183;
			x3194.free();

			// Dealloc(X7552)
			JCudaTensor x3195;
			x3195 = x53;
			x3195.free();

			// val X4348 = X4346 * d_LRN(5,1.0E-4,0.75)(X7551,X7550)/d_X7550
			JCudaTensor x3196;
			JCudaTensor x3197, x3198, x3199;
			x3197 = x3190;
			x3198 = x50;
			x3199 = x47;
			x3196 = x52.backward(x3197, x3198, x3199);

			// Dealloc(X7551)
			JCudaTensor x3200;
			x3200 = x50;
			x3200.free();

			// val X4350 = X4348 * d_ReLU()(X7550)/d_X7549
			JCudaTensor x3201;
			JCudaTensor x3202, x3203;
			x3202 = x3196;
			x3203 = x47;
			x3201 = x49.backward(x3202, x3203);

			// Dealloc(X7550)
			JCudaTensor x3204;
			x3204 = x47;
			x3204.free();

			// val X4351 = X4350 * d_Convolv(1,1)(cv3_W)/d_X7548
			JCudaTensor x3205;
			JCudaTensor x3206, x3207;
			x3206 = x3201;
			x3207 = x44;
			x3205 = x46.backward_data(x3206, x3207);

			// V_cv3_B <~~ X4350 * d_Convolv(1,1)()/d_cv3_B
			float x3209, x3210;
			float x3211;
			float x3212;
			x3211 = 2;
			x3212 = lrn_rate;
			x3209 = x3211 * x3212;
			x3210 = momentum;
			JCudaTensor x3213;
			x3213 = x3201;
			x46.backward_bias(x3213, x3208, x3209, x3210);

			// V_cv3_W <~~ X4350 * d_Convolv(1,1)(X7548)/d_cv3_W
			float x3215, x3216;
			float x3217;
			float x3218;
			x3217 = 1;
			x3218 = lrn_rate;
			x3215 = x3217 * x3218;
			x3216 = momentum;
			JCudaTensor x3219, x3220;
			x3219 = x3201;
			x3220 = x37;
			x46.backward_filter(x3219, x3220, x3214, x3215, x3216);

			// Dealloc(X4350)
			JCudaTensor x3221;
			x3221 = x3201;
			x3221.free();

			// cv3_B <~~ V_cv3_B
			float x3222, x3223;
			x3222 = 1;
			x3223 = 1;
			JCudaTensor x3224;
			x3224 = x3208;
			x45.update(x3224, x3222, x3223);

			// cv3_W <~~ V_cv3_W
			float x3225, x3226;
			x3225 = 1;
			float x3227;
			float x3228;
			x3227 = 1;
			float x3229;
			float x3230;
			float x3231;
			float x3232;
			x3231 = 1;
			x3232 = decay;
			x3229 = x3231 * x3232;
			float x3233;
			float x3234;
			x3233 = 1;
			x3234 = lrn_rate;
			x3230 = x3233 * x3234;
			x3228 = x3229 * x3230;
			x3226 = x3227 + x3228;
			JCudaTensor x3235;
			x3235 = x3214;
			x44.update(x3235, x3225, x3226);

			// val X4353 = X4351 * d_ReLU()(X7548)/d_X7547
			JCudaTensor x3236;
			JCudaTensor x3237, x3238;
			x3237 = x3205;
			x3238 = x37;
			x3236 = x39.backward(x3237, x3238);

			// Dealloc(X7548)
			JCudaTensor x3239;
			x3239 = x37;
			x3239.free();

			// val X4354 = X4353 * d_Convolv(1,0)(cv2_W)/d_X7546
			JCudaTensor x3240;
			JCudaTensor x3241, x3242;
			x3241 = x3236;
			x3242 = x34;
			x3240 = x36.backward_data(x3241, x3242);

			// V_cv2_B <~~ X4353 * d_Convolv(1,0)()/d_cv2_B
			float x3244, x3245;
			float x3246;
			float x3247;
			x3246 = 2;
			x3247 = lrn_rate;
			x3244 = x3246 * x3247;
			x3245 = momentum;
			JCudaTensor x3248;
			x3248 = x3236;
			x36.backward_bias(x3248, x3243, x3244, x3245);

			// V_cv2_W <~~ X4353 * d_Convolv(1,0)(X7546)/d_cv2_W
			float x3250, x3251;
			float x3252;
			float x3253;
			x3252 = 1;
			x3253 = lrn_rate;
			x3250 = x3252 * x3253;
			x3251 = momentum;
			JCudaTensor x3254, x3255;
			x3254 = x3236;
			x3255 = x27;
			x36.backward_filter(x3254, x3255, x3249, x3250, x3251);

			// Dealloc(X4353)
			JCudaTensor x3256;
			x3256 = x3236;
			x3256.free();

			// cv2_B <~~ V_cv2_B
			float x3257, x3258;
			x3257 = 1;
			x3258 = 1;
			JCudaTensor x3259;
			x3259 = x3243;
			x35.update(x3259, x3257, x3258);

			// cv2_W <~~ V_cv2_W
			float x3260, x3261;
			x3260 = 1;
			float x3262;
			float x3263;
			x3262 = 1;
			float x3264;
			float x3265;
			float x3266;
			float x3267;
			x3266 = 1;
			x3267 = decay;
			x3264 = x3266 * x3267;
			float x3268;
			float x3269;
			x3268 = 1;
			x3269 = lrn_rate;
			x3265 = x3268 * x3269;
			x3263 = x3264 * x3265;
			x3261 = x3262 + x3263;
			JCudaTensor x3270;
			x3270 = x3249;
			x34.update(x3270, x3260, x3261);

			// val X4356 = X4354 * d_LRN(5,1.0E-4,0.75)(X7546,X7545)/d_X7545
			JCudaTensor x3271;
			JCudaTensor x3272, x3273, x3274;
			x3272 = x3240;
			x3273 = x27;
			x3274 = x24;
			x3271 = x29.backward(x3272, x3273, x3274);

			// Dealloc(X7546)
			JCudaTensor x3275;
			x3275 = x27;
			x3275.free();

			// val X4358 = X4356 * d_Pooling(3,2,1,true)(X7545,X7544)/d_X7544
			JCudaTensor x3276;
			JCudaTensor x3277, x3278, x3279;
			x3277 = x3271;
			x3278 = x24;
			x3279 = x21;
			x3276 = x26.backward(x3277, x3278, x3279);

			// Dealloc(X4356)
			JCudaTensor x3280;
			x3280 = x3271;
			x3280.free();

			// Dealloc(X7545)
			JCudaTensor x3281;
			x3281 = x24;
			x3281.free();

			// val X4360 = X4358 * d_ReLU()(X7544)/d_X7543
			JCudaTensor x3282;
			JCudaTensor x3283, x3284;
			x3283 = x3276;
			x3284 = x21;
			x3282 = x23.backward(x3283, x3284);

			// Dealloc(X7544)
			JCudaTensor x3285;
			x3285 = x21;
			x3285.free();

			// V_cv1_W <~~ X4360 * d_Convolv(2,3)(X7542)/d_cv1_W
			float x3287, x3288;
			float x3289;
			float x3290;
			x3289 = 1;
			x3290 = lrn_rate;
			x3287 = x3289 * x3290;
			x3288 = momentum;
			JCudaTensor x3291, x3292;
			x3291 = x3282;
			x3292 = x7;
			x17.backward_filter(x3291, x3292, x3286, x3287, x3288);

			// Dealloc(X7542)
			JCudaTensor x3293;
			x3293 = x7;
			x3293.free();

			// V_cv1_B <~~ X4360 * d_Convolv(2,3)()/d_cv1_B
			float x3295, x3296;
			float x3297;
			float x3298;
			x3297 = 2;
			x3298 = lrn_rate;
			x3295 = x3297 * x3298;
			x3296 = momentum;
			JCudaTensor x3299;
			x3299 = x3282;
			x17.backward_bias(x3299, x3294, x3295, x3296);

			// Dealloc(X4360)
			JCudaTensor x3300;
			x3300 = x3282;
			x3300.free();

			// cv1_W <~~ V_cv1_W
			float x3301, x3302;
			x3301 = 1;
			float x3303;
			float x3304;
			x3303 = 1;
			float x3305;
			float x3306;
			float x3307;
			float x3308;
			x3307 = 1;
			x3308 = decay;
			x3305 = x3307 * x3308;
			float x3309;
			float x3310;
			x3309 = 1;
			x3310 = lrn_rate;
			x3306 = x3309 * x3310;
			x3304 = x3305 * x3306;
			x3302 = x3303 + x3304;
			JCudaTensor x3311;
			x3311 = x3286;
			x15.update(x3311, x3301, x3302);

			// cv1_B <~~ V_cv1_B
			float x3312, x3313;
			x3312 = 1;
			x3313 = 1;
			JCudaTensor x3314;
			x3314 = x3294;
			x16.update(x3314, x3312, x3313);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X7714 = Cuda(X)
			JCudaTensor x3315;
			JTensorFloat x3316;
			x3316 = x3;
			x3315 = x3316.asJCudaTensor();

			// val X7715 = Convolv(2,3)(X7714,cv1_W,cv1_B)
			JCudaTensor x3317;
			JCudaTensor x3318, x3319, x3320;
			x3318 = x3315;
			x3319 = x15;
			x3320 = x16;
			x3317 = x17.forward(x3318, x3319, x3320);

			// Dealloc(X7714)
			JCudaTensor x3321;
			x3321 = x3315;
			x3321.free();

			// val X7716 = ReLU()(X7715)
			JCudaTensor x3322;
			JCudaTensor x3323;
			x3323 = x3317;
			x3322 = x23.forward(x3323);

			// val X7717 = Pooling(3,2,1,true)(X7716)
			JCudaTensor x3324;
			JCudaTensor x3325;
			x3325 = x3322;
			x3324 = x26.forward(x3325);

			// Dealloc(X7716)
			JCudaTensor x3326;
			x3326 = x3322;
			x3326.free();

			// val X7718 = LRN(5,1.0E-4,0.75)(X7717)
			JCudaTensor x3327;
			JCudaTensor x3328;
			x3328 = x3324;
			x3327 = x29.forward(x3328);

			// Dealloc(X7717)
			JCudaTensor x3329;
			x3329 = x3324;
			x3329.free();

			// val X7719 = Convolv(1,0)(X7718,cv2_W,cv2_B)
			JCudaTensor x3330;
			JCudaTensor x3331, x3332, x3333;
			x3331 = x3327;
			x3332 = x34;
			x3333 = x35;
			x3330 = x36.forward(x3331, x3332, x3333);

			// Dealloc(X7718)
			JCudaTensor x3334;
			x3334 = x3327;
			x3334.free();

			// val X7720 = ReLU()(X7719)
			JCudaTensor x3335;
			JCudaTensor x3336;
			x3336 = x3330;
			x3335 = x39.forward(x3336);

			// val X7721 = Convolv(1,1)(X7720,cv3_W,cv3_B)
			JCudaTensor x3337;
			JCudaTensor x3338, x3339, x3340;
			x3338 = x3335;
			x3339 = x44;
			x3340 = x45;
			x3337 = x46.forward(x3338, x3339, x3340);

			// Dealloc(X7720)
			JCudaTensor x3341;
			x3341 = x3335;
			x3341.free();

			// val X7722 = ReLU()(X7721)
			JCudaTensor x3342;
			JCudaTensor x3343;
			x3343 = x3337;
			x3342 = x49.forward(x3343);

			// val X7723 = LRN(5,1.0E-4,0.75)(X7722)
			JCudaTensor x3344;
			JCudaTensor x3345;
			x3345 = x3342;
			x3344 = x52.forward(x3345);

			// Dealloc(X7722)
			JCudaTensor x3346;
			x3346 = x3342;
			x3346.free();

			// val X7724 = Pooling(3,2,1,true)(X7723)
			JCudaTensor x3347;
			JCudaTensor x3348;
			x3348 = x3344;
			x3347 = x55.forward(x3348);

			// Dealloc(X7723)
			JCudaTensor x3349;
			x3349 = x3344;
			x3349.free();

			// val X7735 = Pooling(3,1,1,true)(X7724)
			JCudaTensor x3350;
			JCudaTensor x3351;
			x3351 = x3347;
			x3350 = x72.forward(x3351);

			// val X7725 = Convolv(1,0)(X7724,cv11_W,cv11_B)
			JCudaTensor x3352;
			JCudaTensor x3353, x3354, x3355;
			x3353 = x3347;
			x3354 = x67;
			x3355 = x68;
			x3352 = x69.forward(x3353, x3354, x3355);

			// val X7727 = Convolv(1,0)(X7724,cv12_W,cv12_B)
			JCudaTensor x3356;
			JCudaTensor x3357, x3358, x3359;
			x3357 = x3347;
			x3358 = x77;
			x3359 = x78;
			x3356 = x79.forward(x3357, x3358, x3359);

			// val X7731 = Convolv(1,0)(X7724,cv14_W,cv14_B)
			JCudaTensor x3360;
			JCudaTensor x3361, x3362, x3363;
			x3361 = x3347;
			x3362 = x60;
			x3363 = x61;
			x3360 = x62.forward(x3361, x3362, x3363);

			// Dealloc(X7724)
			JCudaTensor x3364;
			x3364 = x3347;
			x3364.free();

			// val X7732 = ReLU()(X7731)
			JCudaTensor x3365;
			JCudaTensor x3366;
			x3366 = x3360;
			x3365 = x92.forward(x3366);

			// val X7736 = Convolv(1,0)(X7735,cv16_W,cv16_B)
			JCudaTensor x3367;
			JCudaTensor x3368, x3369, x3370;
			x3368 = x3350;
			x3369 = x84;
			x3370 = x85;
			x3367 = x86.forward(x3368, x3369, x3370);

			// Dealloc(X7735)
			JCudaTensor x3371;
			x3371 = x3350;
			x3371.free();

			// val X7728 = ReLU()(X7727)
			JCudaTensor x3372;
			JCudaTensor x3373;
			x3373 = x3356;
			x3372 = x89.forward(x3373);

			// val X7729 = Convolv(1,1)(X7728,cv13_W,cv13_B)
			JCudaTensor x3374;
			JCudaTensor x3375, x3376, x3377;
			x3375 = x3372;
			x3376 = x97;
			x3377 = x98;
			x3374 = x99.forward(x3375, x3376, x3377);

			// Dealloc(X7728)
			JCudaTensor x3378;
			x3378 = x3372;
			x3378.free();

			// val X7733 = Convolv(1,2)(X7732,cv15_W,cv15_B)
			JCudaTensor x3379;
			JCudaTensor x3380, x3381, x3382;
			x3380 = x3365;
			x3381 = x104;
			x3382 = x105;
			x3379 = x106.forward(x3380, x3381, x3382);

			// Dealloc(X7732)
			JCudaTensor x3383;
			x3383 = x3365;
			x3383.free();

			// val X7726 = ReLU()(X7725)
			JCudaTensor x3384;
			JCudaTensor x3385;
			x3385 = x3352;
			x3384 = x109.forward(x3385);

			// val X7730 = ReLU()(X7729)
			JCudaTensor x3386;
			JCudaTensor x3387;
			x3387 = x3374;
			x3386 = x112.forward(x3387);

			// val X7734 = ReLU()(X7733)
			JCudaTensor x3388;
			JCudaTensor x3389;
			x3389 = x3379;
			x3388 = x115.forward(x3389);

			// val X7737 = ReLU()(X7736)
			JCudaTensor x3390;
			JCudaTensor x3391;
			x3391 = x3367;
			x3390 = x115.forward(x3391);

			// val X7738 = Concat(X7726,X7730,X7734,X7737)
			JCudaTensor x3392;
			JCudaTensor x3393, x3394, x3395, x3396;
			x3393 = x3384;
			x3394 = x3386;
			x3395 = x3388;
			x3396 = x3390;
			x3392 = x119.forward(x3393,x3394,x3395,x3396);

			// Dealloc(X7737)
			JCudaTensor x3397;
			x3397 = x3390;
			x3397.free();

			// Dealloc(X7734)
			JCudaTensor x3398;
			x3398 = x3388;
			x3398.free();

			// Dealloc(X7730)
			JCudaTensor x3399;
			x3399 = x3386;
			x3399.free();

			// Dealloc(X7726)
			JCudaTensor x3400;
			x3400 = x3384;
			x3400.free();

			// val X7739 = Convolv(1,0)(X7738,cv21_W,cv21_B)
			JCudaTensor x3401;
			JCudaTensor x3402, x3403, x3404;
			x3402 = x3392;
			x3403 = x145;
			x3404 = x146;
			x3401 = x147.forward(x3402, x3403, x3404);

			// val X7745 = Convolv(1,0)(X7738,cv24_W,cv24_B)
			JCudaTensor x3405;
			JCudaTensor x3406, x3407, x3408;
			x3406 = x3392;
			x3407 = x131;
			x3408 = x132;
			x3405 = x133.forward(x3406, x3407, x3408);

			// val X7749 = Pooling(3,1,1,true)(X7738)
			JCudaTensor x3409;
			JCudaTensor x3410;
			x3410 = x3392;
			x3409 = x126.forward(x3410);

			// val X7741 = Convolv(1,0)(X7738,cv22_W,cv22_B)
			JCudaTensor x3411;
			JCudaTensor x3412, x3413, x3414;
			x3412 = x3392;
			x3413 = x138;
			x3414 = x139;
			x3411 = x140.forward(x3412, x3413, x3414);

			// Dealloc(X7738)
			JCudaTensor x3415;
			x3415 = x3392;
			x3415.free();

			// val X7742 = ReLU()(X7741)
			JCudaTensor x3416;
			JCudaTensor x3417;
			x3417 = x3411;
			x3416 = x89.forward(x3417);

			// val X7746 = ReLU()(X7745)
			JCudaTensor x3418;
			JCudaTensor x3419;
			x3419 = x3405;
			x3418 = x92.forward(x3419);

			// val X7750 = Convolv(1,0)(X7749,cv26_W,cv26_B)
			JCudaTensor x3420;
			JCudaTensor x3421, x3422, x3423;
			x3421 = x3409;
			x3422 = x152;
			x3423 = x153;
			x3420 = x154.forward(x3421, x3422, x3423);

			// Dealloc(X7749)
			JCudaTensor x3424;
			x3424 = x3409;
			x3424.free();

			// val X7743 = Convolv(1,1)(X7742,cv23_W,cv23_B)
			JCudaTensor x3425;
			JCudaTensor x3426, x3427, x3428;
			x3426 = x3416;
			x3427 = x169;
			x3428 = x170;
			x3425 = x99.forward(x3426, x3427, x3428);

			// Dealloc(X7742)
			JCudaTensor x3429;
			x3429 = x3416;
			x3429.free();

			// val X7747 = Convolv(1,2)(X7746,cv25_W,cv25_B)
			JCudaTensor x3430;
			JCudaTensor x3431, x3432, x3433;
			x3431 = x3418;
			x3432 = x163;
			x3433 = x164;
			x3430 = x106.forward(x3431, x3432, x3433);

			// Dealloc(X7746)
			JCudaTensor x3434;
			x3434 = x3418;
			x3434.free();

			// val X7740 = ReLU()(X7739)
			JCudaTensor x3435;
			JCudaTensor x3436;
			x3436 = x3401;
			x3435 = x109.forward(x3436);

			// val X7744 = ReLU()(X7743)
			JCudaTensor x3437;
			JCudaTensor x3438;
			x3438 = x3425;
			x3437 = x112.forward(x3438);

			// val X7748 = ReLU()(X7747)
			JCudaTensor x3439;
			JCudaTensor x3440;
			x3440 = x3430;
			x3439 = x115.forward(x3440);

			// val X7751 = ReLU()(X7750)
			JCudaTensor x3441;
			JCudaTensor x3442;
			x3442 = x3420;
			x3441 = x115.forward(x3442);

			// val X7752 = Concat(X7740,X7744,X7748,X7751)
			JCudaTensor x3443;
			JCudaTensor x3444, x3445, x3446, x3447;
			x3444 = x3435;
			x3445 = x3437;
			x3446 = x3439;
			x3447 = x3441;
			x3443 = x119.forward(x3444,x3445,x3446,x3447);

			// Dealloc(X7751)
			JCudaTensor x3448;
			x3448 = x3441;
			x3448.free();

			// Dealloc(X7748)
			JCudaTensor x3449;
			x3449 = x3439;
			x3449.free();

			// Dealloc(X7744)
			JCudaTensor x3450;
			x3450 = x3437;
			x3450.free();

			// Dealloc(X7740)
			JCudaTensor x3451;
			x3451 = x3435;
			x3451.free();

			// val X7753 = Pooling(3,2,1,true)(X7752)
			JCudaTensor x3452;
			JCudaTensor x3453;
			x3453 = x3443;
			x3452 = x186.forward(x3453);

			// Dealloc(X7752)
			JCudaTensor x3454;
			x3454 = x3443;
			x3454.free();

			// val X7764 = Pooling(3,1,1,true)(X7753)
			JCudaTensor x3455;
			JCudaTensor x3456;
			x3456 = x3452;
			x3455 = x203.forward(x3456);

			// val X7756 = Convolv(1,0)(X7753,cv32_W,cv32_B)
			JCudaTensor x3457;
			JCudaTensor x3458, x3459, x3460;
			x3458 = x3452;
			x3459 = x208;
			x3460 = x209;
			x3457 = x210.forward(x3458, x3459, x3460);

			// val X7760 = Convolv(1,0)(X7753,cv34_W,cv34_B)
			JCudaTensor x3461;
			JCudaTensor x3462, x3463, x3464;
			x3462 = x3452;
			x3463 = x191;
			x3464 = x192;
			x3461 = x193.forward(x3462, x3463, x3464);

			// val X7754 = Convolv(1,0)(X7753,cv31_W,cv31_B)
			JCudaTensor x3465;
			JCudaTensor x3466, x3467, x3468;
			x3466 = x3452;
			x3467 = x198;
			x3468 = x199;
			x3465 = x200.forward(x3466, x3467, x3468);

			// Dealloc(X7753)
			JCudaTensor x3469;
			x3469 = x3452;
			x3469.free();

			// val X7757 = ReLU()(X7756)
			JCudaTensor x3470;
			JCudaTensor x3471;
			x3471 = x3457;
			x3470 = x213.forward(x3471);

			// val X7765 = Convolv(1,0)(X7764,cv36_W,cv36_B)
			JCudaTensor x3472;
			JCudaTensor x3473, x3474, x3475;
			x3473 = x3455;
			x3474 = x218;
			x3475 = x219;
			x3472 = x220.forward(x3473, x3474, x3475);

			// Dealloc(X7764)
			JCudaTensor x3476;
			x3476 = x3455;
			x3476.free();

			// val X7761 = ReLU()(X7760)
			JCudaTensor x3477;
			JCudaTensor x3478;
			x3478 = x3461;
			x3477 = x223.forward(x3478);

			// val X7762 = Convolv(1,2)(X7761,cv35_W,cv35_B)
			JCudaTensor x3479;
			JCudaTensor x3480, x3481, x3482;
			x3480 = x3477;
			x3481 = x228;
			x3482 = x229;
			x3479 = x230.forward(x3480, x3481, x3482);

			// Dealloc(X7761)
			JCudaTensor x3483;
			x3483 = x3477;
			x3483.free();

			// val X7758 = Convolv(1,1)(X7757,cv33_W,cv33_B)
			JCudaTensor x3484;
			JCudaTensor x3485, x3486, x3487;
			x3485 = x3470;
			x3486 = x235;
			x3487 = x236;
			x3484 = x237.forward(x3485, x3486, x3487);

			// Dealloc(X7757)
			JCudaTensor x3488;
			x3488 = x3470;
			x3488.free();

			// val X7755 = ReLU()(X7754)
			JCudaTensor x3489;
			JCudaTensor x3490;
			x3490 = x3465;
			x3489 = x240.forward(x3490);

			// val X7759 = ReLU()(X7758)
			JCudaTensor x3491;
			JCudaTensor x3492;
			x3492 = x3484;
			x3491 = x243.forward(x3492);

			// val X7763 = ReLU()(X7762)
			JCudaTensor x3493;
			JCudaTensor x3494;
			x3494 = x3479;
			x3493 = x246.forward(x3494);

			// val X7766 = ReLU()(X7765)
			JCudaTensor x3495;
			JCudaTensor x3496;
			x3496 = x3472;
			x3495 = x246.forward(x3496);

			// val X7767 = Concat(X7755,X7759,X7763,X7766)
			JCudaTensor x3497;
			JCudaTensor x3498, x3499, x3500, x3501;
			x3498 = x3489;
			x3499 = x3491;
			x3500 = x3493;
			x3501 = x3495;
			x3497 = x250.forward(x3498,x3499,x3500,x3501);

			// Dealloc(X7766)
			JCudaTensor x3502;
			x3502 = x3495;
			x3502.free();

			// Dealloc(X7763)
			JCudaTensor x3503;
			x3503 = x3493;
			x3503.free();

			// Dealloc(X7759)
			JCudaTensor x3504;
			x3504 = x3491;
			x3504.free();

			// Dealloc(X7755)
			JCudaTensor x3505;
			x3505 = x3489;
			x3505.free();

			// val X7778 = Pooling(3,1,1,true)(X7767)
			JCudaTensor x3506;
			JCudaTensor x3507;
			x3507 = x3497;
			x3506 = x203.forward(x3507);

			// val X7770 = Convolv(1,0)(X7767,cv42_W,cv42_B)
			JCudaTensor x3508;
			JCudaTensor x3509, x3510, x3511;
			x3509 = x3497;
			x3510 = x259;
			x3511 = x260;
			x3508 = x210.forward(x3509, x3510, x3511);

			// val X7774 = Convolv(1,0)(X7767,cv44_W,cv44_B)
			JCudaTensor x3512;
			JCudaTensor x3513, x3514, x3515;
			x3513 = x3497;
			x3514 = x276;
			x3515 = x277;
			x3512 = x193.forward(x3513, x3514, x3515);

			// val X7768 = Convolv(1,0)(X7767,cv41_W,cv41_B)
			JCudaTensor x3516;
			JCudaTensor x3517, x3518, x3519;
			x3517 = x3497;
			x3518 = x270;
			x3519 = x271;
			x3516 = x200.forward(x3517, x3518, x3519);

			// Dealloc(X7767)
			JCudaTensor x3520;
			x3520 = x3497;
			x3520.free();

			// val X7775 = ReLU()(X7774)
			JCudaTensor x3521;
			JCudaTensor x3522;
			x3522 = x3512;
			x3521 = x223.forward(x3522);

			// val X7779 = Convolv(1,0)(X7778,cv46_W,cv46_B)
			JCudaTensor x3523;
			JCudaTensor x3524, x3525, x3526;
			x3524 = x3506;
			x3525 = x284;
			x3526 = x285;
			x3523 = x220.forward(x3524, x3525, x3526);

			// Dealloc(X7778)
			JCudaTensor x3527;
			x3527 = x3506;
			x3527.free();

			// val X7771 = ReLU()(X7770)
			JCudaTensor x3528;
			JCudaTensor x3529;
			x3529 = x3508;
			x3528 = x213.forward(x3529);

			// val X7776 = Convolv(1,2)(X7775,cv45_W,cv45_B)
			JCudaTensor x3530;
			JCudaTensor x3531, x3532, x3533;
			x3531 = x3521;
			x3532 = x299;
			x3533 = x300;
			x3530 = x230.forward(x3531, x3532, x3533);

			// Dealloc(X7775)
			JCudaTensor x3534;
			x3534 = x3521;
			x3534.free();

			// val X7772 = Convolv(1,1)(X7771,cv43_W,cv43_B)
			JCudaTensor x3535;
			JCudaTensor x3536, x3537, x3538;
			x3536 = x3528;
			x3537 = x305;
			x3538 = x306;
			x3535 = x237.forward(x3536, x3537, x3538);

			// Dealloc(X7771)
			JCudaTensor x3539;
			x3539 = x3528;
			x3539.free();

			// val X7769 = ReLU()(X7768)
			JCudaTensor x3540;
			JCudaTensor x3541;
			x3541 = x3516;
			x3540 = x240.forward(x3541);

			// val X7773 = ReLU()(X7772)
			JCudaTensor x3542;
			JCudaTensor x3543;
			x3543 = x3535;
			x3542 = x243.forward(x3543);

			// val X7777 = ReLU()(X7776)
			JCudaTensor x3544;
			JCudaTensor x3545;
			x3545 = x3530;
			x3544 = x246.forward(x3545);

			// val X7780 = ReLU()(X7779)
			JCudaTensor x3546;
			JCudaTensor x3547;
			x3547 = x3523;
			x3546 = x246.forward(x3547);

			// val X7781 = Concat(X7769,X7773,X7777,X7780)
			JCudaTensor x3548;
			JCudaTensor x3549, x3550, x3551, x3552;
			x3549 = x3540;
			x3550 = x3542;
			x3551 = x3544;
			x3552 = x3546;
			x3548 = x250.forward(x3549,x3550,x3551,x3552);

			// Dealloc(X7780)
			JCudaTensor x3553;
			x3553 = x3546;
			x3553.free();

			// Dealloc(X7777)
			JCudaTensor x3554;
			x3554 = x3544;
			x3554.free();

			// Dealloc(X7773)
			JCudaTensor x3555;
			x3555 = x3542;
			x3555.free();

			// Dealloc(X7769)
			JCudaTensor x3556;
			x3556 = x3540;
			x3556.free();

			// val X7788 = Convolv(1,0)(X7781,cv54_W,cv54_B)
			JCudaTensor x3557;
			JCudaTensor x3558, x3559, x3560;
			x3558 = x3548;
			x3559 = x338;
			x3560 = x339;
			x3557 = x193.forward(x3558, x3559, x3560);

			// val X7792 = Pooling(3,1,1,true)(X7781)
			JCudaTensor x3561;
			JCudaTensor x3562;
			x3562 = x3548;
			x3561 = x203.forward(x3562);

			// val X7784 = Convolv(1,0)(X7781,cv52_W,cv52_B)
			JCudaTensor x3563;
			JCudaTensor x3564, x3565, x3566;
			x3564 = x3548;
			x3565 = x344;
			x3566 = x345;
			x3563 = x210.forward(x3564, x3565, x3566);

			// val X7782 = Convolv(1,0)(X7781,cv51_W,cv51_B)
			JCudaTensor x3567;
			JCudaTensor x3568, x3569, x3570;
			x3568 = x3548;
			x3569 = x355;
			x3570 = x356;
			x3567 = x200.forward(x3568, x3569, x3570);

			// Dealloc(X7781)
			JCudaTensor x3571;
			x3571 = x3548;
			x3571.free();

			// val X7789 = ReLU()(X7788)
			JCudaTensor x3572;
			JCudaTensor x3573;
			x3573 = x3557;
			x3572 = x223.forward(x3573);

			// val X7793 = Convolv(1,0)(X7792,cv56_W,cv56_B)
			JCudaTensor x3574;
			JCudaTensor x3575, x3576, x3577;
			x3575 = x3561;
			x3576 = x368;
			x3577 = x369;
			x3574 = x220.forward(x3575, x3576, x3577);

			// Dealloc(X7792)
			JCudaTensor x3578;
			x3578 = x3561;
			x3578.free();

			// val X7785 = ReLU()(X7784)
			JCudaTensor x3579;
			JCudaTensor x3580;
			x3580 = x3563;
			x3579 = x213.forward(x3580);

			// val X7790 = Convolv(1,2)(X7789,cv55_W,cv55_B)
			JCudaTensor x3581;
			JCudaTensor x3582, x3583, x3584;
			x3582 = x3572;
			x3583 = x374;
			x3584 = x375;
			x3581 = x230.forward(x3582, x3583, x3584);

			// Dealloc(X7789)
			JCudaTensor x3585;
			x3585 = x3572;
			x3585.free();

			// val X7786 = Convolv(1,1)(X7785,cv53_W,cv53_B)
			JCudaTensor x3586;
			JCudaTensor x3587, x3588, x3589;
			x3587 = x3579;
			x3588 = x386;
			x3589 = x387;
			x3586 = x237.forward(x3587, x3588, x3589);

			// Dealloc(X7785)
			JCudaTensor x3590;
			x3590 = x3579;
			x3590.free();

			// val X7783 = ReLU()(X7782)
			JCudaTensor x3591;
			JCudaTensor x3592;
			x3592 = x3567;
			x3591 = x240.forward(x3592);

			// val X7787 = ReLU()(X7786)
			JCudaTensor x3593;
			JCudaTensor x3594;
			x3594 = x3586;
			x3593 = x243.forward(x3594);

			// val X7791 = ReLU()(X7790)
			JCudaTensor x3595;
			JCudaTensor x3596;
			x3596 = x3581;
			x3595 = x246.forward(x3596);

			// val X7794 = ReLU()(X7793)
			JCudaTensor x3597;
			JCudaTensor x3598;
			x3598 = x3574;
			x3597 = x246.forward(x3598);

			// val X7795 = Concat(X7783,X7787,X7791,X7794)
			JCudaTensor x3599;
			JCudaTensor x3600, x3601, x3602, x3603;
			x3600 = x3591;
			x3601 = x3593;
			x3602 = x3595;
			x3603 = x3597;
			x3599 = x250.forward(x3600,x3601,x3602,x3603);

			// Dealloc(X7794)
			JCudaTensor x3604;
			x3604 = x3597;
			x3604.free();

			// Dealloc(X7791)
			JCudaTensor x3605;
			x3605 = x3595;
			x3605.free();

			// Dealloc(X7787)
			JCudaTensor x3606;
			x3606 = x3593;
			x3606.free();

			// Dealloc(X7783)
			JCudaTensor x3607;
			x3607 = x3591;
			x3607.free();

			// val X7806 = Pooling(3,1,1,true)(X7795)
			JCudaTensor x3608;
			JCudaTensor x3609;
			x3609 = x3599;
			x3608 = x203.forward(x3609);

			// val X7802 = Convolv(1,0)(X7795,cv64_W,cv64_B)
			JCudaTensor x3610;
			JCudaTensor x3611, x3612, x3613;
			x3611 = x3599;
			x3612 = x436;
			x3613 = x437;
			x3610 = x193.forward(x3611, x3612, x3613);

			// val X7796 = Convolv(1,0)(X7795,cv61_W,cv61_B)
			JCudaTensor x3614;
			JCudaTensor x3615, x3616, x3617;
			x3615 = x3599;
			x3616 = x428;
			x3617 = x429;
			x3614 = x200.forward(x3615, x3616, x3617);

			// val X7798 = Convolv(1,0)(X7795,cv62_W,cv62_B)
			JCudaTensor x3618;
			JCudaTensor x3619, x3620, x3621;
			x3619 = x3599;
			x3620 = x422;
			x3621 = x423;
			x3618 = x210.forward(x3619, x3620, x3621);

			// Dealloc(X7795)
			JCudaTensor x3622;
			x3622 = x3599;
			x3622.free();

			// val X7799 = ReLU()(X7798)
			JCudaTensor x3623;
			JCudaTensor x3624;
			x3624 = x3618;
			x3623 = x213.forward(x3624);

			// val X7803 = ReLU()(X7802)
			JCudaTensor x3625;
			JCudaTensor x3626;
			x3626 = x3610;
			x3625 = x223.forward(x3626);

			// val X7807 = Convolv(1,0)(X7806,cv66_W,cv66_B)
			JCudaTensor x3627;
			JCudaTensor x3628, x3629, x3630;
			x3628 = x3608;
			x3629 = x444;
			x3630 = x445;
			x3627 = x220.forward(x3628, x3629, x3630);

			// Dealloc(X7806)
			JCudaTensor x3631;
			x3631 = x3608;
			x3631.free();

			// val X7800 = Convolv(1,1)(X7799,cv63_W,cv63_B)
			JCudaTensor x3632;
			JCudaTensor x3633, x3634, x3635;
			x3633 = x3623;
			x3634 = x476;
			x3635 = x477;
			x3632 = x237.forward(x3633, x3634, x3635);

			// Dealloc(X7799)
			JCudaTensor x3636;
			x3636 = x3623;
			x3636.free();

			// val X7804 = Convolv(1,2)(X7803,cv65_W,cv65_B)
			JCudaTensor x3637;
			JCudaTensor x3638, x3639, x3640;
			x3638 = x3625;
			x3639 = x467;
			x3640 = x468;
			x3637 = x230.forward(x3638, x3639, x3640);

			// Dealloc(X7803)
			JCudaTensor x3641;
			x3641 = x3625;
			x3641.free();

			// val X7797 = ReLU()(X7796)
			JCudaTensor x3642;
			JCudaTensor x3643;
			x3643 = x3614;
			x3642 = x240.forward(x3643);

			// val X7801 = ReLU()(X7800)
			JCudaTensor x3644;
			JCudaTensor x3645;
			x3645 = x3632;
			x3644 = x243.forward(x3645);

			// val X7805 = ReLU()(X7804)
			JCudaTensor x3646;
			JCudaTensor x3647;
			x3647 = x3637;
			x3646 = x246.forward(x3647);

			// val X7808 = ReLU()(X7807)
			JCudaTensor x3648;
			JCudaTensor x3649;
			x3649 = x3627;
			x3648 = x246.forward(x3649);

			// val X7809 = Concat(X7797,X7801,X7805,X7808)
			JCudaTensor x3650;
			JCudaTensor x3651, x3652, x3653, x3654;
			x3651 = x3642;
			x3652 = x3644;
			x3653 = x3646;
			x3654 = x3648;
			x3650 = x250.forward(x3651,x3652,x3653,x3654);

			// Dealloc(X7808)
			JCudaTensor x3655;
			x3655 = x3648;
			x3655.free();

			// Dealloc(X7805)
			JCudaTensor x3656;
			x3656 = x3646;
			x3656.free();

			// Dealloc(X7801)
			JCudaTensor x3657;
			x3657 = x3644;
			x3657.free();

			// Dealloc(X7797)
			JCudaTensor x3658;
			x3658 = x3642;
			x3658.free();

			// val X7820 = Pooling(3,1,1,true)(X7809)
			JCudaTensor x3659;
			JCudaTensor x3660;
			x3660 = x3650;
			x3659 = x203.forward(x3660);

			// val X7816 = Convolv(1,0)(X7809,cv74_W,cv74_B)
			JCudaTensor x3661;
			JCudaTensor x3662, x3663, x3664;
			x3662 = x3650;
			x3663 = x544;
			x3664 = x545;
			x3661 = x193.forward(x3662, x3663, x3664);

			// val X7810 = Convolv(1,0)(X7809,cv71_W,cv71_B)
			JCudaTensor x3665;
			JCudaTensor x3666, x3667, x3668;
			x3666 = x3650;
			x3667 = x566;
			x3668 = x567;
			x3665 = x200.forward(x3666, x3667, x3668);

			// val X7812 = Convolv(1,0)(X7809,cv72_W,cv72_B)
			JCudaTensor x3669;
			JCudaTensor x3670, x3671, x3672;
			x3670 = x3650;
			x3671 = x560;
			x3672 = x561;
			x3669 = x210.forward(x3670, x3671, x3672);

			// Dealloc(X7809)
			JCudaTensor x3673;
			x3673 = x3650;
			x3673.free();

			// val X7821 = Convolv(1,0)(X7820,cv76_W,cv76_B)
			JCudaTensor x3674;
			JCudaTensor x3675, x3676, x3677;
			x3675 = x3659;
			x3676 = x588;
			x3677 = x589;
			x3674 = x220.forward(x3675, x3676, x3677);

			// Dealloc(X7820)
			JCudaTensor x3678;
			x3678 = x3659;
			x3678.free();

			// val X7813 = ReLU()(X7812)
			JCudaTensor x3679;
			JCudaTensor x3680;
			x3680 = x3669;
			x3679 = x213.forward(x3680);

			// val X7817 = ReLU()(X7816)
			JCudaTensor x3681;
			JCudaTensor x3682;
			x3682 = x3661;
			x3681 = x223.forward(x3682);

			// val X7814 = Convolv(1,1)(X7813,cv73_W,cv73_B)
			JCudaTensor x3683;
			JCudaTensor x3684, x3685, x3686;
			x3684 = x3679;
			x3685 = x621;
			x3686 = x622;
			x3683 = x237.forward(x3684, x3685, x3686);

			// Dealloc(X7813)
			JCudaTensor x3687;
			x3687 = x3679;
			x3687.free();

			// val X7818 = Convolv(1,2)(X7817,cv75_W,cv75_B)
			JCudaTensor x3688;
			JCudaTensor x3689, x3690, x3691;
			x3689 = x3681;
			x3690 = x634;
			x3691 = x635;
			x3688 = x230.forward(x3689, x3690, x3691);

			// Dealloc(X7817)
			JCudaTensor x3692;
			x3692 = x3681;
			x3692.free();

			// val X7811 = ReLU()(X7810)
			JCudaTensor x3693;
			JCudaTensor x3694;
			x3694 = x3665;
			x3693 = x240.forward(x3694);

			// val X7815 = ReLU()(X7814)
			JCudaTensor x3695;
			JCudaTensor x3696;
			x3696 = x3683;
			x3695 = x243.forward(x3696);

			// val X7819 = ReLU()(X7818)
			JCudaTensor x3697;
			JCudaTensor x3698;
			x3698 = x3688;
			x3697 = x246.forward(x3698);

			// val X7822 = ReLU()(X7821)
			JCudaTensor x3699;
			JCudaTensor x3700;
			x3700 = x3674;
			x3699 = x246.forward(x3700);

			// val X7823 = Concat(X7811,X7815,X7819,X7822)
			JCudaTensor x3701;
			JCudaTensor x3702, x3703, x3704, x3705;
			x3702 = x3693;
			x3703 = x3695;
			x3704 = x3697;
			x3705 = x3699;
			x3701 = x250.forward(x3702,x3703,x3704,x3705);

			// Dealloc(X7822)
			JCudaTensor x3706;
			x3706 = x3699;
			x3706.free();

			// Dealloc(X7819)
			JCudaTensor x3707;
			x3707 = x3697;
			x3707.free();

			// Dealloc(X7815)
			JCudaTensor x3708;
			x3708 = x3695;
			x3708.free();

			// Dealloc(X7811)
			JCudaTensor x3709;
			x3709 = x3693;
			x3709.free();

			// val X7824 = Pooling(3,2,1,true)(X7823)
			JCudaTensor x3710;
			JCudaTensor x3711;
			x3711 = x3701;
			x3710 = x712.forward(x3711);

			// Dealloc(X7823)
			JCudaTensor x3712;
			x3712 = x3701;
			x3712.free();

			// val X7835 = Pooling(3,1,1,true)(X7824)
			JCudaTensor x3713;
			JCudaTensor x3714;
			x3714 = x3710;
			x3713 = x717.forward(x3714);

			// val X7831 = Convolv(1,0)(X7824,cv84_W,cv84_B)
			JCudaTensor x3715;
			JCudaTensor x3716, x3717, x3718;
			x3716 = x3710;
			x3717 = x722;
			x3718 = x723;
			x3715 = x724.forward(x3716, x3717, x3718);

			// val X7827 = Convolv(1,0)(X7824,cv82_W,cv82_B)
			JCudaTensor x3719;
			JCudaTensor x3720, x3721, x3722;
			x3720 = x3710;
			x3721 = x731;
			x3722 = x732;
			x3719 = x733.forward(x3720, x3721, x3722);

			// val X7825 = Convolv(1,0)(X7824,cv81_W,cv81_B)
			JCudaTensor x3723;
			JCudaTensor x3724, x3725, x3726;
			x3724 = x3710;
			x3725 = x738;
			x3726 = x739;
			x3723 = x740.forward(x3724, x3725, x3726);

			// Dealloc(X7824)
			JCudaTensor x3727;
			x3727 = x3710;
			x3727.free();

			// val X7828 = ReLU()(X7827)
			JCudaTensor x3728;
			JCudaTensor x3729;
			x3729 = x3719;
			x3728 = x759.forward(x3729);

			// val X7832 = ReLU()(X7831)
			JCudaTensor x3730;
			JCudaTensor x3731;
			x3731 = x3715;
			x3730 = x749.forward(x3731);

			// val X7836 = Convolv(1,0)(X7835,cv86_W,cv86_B)
			JCudaTensor x3732;
			JCudaTensor x3733, x3734, x3735;
			x3733 = x3713;
			x3734 = x754;
			x3735 = x755;
			x3732 = x756.forward(x3733, x3734, x3735);

			// Dealloc(X7835)
			JCudaTensor x3736;
			x3736 = x3713;
			x3736.free();

			// val X7833 = Convolv(1,2)(X7832,cv85_W,cv85_B)
			JCudaTensor x3737;
			JCudaTensor x3738, x3739, x3740;
			x3738 = x3730;
			x3739 = x764;
			x3740 = x765;
			x3737 = x766.forward(x3738, x3739, x3740);

			// Dealloc(X7832)
			JCudaTensor x3741;
			x3741 = x3730;
			x3741.free();

			// val X7829 = Convolv(1,1)(X7828,cv83_W,cv83_B)
			JCudaTensor x3742;
			JCudaTensor x3743, x3744, x3745;
			x3743 = x3728;
			x3744 = x775;
			x3745 = x776;
			x3742 = x777.forward(x3743, x3744, x3745);

			// Dealloc(X7828)
			JCudaTensor x3746;
			x3746 = x3728;
			x3746.free();

			// val X7826 = ReLU()(X7825)
			JCudaTensor x3747;
			JCudaTensor x3748;
			x3748 = x3723;
			x3747 = x783.forward(x3748);

			// val X7830 = ReLU()(X7829)
			JCudaTensor x3749;
			JCudaTensor x3750;
			x3750 = x3742;
			x3749 = x789.forward(x3750);

			// val X7834 = ReLU()(X7833)
			JCudaTensor x3751;
			JCudaTensor x3752;
			x3752 = x3737;
			x3751 = x780.forward(x3752);

			// val X7837 = ReLU()(X7836)
			JCudaTensor x3753;
			JCudaTensor x3754;
			x3754 = x3732;
			x3753 = x780.forward(x3754);

			// val X7838 = Concat(X7826,X7830,X7834,X7837)
			JCudaTensor x3755;
			JCudaTensor x3756, x3757, x3758, x3759;
			x3756 = x3747;
			x3757 = x3749;
			x3758 = x3751;
			x3759 = x3753;
			x3755 = x793.forward(x3756,x3757,x3758,x3759);

			// Dealloc(X7837)
			JCudaTensor x3760;
			x3760 = x3753;
			x3760.free();

			// Dealloc(X7834)
			JCudaTensor x3761;
			x3761 = x3751;
			x3761.free();

			// Dealloc(X7830)
			JCudaTensor x3762;
			x3762 = x3749;
			x3762.free();

			// Dealloc(X7826)
			JCudaTensor x3763;
			x3763 = x3747;
			x3763.free();

			// val X7845 = Convolv(1,0)(X7838,cv94_W,cv94_B)
			JCudaTensor x3764;
			JCudaTensor x3765, x3766, x3767;
			x3765 = x3755;
			x3766 = x825;
			x3767 = x826;
			x3764 = x724.forward(x3765, x3766, x3767);

			// val X7849 = Pooling(3,1,1,true)(X7838)
			JCudaTensor x3768;
			JCudaTensor x3769;
			x3769 = x3755;
			x3768 = x717.forward(x3769);

			// val X7841 = Convolv(1,0)(X7838,cv92_W,cv92_B)
			JCudaTensor x3770;
			JCudaTensor x3771, x3772, x3773;
			x3771 = x3755;
			x3772 = x813;
			x3773 = x814;
			x3770 = x733.forward(x3771, x3772, x3773);

			// val X7839 = Convolv(1,0)(X7838,cv91_W,cv91_B)
			JCudaTensor x3774;
			JCudaTensor x3775, x3776, x3777;
			x3775 = x3755;
			x3776 = x807;
			x3777 = x808;
			x3774 = x740.forward(x3775, x3776, x3777);

			// Dealloc(X7838)
			JCudaTensor x3778;
			x3778 = x3755;
			x3778.free();

			// val X7850 = Convolv(1,0)(X7849,cv96_W,cv96_B)
			JCudaTensor x3779;
			JCudaTensor x3780, x3781, x3782;
			x3780 = x3768;
			x3781 = x835;
			x3782 = x836;
			x3779 = x756.forward(x3780, x3781, x3782);

			// Dealloc(X7849)
			JCudaTensor x3783;
			x3783 = x3768;
			x3783.free();

			// val X7842 = ReLU()(X7841)
			JCudaTensor x3784;
			JCudaTensor x3785;
			x3785 = x3770;
			x3784 = x759.forward(x3785);

			// val X7846 = ReLU()(X7845)
			JCudaTensor x3786;
			JCudaTensor x3787;
			x3787 = x3764;
			x3786 = x749.forward(x3787);

			// val X7847 = Convolv(1,2)(X7846,cv95_W,cv95_B)
			JCudaTensor x3788;
			JCudaTensor x3789, x3790, x3791;
			x3789 = x3786;
			x3790 = x858;
			x3791 = x859;
			x3788 = x766.forward(x3789, x3790, x3791);

			// Dealloc(X7846)
			JCudaTensor x3792;
			x3792 = x3786;
			x3792.free();

			// val X7843 = Convolv(1,1)(X7842,cv93_W,cv93_B)
			JCudaTensor x3793;
			JCudaTensor x3794, x3795, x3796;
			x3794 = x3784;
			x3795 = x864;
			x3796 = x865;
			x3793 = x777.forward(x3794, x3795, x3796);

			// Dealloc(X7842)
			JCudaTensor x3797;
			x3797 = x3784;
			x3797.free();

			// val X7840 = ReLU()(X7839)
			JCudaTensor x3798;
			JCudaTensor x3799;
			x3799 = x3774;
			x3798 = x783.forward(x3799);

			// val X7844 = ReLU()(X7843)
			JCudaTensor x3800;
			JCudaTensor x3801;
			x3801 = x3793;
			x3800 = x789.forward(x3801);

			// val X7848 = ReLU()(X7847)
			JCudaTensor x3802;
			JCudaTensor x3803;
			x3803 = x3788;
			x3802 = x780.forward(x3803);

			// val X7851 = ReLU()(X7850)
			JCudaTensor x3804;
			JCudaTensor x3805;
			x3805 = x3779;
			x3804 = x780.forward(x3805);

			// val X7852 = Concat(X7840,X7844,X7848,X7851)
			JCudaTensor x3806;
			JCudaTensor x3807, x3808, x3809, x3810;
			x3807 = x3798;
			x3808 = x3800;
			x3809 = x3802;
			x3810 = x3804;
			x3806 = x793.forward(x3807,x3808,x3809,x3810);

			// Dealloc(X7851)
			JCudaTensor x3811;
			x3811 = x3804;
			x3811.free();

			// Dealloc(X7848)
			JCudaTensor x3812;
			x3812 = x3802;
			x3812.free();

			// Dealloc(X7844)
			JCudaTensor x3813;
			x3813 = x3800;
			x3813.free();

			// Dealloc(X7840)
			JCudaTensor x3814;
			x3814 = x3798;
			x3814.free();

			// val X7853 = Pooling(7,1,0,false)(X7852)
			JCudaTensor x3815;
			JCudaTensor x3816;
			x3816 = x3806;
			x3815 = x950.forward(x3816);

			// Dealloc(X7852)
			JCudaTensor x3817;
			x3817 = x3806;
			x3817.free();

			// val X7854 = Dropout(0.4)(X7853)
			JCudaTensor x3818;
			JCudaTensor x3819;
			x3819 = x3815;
			x3818 = x966.forward(x3819);

			// Dealloc(X7853)
			JCudaTensor x3820;
			x3820 = x3815;
			x3820.free();

			// val X7855 = (X7854[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x3821;
			JCudaMatrix x3822;
			JCudaMatrix x3823;
			JCudaTensor x3824;
			JCudaTensor x3825;
			x3825 = x3818;
			x3824 = x3825.flatten(1, new int[]{256, 1, 1});
			x3822 = x3824.asMatrix(1, true);
			JCudaTensor x3826;
			x3826 = x1017;
			x3823 = x3826.asMatrix(1, true);
			x3821 = x3822.times(x3823);

			// Dealloc(X7854)
			JCudaTensor x3827;
			x3827 = x3818;
			x3827.free();

			// val X7857 = (X7855 + (i) => fc_B)
			JCudaTensor x3828;
			JCudaTensor x3829, x3830;
			x3829 = x3821;
			x3830 = x1035;
			x3828 = x3830.copy(128, x3829);

			// Precision(Accuracy(X7857, Y, 1))
			float x3832;
			JCudaTensor x3833;
			JTensorFloat x3834;
			x3833 = x3828;
			x3834 = x4;
			x3832 = x3833.accuracy(x3834, 1);
			System.out.println(x5 + " test precision "  + x3832);
			x3831 += x3832;

			// Dealloc(X7857)
			JCudaTensor x3835;
			x3835 = x3828;
			x3835.free();

		}
		System.out.println();
		System.out.println("average precision: " + x3831/10);
		System.out.println(); 
	}

}