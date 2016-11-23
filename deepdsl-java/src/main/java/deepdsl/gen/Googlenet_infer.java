package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;

// This file is for inference only, which needs trained parameters.
public class Googlenet_infer {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/googlenet";
	// test_data_path
	static String test_data_path = "dataset/imagenet/ilsvrc12_val_lmdb";
	// test_itr
	static int test_itr = 10;
	// test_size
	static int test_size = 10000;

	// (Convolv(1,0),List(List(128, 192, 28, 28), List(16, 192, 1, 1), List(16)))
	static JCudnnConvolution x79 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{16,192,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(32, 192, 1, 1), List(32)))
	static JCudnnConvolution x93 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{32,192,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(64, 192, 1, 1), List(64)))
	static JCudnnConvolution x72 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{64,192,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(96, 192, 1, 1), List(96)))
	static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{96,192,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x220 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x235 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x227 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x210 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x138 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x167 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x148 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x155 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x549 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x573 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x556 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x542 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 64, 56, 56), List(64, 64, 1, 1), List(64)))
	static JCudnnConvolution x32 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{64,64,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,1),List(List(128, 64, 56, 56), List(192, 64, 3, 3), List(192)))
	static JCudnnConvolution x43 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{192,64,3,3},new int[]{192}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 14, 14), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x249 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 28, 28), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x109 = new JCudnnConvolution(new int[]{128,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 7, 7), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x581 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,2),List(List(128, 16, 14, 14), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x257 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(1,2),List(List(128, 16, 28, 28), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x101 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(1,2),List(List(128, 16, 7, 7), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x589 = new JCudnnConvolution(new int[]{128,16,7,7},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(2,3),List(List(128, 3, 224, 224), List(64, 3, 7, 7), List(64)))
	static JCudnnConvolution x13 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{64,3,7,7},new int[]{64}, 2, 3);
	// (Dropout(0.4),List(List(128, 256, 1, 1)))
	static JCudnnDropout x681 = new JCudnnDropout(new int[]{128,256,1,1}, 0.4f);
	// (LRN(5,1.0E-4,0.75),List(List(128, 192, 56, 56)))
	static JCudnnLRN x50 = new JCudnnLRN(new int[]{128,192,56,56}, 5, 1.0E-4, 0.75);
	// (LRN(5,1.0E-4,0.75),List(List(128, 64, 56, 56)))
	static JCudnnLRN x24 = new JCudnnLRN(new int[]{128,64,56,56}, 5, 1.0E-4, 0.75);
	// (Lmdb(1000000,10000,1000),false)
	static LmdbFactory x1 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, 1000, true);
	// (Pooling(3,1,1,true),List(List(128, 192, 28, 28)))
	static JCudnnPooling x65 = new JCudnnPooling(new int[]{128,192,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 14, 14)))
	static JCudnnPooling x213 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 28, 28)))
	static JCudnnPooling x141 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 7, 7)))
	static JCudnnPooling x559 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 192, 56, 56)))
	static JCudnnPooling x54 = new JCudnnPooling(new int[]{128,192,56,56}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 256, 14, 14)))
	static JCudnnPooling x534 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 256, 28, 28)))
	static JCudnnPooling x202 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(3,2,1,true),List(List(128, 64, 112, 112)))
	static JCudnnPooling x20 = new JCudnnPooling(new int[]{128,64,112,112}, 3, 2, 1, PoolingType.MAX);
	// (Pooling(7,1,0,false),List(List(128, 256, 7, 7)))
	static JCudnnPooling x677 = new JCudnnPooling(new int[]{128,256,7,7}, 7, 1, 0, PoolingType.AVERAGE_EXCLUDE_PADDING);
	// (ReLU(),List(List(128, 128, 14, 14)))
	static JCudnnActivation x264 = new JCudnnActivation(new int[]{128,128,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 28, 28)))
	static JCudnnActivation x116 = new JCudnnActivation(new int[]{128,128,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 128, 7, 7)))
	static JCudnnActivation x596 = new JCudnnActivation(new int[]{128,128,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 14, 14)))
	static JCudnnActivation x242 = new JCudnnActivation(new int[]{128,16,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 28, 28)))
	static JCudnnActivation x86 = new JCudnnActivation(new int[]{128,16,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 7, 7)))
	static JCudnnActivation x563 = new JCudnnActivation(new int[]{128,16,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 192, 56, 56)))
	static JCudnnActivation x47 = new JCudnnActivation(new int[]{128,192,56,56}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 14, 14)))
	static JCudnnActivation x267 = new JCudnnActivation(new int[]{128,32,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 28, 28)))
	static JCudnnActivation x119 = new JCudnnActivation(new int[]{128,32,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 32, 7, 7)))
	static JCudnnActivation x599 = new JCudnnActivation(new int[]{128,32,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 112, 112)))
	static JCudnnActivation x17 = new JCudnnActivation(new int[]{128,64,112,112}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 14, 14)))
	static JCudnnActivation x261 = new JCudnnActivation(new int[]{128,64,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 28, 28)))
	static JCudnnActivation x113 = new JCudnnActivation(new int[]{128,64,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 56, 56)))
	static JCudnnActivation x36 = new JCudnnActivation(new int[]{128,64,56,56}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 64, 7, 7)))
	static JCudnnActivation x593 = new JCudnnActivation(new int[]{128,64,7,7}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 14, 14)))
	static JCudnnActivation x239 = new JCudnnActivation(new int[]{128,96,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 28, 28)))
	static JCudnnActivation x83 = new JCudnnActivation(new int[]{128,96,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 7, 7)))
	static JCudnnActivation x566 = new JCudnnActivation(new int[]{128,96,7,7}, ActivationMode.RELU);
	// List(List(128, 64, 14, 14), List(128, 128, 14, 14), List(128, 32, 14, 14), List(128, 32, 14, 14))
	static JCudnnConcat x271 = new JCudnnConcat(new int[]{128,64,14,14},new int[]{128,128,14,14},new int[]{128,32,14,14},new int[]{128,32,14,14});
	// List(List(128, 64, 28, 28), List(128, 128, 28, 28), List(128, 32, 28, 28), List(128, 32, 28, 28))
	static JCudnnConcat x123 = new JCudnnConcat(new int[]{128,64,28,28},new int[]{128,128,28,28},new int[]{128,32,28,28},new int[]{128,32,28,28});
	// List(List(128, 64, 7, 7), List(128, 128, 7, 7), List(128, 32, 7, 7), List(128, 32, 7, 7))
	static JCudnnConcat x603 = new JCudnnConcat(new int[]{128,64,7,7},new int[]{128,128,7,7},new int[]{128,32,7,7},new int[]{128,32,7,7});
	// X
	static JTensorFloat x2;
	// cv11_B
	static JCudaTensor x71 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
	// cv11_W
	static JCudaTensor x70 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 64, 192, 1, 1).load(network_dir + "/cv11_W").asJCudaTensor();
	// cv12_B
	static JCudaTensor x61 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv12_B").asJCudaTensor();
	// cv12_W
	static JCudaTensor x60 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 96, 192, 1, 1).load(network_dir + "/cv12_W").asJCudaTensor();
	// cv13_B
	static JCudaTensor x108 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv13_B").asJCudaTensor();
	// cv13_W
	static JCudaTensor x107 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv13_W").asJCudaTensor();
	// cv14_B
	static JCudaTensor x78 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv14_B").asJCudaTensor();
	// cv14_W
	static JCudaTensor x77 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 16, 192, 1, 1).load(network_dir + "/cv14_W").asJCudaTensor();
	// cv15_B
	static JCudaTensor x100 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv15_B").asJCudaTensor();
	// cv15_W
	static JCudaTensor x99 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv15_W").asJCudaTensor();
	// cv16_B
	static JCudaTensor x92 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv16_B").asJCudaTensor();
	// cv16_W
	static JCudaTensor x91 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 32, 192, 1, 1).load(network_dir + "/cv16_W").asJCudaTensor();
	// cv1_B
	static JCudaTensor x12 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv1_B").asJCudaTensor();
	// cv1_W
	static JCudaTensor x11 = JTensor.randomFloat(-0.11664237f, 0.11664237f, 64, 3, 7, 7).load(network_dir + "/cv1_W").asJCudaTensor();
	// cv21_B
	static JCudaTensor x147 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv21_B").asJCudaTensor();
	// cv21_W
	static JCudaTensor x146 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv21_W").asJCudaTensor();
	// cv22_B
	static JCudaTensor x154 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv22_B").asJCudaTensor();
	// cv22_W
	static JCudaTensor x153 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv22_W").asJCudaTensor();
	// cv23_B
	static JCudaTensor x174 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv23_B").asJCudaTensor();
	// cv23_W
	static JCudaTensor x173 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv23_W").asJCudaTensor();
	// cv24_B
	static JCudaTensor x137 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv24_B").asJCudaTensor();
	// cv24_W
	static JCudaTensor x136 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv24_W").asJCudaTensor();
	// cv25_B
	static JCudaTensor x181 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv25_B").asJCudaTensor();
	// cv25_W
	static JCudaTensor x180 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv25_W").asJCudaTensor();
	// cv26_B
	static JCudaTensor x166 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv26_B").asJCudaTensor();
	// cv26_W
	static JCudaTensor x165 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv26_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x31 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x30 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/cv2_W").asJCudaTensor();
	// cv31_B
	static JCudaTensor x226 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv31_B").asJCudaTensor();
	// cv31_W
	static JCudaTensor x225 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv31_W").asJCudaTensor();
	// cv32_B
	static JCudaTensor x209 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv32_B").asJCudaTensor();
	// cv32_W
	static JCudaTensor x208 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv32_W").asJCudaTensor();
	// cv33_B
	static JCudaTensor x248 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv33_B").asJCudaTensor();
	// cv33_W
	static JCudaTensor x247 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
	// cv34_B
	static JCudaTensor x219 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv34_B").asJCudaTensor();
	// cv34_W
	static JCudaTensor x218 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv34_W").asJCudaTensor();
	// cv35_B
	static JCudaTensor x256 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv35_B").asJCudaTensor();
	// cv35_W
	static JCudaTensor x255 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv35_W").asJCudaTensor();
	// cv36_B
	static JCudaTensor x234 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv36_B").asJCudaTensor();
	// cv36_W
	static JCudaTensor x233 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv36_W").asJCudaTensor();
	// cv3_B
	static JCudaTensor x42 = JTensor.constFloat(0.2f, 192).load(network_dir + "/cv3_B").asJCudaTensor();
	// cv3_W
	static JCudaTensor x41 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 192, 64, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
	// cv41_B
	static JCudaTensor x285 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv41_B").asJCudaTensor();
	// cv41_W
	static JCudaTensor x284 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv41_W").asJCudaTensor();
	// cv42_B
	static JCudaTensor x297 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv42_B").asJCudaTensor();
	// cv42_W
	static JCudaTensor x296 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv42_W").asJCudaTensor();
	// cv43_B
	static JCudaTensor x317 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv43_B").asJCudaTensor();
	// cv43_W
	static JCudaTensor x316 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
	// cv44_B
	static JCudaTensor x291 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv44_B").asJCudaTensor();
	// cv44_W
	static JCudaTensor x290 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv44_W").asJCudaTensor();
	// cv45_B
	static JCudaTensor x324 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv45_B").asJCudaTensor();
	// cv45_W
	static JCudaTensor x323 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv45_W").asJCudaTensor();
	// cv46_B
	static JCudaTensor x310 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv46_B").asJCudaTensor();
	// cv46_W
	static JCudaTensor x309 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv46_W").asJCudaTensor();
	// cv51_B
	static JCudaTensor x356 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv51_B").asJCudaTensor();
	// cv51_W
	static JCudaTensor x355 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv51_W").asJCudaTensor();
	// cv52_B
	static JCudaTensor x362 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv52_B").asJCudaTensor();
	// cv52_W
	static JCudaTensor x361 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv52_W").asJCudaTensor();
	// cv53_B
	static JCudaTensor x380 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv53_B").asJCudaTensor();
	// cv53_W
	static JCudaTensor x379 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
	// cv54_B
	static JCudaTensor x348 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv54_B").asJCudaTensor();
	// cv54_W
	static JCudaTensor x347 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv54_W").asJCudaTensor();
	// cv55_B
	static JCudaTensor x387 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv55_B").asJCudaTensor();
	// cv55_W
	static JCudaTensor x386 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv55_W").asJCudaTensor();
	// cv56_B
	static JCudaTensor x369 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv56_B").asJCudaTensor();
	// cv56_W
	static JCudaTensor x368 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv56_W").asJCudaTensor();
	// cv61_B
	static JCudaTensor x423 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv61_B").asJCudaTensor();
	// cv61_W
	static JCudaTensor x422 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv61_W").asJCudaTensor();
	// cv62_B
	static JCudaTensor x417 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv62_B").asJCudaTensor();
	// cv62_W
	static JCudaTensor x416 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv62_W").asJCudaTensor();
	// cv63_B
	static JCudaTensor x450 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv63_B").asJCudaTensor();
	// cv63_W
	static JCudaTensor x449 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv63_W").asJCudaTensor();
	// cv64_B
	static JCudaTensor x411 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv64_B").asJCudaTensor();
	// cv64_W
	static JCudaTensor x410 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv64_W").asJCudaTensor();
	// cv65_B
	static JCudaTensor x443 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv65_B").asJCudaTensor();
	// cv65_W
	static JCudaTensor x442 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv65_W").asJCudaTensor();
	// cv66_B
	static JCudaTensor x434 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv66_B").asJCudaTensor();
	// cv66_W
	static JCudaTensor x433 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv66_W").asJCudaTensor();
	// cv71_B
	static JCudaTensor x488 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv71_B").asJCudaTensor();
	// cv71_W
	static JCudaTensor x487 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv71_W").asJCudaTensor();
	// cv72_B
	static JCudaTensor x482 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv72_B").asJCudaTensor();
	// cv72_W
	static JCudaTensor x481 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv72_W").asJCudaTensor();
	// cv73_B
	static JCudaTensor x506 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv73_B").asJCudaTensor();
	// cv73_W
	static JCudaTensor x505 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv73_W").asJCudaTensor();
	// cv74_B
	static JCudaTensor x474 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv74_B").asJCudaTensor();
	// cv74_W
	static JCudaTensor x473 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv74_W").asJCudaTensor();
	// cv75_B
	static JCudaTensor x513 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv75_B").asJCudaTensor();
	// cv75_W
	static JCudaTensor x512 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv75_W").asJCudaTensor();
	// cv76_B
	static JCudaTensor x499 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv76_B").asJCudaTensor();
	// cv76_W
	static JCudaTensor x498 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv76_W").asJCudaTensor();
	// cv81_B
	static JCudaTensor x555 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv81_B").asJCudaTensor();
	// cv81_W
	static JCudaTensor x554 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv81_W").asJCudaTensor();
	// cv82_B
	static JCudaTensor x541 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv82_B").asJCudaTensor();
	// cv82_W
	static JCudaTensor x540 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv82_W").asJCudaTensor();
	// cv83_B
	static JCudaTensor x580 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv83_B").asJCudaTensor();
	// cv83_W
	static JCudaTensor x579 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv83_W").asJCudaTensor();
	// cv84_B
	static JCudaTensor x548 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv84_B").asJCudaTensor();
	// cv84_W
	static JCudaTensor x547 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv84_W").asJCudaTensor();
	// cv85_B
	static JCudaTensor x588 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv85_B").asJCudaTensor();
	// cv85_W
	static JCudaTensor x587 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv85_W").asJCudaTensor();
	// cv86_B
	static JCudaTensor x572 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv86_B").asJCudaTensor();
	// cv86_W
	static JCudaTensor x571 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv86_W").asJCudaTensor();
	// cv91_B
	static JCudaTensor x617 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv91_B").asJCudaTensor();
	// cv91_W
	static JCudaTensor x616 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv91_W").asJCudaTensor();
	// cv92_B
	static JCudaTensor x631 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv92_B").asJCudaTensor();
	// cv92_W
	static JCudaTensor x630 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv92_W").asJCudaTensor();
	// cv93_B
	static JCudaTensor x649 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv93_B").asJCudaTensor();
	// cv93_W
	static JCudaTensor x648 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv93_W").asJCudaTensor();
	// cv94_B
	static JCudaTensor x625 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv94_B").asJCudaTensor();
	// cv94_W
	static JCudaTensor x624 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv94_W").asJCudaTensor();
	// cv95_B
	static JCudaTensor x656 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv95_B").asJCudaTensor();
	// cv95_W
	static JCudaTensor x655 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv95_W").asJCudaTensor();
	// cv96_B
	static JCudaTensor x638 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv96_B").asJCudaTensor();
	// cv96_W
	static JCudaTensor x637 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv96_W").asJCudaTensor();
	// fc_B
	static JCudaTensor x694 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
	// fc_W
	static JCudaTensor x689 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1000, 256).load(network_dir + "/fc_W").asJCudaTensor();

	public static void main(String[] args){
		test();
		x61.free();
		x317.free();
		x100.free();
		x316.free();
		x136.free();
		x572.free();
		x356.free();
		x324.free();
		x474.free();
		x219.free();
		x180.free();
		x379.free();
		x154.free();
		x481.free();
		x369.free();
		x247.free();
		x488.free();
		x648.free();
		x587.free();
		x422.free();
		x285.free();
		x147.free();
		x689.free();
		x386.free();
		x482.free();
		x368.free();
		x554.free();
		x181.free();
		x296.free();
		x473.free();
		x255.free();
		x226.free();
		x107.free();
		x624.free();
		x487.free();
		x297.free();
		x71.free();
		x355.free();
		x284.free();
		x656.free();
		x555.free();
		x70.free();
		x137.free();
		x77.free();
		x513.free();
		x505.free();
		x248.free();
		x442.free();
		x380.free();
		x310.free();
		x99.free();
		x165.free();
		x512.free();
		x256.free();
		x637.free();
		x12.free();
		x290.free();
		x410.free();
		x166.free();
		x60.free();
		x580.free();
		x78.free();
		x173.free();
		x631.free();
		x153.free();
		x625.free();
		x92.free();
		x225.free();
		x108.free();
		x91.free();
		x41.free();
		x362.free();
		x309.free();
		x416.free();
		x443.free();
		x547.free();
		x630.free();
		x506.free();
		x616.free();
		x498.free();
		x291.free();
		x499.free();
		x433.free();
		x233.free();
		x617.free();
		x655.free();
		x30.free();
		x146.free();
		x540.free();
		x579.free();
		x638.free();
		x323.free();
		x571.free();
		x449.free();
		x541.free();
		x348.free();
		x209.free();
		x417.free();
		x411.free();
		x450.free();
		x434.free();
		x208.free();
		x649.free();
		x347.free();
		x42.free();
		x694.free();
		x588.free();
		x11.free();
		x174.free();
		x31.free();
		x218.free();
		x423.free();
		x548.free();
		x234.free();
		x387.free();
		x361.free();
		x167.free();
		x119.free();
		x116.free();
		x24.free();
		x573.free();
		x242.free();
		x534.free();
		x235.free();
		x227.free();
		x599.free();
		x138.free();
		x239.free();
		x549.free();
		x213.free();
		x563.free();
		x148.free();
		x36.free();
		x50.free();
		x596.free();
		x267.free();
		x542.free();
		x681.free();
		x141.free();
		x54.free();
		x65.free();
		x261.free();
		x257.free();
		x86.free();
		x566.free();
		x83.free();
		x581.free();
		x113.free();
		x32.free();
		x79.free();
		x20.free();
		x677.free();
		x93.free();
		x17.free();
		x13.free();
		x264.free();
		x62.free();
		x155.free();
		x72.free();
		x202.free();
		x109.free();
		x220.free();
		x589.free();
		x559.free();
		x593.free();
		x101.free();
		x43.free();
		x556.free();
		x210.free();
		x47.free();
		x249.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void test() {
		for(int x3=0; x3<test_itr; x3++) {
			JTensorFloatTuple x4 =  x1.nextFloat();
			x2 = x4.image;

			// val X7858 = Cuda(X)
			JCudaTensor x5;
			JTensorFloat x6;
			x6 = x2;
			x5 = x6.asJCudaTensor();

			// val X7859 = Convolv(2,3)(X7858,cv1_W,cv1_B)
			JCudaTensor x7;
			JCudaTensor x8, x9, x10;
			x8 = x5;
			x9 = x11;
			x10 = x12;
			x7 = x13.forward(x8, x9, x10);

			// Dealloc(X7858)
			JCudaTensor x14;
			x14 = x5;
			x14.free();

			// val X7860 = ReLU()(X7859)
			JCudaTensor x15;
			JCudaTensor x16;
			x16 = x7;
			x15 = x17.forward(x16);

			// val X7861 = Pooling(3,2,1,true)(X7860)
			JCudaTensor x18;
			JCudaTensor x19;
			x19 = x15;
			x18 = x20.forward(x19);

			// Dealloc(X7860)
			JCudaTensor x21;
			x21 = x15;
			x21.free();

			// val X7862 = LRN(5,1.0E-4,0.75)(X7861)
			JCudaTensor x22;
			JCudaTensor x23;
			x23 = x18;
			x22 = x24.forward(x23);

			// Dealloc(X7861)
			JCudaTensor x25;
			x25 = x18;
			x25.free();

			// val X7863 = Convolv(1,0)(X7862,cv2_W,cv2_B)
			JCudaTensor x26;
			JCudaTensor x27, x28, x29;
			x27 = x22;
			x28 = x30;
			x29 = x31;
			x26 = x32.forward(x27, x28, x29);

			// Dealloc(X7862)
			JCudaTensor x33;
			x33 = x22;
			x33.free();

			// val X7864 = ReLU()(X7863)
			JCudaTensor x34;
			JCudaTensor x35;
			x35 = x26;
			x34 = x36.forward(x35);

			// val X7865 = Convolv(1,1)(X7864,cv3_W,cv3_B)
			JCudaTensor x37;
			JCudaTensor x38, x39, x40;
			x38 = x34;
			x39 = x41;
			x40 = x42;
			x37 = x43.forward(x38, x39, x40);

			// Dealloc(X7864)
			JCudaTensor x44;
			x44 = x34;
			x44.free();

			// val X7866 = ReLU()(X7865)
			JCudaTensor x45;
			JCudaTensor x46;
			x46 = x37;
			x45 = x47.forward(x46);

			// val X7867 = LRN(5,1.0E-4,0.75)(X7866)
			JCudaTensor x48;
			JCudaTensor x49;
			x49 = x45;
			x48 = x50.forward(x49);

			// Dealloc(X7866)
			JCudaTensor x51;
			x51 = x45;
			x51.free();

			// val X7868 = Pooling(3,2,1,true)(X7867)
			JCudaTensor x52;
			JCudaTensor x53;
			x53 = x48;
			x52 = x54.forward(x53);

			// Dealloc(X7867)
			JCudaTensor x55;
			x55 = x48;
			x55.free();

			// val X7871 = Convolv(1,0)(X7868,cv12_W,cv12_B)
			JCudaTensor x56;
			JCudaTensor x57, x58, x59;
			x57 = x52;
			x58 = x60;
			x59 = x61;
			x56 = x62.forward(x57, x58, x59);

			// val X7879 = Pooling(3,1,1,true)(X7868)
			JCudaTensor x63;
			JCudaTensor x64;
			x64 = x52;
			x63 = x65.forward(x64);

			// val X7869 = Convolv(1,0)(X7868,cv11_W,cv11_B)
			JCudaTensor x66;
			JCudaTensor x67, x68, x69;
			x67 = x52;
			x68 = x70;
			x69 = x71;
			x66 = x72.forward(x67, x68, x69);

			// val X7875 = Convolv(1,0)(X7868,cv14_W,cv14_B)
			JCudaTensor x73;
			JCudaTensor x74, x75, x76;
			x74 = x52;
			x75 = x77;
			x76 = x78;
			x73 = x79.forward(x74, x75, x76);

			// Dealloc(X7868)
			JCudaTensor x80;
			x80 = x52;
			x80.free();

			// val X7872 = ReLU()(X7871)
			JCudaTensor x81;
			JCudaTensor x82;
			x82 = x56;
			x81 = x83.forward(x82);

			// val X7876 = ReLU()(X7875)
			JCudaTensor x84;
			JCudaTensor x85;
			x85 = x73;
			x84 = x86.forward(x85);

			// val X7880 = Convolv(1,0)(X7879,cv16_W,cv16_B)
			JCudaTensor x87;
			JCudaTensor x88, x89, x90;
			x88 = x63;
			x89 = x91;
			x90 = x92;
			x87 = x93.forward(x88, x89, x90);

			// Dealloc(X7879)
			JCudaTensor x94;
			x94 = x63;
			x94.free();

			// val X7877 = Convolv(1,2)(X7876,cv15_W,cv15_B)
			JCudaTensor x95;
			JCudaTensor x96, x97, x98;
			x96 = x84;
			x97 = x99;
			x98 = x100;
			x95 = x101.forward(x96, x97, x98);

			// Dealloc(X7876)
			JCudaTensor x102;
			x102 = x84;
			x102.free();

			// val X7873 = Convolv(1,1)(X7872,cv13_W,cv13_B)
			JCudaTensor x103;
			JCudaTensor x104, x105, x106;
			x104 = x81;
			x105 = x107;
			x106 = x108;
			x103 = x109.forward(x104, x105, x106);

			// Dealloc(X7872)
			JCudaTensor x110;
			x110 = x81;
			x110.free();

			// val X7870 = ReLU()(X7869)
			JCudaTensor x111;
			JCudaTensor x112;
			x112 = x66;
			x111 = x113.forward(x112);

			// val X7874 = ReLU()(X7873)
			JCudaTensor x114;
			JCudaTensor x115;
			x115 = x103;
			x114 = x116.forward(x115);

			// val X7878 = ReLU()(X7877)
			JCudaTensor x117;
			JCudaTensor x118;
			x118 = x95;
			x117 = x119.forward(x118);

			// val X7881 = ReLU()(X7880)
			JCudaTensor x120;
			JCudaTensor x121;
			x121 = x87;
			x120 = x119.forward(x121);

			// val X7882 = Concat(X7870,X7874,X7878,X7881)
			JCudaTensor x122;
			JCudaTensor x124, x125, x126, x127;
			x124 = x111;
			x125 = x114;
			x126 = x117;
			x127 = x120;
			x122 = x123.forward(x124,x125,x126,x127);

			// Dealloc(X7881)
			JCudaTensor x128;
			x128 = x120;
			x128.free();

			// Dealloc(X7878)
			JCudaTensor x129;
			x129 = x117;
			x129.free();

			// Dealloc(X7874)
			JCudaTensor x130;
			x130 = x114;
			x130.free();

			// Dealloc(X7870)
			JCudaTensor x131;
			x131 = x111;
			x131.free();

			// val X7889 = Convolv(1,0)(X7882,cv24_W,cv24_B)
			JCudaTensor x132;
			JCudaTensor x133, x134, x135;
			x133 = x122;
			x134 = x136;
			x135 = x137;
			x132 = x138.forward(x133, x134, x135);

			// val X7893 = Pooling(3,1,1,true)(X7882)
			JCudaTensor x139;
			JCudaTensor x140;
			x140 = x122;
			x139 = x141.forward(x140);

			// val X7883 = Convolv(1,0)(X7882,cv21_W,cv21_B)
			JCudaTensor x142;
			JCudaTensor x143, x144, x145;
			x143 = x122;
			x144 = x146;
			x145 = x147;
			x142 = x148.forward(x143, x144, x145);

			// val X7885 = Convolv(1,0)(X7882,cv22_W,cv22_B)
			JCudaTensor x149;
			JCudaTensor x150, x151, x152;
			x150 = x122;
			x151 = x153;
			x152 = x154;
			x149 = x155.forward(x150, x151, x152);

			// Dealloc(X7882)
			JCudaTensor x156;
			x156 = x122;
			x156.free();

			// val X7890 = ReLU()(X7889)
			JCudaTensor x157;
			JCudaTensor x158;
			x158 = x132;
			x157 = x86.forward(x158);

			// val X7886 = ReLU()(X7885)
			JCudaTensor x159;
			JCudaTensor x160;
			x160 = x149;
			x159 = x83.forward(x160);

			// val X7894 = Convolv(1,0)(X7893,cv26_W,cv26_B)
			JCudaTensor x161;
			JCudaTensor x162, x163, x164;
			x162 = x139;
			x163 = x165;
			x164 = x166;
			x161 = x167.forward(x162, x163, x164);

			// Dealloc(X7893)
			JCudaTensor x168;
			x168 = x139;
			x168.free();

			// val X7887 = Convolv(1,1)(X7886,cv23_W,cv23_B)
			JCudaTensor x169;
			JCudaTensor x170, x171, x172;
			x170 = x159;
			x171 = x173;
			x172 = x174;
			x169 = x109.forward(x170, x171, x172);

			// Dealloc(X7886)
			JCudaTensor x175;
			x175 = x159;
			x175.free();

			// val X7891 = Convolv(1,2)(X7890,cv25_W,cv25_B)
			JCudaTensor x176;
			JCudaTensor x177, x178, x179;
			x177 = x157;
			x178 = x180;
			x179 = x181;
			x176 = x101.forward(x177, x178, x179);

			// Dealloc(X7890)
			JCudaTensor x182;
			x182 = x157;
			x182.free();

			// val X7884 = ReLU()(X7883)
			JCudaTensor x183;
			JCudaTensor x184;
			x184 = x142;
			x183 = x113.forward(x184);

			// val X7888 = ReLU()(X7887)
			JCudaTensor x185;
			JCudaTensor x186;
			x186 = x169;
			x185 = x116.forward(x186);

			// val X7892 = ReLU()(X7891)
			JCudaTensor x187;
			JCudaTensor x188;
			x188 = x176;
			x187 = x119.forward(x188);

			// val X7895 = ReLU()(X7894)
			JCudaTensor x189;
			JCudaTensor x190;
			x190 = x161;
			x189 = x119.forward(x190);

			// val X7896 = Concat(X7884,X7888,X7892,X7895)
			JCudaTensor x191;
			JCudaTensor x192, x193, x194, x195;
			x192 = x183;
			x193 = x185;
			x194 = x187;
			x195 = x189;
			x191 = x123.forward(x192,x193,x194,x195);

			// Dealloc(X7895)
			JCudaTensor x196;
			x196 = x189;
			x196.free();

			// Dealloc(X7892)
			JCudaTensor x197;
			x197 = x187;
			x197.free();

			// Dealloc(X7888)
			JCudaTensor x198;
			x198 = x185;
			x198.free();

			// Dealloc(X7884)
			JCudaTensor x199;
			x199 = x183;
			x199.free();

			// val X7897 = Pooling(3,2,1,true)(X7896)
			JCudaTensor x200;
			JCudaTensor x201;
			x201 = x191;
			x200 = x202.forward(x201);

			// Dealloc(X7896)
			JCudaTensor x203;
			x203 = x191;
			x203.free();

			// val X7900 = Convolv(1,0)(X7897,cv32_W,cv32_B)
			JCudaTensor x204;
			JCudaTensor x205, x206, x207;
			x205 = x200;
			x206 = x208;
			x207 = x209;
			x204 = x210.forward(x205, x206, x207);

			// val X7908 = Pooling(3,1,1,true)(X7897)
			JCudaTensor x211;
			JCudaTensor x212;
			x212 = x200;
			x211 = x213.forward(x212);

			// val X7904 = Convolv(1,0)(X7897,cv34_W,cv34_B)
			JCudaTensor x214;
			JCudaTensor x215, x216, x217;
			x215 = x200;
			x216 = x218;
			x217 = x219;
			x214 = x220.forward(x215, x216, x217);

			// val X7898 = Convolv(1,0)(X7897,cv31_W,cv31_B)
			JCudaTensor x221;
			JCudaTensor x222, x223, x224;
			x222 = x200;
			x223 = x225;
			x224 = x226;
			x221 = x227.forward(x222, x223, x224);

			// Dealloc(X7897)
			JCudaTensor x228;
			x228 = x200;
			x228.free();

			// val X7909 = Convolv(1,0)(X7908,cv36_W,cv36_B)
			JCudaTensor x229;
			JCudaTensor x230, x231, x232;
			x230 = x211;
			x231 = x233;
			x232 = x234;
			x229 = x235.forward(x230, x231, x232);

			// Dealloc(X7908)
			JCudaTensor x236;
			x236 = x211;
			x236.free();

			// val X7901 = ReLU()(X7900)
			JCudaTensor x237;
			JCudaTensor x238;
			x238 = x204;
			x237 = x239.forward(x238);

			// val X7905 = ReLU()(X7904)
			JCudaTensor x240;
			JCudaTensor x241;
			x241 = x214;
			x240 = x242.forward(x241);

			// val X7902 = Convolv(1,1)(X7901,cv33_W,cv33_B)
			JCudaTensor x243;
			JCudaTensor x244, x245, x246;
			x244 = x237;
			x245 = x247;
			x246 = x248;
			x243 = x249.forward(x244, x245, x246);

			// Dealloc(X7901)
			JCudaTensor x250;
			x250 = x237;
			x250.free();

			// val X7906 = Convolv(1,2)(X7905,cv35_W,cv35_B)
			JCudaTensor x251;
			JCudaTensor x252, x253, x254;
			x252 = x240;
			x253 = x255;
			x254 = x256;
			x251 = x257.forward(x252, x253, x254);

			// Dealloc(X7905)
			JCudaTensor x258;
			x258 = x240;
			x258.free();

			// val X7899 = ReLU()(X7898)
			JCudaTensor x259;
			JCudaTensor x260;
			x260 = x221;
			x259 = x261.forward(x260);

			// val X7903 = ReLU()(X7902)
			JCudaTensor x262;
			JCudaTensor x263;
			x263 = x243;
			x262 = x264.forward(x263);

			// val X7907 = ReLU()(X7906)
			JCudaTensor x265;
			JCudaTensor x266;
			x266 = x251;
			x265 = x267.forward(x266);

			// val X7910 = ReLU()(X7909)
			JCudaTensor x268;
			JCudaTensor x269;
			x269 = x229;
			x268 = x267.forward(x269);

			// val X7911 = Concat(X7899,X7903,X7907,X7910)
			JCudaTensor x270;
			JCudaTensor x272, x273, x274, x275;
			x272 = x259;
			x273 = x262;
			x274 = x265;
			x275 = x268;
			x270 = x271.forward(x272,x273,x274,x275);

			// Dealloc(X7910)
			JCudaTensor x276;
			x276 = x268;
			x276.free();

			// Dealloc(X7907)
			JCudaTensor x277;
			x277 = x265;
			x277.free();

			// Dealloc(X7903)
			JCudaTensor x278;
			x278 = x262;
			x278.free();

			// Dealloc(X7899)
			JCudaTensor x279;
			x279 = x259;
			x279.free();

			// val X7912 = Convolv(1,0)(X7911,cv41_W,cv41_B)
			JCudaTensor x280;
			JCudaTensor x281, x282, x283;
			x281 = x270;
			x282 = x284;
			x283 = x285;
			x280 = x227.forward(x281, x282, x283);

			// val X7918 = Convolv(1,0)(X7911,cv44_W,cv44_B)
			JCudaTensor x286;
			JCudaTensor x287, x288, x289;
			x287 = x270;
			x288 = x290;
			x289 = x291;
			x286 = x220.forward(x287, x288, x289);

			// val X7914 = Convolv(1,0)(X7911,cv42_W,cv42_B)
			JCudaTensor x292;
			JCudaTensor x293, x294, x295;
			x293 = x270;
			x294 = x296;
			x295 = x297;
			x292 = x210.forward(x293, x294, x295);

			// val X7922 = Pooling(3,1,1,true)(X7911)
			JCudaTensor x298;
			JCudaTensor x299;
			x299 = x270;
			x298 = x213.forward(x299);

			// Dealloc(X7911)
			JCudaTensor x300;
			x300 = x270;
			x300.free();

			// val X7915 = ReLU()(X7914)
			JCudaTensor x301;
			JCudaTensor x302;
			x302 = x292;
			x301 = x239.forward(x302);

			// val X7919 = ReLU()(X7918)
			JCudaTensor x303;
			JCudaTensor x304;
			x304 = x286;
			x303 = x242.forward(x304);

			// val X7923 = Convolv(1,0)(X7922,cv46_W,cv46_B)
			JCudaTensor x305;
			JCudaTensor x306, x307, x308;
			x306 = x298;
			x307 = x309;
			x308 = x310;
			x305 = x235.forward(x306, x307, x308);

			// Dealloc(X7922)
			JCudaTensor x311;
			x311 = x298;
			x311.free();

			// val X7916 = Convolv(1,1)(X7915,cv43_W,cv43_B)
			JCudaTensor x312;
			JCudaTensor x313, x314, x315;
			x313 = x301;
			x314 = x316;
			x315 = x317;
			x312 = x249.forward(x313, x314, x315);

			// Dealloc(X7915)
			JCudaTensor x318;
			x318 = x301;
			x318.free();

			// val X7920 = Convolv(1,2)(X7919,cv45_W,cv45_B)
			JCudaTensor x319;
			JCudaTensor x320, x321, x322;
			x320 = x303;
			x321 = x323;
			x322 = x324;
			x319 = x257.forward(x320, x321, x322);

			// Dealloc(X7919)
			JCudaTensor x325;
			x325 = x303;
			x325.free();

			// val X7913 = ReLU()(X7912)
			JCudaTensor x326;
			JCudaTensor x327;
			x327 = x280;
			x326 = x261.forward(x327);

			// val X7917 = ReLU()(X7916)
			JCudaTensor x328;
			JCudaTensor x329;
			x329 = x312;
			x328 = x264.forward(x329);

			// val X7921 = ReLU()(X7920)
			JCudaTensor x330;
			JCudaTensor x331;
			x331 = x319;
			x330 = x267.forward(x331);

			// val X7924 = ReLU()(X7923)
			JCudaTensor x332;
			JCudaTensor x333;
			x333 = x305;
			x332 = x267.forward(x333);

			// val X7925 = Concat(X7913,X7917,X7921,X7924)
			JCudaTensor x334;
			JCudaTensor x335, x336, x337, x338;
			x335 = x326;
			x336 = x328;
			x337 = x330;
			x338 = x332;
			x334 = x271.forward(x335,x336,x337,x338);

			// Dealloc(X7924)
			JCudaTensor x339;
			x339 = x332;
			x339.free();

			// Dealloc(X7921)
			JCudaTensor x340;
			x340 = x330;
			x340.free();

			// Dealloc(X7917)
			JCudaTensor x341;
			x341 = x328;
			x341.free();

			// Dealloc(X7913)
			JCudaTensor x342;
			x342 = x326;
			x342.free();

			// val X7932 = Convolv(1,0)(X7925,cv54_W,cv54_B)
			JCudaTensor x343;
			JCudaTensor x344, x345, x346;
			x344 = x334;
			x345 = x347;
			x346 = x348;
			x343 = x220.forward(x344, x345, x346);

			// val X7936 = Pooling(3,1,1,true)(X7925)
			JCudaTensor x349;
			JCudaTensor x350;
			x350 = x334;
			x349 = x213.forward(x350);

			// val X7926 = Convolv(1,0)(X7925,cv51_W,cv51_B)
			JCudaTensor x351;
			JCudaTensor x352, x353, x354;
			x352 = x334;
			x353 = x355;
			x354 = x356;
			x351 = x227.forward(x352, x353, x354);

			// val X7928 = Convolv(1,0)(X7925,cv52_W,cv52_B)
			JCudaTensor x357;
			JCudaTensor x358, x359, x360;
			x358 = x334;
			x359 = x361;
			x360 = x362;
			x357 = x210.forward(x358, x359, x360);

			// Dealloc(X7925)
			JCudaTensor x363;
			x363 = x334;
			x363.free();

			// val X7937 = Convolv(1,0)(X7936,cv56_W,cv56_B)
			JCudaTensor x364;
			JCudaTensor x365, x366, x367;
			x365 = x349;
			x366 = x368;
			x367 = x369;
			x364 = x235.forward(x365, x366, x367);

			// Dealloc(X7936)
			JCudaTensor x370;
			x370 = x349;
			x370.free();

			// val X7933 = ReLU()(X7932)
			JCudaTensor x371;
			JCudaTensor x372;
			x372 = x343;
			x371 = x242.forward(x372);

			// val X7929 = ReLU()(X7928)
			JCudaTensor x373;
			JCudaTensor x374;
			x374 = x357;
			x373 = x239.forward(x374);

			// val X7930 = Convolv(1,1)(X7929,cv53_W,cv53_B)
			JCudaTensor x375;
			JCudaTensor x376, x377, x378;
			x376 = x373;
			x377 = x379;
			x378 = x380;
			x375 = x249.forward(x376, x377, x378);

			// Dealloc(X7929)
			JCudaTensor x381;
			x381 = x373;
			x381.free();

			// val X7934 = Convolv(1,2)(X7933,cv55_W,cv55_B)
			JCudaTensor x382;
			JCudaTensor x383, x384, x385;
			x383 = x371;
			x384 = x386;
			x385 = x387;
			x382 = x257.forward(x383, x384, x385);

			// Dealloc(X7933)
			JCudaTensor x388;
			x388 = x371;
			x388.free();

			// val X7927 = ReLU()(X7926)
			JCudaTensor x389;
			JCudaTensor x390;
			x390 = x351;
			x389 = x261.forward(x390);

			// val X7931 = ReLU()(X7930)
			JCudaTensor x391;
			JCudaTensor x392;
			x392 = x375;
			x391 = x264.forward(x392);

			// val X7935 = ReLU()(X7934)
			JCudaTensor x393;
			JCudaTensor x394;
			x394 = x382;
			x393 = x267.forward(x394);

			// val X7938 = ReLU()(X7937)
			JCudaTensor x395;
			JCudaTensor x396;
			x396 = x364;
			x395 = x267.forward(x396);

			// val X7939 = Concat(X7927,X7931,X7935,X7938)
			JCudaTensor x397;
			JCudaTensor x398, x399, x400, x401;
			x398 = x389;
			x399 = x391;
			x400 = x393;
			x401 = x395;
			x397 = x271.forward(x398,x399,x400,x401);

			// Dealloc(X7938)
			JCudaTensor x402;
			x402 = x395;
			x402.free();

			// Dealloc(X7935)
			JCudaTensor x403;
			x403 = x393;
			x403.free();

			// Dealloc(X7931)
			JCudaTensor x404;
			x404 = x391;
			x404.free();

			// Dealloc(X7927)
			JCudaTensor x405;
			x405 = x389;
			x405.free();

			// val X7946 = Convolv(1,0)(X7939,cv64_W,cv64_B)
			JCudaTensor x406;
			JCudaTensor x407, x408, x409;
			x407 = x397;
			x408 = x410;
			x409 = x411;
			x406 = x220.forward(x407, x408, x409);

			// val X7942 = Convolv(1,0)(X7939,cv62_W,cv62_B)
			JCudaTensor x412;
			JCudaTensor x413, x414, x415;
			x413 = x397;
			x414 = x416;
			x415 = x417;
			x412 = x210.forward(x413, x414, x415);

			// val X7940 = Convolv(1,0)(X7939,cv61_W,cv61_B)
			JCudaTensor x418;
			JCudaTensor x419, x420, x421;
			x419 = x397;
			x420 = x422;
			x421 = x423;
			x418 = x227.forward(x419, x420, x421);

			// val X7950 = Pooling(3,1,1,true)(X7939)
			JCudaTensor x424;
			JCudaTensor x425;
			x425 = x397;
			x424 = x213.forward(x425);

			// Dealloc(X7939)
			JCudaTensor x426;
			x426 = x397;
			x426.free();

			// val X7947 = ReLU()(X7946)
			JCudaTensor x427;
			JCudaTensor x428;
			x428 = x406;
			x427 = x242.forward(x428);

			// val X7951 = Convolv(1,0)(X7950,cv66_W,cv66_B)
			JCudaTensor x429;
			JCudaTensor x430, x431, x432;
			x430 = x424;
			x431 = x433;
			x432 = x434;
			x429 = x235.forward(x430, x431, x432);

			// Dealloc(X7950)
			JCudaTensor x435;
			x435 = x424;
			x435.free();

			// val X7943 = ReLU()(X7942)
			JCudaTensor x436;
			JCudaTensor x437;
			x437 = x412;
			x436 = x239.forward(x437);

			// val X7948 = Convolv(1,2)(X7947,cv65_W,cv65_B)
			JCudaTensor x438;
			JCudaTensor x439, x440, x441;
			x439 = x427;
			x440 = x442;
			x441 = x443;
			x438 = x257.forward(x439, x440, x441);

			// Dealloc(X7947)
			JCudaTensor x444;
			x444 = x427;
			x444.free();

			// val X7944 = Convolv(1,1)(X7943,cv63_W,cv63_B)
			JCudaTensor x445;
			JCudaTensor x446, x447, x448;
			x446 = x436;
			x447 = x449;
			x448 = x450;
			x445 = x249.forward(x446, x447, x448);

			// Dealloc(X7943)
			JCudaTensor x451;
			x451 = x436;
			x451.free();

			// val X7941 = ReLU()(X7940)
			JCudaTensor x452;
			JCudaTensor x453;
			x453 = x418;
			x452 = x261.forward(x453);

			// val X7945 = ReLU()(X7944)
			JCudaTensor x454;
			JCudaTensor x455;
			x455 = x445;
			x454 = x264.forward(x455);

			// val X7949 = ReLU()(X7948)
			JCudaTensor x456;
			JCudaTensor x457;
			x457 = x438;
			x456 = x267.forward(x457);

			// val X7952 = ReLU()(X7951)
			JCudaTensor x458;
			JCudaTensor x459;
			x459 = x429;
			x458 = x267.forward(x459);

			// val X7953 = Concat(X7941,X7945,X7949,X7952)
			JCudaTensor x460;
			JCudaTensor x461, x462, x463, x464;
			x461 = x452;
			x462 = x454;
			x463 = x456;
			x464 = x458;
			x460 = x271.forward(x461,x462,x463,x464);

			// Dealloc(X7952)
			JCudaTensor x465;
			x465 = x458;
			x465.free();

			// Dealloc(X7949)
			JCudaTensor x466;
			x466 = x456;
			x466.free();

			// Dealloc(X7945)
			JCudaTensor x467;
			x467 = x454;
			x467.free();

			// Dealloc(X7941)
			JCudaTensor x468;
			x468 = x452;
			x468.free();

			// val X7960 = Convolv(1,0)(X7953,cv74_W,cv74_B)
			JCudaTensor x469;
			JCudaTensor x470, x471, x472;
			x470 = x460;
			x471 = x473;
			x472 = x474;
			x469 = x220.forward(x470, x471, x472);

			// val X7964 = Pooling(3,1,1,true)(X7953)
			JCudaTensor x475;
			JCudaTensor x476;
			x476 = x460;
			x475 = x213.forward(x476);

			// val X7956 = Convolv(1,0)(X7953,cv72_W,cv72_B)
			JCudaTensor x477;
			JCudaTensor x478, x479, x480;
			x478 = x460;
			x479 = x481;
			x480 = x482;
			x477 = x210.forward(x478, x479, x480);

			// val X7954 = Convolv(1,0)(X7953,cv71_W,cv71_B)
			JCudaTensor x483;
			JCudaTensor x484, x485, x486;
			x484 = x460;
			x485 = x487;
			x486 = x488;
			x483 = x227.forward(x484, x485, x486);

			// Dealloc(X7953)
			JCudaTensor x489;
			x489 = x460;
			x489.free();

			// val X7957 = ReLU()(X7956)
			JCudaTensor x490;
			JCudaTensor x491;
			x491 = x477;
			x490 = x239.forward(x491);

			// val X7961 = ReLU()(X7960)
			JCudaTensor x492;
			JCudaTensor x493;
			x493 = x469;
			x492 = x242.forward(x493);

			// val X7965 = Convolv(1,0)(X7964,cv76_W,cv76_B)
			JCudaTensor x494;
			JCudaTensor x495, x496, x497;
			x495 = x475;
			x496 = x498;
			x497 = x499;
			x494 = x235.forward(x495, x496, x497);

			// Dealloc(X7964)
			JCudaTensor x500;
			x500 = x475;
			x500.free();

			// val X7958 = Convolv(1,1)(X7957,cv73_W,cv73_B)
			JCudaTensor x501;
			JCudaTensor x502, x503, x504;
			x502 = x490;
			x503 = x505;
			x504 = x506;
			x501 = x249.forward(x502, x503, x504);

			// Dealloc(X7957)
			JCudaTensor x507;
			x507 = x490;
			x507.free();

			// val X7962 = Convolv(1,2)(X7961,cv75_W,cv75_B)
			JCudaTensor x508;
			JCudaTensor x509, x510, x511;
			x509 = x492;
			x510 = x512;
			x511 = x513;
			x508 = x257.forward(x509, x510, x511);

			// Dealloc(X7961)
			JCudaTensor x514;
			x514 = x492;
			x514.free();

			// val X7955 = ReLU()(X7954)
			JCudaTensor x515;
			JCudaTensor x516;
			x516 = x483;
			x515 = x261.forward(x516);

			// val X7959 = ReLU()(X7958)
			JCudaTensor x517;
			JCudaTensor x518;
			x518 = x501;
			x517 = x264.forward(x518);

			// val X7963 = ReLU()(X7962)
			JCudaTensor x519;
			JCudaTensor x520;
			x520 = x508;
			x519 = x267.forward(x520);

			// val X7966 = ReLU()(X7965)
			JCudaTensor x521;
			JCudaTensor x522;
			x522 = x494;
			x521 = x267.forward(x522);

			// val X7967 = Concat(X7955,X7959,X7963,X7966)
			JCudaTensor x523;
			JCudaTensor x524, x525, x526, x527;
			x524 = x515;
			x525 = x517;
			x526 = x519;
			x527 = x521;
			x523 = x271.forward(x524,x525,x526,x527);

			// Dealloc(X7966)
			JCudaTensor x528;
			x528 = x521;
			x528.free();

			// Dealloc(X7963)
			JCudaTensor x529;
			x529 = x519;
			x529.free();

			// Dealloc(X7959)
			JCudaTensor x530;
			x530 = x517;
			x530.free();

			// Dealloc(X7955)
			JCudaTensor x531;
			x531 = x515;
			x531.free();

			// val X7968 = Pooling(3,2,1,true)(X7967)
			JCudaTensor x532;
			JCudaTensor x533;
			x533 = x523;
			x532 = x534.forward(x533);

			// Dealloc(X7967)
			JCudaTensor x535;
			x535 = x523;
			x535.free();

			// val X7971 = Convolv(1,0)(X7968,cv82_W,cv82_B)
			JCudaTensor x536;
			JCudaTensor x537, x538, x539;
			x537 = x532;
			x538 = x540;
			x539 = x541;
			x536 = x542.forward(x537, x538, x539);

			// val X7975 = Convolv(1,0)(X7968,cv84_W,cv84_B)
			JCudaTensor x543;
			JCudaTensor x544, x545, x546;
			x544 = x532;
			x545 = x547;
			x546 = x548;
			x543 = x549.forward(x544, x545, x546);

			// val X7969 = Convolv(1,0)(X7968,cv81_W,cv81_B)
			JCudaTensor x550;
			JCudaTensor x551, x552, x553;
			x551 = x532;
			x552 = x554;
			x553 = x555;
			x550 = x556.forward(x551, x552, x553);

			// val X7979 = Pooling(3,1,1,true)(X7968)
			JCudaTensor x557;
			JCudaTensor x558;
			x558 = x532;
			x557 = x559.forward(x558);

			// Dealloc(X7968)
			JCudaTensor x560;
			x560 = x532;
			x560.free();

			// val X7976 = ReLU()(X7975)
			JCudaTensor x561;
			JCudaTensor x562;
			x562 = x543;
			x561 = x563.forward(x562);

			// val X7972 = ReLU()(X7971)
			JCudaTensor x564;
			JCudaTensor x565;
			x565 = x536;
			x564 = x566.forward(x565);

			// val X7980 = Convolv(1,0)(X7979,cv86_W,cv86_B)
			JCudaTensor x567;
			JCudaTensor x568, x569, x570;
			x568 = x557;
			x569 = x571;
			x570 = x572;
			x567 = x573.forward(x568, x569, x570);

			// Dealloc(X7979)
			JCudaTensor x574;
			x574 = x557;
			x574.free();

			// val X7973 = Convolv(1,1)(X7972,cv83_W,cv83_B)
			JCudaTensor x575;
			JCudaTensor x576, x577, x578;
			x576 = x564;
			x577 = x579;
			x578 = x580;
			x575 = x581.forward(x576, x577, x578);

			// Dealloc(X7972)
			JCudaTensor x582;
			x582 = x564;
			x582.free();

			// val X7977 = Convolv(1,2)(X7976,cv85_W,cv85_B)
			JCudaTensor x583;
			JCudaTensor x584, x585, x586;
			x584 = x561;
			x585 = x587;
			x586 = x588;
			x583 = x589.forward(x584, x585, x586);

			// Dealloc(X7976)
			JCudaTensor x590;
			x590 = x561;
			x590.free();

			// val X7970 = ReLU()(X7969)
			JCudaTensor x591;
			JCudaTensor x592;
			x592 = x550;
			x591 = x593.forward(x592);

			// val X7974 = ReLU()(X7973)
			JCudaTensor x594;
			JCudaTensor x595;
			x595 = x575;
			x594 = x596.forward(x595);

			// val X7978 = ReLU()(X7977)
			JCudaTensor x597;
			JCudaTensor x598;
			x598 = x583;
			x597 = x599.forward(x598);

			// val X7981 = ReLU()(X7980)
			JCudaTensor x600;
			JCudaTensor x601;
			x601 = x567;
			x600 = x599.forward(x601);

			// val X7982 = Concat(X7970,X7974,X7978,X7981)
			JCudaTensor x602;
			JCudaTensor x604, x605, x606, x607;
			x604 = x591;
			x605 = x594;
			x606 = x597;
			x607 = x600;
			x602 = x603.forward(x604,x605,x606,x607);

			// Dealloc(X7981)
			JCudaTensor x608;
			x608 = x600;
			x608.free();

			// Dealloc(X7978)
			JCudaTensor x609;
			x609 = x597;
			x609.free();

			// Dealloc(X7974)
			JCudaTensor x610;
			x610 = x594;
			x610.free();

			// Dealloc(X7970)
			JCudaTensor x611;
			x611 = x591;
			x611.free();

			// val X7983 = Convolv(1,0)(X7982,cv91_W,cv91_B)
			JCudaTensor x612;
			JCudaTensor x613, x614, x615;
			x613 = x602;
			x614 = x616;
			x615 = x617;
			x612 = x556.forward(x613, x614, x615);

			// val X7993 = Pooling(3,1,1,true)(X7982)
			JCudaTensor x618;
			JCudaTensor x619;
			x619 = x602;
			x618 = x559.forward(x619);

			// val X7989 = Convolv(1,0)(X7982,cv94_W,cv94_B)
			JCudaTensor x620;
			JCudaTensor x621, x622, x623;
			x621 = x602;
			x622 = x624;
			x623 = x625;
			x620 = x549.forward(x621, x622, x623);

			// val X7985 = Convolv(1,0)(X7982,cv92_W,cv92_B)
			JCudaTensor x626;
			JCudaTensor x627, x628, x629;
			x627 = x602;
			x628 = x630;
			x629 = x631;
			x626 = x542.forward(x627, x628, x629);

			// Dealloc(X7982)
			JCudaTensor x632;
			x632 = x602;
			x632.free();

			// val X7994 = Convolv(1,0)(X7993,cv96_W,cv96_B)
			JCudaTensor x633;
			JCudaTensor x634, x635, x636;
			x634 = x618;
			x635 = x637;
			x636 = x638;
			x633 = x573.forward(x634, x635, x636);

			// Dealloc(X7993)
			JCudaTensor x639;
			x639 = x618;
			x639.free();

			// val X7986 = ReLU()(X7985)
			JCudaTensor x640;
			JCudaTensor x641;
			x641 = x626;
			x640 = x566.forward(x641);

			// val X7990 = ReLU()(X7989)
			JCudaTensor x642;
			JCudaTensor x643;
			x643 = x620;
			x642 = x563.forward(x643);

			// val X7987 = Convolv(1,1)(X7986,cv93_W,cv93_B)
			JCudaTensor x644;
			JCudaTensor x645, x646, x647;
			x645 = x640;
			x646 = x648;
			x647 = x649;
			x644 = x581.forward(x645, x646, x647);

			// Dealloc(X7986)
			JCudaTensor x650;
			x650 = x640;
			x650.free();

			// val X7991 = Convolv(1,2)(X7990,cv95_W,cv95_B)
			JCudaTensor x651;
			JCudaTensor x652, x653, x654;
			x652 = x642;
			x653 = x655;
			x654 = x656;
			x651 = x589.forward(x652, x653, x654);

			// Dealloc(X7990)
			JCudaTensor x657;
			x657 = x642;
			x657.free();

			// val X7984 = ReLU()(X7983)
			JCudaTensor x658;
			JCudaTensor x659;
			x659 = x612;
			x658 = x593.forward(x659);

			// val X7988 = ReLU()(X7987)
			JCudaTensor x660;
			JCudaTensor x661;
			x661 = x644;
			x660 = x596.forward(x661);

			// val X7992 = ReLU()(X7991)
			JCudaTensor x662;
			JCudaTensor x663;
			x663 = x651;
			x662 = x599.forward(x663);

			// val X7995 = ReLU()(X7994)
			JCudaTensor x664;
			JCudaTensor x665;
			x665 = x633;
			x664 = x599.forward(x665);

			// val X7996 = Concat(X7984,X7988,X7992,X7995)
			JCudaTensor x666;
			JCudaTensor x667, x668, x669, x670;
			x667 = x658;
			x668 = x660;
			x669 = x662;
			x670 = x664;
			x666 = x603.forward(x667,x668,x669,x670);

			// Dealloc(X7995)
			JCudaTensor x671;
			x671 = x664;
			x671.free();

			// Dealloc(X7992)
			JCudaTensor x672;
			x672 = x662;
			x672.free();

			// Dealloc(X7988)
			JCudaTensor x673;
			x673 = x660;
			x673.free();

			// Dealloc(X7984)
			JCudaTensor x674;
			x674 = x658;
			x674.free();

			// val X7997 = Pooling(7,1,0,false)(X7996)
			JCudaTensor x675;
			JCudaTensor x676;
			x676 = x666;
			x675 = x677.forward(x676);

			// Dealloc(X7996)
			JCudaTensor x678;
			x678 = x666;
			x678.free();

			// val X7998 = Dropout(0.4)(X7997)
			JCudaTensor x679;
			JCudaTensor x680;
			x680 = x675;
			x679 = x681.forward(x680);

			// Dealloc(X7997)
			JCudaTensor x682;
			x682 = x675;
			x682.free();

			// val X7999 = (X7998[1><3])(i | @) * (fc_W)(j | @)
			JCudaTensor x683;
			JCudaMatrix x684;
			JCudaMatrix x685;
			JCudaTensor x686;
			JCudaTensor x687;
			x687 = x679;
			x686 = x687.flatten(1, new int[]{256, 1, 1});
			x684 = x686.asMatrix(1, true);
			JCudaTensor x688;
			x688 = x689;
			x685 = x688.asMatrix(1, true);
			x683 = x684.times(x685);

			// Dealloc(X7998)
			JCudaTensor x690;
			x690 = x679;
			x690.free();

			// val X8001 = (X7999 + (i) => fc_B)
			JCudaTensor x691;
			JCudaTensor x692, x693;
			x692 = x683;
			x693 = x694;
			x691 = x693.copy(128, x692);

			// Prediction(X8001)
			JCudaTensor x695;
			x695 = x691;
			System.out.println(x3 + " inference " + java.util.Arrays.toString(x695.prediction()));

			// Dealloc(X8001)
			JCudaTensor x696;
			x696 = x691;
			x696.free();

		}

	}

}