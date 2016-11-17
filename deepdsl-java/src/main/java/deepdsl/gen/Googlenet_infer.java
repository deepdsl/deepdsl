package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;

// This file is for inference only, which needs trained parameters.
public class Googlenet_infer {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
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

	// (Convolv(1,0),List(List(128, 192, 28, 28), List(16, 192, 1, 1), List(16)))
	static JCudnnConvolution x79 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{16,192,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(32, 192, 1, 1), List(32)))
	static JCudnnConvolution x87 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{32,192,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(64, 192, 1, 1), List(64)))
	static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{64,192,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 192, 28, 28), List(96, 192, 1, 1), List(96)))
	static JCudnnConvolution x69 = new JCudnnConvolution(new int[]{128,192,28,28},new int[]{96,192,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x227 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x241 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x210 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 14, 14), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x217 = new JCudnnConvolution(new int[]{128,256,14,14},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x141 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x165 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x148 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 28, 28), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x155 = new JCudnnConvolution(new int[]{128,256,28,28},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(16, 256, 1, 1), List(16)))
	static JCudnnConvolution x545 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{16,256,1,1},new int[]{16}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(32, 256, 1, 1), List(32)))
	static JCudnnConvolution x567 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{32,256,1,1},new int[]{32}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(64, 256, 1, 1), List(64)))
	static JCudnnConvolution x559 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{64,256,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,0),List(List(128, 256, 7, 7), List(96, 256, 1, 1), List(96)))
	static JCudnnConvolution x552 = new JCudnnConvolution(new int[]{128,256,7,7},new int[]{96,256,1,1},new int[]{96}, 1, 0);
	// (Convolv(1,0),List(List(128, 64, 56, 56), List(64, 64, 1, 1), List(64)))
	static JCudnnConvolution x32 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{64,64,1,1},new int[]{64}, 1, 0);
	// (Convolv(1,1),List(List(128, 64, 56, 56), List(192, 64, 3, 3), List(192)))
	static JCudnnConvolution x43 = new JCudnnConvolution(new int[]{128,64,56,56},new int[]{192,64,3,3},new int[]{192}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 14, 14), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x257 = new JCudnnConvolution(new int[]{128,96,14,14},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 28, 28), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x101 = new JCudnnConvolution(new int[]{128,96,28,28},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(128, 96, 7, 7), List(128, 96, 3, 3), List(128)))
	static JCudnnConvolution x581 = new JCudnnConvolution(new int[]{128,96,7,7},new int[]{128,96,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,2),List(List(128, 16, 14, 14), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x249 = new JCudnnConvolution(new int[]{128,16,14,14},new int[]{32,16,5,5},new int[]{32}, 1, 2);
	// (Convolv(1,2),List(List(128, 16, 28, 28), List(32, 16, 5, 5), List(32)))
	static JCudnnConvolution x109 = new JCudnnConvolution(new int[]{128,16,28,28},new int[]{32,16,5,5},new int[]{32}, 1, 2);
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
	// (Lmdb(1000000,10000,Win32,1000),false)
	static LmdbFactory x1 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, platform, 1000, true);
	// (Pooling(3,1,1,true),List(List(128, 192, 28, 28)))
	static JCudnnPooling x72 = new JCudnnPooling(new int[]{128,192,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 14, 14)))
	static JCudnnPooling x220 = new JCudnnPooling(new int[]{128,256,14,14}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 28, 28)))
	static JCudnnPooling x134 = new JCudnnPooling(new int[]{128,256,28,28}, 3, 1, 1, PoolingType.MAX);
	// (Pooling(3,1,1,true),List(List(128, 256, 7, 7)))
	static JCudnnPooling x538 = new JCudnnPooling(new int[]{128,256,7,7}, 3, 1, 1, PoolingType.MAX);
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
	static JCudnnActivation x231 = new JCudnnActivation(new int[]{128,16,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 28, 28)))
	static JCudnnActivation x94 = new JCudnnActivation(new int[]{128,16,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 16, 7, 7)))
	static JCudnnActivation x574 = new JCudnnActivation(new int[]{128,16,7,7}, ActivationMode.RELU);
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
	static JCudnnActivation x234 = new JCudnnActivation(new int[]{128,96,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 28, 28)))
	static JCudnnActivation x91 = new JCudnnActivation(new int[]{128,96,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 7, 7)))
	static JCudnnActivation x571 = new JCudnnActivation(new int[]{128,96,7,7}, ActivationMode.RELU);
	// List(List(128, 64, 14, 14), List(128, 128, 14, 14), List(128, 32, 14, 14), List(128, 32, 14, 14))
	static JCudnnConcat x271 = new JCudnnConcat(new int[]{128,64,14,14},new int[]{128,128,14,14},new int[]{128,32,14,14},new int[]{128,32,14,14});
	// List(List(128, 64, 28, 28), List(128, 128, 28, 28), List(128, 32, 28, 28), List(128, 32, 28, 28))
	static JCudnnConcat x123 = new JCudnnConcat(new int[]{128,64,28,28},new int[]{128,128,28,28},new int[]{128,32,28,28},new int[]{128,32,28,28});
	// List(List(128, 64, 7, 7), List(128, 128, 7, 7), List(128, 32, 7, 7), List(128, 32, 7, 7))
	static JCudnnConcat x603 = new JCudnnConcat(new int[]{128,64,7,7},new int[]{128,128,7,7},new int[]{128,32,7,7},new int[]{128,32,7,7});
	// X
	static JTensorFloat x2;
	// cv11_B
	static JCudaTensor x61 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
	// cv11_W
	static JCudaTensor x60 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 64, 192, 1, 1).load(network_dir + "/cv11_W").asJCudaTensor();
	// cv12_B
	static JCudaTensor x68 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv12_B").asJCudaTensor();
	// cv12_W
	static JCudaTensor x67 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 96, 192, 1, 1).load(network_dir + "/cv12_W").asJCudaTensor();
	// cv13_B
	static JCudaTensor x100 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv13_B").asJCudaTensor();
	// cv13_W
	static JCudaTensor x99 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv13_W").asJCudaTensor();
	// cv14_B
	static JCudaTensor x78 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv14_B").asJCudaTensor();
	// cv14_W
	static JCudaTensor x77 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 16, 192, 1, 1).load(network_dir + "/cv14_W").asJCudaTensor();
	// cv15_B
	static JCudaTensor x108 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv15_B").asJCudaTensor();
	// cv15_W
	static JCudaTensor x107 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv15_W").asJCudaTensor();
	// cv16_B
	static JCudaTensor x86 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv16_B").asJCudaTensor();
	// cv16_W
	static JCudaTensor x85 = JTensor.randomFloat(-0.10206208f, 0.10206208f, 32, 192, 1, 1).load(network_dir + "/cv16_W").asJCudaTensor();
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
	static JCudaTensor x140 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv24_B").asJCudaTensor();
	// cv24_W
	static JCudaTensor x139 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv24_W").asJCudaTensor();
	// cv25_B
	static JCudaTensor x181 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv25_B").asJCudaTensor();
	// cv25_W
	static JCudaTensor x180 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv25_W").asJCudaTensor();
	// cv26_B
	static JCudaTensor x164 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv26_B").asJCudaTensor();
	// cv26_W
	static JCudaTensor x163 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv26_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x31 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x30 = JTensor.randomFloat(-0.17677669f, 0.17677669f, 64, 64, 1, 1).load(network_dir + "/cv2_W").asJCudaTensor();
	// cv31_B
	static JCudaTensor x209 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv31_B").asJCudaTensor();
	// cv31_W
	static JCudaTensor x208 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv31_W").asJCudaTensor();
	// cv32_B
	static JCudaTensor x216 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv32_B").asJCudaTensor();
	// cv32_W
	static JCudaTensor x215 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv32_W").asJCudaTensor();
	// cv33_B
	static JCudaTensor x256 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv33_B").asJCudaTensor();
	// cv33_W
	static JCudaTensor x255 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
	// cv34_B
	static JCudaTensor x226 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv34_B").asJCudaTensor();
	// cv34_W
	static JCudaTensor x225 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv34_W").asJCudaTensor();
	// cv35_B
	static JCudaTensor x248 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv35_B").asJCudaTensor();
	// cv35_W
	static JCudaTensor x247 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv35_W").asJCudaTensor();
	// cv36_B
	static JCudaTensor x240 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv36_B").asJCudaTensor();
	// cv36_W
	static JCudaTensor x239 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv36_W").asJCudaTensor();
	// cv3_B
	static JCudaTensor x42 = JTensor.constFloat(0.2f, 192).load(network_dir + "/cv3_B").asJCudaTensor();
	// cv3_W
	static JCudaTensor x41 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 192, 64, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
	// cv41_B
	static JCudaTensor x287 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv41_B").asJCudaTensor();
	// cv41_W
	static JCudaTensor x286 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv41_W").asJCudaTensor();
	// cv42_B
	static JCudaTensor x293 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv42_B").asJCudaTensor();
	// cv42_W
	static JCudaTensor x292 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv42_W").asJCudaTensor();
	// cv43_B
	static JCudaTensor x324 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv43_B").asJCudaTensor();
	// cv43_W
	static JCudaTensor x323 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
	// cv44_B
	static JCudaTensor x299 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv44_B").asJCudaTensor();
	// cv44_W
	static JCudaTensor x298 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv44_W").asJCudaTensor();
	// cv45_B
	static JCudaTensor x317 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv45_B").asJCudaTensor();
	// cv45_W
	static JCudaTensor x316 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv45_W").asJCudaTensor();
	// cv46_B
	static JCudaTensor x310 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv46_B").asJCudaTensor();
	// cv46_W
	static JCudaTensor x309 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv46_W").asJCudaTensor();
	// cv51_B
	static JCudaTensor x354 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv51_B").asJCudaTensor();
	// cv51_W
	static JCudaTensor x353 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv51_W").asJCudaTensor();
	// cv52_B
	static JCudaTensor x360 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv52_B").asJCudaTensor();
	// cv52_W
	static JCudaTensor x359 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv52_W").asJCudaTensor();
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
	static JCudaTensor x371 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv56_B").asJCudaTensor();
	// cv56_W
	static JCudaTensor x370 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv56_W").asJCudaTensor();
	// cv61_B
	static JCudaTensor x413 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv61_B").asJCudaTensor();
	// cv61_W
	static JCudaTensor x412 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv61_W").asJCudaTensor();
	// cv62_B
	static JCudaTensor x419 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv62_B").asJCudaTensor();
	// cv62_W
	static JCudaTensor x418 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv62_W").asJCudaTensor();
	// cv63_B
	static JCudaTensor x443 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv63_B").asJCudaTensor();
	// cv63_W
	static JCudaTensor x442 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv63_W").asJCudaTensor();
	// cv64_B
	static JCudaTensor x425 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv64_B").asJCudaTensor();
	// cv64_W
	static JCudaTensor x424 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv64_W").asJCudaTensor();
	// cv65_B
	static JCudaTensor x450 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv65_B").asJCudaTensor();
	// cv65_W
	static JCudaTensor x449 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv65_W").asJCudaTensor();
	// cv66_B
	static JCudaTensor x432 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv66_B").asJCudaTensor();
	// cv66_W
	static JCudaTensor x431 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv66_W").asJCudaTensor();
	// cv71_B
	static JCudaTensor x488 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv71_B").asJCudaTensor();
	// cv71_W
	static JCudaTensor x487 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv71_W").asJCudaTensor();
	// cv72_B
	static JCudaTensor x474 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv72_B").asJCudaTensor();
	// cv72_W
	static JCudaTensor x473 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv72_W").asJCudaTensor();
	// cv73_B
	static JCudaTensor x513 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv73_B").asJCudaTensor();
	// cv73_W
	static JCudaTensor x512 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv73_W").asJCudaTensor();
	// cv74_B
	static JCudaTensor x480 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv74_B").asJCudaTensor();
	// cv74_W
	static JCudaTensor x479 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv74_W").asJCudaTensor();
	// cv75_B
	static JCudaTensor x506 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv75_B").asJCudaTensor();
	// cv75_W
	static JCudaTensor x505 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv75_W").asJCudaTensor();
	// cv76_B
	static JCudaTensor x495 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv76_B").asJCudaTensor();
	// cv76_W
	static JCudaTensor x494 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv76_W").asJCudaTensor();
	// cv81_B
	static JCudaTensor x558 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv81_B").asJCudaTensor();
	// cv81_W
	static JCudaTensor x557 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv81_W").asJCudaTensor();
	// cv82_B
	static JCudaTensor x551 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv82_B").asJCudaTensor();
	// cv82_W
	static JCudaTensor x550 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv82_W").asJCudaTensor();
	// cv83_B
	static JCudaTensor x580 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv83_B").asJCudaTensor();
	// cv83_W
	static JCudaTensor x579 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv83_W").asJCudaTensor();
	// cv84_B
	static JCudaTensor x544 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv84_B").asJCudaTensor();
	// cv84_W
	static JCudaTensor x543 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv84_W").asJCudaTensor();
	// cv85_B
	static JCudaTensor x588 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv85_B").asJCudaTensor();
	// cv85_W
	static JCudaTensor x587 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv85_W").asJCudaTensor();
	// cv86_B
	static JCudaTensor x566 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv86_B").asJCudaTensor();
	// cv86_W
	static JCudaTensor x565 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv86_W").asJCudaTensor();
	// cv91_B
	static JCudaTensor x623 = JTensor.constFloat(0.2f, 64).load(network_dir + "/cv91_B").asJCudaTensor();
	// cv91_W
	static JCudaTensor x622 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 64, 256, 1, 1).load(network_dir + "/cv91_W").asJCudaTensor();
	// cv92_B
	static JCudaTensor x631 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv92_B").asJCudaTensor();
	// cv92_W
	static JCudaTensor x630 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 96, 256, 1, 1).load(network_dir + "/cv92_W").asJCudaTensor();
	// cv93_B
	static JCudaTensor x649 = JTensor.constFloat(0.2f, 128).load(network_dir + "/cv93_B").asJCudaTensor();
	// cv93_W
	static JCudaTensor x648 = JTensor.randomFloat(-0.048112523f, 0.048112523f, 128, 96, 3, 3).load(network_dir + "/cv93_W").asJCudaTensor();
	// cv94_B
	static JCudaTensor x617 = JTensor.constFloat(0.2f, 16).load(network_dir + "/cv94_B").asJCudaTensor();
	// cv94_W
	static JCudaTensor x616 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 16, 256, 1, 1).load(network_dir + "/cv94_W").asJCudaTensor();
	// cv95_B
	static JCudaTensor x656 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv95_B").asJCudaTensor();
	// cv95_W
	static JCudaTensor x655 = JTensor.randomFloat(-0.07071068f, 0.07071068f, 32, 16, 5, 5).load(network_dir + "/cv95_W").asJCudaTensor();
	// cv96_B
	static JCudaTensor x642 = JTensor.constFloat(0.2f, 32).load(network_dir + "/cv96_B").asJCudaTensor();
	// cv96_W
	static JCudaTensor x641 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 32, 256, 1, 1).load(network_dir + "/cv96_W").asJCudaTensor();
	// fc_B
	static JCudaTensor x694 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc_B").asJCudaTensor();
	// fc_W
	static JCudaTensor x689 = JTensor.randomFloat(-0.088388346f, 0.088388346f, 1000, 256).load(network_dir + "/fc_W").asJCudaTensor();

	public static void main(String[] args){
		test();
		x61.free();
		x292.free();
		x317.free();
		x100.free();
		x494.free();
		x316.free();
		x424.free();
		x293.free();
		x140.free();
		x324.free();
		x474.free();
		x180.free();
		x379.free();
		x154.free();
		x164.free();
		x247.free();
		x488.free();
		x371.free();
		x648.free();
		x587.free();
		x239.free();
		x299.free();
		x147.free();
		x689.free();
		x386.free();
		x215.free();
		x181.free();
		x622.free();
		x473.free();
		x255.free();
		x370.free();
		x226.free();
		x107.free();
		x353.free();
		x67.free();
		x551.free();
		x558.free();
		x487.free();
		x656.free();
		x163.free();
		x77.free();
		x513.free();
		x505.free();
		x248.free();
		x442.free();
		x380.free();
		x310.free();
		x432.free();
		x99.free();
		x512.free();
		x256.free();
		x12.free();
		x543.free();
		x85.free();
		x60.free();
		x479.free();
		x580.free();
		x544.free();
		x78.free();
		x354.free();
		x86.free();
		x173.free();
		x631.free();
		x153.free();
		x566.free();
		x225.free();
		x108.free();
		x41.free();
		x309.free();
		x425.free();
		x359.free();
		x443.free();
		x630.free();
		x506.free();
		x139.free();
		x616.free();
		x642.free();
		x480.free();
		x565.free();
		x286.free();
		x617.free();
		x298.free();
		x655.free();
		x557.free();
		x30.free();
		x146.free();
		x579.free();
		x418.free();
		x623.free();
		x323.free();
		x287.free();
		x419.free();
		x449.free();
		x348.free();
		x216.free();
		x209.free();
		x450.free();
		x550.free();
		x68.free();
		x208.free();
		x649.free();
		x347.free();
		x42.free();
		x495.free();
		x412.free();
		x360.free();
		x694.free();
		x588.free();
		x641.free();
		x11.free();
		x174.free();
		x413.free();
		x31.free();
		x240.free();
		x431.free();
		x387.free();
		x119.free();
		x116.free();
		x24.free();
		x241.free();
		x94.free();
		x534.free();
		x227.free();
		x599.free();
		x134.free();
		x552.free();
		x231.free();
		x148.free();
		x36.free();
		x50.free();
		x596.free();
		x267.free();
		x681.free();
		x141.free();
		x574.free();
		x54.free();
		x165.free();
		x261.free();
		x257.free();
		x545.free();
		x217.free();
		x581.free();
		x91.free();
		x113.free();
		x538.free();
		x32.free();
		x79.free();
		x20.free();
		x677.free();
		x17.free();
		x13.free();
		x264.free();
		x62.free();
		x69.free();
		x155.free();
		x72.free();
		x202.free();
		x109.free();
		x571.free();
		x567.free();
		x220.free();
		x589.free();
		x87.free();
		x559.free();
		x593.free();
		x101.free();
		x43.free();
		x210.free();
		x47.free();
		x234.free();
		x249.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void test() {
		for(int x3=0; x3<test_itr; x3++) {
			JTensorFloatTuple x4 =  x1.nextFloat();
			x2 = x4.image;

			// val X82 = Cuda(X)
			JCudaTensor x5;
			JTensorFloat x6;
			x6 = x2;
			x5 = x6.asJCudaTensor();

			// val X83 = Convolv(2,3)(X82,cv1_W,cv1_B)
			JCudaTensor x7;
			JCudaTensor x8, x9, x10;
			x8 = x5;
			x9 = x11;
			x10 = x12;
			x7 = x13.forward(x8, x9, x10);

			// Dealloc(X82)
			JCudaTensor x14;
			x14 = x5;
			x14.free();

			// val X84 = ReLU()(X83)
			JCudaTensor x15;
			JCudaTensor x16;
			x16 = x7;
			x15 = x17.forward(x16);

			// val X85 = Pooling(3,2,1,true)(X84)
			JCudaTensor x18;
			JCudaTensor x19;
			x19 = x15;
			x18 = x20.forward(x19);

			// Dealloc(X84)
			JCudaTensor x21;
			x21 = x15;
			x21.free();

			// val X86 = LRN(5,1.0E-4,0.75)(X85)
			JCudaTensor x22;
			JCudaTensor x23;
			x23 = x18;
			x22 = x24.forward(x23);

			// Dealloc(X85)
			JCudaTensor x25;
			x25 = x18;
			x25.free();

			// val X87 = Convolv(1,0)(X86,cv2_W,cv2_B)
			JCudaTensor x26;
			JCudaTensor x27, x28, x29;
			x27 = x22;
			x28 = x30;
			x29 = x31;
			x26 = x32.forward(x27, x28, x29);

			// Dealloc(X86)
			JCudaTensor x33;
			x33 = x22;
			x33.free();

			// val X88 = ReLU()(X87)
			JCudaTensor x34;
			JCudaTensor x35;
			x35 = x26;
			x34 = x36.forward(x35);

			// val X89 = Convolv(1,1)(X88,cv3_W,cv3_B)
			JCudaTensor x37;
			JCudaTensor x38, x39, x40;
			x38 = x34;
			x39 = x41;
			x40 = x42;
			x37 = x43.forward(x38, x39, x40);

			// Dealloc(X88)
			JCudaTensor x44;
			x44 = x34;
			x44.free();

			// val X90 = ReLU()(X89)
			JCudaTensor x45;
			JCudaTensor x46;
			x46 = x37;
			x45 = x47.forward(x46);

			// val X91 = LRN(5,1.0E-4,0.75)(X90)
			JCudaTensor x48;
			JCudaTensor x49;
			x49 = x45;
			x48 = x50.forward(x49);

			// Dealloc(X90)
			JCudaTensor x51;
			x51 = x45;
			x51.free();

			// val X92 = Pooling(3,2,1,true)(X91)
			JCudaTensor x52;
			JCudaTensor x53;
			x53 = x48;
			x52 = x54.forward(x53);

			// Dealloc(X91)
			JCudaTensor x55;
			x55 = x48;
			x55.free();

			// val X93 = Convolv(1,0)(X92,cv11_W,cv11_B)
			JCudaTensor x56;
			JCudaTensor x57, x58, x59;
			x57 = x52;
			x58 = x60;
			x59 = x61;
			x56 = x62.forward(x57, x58, x59);

			// val X95 = Convolv(1,0)(X92,cv12_W,cv12_B)
			JCudaTensor x63;
			JCudaTensor x64, x65, x66;
			x64 = x52;
			x65 = x67;
			x66 = x68;
			x63 = x69.forward(x64, x65, x66);

			// val X103 = Pooling(3,1,1,true)(X92)
			JCudaTensor x70;
			JCudaTensor x71;
			x71 = x52;
			x70 = x72.forward(x71);

			// val X99 = Convolv(1,0)(X92,cv14_W,cv14_B)
			JCudaTensor x73;
			JCudaTensor x74, x75, x76;
			x74 = x52;
			x75 = x77;
			x76 = x78;
			x73 = x79.forward(x74, x75, x76);

			// Dealloc(X92)
			JCudaTensor x80;
			x80 = x52;
			x80.free();

			// val X104 = Convolv(1,0)(X103,cv16_W,cv16_B)
			JCudaTensor x81;
			JCudaTensor x82, x83, x84;
			x82 = x70;
			x83 = x85;
			x84 = x86;
			x81 = x87.forward(x82, x83, x84);

			// Dealloc(X103)
			JCudaTensor x88;
			x88 = x70;
			x88.free();

			// val X96 = ReLU()(X95)
			JCudaTensor x89;
			JCudaTensor x90;
			x90 = x63;
			x89 = x91.forward(x90);

			// val X100 = ReLU()(X99)
			JCudaTensor x92;
			JCudaTensor x93;
			x93 = x73;
			x92 = x94.forward(x93);

			// val X97 = Convolv(1,1)(X96,cv13_W,cv13_B)
			JCudaTensor x95;
			JCudaTensor x96, x97, x98;
			x96 = x89;
			x97 = x99;
			x98 = x100;
			x95 = x101.forward(x96, x97, x98);

			// Dealloc(X96)
			JCudaTensor x102;
			x102 = x89;
			x102.free();

			// val X101 = Convolv(1,2)(X100,cv15_W,cv15_B)
			JCudaTensor x103;
			JCudaTensor x104, x105, x106;
			x104 = x92;
			x105 = x107;
			x106 = x108;
			x103 = x109.forward(x104, x105, x106);

			// Dealloc(X100)
			JCudaTensor x110;
			x110 = x92;
			x110.free();

			// val X94 = ReLU()(X93)
			JCudaTensor x111;
			JCudaTensor x112;
			x112 = x56;
			x111 = x113.forward(x112);

			// val X98 = ReLU()(X97)
			JCudaTensor x114;
			JCudaTensor x115;
			x115 = x95;
			x114 = x116.forward(x115);

			// val X102 = ReLU()(X101)
			JCudaTensor x117;
			JCudaTensor x118;
			x118 = x103;
			x117 = x119.forward(x118);

			// val X105 = ReLU()(X104)
			JCudaTensor x120;
			JCudaTensor x121;
			x121 = x81;
			x120 = x119.forward(x121);

			// val X106 = Concat(X94,X98,X102,X105)
			JCudaTensor x122;
			JCudaTensor x124, x125, x126, x127;
			x124 = x111;
			x125 = x114;
			x126 = x117;
			x127 = x120;
			x122 = x123.forward(x124,x125,x126,x127);

			// Dealloc(X105)
			JCudaTensor x128;
			x128 = x120;
			x128.free();

			// Dealloc(X102)
			JCudaTensor x129;
			x129 = x117;
			x129.free();

			// Dealloc(X98)
			JCudaTensor x130;
			x130 = x114;
			x130.free();

			// Dealloc(X94)
			JCudaTensor x131;
			x131 = x111;
			x131.free();

			// val X117 = Pooling(3,1,1,true)(X106)
			JCudaTensor x132;
			JCudaTensor x133;
			x133 = x122;
			x132 = x134.forward(x133);

			// val X113 = Convolv(1,0)(X106,cv24_W,cv24_B)
			JCudaTensor x135;
			JCudaTensor x136, x137, x138;
			x136 = x122;
			x137 = x139;
			x138 = x140;
			x135 = x141.forward(x136, x137, x138);

			// val X107 = Convolv(1,0)(X106,cv21_W,cv21_B)
			JCudaTensor x142;
			JCudaTensor x143, x144, x145;
			x143 = x122;
			x144 = x146;
			x145 = x147;
			x142 = x148.forward(x143, x144, x145);

			// val X109 = Convolv(1,0)(X106,cv22_W,cv22_B)
			JCudaTensor x149;
			JCudaTensor x150, x151, x152;
			x150 = x122;
			x151 = x153;
			x152 = x154;
			x149 = x155.forward(x150, x151, x152);

			// Dealloc(X106)
			JCudaTensor x156;
			x156 = x122;
			x156.free();

			// val X110 = ReLU()(X109)
			JCudaTensor x157;
			JCudaTensor x158;
			x158 = x149;
			x157 = x91.forward(x158);

			// val X118 = Convolv(1,0)(X117,cv26_W,cv26_B)
			JCudaTensor x159;
			JCudaTensor x160, x161, x162;
			x160 = x132;
			x161 = x163;
			x162 = x164;
			x159 = x165.forward(x160, x161, x162);

			// Dealloc(X117)
			JCudaTensor x166;
			x166 = x132;
			x166.free();

			// val X114 = ReLU()(X113)
			JCudaTensor x167;
			JCudaTensor x168;
			x168 = x135;
			x167 = x94.forward(x168);

			// val X111 = Convolv(1,1)(X110,cv23_W,cv23_B)
			JCudaTensor x169;
			JCudaTensor x170, x171, x172;
			x170 = x157;
			x171 = x173;
			x172 = x174;
			x169 = x101.forward(x170, x171, x172);

			// Dealloc(X110)
			JCudaTensor x175;
			x175 = x157;
			x175.free();

			// val X115 = Convolv(1,2)(X114,cv25_W,cv25_B)
			JCudaTensor x176;
			JCudaTensor x177, x178, x179;
			x177 = x167;
			x178 = x180;
			x179 = x181;
			x176 = x109.forward(x177, x178, x179);

			// Dealloc(X114)
			JCudaTensor x182;
			x182 = x167;
			x182.free();

			// val X108 = ReLU()(X107)
			JCudaTensor x183;
			JCudaTensor x184;
			x184 = x142;
			x183 = x113.forward(x184);

			// val X112 = ReLU()(X111)
			JCudaTensor x185;
			JCudaTensor x186;
			x186 = x169;
			x185 = x116.forward(x186);

			// val X116 = ReLU()(X115)
			JCudaTensor x187;
			JCudaTensor x188;
			x188 = x176;
			x187 = x119.forward(x188);

			// val X119 = ReLU()(X118)
			JCudaTensor x189;
			JCudaTensor x190;
			x190 = x159;
			x189 = x119.forward(x190);

			// val X120 = Concat(X108,X112,X116,X119)
			JCudaTensor x191;
			JCudaTensor x192, x193, x194, x195;
			x192 = x183;
			x193 = x185;
			x194 = x187;
			x195 = x189;
			x191 = x123.forward(x192,x193,x194,x195);

			// Dealloc(X119)
			JCudaTensor x196;
			x196 = x189;
			x196.free();

			// Dealloc(X116)
			JCudaTensor x197;
			x197 = x187;
			x197.free();

			// Dealloc(X112)
			JCudaTensor x198;
			x198 = x185;
			x198.free();

			// Dealloc(X108)
			JCudaTensor x199;
			x199 = x183;
			x199.free();

			// val X121 = Pooling(3,2,1,true)(X120)
			JCudaTensor x200;
			JCudaTensor x201;
			x201 = x191;
			x200 = x202.forward(x201);

			// Dealloc(X120)
			JCudaTensor x203;
			x203 = x191;
			x203.free();

			// val X122 = Convolv(1,0)(X121,cv31_W,cv31_B)
			JCudaTensor x204;
			JCudaTensor x205, x206, x207;
			x205 = x200;
			x206 = x208;
			x207 = x209;
			x204 = x210.forward(x205, x206, x207);

			// val X124 = Convolv(1,0)(X121,cv32_W,cv32_B)
			JCudaTensor x211;
			JCudaTensor x212, x213, x214;
			x212 = x200;
			x213 = x215;
			x214 = x216;
			x211 = x217.forward(x212, x213, x214);

			// val X132 = Pooling(3,1,1,true)(X121)
			JCudaTensor x218;
			JCudaTensor x219;
			x219 = x200;
			x218 = x220.forward(x219);

			// val X128 = Convolv(1,0)(X121,cv34_W,cv34_B)
			JCudaTensor x221;
			JCudaTensor x222, x223, x224;
			x222 = x200;
			x223 = x225;
			x224 = x226;
			x221 = x227.forward(x222, x223, x224);

			// Dealloc(X121)
			JCudaTensor x228;
			x228 = x200;
			x228.free();

			// val X129 = ReLU()(X128)
			JCudaTensor x229;
			JCudaTensor x230;
			x230 = x221;
			x229 = x231.forward(x230);

			// val X125 = ReLU()(X124)
			JCudaTensor x232;
			JCudaTensor x233;
			x233 = x211;
			x232 = x234.forward(x233);

			// val X133 = Convolv(1,0)(X132,cv36_W,cv36_B)
			JCudaTensor x235;
			JCudaTensor x236, x237, x238;
			x236 = x218;
			x237 = x239;
			x238 = x240;
			x235 = x241.forward(x236, x237, x238);

			// Dealloc(X132)
			JCudaTensor x242;
			x242 = x218;
			x242.free();

			// val X130 = Convolv(1,2)(X129,cv35_W,cv35_B)
			JCudaTensor x243;
			JCudaTensor x244, x245, x246;
			x244 = x229;
			x245 = x247;
			x246 = x248;
			x243 = x249.forward(x244, x245, x246);

			// Dealloc(X129)
			JCudaTensor x250;
			x250 = x229;
			x250.free();

			// val X126 = Convolv(1,1)(X125,cv33_W,cv33_B)
			JCudaTensor x251;
			JCudaTensor x252, x253, x254;
			x252 = x232;
			x253 = x255;
			x254 = x256;
			x251 = x257.forward(x252, x253, x254);

			// Dealloc(X125)
			JCudaTensor x258;
			x258 = x232;
			x258.free();

			// val X123 = ReLU()(X122)
			JCudaTensor x259;
			JCudaTensor x260;
			x260 = x204;
			x259 = x261.forward(x260);

			// val X127 = ReLU()(X126)
			JCudaTensor x262;
			JCudaTensor x263;
			x263 = x251;
			x262 = x264.forward(x263);

			// val X131 = ReLU()(X130)
			JCudaTensor x265;
			JCudaTensor x266;
			x266 = x243;
			x265 = x267.forward(x266);

			// val X134 = ReLU()(X133)
			JCudaTensor x268;
			JCudaTensor x269;
			x269 = x235;
			x268 = x267.forward(x269);

			// val X135 = Concat(X123,X127,X131,X134)
			JCudaTensor x270;
			JCudaTensor x272, x273, x274, x275;
			x272 = x259;
			x273 = x262;
			x274 = x265;
			x275 = x268;
			x270 = x271.forward(x272,x273,x274,x275);

			// Dealloc(X134)
			JCudaTensor x276;
			x276 = x268;
			x276.free();

			// Dealloc(X131)
			JCudaTensor x277;
			x277 = x265;
			x277.free();

			// Dealloc(X127)
			JCudaTensor x278;
			x278 = x262;
			x278.free();

			// Dealloc(X123)
			JCudaTensor x279;
			x279 = x259;
			x279.free();

			// val X146 = Pooling(3,1,1,true)(X135)
			JCudaTensor x280;
			JCudaTensor x281;
			x281 = x270;
			x280 = x220.forward(x281);

			// val X136 = Convolv(1,0)(X135,cv41_W,cv41_B)
			JCudaTensor x282;
			JCudaTensor x283, x284, x285;
			x283 = x270;
			x284 = x286;
			x285 = x287;
			x282 = x210.forward(x283, x284, x285);

			// val X138 = Convolv(1,0)(X135,cv42_W,cv42_B)
			JCudaTensor x288;
			JCudaTensor x289, x290, x291;
			x289 = x270;
			x290 = x292;
			x291 = x293;
			x288 = x217.forward(x289, x290, x291);

			// val X142 = Convolv(1,0)(X135,cv44_W,cv44_B)
			JCudaTensor x294;
			JCudaTensor x295, x296, x297;
			x295 = x270;
			x296 = x298;
			x297 = x299;
			x294 = x227.forward(x295, x296, x297);

			// Dealloc(X135)
			JCudaTensor x300;
			x300 = x270;
			x300.free();

			// val X143 = ReLU()(X142)
			JCudaTensor x301;
			JCudaTensor x302;
			x302 = x294;
			x301 = x231.forward(x302);

			// val X139 = ReLU()(X138)
			JCudaTensor x303;
			JCudaTensor x304;
			x304 = x288;
			x303 = x234.forward(x304);

			// val X147 = Convolv(1,0)(X146,cv46_W,cv46_B)
			JCudaTensor x305;
			JCudaTensor x306, x307, x308;
			x306 = x280;
			x307 = x309;
			x308 = x310;
			x305 = x241.forward(x306, x307, x308);

			// Dealloc(X146)
			JCudaTensor x311;
			x311 = x280;
			x311.free();

			// val X144 = Convolv(1,2)(X143,cv45_W,cv45_B)
			JCudaTensor x312;
			JCudaTensor x313, x314, x315;
			x313 = x301;
			x314 = x316;
			x315 = x317;
			x312 = x249.forward(x313, x314, x315);

			// Dealloc(X143)
			JCudaTensor x318;
			x318 = x301;
			x318.free();

			// val X140 = Convolv(1,1)(X139,cv43_W,cv43_B)
			JCudaTensor x319;
			JCudaTensor x320, x321, x322;
			x320 = x303;
			x321 = x323;
			x322 = x324;
			x319 = x257.forward(x320, x321, x322);

			// Dealloc(X139)
			JCudaTensor x325;
			x325 = x303;
			x325.free();

			// val X137 = ReLU()(X136)
			JCudaTensor x326;
			JCudaTensor x327;
			x327 = x282;
			x326 = x261.forward(x327);

			// val X141 = ReLU()(X140)
			JCudaTensor x328;
			JCudaTensor x329;
			x329 = x319;
			x328 = x264.forward(x329);

			// val X145 = ReLU()(X144)
			JCudaTensor x330;
			JCudaTensor x331;
			x331 = x312;
			x330 = x267.forward(x331);

			// val X148 = ReLU()(X147)
			JCudaTensor x332;
			JCudaTensor x333;
			x333 = x305;
			x332 = x267.forward(x333);

			// val X149 = Concat(X137,X141,X145,X148)
			JCudaTensor x334;
			JCudaTensor x335, x336, x337, x338;
			x335 = x326;
			x336 = x328;
			x337 = x330;
			x338 = x332;
			x334 = x271.forward(x335,x336,x337,x338);

			// Dealloc(X148)
			JCudaTensor x339;
			x339 = x332;
			x339.free();

			// Dealloc(X145)
			JCudaTensor x340;
			x340 = x330;
			x340.free();

			// Dealloc(X141)
			JCudaTensor x341;
			x341 = x328;
			x341.free();

			// Dealloc(X137)
			JCudaTensor x342;
			x342 = x326;
			x342.free();

			// val X156 = Convolv(1,0)(X149,cv54_W,cv54_B)
			JCudaTensor x343;
			JCudaTensor x344, x345, x346;
			x344 = x334;
			x345 = x347;
			x346 = x348;
			x343 = x227.forward(x344, x345, x346);

			// val X150 = Convolv(1,0)(X149,cv51_W,cv51_B)
			JCudaTensor x349;
			JCudaTensor x350, x351, x352;
			x350 = x334;
			x351 = x353;
			x352 = x354;
			x349 = x210.forward(x350, x351, x352);

			// val X152 = Convolv(1,0)(X149,cv52_W,cv52_B)
			JCudaTensor x355;
			JCudaTensor x356, x357, x358;
			x356 = x334;
			x357 = x359;
			x358 = x360;
			x355 = x217.forward(x356, x357, x358);

			// val X160 = Pooling(3,1,1,true)(X149)
			JCudaTensor x361;
			JCudaTensor x362;
			x362 = x334;
			x361 = x220.forward(x362);

			// Dealloc(X149)
			JCudaTensor x363;
			x363 = x334;
			x363.free();

			// val X157 = ReLU()(X156)
			JCudaTensor x364;
			JCudaTensor x365;
			x365 = x343;
			x364 = x231.forward(x365);

			// val X161 = Convolv(1,0)(X160,cv56_W,cv56_B)
			JCudaTensor x366;
			JCudaTensor x367, x368, x369;
			x367 = x361;
			x368 = x370;
			x369 = x371;
			x366 = x241.forward(x367, x368, x369);

			// Dealloc(X160)
			JCudaTensor x372;
			x372 = x361;
			x372.free();

			// val X153 = ReLU()(X152)
			JCudaTensor x373;
			JCudaTensor x374;
			x374 = x355;
			x373 = x234.forward(x374);

			// val X154 = Convolv(1,1)(X153,cv53_W,cv53_B)
			JCudaTensor x375;
			JCudaTensor x376, x377, x378;
			x376 = x373;
			x377 = x379;
			x378 = x380;
			x375 = x257.forward(x376, x377, x378);

			// Dealloc(X153)
			JCudaTensor x381;
			x381 = x373;
			x381.free();

			// val X158 = Convolv(1,2)(X157,cv55_W,cv55_B)
			JCudaTensor x382;
			JCudaTensor x383, x384, x385;
			x383 = x364;
			x384 = x386;
			x385 = x387;
			x382 = x249.forward(x383, x384, x385);

			// Dealloc(X157)
			JCudaTensor x388;
			x388 = x364;
			x388.free();

			// val X151 = ReLU()(X150)
			JCudaTensor x389;
			JCudaTensor x390;
			x390 = x349;
			x389 = x261.forward(x390);

			// val X155 = ReLU()(X154)
			JCudaTensor x391;
			JCudaTensor x392;
			x392 = x375;
			x391 = x264.forward(x392);

			// val X159 = ReLU()(X158)
			JCudaTensor x393;
			JCudaTensor x394;
			x394 = x382;
			x393 = x267.forward(x394);

			// val X162 = ReLU()(X161)
			JCudaTensor x395;
			JCudaTensor x396;
			x396 = x366;
			x395 = x267.forward(x396);

			// val X163 = Concat(X151,X155,X159,X162)
			JCudaTensor x397;
			JCudaTensor x398, x399, x400, x401;
			x398 = x389;
			x399 = x391;
			x400 = x393;
			x401 = x395;
			x397 = x271.forward(x398,x399,x400,x401);

			// Dealloc(X162)
			JCudaTensor x402;
			x402 = x395;
			x402.free();

			// Dealloc(X159)
			JCudaTensor x403;
			x403 = x393;
			x403.free();

			// Dealloc(X155)
			JCudaTensor x404;
			x404 = x391;
			x404.free();

			// Dealloc(X151)
			JCudaTensor x405;
			x405 = x389;
			x405.free();

			// val X174 = Pooling(3,1,1,true)(X163)
			JCudaTensor x406;
			JCudaTensor x407;
			x407 = x397;
			x406 = x220.forward(x407);

			// val X164 = Convolv(1,0)(X163,cv61_W,cv61_B)
			JCudaTensor x408;
			JCudaTensor x409, x410, x411;
			x409 = x397;
			x410 = x412;
			x411 = x413;
			x408 = x210.forward(x409, x410, x411);

			// val X166 = Convolv(1,0)(X163,cv62_W,cv62_B)
			JCudaTensor x414;
			JCudaTensor x415, x416, x417;
			x415 = x397;
			x416 = x418;
			x417 = x419;
			x414 = x217.forward(x415, x416, x417);

			// val X170 = Convolv(1,0)(X163,cv64_W,cv64_B)
			JCudaTensor x420;
			JCudaTensor x421, x422, x423;
			x421 = x397;
			x422 = x424;
			x423 = x425;
			x420 = x227.forward(x421, x422, x423);

			// Dealloc(X163)
			JCudaTensor x426;
			x426 = x397;
			x426.free();

			// val X175 = Convolv(1,0)(X174,cv66_W,cv66_B)
			JCudaTensor x427;
			JCudaTensor x428, x429, x430;
			x428 = x406;
			x429 = x431;
			x430 = x432;
			x427 = x241.forward(x428, x429, x430);

			// Dealloc(X174)
			JCudaTensor x433;
			x433 = x406;
			x433.free();

			// val X171 = ReLU()(X170)
			JCudaTensor x434;
			JCudaTensor x435;
			x435 = x420;
			x434 = x231.forward(x435);

			// val X167 = ReLU()(X166)
			JCudaTensor x436;
			JCudaTensor x437;
			x437 = x414;
			x436 = x234.forward(x437);

			// val X168 = Convolv(1,1)(X167,cv63_W,cv63_B)
			JCudaTensor x438;
			JCudaTensor x439, x440, x441;
			x439 = x436;
			x440 = x442;
			x441 = x443;
			x438 = x257.forward(x439, x440, x441);

			// Dealloc(X167)
			JCudaTensor x444;
			x444 = x436;
			x444.free();

			// val X172 = Convolv(1,2)(X171,cv65_W,cv65_B)
			JCudaTensor x445;
			JCudaTensor x446, x447, x448;
			x446 = x434;
			x447 = x449;
			x448 = x450;
			x445 = x249.forward(x446, x447, x448);

			// Dealloc(X171)
			JCudaTensor x451;
			x451 = x434;
			x451.free();

			// val X165 = ReLU()(X164)
			JCudaTensor x452;
			JCudaTensor x453;
			x453 = x408;
			x452 = x261.forward(x453);

			// val X169 = ReLU()(X168)
			JCudaTensor x454;
			JCudaTensor x455;
			x455 = x438;
			x454 = x264.forward(x455);

			// val X173 = ReLU()(X172)
			JCudaTensor x456;
			JCudaTensor x457;
			x457 = x445;
			x456 = x267.forward(x457);

			// val X176 = ReLU()(X175)
			JCudaTensor x458;
			JCudaTensor x459;
			x459 = x427;
			x458 = x267.forward(x459);

			// val X177 = Concat(X165,X169,X173,X176)
			JCudaTensor x460;
			JCudaTensor x461, x462, x463, x464;
			x461 = x452;
			x462 = x454;
			x463 = x456;
			x464 = x458;
			x460 = x271.forward(x461,x462,x463,x464);

			// Dealloc(X176)
			JCudaTensor x465;
			x465 = x458;
			x465.free();

			// Dealloc(X173)
			JCudaTensor x466;
			x466 = x456;
			x466.free();

			// Dealloc(X169)
			JCudaTensor x467;
			x467 = x454;
			x467.free();

			// Dealloc(X165)
			JCudaTensor x468;
			x468 = x452;
			x468.free();

			// val X180 = Convolv(1,0)(X177,cv72_W,cv72_B)
			JCudaTensor x469;
			JCudaTensor x470, x471, x472;
			x470 = x460;
			x471 = x473;
			x472 = x474;
			x469 = x217.forward(x470, x471, x472);

			// val X184 = Convolv(1,0)(X177,cv74_W,cv74_B)
			JCudaTensor x475;
			JCudaTensor x476, x477, x478;
			x476 = x460;
			x477 = x479;
			x478 = x480;
			x475 = x227.forward(x476, x477, x478);

			// val X188 = Pooling(3,1,1,true)(X177)
			JCudaTensor x481;
			JCudaTensor x482;
			x482 = x460;
			x481 = x220.forward(x482);

			// val X178 = Convolv(1,0)(X177,cv71_W,cv71_B)
			JCudaTensor x483;
			JCudaTensor x484, x485, x486;
			x484 = x460;
			x485 = x487;
			x486 = x488;
			x483 = x210.forward(x484, x485, x486);

			// Dealloc(X177)
			JCudaTensor x489;
			x489 = x460;
			x489.free();

			// val X189 = Convolv(1,0)(X188,cv76_W,cv76_B)
			JCudaTensor x490;
			JCudaTensor x491, x492, x493;
			x491 = x481;
			x492 = x494;
			x493 = x495;
			x490 = x241.forward(x491, x492, x493);

			// Dealloc(X188)
			JCudaTensor x496;
			x496 = x481;
			x496.free();

			// val X185 = ReLU()(X184)
			JCudaTensor x497;
			JCudaTensor x498;
			x498 = x475;
			x497 = x231.forward(x498);

			// val X181 = ReLU()(X180)
			JCudaTensor x499;
			JCudaTensor x500;
			x500 = x469;
			x499 = x234.forward(x500);

			// val X186 = Convolv(1,2)(X185,cv75_W,cv75_B)
			JCudaTensor x501;
			JCudaTensor x502, x503, x504;
			x502 = x497;
			x503 = x505;
			x504 = x506;
			x501 = x249.forward(x502, x503, x504);

			// Dealloc(X185)
			JCudaTensor x507;
			x507 = x497;
			x507.free();

			// val X182 = Convolv(1,1)(X181,cv73_W,cv73_B)
			JCudaTensor x508;
			JCudaTensor x509, x510, x511;
			x509 = x499;
			x510 = x512;
			x511 = x513;
			x508 = x257.forward(x509, x510, x511);

			// Dealloc(X181)
			JCudaTensor x514;
			x514 = x499;
			x514.free();

			// val X179 = ReLU()(X178)
			JCudaTensor x515;
			JCudaTensor x516;
			x516 = x483;
			x515 = x261.forward(x516);

			// val X183 = ReLU()(X182)
			JCudaTensor x517;
			JCudaTensor x518;
			x518 = x508;
			x517 = x264.forward(x518);

			// val X187 = ReLU()(X186)
			JCudaTensor x519;
			JCudaTensor x520;
			x520 = x501;
			x519 = x267.forward(x520);

			// val X190 = ReLU()(X189)
			JCudaTensor x521;
			JCudaTensor x522;
			x522 = x490;
			x521 = x267.forward(x522);

			// val X191 = Concat(X179,X183,X187,X190)
			JCudaTensor x523;
			JCudaTensor x524, x525, x526, x527;
			x524 = x515;
			x525 = x517;
			x526 = x519;
			x527 = x521;
			x523 = x271.forward(x524,x525,x526,x527);

			// Dealloc(X190)
			JCudaTensor x528;
			x528 = x521;
			x528.free();

			// Dealloc(X187)
			JCudaTensor x529;
			x529 = x519;
			x529.free();

			// Dealloc(X183)
			JCudaTensor x530;
			x530 = x517;
			x530.free();

			// Dealloc(X179)
			JCudaTensor x531;
			x531 = x515;
			x531.free();

			// val X192 = Pooling(3,2,1,true)(X191)
			JCudaTensor x532;
			JCudaTensor x533;
			x533 = x523;
			x532 = x534.forward(x533);

			// Dealloc(X191)
			JCudaTensor x535;
			x535 = x523;
			x535.free();

			// val X203 = Pooling(3,1,1,true)(X192)
			JCudaTensor x536;
			JCudaTensor x537;
			x537 = x532;
			x536 = x538.forward(x537);

			// val X199 = Convolv(1,0)(X192,cv84_W,cv84_B)
			JCudaTensor x539;
			JCudaTensor x540, x541, x542;
			x540 = x532;
			x541 = x543;
			x542 = x544;
			x539 = x545.forward(x540, x541, x542);

			// val X195 = Convolv(1,0)(X192,cv82_W,cv82_B)
			JCudaTensor x546;
			JCudaTensor x547, x548, x549;
			x547 = x532;
			x548 = x550;
			x549 = x551;
			x546 = x552.forward(x547, x548, x549);

			// val X193 = Convolv(1,0)(X192,cv81_W,cv81_B)
			JCudaTensor x553;
			JCudaTensor x554, x555, x556;
			x554 = x532;
			x555 = x557;
			x556 = x558;
			x553 = x559.forward(x554, x555, x556);

			// Dealloc(X192)
			JCudaTensor x560;
			x560 = x532;
			x560.free();

			// val X204 = Convolv(1,0)(X203,cv86_W,cv86_B)
			JCudaTensor x561;
			JCudaTensor x562, x563, x564;
			x562 = x536;
			x563 = x565;
			x564 = x566;
			x561 = x567.forward(x562, x563, x564);

			// Dealloc(X203)
			JCudaTensor x568;
			x568 = x536;
			x568.free();

			// val X196 = ReLU()(X195)
			JCudaTensor x569;
			JCudaTensor x570;
			x570 = x546;
			x569 = x571.forward(x570);

			// val X200 = ReLU()(X199)
			JCudaTensor x572;
			JCudaTensor x573;
			x573 = x539;
			x572 = x574.forward(x573);

			// val X197 = Convolv(1,1)(X196,cv83_W,cv83_B)
			JCudaTensor x575;
			JCudaTensor x576, x577, x578;
			x576 = x569;
			x577 = x579;
			x578 = x580;
			x575 = x581.forward(x576, x577, x578);

			// Dealloc(X196)
			JCudaTensor x582;
			x582 = x569;
			x582.free();

			// val X201 = Convolv(1,2)(X200,cv85_W,cv85_B)
			JCudaTensor x583;
			JCudaTensor x584, x585, x586;
			x584 = x572;
			x585 = x587;
			x586 = x588;
			x583 = x589.forward(x584, x585, x586);

			// Dealloc(X200)
			JCudaTensor x590;
			x590 = x572;
			x590.free();

			// val X194 = ReLU()(X193)
			JCudaTensor x591;
			JCudaTensor x592;
			x592 = x553;
			x591 = x593.forward(x592);

			// val X198 = ReLU()(X197)
			JCudaTensor x594;
			JCudaTensor x595;
			x595 = x575;
			x594 = x596.forward(x595);

			// val X202 = ReLU()(X201)
			JCudaTensor x597;
			JCudaTensor x598;
			x598 = x583;
			x597 = x599.forward(x598);

			// val X205 = ReLU()(X204)
			JCudaTensor x600;
			JCudaTensor x601;
			x601 = x561;
			x600 = x599.forward(x601);

			// val X206 = Concat(X194,X198,X202,X205)
			JCudaTensor x602;
			JCudaTensor x604, x605, x606, x607;
			x604 = x591;
			x605 = x594;
			x606 = x597;
			x607 = x600;
			x602 = x603.forward(x604,x605,x606,x607);

			// Dealloc(X205)
			JCudaTensor x608;
			x608 = x600;
			x608.free();

			// Dealloc(X202)
			JCudaTensor x609;
			x609 = x597;
			x609.free();

			// Dealloc(X198)
			JCudaTensor x610;
			x610 = x594;
			x610.free();

			// Dealloc(X194)
			JCudaTensor x611;
			x611 = x591;
			x611.free();

			// val X213 = Convolv(1,0)(X206,cv94_W,cv94_B)
			JCudaTensor x612;
			JCudaTensor x613, x614, x615;
			x613 = x602;
			x614 = x616;
			x615 = x617;
			x612 = x545.forward(x613, x614, x615);

			// val X207 = Convolv(1,0)(X206,cv91_W,cv91_B)
			JCudaTensor x618;
			JCudaTensor x619, x620, x621;
			x619 = x602;
			x620 = x622;
			x621 = x623;
			x618 = x559.forward(x619, x620, x621);

			// val X217 = Pooling(3,1,1,true)(X206)
			JCudaTensor x624;
			JCudaTensor x625;
			x625 = x602;
			x624 = x538.forward(x625);

			// val X209 = Convolv(1,0)(X206,cv92_W,cv92_B)
			JCudaTensor x626;
			JCudaTensor x627, x628, x629;
			x627 = x602;
			x628 = x630;
			x629 = x631;
			x626 = x552.forward(x627, x628, x629);

			// Dealloc(X206)
			JCudaTensor x632;
			x632 = x602;
			x632.free();

			// val X210 = ReLU()(X209)
			JCudaTensor x633;
			JCudaTensor x634;
			x634 = x626;
			x633 = x571.forward(x634);

			// val X214 = ReLU()(X213)
			JCudaTensor x635;
			JCudaTensor x636;
			x636 = x612;
			x635 = x574.forward(x636);

			// val X218 = Convolv(1,0)(X217,cv96_W,cv96_B)
			JCudaTensor x637;
			JCudaTensor x638, x639, x640;
			x638 = x624;
			x639 = x641;
			x640 = x642;
			x637 = x567.forward(x638, x639, x640);

			// Dealloc(X217)
			JCudaTensor x643;
			x643 = x624;
			x643.free();

			// val X211 = Convolv(1,1)(X210,cv93_W,cv93_B)
			JCudaTensor x644;
			JCudaTensor x645, x646, x647;
			x645 = x633;
			x646 = x648;
			x647 = x649;
			x644 = x581.forward(x645, x646, x647);

			// Dealloc(X210)
			JCudaTensor x650;
			x650 = x633;
			x650.free();

			// val X215 = Convolv(1,2)(X214,cv95_W,cv95_B)
			JCudaTensor x651;
			JCudaTensor x652, x653, x654;
			x652 = x635;
			x653 = x655;
			x654 = x656;
			x651 = x589.forward(x652, x653, x654);

			// Dealloc(X214)
			JCudaTensor x657;
			x657 = x635;
			x657.free();

			// val X208 = ReLU()(X207)
			JCudaTensor x658;
			JCudaTensor x659;
			x659 = x618;
			x658 = x593.forward(x659);

			// val X212 = ReLU()(X211)
			JCudaTensor x660;
			JCudaTensor x661;
			x661 = x644;
			x660 = x596.forward(x661);

			// val X216 = ReLU()(X215)
			JCudaTensor x662;
			JCudaTensor x663;
			x663 = x651;
			x662 = x599.forward(x663);

			// val X219 = ReLU()(X218)
			JCudaTensor x664;
			JCudaTensor x665;
			x665 = x637;
			x664 = x599.forward(x665);

			// val X220 = Concat(X208,X212,X216,X219)
			JCudaTensor x666;
			JCudaTensor x667, x668, x669, x670;
			x667 = x658;
			x668 = x660;
			x669 = x662;
			x670 = x664;
			x666 = x603.forward(x667,x668,x669,x670);

			// Dealloc(X219)
			JCudaTensor x671;
			x671 = x664;
			x671.free();

			// Dealloc(X216)
			JCudaTensor x672;
			x672 = x662;
			x672.free();

			// Dealloc(X212)
			JCudaTensor x673;
			x673 = x660;
			x673.free();

			// Dealloc(X208)
			JCudaTensor x674;
			x674 = x658;
			x674.free();

			// val X221 = Pooling(7,1,0,false)(X220)
			JCudaTensor x675;
			JCudaTensor x676;
			x676 = x666;
			x675 = x677.forward(x676);

			// Dealloc(X220)
			JCudaTensor x678;
			x678 = x666;
			x678.free();

			// val X222 = Dropout(0.4)(X221)
			JCudaTensor x679;
			JCudaTensor x680;
			x680 = x675;
			x679 = x681.forward(x680);

			// Dealloc(X221)
			JCudaTensor x682;
			x682 = x675;
			x682.free();

			// val X223 = (X222[1><3])(i | @) * (fc_W)(j | @)
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

			// Dealloc(X222)
			JCudaTensor x690;
			x690 = x679;
			x690.free();

			// val X225 = (X223 + (i) => fc_B)
			JCudaTensor x691;
			JCudaTensor x692, x693;
			x692 = x683;
			x693 = x694;
			x691 = x693.copy(128, x692);

			// Prediction(X225)
			JCudaTensor x695;
			x695 = x691;
			System.out.println(x3 + " inference " + java.util.Arrays.toString(x695.asJTensor().prediction()));

			// Dealloc(X225)
			JCudaTensor x696;
			x696 = x691;
			x696.free();

		}

	}

}