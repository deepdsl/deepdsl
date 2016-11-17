package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;

// This file is for inference only, which needs trained parameters.
public class Alexnet_infer {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/alexnet";
	// platform
	static LmdbUtils.OS platform = LmdbUtils.OS.WINDOWS;
	// test_data_path
	static String test_data_path = "dataset/imagenet/ilsvrc12_val_lmdb";
	// test_itr
	static int test_itr = 10;
	// test_size
	static int test_size = 10000;

	// (Convolv(1,1),List(List(128, 256, 13, 13), List(384, 256, 3, 3), List(384)))
	static JCudnnConvolution x51 = new JCudnnConvolution(new int[]{128,256,13,13},new int[]{384,256,3,3},new int[]{384}, 1, 1);
	// (Convolv(1,1),List(List(128, 384, 13, 13), List(256, 384, 3, 3), List(256)))
	static JCudnnConvolution x72 = new JCudnnConvolution(new int[]{128,384,13,13},new int[]{256,384,3,3},new int[]{256}, 1, 1);
	// (Convolv(1,1),List(List(128, 384, 13, 13), List(384, 384, 3, 3), List(384)))
	static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,384,13,13},new int[]{384,384,3,3},new int[]{384}, 1, 1);
	// (Convolv(1,2),List(List(128, 96, 27, 27), List(256, 96, 5, 5), List(256)))
	static JCudnnConvolution x32 = new JCudnnConvolution(new int[]{128,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
	// (Convolv(4,2),List(List(128, 3, 224, 224), List(96, 3, 11, 11), List(96)))
	static JCudnnConvolution x13 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 2);
	// (Dropout(0.5),List(List(128, 4096)))
	static JCudnnDropout x98 = new JCudnnDropout(new int[]{128,4096}, 0.5f);
	// (LRN(5,1.0E-4,0.75),List(List(128, 256, 27, 27)))
	static JCudnnLRN x39 = new JCudnnLRN(new int[]{128,256,27,27}, 5, 1.0E-4, 0.75);
	// (LRN(5,1.0E-4,0.75),List(List(128, 96, 55, 55)))
	static JCudnnLRN x20 = new JCudnnLRN(new int[]{128,96,55,55}, 5, 1.0E-4, 0.75);
	// (Lmdb(1000000,10000,Win32,1000),false)
	static LmdbFactory x1 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, platform, 1000, true);
	// (Pooling(3,2,0,true),List(List(128, 256, 13, 13)))
	static JCudnnPooling x79 = new JCudnnPooling(new int[]{128,256,13,13}, 3, 2, 0, PoolingType.MAX);
	// (Pooling(3,2,0,true),List(List(128, 256, 27, 27)))
	static JCudnnPooling x43 = new JCudnnPooling(new int[]{128,256,27,27}, 3, 2, 0, PoolingType.MAX);
	// (Pooling(3,2,0,true),List(List(128, 96, 55, 55)))
	static JCudnnPooling x24 = new JCudnnPooling(new int[]{128,96,55,55}, 3, 2, 0, PoolingType.MAX);
	// (ReLU(),List(List(128, 256, 13, 13)))
	static JCudnnActivation x76 = new JCudnnActivation(new int[]{128,256,13,13}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 256, 27, 27)))
	static JCudnnActivation x36 = new JCudnnActivation(new int[]{128,256,27,27}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 384, 13, 13)))
	static JCudnnActivation x55 = new JCudnnActivation(new int[]{128,384,13,13}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 4096)))
	static JCudnnActivation x95 = new JCudnnActivation(new int[]{128,4096}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 55, 55)))
	static JCudnnActivation x17 = new JCudnnActivation(new int[]{128,96,55,55}, ActivationMode.RELU);
	// X
	static JTensorFloat x2;
	// cv1_B
	static JCudaTensor x12 = JTensor.constFloat(0.0f, 96).load(network_dir + "/cv1_B").asJCudaTensor();
	// cv1_W
	static JCudaTensor x11 = JTensor.gaussianFloat(0.01f, 96, 3, 11, 11).load(network_dir + "/cv1_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x31 = JTensor.constFloat(0.1f, 256).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x30 = JTensor.gaussianFloat(0.01f, 256, 96, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
	// cv3_B
	static JCudaTensor x50 = JTensor.constFloat(0.0f, 384).load(network_dir + "/cv3_B").asJCudaTensor();
	// cv3_W
	static JCudaTensor x49 = JTensor.gaussianFloat(0.01f, 384, 256, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
	// cv4_B
	static JCudaTensor x61 = JTensor.constFloat(0.1f, 384).load(network_dir + "/cv4_B").asJCudaTensor();
	// cv4_W
	static JCudaTensor x60 = JTensor.gaussianFloat(0.01f, 384, 384, 3, 3).load(network_dir + "/cv4_W").asJCudaTensor();
	// cv5_B
	static JCudaTensor x71 = JTensor.constFloat(0.1f, 256).load(network_dir + "/cv5_B").asJCudaTensor();
	// cv5_W
	static JCudaTensor x70 = JTensor.gaussianFloat(0.01f, 256, 384, 3, 3).load(network_dir + "/cv5_W").asJCudaTensor();
	// fc6_B
	static JCudaTensor x92 = JTensor.constFloat(0.1f, 4096).load(network_dir + "/fc6_B").asJCudaTensor();
	// fc6_W
	static JCudaTensor x87 = JTensor.gaussianFloat(0.005f, 4096, 9216).load(network_dir + "/fc6_W").asJCudaTensor();
	// fc7_B
	static JCudaTensor x110 = JTensor.constFloat(0.1f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
	// fc7_W
	static JCudaTensor x105 = JTensor.gaussianFloat(0.005f, 4096, 4096).load(network_dir + "/fc7_W").asJCudaTensor();
	// fc8_B
	static JCudaTensor x126 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
	// fc8_W
	static JCudaTensor x121 = JTensor.gaussianFloat(0.01f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

	public static void main(String[] args){
		test();
		x61.free();
		x126.free();
		x49.free();
		x71.free();
		x50.free();
		x70.free();
		x110.free();
		x12.free();
		x121.free();
		x60.free();
		x92.free();
		x30.free();
		x105.free();
		x87.free();
		x11.free();
		x31.free();
		x24.free();
		x39.free();
		x51.free();
		x98.free();
		x36.free();
		x55.free();
		x32.free();
		x79.free();
		x20.free();
		x17.free();
		x13.free();
		x62.free();
		x72.free();
		x95.free();
		x76.free();
		x43.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void test() {
		for(int x3=0; x3<test_itr; x3++) {
			JTensorFloatTuple x4 =  x1.nextFloat();
			x2 = x4.image;

			// val X2342 = Cuda(X)
			JCudaTensor x5;
			JTensorFloat x6;
			x6 = x2;
			x5 = x6.asJCudaTensor();

			// val X2343 = Convolv(4,2)(X2342,cv1_W,cv1_B)
			JCudaTensor x7;
			JCudaTensor x8, x9, x10;
			x8 = x5;
			x9 = x11;
			x10 = x12;
			x7 = x13.forward(x8, x9, x10);

			// Dealloc(X2342)
			JCudaTensor x14;
			x14 = x5;
			x14.free();

			// val X2344 = ReLU()(X2343)
			JCudaTensor x15;
			JCudaTensor x16;
			x16 = x7;
			x15 = x17.forward(x16);

			// val X2345 = LRN(5,1.0E-4,0.75)(X2344)
			JCudaTensor x18;
			JCudaTensor x19;
			x19 = x15;
			x18 = x20.forward(x19);

			// Dealloc(X2344)
			JCudaTensor x21;
			x21 = x15;
			x21.free();

			// val X2346 = Pooling(3,2,0,true)(X2345)
			JCudaTensor x22;
			JCudaTensor x23;
			x23 = x18;
			x22 = x24.forward(x23);

			// Dealloc(X2345)
			JCudaTensor x25;
			x25 = x18;
			x25.free();

			// val X2347 = Convolv(1,2)(X2346,cv2_W,cv2_B)
			JCudaTensor x26;
			JCudaTensor x27, x28, x29;
			x27 = x22;
			x28 = x30;
			x29 = x31;
			x26 = x32.forward(x27, x28, x29);

			// Dealloc(X2346)
			JCudaTensor x33;
			x33 = x22;
			x33.free();

			// val X2348 = ReLU()(X2347)
			JCudaTensor x34;
			JCudaTensor x35;
			x35 = x26;
			x34 = x36.forward(x35);

			// val X2349 = LRN(5,1.0E-4,0.75)(X2348)
			JCudaTensor x37;
			JCudaTensor x38;
			x38 = x34;
			x37 = x39.forward(x38);

			// Dealloc(X2348)
			JCudaTensor x40;
			x40 = x34;
			x40.free();

			// val X2350 = Pooling(3,2,0,true)(X2349)
			JCudaTensor x41;
			JCudaTensor x42;
			x42 = x37;
			x41 = x43.forward(x42);

			// Dealloc(X2349)
			JCudaTensor x44;
			x44 = x37;
			x44.free();

			// val X2351 = Convolv(1,1)(X2350,cv3_W,cv3_B)
			JCudaTensor x45;
			JCudaTensor x46, x47, x48;
			x46 = x41;
			x47 = x49;
			x48 = x50;
			x45 = x51.forward(x46, x47, x48);

			// Dealloc(X2350)
			JCudaTensor x52;
			x52 = x41;
			x52.free();

			// val X2352 = ReLU()(X2351)
			JCudaTensor x53;
			JCudaTensor x54;
			x54 = x45;
			x53 = x55.forward(x54);

			// val X2353 = Convolv(1,1)(X2352,cv4_W,cv4_B)
			JCudaTensor x56;
			JCudaTensor x57, x58, x59;
			x57 = x53;
			x58 = x60;
			x59 = x61;
			x56 = x62.forward(x57, x58, x59);

			// Dealloc(X2352)
			JCudaTensor x63;
			x63 = x53;
			x63.free();

			// val X2354 = ReLU()(X2353)
			JCudaTensor x64;
			JCudaTensor x65;
			x65 = x56;
			x64 = x55.forward(x65);

			// val X2355 = Convolv(1,1)(X2354,cv5_W,cv5_B)
			JCudaTensor x66;
			JCudaTensor x67, x68, x69;
			x67 = x64;
			x68 = x70;
			x69 = x71;
			x66 = x72.forward(x67, x68, x69);

			// Dealloc(X2354)
			JCudaTensor x73;
			x73 = x64;
			x73.free();

			// val X2356 = ReLU()(X2355)
			JCudaTensor x74;
			JCudaTensor x75;
			x75 = x66;
			x74 = x76.forward(x75);

			// val X2357 = Pooling(3,2,0,true)(X2356)
			JCudaTensor x77;
			JCudaTensor x78;
			x78 = x74;
			x77 = x79.forward(x78);

			// Dealloc(X2356)
			JCudaTensor x80;
			x80 = x74;
			x80.free();

			// val X2358 = (X2357[1><3])(i | @) * (fc6_W)(j | @)
			JCudaTensor x81;
			JCudaMatrix x82;
			JCudaMatrix x83;
			JCudaTensor x84;
			JCudaTensor x85;
			x85 = x77;
			x84 = x85.flatten(1, new int[]{256, 6, 6});
			x82 = x84.asMatrix(1, true);
			JCudaTensor x86;
			x86 = x87;
			x83 = x86.asMatrix(1, true);
			x81 = x82.times(x83);

			// Dealloc(X2357)
			JCudaTensor x88;
			x88 = x77;
			x88.free();

			// val X2360 = (X2358 + (i) => fc6_B)
			JCudaTensor x89;
			JCudaTensor x90, x91;
			x90 = x81;
			x91 = x92;
			x89 = x91.copy(128, x90);

			// val X2361 = ReLU()(X2360)
			JCudaTensor x93;
			JCudaTensor x94;
			x94 = x89;
			x93 = x95.forward(x94);

			// val X2362 = Dropout(0.5)(X2361)
			JCudaTensor x96;
			JCudaTensor x97;
			x97 = x93;
			x96 = x98.forward(x97);

			// Dealloc(X2361)
			JCudaTensor x99;
			x99 = x93;
			x99.free();

			// val X2363 = (X2362)(i | @) * (fc7_W)(j | @)
			JCudaTensor x100;
			JCudaMatrix x101;
			JCudaMatrix x102;
			JCudaTensor x103;
			x103 = x96;
			x101 = x103.asMatrix(1, true);
			JCudaTensor x104;
			x104 = x105;
			x102 = x104.asMatrix(1, true);
			x100 = x101.times(x102);

			// Dealloc(X2362)
			JCudaTensor x106;
			x106 = x96;
			x106.free();

			// val X2365 = (X2363 + (i) => fc7_B)
			JCudaTensor x107;
			JCudaTensor x108, x109;
			x108 = x100;
			x109 = x110;
			x107 = x109.copy(128, x108);

			// val X2366 = ReLU()(X2365)
			JCudaTensor x111;
			JCudaTensor x112;
			x112 = x107;
			x111 = x95.forward(x112);

			// val X2367 = Dropout(0.5)(X2366)
			JCudaTensor x113;
			JCudaTensor x114;
			x114 = x111;
			x113 = x98.forward(x114);

			// Dealloc(X2366)
			JCudaTensor x115;
			x115 = x111;
			x115.free();

			// val X2368 = (X2367)(i | @) * (fc8_W)(j | @)
			JCudaTensor x116;
			JCudaMatrix x117;
			JCudaMatrix x118;
			JCudaTensor x119;
			x119 = x113;
			x117 = x119.asMatrix(1, true);
			JCudaTensor x120;
			x120 = x121;
			x118 = x120.asMatrix(1, true);
			x116 = x117.times(x118);

			// Dealloc(X2367)
			JCudaTensor x122;
			x122 = x113;
			x122.free();

			// val X2370 = (X2368 + (i) => fc8_B)
			JCudaTensor x123;
			JCudaTensor x124, x125;
			x124 = x116;
			x125 = x126;
			x123 = x125.copy(128, x124);

			// Prediction(X2370)
			JCudaTensor x127;
			x127 = x123;
			System.out.println(x3 + " inference " + java.util.Arrays.toString(x127.asJTensor().prediction()));

			// Dealloc(X2370)
			JCudaTensor x128;
			x128 = x123;
			x128.free();

		}

	}

}