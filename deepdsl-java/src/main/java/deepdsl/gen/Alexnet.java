package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;


public class Alexnet {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// decay_1
	static float decay_1 = 0.999995f;
	// lrn_rate_1
	static float lrn_rate_1 = -0.01f;
	// lrn_rate_2
	static float lrn_rate_2 = -0.02f;
	// momentum
	static float momentum = 0.9f;
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
	// train_data_path
	static String train_data_path = "dataset/imagenet/ilsvrc12_train_lmdb";
	// train_itr
	static int train_itr = 1000;
	// train_size
	static int train_size = 1000000;

	// (Convolv(1,1),List(List(128, 256, 13, 13), List(384, 256, 3, 3), List(384)))
	static JCudnnConvolution x52 = new JCudnnConvolution(new int[]{128,256,13,13},new int[]{384,256,3,3},new int[]{384}, 1, 1);
	// (Convolv(1,1),List(List(128, 384, 13, 13), List(256, 384, 3, 3), List(256)))
	static JCudnnConvolution x71 = new JCudnnConvolution(new int[]{128,384,13,13},new int[]{256,384,3,3},new int[]{256}, 1, 1);
	// (Convolv(1,1),List(List(128, 384, 13, 13), List(384, 384, 3, 3), List(384)))
	static JCudnnConvolution x62 = new JCudnnConvolution(new int[]{128,384,13,13},new int[]{384,384,3,3},new int[]{384}, 1, 1);
	// (Convolv(1,2),List(List(128, 96, 27, 27), List(256, 96, 5, 5), List(256)))
	static JCudnnConvolution x36 = new JCudnnConvolution(new int[]{128,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
	// (Convolv(4,2),List(List(128, 3, 224, 224), List(96, 3, 11, 11), List(96)))
	static JCudnnConvolution x20 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 2);
	// (Dropout(0.5),List(List(128, 4096)))
	static JCudnnDropout x94 = new JCudnnDropout(new int[]{128,4096}, 0.5f);
	// (LRN(5,1.0E-4,0.75),List(List(128, 256, 27, 27)))
	static JCudnnLRN x42 = new JCudnnLRN(new int[]{128,256,27,27}, 5, 1.0E-4, 0.75);
	// (LRN(5,1.0E-4,0.75),List(List(128, 96, 55, 55)))
	static JCudnnLRN x26 = new JCudnnLRN(new int[]{128,96,55,55}, 5, 1.0E-4, 0.75);
	// (Lmdb(1000000,10000,Win32,1000),false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, platform, 1000, true);
	// (Lmdb(1000000,10000,Win32,1000),true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, platform, 1000, false);
	// (LogSoftmax(),List(List(128, 1000)))
	static JCudnnSoftmax x121 = new JCudnnSoftmax(new int[]{128,1000}, SoftmaxAlgorithm.LOG);
	// (Pooling(3,2,0,true),List(List(128, 256, 13, 13)))
	static JCudnnPooling x77 = new JCudnnPooling(new int[]{128,256,13,13}, 3, 2, 0, PoolingType.MAX);
	// (Pooling(3,2,0,true),List(List(128, 256, 27, 27)))
	static JCudnnPooling x45 = new JCudnnPooling(new int[]{128,256,27,27}, 3, 2, 0, PoolingType.MAX);
	// (Pooling(3,2,0,true),List(List(128, 96, 55, 55)))
	static JCudnnPooling x29 = new JCudnnPooling(new int[]{128,96,55,55}, 3, 2, 0, PoolingType.MAX);
	// (ReLU(),List(List(128, 256, 13, 13)))
	static JCudnnActivation x74 = new JCudnnActivation(new int[]{128,256,13,13}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 256, 27, 27)))
	static JCudnnActivation x39 = new JCudnnActivation(new int[]{128,256,27,27}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 384, 13, 13)))
	static JCudnnActivation x55 = new JCudnnActivation(new int[]{128,384,13,13}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 4096)))
	static JCudnnActivation x91 = new JCudnnActivation(new int[]{128,4096}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 55, 55)))
	static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,96,55,55}, ActivationMode.RELU);
	// Precision((Sum(X2341) / |128|))
	static float x483;
	// V_cv1_B
	static JCudaTensor x365 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x359 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x328 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x332 = JTensor.constFloat(0.0f, 256, 96, 5, 5).asJCudaTensor();
	// V_cv3_B
	static JCudaTensor x291 = JTensor.constFloat(0.0f, 384).asJCudaTensor();
	// V_cv3_W
	static JCudaTensor x298 = JTensor.constFloat(0.0f, 384, 256, 3, 3).asJCudaTensor();
	// V_cv4_B
	static JCudaTensor x271 = JTensor.constFloat(0.0f, 384).asJCudaTensor();
	// V_cv4_W
	static JCudaTensor x275 = JTensor.constFloat(0.0f, 384, 384, 3, 3).asJCudaTensor();
	// V_cv5_B
	static JCudaTensor x245 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// V_cv5_W
	static JCudaTensor x252 = JTensor.constFloat(0.0f, 256, 384, 3, 3).asJCudaTensor();
	// V_fc6_B
	static JCudaTensor x230 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
	// V_fc6_W
	static JCudaTensor x218 = JTensor.constFloat(0.0f, 4096, 9216).asJCudaTensor();
	// V_fc7_B
	static JCudaTensor x183 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
	// V_fc7_W
	static JCudaTensor x187 = JTensor.constFloat(0.0f, 4096, 4096).asJCudaTensor();
	// V_fc8_B
	static JCudaTensor x149 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc8_W
	static JCudaTensor x156 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;
	// cv1_B
	static JCudaTensor x19 = JTensor.constFloat(0.0f, 96).load(network_dir + "/cv1_B").asJCudaTensor();
	// cv1_W
	static JCudaTensor x18 = JTensor.gaussianFloat(0.01f, 96, 3, 11, 11).load(network_dir + "/cv1_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x35 = JTensor.constFloat(0.1f, 256).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x34 = JTensor.gaussianFloat(0.01f, 256, 96, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
	// cv3_B
	static JCudaTensor x51 = JTensor.constFloat(0.0f, 384).load(network_dir + "/cv3_B").asJCudaTensor();
	// cv3_W
	static JCudaTensor x50 = JTensor.gaussianFloat(0.01f, 384, 256, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
	// cv4_B
	static JCudaTensor x61 = JTensor.constFloat(0.1f, 384).load(network_dir + "/cv4_B").asJCudaTensor();
	// cv4_W
	static JCudaTensor x60 = JTensor.gaussianFloat(0.01f, 384, 384, 3, 3).load(network_dir + "/cv4_W").asJCudaTensor();
	// cv5_B
	static JCudaTensor x70 = JTensor.constFloat(0.1f, 256).load(network_dir + "/cv5_B").asJCudaTensor();
	// cv5_W
	static JCudaTensor x69 = JTensor.gaussianFloat(0.01f, 256, 384, 3, 3).load(network_dir + "/cv5_W").asJCudaTensor();
	// fc6_B
	static JCudaTensor x88 = JTensor.constFloat(0.1f, 4096).load(network_dir + "/fc6_B").asJCudaTensor();
	// fc6_W
	static JCudaTensor x84 = JTensor.gaussianFloat(0.005f, 4096, 9216).load(network_dir + "/fc6_W").asJCudaTensor();
	// fc7_B
	static JCudaTensor x104 = JTensor.constFloat(0.1f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
	// fc7_W
	static JCudaTensor x100 = JTensor.gaussianFloat(0.005f, 4096, 4096).load(network_dir + "/fc7_W").asJCudaTensor();
	// fc8_B
	static JCudaTensor x118 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
	// fc8_W
	static JCudaTensor x114 = JTensor.gaussianFloat(0.01f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

	public static void main(String[] args){
		double t = System.nanoTime();
		train();
		System.out.println((System.nanoTime() - t) / 1.0E9);
		test();
		x19.save(network_dir + "/cv1_B");
		x18.save(network_dir + "/cv1_W");
		x35.save(network_dir + "/cv2_B");
		x34.save(network_dir + "/cv2_W");
		x51.save(network_dir + "/cv3_B");
		x50.save(network_dir + "/cv3_W");
		x61.save(network_dir + "/cv4_B");
		x60.save(network_dir + "/cv4_W");
		x70.save(network_dir + "/cv5_B");
		x69.save(network_dir + "/cv5_W");
		x88.save(network_dir + "/fc6_B");
		x84.save(network_dir + "/fc6_W");
		x104.save(network_dir + "/fc7_B");
		x100.save(network_dir + "/fc7_W");
		x118.save(network_dir + "/fc8_B");
		x114.save(network_dir + "/fc8_W");
		x61.free();
		x252.free();
		x187.free();
		x100.free();
		x51.free();
		x34.free();
		x18.free();
		x19.free();
		x84.free();
		x275.free();
		x230.free();
		x332.free();
		x35.free();
		x271.free();
		x50.free();
		x183.free();
		x70.free();
		x60.free();
		x88.free();
		x104.free();
		x359.free();
		x291.free();
		x69.free();
		x149.free();
		x298.free();
		x118.free();
		x156.free();
		x328.free();
		x245.free();
		x114.free();
		x365.free();
		x218.free();
		x94.free();
		x39.free();
		x29.free();
		x36.free();
		x71.free();
		x77.free();
		x26.free();
		x23.free();
		x74.free();
		x45.free();
		x121.free();
		x55.free();
		x91.free();
		x52.free();
		x20.free();
		x62.free();
		x42.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void train() {
		for(int x5=0; x5<train_itr; x5++) {
			JTensorFloatTuple x6 =  x1.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X13 = Cuda(X)
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x3;
			x7 = x8.asJCudaTensor();

			// val X43 = Cuda(Indicator(Y, 1000))
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x4.asIndicator(1000);
			x9 = x10.asJCudaTensor();

			// val X123 = - X43.copy
			JCudaTensor x11;
			JCudaTensor x12;
			float x13;
			x12 = x9;
			x12 = x12.clone();
			x13 = -1;
			x11 = x12.times_i(x13);

			// val X14 = Convolv(4,2)(X13,cv1_W,cv1_B)
			JCudaTensor x14;
			JCudaTensor x15, x16, x17;
			x15 = x7;
			x16 = x18;
			x17 = x19;
			x14 = x20.forward(x15, x16, x17);

			// val X15 = ReLU()(X14)
			JCudaTensor x21;
			JCudaTensor x22;
			x22 = x14;
			x21 = x23.forward(x22);

			// val X16 = LRN(5,1.0E-4,0.75)(X15)
			JCudaTensor x24;
			JCudaTensor x25;
			x25 = x21;
			x24 = x26.forward(x25);

			// val X17 = Pooling(3,2,0,true)(X16)
			JCudaTensor x27;
			JCudaTensor x28;
			x28 = x24;
			x27 = x29.forward(x28);

			// val X18 = Convolv(1,2)(X17,cv2_W,cv2_B)
			JCudaTensor x30;
			JCudaTensor x31, x32, x33;
			x31 = x27;
			x32 = x34;
			x33 = x35;
			x30 = x36.forward(x31, x32, x33);

			// val X19 = ReLU()(X18)
			JCudaTensor x37;
			JCudaTensor x38;
			x38 = x30;
			x37 = x39.forward(x38);

			// val X20 = LRN(5,1.0E-4,0.75)(X19)
			JCudaTensor x40;
			JCudaTensor x41;
			x41 = x37;
			x40 = x42.forward(x41);

			// val X21 = Pooling(3,2,0,true)(X20)
			JCudaTensor x43;
			JCudaTensor x44;
			x44 = x40;
			x43 = x45.forward(x44);

			// val X22 = Convolv(1,1)(X21,cv3_W,cv3_B)
			JCudaTensor x46;
			JCudaTensor x47, x48, x49;
			x47 = x43;
			x48 = x50;
			x49 = x51;
			x46 = x52.forward(x47, x48, x49);

			// val X23 = ReLU()(X22)
			JCudaTensor x53;
			JCudaTensor x54;
			x54 = x46;
			x53 = x55.forward(x54);

			// val X24 = Convolv(1,1)(X23,cv4_W,cv4_B)
			JCudaTensor x56;
			JCudaTensor x57, x58, x59;
			x57 = x53;
			x58 = x60;
			x59 = x61;
			x56 = x62.forward(x57, x58, x59);

			// val X25 = ReLU()(X24)
			JCudaTensor x63;
			JCudaTensor x64;
			x64 = x56;
			x63 = x55.forward(x64);

			// val X26 = Convolv(1,1)(X25,cv5_W,cv5_B)
			JCudaTensor x65;
			JCudaTensor x66, x67, x68;
			x66 = x63;
			x67 = x69;
			x68 = x70;
			x65 = x71.forward(x66, x67, x68);

			// val X27 = ReLU()(X26)
			JCudaTensor x72;
			JCudaTensor x73;
			x73 = x65;
			x72 = x74.forward(x73);

			// val X28 = Pooling(3,2,0,true)(X27)
			JCudaTensor x75;
			JCudaTensor x76;
			x76 = x72;
			x75 = x77.forward(x76);

			// val X29 = (X28[1><3])(i | @) * (fc6_W)(j | @)
			JCudaTensor x78;
			JCudaMatrix x79;
			JCudaMatrix x80;
			JCudaTensor x81;
			JCudaTensor x82;
			x82 = x75;
			x81 = x82.flatten(1, new int[]{256, 6, 6});
			x79 = x81.asMatrix(1, true);
			JCudaTensor x83;
			x83 = x84;
			x80 = x83.asMatrix(1, true);
			x78 = x79.times(x80);

			// val X31 = (X29 + (i) => fc6_B)
			JCudaTensor x85;
			JCudaTensor x86, x87;
			x86 = x78;
			x87 = x88;
			x85 = x87.copy(128, x86);

			// val X32 = ReLU()(X31)
			JCudaTensor x89;
			JCudaTensor x90;
			x90 = x85;
			x89 = x91.forward(x90);

			// val X33 = Dropout(0.5)(X32)
			JCudaTensor x92;
			JCudaTensor x93;
			x93 = x89;
			x92 = x94.forward(x93);

			// val X34 = (X33)(i | @) * (fc7_W)(j | @)
			JCudaTensor x95;
			JCudaMatrix x96;
			JCudaMatrix x97;
			JCudaTensor x98;
			x98 = x92;
			x96 = x98.asMatrix(1, true);
			JCudaTensor x99;
			x99 = x100;
			x97 = x99.asMatrix(1, true);
			x95 = x96.times(x97);

			// val X36 = (X34 + (i) => fc7_B)
			JCudaTensor x101;
			JCudaTensor x102, x103;
			x102 = x95;
			x103 = x104;
			x101 = x103.copy(128, x102);

			// val X37 = ReLU()(X36)
			JCudaTensor x105;
			JCudaTensor x106;
			x106 = x101;
			x105 = x91.forward(x106);

			// val X38 = Dropout(0.5)(X37)
			JCudaTensor x107;
			JCudaTensor x108;
			x108 = x105;
			x107 = x94.forward(x108);

			// val X39 = (X38)(i | @) * (fc8_W)(j | @)
			JCudaTensor x109;
			JCudaMatrix x110;
			JCudaMatrix x111;
			JCudaTensor x112;
			x112 = x107;
			x110 = x112.asMatrix(1, true);
			JCudaTensor x113;
			x113 = x114;
			x111 = x113.asMatrix(1, true);
			x109 = x110.times(x111);

			// val X41 = (X39 + (i) => fc8_B)
			JCudaTensor x115;
			JCudaTensor x116, x117;
			x116 = x109;
			x117 = x118;
			x115 = x117.copy(128, x116);

			// val X42 = LogSoftmax()(X41)
			JCudaTensor x119;
			JCudaTensor x120;
			x120 = x115;
			x119 = x121.forward(x120);

			// Dealloc(X41)
			JCudaTensor x122;
			x122 = x115;
			x122.free();

			// val X124 = (X123 / |128|)
			JCudaTensor x123;
			JCudaTensor x124;
			float x125;
			x124 = x11;
			float x126;
			x126 = 128;
			x125 = 1 / x126;
			x123 = x124.times_i(x125);

			// Cost(((0 - (X43 . X42)) / |128|))
			float x127;
			float x128;
			float x129;
			float x130;
			JCudaTensor x131, x132;
			x131 = x9;
			x132 = x119;
			x130 = x131.dot(x132);
			x128 = - x130;
			x129 = 128;
			x127 = x128 / x129;
			System.out.println(x5 + " " + x127);
			if (Float.isNaN(x127)) { System.exit(-1); }

			// Dealloc(X43)
			JCudaTensor x133;
			x133 = x9;
			x133.free();

			// val X155 = X124 * d_LogSoftmax()(X42)/d_X41
			JCudaTensor x134;
			JCudaTensor x135, x136;
			x135 = x123;
			x136 = x119;
			x134 = x121.backward(x135, x136);

			// Dealloc(X124)
			JCudaTensor x137;
			x137 = x123;
			x137.free();

			// Dealloc(X42)
			JCudaTensor x138;
			x138 = x119;
			x138.free();

			// val m1 = (i21) => fc8_W[@, i21]
			JCudaMatrix x139;
			JCudaTensor x140;
			x140 = x114;
			x139 = x140.asMatrix(1, false);

			// val m43 = (i490) => X155[@, i490]
			JCudaMatrix x141;
			JCudaTensor x142;
			x142 = x134;
			x141 = x142.asMatrix(1, false);

			// val m45 = (i498) => X38[@, i498]
			JCudaMatrix x143;
			JCudaTensor x144;
			x144 = x107;
			x143 = x144.asMatrix(1, false);

			// val X181 = (X155)(i20 | @) * m1
			JCudaTensor x145;
			JCudaMatrix x146;
			JCudaMatrix x147;
			JCudaTensor x148;
			x148 = x134;
			x146 = x148.asMatrix(1, true);
			x147 = x139;
			x145 = x146.times(x147);

			// V_fc8_B <~~ Sum(m43)
			float x150, x151;
			x150 = lrn_rate_2;
			x151 = momentum;
			JCudaMatrix x152;
			x152 = x141;
			x152.sum(x149, x150, x151);

			// val X182 = X181 * d_Dropout(0.5)()/d_X37
			JCudaTensor x153;
			JCudaTensor x154;
			x154 = x145;
			x153 = x94.backward(x154);

			// Dealloc(X181)
			JCudaTensor x155;
			x155 = x145;
			x155.free();

			// V_fc8_W <~~ m43 * m45
			float x157, x158;
			x157 = lrn_rate_1;
			x158 = momentum;
			JCudaMatrix x159;
			JCudaMatrix x160;
			x159 = x141;
			x160 = x143;
			x159.times(x160, x156, x157, x158);

			// Dealloc(X155)
			JCudaTensor x161;
			x161 = x134;
			x161.free();

			// Dealloc(X38)
			JCudaTensor x162;
			x162 = x107;
			x162.free();

			// fc8_B <~~ V_fc8_B
			float x163, x164;
			x163 = 1;
			x164 = 1;
			JCudaTensor x165;
			x165 = x149;
			x118.update(x165, x163, x164);

			// fc8_W <~~ V_fc8_W
			float x166, x167;
			x166 = 1;
			x167 = decay_1;
			JCudaTensor x168;
			x168 = x156;
			x114.update(x168, x166, x167);

			// val X184 = X182 * d_ReLU()(X37)/d_X36
			JCudaTensor x169;
			JCudaTensor x170, x171;
			x170 = x153;
			x171 = x105;
			x169 = x91.backward(x170, x171);

			// Dealloc(X37)
			JCudaTensor x172;
			x172 = x105;
			x172.free();

			// val m2 = (i25) => fc7_W[@, i25]
			JCudaMatrix x173;
			JCudaTensor x174;
			x174 = x100;
			x173 = x174.asMatrix(1, false);

			// val m39 = (i453) => X184[@, i453]
			JCudaMatrix x175;
			JCudaTensor x176;
			x176 = x169;
			x175 = x176.asMatrix(1, false);

			// val m42 = (i471) => X33[@, i471]
			JCudaMatrix x177;
			JCudaTensor x178;
			x178 = x92;
			x177 = x178.asMatrix(1, false);

			// val X185 = (X184)(i24 | @) * m2
			JCudaTensor x179;
			JCudaMatrix x180;
			JCudaMatrix x181;
			JCudaTensor x182;
			x182 = x169;
			x180 = x182.asMatrix(1, true);
			x181 = x173;
			x179 = x180.times(x181);

			// V_fc7_B <~~ Sum(m39)
			float x184, x185;
			x184 = lrn_rate_2;
			x185 = momentum;
			JCudaMatrix x186;
			x186 = x175;
			x186.sum(x183, x184, x185);

			// V_fc7_W <~~ m39 * m42
			float x188, x189;
			x188 = lrn_rate_1;
			x189 = momentum;
			JCudaMatrix x190;
			JCudaMatrix x191;
			x190 = x175;
			x191 = x177;
			x190.times(x191, x187, x188, x189);

			// Dealloc(X184)
			JCudaTensor x192;
			x192 = x169;
			x192.free();

			// Dealloc(X33)
			JCudaTensor x193;
			x193 = x92;
			x193.free();

			// val X186 = X185 * d_Dropout(0.5)()/d_X32
			JCudaTensor x194;
			JCudaTensor x195;
			x195 = x179;
			x194 = x94.backward(x195);

			// Dealloc(X185)
			JCudaTensor x196;
			x196 = x179;
			x196.free();

			// fc7_B <~~ V_fc7_B
			float x197, x198;
			x197 = 1;
			x198 = 1;
			JCudaTensor x199;
			x199 = x183;
			x104.update(x199, x197, x198);

			// fc7_W <~~ V_fc7_W
			float x200, x201;
			x200 = 1;
			x201 = decay_1;
			JCudaTensor x202;
			x202 = x187;
			x100.update(x202, x200, x201);

			// val X188 = X186 * d_ReLU()(X32)/d_X31
			JCudaTensor x203;
			JCudaTensor x204, x205;
			x204 = x194;
			x205 = x89;
			x203 = x91.backward(x204, x205);

			// Dealloc(X32)
			JCudaTensor x206;
			x206 = x89;
			x206.free();

			// val m3 = (i29) => fc6_W[@, i29]
			JCudaMatrix x207;
			JCudaTensor x208;
			x208 = x84;
			x207 = x208.asMatrix(1, false);

			// val m33 = (i396) => X188[@, i396]
			JCudaMatrix x209;
			JCudaTensor x210;
			x210 = x203;
			x209 = x210.asMatrix(1, false);

			// val m37 = (i424) => X28[1><3][@, i424]
			JCudaMatrix x211;
			JCudaTensor x212;
			JCudaTensor x213;
			x213 = x75;
			x212 = x213.flatten(1, new int[]{256, 6, 6});
			x211 = x212.asMatrix(1, false);

			// val X189 = (X188)(i28 | @) * m3
			JCudaTensor x214;
			JCudaMatrix x215;
			JCudaMatrix x216;
			JCudaTensor x217;
			x217 = x203;
			x215 = x217.asMatrix(1, true);
			x216 = x207;
			x214 = x215.times(x216);

			// V_fc6_W <~~ m33 * m37
			float x219, x220;
			x219 = lrn_rate_1;
			x220 = momentum;
			JCudaMatrix x221;
			JCudaMatrix x222;
			x221 = x209;
			x222 = x211;
			x221.times(x222, x218, x219, x220);

			// val X191 = X189[1<>3] * d_Pooling(3,2,0,true)(X28,X27)/d_X27
			JCudaTensor x223;
			JCudaTensor x224, x225, x226;
			JCudaTensor x227;
			x227 = x214;
			x224 = x227.unflatten(1, new int[]{256, 6, 6});
			x225 = x75;
			x226 = x72;
			x223 = x77.backward(x224, x225, x226);

			// Dealloc(X189)
			JCudaTensor x228;
			x228 = x214;
			x228.free();

			// Dealloc(X28)
			JCudaTensor x229;
			x229 = x75;
			x229.free();

			// V_fc6_B <~~ Sum(m33)
			float x231, x232;
			x231 = lrn_rate_2;
			x232 = momentum;
			JCudaMatrix x233;
			x233 = x209;
			x233.sum(x230, x231, x232);

			// Dealloc(X188)
			JCudaTensor x234;
			x234 = x203;
			x234.free();

			// fc6_W <~~ V_fc6_W
			float x235, x236;
			x235 = 1;
			x236 = decay_1;
			JCudaTensor x237;
			x237 = x218;
			x84.update(x237, x235, x236);

			// fc6_B <~~ V_fc6_B
			float x238, x239;
			x238 = 1;
			x239 = 1;
			JCudaTensor x240;
			x240 = x230;
			x88.update(x240, x238, x239);

			// val X193 = X191 * d_ReLU()(X27)/d_X26
			JCudaTensor x241;
			JCudaTensor x242, x243;
			x242 = x223;
			x243 = x72;
			x241 = x74.backward(x242, x243);

			// Dealloc(X27)
			JCudaTensor x244;
			x244 = x72;
			x244.free();

			// V_cv5_B <~~ X193 * d_Convolv(1,1)()/d_cv5_B
			float x246, x247;
			x246 = lrn_rate_2;
			x247 = momentum;
			JCudaTensor x248;
			x248 = x241;
			x71.backward_bias(x248, x245, x246, x247);

			// val X194 = X193 * d_Convolv(1,1)(cv5_W)/d_X25
			JCudaTensor x249;
			JCudaTensor x250, x251;
			x250 = x241;
			x251 = x69;
			x249 = x71.backward_data(x250, x251);

			// V_cv5_W <~~ X193 * d_Convolv(1,1)(X25)/d_cv5_W
			float x253, x254;
			x253 = lrn_rate_1;
			x254 = momentum;
			JCudaTensor x255, x256;
			x255 = x241;
			x256 = x63;
			x71.backward_filter(x255, x256, x252, x253, x254);

			// Dealloc(X193)
			JCudaTensor x257;
			x257 = x241;
			x257.free();

			// cv5_B <~~ V_cv5_B
			float x258, x259;
			x258 = 1;
			x259 = 1;
			JCudaTensor x260;
			x260 = x245;
			x70.update(x260, x258, x259);

			// cv5_W <~~ V_cv5_W
			float x261, x262;
			x261 = 1;
			x262 = decay_1;
			JCudaTensor x263;
			x263 = x252;
			x69.update(x263, x261, x262);

			// val X196 = X194 * d_ReLU()(X25)/d_X24
			JCudaTensor x264;
			JCudaTensor x265, x266;
			x265 = x249;
			x266 = x63;
			x264 = x55.backward(x265, x266);

			// Dealloc(X25)
			JCudaTensor x267;
			x267 = x63;
			x267.free();

			// val X197 = X196 * d_Convolv(1,1)(cv4_W)/d_X23
			JCudaTensor x268;
			JCudaTensor x269, x270;
			x269 = x264;
			x270 = x60;
			x268 = x62.backward_data(x269, x270);

			// V_cv4_B <~~ X196 * d_Convolv(1,1)()/d_cv4_B
			float x272, x273;
			x272 = lrn_rate_2;
			x273 = momentum;
			JCudaTensor x274;
			x274 = x264;
			x62.backward_bias(x274, x271, x272, x273);

			// V_cv4_W <~~ X196 * d_Convolv(1,1)(X23)/d_cv4_W
			float x276, x277;
			x276 = lrn_rate_1;
			x277 = momentum;
			JCudaTensor x278, x279;
			x278 = x264;
			x279 = x53;
			x62.backward_filter(x278, x279, x275, x276, x277);

			// Dealloc(X196)
			JCudaTensor x280;
			x280 = x264;
			x280.free();

			// cv4_B <~~ V_cv4_B
			float x281, x282;
			x281 = 1;
			x282 = 1;
			JCudaTensor x283;
			x283 = x271;
			x61.update(x283, x281, x282);

			// cv4_W <~~ V_cv4_W
			float x284, x285;
			x284 = 1;
			x285 = decay_1;
			JCudaTensor x286;
			x286 = x275;
			x60.update(x286, x284, x285);

			// val X199 = X197 * d_ReLU()(X23)/d_X22
			JCudaTensor x287;
			JCudaTensor x288, x289;
			x288 = x268;
			x289 = x53;
			x287 = x55.backward(x288, x289);

			// Dealloc(X23)
			JCudaTensor x290;
			x290 = x53;
			x290.free();

			// V_cv3_B <~~ X199 * d_Convolv(1,1)()/d_cv3_B
			float x292, x293;
			x292 = lrn_rate_2;
			x293 = momentum;
			JCudaTensor x294;
			x294 = x287;
			x52.backward_bias(x294, x291, x292, x293);

			// val X200 = X199 * d_Convolv(1,1)(cv3_W)/d_X21
			JCudaTensor x295;
			JCudaTensor x296, x297;
			x296 = x287;
			x297 = x50;
			x295 = x52.backward_data(x296, x297);

			// V_cv3_W <~~ X199 * d_Convolv(1,1)(X21)/d_cv3_W
			float x299, x300;
			x299 = lrn_rate_1;
			x300 = momentum;
			JCudaTensor x301, x302;
			x301 = x287;
			x302 = x43;
			x52.backward_filter(x301, x302, x298, x299, x300);

			// Dealloc(X199)
			JCudaTensor x303;
			x303 = x287;
			x303.free();

			// cv3_B <~~ V_cv3_B
			float x304, x305;
			x304 = 1;
			x305 = 1;
			JCudaTensor x306;
			x306 = x291;
			x51.update(x306, x304, x305);

			// cv3_W <~~ V_cv3_W
			float x307, x308;
			x307 = 1;
			x308 = decay_1;
			JCudaTensor x309;
			x309 = x298;
			x50.update(x309, x307, x308);

			// val X202 = X200 * d_Pooling(3,2,0,true)(X21,X20)/d_X20
			JCudaTensor x310;
			JCudaTensor x311, x312, x313;
			x311 = x295;
			x312 = x43;
			x313 = x40;
			x310 = x45.backward(x311, x312, x313);

			// Dealloc(X200)
			JCudaTensor x314;
			x314 = x295;
			x314.free();

			// Dealloc(X21)
			JCudaTensor x315;
			x315 = x43;
			x315.free();

			// val X204 = X202 * d_LRN(5,1.0E-4,0.75)(X20,X19)/d_X19
			JCudaTensor x316;
			JCudaTensor x317, x318, x319;
			x317 = x310;
			x318 = x40;
			x319 = x37;
			x316 = x42.backward(x317, x318, x319);

			// Dealloc(X20)
			JCudaTensor x320;
			x320 = x40;
			x320.free();

			// val X206 = X204 * d_ReLU()(X19)/d_X18
			JCudaTensor x321;
			JCudaTensor x322, x323;
			x322 = x316;
			x323 = x37;
			x321 = x39.backward(x322, x323);

			// Dealloc(X19)
			JCudaTensor x324;
			x324 = x37;
			x324.free();

			// val X207 = X206 * d_Convolv(1,2)(cv2_W)/d_X17
			JCudaTensor x325;
			JCudaTensor x326, x327;
			x326 = x321;
			x327 = x34;
			x325 = x36.backward_data(x326, x327);

			// V_cv2_B <~~ X206 * d_Convolv(1,2)()/d_cv2_B
			float x329, x330;
			x329 = lrn_rate_2;
			x330 = momentum;
			JCudaTensor x331;
			x331 = x321;
			x36.backward_bias(x331, x328, x329, x330);

			// V_cv2_W <~~ X206 * d_Convolv(1,2)(X17)/d_cv2_W
			float x333, x334;
			x333 = lrn_rate_1;
			x334 = momentum;
			JCudaTensor x335, x336;
			x335 = x321;
			x336 = x27;
			x36.backward_filter(x335, x336, x332, x333, x334);

			// Dealloc(X206)
			JCudaTensor x337;
			x337 = x321;
			x337.free();

			// cv2_B <~~ V_cv2_B
			float x338, x339;
			x338 = 1;
			x339 = 1;
			JCudaTensor x340;
			x340 = x328;
			x35.update(x340, x338, x339);

			// cv2_W <~~ V_cv2_W
			float x341, x342;
			x341 = 1;
			x342 = decay_1;
			JCudaTensor x343;
			x343 = x332;
			x34.update(x343, x341, x342);

			// val X209 = X207 * d_Pooling(3,2,0,true)(X17,X16)/d_X16
			JCudaTensor x344;
			JCudaTensor x345, x346, x347;
			x345 = x325;
			x346 = x27;
			x347 = x24;
			x344 = x29.backward(x345, x346, x347);

			// Dealloc(X207)
			JCudaTensor x348;
			x348 = x325;
			x348.free();

			// Dealloc(X17)
			JCudaTensor x349;
			x349 = x27;
			x349.free();

			// val X211 = X209 * d_LRN(5,1.0E-4,0.75)(X16,X15)/d_X15
			JCudaTensor x350;
			JCudaTensor x351, x352, x353;
			x351 = x344;
			x352 = x24;
			x353 = x21;
			x350 = x26.backward(x351, x352, x353);

			// Dealloc(X16)
			JCudaTensor x354;
			x354 = x24;
			x354.free();

			// val X213 = X211 * d_ReLU()(X15)/d_X14
			JCudaTensor x355;
			JCudaTensor x356, x357;
			x356 = x350;
			x357 = x21;
			x355 = x23.backward(x356, x357);

			// Dealloc(X15)
			JCudaTensor x358;
			x358 = x21;
			x358.free();

			// V_cv1_W <~~ X213 * d_Convolv(4,2)(X13)/d_cv1_W
			float x360, x361;
			x360 = lrn_rate_1;
			x361 = momentum;
			JCudaTensor x362, x363;
			x362 = x355;
			x363 = x7;
			x20.backward_filter(x362, x363, x359, x360, x361);

			// Dealloc(X13)
			JCudaTensor x364;
			x364 = x7;
			x364.free();

			// V_cv1_B <~~ X213 * d_Convolv(4,2)()/d_cv1_B
			float x366, x367;
			x366 = lrn_rate_2;
			x367 = momentum;
			JCudaTensor x368;
			x368 = x355;
			x20.backward_bias(x368, x365, x366, x367);

			// Dealloc(X213)
			JCudaTensor x369;
			x369 = x355;
			x369.free();

			// cv1_W <~~ V_cv1_W
			float x370, x371;
			x370 = 1;
			x371 = decay_1;
			JCudaTensor x372;
			x372 = x359;
			x18.update(x372, x370, x371);

			// cv1_B <~~ V_cv1_B
			float x373, x374;
			x373 = 1;
			x374 = 1;
			JCudaTensor x375;
			x375 = x365;
			x19.update(x375, x373, x374);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X2308 = Cuda(X)
			JCudaTensor x376;
			JTensorFloat x377;
			x377 = x3;
			x376 = x377.asJCudaTensor();

			// val X2309 = Convolv(4,2)(X2308,cv1_W,cv1_B)
			JCudaTensor x378;
			JCudaTensor x379, x380, x381;
			x379 = x376;
			x380 = x18;
			x381 = x19;
			x378 = x20.forward(x379, x380, x381);

			// Dealloc(X2308)
			JCudaTensor x382;
			x382 = x376;
			x382.free();

			// val X2310 = ReLU()(X2309)
			JCudaTensor x383;
			JCudaTensor x384;
			x384 = x378;
			x383 = x23.forward(x384);

			// val X2311 = LRN(5,1.0E-4,0.75)(X2310)
			JCudaTensor x385;
			JCudaTensor x386;
			x386 = x383;
			x385 = x26.forward(x386);

			// Dealloc(X2310)
			JCudaTensor x387;
			x387 = x383;
			x387.free();

			// val X2312 = Pooling(3,2,0,true)(X2311)
			JCudaTensor x388;
			JCudaTensor x389;
			x389 = x385;
			x388 = x29.forward(x389);

			// Dealloc(X2311)
			JCudaTensor x390;
			x390 = x385;
			x390.free();

			// val X2313 = Convolv(1,2)(X2312,cv2_W,cv2_B)
			JCudaTensor x391;
			JCudaTensor x392, x393, x394;
			x392 = x388;
			x393 = x34;
			x394 = x35;
			x391 = x36.forward(x392, x393, x394);

			// Dealloc(X2312)
			JCudaTensor x395;
			x395 = x388;
			x395.free();

			// val X2314 = ReLU()(X2313)
			JCudaTensor x396;
			JCudaTensor x397;
			x397 = x391;
			x396 = x39.forward(x397);

			// val X2315 = LRN(5,1.0E-4,0.75)(X2314)
			JCudaTensor x398;
			JCudaTensor x399;
			x399 = x396;
			x398 = x42.forward(x399);

			// Dealloc(X2314)
			JCudaTensor x400;
			x400 = x396;
			x400.free();

			// val X2316 = Pooling(3,2,0,true)(X2315)
			JCudaTensor x401;
			JCudaTensor x402;
			x402 = x398;
			x401 = x45.forward(x402);

			// Dealloc(X2315)
			JCudaTensor x403;
			x403 = x398;
			x403.free();

			// val X2317 = Convolv(1,1)(X2316,cv3_W,cv3_B)
			JCudaTensor x404;
			JCudaTensor x405, x406, x407;
			x405 = x401;
			x406 = x50;
			x407 = x51;
			x404 = x52.forward(x405, x406, x407);

			// Dealloc(X2316)
			JCudaTensor x408;
			x408 = x401;
			x408.free();

			// val X2318 = ReLU()(X2317)
			JCudaTensor x409;
			JCudaTensor x410;
			x410 = x404;
			x409 = x55.forward(x410);

			// val X2319 = Convolv(1,1)(X2318,cv4_W,cv4_B)
			JCudaTensor x411;
			JCudaTensor x412, x413, x414;
			x412 = x409;
			x413 = x60;
			x414 = x61;
			x411 = x62.forward(x412, x413, x414);

			// Dealloc(X2318)
			JCudaTensor x415;
			x415 = x409;
			x415.free();

			// val X2320 = ReLU()(X2319)
			JCudaTensor x416;
			JCudaTensor x417;
			x417 = x411;
			x416 = x55.forward(x417);

			// val X2321 = Convolv(1,1)(X2320,cv5_W,cv5_B)
			JCudaTensor x418;
			JCudaTensor x419, x420, x421;
			x419 = x416;
			x420 = x69;
			x421 = x70;
			x418 = x71.forward(x419, x420, x421);

			// Dealloc(X2320)
			JCudaTensor x422;
			x422 = x416;
			x422.free();

			// val X2322 = ReLU()(X2321)
			JCudaTensor x423;
			JCudaTensor x424;
			x424 = x418;
			x423 = x74.forward(x424);

			// val X2323 = Pooling(3,2,0,true)(X2322)
			JCudaTensor x425;
			JCudaTensor x426;
			x426 = x423;
			x425 = x77.forward(x426);

			// Dealloc(X2322)
			JCudaTensor x427;
			x427 = x423;
			x427.free();

			// val X2324 = (X2323[1><3])(i | @) * (fc6_W)(j | @)
			JCudaTensor x428;
			JCudaMatrix x429;
			JCudaMatrix x430;
			JCudaTensor x431;
			JCudaTensor x432;
			x432 = x425;
			x431 = x432.flatten(1, new int[]{256, 6, 6});
			x429 = x431.asMatrix(1, true);
			JCudaTensor x433;
			x433 = x84;
			x430 = x433.asMatrix(1, true);
			x428 = x429.times(x430);

			// Dealloc(X2323)
			JCudaTensor x434;
			x434 = x425;
			x434.free();

			// val X2326 = (X2324 + (i) => fc6_B)
			JCudaTensor x435;
			JCudaTensor x436, x437;
			x436 = x428;
			x437 = x88;
			x435 = x437.copy(128, x436);

			// val X2327 = ReLU()(X2326)
			JCudaTensor x438;
			JCudaTensor x439;
			x439 = x435;
			x438 = x91.forward(x439);

			// val X2328 = Dropout(0.5)(X2327)
			JCudaTensor x440;
			JCudaTensor x441;
			x441 = x438;
			x440 = x94.forward(x441);

			// Dealloc(X2327)
			JCudaTensor x442;
			x442 = x438;
			x442.free();

			// val X2329 = (X2328)(i | @) * (fc7_W)(j | @)
			JCudaTensor x443;
			JCudaMatrix x444;
			JCudaMatrix x445;
			JCudaTensor x446;
			x446 = x440;
			x444 = x446.asMatrix(1, true);
			JCudaTensor x447;
			x447 = x100;
			x445 = x447.asMatrix(1, true);
			x443 = x444.times(x445);

			// Dealloc(X2328)
			JCudaTensor x448;
			x448 = x440;
			x448.free();

			// val X2331 = (X2329 + (i) => fc7_B)
			JCudaTensor x449;
			JCudaTensor x450, x451;
			x450 = x443;
			x451 = x104;
			x449 = x451.copy(128, x450);

			// val X2332 = ReLU()(X2331)
			JCudaTensor x452;
			JCudaTensor x453;
			x453 = x449;
			x452 = x91.forward(x453);

			// val X2333 = Dropout(0.5)(X2332)
			JCudaTensor x454;
			JCudaTensor x455;
			x455 = x452;
			x454 = x94.forward(x455);

			// Dealloc(X2332)
			JCudaTensor x456;
			x456 = x452;
			x456.free();

			// val X2334 = (X2333)(i | @) * (fc8_W)(j | @)
			JCudaTensor x457;
			JCudaMatrix x458;
			JCudaMatrix x459;
			JCudaTensor x460;
			x460 = x454;
			x458 = x460.asMatrix(1, true);
			JCudaTensor x461;
			x461 = x114;
			x459 = x461.asMatrix(1, true);
			x457 = x458.times(x459);

			// Dealloc(X2333)
			JCudaTensor x462;
			x462 = x454;
			x462.free();

			// val X2336 = (X2334 + (i) => fc8_B)
			JCudaTensor x463;
			JCudaTensor x464, x465;
			x464 = x457;
			x465 = x118;
			x463 = x465.copy(128, x464);

			// val X2337 = Cuda(Indicator(Y, 1000))
			JCudaTensor x466;
			JTensorFloat x467;
			x467 = x4.asIndicator(1000);
			x466 = x467.asJCudaTensor();

			// val X2338 = X2337 .* X2336
			JCudaTensor x468;
			JCudaTensor x469, x470;
			x469 = x466;
			x470 = x463;
			x468 = x469.times_i(x470);

			// val X2339 = Sum((X2338)(i15 | @))
			JCudaTensor x471;
			JCudaMatrix x472;
			JCudaTensor x473;
			x473 = x468;
			x472 = x473.asMatrix(1, true);
			x471 = x472.sum();

			// Dealloc(X2338)
			JCudaTensor x474;
			x474 = x468;
			x474.free();

			// val X2340 = Max((X2336)(i15 | @))
			JCudaTensor x475;
			JCudaMatrix x476;
			JCudaTensor x477;
			x477 = x463;
			x476 = x477.asMatrix(1, true);
			x475 = x476.max();

			// Dealloc(X2336)
			JCudaTensor x478;
			x478 = x463;
			x478.free();

			// val X2341 = 1{X2339 == X2340}
			JCudaTensor x479;
			JCudaTensor x480, x481;
			x480 = x471;
			x481 = x475;
			x479 = x480.eq(x481);

			// Dealloc(X2340)
			JCudaTensor x482;
			x482 = x475;
			x482.free();

			// Precision((Sum(X2341) / |128|))
			float x484;
			float x485;
			float x486;
			JCudaTensor x487;
			x487 = x479;
			x485 = x487.sum();
			x486 = 128;
			x484 = x485 / x486;
			System.out.println(x5 + " test precision "  + x484);
			x483 += x484;

			// Dealloc(X2341)
			JCudaTensor x488;
			x488 = x479;
			x488.free();

		}
		System.out.println();
		System.out.println("average precision: " + x483/10);
		System.out.println(); 
	}

}