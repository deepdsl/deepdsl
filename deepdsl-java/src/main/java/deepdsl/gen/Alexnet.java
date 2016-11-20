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
	static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 2);
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
	// Precision(Accuracy(1))
	static float x466;
	// V_cv1_B
	static JCudaTensor x365 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x359 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x325 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
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
	static JCudaTensor x190 = JTensor.constFloat(0.0f, 4096, 4096).asJCudaTensor();
	// V_fc8_B
	static JCudaTensor x152 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc8_W
	static JCudaTensor x156 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;
	// cv1_B
	static JCudaTensor x16 = JTensor.constFloat(0.0f, 96).load(network_dir + "/cv1_B").asJCudaTensor();
	// cv1_W
	static JCudaTensor x15 = JTensor.gaussianFloat(0.01f, 96, 3, 11, 11).load(network_dir + "/cv1_W").asJCudaTensor();
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
		x16.save(network_dir + "/cv1_B");
		x15.save(network_dir + "/cv1_W");
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
		x15.free();
		x100.free();
		x51.free();
		x34.free();
		x84.free();
		x275.free();
		x230.free();
		x152.free();
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
		x298.free();
		x325.free();
		x118.free();
		x156.free();
		x16.free();
		x245.free();
		x114.free();
		x190.free();
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
		x17.free();
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

			// val X666 = Cuda(Indicator(Y, 1000))
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x4.asIndicator(1000);
			x7 = x8.asJCudaTensor();

			// val X636 = Cuda(X)
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x3;
			x9 = x10.asJCudaTensor();

			// val X637 = Convolv(4,2)(X636,cv1_W,cv1_B)
			JCudaTensor x11;
			JCudaTensor x12, x13, x14;
			x12 = x9;
			x13 = x15;
			x14 = x16;
			x11 = x17.forward(x12, x13, x14);

			// val X348 = - X666.copy
			JCudaTensor x18;
			JCudaTensor x19;
			float x20;
			x19 = x7;
			x19 = x19.clone();
			x20 = -1;
			x18 = x19.times_i(x20);

			// val X638 = ReLU()(X637)
			JCudaTensor x21;
			JCudaTensor x22;
			x22 = x11;
			x21 = x23.forward(x22);

			// val X639 = LRN(5,1.0E-4,0.75)(X638)
			JCudaTensor x24;
			JCudaTensor x25;
			x25 = x21;
			x24 = x26.forward(x25);

			// val X640 = Pooling(3,2,0,true)(X639)
			JCudaTensor x27;
			JCudaTensor x28;
			x28 = x24;
			x27 = x29.forward(x28);

			// val X641 = Convolv(1,2)(X640,cv2_W,cv2_B)
			JCudaTensor x30;
			JCudaTensor x31, x32, x33;
			x31 = x27;
			x32 = x34;
			x33 = x35;
			x30 = x36.forward(x31, x32, x33);

			// val X642 = ReLU()(X641)
			JCudaTensor x37;
			JCudaTensor x38;
			x38 = x30;
			x37 = x39.forward(x38);

			// val X643 = LRN(5,1.0E-4,0.75)(X642)
			JCudaTensor x40;
			JCudaTensor x41;
			x41 = x37;
			x40 = x42.forward(x41);

			// val X644 = Pooling(3,2,0,true)(X643)
			JCudaTensor x43;
			JCudaTensor x44;
			x44 = x40;
			x43 = x45.forward(x44);

			// val X645 = Convolv(1,1)(X644,cv3_W,cv3_B)
			JCudaTensor x46;
			JCudaTensor x47, x48, x49;
			x47 = x43;
			x48 = x50;
			x49 = x51;
			x46 = x52.forward(x47, x48, x49);

			// val X646 = ReLU()(X645)
			JCudaTensor x53;
			JCudaTensor x54;
			x54 = x46;
			x53 = x55.forward(x54);

			// val X647 = Convolv(1,1)(X646,cv4_W,cv4_B)
			JCudaTensor x56;
			JCudaTensor x57, x58, x59;
			x57 = x53;
			x58 = x60;
			x59 = x61;
			x56 = x62.forward(x57, x58, x59);

			// val X648 = ReLU()(X647)
			JCudaTensor x63;
			JCudaTensor x64;
			x64 = x56;
			x63 = x55.forward(x64);

			// val X649 = Convolv(1,1)(X648,cv5_W,cv5_B)
			JCudaTensor x65;
			JCudaTensor x66, x67, x68;
			x66 = x63;
			x67 = x69;
			x68 = x70;
			x65 = x71.forward(x66, x67, x68);

			// val X650 = ReLU()(X649)
			JCudaTensor x72;
			JCudaTensor x73;
			x73 = x65;
			x72 = x74.forward(x73);

			// val X651 = Pooling(3,2,0,true)(X650)
			JCudaTensor x75;
			JCudaTensor x76;
			x76 = x72;
			x75 = x77.forward(x76);

			// val X652 = (X651[1><3])(i | @) * (fc6_W)(j | @)
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

			// val X654 = (X652 + (i) => fc6_B)
			JCudaTensor x85;
			JCudaTensor x86, x87;
			x86 = x78;
			x87 = x88;
			x85 = x87.copy(128, x86);

			// val X655 = ReLU()(X654)
			JCudaTensor x89;
			JCudaTensor x90;
			x90 = x85;
			x89 = x91.forward(x90);

			// val X656 = Dropout(0.5)(X655)
			JCudaTensor x92;
			JCudaTensor x93;
			x93 = x89;
			x92 = x94.forward(x93);

			// val X657 = (X656)(i | @) * (fc7_W)(j | @)
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

			// val X659 = (X657 + (i) => fc7_B)
			JCudaTensor x101;
			JCudaTensor x102, x103;
			x102 = x95;
			x103 = x104;
			x101 = x103.copy(128, x102);

			// val X660 = ReLU()(X659)
			JCudaTensor x105;
			JCudaTensor x106;
			x106 = x101;
			x105 = x91.forward(x106);

			// val X661 = Dropout(0.5)(X660)
			JCudaTensor x107;
			JCudaTensor x108;
			x108 = x105;
			x107 = x94.forward(x108);

			// val X662 = (X661)(i | @) * (fc8_W)(j | @)
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

			// val X664 = (X662 + (i) => fc8_B)
			JCudaTensor x115;
			JCudaTensor x116, x117;
			x116 = x109;
			x117 = x118;
			x115 = x117.copy(128, x116);

			// val X665 = LogSoftmax()(X664)
			JCudaTensor x119;
			JCudaTensor x120;
			x120 = x115;
			x119 = x121.forward(x120);

			// Dealloc(X664)
			JCudaTensor x122;
			x122 = x115;
			x122.free();

			// val X349 = (X348 / |128|)
			JCudaTensor x123;
			JCudaTensor x124;
			float x125;
			x124 = x18;
			float x126;
			x126 = 128;
			x125 = 1 / x126;
			x123 = x124.times_i(x125);

			// Cost(((0 - (X666 . X665)) / |128|))
			float x127;
			float x128;
			float x129;
			float x130;
			JCudaTensor x131, x132;
			x131 = x7;
			x132 = x119;
			x130 = x131.dot(x132);
			x128 = - x130;
			x129 = 128;
			x127 = x128 / x129;
			System.out.println(x5 + " " + x127);
			if (Float.isNaN(x127)) { System.exit(-1); }

			// Dealloc(X666)
			JCudaTensor x133;
			x133 = x7;
			x133.free();

			// val X380 = X349 * d_LogSoftmax()(X665)/d_X664
			JCudaTensor x134;
			JCudaTensor x135, x136;
			x135 = x123;
			x136 = x119;
			x134 = x121.backward(x135, x136);

			// Dealloc(X349)
			JCudaTensor x137;
			x137 = x123;
			x137.free();

			// Dealloc(X665)
			JCudaTensor x138;
			x138 = x119;
			x138.free();

			// val m1 = (i337) => fc8_W[@, i337]
			JCudaMatrix x139;
			JCudaTensor x140;
			x140 = x114;
			x139 = x140.asMatrix(1, false);

			// val X406 = (X380)(i336 | @) * m1
			JCudaTensor x141;
			JCudaMatrix x142;
			JCudaMatrix x143;
			JCudaTensor x144;
			x144 = x134;
			x142 = x144.asMatrix(1, true);
			x143 = x139;
			x141 = x142.times(x143);

			// val m25 = (i17) => X380[@, i17]
			JCudaMatrix x145;
			JCudaTensor x146;
			x146 = x134;
			x145 = x146.asMatrix(1, false);

			// val m27 = (i21) => X661[@, i21]
			JCudaMatrix x147;
			JCudaTensor x148;
			x148 = x107;
			x147 = x148.asMatrix(1, false);

			// val X407 = X406 * d_Dropout(0.5)()/d_X660
			JCudaTensor x149;
			JCudaTensor x150;
			x150 = x141;
			x149 = x94.backward(x150);

			// Dealloc(X406)
			JCudaTensor x151;
			x151 = x141;
			x151.free();

			// V_fc8_B <~~ Sum(m25)
			float x153, x154;
			x153 = lrn_rate_2;
			x154 = momentum;
			JCudaMatrix x155;
			x155 = x145;
			x155.sum(x152, x153, x154);

			// V_fc8_W <~~ m25 * m27
			float x157, x158;
			x157 = lrn_rate_1;
			x158 = momentum;
			JCudaMatrix x159;
			JCudaMatrix x160;
			x159 = x145;
			x160 = x147;
			x159.times(x160, x156, x157, x158);

			// Dealloc(X380)
			JCudaTensor x161;
			x161 = x134;
			x161.free();

			// Dealloc(X661)
			JCudaTensor x162;
			x162 = x107;
			x162.free();

			// fc8_B <~~ V_fc8_B
			float x163, x164;
			x163 = 1;
			x164 = 1;
			JCudaTensor x165;
			x165 = x152;
			x118.update(x165, x163, x164);

			// fc8_W <~~ V_fc8_W
			float x166, x167;
			x166 = 1;
			x167 = decay_1;
			JCudaTensor x168;
			x168 = x156;
			x114.update(x168, x166, x167);

			// val X409 = X407 * d_ReLU()(X660)/d_X659
			JCudaTensor x169;
			JCudaTensor x170, x171;
			x170 = x149;
			x171 = x105;
			x169 = x91.backward(x170, x171);

			// Dealloc(X660)
			JCudaTensor x172;
			x172 = x105;
			x172.free();

			// val m2 = (i341) => fc7_W[@, i341]
			JCudaMatrix x173;
			JCudaTensor x174;
			x174 = x100;
			x173 = x174.asMatrix(1, false);

			// val m22 = (i30) => X409[@, i30]
			JCudaMatrix x175;
			JCudaTensor x176;
			x176 = x169;
			x175 = x176.asMatrix(1, false);

			// val m24 = (i34) => X656[@, i34]
			JCudaMatrix x177;
			JCudaTensor x178;
			x178 = x92;
			x177 = x178.asMatrix(1, false);

			// val X410 = (X409)(i340 | @) * m2
			JCudaTensor x179;
			JCudaMatrix x180;
			JCudaMatrix x181;
			JCudaTensor x182;
			x182 = x169;
			x180 = x182.asMatrix(1, true);
			x181 = x173;
			x179 = x180.times(x181);

			// V_fc7_B <~~ Sum(m22)
			float x184, x185;
			x184 = lrn_rate_2;
			x185 = momentum;
			JCudaMatrix x186;
			x186 = x175;
			x186.sum(x183, x184, x185);

			// val X411 = X410 * d_Dropout(0.5)()/d_X655
			JCudaTensor x187;
			JCudaTensor x188;
			x188 = x179;
			x187 = x94.backward(x188);

			// Dealloc(X410)
			JCudaTensor x189;
			x189 = x179;
			x189.free();

			// V_fc7_W <~~ m22 * m24
			float x191, x192;
			x191 = lrn_rate_1;
			x192 = momentum;
			JCudaMatrix x193;
			JCudaMatrix x194;
			x193 = x175;
			x194 = x177;
			x193.times(x194, x190, x191, x192);

			// Dealloc(X409)
			JCudaTensor x195;
			x195 = x169;
			x195.free();

			// Dealloc(X656)
			JCudaTensor x196;
			x196 = x92;
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
			x202 = x190;
			x100.update(x202, x200, x201);

			// val X413 = X411 * d_ReLU()(X655)/d_X654
			JCudaTensor x203;
			JCudaTensor x204, x205;
			x204 = x187;
			x205 = x89;
			x203 = x91.backward(x204, x205);

			// Dealloc(X655)
			JCudaTensor x206;
			x206 = x89;
			x206.free();

			// val m3 = (i345) => fc6_W[@, i345]
			JCudaMatrix x207;
			JCudaTensor x208;
			x208 = x84;
			x207 = x208.asMatrix(1, false);

			// val m19 = (i57) => X413[@, i57]
			JCudaMatrix x209;
			JCudaTensor x210;
			x210 = x203;
			x209 = x210.asMatrix(1, false);

			// val m21 = (i61) => X651[1><3][@, i61]
			JCudaMatrix x211;
			JCudaTensor x212;
			JCudaTensor x213;
			x213 = x75;
			x212 = x213.flatten(1, new int[]{256, 6, 6});
			x211 = x212.asMatrix(1, false);

			// val X414 = (X413)(i344 | @) * m3
			JCudaTensor x214;
			JCudaMatrix x215;
			JCudaMatrix x216;
			JCudaTensor x217;
			x217 = x203;
			x215 = x217.asMatrix(1, true);
			x216 = x207;
			x214 = x215.times(x216);

			// V_fc6_W <~~ m19 * m21
			float x219, x220;
			x219 = lrn_rate_1;
			x220 = momentum;
			JCudaMatrix x221;
			JCudaMatrix x222;
			x221 = x209;
			x222 = x211;
			x221.times(x222, x218, x219, x220);

			// val X416 = X414[1<>3] * d_Pooling(3,2,0,true)(X651,X650)/d_X650
			JCudaTensor x223;
			JCudaTensor x224, x225, x226;
			JCudaTensor x227;
			x227 = x214;
			x224 = x227.unflatten(1, new int[]{256, 6, 6});
			x225 = x75;
			x226 = x72;
			x223 = x77.backward(x224, x225, x226);

			// Dealloc(X414)
			JCudaTensor x228;
			x228 = x214;
			x228.free();

			// Dealloc(X651)
			JCudaTensor x229;
			x229 = x75;
			x229.free();

			// V_fc6_B <~~ Sum(m19)
			float x231, x232;
			x231 = lrn_rate_2;
			x232 = momentum;
			JCudaMatrix x233;
			x233 = x209;
			x233.sum(x230, x231, x232);

			// Dealloc(X413)
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

			// val X418 = X416 * d_ReLU()(X650)/d_X649
			JCudaTensor x241;
			JCudaTensor x242, x243;
			x242 = x223;
			x243 = x72;
			x241 = x74.backward(x242, x243);

			// Dealloc(X650)
			JCudaTensor x244;
			x244 = x72;
			x244.free();

			// V_cv5_B <~~ X418 * d_Convolv(1,1)()/d_cv5_B
			float x246, x247;
			x246 = lrn_rate_2;
			x247 = momentum;
			JCudaTensor x248;
			x248 = x241;
			x71.backward_bias(x248, x245, x246, x247);

			// val X419 = X418 * d_Convolv(1,1)(cv5_W)/d_X648
			JCudaTensor x249;
			JCudaTensor x250, x251;
			x250 = x241;
			x251 = x69;
			x249 = x71.backward_data(x250, x251);

			// V_cv5_W <~~ X418 * d_Convolv(1,1)(X648)/d_cv5_W
			float x253, x254;
			x253 = lrn_rate_1;
			x254 = momentum;
			JCudaTensor x255, x256;
			x255 = x241;
			x256 = x63;
			x71.backward_filter(x255, x256, x252, x253, x254);

			// Dealloc(X418)
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

			// val X421 = X419 * d_ReLU()(X648)/d_X647
			JCudaTensor x264;
			JCudaTensor x265, x266;
			x265 = x249;
			x266 = x63;
			x264 = x55.backward(x265, x266);

			// Dealloc(X648)
			JCudaTensor x267;
			x267 = x63;
			x267.free();

			// val X422 = X421 * d_Convolv(1,1)(cv4_W)/d_X646
			JCudaTensor x268;
			JCudaTensor x269, x270;
			x269 = x264;
			x270 = x60;
			x268 = x62.backward_data(x269, x270);

			// V_cv4_B <~~ X421 * d_Convolv(1,1)()/d_cv4_B
			float x272, x273;
			x272 = lrn_rate_2;
			x273 = momentum;
			JCudaTensor x274;
			x274 = x264;
			x62.backward_bias(x274, x271, x272, x273);

			// V_cv4_W <~~ X421 * d_Convolv(1,1)(X646)/d_cv4_W
			float x276, x277;
			x276 = lrn_rate_1;
			x277 = momentum;
			JCudaTensor x278, x279;
			x278 = x264;
			x279 = x53;
			x62.backward_filter(x278, x279, x275, x276, x277);

			// Dealloc(X421)
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

			// val X424 = X422 * d_ReLU()(X646)/d_X645
			JCudaTensor x287;
			JCudaTensor x288, x289;
			x288 = x268;
			x289 = x53;
			x287 = x55.backward(x288, x289);

			// Dealloc(X646)
			JCudaTensor x290;
			x290 = x53;
			x290.free();

			// V_cv3_B <~~ X424 * d_Convolv(1,1)()/d_cv3_B
			float x292, x293;
			x292 = lrn_rate_2;
			x293 = momentum;
			JCudaTensor x294;
			x294 = x287;
			x52.backward_bias(x294, x291, x292, x293);

			// val X425 = X424 * d_Convolv(1,1)(cv3_W)/d_X644
			JCudaTensor x295;
			JCudaTensor x296, x297;
			x296 = x287;
			x297 = x50;
			x295 = x52.backward_data(x296, x297);

			// V_cv3_W <~~ X424 * d_Convolv(1,1)(X644)/d_cv3_W
			float x299, x300;
			x299 = lrn_rate_1;
			x300 = momentum;
			JCudaTensor x301, x302;
			x301 = x287;
			x302 = x43;
			x52.backward_filter(x301, x302, x298, x299, x300);

			// Dealloc(X424)
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

			// val X427 = X425 * d_Pooling(3,2,0,true)(X644,X643)/d_X643
			JCudaTensor x310;
			JCudaTensor x311, x312, x313;
			x311 = x295;
			x312 = x43;
			x313 = x40;
			x310 = x45.backward(x311, x312, x313);

			// Dealloc(X425)
			JCudaTensor x314;
			x314 = x295;
			x314.free();

			// Dealloc(X644)
			JCudaTensor x315;
			x315 = x43;
			x315.free();

			// val X429 = X427 * d_LRN(5,1.0E-4,0.75)(X643,X642)/d_X642
			JCudaTensor x316;
			JCudaTensor x317, x318, x319;
			x317 = x310;
			x318 = x40;
			x319 = x37;
			x316 = x42.backward(x317, x318, x319);

			// Dealloc(X643)
			JCudaTensor x320;
			x320 = x40;
			x320.free();

			// val X431 = X429 * d_ReLU()(X642)/d_X641
			JCudaTensor x321;
			JCudaTensor x322, x323;
			x322 = x316;
			x323 = x37;
			x321 = x39.backward(x322, x323);

			// Dealloc(X642)
			JCudaTensor x324;
			x324 = x37;
			x324.free();

			// V_cv2_B <~~ X431 * d_Convolv(1,2)()/d_cv2_B
			float x326, x327;
			x326 = lrn_rate_2;
			x327 = momentum;
			JCudaTensor x328;
			x328 = x321;
			x36.backward_bias(x328, x325, x326, x327);

			// val X432 = X431 * d_Convolv(1,2)(cv2_W)/d_X640
			JCudaTensor x329;
			JCudaTensor x330, x331;
			x330 = x321;
			x331 = x34;
			x329 = x36.backward_data(x330, x331);

			// V_cv2_W <~~ X431 * d_Convolv(1,2)(X640)/d_cv2_W
			float x333, x334;
			x333 = lrn_rate_1;
			x334 = momentum;
			JCudaTensor x335, x336;
			x335 = x321;
			x336 = x27;
			x36.backward_filter(x335, x336, x332, x333, x334);

			// Dealloc(X431)
			JCudaTensor x337;
			x337 = x321;
			x337.free();

			// cv2_B <~~ V_cv2_B
			float x338, x339;
			x338 = 1;
			x339 = 1;
			JCudaTensor x340;
			x340 = x325;
			x35.update(x340, x338, x339);

			// cv2_W <~~ V_cv2_W
			float x341, x342;
			x341 = 1;
			x342 = decay_1;
			JCudaTensor x343;
			x343 = x332;
			x34.update(x343, x341, x342);

			// val X434 = X432 * d_Pooling(3,2,0,true)(X640,X639)/d_X639
			JCudaTensor x344;
			JCudaTensor x345, x346, x347;
			x345 = x329;
			x346 = x27;
			x347 = x24;
			x344 = x29.backward(x345, x346, x347);

			// Dealloc(X432)
			JCudaTensor x348;
			x348 = x329;
			x348.free();

			// Dealloc(X640)
			JCudaTensor x349;
			x349 = x27;
			x349.free();

			// val X436 = X434 * d_LRN(5,1.0E-4,0.75)(X639,X638)/d_X638
			JCudaTensor x350;
			JCudaTensor x351, x352, x353;
			x351 = x344;
			x352 = x24;
			x353 = x21;
			x350 = x26.backward(x351, x352, x353);

			// Dealloc(X639)
			JCudaTensor x354;
			x354 = x24;
			x354.free();

			// val X438 = X436 * d_ReLU()(X638)/d_X637
			JCudaTensor x355;
			JCudaTensor x356, x357;
			x356 = x350;
			x357 = x21;
			x355 = x23.backward(x356, x357);

			// Dealloc(X638)
			JCudaTensor x358;
			x358 = x21;
			x358.free();

			// V_cv1_W <~~ X438 * d_Convolv(4,2)(X636)/d_cv1_W
			float x360, x361;
			x360 = lrn_rate_1;
			x361 = momentum;
			JCudaTensor x362, x363;
			x362 = x355;
			x363 = x9;
			x17.backward_filter(x362, x363, x359, x360, x361);

			// Dealloc(X636)
			JCudaTensor x364;
			x364 = x9;
			x364.free();

			// V_cv1_B <~~ X438 * d_Convolv(4,2)()/d_cv1_B
			float x366, x367;
			x366 = lrn_rate_2;
			x367 = momentum;
			JCudaTensor x368;
			x368 = x355;
			x17.backward_bias(x368, x365, x366, x367);

			// Dealloc(X438)
			JCudaTensor x369;
			x369 = x355;
			x369.free();

			// cv1_W <~~ V_cv1_W
			float x370, x371;
			x370 = 1;
			x371 = decay_1;
			JCudaTensor x372;
			x372 = x359;
			x15.update(x372, x370, x371);

			// cv1_B <~~ V_cv1_B
			float x373, x374;
			x373 = 1;
			x374 = 1;
			JCudaTensor x375;
			x375 = x365;
			x16.update(x375, x373, x374);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X667 = Cuda(X)
			JCudaTensor x376;
			JTensorFloat x377;
			x377 = x3;
			x376 = x377.asJCudaTensor();

			// val X668 = Convolv(4,2)(X667,cv1_W,cv1_B)
			JCudaTensor x378;
			JCudaTensor x379, x380, x381;
			x379 = x376;
			x380 = x15;
			x381 = x16;
			x378 = x17.forward(x379, x380, x381);

			// Dealloc(X667)
			JCudaTensor x382;
			x382 = x376;
			x382.free();

			// val X669 = ReLU()(X668)
			JCudaTensor x383;
			JCudaTensor x384;
			x384 = x378;
			x383 = x23.forward(x384);

			// val X670 = LRN(5,1.0E-4,0.75)(X669)
			JCudaTensor x385;
			JCudaTensor x386;
			x386 = x383;
			x385 = x26.forward(x386);

			// Dealloc(X669)
			JCudaTensor x387;
			x387 = x383;
			x387.free();

			// val X671 = Pooling(3,2,0,true)(X670)
			JCudaTensor x388;
			JCudaTensor x389;
			x389 = x385;
			x388 = x29.forward(x389);

			// Dealloc(X670)
			JCudaTensor x390;
			x390 = x385;
			x390.free();

			// val X672 = Convolv(1,2)(X671,cv2_W,cv2_B)
			JCudaTensor x391;
			JCudaTensor x392, x393, x394;
			x392 = x388;
			x393 = x34;
			x394 = x35;
			x391 = x36.forward(x392, x393, x394);

			// Dealloc(X671)
			JCudaTensor x395;
			x395 = x388;
			x395.free();

			// val X673 = ReLU()(X672)
			JCudaTensor x396;
			JCudaTensor x397;
			x397 = x391;
			x396 = x39.forward(x397);

			// val X674 = LRN(5,1.0E-4,0.75)(X673)
			JCudaTensor x398;
			JCudaTensor x399;
			x399 = x396;
			x398 = x42.forward(x399);

			// Dealloc(X673)
			JCudaTensor x400;
			x400 = x396;
			x400.free();

			// val X675 = Pooling(3,2,0,true)(X674)
			JCudaTensor x401;
			JCudaTensor x402;
			x402 = x398;
			x401 = x45.forward(x402);

			// Dealloc(X674)
			JCudaTensor x403;
			x403 = x398;
			x403.free();

			// val X676 = Convolv(1,1)(X675,cv3_W,cv3_B)
			JCudaTensor x404;
			JCudaTensor x405, x406, x407;
			x405 = x401;
			x406 = x50;
			x407 = x51;
			x404 = x52.forward(x405, x406, x407);

			// Dealloc(X675)
			JCudaTensor x408;
			x408 = x401;
			x408.free();

			// val X677 = ReLU()(X676)
			JCudaTensor x409;
			JCudaTensor x410;
			x410 = x404;
			x409 = x55.forward(x410);

			// val X678 = Convolv(1,1)(X677,cv4_W,cv4_B)
			JCudaTensor x411;
			JCudaTensor x412, x413, x414;
			x412 = x409;
			x413 = x60;
			x414 = x61;
			x411 = x62.forward(x412, x413, x414);

			// Dealloc(X677)
			JCudaTensor x415;
			x415 = x409;
			x415.free();

			// val X679 = ReLU()(X678)
			JCudaTensor x416;
			JCudaTensor x417;
			x417 = x411;
			x416 = x55.forward(x417);

			// val X680 = Convolv(1,1)(X679,cv5_W,cv5_B)
			JCudaTensor x418;
			JCudaTensor x419, x420, x421;
			x419 = x416;
			x420 = x69;
			x421 = x70;
			x418 = x71.forward(x419, x420, x421);

			// Dealloc(X679)
			JCudaTensor x422;
			x422 = x416;
			x422.free();

			// val X681 = ReLU()(X680)
			JCudaTensor x423;
			JCudaTensor x424;
			x424 = x418;
			x423 = x74.forward(x424);

			// val X682 = Pooling(3,2,0,true)(X681)
			JCudaTensor x425;
			JCudaTensor x426;
			x426 = x423;
			x425 = x77.forward(x426);

			// Dealloc(X681)
			JCudaTensor x427;
			x427 = x423;
			x427.free();

			// val X683 = (X682[1><3])(i | @) * (fc6_W)(j | @)
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

			// Dealloc(X682)
			JCudaTensor x434;
			x434 = x425;
			x434.free();

			// val X685 = (X683 + (i) => fc6_B)
			JCudaTensor x435;
			JCudaTensor x436, x437;
			x436 = x428;
			x437 = x88;
			x435 = x437.copy(128, x436);

			// val X686 = ReLU()(X685)
			JCudaTensor x438;
			JCudaTensor x439;
			x439 = x435;
			x438 = x91.forward(x439);

			// val X687 = Dropout(0.5)(X686)
			JCudaTensor x440;
			JCudaTensor x441;
			x441 = x438;
			x440 = x94.forward(x441);

			// Dealloc(X686)
			JCudaTensor x442;
			x442 = x438;
			x442.free();

			// val X688 = (X687)(i | @) * (fc7_W)(j | @)
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

			// Dealloc(X687)
			JCudaTensor x448;
			x448 = x440;
			x448.free();

			// val X690 = (X688 + (i) => fc7_B)
			JCudaTensor x449;
			JCudaTensor x450, x451;
			x450 = x443;
			x451 = x104;
			x449 = x451.copy(128, x450);

			// val X691 = ReLU()(X690)
			JCudaTensor x452;
			JCudaTensor x453;
			x453 = x449;
			x452 = x91.forward(x453);

			// val X692 = Dropout(0.5)(X691)
			JCudaTensor x454;
			JCudaTensor x455;
			x455 = x452;
			x454 = x94.forward(x455);

			// Dealloc(X691)
			JCudaTensor x456;
			x456 = x452;
			x456.free();

			// val X693 = (X692)(i | @) * (fc8_W)(j | @)
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

			// Dealloc(X692)
			JCudaTensor x462;
			x462 = x454;
			x462.free();

			// val X695 = (X693 + (i) => fc8_B)
			JCudaTensor x463;
			JCudaTensor x464, x465;
			x464 = x457;
			x465 = x118;
			x463 = x465.copy(128, x464);

			// Precision(Accuracy(1))
			float x467;
			JCudaTensor x468;
			JTensorFloat x469;
			x468 = x463;
			x469 = x4;
			x467 = x468.accuracy(x469, 1);
			System.out.println(x5 + " test precision "  + x467);
			x466 += x467;

			// Dealloc(X695)
			JCudaTensor x470;
			x470 = x463;
			x470.free();

		}
		System.out.println();
		System.out.println("average precision: " + x466/10);
		System.out.println(); 
	}

}