package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Alexnet {
	static{
		// comment the first or both lines below for memory efficient mode
		JCudaTensor.enableMemoryCache();
		JCudaTensor.enableWorkspaceCache();
	}
	// decay
	static float decay = 5.0E-4f;
	// lrn_rate
	static float lrn_rate = -0.01f;
	// momentum
	static float momentum = 0.1f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/alexnet";
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
	// (LMDB,false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, 1000, true);
	// (LMDB,true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, 1000, false);
	// (LRN(5,1.0E-4,0.75),List(List(128, 256, 27, 27)))
	static JCudnnLRN x42 = new JCudnnLRN(new int[]{128,256,27,27}, 5, 1.0E-4, 0.75);
	// (LRN(5,1.0E-4,0.75),List(List(128, 96, 55, 55)))
	static JCudnnLRN x26 = new JCudnnLRN(new int[]{128,96,55,55}, 5, 1.0E-4, 0.75);
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
	// Precision(Accuracy(X699, Y, 1))
	static float x562;
	// V_cv1_B
	static JCudaTensor x451 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x443 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x407 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x400 = JTensor.constFloat(0.0f, 256, 96, 5, 5).asJCudaTensor();
	// V_cv3_B
	static JCudaTensor x351 = JTensor.constFloat(0.0f, 384).asJCudaTensor();
	// V_cv3_W
	static JCudaTensor x360 = JTensor.constFloat(0.0f, 384, 256, 3, 3).asJCudaTensor();
	// V_cv4_B
	static JCudaTensor x326 = JTensor.constFloat(0.0f, 384).asJCudaTensor();
	// V_cv4_W
	static JCudaTensor x319 = JTensor.constFloat(0.0f, 384, 384, 3, 3).asJCudaTensor();
	// V_cv5_B
	static JCudaTensor x281 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// V_cv5_W
	static JCudaTensor x290 = JTensor.constFloat(0.0f, 256, 384, 3, 3).asJCudaTensor();
	// V_fc6_B
	static JCudaTensor x256 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
	// V_fc6_W
	static JCudaTensor x242 = JTensor.constFloat(0.0f, 4096, 9216).asJCudaTensor();
	// V_fc7_B
	static JCudaTensor x195 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
	// V_fc7_W
	static JCudaTensor x201 = JTensor.constFloat(0.0f, 4096, 4096).asJCudaTensor();
	// V_fc8_B
	static JCudaTensor x152 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc8_W
	static JCudaTensor x158 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
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
		x407.free();
		x15.free();
		x100.free();
		x242.free();
		x351.free();
		x51.free();
		x34.free();
		x84.free();
		x451.free();
		x152.free();
		x319.free();
		x35.free();
		x50.free();
		x70.free();
		x195.free();
		x256.free();
		x201.free();
		x400.free();
		x290.free();
		x60.free();
		x88.free();
		x104.free();
		x326.free();
		x443.free();
		x69.free();
		x118.free();
		x281.free();
		x16.free();
		x114.free();
		x360.free();
		x158.free();
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

			// val X670 = Cuda(Indicator(Y, 1000))
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x4.asIndicator(1000);
			x7 = x8.asJCudaTensor();

			// val X640 = Cuda(X)
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x3;
			x9 = x10.asJCudaTensor();

			// val X641 = Convolv(4,2)(X640,cv1_W,cv1_B)
			JCudaTensor x11;
			JCudaTensor x12, x13, x14;
			x12 = x9;
			x13 = x15;
			x14 = x16;
			x11 = x17.forward(x12, x13, x14);

			// val X352 = - X670.copy
			JCudaTensor x18;
			JCudaTensor x19;
			float x20;
			x19 = x7;
			x19 = x19.clone();
			x20 = -1;
			x18 = x19.times_i(x20);

			// val X642 = ReLU()(X641)
			JCudaTensor x21;
			JCudaTensor x22;
			x22 = x11;
			x21 = x23.forward(x22);

			// val X643 = LRN(5,1.0E-4,0.75)(X642)
			JCudaTensor x24;
			JCudaTensor x25;
			x25 = x21;
			x24 = x26.forward(x25);

			// val X644 = Pooling(3,2,0,true)(X643)
			JCudaTensor x27;
			JCudaTensor x28;
			x28 = x24;
			x27 = x29.forward(x28);

			// val X645 = Convolv(1,2)(X644,cv2_W,cv2_B)
			JCudaTensor x30;
			JCudaTensor x31, x32, x33;
			x31 = x27;
			x32 = x34;
			x33 = x35;
			x30 = x36.forward(x31, x32, x33);

			// val X646 = ReLU()(X645)
			JCudaTensor x37;
			JCudaTensor x38;
			x38 = x30;
			x37 = x39.forward(x38);

			// val X647 = LRN(5,1.0E-4,0.75)(X646)
			JCudaTensor x40;
			JCudaTensor x41;
			x41 = x37;
			x40 = x42.forward(x41);

			// val X648 = Pooling(3,2,0,true)(X647)
			JCudaTensor x43;
			JCudaTensor x44;
			x44 = x40;
			x43 = x45.forward(x44);

			// val X649 = Convolv(1,1)(X648,cv3_W,cv3_B)
			JCudaTensor x46;
			JCudaTensor x47, x48, x49;
			x47 = x43;
			x48 = x50;
			x49 = x51;
			x46 = x52.forward(x47, x48, x49);

			// val X650 = ReLU()(X649)
			JCudaTensor x53;
			JCudaTensor x54;
			x54 = x46;
			x53 = x55.forward(x54);

			// val X651 = Convolv(1,1)(X650,cv4_W,cv4_B)
			JCudaTensor x56;
			JCudaTensor x57, x58, x59;
			x57 = x53;
			x58 = x60;
			x59 = x61;
			x56 = x62.forward(x57, x58, x59);

			// val X652 = ReLU()(X651)
			JCudaTensor x63;
			JCudaTensor x64;
			x64 = x56;
			x63 = x55.forward(x64);

			// val X653 = Convolv(1,1)(X652,cv5_W,cv5_B)
			JCudaTensor x65;
			JCudaTensor x66, x67, x68;
			x66 = x63;
			x67 = x69;
			x68 = x70;
			x65 = x71.forward(x66, x67, x68);

			// val X654 = ReLU()(X653)
			JCudaTensor x72;
			JCudaTensor x73;
			x73 = x65;
			x72 = x74.forward(x73);

			// val X655 = Pooling(3,2,0,true)(X654)
			JCudaTensor x75;
			JCudaTensor x76;
			x76 = x72;
			x75 = x77.forward(x76);

			// val X656 = (X655[1><3])(i10 | @) * (fc6_W)(i11 | @)
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

			// val X658 = (X656 + (i10) => fc6_B)
			JCudaTensor x85;
			JCudaTensor x86, x87;
			x86 = x78;
			x87 = x88;
			x85 = x87.copy(128, x86);

			// val X659 = ReLU()(X658)
			JCudaTensor x89;
			JCudaTensor x90;
			x90 = x85;
			x89 = x91.forward(x90);

			// val X660 = Dropout(0.5)(X659)
			JCudaTensor x92;
			JCudaTensor x93;
			x93 = x89;
			x92 = x94.forward(x93);

			// val X661 = (X660)(i13 | @) * (fc7_W)(i14 | @)
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

			// val X663 = (X661 + (i13) => fc7_B)
			JCudaTensor x101;
			JCudaTensor x102, x103;
			x102 = x95;
			x103 = x104;
			x101 = x103.copy(128, x102);

			// val X664 = ReLU()(X663)
			JCudaTensor x105;
			JCudaTensor x106;
			x106 = x101;
			x105 = x91.forward(x106);

			// val X665 = Dropout(0.5)(X664)
			JCudaTensor x107;
			JCudaTensor x108;
			x108 = x105;
			x107 = x94.forward(x108);

			// val X666 = (X665)(i16 | @) * (fc8_W)(i17 | @)
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

			// val X668 = (X666 + (i16) => fc8_B)
			JCudaTensor x115;
			JCudaTensor x116, x117;
			x116 = x109;
			x117 = x118;
			x115 = x117.copy(128, x116);

			// val X669 = LogSoftmax()(X668)
			JCudaTensor x119;
			JCudaTensor x120;
			x120 = x115;
			x119 = x121.forward(x120);

			// Dealloc(X668)
			JCudaTensor x122;
			x122 = x115;
			x122.free();

			// val X353 = (X352 / |128|)
			JCudaTensor x123;
			JCudaTensor x124;
			float x125;
			x124 = x18;
			float x126;
			x126 = 128;
			x125 = 1 / x126;
			x123 = x124.times_i(x125);

			// Cost(((0 - (X670 . X669)) / |128|))
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

			// Dealloc(X670)
			JCudaTensor x133;
			x133 = x7;
			x133.free();

			// val X384 = X353 * d_LogSoftmax()(X669)/d_X668
			JCudaTensor x134;
			JCudaTensor x135, x136;
			x135 = x123;
			x136 = x119;
			x134 = x121.backward(x135, x136);

			// Dealloc(X353)
			JCudaTensor x137;
			x137 = x123;
			x137.free();

			// Dealloc(X669)
			JCudaTensor x138;
			x138 = x119;
			x138.free();

			// val m1 = (i343) => fc8_W[@, i343]
			JCudaMatrix x139;
			JCudaTensor x140;
			x140 = x114;
			x139 = x140.asMatrix(1, false);

			// val X410 = (X384)(i342 | @) * m1
			JCudaTensor x141;
			JCudaMatrix x142;
			JCudaMatrix x143;
			JCudaTensor x144;
			x144 = x134;
			x142 = x144.asMatrix(1, true);
			x143 = x139;
			x141 = x142.times(x143);

			// val m25 = (i23) => X384[@, i23]
			JCudaMatrix x145;
			JCudaTensor x146;
			x146 = x134;
			x145 = x146.asMatrix(1, false);

			// val m27 = (i27) => X665[@, i27]
			JCudaMatrix x147;
			JCudaTensor x148;
			x148 = x107;
			x147 = x148.asMatrix(1, false);

			// val X411 = X410 * d_Dropout(0.5)()/d_X664
			JCudaTensor x149;
			JCudaTensor x150;
			x150 = x141;
			x149 = x94.backward(x150);

			// Dealloc(X410)
			JCudaTensor x151;
			x151 = x141;
			x151.free();

			// V_fc8_B <~~ Sum(m25)
			float x153, x154;
			float x155;
			float x156;
			x155 = 2;
			x156 = lrn_rate;
			x153 = x155 * x156;
			x154 = momentum;
			JCudaMatrix x157;
			x157 = x145;
			x157.sum(x152, x153, x154);

			// V_fc8_W <~~ m25 * m27
			float x159, x160;
			float x161;
			float x162;
			x161 = 1;
			x162 = lrn_rate;
			x159 = x161 * x162;
			x160 = momentum;
			JCudaMatrix x163;
			JCudaMatrix x164;
			x163 = x145;
			x164 = x147;
			x163.times(x164, x158, x159, x160);

			// Dealloc(X384)
			JCudaTensor x165;
			x165 = x134;
			x165.free();

			// Dealloc(X665)
			JCudaTensor x166;
			x166 = x107;
			x166.free();

			// fc8_B <~~ V_fc8_B
			float x167, x168;
			x167 = 1;
			x168 = 1;
			JCudaTensor x169;
			x169 = x152;
			x118.update(x169, x167, x168);

			// fc8_W <~~ V_fc8_W
			float x170, x171;
			x170 = 1;
			float x172;
			float x173;
			x172 = 1;
			float x174;
			float x175;
			float x176;
			float x177;
			x176 = 1;
			x177 = decay;
			x174 = x176 * x177;
			float x178;
			float x179;
			x178 = 1;
			x179 = lrn_rate;
			x175 = x178 * x179;
			x173 = x174 * x175;
			x171 = x172 + x173;
			JCudaTensor x180;
			x180 = x158;
			x114.update(x180, x170, x171);

			// val X413 = X411 * d_ReLU()(X664)/d_X663
			JCudaTensor x181;
			JCudaTensor x182, x183;
			x182 = x149;
			x183 = x105;
			x181 = x91.backward(x182, x183);

			// Dealloc(X664)
			JCudaTensor x184;
			x184 = x105;
			x184.free();

			// val m2 = (i347) => fc7_W[@, i347]
			JCudaMatrix x185;
			JCudaTensor x186;
			x186 = x100;
			x185 = x186.asMatrix(1, false);

			// val m22 = (i36) => X413[@, i36]
			JCudaMatrix x187;
			JCudaTensor x188;
			x188 = x181;
			x187 = x188.asMatrix(1, false);

			// val m24 = (i40) => X660[@, i40]
			JCudaMatrix x189;
			JCudaTensor x190;
			x190 = x92;
			x189 = x190.asMatrix(1, false);

			// val X414 = (X413)(i346 | @) * m2
			JCudaTensor x191;
			JCudaMatrix x192;
			JCudaMatrix x193;
			JCudaTensor x194;
			x194 = x181;
			x192 = x194.asMatrix(1, true);
			x193 = x185;
			x191 = x192.times(x193);

			// V_fc7_B <~~ Sum(m22)
			float x196, x197;
			float x198;
			float x199;
			x198 = 2;
			x199 = lrn_rate;
			x196 = x198 * x199;
			x197 = momentum;
			JCudaMatrix x200;
			x200 = x187;
			x200.sum(x195, x196, x197);

			// V_fc7_W <~~ m22 * m24
			float x202, x203;
			float x204;
			float x205;
			x204 = 1;
			x205 = lrn_rate;
			x202 = x204 * x205;
			x203 = momentum;
			JCudaMatrix x206;
			JCudaMatrix x207;
			x206 = x187;
			x207 = x189;
			x206.times(x207, x201, x202, x203);

			// Dealloc(X413)
			JCudaTensor x208;
			x208 = x181;
			x208.free();

			// Dealloc(X660)
			JCudaTensor x209;
			x209 = x92;
			x209.free();

			// val X415 = X414 * d_Dropout(0.5)()/d_X659
			JCudaTensor x210;
			JCudaTensor x211;
			x211 = x191;
			x210 = x94.backward(x211);

			// Dealloc(X414)
			JCudaTensor x212;
			x212 = x191;
			x212.free();

			// fc7_B <~~ V_fc7_B
			float x213, x214;
			x213 = 1;
			x214 = 1;
			JCudaTensor x215;
			x215 = x195;
			x104.update(x215, x213, x214);

			// fc7_W <~~ V_fc7_W
			float x216, x217;
			x216 = 1;
			float x218;
			float x219;
			x218 = 1;
			float x220;
			float x221;
			float x222;
			float x223;
			x222 = 1;
			x223 = decay;
			x220 = x222 * x223;
			float x224;
			float x225;
			x224 = 1;
			x225 = lrn_rate;
			x221 = x224 * x225;
			x219 = x220 * x221;
			x217 = x218 + x219;
			JCudaTensor x226;
			x226 = x201;
			x100.update(x226, x216, x217);

			// val X417 = X415 * d_ReLU()(X659)/d_X658
			JCudaTensor x227;
			JCudaTensor x228, x229;
			x228 = x210;
			x229 = x89;
			x227 = x91.backward(x228, x229);

			// Dealloc(X659)
			JCudaTensor x230;
			x230 = x89;
			x230.free();

			// val m3 = (i351) => fc6_W[@, i351]
			JCudaMatrix x231;
			JCudaTensor x232;
			x232 = x84;
			x231 = x232.asMatrix(1, false);

			// val m19 = (i63) => X417[@, i63]
			JCudaMatrix x233;
			JCudaTensor x234;
			x234 = x227;
			x233 = x234.asMatrix(1, false);

			// val m21 = (i67) => X655[1><3][@, i67]
			JCudaMatrix x235;
			JCudaTensor x236;
			JCudaTensor x237;
			x237 = x75;
			x236 = x237.flatten(1, new int[]{256, 6, 6});
			x235 = x236.asMatrix(1, false);

			// val X418 = (X417)(i350 | @) * m3
			JCudaTensor x238;
			JCudaMatrix x239;
			JCudaMatrix x240;
			JCudaTensor x241;
			x241 = x227;
			x239 = x241.asMatrix(1, true);
			x240 = x231;
			x238 = x239.times(x240);

			// V_fc6_W <~~ m19 * m21
			float x243, x244;
			float x245;
			float x246;
			x245 = 1;
			x246 = lrn_rate;
			x243 = x245 * x246;
			x244 = momentum;
			JCudaMatrix x247;
			JCudaMatrix x248;
			x247 = x233;
			x248 = x235;
			x247.times(x248, x242, x243, x244);

			// val X420 = X418[1<>3] * d_Pooling(3,2,0,true)(X655,X654)/d_X654
			JCudaTensor x249;
			JCudaTensor x250, x251, x252;
			JCudaTensor x253;
			x253 = x238;
			x250 = x253.unflatten(1, new int[]{256, 6, 6});
			x251 = x75;
			x252 = x72;
			x249 = x77.backward(x250, x251, x252);

			// Dealloc(X418)
			JCudaTensor x254;
			x254 = x238;
			x254.free();

			// Dealloc(X655)
			JCudaTensor x255;
			x255 = x75;
			x255.free();

			// V_fc6_B <~~ Sum(m19)
			float x257, x258;
			float x259;
			float x260;
			x259 = 2;
			x260 = lrn_rate;
			x257 = x259 * x260;
			x258 = momentum;
			JCudaMatrix x261;
			x261 = x233;
			x261.sum(x256, x257, x258);

			// Dealloc(X417)
			JCudaTensor x262;
			x262 = x227;
			x262.free();

			// fc6_W <~~ V_fc6_W
			float x263, x264;
			x263 = 1;
			float x265;
			float x266;
			x265 = 1;
			float x267;
			float x268;
			float x269;
			float x270;
			x269 = 1;
			x270 = decay;
			x267 = x269 * x270;
			float x271;
			float x272;
			x271 = 1;
			x272 = lrn_rate;
			x268 = x271 * x272;
			x266 = x267 * x268;
			x264 = x265 + x266;
			JCudaTensor x273;
			x273 = x242;
			x84.update(x273, x263, x264);

			// fc6_B <~~ V_fc6_B
			float x274, x275;
			x274 = 1;
			x275 = 1;
			JCudaTensor x276;
			x276 = x256;
			x88.update(x276, x274, x275);

			// val X422 = X420 * d_ReLU()(X654)/d_X653
			JCudaTensor x277;
			JCudaTensor x278, x279;
			x278 = x249;
			x279 = x72;
			x277 = x74.backward(x278, x279);

			// Dealloc(X654)
			JCudaTensor x280;
			x280 = x72;
			x280.free();

			// V_cv5_B <~~ X422 * d_Convolv(1,1)()/d_cv5_B
			float x282, x283;
			float x284;
			float x285;
			x284 = 2;
			x285 = lrn_rate;
			x282 = x284 * x285;
			x283 = momentum;
			JCudaTensor x286;
			x286 = x277;
			x71.backward_bias(x286, x281, x282, x283);

			// val X423 = X422 * d_Convolv(1,1)(cv5_W)/d_X652
			JCudaTensor x287;
			JCudaTensor x288, x289;
			x288 = x277;
			x289 = x69;
			x287 = x71.backward_data(x288, x289);

			// V_cv5_W <~~ X422 * d_Convolv(1,1)(X652)/d_cv5_W
			float x291, x292;
			float x293;
			float x294;
			x293 = 1;
			x294 = lrn_rate;
			x291 = x293 * x294;
			x292 = momentum;
			JCudaTensor x295, x296;
			x295 = x277;
			x296 = x63;
			x71.backward_filter(x295, x296, x290, x291, x292);

			// Dealloc(X422)
			JCudaTensor x297;
			x297 = x277;
			x297.free();

			// cv5_B <~~ V_cv5_B
			float x298, x299;
			x298 = 1;
			x299 = 1;
			JCudaTensor x300;
			x300 = x281;
			x70.update(x300, x298, x299);

			// cv5_W <~~ V_cv5_W
			float x301, x302;
			x301 = 1;
			float x303;
			float x304;
			x303 = 1;
			float x305;
			float x306;
			float x307;
			float x308;
			x307 = 1;
			x308 = decay;
			x305 = x307 * x308;
			float x309;
			float x310;
			x309 = 1;
			x310 = lrn_rate;
			x306 = x309 * x310;
			x304 = x305 * x306;
			x302 = x303 + x304;
			JCudaTensor x311;
			x311 = x290;
			x69.update(x311, x301, x302);

			// val X425 = X423 * d_ReLU()(X652)/d_X651
			JCudaTensor x312;
			JCudaTensor x313, x314;
			x313 = x287;
			x314 = x63;
			x312 = x55.backward(x313, x314);

			// Dealloc(X652)
			JCudaTensor x315;
			x315 = x63;
			x315.free();

			// val X426 = X425 * d_Convolv(1,1)(cv4_W)/d_X650
			JCudaTensor x316;
			JCudaTensor x317, x318;
			x317 = x312;
			x318 = x60;
			x316 = x62.backward_data(x317, x318);

			// V_cv4_W <~~ X425 * d_Convolv(1,1)(X650)/d_cv4_W
			float x320, x321;
			float x322;
			float x323;
			x322 = 1;
			x323 = lrn_rate;
			x320 = x322 * x323;
			x321 = momentum;
			JCudaTensor x324, x325;
			x324 = x312;
			x325 = x53;
			x62.backward_filter(x324, x325, x319, x320, x321);

			// V_cv4_B <~~ X425 * d_Convolv(1,1)()/d_cv4_B
			float x327, x328;
			float x329;
			float x330;
			x329 = 2;
			x330 = lrn_rate;
			x327 = x329 * x330;
			x328 = momentum;
			JCudaTensor x331;
			x331 = x312;
			x62.backward_bias(x331, x326, x327, x328);

			// Dealloc(X425)
			JCudaTensor x332;
			x332 = x312;
			x332.free();

			// cv4_W <~~ V_cv4_W
			float x333, x334;
			x333 = 1;
			float x335;
			float x336;
			x335 = 1;
			float x337;
			float x338;
			float x339;
			float x340;
			x339 = 1;
			x340 = decay;
			x337 = x339 * x340;
			float x341;
			float x342;
			x341 = 1;
			x342 = lrn_rate;
			x338 = x341 * x342;
			x336 = x337 * x338;
			x334 = x335 + x336;
			JCudaTensor x343;
			x343 = x319;
			x60.update(x343, x333, x334);

			// cv4_B <~~ V_cv4_B
			float x344, x345;
			x344 = 1;
			x345 = 1;
			JCudaTensor x346;
			x346 = x326;
			x61.update(x346, x344, x345);

			// val X428 = X426 * d_ReLU()(X650)/d_X649
			JCudaTensor x347;
			JCudaTensor x348, x349;
			x348 = x316;
			x349 = x53;
			x347 = x55.backward(x348, x349);

			// Dealloc(X650)
			JCudaTensor x350;
			x350 = x53;
			x350.free();

			// V_cv3_B <~~ X428 * d_Convolv(1,1)()/d_cv3_B
			float x352, x353;
			float x354;
			float x355;
			x354 = 2;
			x355 = lrn_rate;
			x352 = x354 * x355;
			x353 = momentum;
			JCudaTensor x356;
			x356 = x347;
			x52.backward_bias(x356, x351, x352, x353);

			// val X429 = X428 * d_Convolv(1,1)(cv3_W)/d_X648
			JCudaTensor x357;
			JCudaTensor x358, x359;
			x358 = x347;
			x359 = x50;
			x357 = x52.backward_data(x358, x359);

			// V_cv3_W <~~ X428 * d_Convolv(1,1)(X648)/d_cv3_W
			float x361, x362;
			float x363;
			float x364;
			x363 = 1;
			x364 = lrn_rate;
			x361 = x363 * x364;
			x362 = momentum;
			JCudaTensor x365, x366;
			x365 = x347;
			x366 = x43;
			x52.backward_filter(x365, x366, x360, x361, x362);

			// Dealloc(X428)
			JCudaTensor x367;
			x367 = x347;
			x367.free();

			// cv3_B <~~ V_cv3_B
			float x368, x369;
			x368 = 1;
			x369 = 1;
			JCudaTensor x370;
			x370 = x351;
			x51.update(x370, x368, x369);

			// cv3_W <~~ V_cv3_W
			float x371, x372;
			x371 = 1;
			float x373;
			float x374;
			x373 = 1;
			float x375;
			float x376;
			float x377;
			float x378;
			x377 = 1;
			x378 = decay;
			x375 = x377 * x378;
			float x379;
			float x380;
			x379 = 1;
			x380 = lrn_rate;
			x376 = x379 * x380;
			x374 = x375 * x376;
			x372 = x373 + x374;
			JCudaTensor x381;
			x381 = x360;
			x50.update(x381, x371, x372);

			// val X431 = X429 * d_Pooling(3,2,0,true)(X648,X647)/d_X647
			JCudaTensor x382;
			JCudaTensor x383, x384, x385;
			x383 = x357;
			x384 = x43;
			x385 = x40;
			x382 = x45.backward(x383, x384, x385);

			// Dealloc(X429)
			JCudaTensor x386;
			x386 = x357;
			x386.free();

			// Dealloc(X648)
			JCudaTensor x387;
			x387 = x43;
			x387.free();

			// val X433 = X431 * d_LRN(5,1.0E-4,0.75)(X647,X646)/d_X646
			JCudaTensor x388;
			JCudaTensor x389, x390, x391;
			x389 = x382;
			x390 = x40;
			x391 = x37;
			x388 = x42.backward(x389, x390, x391);

			// Dealloc(X647)
			JCudaTensor x392;
			x392 = x40;
			x392.free();

			// val X435 = X433 * d_ReLU()(X646)/d_X645
			JCudaTensor x393;
			JCudaTensor x394, x395;
			x394 = x388;
			x395 = x37;
			x393 = x39.backward(x394, x395);

			// Dealloc(X646)
			JCudaTensor x396;
			x396 = x37;
			x396.free();

			// val X436 = X435 * d_Convolv(1,2)(cv2_W)/d_X644
			JCudaTensor x397;
			JCudaTensor x398, x399;
			x398 = x393;
			x399 = x34;
			x397 = x36.backward_data(x398, x399);

			// V_cv2_W <~~ X435 * d_Convolv(1,2)(X644)/d_cv2_W
			float x401, x402;
			float x403;
			float x404;
			x403 = 1;
			x404 = lrn_rate;
			x401 = x403 * x404;
			x402 = momentum;
			JCudaTensor x405, x406;
			x405 = x393;
			x406 = x27;
			x36.backward_filter(x405, x406, x400, x401, x402);

			// V_cv2_B <~~ X435 * d_Convolv(1,2)()/d_cv2_B
			float x408, x409;
			float x410;
			float x411;
			x410 = 2;
			x411 = lrn_rate;
			x408 = x410 * x411;
			x409 = momentum;
			JCudaTensor x412;
			x412 = x393;
			x36.backward_bias(x412, x407, x408, x409);

			// Dealloc(X435)
			JCudaTensor x413;
			x413 = x393;
			x413.free();

			// cv2_W <~~ V_cv2_W
			float x414, x415;
			x414 = 1;
			float x416;
			float x417;
			x416 = 1;
			float x418;
			float x419;
			float x420;
			float x421;
			x420 = 1;
			x421 = decay;
			x418 = x420 * x421;
			float x422;
			float x423;
			x422 = 1;
			x423 = lrn_rate;
			x419 = x422 * x423;
			x417 = x418 * x419;
			x415 = x416 + x417;
			JCudaTensor x424;
			x424 = x400;
			x34.update(x424, x414, x415);

			// cv2_B <~~ V_cv2_B
			float x425, x426;
			x425 = 1;
			x426 = 1;
			JCudaTensor x427;
			x427 = x407;
			x35.update(x427, x425, x426);

			// val X438 = X436 * d_Pooling(3,2,0,true)(X644,X643)/d_X643
			JCudaTensor x428;
			JCudaTensor x429, x430, x431;
			x429 = x397;
			x430 = x27;
			x431 = x24;
			x428 = x29.backward(x429, x430, x431);

			// Dealloc(X436)
			JCudaTensor x432;
			x432 = x397;
			x432.free();

			// Dealloc(X644)
			JCudaTensor x433;
			x433 = x27;
			x433.free();

			// val X440 = X438 * d_LRN(5,1.0E-4,0.75)(X643,X642)/d_X642
			JCudaTensor x434;
			JCudaTensor x435, x436, x437;
			x435 = x428;
			x436 = x24;
			x437 = x21;
			x434 = x26.backward(x435, x436, x437);

			// Dealloc(X643)
			JCudaTensor x438;
			x438 = x24;
			x438.free();

			// val X442 = X440 * d_ReLU()(X642)/d_X641
			JCudaTensor x439;
			JCudaTensor x440, x441;
			x440 = x434;
			x441 = x21;
			x439 = x23.backward(x440, x441);

			// Dealloc(X642)
			JCudaTensor x442;
			x442 = x21;
			x442.free();

			// V_cv1_W <~~ X442 * d_Convolv(4,2)(X640)/d_cv1_W
			float x444, x445;
			float x446;
			float x447;
			x446 = 1;
			x447 = lrn_rate;
			x444 = x446 * x447;
			x445 = momentum;
			JCudaTensor x448, x449;
			x448 = x439;
			x449 = x9;
			x17.backward_filter(x448, x449, x443, x444, x445);

			// Dealloc(X640)
			JCudaTensor x450;
			x450 = x9;
			x450.free();

			// V_cv1_B <~~ X442 * d_Convolv(4,2)()/d_cv1_B
			float x452, x453;
			float x454;
			float x455;
			x454 = 2;
			x455 = lrn_rate;
			x452 = x454 * x455;
			x453 = momentum;
			JCudaTensor x456;
			x456 = x439;
			x17.backward_bias(x456, x451, x452, x453);

			// Dealloc(X442)
			JCudaTensor x457;
			x457 = x439;
			x457.free();

			// cv1_W <~~ V_cv1_W
			float x458, x459;
			x458 = 1;
			float x460;
			float x461;
			x460 = 1;
			float x462;
			float x463;
			float x464;
			float x465;
			x464 = 1;
			x465 = decay;
			x462 = x464 * x465;
			float x466;
			float x467;
			x466 = 1;
			x467 = lrn_rate;
			x463 = x466 * x467;
			x461 = x462 * x463;
			x459 = x460 + x461;
			JCudaTensor x468;
			x468 = x443;
			x15.update(x468, x458, x459);

			// cv1_B <~~ V_cv1_B
			float x469, x470;
			x469 = 1;
			x470 = 1;
			JCudaTensor x471;
			x471 = x451;
			x16.update(x471, x469, x470);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X671 = Cuda(X)
			JCudaTensor x472;
			JTensorFloat x473;
			x473 = x3;
			x472 = x473.asJCudaTensor();

			// val X672 = Convolv(4,2)(X671,cv1_W,cv1_B)
			JCudaTensor x474;
			JCudaTensor x475, x476, x477;
			x475 = x472;
			x476 = x15;
			x477 = x16;
			x474 = x17.forward(x475, x476, x477);

			// Dealloc(X671)
			JCudaTensor x478;
			x478 = x472;
			x478.free();

			// val X673 = ReLU()(X672)
			JCudaTensor x479;
			JCudaTensor x480;
			x480 = x474;
			x479 = x23.forward(x480);

			// val X674 = LRN(5,1.0E-4,0.75)(X673)
			JCudaTensor x481;
			JCudaTensor x482;
			x482 = x479;
			x481 = x26.forward(x482);

			// Dealloc(X673)
			JCudaTensor x483;
			x483 = x479;
			x483.free();

			// val X675 = Pooling(3,2,0,true)(X674)
			JCudaTensor x484;
			JCudaTensor x485;
			x485 = x481;
			x484 = x29.forward(x485);

			// Dealloc(X674)
			JCudaTensor x486;
			x486 = x481;
			x486.free();

			// val X676 = Convolv(1,2)(X675,cv2_W,cv2_B)
			JCudaTensor x487;
			JCudaTensor x488, x489, x490;
			x488 = x484;
			x489 = x34;
			x490 = x35;
			x487 = x36.forward(x488, x489, x490);

			// Dealloc(X675)
			JCudaTensor x491;
			x491 = x484;
			x491.free();

			// val X677 = ReLU()(X676)
			JCudaTensor x492;
			JCudaTensor x493;
			x493 = x487;
			x492 = x39.forward(x493);

			// val X678 = LRN(5,1.0E-4,0.75)(X677)
			JCudaTensor x494;
			JCudaTensor x495;
			x495 = x492;
			x494 = x42.forward(x495);

			// Dealloc(X677)
			JCudaTensor x496;
			x496 = x492;
			x496.free();

			// val X679 = Pooling(3,2,0,true)(X678)
			JCudaTensor x497;
			JCudaTensor x498;
			x498 = x494;
			x497 = x45.forward(x498);

			// Dealloc(X678)
			JCudaTensor x499;
			x499 = x494;
			x499.free();

			// val X680 = Convolv(1,1)(X679,cv3_W,cv3_B)
			JCudaTensor x500;
			JCudaTensor x501, x502, x503;
			x501 = x497;
			x502 = x50;
			x503 = x51;
			x500 = x52.forward(x501, x502, x503);

			// Dealloc(X679)
			JCudaTensor x504;
			x504 = x497;
			x504.free();

			// val X681 = ReLU()(X680)
			JCudaTensor x505;
			JCudaTensor x506;
			x506 = x500;
			x505 = x55.forward(x506);

			// val X682 = Convolv(1,1)(X681,cv4_W,cv4_B)
			JCudaTensor x507;
			JCudaTensor x508, x509, x510;
			x508 = x505;
			x509 = x60;
			x510 = x61;
			x507 = x62.forward(x508, x509, x510);

			// Dealloc(X681)
			JCudaTensor x511;
			x511 = x505;
			x511.free();

			// val X683 = ReLU()(X682)
			JCudaTensor x512;
			JCudaTensor x513;
			x513 = x507;
			x512 = x55.forward(x513);

			// val X684 = Convolv(1,1)(X683,cv5_W,cv5_B)
			JCudaTensor x514;
			JCudaTensor x515, x516, x517;
			x515 = x512;
			x516 = x69;
			x517 = x70;
			x514 = x71.forward(x515, x516, x517);

			// Dealloc(X683)
			JCudaTensor x518;
			x518 = x512;
			x518.free();

			// val X685 = ReLU()(X684)
			JCudaTensor x519;
			JCudaTensor x520;
			x520 = x514;
			x519 = x74.forward(x520);

			// val X686 = Pooling(3,2,0,true)(X685)
			JCudaTensor x521;
			JCudaTensor x522;
			x522 = x519;
			x521 = x77.forward(x522);

			// Dealloc(X685)
			JCudaTensor x523;
			x523 = x519;
			x523.free();

			// val X687 = (X686[1><3])(i10 | @) * (fc6_W)(i11 | @)
			JCudaTensor x524;
			JCudaMatrix x525;
			JCudaMatrix x526;
			JCudaTensor x527;
			JCudaTensor x528;
			x528 = x521;
			x527 = x528.flatten(1, new int[]{256, 6, 6});
			x525 = x527.asMatrix(1, true);
			JCudaTensor x529;
			x529 = x84;
			x526 = x529.asMatrix(1, true);
			x524 = x525.times(x526);

			// Dealloc(X686)
			JCudaTensor x530;
			x530 = x521;
			x530.free();

			// val X689 = (X687 + (i10) => fc6_B)
			JCudaTensor x531;
			JCudaTensor x532, x533;
			x532 = x524;
			x533 = x88;
			x531 = x533.copy(128, x532);

			// val X690 = ReLU()(X689)
			JCudaTensor x534;
			JCudaTensor x535;
			x535 = x531;
			x534 = x91.forward(x535);

			// val X691 = Dropout(0.5)(X690)
			JCudaTensor x536;
			JCudaTensor x537;
			x537 = x534;
			x536 = x94.forward(x537);

			// Dealloc(X690)
			JCudaTensor x538;
			x538 = x534;
			x538.free();

			// val X692 = (X691)(i13 | @) * (fc7_W)(i14 | @)
			JCudaTensor x539;
			JCudaMatrix x540;
			JCudaMatrix x541;
			JCudaTensor x542;
			x542 = x536;
			x540 = x542.asMatrix(1, true);
			JCudaTensor x543;
			x543 = x100;
			x541 = x543.asMatrix(1, true);
			x539 = x540.times(x541);

			// Dealloc(X691)
			JCudaTensor x544;
			x544 = x536;
			x544.free();

			// val X694 = (X692 + (i13) => fc7_B)
			JCudaTensor x545;
			JCudaTensor x546, x547;
			x546 = x539;
			x547 = x104;
			x545 = x547.copy(128, x546);

			// val X695 = ReLU()(X694)
			JCudaTensor x548;
			JCudaTensor x549;
			x549 = x545;
			x548 = x91.forward(x549);

			// val X696 = Dropout(0.5)(X695)
			JCudaTensor x550;
			JCudaTensor x551;
			x551 = x548;
			x550 = x94.forward(x551);

			// Dealloc(X695)
			JCudaTensor x552;
			x552 = x548;
			x552.free();

			// val X697 = (X696)(i16 | @) * (fc8_W)(i17 | @)
			JCudaTensor x553;
			JCudaMatrix x554;
			JCudaMatrix x555;
			JCudaTensor x556;
			x556 = x550;
			x554 = x556.asMatrix(1, true);
			JCudaTensor x557;
			x557 = x114;
			x555 = x557.asMatrix(1, true);
			x553 = x554.times(x555);

			// Dealloc(X696)
			JCudaTensor x558;
			x558 = x550;
			x558.free();

			// val X699 = (X697 + (i16) => fc8_B)
			JCudaTensor x559;
			JCudaTensor x560, x561;
			x560 = x553;
			x561 = x118;
			x559 = x561.copy(128, x560);

			// Precision(Accuracy(X699, Y, 1))
			float x563;
			JCudaTensor x564;
			JTensorFloat x565;
			x564 = x559;
			x565 = x4;
			x563 = x564.accuracy(x565, 1);
			System.out.println(x5 + " test precision "  + x563);
			x562 += x563;

			// Dealloc(X699)
			JCudaTensor x566;
			x566 = x559;
			x566.free();

		}
		System.out.println();
		System.out.println("average precision: " + x562/test_itr);
		System.out.println(); 
	}

}