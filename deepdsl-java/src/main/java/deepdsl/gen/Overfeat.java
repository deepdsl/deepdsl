package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Overfeat {
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
	static float momentum = 0.9f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/overfeat";
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

	// (Convolv(1,1),List(List(128, 1024, 13, 13), List(1024, 1024, 3, 3), List(1024)))
	static JCudnnConvolution x66 = new JCudnnConvolution(new int[]{128,1024,13,13},new int[]{1024,1024,3,3},new int[]{1024}, 1, 1);
	// (Convolv(1,1),List(List(128, 256, 13, 13), List(512, 256, 3, 3), List(512)))
	static JCudnnConvolution x46 = new JCudnnConvolution(new int[]{128,256,13,13},new int[]{512,256,3,3},new int[]{512}, 1, 1);
	// (Convolv(1,1),List(List(128, 512, 13, 13), List(1024, 512, 3, 3), List(1024)))
	static JCudnnConvolution x56 = new JCudnnConvolution(new int[]{128,512,13,13},new int[]{1024,512,3,3},new int[]{1024}, 1, 1);
	// (Convolv(1,2),List(List(128, 96, 27, 27), List(256, 96, 5, 5), List(256)))
	static JCudnnConvolution x33 = new JCudnnConvolution(new int[]{128,96,27,27},new int[]{256,96,5,5},new int[]{256}, 1, 2);
	// (Convolv(4,0),List(List(128, 3, 224, 224), List(96, 3, 11, 11), List(96)))
	static JCudnnConvolution x20 = new JCudnnConvolution(new int[]{128,3,224,224},new int[]{96,3,11,11},new int[]{96}, 4, 0);
	// (LMDB,false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, 1000, true);
	// (LMDB,true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, 1000, false);
	// (LogSoftmax(),List(List(128, 1000)))
	static JCudnnSoftmax x105 = new JCudnnSoftmax(new int[]{128,1000}, SoftmaxAlgorithm.LOG);
	// (Pooling(2,2,0,true),List(List(128, 1024, 13, 13)))
	static JCudnnPooling x71 = new JCudnnPooling(new int[]{128,1024,13,13}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(128, 256, 27, 27)))
	static JCudnnPooling x39 = new JCudnnPooling(new int[]{128,256,27,27}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(128, 96, 54, 54)))
	static JCudnnPooling x26 = new JCudnnPooling(new int[]{128,96,54,54}, 2, 2, 0, PoolingType.MAX);
	// (ReLU(),List(List(128, 1024, 13, 13)))
	static JCudnnActivation x59 = new JCudnnActivation(new int[]{128,1024,13,13}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 256, 27, 27)))
	static JCudnnActivation x36 = new JCudnnActivation(new int[]{128,256,27,27}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 512, 13, 13)))
	static JCudnnActivation x49 = new JCudnnActivation(new int[]{128,512,13,13}, ActivationMode.RELU);
	// (ReLU(),List(List(128, 96, 54, 54)))
	static JCudnnActivation x23 = new JCudnnActivation(new int[]{128,96,54,54}, ActivationMode.RELU);
	// Precision(Accuracy(X463, Y, 1))
	static float x530;
	// V_cv1_B
	static JCudaTensor x435 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x427 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x396 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x389 = JTensor.constFloat(0.0f, 256, 96, 5, 5).asJCudaTensor();
	// V_cv3_B
	static JCudaTensor x345 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// V_cv3_W
	static JCudaTensor x354 = JTensor.constFloat(0.0f, 512, 256, 3, 3).asJCudaTensor();
	// V_cv4_B
	static JCudaTensor x320 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_cv4_W
	static JCudaTensor x313 = JTensor.constFloat(0.0f, 1024, 512, 3, 3).asJCudaTensor();
	// V_cv5_B
	static JCudaTensor x275 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_cv5_W
	static JCudaTensor x284 = JTensor.constFloat(0.0f, 1024, 1024, 3, 3).asJCudaTensor();
	// V_fc6_B
	static JCudaTensor x234 = JTensor.constFloat(0.0f, 3072).asJCudaTensor();
	// V_fc6_W
	static JCudaTensor x240 = JTensor.constFloat(0.0f, 3072, 36864).asJCudaTensor();
	// V_fc7_B
	static JCudaTensor x182 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
	// V_fc7_W
	static JCudaTensor x197 = JTensor.constFloat(0.0f, 4096, 3072).asJCudaTensor();
	// V_fc8_B
	static JCudaTensor x135 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc8_W
	static JCudaTensor x145 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;
	// cv1_B
	static JCudaTensor x19 = JTensor.constFloat(0.2f, 96).load(network_dir + "/cv1_B").asJCudaTensor();
	// cv1_W
	static JCudaTensor x18 = JTensor.randomFloat(-0.07422696f, 0.07422696f, 96, 3, 11, 11).load(network_dir + "/cv1_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x32 = JTensor.constFloat(0.2f, 256).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x31 = JTensor.randomFloat(-0.028867513f, 0.028867513f, 256, 96, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
	// cv3_B
	static JCudaTensor x45 = JTensor.constFloat(0.2f, 512).load(network_dir + "/cv3_B").asJCudaTensor();
	// cv3_W
	static JCudaTensor x44 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 512, 256, 3, 3).load(network_dir + "/cv3_W").asJCudaTensor();
	// cv4_B
	static JCudaTensor x55 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/cv4_B").asJCudaTensor();
	// cv4_W
	static JCudaTensor x54 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 1024, 512, 3, 3).load(network_dir + "/cv4_W").asJCudaTensor();
	// cv5_B
	static JCudaTensor x65 = JTensor.constFloat(0.2f, 1024).load(network_dir + "/cv5_B").asJCudaTensor();
	// cv5_W
	static JCudaTensor x64 = JTensor.randomFloat(-0.014731391f, 0.014731391f, 1024, 1024, 3, 3).load(network_dir + "/cv5_W").asJCudaTensor();
	// fc6_B
	static JCudaTensor x82 = JTensor.constFloat(0.0f, 3072).load(network_dir + "/fc6_B").asJCudaTensor();
	// fc6_W
	static JCudaTensor x78 = JTensor.randomFloat(-0.0073656957f, 0.0073656957f, 3072, 36864).load(network_dir + "/fc6_W").asJCudaTensor();
	// fc7_B
	static JCudaTensor x92 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
	// fc7_W
	static JCudaTensor x88 = JTensor.randomFloat(-0.02551552f, 0.02551552f, 4096, 3072).load(network_dir + "/fc7_W").asJCudaTensor();
	// fc8_B
	static JCudaTensor x102 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
	// fc8_W
	static JCudaTensor x98 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

	public static void main(String[] args){
		double t = System.nanoTime();
		train();
		System.out.println((System.nanoTime() - t) / 1.0E9);
		test();
		x19.save(network_dir + "/cv1_B");
		x18.save(network_dir + "/cv1_W");
		x32.save(network_dir + "/cv2_B");
		x31.save(network_dir + "/cv2_W");
		x45.save(network_dir + "/cv3_B");
		x44.save(network_dir + "/cv3_W");
		x55.save(network_dir + "/cv4_B");
		x54.save(network_dir + "/cv4_W");
		x65.save(network_dir + "/cv5_B");
		x64.save(network_dir + "/cv5_W");
		x82.save(network_dir + "/fc6_B");
		x78.save(network_dir + "/fc6_W");
		x92.save(network_dir + "/fc7_B");
		x88.save(network_dir + "/fc7_W");
		x102.save(network_dir + "/fc8_B");
		x98.save(network_dir + "/fc8_W");
		x435.free();
		x64.free();
		x18.free();
		x19.free();
		x396.free();
		x98.free();
		x275.free();
		x145.free();
		x284.free();
		x313.free();
		x54.free();
		x65.free();
		x45.free();
		x78.free();
		x354.free();
		x55.free();
		x88.free();
		x92.free();
		x82.free();
		x32.free();
		x182.free();
		x44.free();
		x345.free();
		x427.free();
		x135.free();
		x102.free();
		x389.free();
		x31.free();
		x197.free();
		x320.free();
		x240.free();
		x234.free();
		x39.free();
		x59.free();
		x49.free();
		x36.free();
		x71.free();
		x26.free();
		x23.free();
		x20.free();
		x46.free();
		x66.free();
		x105.free();
		x33.free();
		x56.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void train() {
		for(int x5=0; x5<train_itr; x5++) {
			JTensorFloatTuple x6 =  x1.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X440 = Cuda(Indicator(Y, 1000))
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x4.asIndicator(1000);
			x7 = x8.asJCudaTensor();

			// val X416 = Cuda(X)
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x3;
			x9 = x10.asJCudaTensor();

			// val X214 = - X440.copy
			JCudaTensor x11;
			JCudaTensor x12;
			float x13;
			x12 = x7;
			x12 = x12.clone();
			x13 = -1;
			x11 = x12.times_i(x13);

			// val X417 = Convolv(4,0)(X416,cv1_W,cv1_B)
			JCudaTensor x14;
			JCudaTensor x15, x16, x17;
			x15 = x9;
			x16 = x18;
			x17 = x19;
			x14 = x20.forward(x15, x16, x17);

			// val X418 = ReLU()(X417)
			JCudaTensor x21;
			JCudaTensor x22;
			x22 = x14;
			x21 = x23.forward(x22);

			// val X419 = Pooling(2,2,0,true)(X418)
			JCudaTensor x24;
			JCudaTensor x25;
			x25 = x21;
			x24 = x26.forward(x25);

			// val X420 = Convolv(1,2)(X419,cv2_W,cv2_B)
			JCudaTensor x27;
			JCudaTensor x28, x29, x30;
			x28 = x24;
			x29 = x31;
			x30 = x32;
			x27 = x33.forward(x28, x29, x30);

			// val X421 = ReLU()(X420)
			JCudaTensor x34;
			JCudaTensor x35;
			x35 = x27;
			x34 = x36.forward(x35);

			// val X422 = Pooling(2,2,0,true)(X421)
			JCudaTensor x37;
			JCudaTensor x38;
			x38 = x34;
			x37 = x39.forward(x38);

			// val X423 = Convolv(1,1)(X422,cv3_W,cv3_B)
			JCudaTensor x40;
			JCudaTensor x41, x42, x43;
			x41 = x37;
			x42 = x44;
			x43 = x45;
			x40 = x46.forward(x41, x42, x43);

			// val X424 = ReLU()(X423)
			JCudaTensor x47;
			JCudaTensor x48;
			x48 = x40;
			x47 = x49.forward(x48);

			// val X425 = Convolv(1,1)(X424,cv4_W,cv4_B)
			JCudaTensor x50;
			JCudaTensor x51, x52, x53;
			x51 = x47;
			x52 = x54;
			x53 = x55;
			x50 = x56.forward(x51, x52, x53);

			// val X426 = ReLU()(X425)
			JCudaTensor x57;
			JCudaTensor x58;
			x58 = x50;
			x57 = x59.forward(x58);

			// val X427 = Convolv(1,1)(X426,cv5_W,cv5_B)
			JCudaTensor x60;
			JCudaTensor x61, x62, x63;
			x61 = x57;
			x62 = x64;
			x63 = x65;
			x60 = x66.forward(x61, x62, x63);

			// val X428 = ReLU()(X427)
			JCudaTensor x67;
			JCudaTensor x68;
			x68 = x60;
			x67 = x59.forward(x68);

			// val X429 = Pooling(2,2,0,true)(X428)
			JCudaTensor x69;
			JCudaTensor x70;
			x70 = x67;
			x69 = x71.forward(x70);

			// val X430 = (X429[1><3])(i1 | @) * (fc6_W)(i2 | @)
			JCudaTensor x72;
			JCudaMatrix x73;
			JCudaMatrix x74;
			JCudaTensor x75;
			JCudaTensor x76;
			x76 = x69;
			x75 = x76.flatten(1, new int[]{1024, 6, 6});
			x73 = x75.asMatrix(1, true);
			JCudaTensor x77;
			x77 = x78;
			x74 = x77.asMatrix(1, true);
			x72 = x73.times(x74);

			// val X432 = (X430 + (i1) => fc6_B)
			JCudaTensor x79;
			JCudaTensor x80, x81;
			x80 = x72;
			x81 = x82;
			x79 = x81.copy(128, x80);

			// val X433 = (X432)(i4 | @) * (fc7_W)(i5 | @)
			JCudaTensor x83;
			JCudaMatrix x84;
			JCudaMatrix x85;
			JCudaTensor x86;
			x86 = x79;
			x84 = x86.asMatrix(1, true);
			JCudaTensor x87;
			x87 = x88;
			x85 = x87.asMatrix(1, true);
			x83 = x84.times(x85);

			// val X435 = (X433 + (i4) => fc7_B)
			JCudaTensor x89;
			JCudaTensor x90, x91;
			x90 = x83;
			x91 = x92;
			x89 = x91.copy(128, x90);

			// val X436 = (X435)(i7 | @) * (fc8_W)(i8 | @)
			JCudaTensor x93;
			JCudaMatrix x94;
			JCudaMatrix x95;
			JCudaTensor x96;
			x96 = x89;
			x94 = x96.asMatrix(1, true);
			JCudaTensor x97;
			x97 = x98;
			x95 = x97.asMatrix(1, true);
			x93 = x94.times(x95);

			// val X438 = (X436 + (i7) => fc8_B)
			JCudaTensor x99;
			JCudaTensor x100, x101;
			x100 = x93;
			x101 = x102;
			x99 = x101.copy(128, x100);

			// val X439 = LogSoftmax()(X438)
			JCudaTensor x103;
			JCudaTensor x104;
			x104 = x99;
			x103 = x105.forward(x104);

			// Dealloc(X438)
			JCudaTensor x106;
			x106 = x99;
			x106.free();

			// val X215 = (X214 / |128|)
			JCudaTensor x107;
			JCudaTensor x108;
			float x109;
			x108 = x11;
			float x110;
			x110 = 128;
			x109 = 1 / x110;
			x107 = x108.times_i(x109);

			// Cost(((0 - (X440 . X439)) / |128|))
			float x111;
			float x112;
			float x113;
			float x114;
			JCudaTensor x115, x116;
			x115 = x7;
			x116 = x103;
			x114 = x115.dot(x116);
			x112 = - x114;
			x113 = 128;
			x111 = x112 / x113;
			System.out.println(x5 + " " + x111);
			if (Float.isNaN(x111)) { System.exit(-1); }

			// Dealloc(X440)
			JCudaTensor x117;
			x117 = x7;
			x117.free();

			// val X240 = X215 * d_LogSoftmax()(X439)/d_X438
			JCudaTensor x118;
			JCudaTensor x119, x120;
			x119 = x107;
			x120 = x103;
			x118 = x105.backward(x119, x120);

			// Dealloc(X215)
			JCudaTensor x121;
			x121 = x107;
			x121.free();

			// Dealloc(X439)
			JCudaTensor x122;
			x122 = x103;
			x122.free();

			// val m1 = (i283) => fc8_W[@, i283]
			JCudaMatrix x123;
			JCudaTensor x124;
			x124 = x98;
			x123 = x124.asMatrix(1, false);

			// val m25 = (i23) => X240[@, i23]
			JCudaMatrix x125;
			JCudaTensor x126;
			x126 = x118;
			x125 = x126.asMatrix(1, false);

			// val X260 = (X240)(i282 | @) * m1
			JCudaTensor x127;
			JCudaMatrix x128;
			JCudaMatrix x129;
			JCudaTensor x130;
			x130 = x118;
			x128 = x130.asMatrix(1, true);
			x129 = x123;
			x127 = x128.times(x129);

			// val m2 = (i287) => fc7_W[@, i287]
			JCudaMatrix x131;
			JCudaTensor x132;
			x132 = x88;
			x131 = x132.asMatrix(1, false);

			// val m27 = (i27) => X435[@, i27]
			JCudaMatrix x133;
			JCudaTensor x134;
			x134 = x89;
			x133 = x134.asMatrix(1, false);

			// V_fc8_B <~~ Sum(m25)
			float x136, x137;
			float x138;
			float x139;
			x138 = 1;
			x139 = lrn_rate;
			x136 = x138 * x139;
			x137 = momentum;
			JCudaMatrix x140;
			x140 = x125;
			x140.sum(x135, x136, x137);

			// val X261 = (X260)(i286 | @) * m2
			JCudaTensor x141;
			JCudaMatrix x142;
			JCudaMatrix x143;
			JCudaTensor x144;
			x144 = x127;
			x142 = x144.asMatrix(1, true);
			x143 = x131;
			x141 = x142.times(x143);

			// V_fc8_W <~~ m25 * m27
			float x146, x147;
			float x148;
			float x149;
			x148 = 1;
			x149 = lrn_rate;
			x146 = x148 * x149;
			x147 = momentum;
			JCudaMatrix x150;
			JCudaMatrix x151;
			x150 = x125;
			x151 = x133;
			x150.times(x151, x145, x146, x147);

			// Dealloc(X240)
			JCudaTensor x152;
			x152 = x118;
			x152.free();

			// Dealloc(X435)
			JCudaTensor x153;
			x153 = x89;
			x153.free();

			// val m22 = (i30) => X260[@, i30]
			JCudaMatrix x154;
			JCudaTensor x155;
			x155 = x127;
			x154 = x155.asMatrix(1, false);

			// val m3 = (i291) => fc6_W[@, i291]
			JCudaMatrix x156;
			JCudaTensor x157;
			x157 = x78;
			x156 = x157.asMatrix(1, false);

			// val m24 = (i34) => X432[@, i34]
			JCudaMatrix x158;
			JCudaTensor x159;
			x159 = x79;
			x158 = x159.asMatrix(1, false);

			// fc8_B <~~ V_fc8_B
			float x160, x161;
			x160 = 1;
			float x162;
			float x163;
			x162 = 1;
			float x164;
			float x165;
			float x166;
			float x167;
			x166 = 1;
			x167 = decay;
			x164 = x166 * x167;
			float x168;
			float x169;
			x168 = 1;
			x169 = lrn_rate;
			x165 = x168 * x169;
			x163 = x164 * x165;
			x161 = x162 + x163;
			JCudaTensor x170;
			x170 = x135;
			x102.update(x170, x160, x161);

			// fc8_W <~~ V_fc8_W
			float x171, x172;
			x171 = 1;
			float x173;
			float x174;
			x173 = 1;
			float x175;
			float x176;
			float x177;
			float x178;
			x177 = 1;
			x178 = decay;
			x175 = x177 * x178;
			float x179;
			float x180;
			x179 = 1;
			x180 = lrn_rate;
			x176 = x179 * x180;
			x174 = x175 * x176;
			x172 = x173 + x174;
			JCudaTensor x181;
			x181 = x145;
			x98.update(x181, x171, x172);

			// V_fc7_B <~~ Sum(m22)
			float x183, x184;
			float x185;
			float x186;
			x185 = 1;
			x186 = lrn_rate;
			x183 = x185 * x186;
			x184 = momentum;
			JCudaMatrix x187;
			x187 = x154;
			x187.sum(x182, x183, x184);

			// val m19 = (i39) => X261[@, i39]
			JCudaMatrix x188;
			JCudaTensor x189;
			x189 = x141;
			x188 = x189.asMatrix(1, false);

			// val m21 = (i43) => X429[1><3][@, i43]
			JCudaMatrix x190;
			JCudaTensor x191;
			JCudaTensor x192;
			x192 = x69;
			x191 = x192.flatten(1, new int[]{1024, 6, 6});
			x190 = x191.asMatrix(1, false);

			// val X262 = (X261)(i290 | @) * m3
			JCudaTensor x193;
			JCudaMatrix x194;
			JCudaMatrix x195;
			JCudaTensor x196;
			x196 = x141;
			x194 = x196.asMatrix(1, true);
			x195 = x156;
			x193 = x194.times(x195);

			// V_fc7_W <~~ m22 * m24
			float x198, x199;
			float x200;
			float x201;
			x200 = 1;
			x201 = lrn_rate;
			x198 = x200 * x201;
			x199 = momentum;
			JCudaMatrix x202;
			JCudaMatrix x203;
			x202 = x154;
			x203 = x158;
			x202.times(x203, x197, x198, x199);

			// Dealloc(X260)
			JCudaTensor x204;
			x204 = x127;
			x204.free();

			// Dealloc(X432)
			JCudaTensor x205;
			x205 = x79;
			x205.free();

			// fc7_W <~~ V_fc7_W
			float x206, x207;
			x206 = 1;
			float x208;
			float x209;
			x208 = 1;
			float x210;
			float x211;
			float x212;
			float x213;
			x212 = 1;
			x213 = decay;
			x210 = x212 * x213;
			float x214;
			float x215;
			x214 = 1;
			x215 = lrn_rate;
			x211 = x214 * x215;
			x209 = x210 * x211;
			x207 = x208 + x209;
			JCudaTensor x216;
			x216 = x197;
			x88.update(x216, x206, x207);

			// fc7_B <~~ V_fc7_B
			float x217, x218;
			x217 = 1;
			float x219;
			float x220;
			x219 = 1;
			float x221;
			float x222;
			float x223;
			float x224;
			x223 = 1;
			x224 = decay;
			x221 = x223 * x224;
			float x225;
			float x226;
			x225 = 1;
			x226 = lrn_rate;
			x222 = x225 * x226;
			x220 = x221 * x222;
			x218 = x219 + x220;
			JCudaTensor x227;
			x227 = x182;
			x92.update(x227, x217, x218);

			// val X264 = X262[1<>3] * d_Pooling(2,2,0,true)(X429,X428)/d_X428
			JCudaTensor x228;
			JCudaTensor x229, x230, x231;
			JCudaTensor x232;
			x232 = x193;
			x229 = x232.unflatten(1, new int[]{1024, 6, 6});
			x230 = x69;
			x231 = x67;
			x228 = x71.backward(x229, x230, x231);

			// Dealloc(X262)
			JCudaTensor x233;
			x233 = x193;
			x233.free();

			// V_fc6_B <~~ Sum(m19)
			float x235, x236;
			float x237;
			float x238;
			x237 = 1;
			x238 = lrn_rate;
			x235 = x237 * x238;
			x236 = momentum;
			JCudaMatrix x239;
			x239 = x188;
			x239.sum(x234, x235, x236);

			// V_fc6_W <~~ m19 * m21
			float x241, x242;
			float x243;
			float x244;
			x243 = 1;
			x244 = lrn_rate;
			x241 = x243 * x244;
			x242 = momentum;
			JCudaMatrix x245;
			JCudaMatrix x246;
			x245 = x188;
			x246 = x190;
			x245.times(x246, x240, x241, x242);

			// Dealloc(X261)
			JCudaTensor x247;
			x247 = x141;
			x247.free();

			// Dealloc(X429)
			JCudaTensor x248;
			x248 = x69;
			x248.free();

			// fc6_B <~~ V_fc6_B
			float x249, x250;
			x249 = 1;
			float x251;
			float x252;
			x251 = 1;
			float x253;
			float x254;
			float x255;
			float x256;
			x255 = 1;
			x256 = decay;
			x253 = x255 * x256;
			float x257;
			float x258;
			x257 = 1;
			x258 = lrn_rate;
			x254 = x257 * x258;
			x252 = x253 * x254;
			x250 = x251 + x252;
			JCudaTensor x259;
			x259 = x234;
			x82.update(x259, x249, x250);

			// fc6_W <~~ V_fc6_W
			float x260, x261;
			x260 = 1;
			float x262;
			float x263;
			x262 = 1;
			float x264;
			float x265;
			float x266;
			float x267;
			x266 = 1;
			x267 = decay;
			x264 = x266 * x267;
			float x268;
			float x269;
			x268 = 1;
			x269 = lrn_rate;
			x265 = x268 * x269;
			x263 = x264 * x265;
			x261 = x262 + x263;
			JCudaTensor x270;
			x270 = x240;
			x78.update(x270, x260, x261);

			// val X266 = X264 * d_ReLU()(X428)/d_X427
			JCudaTensor x271;
			JCudaTensor x272, x273;
			x272 = x228;
			x273 = x67;
			x271 = x59.backward(x272, x273);

			// Dealloc(X428)
			JCudaTensor x274;
			x274 = x67;
			x274.free();

			// V_cv5_B <~~ X266 * d_Convolv(1,1)()/d_cv5_B
			float x276, x277;
			float x278;
			float x279;
			x278 = 2;
			x279 = lrn_rate;
			x276 = x278 * x279;
			x277 = momentum;
			JCudaTensor x280;
			x280 = x271;
			x66.backward_bias(x280, x275, x276, x277);

			// val X267 = X266 * d_Convolv(1,1)(cv5_W)/d_X426
			JCudaTensor x281;
			JCudaTensor x282, x283;
			x282 = x271;
			x283 = x64;
			x281 = x66.backward_data(x282, x283);

			// V_cv5_W <~~ X266 * d_Convolv(1,1)(X426)/d_cv5_W
			float x285, x286;
			float x287;
			float x288;
			x287 = 1;
			x288 = lrn_rate;
			x285 = x287 * x288;
			x286 = momentum;
			JCudaTensor x289, x290;
			x289 = x271;
			x290 = x57;
			x66.backward_filter(x289, x290, x284, x285, x286);

			// Dealloc(X266)
			JCudaTensor x291;
			x291 = x271;
			x291.free();

			// cv5_B <~~ V_cv5_B
			float x292, x293;
			x292 = 1;
			x293 = 1;
			JCudaTensor x294;
			x294 = x275;
			x65.update(x294, x292, x293);

			// cv5_W <~~ V_cv5_W
			float x295, x296;
			x295 = 1;
			float x297;
			float x298;
			x297 = 1;
			float x299;
			float x300;
			float x301;
			float x302;
			x301 = 1;
			x302 = decay;
			x299 = x301 * x302;
			float x303;
			float x304;
			x303 = 1;
			x304 = lrn_rate;
			x300 = x303 * x304;
			x298 = x299 * x300;
			x296 = x297 + x298;
			JCudaTensor x305;
			x305 = x284;
			x64.update(x305, x295, x296);

			// val X269 = X267 * d_ReLU()(X426)/d_X425
			JCudaTensor x306;
			JCudaTensor x307, x308;
			x307 = x281;
			x308 = x57;
			x306 = x59.backward(x307, x308);

			// Dealloc(X426)
			JCudaTensor x309;
			x309 = x57;
			x309.free();

			// val X270 = X269 * d_Convolv(1,1)(cv4_W)/d_X424
			JCudaTensor x310;
			JCudaTensor x311, x312;
			x311 = x306;
			x312 = x54;
			x310 = x56.backward_data(x311, x312);

			// V_cv4_W <~~ X269 * d_Convolv(1,1)(X424)/d_cv4_W
			float x314, x315;
			float x316;
			float x317;
			x316 = 1;
			x317 = lrn_rate;
			x314 = x316 * x317;
			x315 = momentum;
			JCudaTensor x318, x319;
			x318 = x306;
			x319 = x47;
			x56.backward_filter(x318, x319, x313, x314, x315);

			// V_cv4_B <~~ X269 * d_Convolv(1,1)()/d_cv4_B
			float x321, x322;
			float x323;
			float x324;
			x323 = 2;
			x324 = lrn_rate;
			x321 = x323 * x324;
			x322 = momentum;
			JCudaTensor x325;
			x325 = x306;
			x56.backward_bias(x325, x320, x321, x322);

			// Dealloc(X269)
			JCudaTensor x326;
			x326 = x306;
			x326.free();

			// cv4_W <~~ V_cv4_W
			float x327, x328;
			x327 = 1;
			float x329;
			float x330;
			x329 = 1;
			float x331;
			float x332;
			float x333;
			float x334;
			x333 = 1;
			x334 = decay;
			x331 = x333 * x334;
			float x335;
			float x336;
			x335 = 1;
			x336 = lrn_rate;
			x332 = x335 * x336;
			x330 = x331 * x332;
			x328 = x329 + x330;
			JCudaTensor x337;
			x337 = x313;
			x54.update(x337, x327, x328);

			// cv4_B <~~ V_cv4_B
			float x338, x339;
			x338 = 1;
			x339 = 1;
			JCudaTensor x340;
			x340 = x320;
			x55.update(x340, x338, x339);

			// val X272 = X270 * d_ReLU()(X424)/d_X423
			JCudaTensor x341;
			JCudaTensor x342, x343;
			x342 = x310;
			x343 = x47;
			x341 = x49.backward(x342, x343);

			// Dealloc(X424)
			JCudaTensor x344;
			x344 = x47;
			x344.free();

			// V_cv3_B <~~ X272 * d_Convolv(1,1)()/d_cv3_B
			float x346, x347;
			float x348;
			float x349;
			x348 = 2;
			x349 = lrn_rate;
			x346 = x348 * x349;
			x347 = momentum;
			JCudaTensor x350;
			x350 = x341;
			x46.backward_bias(x350, x345, x346, x347);

			// val X273 = X272 * d_Convolv(1,1)(cv3_W)/d_X422
			JCudaTensor x351;
			JCudaTensor x352, x353;
			x352 = x341;
			x353 = x44;
			x351 = x46.backward_data(x352, x353);

			// V_cv3_W <~~ X272 * d_Convolv(1,1)(X422)/d_cv3_W
			float x355, x356;
			float x357;
			float x358;
			x357 = 1;
			x358 = lrn_rate;
			x355 = x357 * x358;
			x356 = momentum;
			JCudaTensor x359, x360;
			x359 = x341;
			x360 = x37;
			x46.backward_filter(x359, x360, x354, x355, x356);

			// Dealloc(X272)
			JCudaTensor x361;
			x361 = x341;
			x361.free();

			// cv3_B <~~ V_cv3_B
			float x362, x363;
			x362 = 1;
			x363 = 1;
			JCudaTensor x364;
			x364 = x345;
			x45.update(x364, x362, x363);

			// cv3_W <~~ V_cv3_W
			float x365, x366;
			x365 = 1;
			float x367;
			float x368;
			x367 = 1;
			float x369;
			float x370;
			float x371;
			float x372;
			x371 = 1;
			x372 = decay;
			x369 = x371 * x372;
			float x373;
			float x374;
			x373 = 1;
			x374 = lrn_rate;
			x370 = x373 * x374;
			x368 = x369 * x370;
			x366 = x367 + x368;
			JCudaTensor x375;
			x375 = x354;
			x44.update(x375, x365, x366);

			// val X275 = X273 * d_Pooling(2,2,0,true)(X422,X421)/d_X421
			JCudaTensor x376;
			JCudaTensor x377, x378, x379;
			x377 = x351;
			x378 = x37;
			x379 = x34;
			x376 = x39.backward(x377, x378, x379);

			// Dealloc(X273)
			JCudaTensor x380;
			x380 = x351;
			x380.free();

			// Dealloc(X422)
			JCudaTensor x381;
			x381 = x37;
			x381.free();

			// val X277 = X275 * d_ReLU()(X421)/d_X420
			JCudaTensor x382;
			JCudaTensor x383, x384;
			x383 = x376;
			x384 = x34;
			x382 = x36.backward(x383, x384);

			// Dealloc(X421)
			JCudaTensor x385;
			x385 = x34;
			x385.free();

			// val X278 = X277 * d_Convolv(1,2)(cv2_W)/d_X419
			JCudaTensor x386;
			JCudaTensor x387, x388;
			x387 = x382;
			x388 = x31;
			x386 = x33.backward_data(x387, x388);

			// V_cv2_W <~~ X277 * d_Convolv(1,2)(X419)/d_cv2_W
			float x390, x391;
			float x392;
			float x393;
			x392 = 1;
			x393 = lrn_rate;
			x390 = x392 * x393;
			x391 = momentum;
			JCudaTensor x394, x395;
			x394 = x382;
			x395 = x24;
			x33.backward_filter(x394, x395, x389, x390, x391);

			// V_cv2_B <~~ X277 * d_Convolv(1,2)()/d_cv2_B
			float x397, x398;
			float x399;
			float x400;
			x399 = 2;
			x400 = lrn_rate;
			x397 = x399 * x400;
			x398 = momentum;
			JCudaTensor x401;
			x401 = x382;
			x33.backward_bias(x401, x396, x397, x398);

			// Dealloc(X277)
			JCudaTensor x402;
			x402 = x382;
			x402.free();

			// cv2_W <~~ V_cv2_W
			float x403, x404;
			x403 = 1;
			float x405;
			float x406;
			x405 = 1;
			float x407;
			float x408;
			float x409;
			float x410;
			x409 = 1;
			x410 = decay;
			x407 = x409 * x410;
			float x411;
			float x412;
			x411 = 1;
			x412 = lrn_rate;
			x408 = x411 * x412;
			x406 = x407 * x408;
			x404 = x405 + x406;
			JCudaTensor x413;
			x413 = x389;
			x31.update(x413, x403, x404);

			// cv2_B <~~ V_cv2_B
			float x414, x415;
			x414 = 1;
			x415 = 1;
			JCudaTensor x416;
			x416 = x396;
			x32.update(x416, x414, x415);

			// val X280 = X278 * d_Pooling(2,2,0,true)(X419,X418)/d_X418
			JCudaTensor x417;
			JCudaTensor x418, x419, x420;
			x418 = x386;
			x419 = x24;
			x420 = x21;
			x417 = x26.backward(x418, x419, x420);

			// Dealloc(X278)
			JCudaTensor x421;
			x421 = x386;
			x421.free();

			// Dealloc(X419)
			JCudaTensor x422;
			x422 = x24;
			x422.free();

			// val X282 = X280 * d_ReLU()(X418)/d_X417
			JCudaTensor x423;
			JCudaTensor x424, x425;
			x424 = x417;
			x425 = x21;
			x423 = x23.backward(x424, x425);

			// Dealloc(X418)
			JCudaTensor x426;
			x426 = x21;
			x426.free();

			// V_cv1_W <~~ X282 * d_Convolv(4,0)(X416)/d_cv1_W
			float x428, x429;
			float x430;
			float x431;
			x430 = 1;
			x431 = lrn_rate;
			x428 = x430 * x431;
			x429 = momentum;
			JCudaTensor x432, x433;
			x432 = x423;
			x433 = x9;
			x20.backward_filter(x432, x433, x427, x428, x429);

			// Dealloc(X416)
			JCudaTensor x434;
			x434 = x9;
			x434.free();

			// V_cv1_B <~~ X282 * d_Convolv(4,0)()/d_cv1_B
			float x436, x437;
			float x438;
			float x439;
			x438 = 2;
			x439 = lrn_rate;
			x436 = x438 * x439;
			x437 = momentum;
			JCudaTensor x440;
			x440 = x423;
			x20.backward_bias(x440, x435, x436, x437);

			// Dealloc(X282)
			JCudaTensor x441;
			x441 = x423;
			x441.free();

			// cv1_W <~~ V_cv1_W
			float x442, x443;
			x442 = 1;
			float x444;
			float x445;
			x444 = 1;
			float x446;
			float x447;
			float x448;
			float x449;
			x448 = 1;
			x449 = decay;
			x446 = x448 * x449;
			float x450;
			float x451;
			x450 = 1;
			x451 = lrn_rate;
			x447 = x450 * x451;
			x445 = x446 * x447;
			x443 = x444 + x445;
			JCudaTensor x452;
			x452 = x427;
			x18.update(x452, x442, x443);

			// cv1_B <~~ V_cv1_B
			float x453, x454;
			x453 = 1;
			x454 = 1;
			JCudaTensor x455;
			x455 = x435;
			x19.update(x455, x453, x454);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X441 = Cuda(X)
			JCudaTensor x456;
			JTensorFloat x457;
			x457 = x3;
			x456 = x457.asJCudaTensor();

			// val X442 = Convolv(4,0)(X441,cv1_W,cv1_B)
			JCudaTensor x458;
			JCudaTensor x459, x460, x461;
			x459 = x456;
			x460 = x18;
			x461 = x19;
			x458 = x20.forward(x459, x460, x461);

			// Dealloc(X441)
			JCudaTensor x462;
			x462 = x456;
			x462.free();

			// val X443 = ReLU()(X442)
			JCudaTensor x463;
			JCudaTensor x464;
			x464 = x458;
			x463 = x23.forward(x464);

			// val X444 = Pooling(2,2,0,true)(X443)
			JCudaTensor x465;
			JCudaTensor x466;
			x466 = x463;
			x465 = x26.forward(x466);

			// Dealloc(X443)
			JCudaTensor x467;
			x467 = x463;
			x467.free();

			// val X445 = Convolv(1,2)(X444,cv2_W,cv2_B)
			JCudaTensor x468;
			JCudaTensor x469, x470, x471;
			x469 = x465;
			x470 = x31;
			x471 = x32;
			x468 = x33.forward(x469, x470, x471);

			// Dealloc(X444)
			JCudaTensor x472;
			x472 = x465;
			x472.free();

			// val X446 = ReLU()(X445)
			JCudaTensor x473;
			JCudaTensor x474;
			x474 = x468;
			x473 = x36.forward(x474);

			// val X447 = Pooling(2,2,0,true)(X446)
			JCudaTensor x475;
			JCudaTensor x476;
			x476 = x473;
			x475 = x39.forward(x476);

			// Dealloc(X446)
			JCudaTensor x477;
			x477 = x473;
			x477.free();

			// val X448 = Convolv(1,1)(X447,cv3_W,cv3_B)
			JCudaTensor x478;
			JCudaTensor x479, x480, x481;
			x479 = x475;
			x480 = x44;
			x481 = x45;
			x478 = x46.forward(x479, x480, x481);

			// Dealloc(X447)
			JCudaTensor x482;
			x482 = x475;
			x482.free();

			// val X449 = ReLU()(X448)
			JCudaTensor x483;
			JCudaTensor x484;
			x484 = x478;
			x483 = x49.forward(x484);

			// val X450 = Convolv(1,1)(X449,cv4_W,cv4_B)
			JCudaTensor x485;
			JCudaTensor x486, x487, x488;
			x486 = x483;
			x487 = x54;
			x488 = x55;
			x485 = x56.forward(x486, x487, x488);

			// Dealloc(X449)
			JCudaTensor x489;
			x489 = x483;
			x489.free();

			// val X451 = ReLU()(X450)
			JCudaTensor x490;
			JCudaTensor x491;
			x491 = x485;
			x490 = x59.forward(x491);

			// val X452 = Convolv(1,1)(X451,cv5_W,cv5_B)
			JCudaTensor x492;
			JCudaTensor x493, x494, x495;
			x493 = x490;
			x494 = x64;
			x495 = x65;
			x492 = x66.forward(x493, x494, x495);

			// Dealloc(X451)
			JCudaTensor x496;
			x496 = x490;
			x496.free();

			// val X453 = ReLU()(X452)
			JCudaTensor x497;
			JCudaTensor x498;
			x498 = x492;
			x497 = x59.forward(x498);

			// val X454 = Pooling(2,2,0,true)(X453)
			JCudaTensor x499;
			JCudaTensor x500;
			x500 = x497;
			x499 = x71.forward(x500);

			// Dealloc(X453)
			JCudaTensor x501;
			x501 = x497;
			x501.free();

			// val X455 = (X454[1><3])(i1 | @) * (fc6_W)(i2 | @)
			JCudaTensor x502;
			JCudaMatrix x503;
			JCudaMatrix x504;
			JCudaTensor x505;
			JCudaTensor x506;
			x506 = x499;
			x505 = x506.flatten(1, new int[]{1024, 6, 6});
			x503 = x505.asMatrix(1, true);
			JCudaTensor x507;
			x507 = x78;
			x504 = x507.asMatrix(1, true);
			x502 = x503.times(x504);

			// Dealloc(X454)
			JCudaTensor x508;
			x508 = x499;
			x508.free();

			// val X457 = (X455 + (i1) => fc6_B)
			JCudaTensor x509;
			JCudaTensor x510, x511;
			x510 = x502;
			x511 = x82;
			x509 = x511.copy(128, x510);

			// val X458 = (X457)(i4 | @) * (fc7_W)(i5 | @)
			JCudaTensor x512;
			JCudaMatrix x513;
			JCudaMatrix x514;
			JCudaTensor x515;
			x515 = x509;
			x513 = x515.asMatrix(1, true);
			JCudaTensor x516;
			x516 = x88;
			x514 = x516.asMatrix(1, true);
			x512 = x513.times(x514);

			// Dealloc(X457)
			JCudaTensor x517;
			x517 = x509;
			x517.free();

			// val X460 = (X458 + (i4) => fc7_B)
			JCudaTensor x518;
			JCudaTensor x519, x520;
			x519 = x512;
			x520 = x92;
			x518 = x520.copy(128, x519);

			// val X461 = (X460)(i7 | @) * (fc8_W)(i8 | @)
			JCudaTensor x521;
			JCudaMatrix x522;
			JCudaMatrix x523;
			JCudaTensor x524;
			x524 = x518;
			x522 = x524.asMatrix(1, true);
			JCudaTensor x525;
			x525 = x98;
			x523 = x525.asMatrix(1, true);
			x521 = x522.times(x523);

			// Dealloc(X460)
			JCudaTensor x526;
			x526 = x518;
			x526.free();

			// val X463 = (X461 + (i7) => fc8_B)
			JCudaTensor x527;
			JCudaTensor x528, x529;
			x528 = x521;
			x529 = x102;
			x527 = x529.copy(128, x528);

			// Precision(Accuracy(X463, Y, 1))
			float x531;
			JCudaTensor x532;
			JTensorFloat x533;
			x532 = x527;
			x533 = x4;
			x531 = x532.accuracy(x533, 1);
			System.out.println(x5 + " test precision "  + x531);
			x530 += x531;

			// Dealloc(X463)
			JCudaTensor x534;
			x534 = x527;
			x534.free();

		}
		System.out.println();
		System.out.println("average precision: " + x530/test_itr);
		System.out.println(); 
	}

}