package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;


public class Overfeat {
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
	static String network_dir = "src/main/java/deepdsl/gen/overfeat";
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
	// (Lmdb(1000000,10000,Win32,1000),false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{128, 3, 224, 224}, platform, 1000, true);
	// (Lmdb(1000000,10000,Win32,1000),true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{128, 3, 224, 224}, platform, 1000, false);
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
	// Precision(Accuracy(1))
	static float x410;
	// V_cv1_B
	static JCudaTensor x325 = JTensor.constFloat(0.0f, 96).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x319 = JTensor.constFloat(0.0f, 96, 3, 11, 11).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x290 = JTensor.constFloat(0.0f, 256).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x297 = JTensor.constFloat(0.0f, 256, 96, 5, 5).asJCudaTensor();
	// V_cv3_B
	static JCudaTensor x264 = JTensor.constFloat(0.0f, 512).asJCudaTensor();
	// V_cv3_W
	static JCudaTensor x268 = JTensor.constFloat(0.0f, 512, 256, 3, 3).asJCudaTensor();
	// V_cv4_B
	static JCudaTensor x246 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_cv4_W
	static JCudaTensor x241 = JTensor.constFloat(0.0f, 1024, 512, 3, 3).asJCudaTensor();
	// V_cv5_B
	static JCudaTensor x223 = JTensor.constFloat(0.0f, 1024).asJCudaTensor();
	// V_cv5_W
	static JCudaTensor x218 = JTensor.constFloat(0.0f, 1024, 1024, 3, 3).asJCudaTensor();
	// V_fc6_B
	static JCudaTensor x188 = JTensor.constFloat(0.0f, 3072).asJCudaTensor();
	// V_fc6_W
	static JCudaTensor x192 = JTensor.constFloat(0.0f, 3072, 36864).asJCudaTensor();
	// V_fc7_B
	static JCudaTensor x168 = JTensor.constFloat(0.0f, 4096).asJCudaTensor();
	// V_fc7_W
	static JCudaTensor x162 = JTensor.constFloat(0.0f, 4096, 3072).asJCudaTensor();
	// V_fc8_B
	static JCudaTensor x145 = JTensor.constFloat(0.0f, 1000).asJCudaTensor();
	// V_fc8_W
	static JCudaTensor x149 = JTensor.constFloat(0.0f, 1000, 4096).asJCudaTensor();
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
		x241.free();
		x64.free();
		x18.free();
		x19.free();
		x98.free();
		x188.free();
		x319.free();
		x297.free();
		x145.free();
		x54.free();
		x65.free();
		x192.free();
		x290.free();
		x223.free();
		x45.free();
		x78.free();
		x55.free();
		x88.free();
		x92.free();
		x82.free();
		x32.free();
		x268.free();
		x264.free();
		x149.free();
		x325.free();
		x44.free();
		x246.free();
		x162.free();
		x102.free();
		x31.free();
		x218.free();
		x168.free();
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

			// val X412 = Cuda(X)
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x3;
			x7 = x8.asJCudaTensor();

			// val X436 = Cuda(Indicator(Y, 1000))
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x4.asIndicator(1000);
			x9 = x10.asJCudaTensor();

			// val X210 = - X436.copy
			JCudaTensor x11;
			JCudaTensor x12;
			float x13;
			x12 = x9;
			x12 = x12.clone();
			x13 = -1;
			x11 = x12.times_i(x13);

			// val X413 = Convolv(4,0)(X412,cv1_W,cv1_B)
			JCudaTensor x14;
			JCudaTensor x15, x16, x17;
			x15 = x7;
			x16 = x18;
			x17 = x19;
			x14 = x20.forward(x15, x16, x17);

			// val X414 = ReLU()(X413)
			JCudaTensor x21;
			JCudaTensor x22;
			x22 = x14;
			x21 = x23.forward(x22);

			// val X415 = Pooling(2,2,0,true)(X414)
			JCudaTensor x24;
			JCudaTensor x25;
			x25 = x21;
			x24 = x26.forward(x25);

			// val X416 = Convolv(1,2)(X415,cv2_W,cv2_B)
			JCudaTensor x27;
			JCudaTensor x28, x29, x30;
			x28 = x24;
			x29 = x31;
			x30 = x32;
			x27 = x33.forward(x28, x29, x30);

			// val X417 = ReLU()(X416)
			JCudaTensor x34;
			JCudaTensor x35;
			x35 = x27;
			x34 = x36.forward(x35);

			// val X418 = Pooling(2,2,0,true)(X417)
			JCudaTensor x37;
			JCudaTensor x38;
			x38 = x34;
			x37 = x39.forward(x38);

			// val X419 = Convolv(1,1)(X418,cv3_W,cv3_B)
			JCudaTensor x40;
			JCudaTensor x41, x42, x43;
			x41 = x37;
			x42 = x44;
			x43 = x45;
			x40 = x46.forward(x41, x42, x43);

			// val X420 = ReLU()(X419)
			JCudaTensor x47;
			JCudaTensor x48;
			x48 = x40;
			x47 = x49.forward(x48);

			// val X421 = Convolv(1,1)(X420,cv4_W,cv4_B)
			JCudaTensor x50;
			JCudaTensor x51, x52, x53;
			x51 = x47;
			x52 = x54;
			x53 = x55;
			x50 = x56.forward(x51, x52, x53);

			// val X422 = ReLU()(X421)
			JCudaTensor x57;
			JCudaTensor x58;
			x58 = x50;
			x57 = x59.forward(x58);

			// val X423 = Convolv(1,1)(X422,cv5_W,cv5_B)
			JCudaTensor x60;
			JCudaTensor x61, x62, x63;
			x61 = x57;
			x62 = x64;
			x63 = x65;
			x60 = x66.forward(x61, x62, x63);

			// val X424 = ReLU()(X423)
			JCudaTensor x67;
			JCudaTensor x68;
			x68 = x60;
			x67 = x59.forward(x68);

			// val X425 = Pooling(2,2,0,true)(X424)
			JCudaTensor x69;
			JCudaTensor x70;
			x70 = x67;
			x69 = x71.forward(x70);

			// val X426 = (X425[1><3])(i | @) * (fc6_W)(j | @)
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

			// val X428 = (X426 + (i) => fc6_B)
			JCudaTensor x79;
			JCudaTensor x80, x81;
			x80 = x72;
			x81 = x82;
			x79 = x81.copy(128, x80);

			// val X429 = (X428)(i | @) * (fc7_W)(j | @)
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

			// val X431 = (X429 + (i) => fc7_B)
			JCudaTensor x89;
			JCudaTensor x90, x91;
			x90 = x83;
			x91 = x92;
			x89 = x91.copy(128, x90);

			// val X432 = (X431)(i | @) * (fc8_W)(j | @)
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

			// val X434 = (X432 + (i) => fc8_B)
			JCudaTensor x99;
			JCudaTensor x100, x101;
			x100 = x93;
			x101 = x102;
			x99 = x101.copy(128, x100);

			// val X435 = LogSoftmax()(X434)
			JCudaTensor x103;
			JCudaTensor x104;
			x104 = x99;
			x103 = x105.forward(x104);

			// Dealloc(X434)
			JCudaTensor x106;
			x106 = x99;
			x106.free();

			// val X211 = (X210 / |128|)
			JCudaTensor x107;
			JCudaTensor x108;
			float x109;
			x108 = x11;
			float x110;
			x110 = 128;
			x109 = 1 / x110;
			x107 = x108.times_i(x109);

			// Cost(((0 - (X436 . X435)) / |128|))
			float x111;
			float x112;
			float x113;
			float x114;
			JCudaTensor x115, x116;
			x115 = x9;
			x116 = x103;
			x114 = x115.dot(x116);
			x112 = - x114;
			x113 = 128;
			x111 = x112 / x113;
			System.out.println(x5 + " " + x111);
			if (Float.isNaN(x111)) { System.exit(-1); }

			// Dealloc(X436)
			JCudaTensor x117;
			x117 = x9;
			x117.free();

			// val X236 = X211 * d_LogSoftmax()(X435)/d_X434
			JCudaTensor x118;
			JCudaTensor x119, x120;
			x119 = x107;
			x120 = x103;
			x118 = x105.backward(x119, x120);

			// Dealloc(X211)
			JCudaTensor x121;
			x121 = x107;
			x121.free();

			// Dealloc(X435)
			JCudaTensor x122;
			x122 = x103;
			x122.free();

			// val m1 = (i277) => fc8_W[@, i277]
			JCudaMatrix x123;
			JCudaTensor x124;
			x124 = x98;
			x123 = x124.asMatrix(1, false);

			// val X256 = (X236)(i276 | @) * m1
			JCudaTensor x125;
			JCudaMatrix x126;
			JCudaMatrix x127;
			JCudaTensor x128;
			x128 = x118;
			x126 = x128.asMatrix(1, true);
			x127 = x123;
			x125 = x126.times(x127);

			// val m2 = (i281) => fc7_W[@, i281]
			JCudaMatrix x129;
			JCudaTensor x130;
			x130 = x88;
			x129 = x130.asMatrix(1, false);

			// val m25 = (i17) => X236[@, i17]
			JCudaMatrix x131;
			JCudaTensor x132;
			x132 = x118;
			x131 = x132.asMatrix(1, false);

			// val m27 = (i21) => X431[@, i21]
			JCudaMatrix x133;
			JCudaTensor x134;
			x134 = x89;
			x133 = x134.asMatrix(1, false);

			// val m22 = (i24) => X256[@, i24]
			JCudaMatrix x135;
			JCudaTensor x136;
			x136 = x125;
			x135 = x136.asMatrix(1, false);

			// val m3 = (i285) => fc6_W[@, i285]
			JCudaMatrix x137;
			JCudaTensor x138;
			x138 = x78;
			x137 = x138.asMatrix(1, false);

			// val m24 = (i28) => X428[@, i28]
			JCudaMatrix x139;
			JCudaTensor x140;
			x140 = x79;
			x139 = x140.asMatrix(1, false);

			// val X257 = (X256)(i280 | @) * m2
			JCudaTensor x141;
			JCudaMatrix x142;
			JCudaMatrix x143;
			JCudaTensor x144;
			x144 = x125;
			x142 = x144.asMatrix(1, true);
			x143 = x129;
			x141 = x142.times(x143);

			// V_fc8_B <~~ Sum(m25)
			float x146, x147;
			x146 = lrn_rate_1;
			x147 = momentum;
			JCudaMatrix x148;
			x148 = x131;
			x148.sum(x145, x146, x147);

			// V_fc8_W <~~ m25 * m27
			float x150, x151;
			x150 = lrn_rate_1;
			x151 = momentum;
			JCudaMatrix x152;
			JCudaMatrix x153;
			x152 = x131;
			x153 = x133;
			x152.times(x153, x149, x150, x151);

			// Dealloc(X236)
			JCudaTensor x154;
			x154 = x118;
			x154.free();

			// Dealloc(X431)
			JCudaTensor x155;
			x155 = x89;
			x155.free();

			// fc8_B <~~ V_fc8_B
			float x156, x157;
			x156 = 1;
			x157 = decay_1;
			JCudaTensor x158;
			x158 = x145;
			x102.update(x158, x156, x157);

			// fc8_W <~~ V_fc8_W
			float x159, x160;
			x159 = 1;
			x160 = decay_1;
			JCudaTensor x161;
			x161 = x149;
			x98.update(x161, x159, x160);

			// V_fc7_W <~~ m22 * m24
			float x163, x164;
			x163 = lrn_rate_1;
			x164 = momentum;
			JCudaMatrix x165;
			JCudaMatrix x166;
			x165 = x135;
			x166 = x139;
			x165.times(x166, x162, x163, x164);

			// Dealloc(X428)
			JCudaTensor x167;
			x167 = x79;
			x167.free();

			// V_fc7_B <~~ Sum(m22)
			float x169, x170;
			x169 = lrn_rate_1;
			x170 = momentum;
			JCudaMatrix x171;
			x171 = x135;
			x171.sum(x168, x169, x170);

			// Dealloc(X256)
			JCudaTensor x172;
			x172 = x125;
			x172.free();

			// val m21 = (i37) => X425[1><3][@, i37]
			JCudaMatrix x173;
			JCudaTensor x174;
			JCudaTensor x175;
			x175 = x69;
			x174 = x175.flatten(1, new int[]{1024, 6, 6});
			x173 = x174.asMatrix(1, false);

			// val m19 = (i33) => X257[@, i33]
			JCudaMatrix x176;
			JCudaTensor x177;
			x177 = x141;
			x176 = x177.asMatrix(1, false);

			// val X258 = (X257)(i284 | @) * m3
			JCudaTensor x178;
			JCudaMatrix x179;
			JCudaMatrix x180;
			JCudaTensor x181;
			x181 = x141;
			x179 = x181.asMatrix(1, true);
			x180 = x137;
			x178 = x179.times(x180);

			// fc7_B <~~ V_fc7_B
			float x182, x183;
			x182 = 1;
			x183 = decay_1;
			JCudaTensor x184;
			x184 = x168;
			x92.update(x184, x182, x183);

			// fc7_W <~~ V_fc7_W
			float x185, x186;
			x185 = 1;
			x186 = decay_1;
			JCudaTensor x187;
			x187 = x162;
			x88.update(x187, x185, x186);

			// V_fc6_B <~~ Sum(m19)
			float x189, x190;
			x189 = lrn_rate_1;
			x190 = momentum;
			JCudaMatrix x191;
			x191 = x176;
			x191.sum(x188, x189, x190);

			// V_fc6_W <~~ m19 * m21
			float x193, x194;
			x193 = lrn_rate_1;
			x194 = momentum;
			JCudaMatrix x195;
			JCudaMatrix x196;
			x195 = x176;
			x196 = x173;
			x195.times(x196, x192, x193, x194);

			// Dealloc(X257)
			JCudaTensor x197;
			x197 = x141;
			x197.free();

			// val X260 = X258[1<>3] * d_Pooling(2,2,0,true)(X425,X424)/d_X424
			JCudaTensor x198;
			JCudaTensor x199, x200, x201;
			JCudaTensor x202;
			x202 = x178;
			x199 = x202.unflatten(1, new int[]{1024, 6, 6});
			x200 = x69;
			x201 = x67;
			x198 = x71.backward(x199, x200, x201);

			// Dealloc(X258)
			JCudaTensor x203;
			x203 = x178;
			x203.free();

			// Dealloc(X425)
			JCudaTensor x204;
			x204 = x69;
			x204.free();

			// fc6_B <~~ V_fc6_B
			float x205, x206;
			x205 = 1;
			x206 = decay_1;
			JCudaTensor x207;
			x207 = x188;
			x82.update(x207, x205, x206);

			// fc6_W <~~ V_fc6_W
			float x208, x209;
			x208 = 1;
			x209 = decay_1;
			JCudaTensor x210;
			x210 = x192;
			x78.update(x210, x208, x209);

			// val X262 = X260 * d_ReLU()(X424)/d_X423
			JCudaTensor x211;
			JCudaTensor x212, x213;
			x212 = x198;
			x213 = x67;
			x211 = x59.backward(x212, x213);

			// Dealloc(X424)
			JCudaTensor x214;
			x214 = x67;
			x214.free();

			// val X263 = X262 * d_Convolv(1,1)(cv5_W)/d_X422
			JCudaTensor x215;
			JCudaTensor x216, x217;
			x216 = x211;
			x217 = x64;
			x215 = x66.backward_data(x216, x217);

			// V_cv5_W <~~ X262 * d_Convolv(1,1)(X422)/d_cv5_W
			float x219, x220;
			x219 = lrn_rate_1;
			x220 = momentum;
			JCudaTensor x221, x222;
			x221 = x211;
			x222 = x57;
			x66.backward_filter(x221, x222, x218, x219, x220);

			// V_cv5_B <~~ X262 * d_Convolv(1,1)()/d_cv5_B
			float x224, x225;
			x224 = lrn_rate_2;
			x225 = momentum;
			JCudaTensor x226;
			x226 = x211;
			x66.backward_bias(x226, x223, x224, x225);

			// Dealloc(X262)
			JCudaTensor x227;
			x227 = x211;
			x227.free();

			// cv5_W <~~ V_cv5_W
			float x228, x229;
			x228 = 1;
			x229 = decay_1;
			JCudaTensor x230;
			x230 = x218;
			x64.update(x230, x228, x229);

			// cv5_B <~~ V_cv5_B
			float x231, x232;
			x231 = 1;
			x232 = 1;
			JCudaTensor x233;
			x233 = x223;
			x65.update(x233, x231, x232);

			// val X265 = X263 * d_ReLU()(X422)/d_X421
			JCudaTensor x234;
			JCudaTensor x235, x236;
			x235 = x215;
			x236 = x57;
			x234 = x59.backward(x235, x236);

			// Dealloc(X422)
			JCudaTensor x237;
			x237 = x57;
			x237.free();

			// val X266 = X265 * d_Convolv(1,1)(cv4_W)/d_X420
			JCudaTensor x238;
			JCudaTensor x239, x240;
			x239 = x234;
			x240 = x54;
			x238 = x56.backward_data(x239, x240);

			// V_cv4_W <~~ X265 * d_Convolv(1,1)(X420)/d_cv4_W
			float x242, x243;
			x242 = lrn_rate_1;
			x243 = momentum;
			JCudaTensor x244, x245;
			x244 = x234;
			x245 = x47;
			x56.backward_filter(x244, x245, x241, x242, x243);

			// V_cv4_B <~~ X265 * d_Convolv(1,1)()/d_cv4_B
			float x247, x248;
			x247 = lrn_rate_2;
			x248 = momentum;
			JCudaTensor x249;
			x249 = x234;
			x56.backward_bias(x249, x246, x247, x248);

			// Dealloc(X265)
			JCudaTensor x250;
			x250 = x234;
			x250.free();

			// cv4_W <~~ V_cv4_W
			float x251, x252;
			x251 = 1;
			x252 = decay_1;
			JCudaTensor x253;
			x253 = x241;
			x54.update(x253, x251, x252);

			// cv4_B <~~ V_cv4_B
			float x254, x255;
			x254 = 1;
			x255 = 1;
			JCudaTensor x256;
			x256 = x246;
			x55.update(x256, x254, x255);

			// val X268 = X266 * d_ReLU()(X420)/d_X419
			JCudaTensor x257;
			JCudaTensor x258, x259;
			x258 = x238;
			x259 = x47;
			x257 = x49.backward(x258, x259);

			// Dealloc(X420)
			JCudaTensor x260;
			x260 = x47;
			x260.free();

			// val X269 = X268 * d_Convolv(1,1)(cv3_W)/d_X418
			JCudaTensor x261;
			JCudaTensor x262, x263;
			x262 = x257;
			x263 = x44;
			x261 = x46.backward_data(x262, x263);

			// V_cv3_B <~~ X268 * d_Convolv(1,1)()/d_cv3_B
			float x265, x266;
			x265 = lrn_rate_2;
			x266 = momentum;
			JCudaTensor x267;
			x267 = x257;
			x46.backward_bias(x267, x264, x265, x266);

			// V_cv3_W <~~ X268 * d_Convolv(1,1)(X418)/d_cv3_W
			float x269, x270;
			x269 = lrn_rate_1;
			x270 = momentum;
			JCudaTensor x271, x272;
			x271 = x257;
			x272 = x37;
			x46.backward_filter(x271, x272, x268, x269, x270);

			// Dealloc(X268)
			JCudaTensor x273;
			x273 = x257;
			x273.free();

			// cv3_B <~~ V_cv3_B
			float x274, x275;
			x274 = 1;
			x275 = 1;
			JCudaTensor x276;
			x276 = x264;
			x45.update(x276, x274, x275);

			// cv3_W <~~ V_cv3_W
			float x277, x278;
			x277 = 1;
			x278 = decay_1;
			JCudaTensor x279;
			x279 = x268;
			x44.update(x279, x277, x278);

			// val X271 = X269 * d_Pooling(2,2,0,true)(X418,X417)/d_X417
			JCudaTensor x280;
			JCudaTensor x281, x282, x283;
			x281 = x261;
			x282 = x37;
			x283 = x34;
			x280 = x39.backward(x281, x282, x283);

			// Dealloc(X269)
			JCudaTensor x284;
			x284 = x261;
			x284.free();

			// Dealloc(X418)
			JCudaTensor x285;
			x285 = x37;
			x285.free();

			// val X273 = X271 * d_ReLU()(X417)/d_X416
			JCudaTensor x286;
			JCudaTensor x287, x288;
			x287 = x280;
			x288 = x34;
			x286 = x36.backward(x287, x288);

			// Dealloc(X417)
			JCudaTensor x289;
			x289 = x34;
			x289.free();

			// V_cv2_B <~~ X273 * d_Convolv(1,2)()/d_cv2_B
			float x291, x292;
			x291 = lrn_rate_2;
			x292 = momentum;
			JCudaTensor x293;
			x293 = x286;
			x33.backward_bias(x293, x290, x291, x292);

			// val X274 = X273 * d_Convolv(1,2)(cv2_W)/d_X415
			JCudaTensor x294;
			JCudaTensor x295, x296;
			x295 = x286;
			x296 = x31;
			x294 = x33.backward_data(x295, x296);

			// V_cv2_W <~~ X273 * d_Convolv(1,2)(X415)/d_cv2_W
			float x298, x299;
			x298 = lrn_rate_1;
			x299 = momentum;
			JCudaTensor x300, x301;
			x300 = x286;
			x301 = x24;
			x33.backward_filter(x300, x301, x297, x298, x299);

			// Dealloc(X273)
			JCudaTensor x302;
			x302 = x286;
			x302.free();

			// cv2_B <~~ V_cv2_B
			float x303, x304;
			x303 = 1;
			x304 = 1;
			JCudaTensor x305;
			x305 = x290;
			x32.update(x305, x303, x304);

			// cv2_W <~~ V_cv2_W
			float x306, x307;
			x306 = 1;
			x307 = decay_1;
			JCudaTensor x308;
			x308 = x297;
			x31.update(x308, x306, x307);

			// val X276 = X274 * d_Pooling(2,2,0,true)(X415,X414)/d_X414
			JCudaTensor x309;
			JCudaTensor x310, x311, x312;
			x310 = x294;
			x311 = x24;
			x312 = x21;
			x309 = x26.backward(x310, x311, x312);

			// Dealloc(X274)
			JCudaTensor x313;
			x313 = x294;
			x313.free();

			// Dealloc(X415)
			JCudaTensor x314;
			x314 = x24;
			x314.free();

			// val X278 = X276 * d_ReLU()(X414)/d_X413
			JCudaTensor x315;
			JCudaTensor x316, x317;
			x316 = x309;
			x317 = x21;
			x315 = x23.backward(x316, x317);

			// Dealloc(X414)
			JCudaTensor x318;
			x318 = x21;
			x318.free();

			// V_cv1_W <~~ X278 * d_Convolv(4,0)(X412)/d_cv1_W
			float x320, x321;
			x320 = lrn_rate_1;
			x321 = momentum;
			JCudaTensor x322, x323;
			x322 = x315;
			x323 = x7;
			x20.backward_filter(x322, x323, x319, x320, x321);

			// Dealloc(X412)
			JCudaTensor x324;
			x324 = x7;
			x324.free();

			// V_cv1_B <~~ X278 * d_Convolv(4,0)()/d_cv1_B
			float x326, x327;
			x326 = lrn_rate_2;
			x327 = momentum;
			JCudaTensor x328;
			x328 = x315;
			x20.backward_bias(x328, x325, x326, x327);

			// Dealloc(X278)
			JCudaTensor x329;
			x329 = x315;
			x329.free();

			// cv1_W <~~ V_cv1_W
			float x330, x331;
			x330 = 1;
			x331 = decay_1;
			JCudaTensor x332;
			x332 = x319;
			x18.update(x332, x330, x331);

			// cv1_B <~~ V_cv1_B
			float x333, x334;
			x333 = 1;
			x334 = 1;
			JCudaTensor x335;
			x335 = x325;
			x19.update(x335, x333, x334);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X437 = Cuda(X)
			JCudaTensor x336;
			JTensorFloat x337;
			x337 = x3;
			x336 = x337.asJCudaTensor();

			// val X438 = Convolv(4,0)(X437,cv1_W,cv1_B)
			JCudaTensor x338;
			JCudaTensor x339, x340, x341;
			x339 = x336;
			x340 = x18;
			x341 = x19;
			x338 = x20.forward(x339, x340, x341);

			// Dealloc(X437)
			JCudaTensor x342;
			x342 = x336;
			x342.free();

			// val X439 = ReLU()(X438)
			JCudaTensor x343;
			JCudaTensor x344;
			x344 = x338;
			x343 = x23.forward(x344);

			// val X440 = Pooling(2,2,0,true)(X439)
			JCudaTensor x345;
			JCudaTensor x346;
			x346 = x343;
			x345 = x26.forward(x346);

			// Dealloc(X439)
			JCudaTensor x347;
			x347 = x343;
			x347.free();

			// val X441 = Convolv(1,2)(X440,cv2_W,cv2_B)
			JCudaTensor x348;
			JCudaTensor x349, x350, x351;
			x349 = x345;
			x350 = x31;
			x351 = x32;
			x348 = x33.forward(x349, x350, x351);

			// Dealloc(X440)
			JCudaTensor x352;
			x352 = x345;
			x352.free();

			// val X442 = ReLU()(X441)
			JCudaTensor x353;
			JCudaTensor x354;
			x354 = x348;
			x353 = x36.forward(x354);

			// val X443 = Pooling(2,2,0,true)(X442)
			JCudaTensor x355;
			JCudaTensor x356;
			x356 = x353;
			x355 = x39.forward(x356);

			// Dealloc(X442)
			JCudaTensor x357;
			x357 = x353;
			x357.free();

			// val X444 = Convolv(1,1)(X443,cv3_W,cv3_B)
			JCudaTensor x358;
			JCudaTensor x359, x360, x361;
			x359 = x355;
			x360 = x44;
			x361 = x45;
			x358 = x46.forward(x359, x360, x361);

			// Dealloc(X443)
			JCudaTensor x362;
			x362 = x355;
			x362.free();

			// val X445 = ReLU()(X444)
			JCudaTensor x363;
			JCudaTensor x364;
			x364 = x358;
			x363 = x49.forward(x364);

			// val X446 = Convolv(1,1)(X445,cv4_W,cv4_B)
			JCudaTensor x365;
			JCudaTensor x366, x367, x368;
			x366 = x363;
			x367 = x54;
			x368 = x55;
			x365 = x56.forward(x366, x367, x368);

			// Dealloc(X445)
			JCudaTensor x369;
			x369 = x363;
			x369.free();

			// val X447 = ReLU()(X446)
			JCudaTensor x370;
			JCudaTensor x371;
			x371 = x365;
			x370 = x59.forward(x371);

			// val X448 = Convolv(1,1)(X447,cv5_W,cv5_B)
			JCudaTensor x372;
			JCudaTensor x373, x374, x375;
			x373 = x370;
			x374 = x64;
			x375 = x65;
			x372 = x66.forward(x373, x374, x375);

			// Dealloc(X447)
			JCudaTensor x376;
			x376 = x370;
			x376.free();

			// val X449 = ReLU()(X448)
			JCudaTensor x377;
			JCudaTensor x378;
			x378 = x372;
			x377 = x59.forward(x378);

			// val X450 = Pooling(2,2,0,true)(X449)
			JCudaTensor x379;
			JCudaTensor x380;
			x380 = x377;
			x379 = x71.forward(x380);

			// Dealloc(X449)
			JCudaTensor x381;
			x381 = x377;
			x381.free();

			// val X451 = (X450[1><3])(i | @) * (fc6_W)(j | @)
			JCudaTensor x382;
			JCudaMatrix x383;
			JCudaMatrix x384;
			JCudaTensor x385;
			JCudaTensor x386;
			x386 = x379;
			x385 = x386.flatten(1, new int[]{1024, 6, 6});
			x383 = x385.asMatrix(1, true);
			JCudaTensor x387;
			x387 = x78;
			x384 = x387.asMatrix(1, true);
			x382 = x383.times(x384);

			// Dealloc(X450)
			JCudaTensor x388;
			x388 = x379;
			x388.free();

			// val X453 = (X451 + (i) => fc6_B)
			JCudaTensor x389;
			JCudaTensor x390, x391;
			x390 = x382;
			x391 = x82;
			x389 = x391.copy(128, x390);

			// val X454 = (X453)(i | @) * (fc7_W)(j | @)
			JCudaTensor x392;
			JCudaMatrix x393;
			JCudaMatrix x394;
			JCudaTensor x395;
			x395 = x389;
			x393 = x395.asMatrix(1, true);
			JCudaTensor x396;
			x396 = x88;
			x394 = x396.asMatrix(1, true);
			x392 = x393.times(x394);

			// Dealloc(X453)
			JCudaTensor x397;
			x397 = x389;
			x397.free();

			// val X456 = (X454 + (i) => fc7_B)
			JCudaTensor x398;
			JCudaTensor x399, x400;
			x399 = x392;
			x400 = x92;
			x398 = x400.copy(128, x399);

			// val X457 = (X456)(i | @) * (fc8_W)(j | @)
			JCudaTensor x401;
			JCudaMatrix x402;
			JCudaMatrix x403;
			JCudaTensor x404;
			x404 = x398;
			x402 = x404.asMatrix(1, true);
			JCudaTensor x405;
			x405 = x98;
			x403 = x405.asMatrix(1, true);
			x401 = x402.times(x403);

			// Dealloc(X456)
			JCudaTensor x406;
			x406 = x398;
			x406.free();

			// val X459 = (X457 + (i) => fc8_B)
			JCudaTensor x407;
			JCudaTensor x408, x409;
			x408 = x401;
			x409 = x102;
			x407 = x409.copy(128, x408);

			// Precision(Accuracy(1))
			float x411;
			JCudaTensor x412;
			JTensorFloat x413;
			x412 = x407;
			x413 = x4;
			x411 = x412.accuracy(x413, 1);
			System.out.println(x5 + " test precision "  + x411);
			x410 += x411;

			// Dealloc(X459)
			JCudaTensor x414;
			x414 = x407;
			x414.free();

		}
		System.out.println();
		System.out.println("average precision: " + x410/10);
		System.out.println(); 
	}

}