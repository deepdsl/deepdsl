package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Lenet {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// decay_1
	static float decay_1 = 0.999995f;
	// lrn_rate_1
	static float lrn_rate_1 = -0.01f;
	// momentum
	static float momentum = 0.9f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/lenet";
	// test_itr
	static int test_itr = 10;
	// train_itr
	static int train_itr = 100;

	// (Convolv(1,0),List(List(500, 1, 28, 28), List(20, 1, 5, 5), List(20)))
	static JCudnnConvolution x15 = new JCudnnConvolution(new int[]{500,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
	// (Convolv(1,0),List(List(500, 20, 12, 12), List(50, 20, 5, 5), List(50)))
	static JCudnnConvolution x25 = new JCudnnConvolution(new int[]{500,20,12,12},new int[]{50,20,5,5},new int[]{50}, 1, 0);
	// (MNIST,false)
	static MnistFactory x2 = MnistFactory.getFactory(false, new int[]{500, 1, 28, 28});
	// (MNIST,true)
	static MnistFactory x1 = MnistFactory.getFactory(true, new int[]{500, 1, 28, 28});
	// (Pooling(2,2,0,true),List(List(500, 20, 24, 24)))
	static JCudnnPooling x18 = new JCudnnPooling(new int[]{500,20,24,24}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(500, 50, 8, 8)))
	static JCudnnPooling x28 = new JCudnnPooling(new int[]{500,50,8,8}, 2, 2, 0, PoolingType.MAX);
	// (ReLU(),List(List(500, 500)))
	static JCudnnActivation x42 = new JCudnnActivation(new int[]{500,500}, ActivationMode.RELU);
	// (Softmax(),List(List(500, 10)))
	static JCudnnSoftmax x55 = new JCudnnSoftmax(new int[]{500,10}, SoftmaxAlgorithm.ACCURATE);
	// Precision(Accuracy(1))
	static float x235;
	// V_cv1_B
	static JCudaTensor x185 = JTensor.constFloat(0.0f, 20).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x179 = JTensor.constFloat(0.0f, 20, 1, 5, 5).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x161 = JTensor.constFloat(0.0f, 50).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x156 = JTensor.constFloat(0.0f, 50, 20, 5, 5).asJCudaTensor();
	// V_fc1_B
	static JCudaTensor x142 = JTensor.constFloat(0.0f, 500).asJCudaTensor();
	// V_fc1_W
	static JCudaTensor x136 = JTensor.constFloat(0.0f, 500, 800).asJCudaTensor();
	// V_fc2_B
	static JCudaTensor x98 = JTensor.constFloat(0.0f, 10).asJCudaTensor();
	// V_fc2_W
	static JCudaTensor x102 = JTensor.constFloat(0.0f, 10, 500).asJCudaTensor();
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;
	// cv1_B
	static JCudaTensor x14 = JTensor.constFloat(0.0f, 20).load(network_dir + "/cv1_B").asJCudaTensor();
	// cv1_W
	static JCudaTensor x13 = JTensor.randomFloat(-0.28284273f, 0.28284273f, 20, 1, 5, 5).load(network_dir + "/cv1_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x24 = JTensor.constFloat(0.0f, 50).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x23 = JTensor.randomFloat(-0.06324555f, 0.06324555f, 50, 20, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
	// fc1_B
	static JCudaTensor x39 = JTensor.constFloat(0.0f, 500).load(network_dir + "/fc1_B").asJCudaTensor();
	// fc1_W
	static JCudaTensor x35 = JTensor.randomFloat(-0.05f, 0.05f, 500, 800).load(network_dir + "/fc1_W").asJCudaTensor();
	// fc2_B
	static JCudaTensor x52 = JTensor.constFloat(0.0f, 10).load(network_dir + "/fc2_B").asJCudaTensor();
	// fc2_W
	static JCudaTensor x48 = JTensor.randomFloat(-0.06324555f, 0.06324555f, 10, 500).load(network_dir + "/fc2_W").asJCudaTensor();

	public static void main(String[] args){
		double t = System.nanoTime();
		train();
		System.out.println((System.nanoTime() - t) / 1.0E9);
		test();
		x13.save(network_dir + "/cv1_W");
		x48.save(network_dir + "/fc2_W");
		x23.save(network_dir + "/cv2_W");
		x35.save(network_dir + "/fc1_W");
		x39.save(network_dir + "/fc1_B");
		x24.save(network_dir + "/cv2_B");
		x52.save(network_dir + "/fc2_B");
		x14.save(network_dir + "/cv1_B");
		x24.free();
		x136.free();
		x39.free();
		x142.free();
		x98.free();
		x161.free();
		x35.free();
		x23.free();
		x52.free();
		x48.free();
		x13.free();
		x156.free();
		x14.free();
		x185.free();
		x102.free();
		x179.free();
		x15.free();
		x18.free();
		x25.free();
		x55.free();
		x28.free();
		x42.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void train() {
		for(int x5=0; x5<train_itr; x5++) {
			JTensorFloatTuple x6 =  x1.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X126 = Cuda(X)
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x3;
			x7 = x8.asJCudaTensor();

			// val X127 = Convolv(1,0)(X126,cv1_W,cv1_B)
			JCudaTensor x9;
			JCudaTensor x10, x11, x12;
			x10 = x7;
			x11 = x13;
			x12 = x14;
			x9 = x15.forward(x10, x11, x12);

			// val X128 = Pooling(2,2,0,true)(X127)
			JCudaTensor x16;
			JCudaTensor x17;
			x17 = x9;
			x16 = x18.forward(x17);

			// val X129 = Convolv(1,0)(X128,cv2_W,cv2_B)
			JCudaTensor x19;
			JCudaTensor x20, x21, x22;
			x20 = x16;
			x21 = x23;
			x22 = x24;
			x19 = x25.forward(x20, x21, x22);

			// val X130 = Pooling(2,2,0,true)(X129)
			JCudaTensor x26;
			JCudaTensor x27;
			x27 = x19;
			x26 = x28.forward(x27);

			// val X131 = (X130[1><3])(i | @) * (fc1_W)(j | @)
			JCudaTensor x29;
			JCudaMatrix x30;
			JCudaMatrix x31;
			JCudaTensor x32;
			JCudaTensor x33;
			x33 = x26;
			x32 = x33.flatten(1, new int[]{50, 4, 4});
			x30 = x32.asMatrix(1, true);
			JCudaTensor x34;
			x34 = x35;
			x31 = x34.asMatrix(1, true);
			x29 = x30.times(x31);

			// val X133 = (X131 + (i) => fc1_B)
			JCudaTensor x36;
			JCudaTensor x37, x38;
			x37 = x29;
			x38 = x39;
			x36 = x38.copy(500, x37);

			// val X134 = ReLU()(X133)
			JCudaTensor x40;
			JCudaTensor x41;
			x41 = x36;
			x40 = x42.forward(x41);

			// val X135 = (X134)(i | @) * (fc2_W)(j | @)
			JCudaTensor x43;
			JCudaMatrix x44;
			JCudaMatrix x45;
			JCudaTensor x46;
			x46 = x40;
			x44 = x46.asMatrix(1, true);
			JCudaTensor x47;
			x47 = x48;
			x45 = x47.asMatrix(1, true);
			x43 = x44.times(x45);

			// val X137 = (X135 + (i) => fc2_B)
			JCudaTensor x49;
			JCudaTensor x50, x51;
			x50 = x43;
			x51 = x52;
			x49 = x51.copy(500, x50);

			// val X138 = Softmax()(X137)
			JCudaTensor x53;
			JCudaTensor x54;
			x54 = x49;
			x53 = x55.forward(x54);

			// Dealloc(X137)
			JCudaTensor x56;
			x56 = x49;
			x56.free();

			// val X139 = Cuda(Indicator(Y, 10))
			JCudaTensor x57;
			JTensorFloat x58;
			x58 = x4.asIndicator(10);
			x57 = x58.asJCudaTensor();

			// val X140 = Log X138.copy
			JCudaTensor x59;
			JCudaTensor x60;
			x60 = x53;
			x60 = x60.clone();
			x59 = x60.log();

			// val X72 = 1/(X138.copy)
			JCudaTensor x61;
			JCudaTensor x62;
			float x63;
			x62 = x53;
			x62 = x62.clone();
			x63 = -1;
			x61 = x62.pow(x63);

			// Cost(((0 - (X139 . X140)) / |500|))
			float x64;
			float x65;
			float x66;
			float x67;
			JCudaTensor x68, x69;
			x68 = x57;
			x69 = x59;
			x67 = x68.dot(x69);
			x65 = - x67;
			x66 = 500;
			x64 = x65 / x66;
			System.out.println(x5 + " " + x64);
			if (Float.isNaN(x64)) { System.exit(-1); }

			// Dealloc(X140)
			JCudaTensor x70;
			x70 = x59;
			x70.free();

			// val X73 = X139.copy .* X72
			JCudaTensor x71;
			JCudaTensor x72, x73;
			x72 = x57;
			x72 = x72.clone();
			x73 = x61;
			x71 = x72.times_i(x73);

			// Dealloc(X72)
			JCudaTensor x74;
			x74 = x61;
			x74.free();

			// Dealloc(X139)
			JCudaTensor x75;
			x75 = x57;
			x75.free();

			// val X74 = - X73
			JCudaTensor x76;
			JCudaTensor x77;
			float x78;
			x77 = x71;
			x78 = -1;
			x76 = x77.times_i(x78);

			// val X75 = (X74 / |500|)
			JCudaTensor x79;
			JCudaTensor x80;
			float x81;
			x80 = x76;
			float x82;
			x82 = 500;
			x81 = 1 / x82;
			x79 = x80.times_i(x81);

			// val X77 = X75 * d_Softmax()(X138)/d_X137
			JCudaTensor x83;
			JCudaTensor x84, x85;
			x84 = x79;
			x85 = x53;
			x83 = x55.backward(x84, x85);

			// Dealloc(X75)
			JCudaTensor x86;
			x86 = x79;
			x86.free();

			// Dealloc(X138)
			JCudaTensor x87;
			x87 = x53;
			x87.free();

			// val m1 = (i89) => fc2_W[@, i89]
			JCudaMatrix x88;
			JCudaTensor x89;
			x89 = x48;
			x88 = x89.asMatrix(1, false);

			// val m6 = (i18) => X77[@, i18]
			JCudaMatrix x90;
			JCudaTensor x91;
			x91 = x83;
			x90 = x91.asMatrix(1, false);

			// val m7 = (i19) => X134[@, i19]
			JCudaMatrix x92;
			JCudaTensor x93;
			x93 = x40;
			x92 = x93.asMatrix(1, false);

			// val X87 = (X77)(i88 | @) * m1
			JCudaTensor x94;
			JCudaMatrix x95;
			JCudaMatrix x96;
			JCudaTensor x97;
			x97 = x83;
			x95 = x97.asMatrix(1, true);
			x96 = x88;
			x94 = x95.times(x96);

			// V_fc2_B <~~ Sum(m6)
			float x99, x100;
			x99 = lrn_rate_1;
			x100 = momentum;
			JCudaMatrix x101;
			x101 = x90;
			x101.sum(x98, x99, x100);

			// V_fc2_W <~~ m6 * m7
			float x103, x104;
			x103 = lrn_rate_1;
			x104 = momentum;
			JCudaMatrix x105;
			JCudaMatrix x106;
			x105 = x90;
			x106 = x92;
			x105.times(x106, x102, x103, x104);

			// Dealloc(X77)
			JCudaTensor x107;
			x107 = x83;
			x107.free();

			// val X89 = X87 * d_ReLU()(X134)/d_X133
			JCudaTensor x108;
			JCudaTensor x109, x110;
			x109 = x94;
			x110 = x40;
			x108 = x42.backward(x109, x110);

			// Dealloc(X134)
			JCudaTensor x111;
			x111 = x40;
			x111.free();

			// val m2 = (i93) => fc1_W[@, i93]
			JCudaMatrix x112;
			JCudaTensor x113;
			x113 = x35;
			x112 = x113.asMatrix(1, false);

			// fc2_B <~~ V_fc2_B
			float x114, x115;
			x114 = 1;
			x115 = decay_1;
			JCudaTensor x116;
			x116 = x98;
			x52.update(x116, x114, x115);

			// fc2_W <~~ V_fc2_W
			float x117, x118;
			x117 = 1;
			x118 = decay_1;
			JCudaTensor x119;
			x119 = x102;
			x48.update(x119, x117, x118);

			// val m8 = (i27) => X89[@, i27]
			JCudaMatrix x120;
			JCudaTensor x121;
			x121 = x108;
			x120 = x121.asMatrix(1, false);

			// val m9 = (i28) => X130[1><3][@, i28]
			JCudaMatrix x122;
			JCudaTensor x123;
			JCudaTensor x124;
			x124 = x26;
			x123 = x124.flatten(1, new int[]{50, 4, 4});
			x122 = x123.asMatrix(1, false);

			// val X90 = (X89)(i92 | @) * m2
			JCudaTensor x125;
			JCudaMatrix x126;
			JCudaMatrix x127;
			JCudaTensor x128;
			x128 = x108;
			x126 = x128.asMatrix(1, true);
			x127 = x112;
			x125 = x126.times(x127);

			// val X92 = X90[1<>3] * d_Pooling(2,2,0,true)(X130,X129)/d_X129
			JCudaTensor x129;
			JCudaTensor x130, x131, x132;
			JCudaTensor x133;
			x133 = x125;
			x130 = x133.unflatten(1, new int[]{50, 4, 4});
			x131 = x26;
			x132 = x19;
			x129 = x28.backward(x130, x131, x132);

			// Dealloc(X90)
			JCudaTensor x134;
			x134 = x125;
			x134.free();

			// Dealloc(X129)
			JCudaTensor x135;
			x135 = x19;
			x135.free();

			// V_fc1_W <~~ m8 * m9
			float x137, x138;
			x137 = lrn_rate_1;
			x138 = momentum;
			JCudaMatrix x139;
			JCudaMatrix x140;
			x139 = x120;
			x140 = x122;
			x139.times(x140, x136, x137, x138);

			// Dealloc(X130)
			JCudaTensor x141;
			x141 = x26;
			x141.free();

			// V_fc1_B <~~ Sum(m8)
			float x143, x144;
			x143 = lrn_rate_1;
			x144 = momentum;
			JCudaMatrix x145;
			x145 = x120;
			x145.sum(x142, x143, x144);

			// Dealloc(X89)
			JCudaTensor x146;
			x146 = x108;
			x146.free();

			// fc1_W <~~ V_fc1_W
			float x147, x148;
			x147 = 1;
			x148 = decay_1;
			JCudaTensor x149;
			x149 = x136;
			x35.update(x149, x147, x148);

			// fc1_B <~~ V_fc1_B
			float x150, x151;
			x150 = 1;
			x151 = decay_1;
			JCudaTensor x152;
			x152 = x142;
			x39.update(x152, x150, x151);

			// val X93 = X92 * d_Convolv(1,0)(cv2_W)/d_X128
			JCudaTensor x153;
			JCudaTensor x154, x155;
			x154 = x129;
			x155 = x23;
			x153 = x25.backward_data(x154, x155);

			// V_cv2_W <~~ X92 * d_Convolv(1,0)(X128)/d_cv2_W
			float x157, x158;
			x157 = lrn_rate_1;
			x158 = momentum;
			JCudaTensor x159, x160;
			x159 = x129;
			x160 = x16;
			x25.backward_filter(x159, x160, x156, x157, x158);

			// V_cv2_B <~~ X92 * d_Convolv(1,0)()/d_cv2_B
			float x162, x163;
			x162 = lrn_rate_1;
			x163 = momentum;
			JCudaTensor x164;
			x164 = x129;
			x25.backward_bias(x164, x161, x162, x163);

			// Dealloc(X92)
			JCudaTensor x165;
			x165 = x129;
			x165.free();

			// cv2_W <~~ V_cv2_W
			float x166, x167;
			x166 = 1;
			x167 = decay_1;
			JCudaTensor x168;
			x168 = x156;
			x23.update(x168, x166, x167);

			// cv2_B <~~ V_cv2_B
			float x169, x170;
			x169 = 1;
			x170 = decay_1;
			JCudaTensor x171;
			x171 = x161;
			x24.update(x171, x169, x170);

			// val X95 = X93 * d_Pooling(2,2,0,true)(X128,X127)/d_X127
			JCudaTensor x172;
			JCudaTensor x173, x174, x175;
			x173 = x153;
			x174 = x16;
			x175 = x9;
			x172 = x18.backward(x173, x174, x175);

			// Dealloc(X93)
			JCudaTensor x176;
			x176 = x153;
			x176.free();

			// Dealloc(X128)
			JCudaTensor x177;
			x177 = x16;
			x177.free();

			// Dealloc(X127)
			JCudaTensor x178;
			x178 = x9;
			x178.free();

			// V_cv1_W <~~ X95 * d_Convolv(1,0)(X126)/d_cv1_W
			float x180, x181;
			x180 = lrn_rate_1;
			x181 = momentum;
			JCudaTensor x182, x183;
			x182 = x172;
			x183 = x7;
			x15.backward_filter(x182, x183, x179, x180, x181);

			// Dealloc(X126)
			JCudaTensor x184;
			x184 = x7;
			x184.free();

			// V_cv1_B <~~ X95 * d_Convolv(1,0)()/d_cv1_B
			float x186, x187;
			x186 = lrn_rate_1;
			x187 = momentum;
			JCudaTensor x188;
			x188 = x172;
			x15.backward_bias(x188, x185, x186, x187);

			// Dealloc(X95)
			JCudaTensor x189;
			x189 = x172;
			x189.free();

			// cv1_W <~~ V_cv1_W
			float x190, x191;
			x190 = 1;
			x191 = decay_1;
			JCudaTensor x192;
			x192 = x179;
			x13.update(x192, x190, x191);

			// cv1_B <~~ V_cv1_B
			float x193, x194;
			x193 = 1;
			x194 = decay_1;
			JCudaTensor x195;
			x195 = x185;
			x14.update(x195, x193, x194);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X141 = Cuda(X)
			JCudaTensor x196;
			JTensorFloat x197;
			x197 = x3;
			x196 = x197.asJCudaTensor();

			// val X142 = Convolv(1,0)(X141,cv1_W,cv1_B)
			JCudaTensor x198;
			JCudaTensor x199, x200, x201;
			x199 = x196;
			x200 = x13;
			x201 = x14;
			x198 = x15.forward(x199, x200, x201);

			// Dealloc(X141)
			JCudaTensor x202;
			x202 = x196;
			x202.free();

			// val X143 = Pooling(2,2,0,true)(X142)
			JCudaTensor x203;
			JCudaTensor x204;
			x204 = x198;
			x203 = x18.forward(x204);

			// Dealloc(X142)
			JCudaTensor x205;
			x205 = x198;
			x205.free();

			// val X144 = Convolv(1,0)(X143,cv2_W,cv2_B)
			JCudaTensor x206;
			JCudaTensor x207, x208, x209;
			x207 = x203;
			x208 = x23;
			x209 = x24;
			x206 = x25.forward(x207, x208, x209);

			// Dealloc(X143)
			JCudaTensor x210;
			x210 = x203;
			x210.free();

			// val X145 = Pooling(2,2,0,true)(X144)
			JCudaTensor x211;
			JCudaTensor x212;
			x212 = x206;
			x211 = x28.forward(x212);

			// Dealloc(X144)
			JCudaTensor x213;
			x213 = x206;
			x213.free();

			// val X146 = (X145[1><3])(i | @) * (fc1_W)(j | @)
			JCudaTensor x214;
			JCudaMatrix x215;
			JCudaMatrix x216;
			JCudaTensor x217;
			JCudaTensor x218;
			x218 = x211;
			x217 = x218.flatten(1, new int[]{50, 4, 4});
			x215 = x217.asMatrix(1, true);
			JCudaTensor x219;
			x219 = x35;
			x216 = x219.asMatrix(1, true);
			x214 = x215.times(x216);

			// Dealloc(X145)
			JCudaTensor x220;
			x220 = x211;
			x220.free();

			// val X148 = (X146 + (i) => fc1_B)
			JCudaTensor x221;
			JCudaTensor x222, x223;
			x222 = x214;
			x223 = x39;
			x221 = x223.copy(500, x222);

			// val X149 = ReLU()(X148)
			JCudaTensor x224;
			JCudaTensor x225;
			x225 = x221;
			x224 = x42.forward(x225);

			// val X150 = (X149)(i | @) * (fc2_W)(j | @)
			JCudaTensor x226;
			JCudaMatrix x227;
			JCudaMatrix x228;
			JCudaTensor x229;
			x229 = x224;
			x227 = x229.asMatrix(1, true);
			JCudaTensor x230;
			x230 = x48;
			x228 = x230.asMatrix(1, true);
			x226 = x227.times(x228);

			// Dealloc(X149)
			JCudaTensor x231;
			x231 = x224;
			x231.free();

			// val X152 = (X150 + (i) => fc2_B)
			JCudaTensor x232;
			JCudaTensor x233, x234;
			x233 = x226;
			x234 = x52;
			x232 = x234.copy(500, x233);

			// Precision(Accuracy(1))
			float x236;
			JCudaTensor x237;
			JTensorFloat x238;
			x237 = x232;
			x238 = x4;
			x236 = x237.accuracy(x238, 1);
			System.out.println(x5 + " test precision "  + x236);
			x235 += x236;

			// Dealloc(X152)
			JCudaTensor x239;
			x239 = x232;
			x239.free();

		}
		System.out.println();
		System.out.println("average precision: " + x235/10);
		System.out.println(); 
	}

}