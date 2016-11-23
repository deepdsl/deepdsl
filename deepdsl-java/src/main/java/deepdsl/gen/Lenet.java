package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Lenet {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// decay
	static float decay = 5.0E-4f;
	// lrn_rate
	static float lrn_rate = -0.01f;
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
	// Precision(Accuracy(X152, Y, 1))
	static float x315;
	// V_cv1_B
	static JCudaTensor x239 = JTensor.constFloat(0.0f, 20).asJCudaTensor();
	// V_cv1_W
	static JCudaTensor x245 = JTensor.constFloat(0.0f, 20, 1, 5, 5).asJCudaTensor();
	// V_cv2_B
	static JCudaTensor x203 = JTensor.constFloat(0.0f, 50).asJCudaTensor();
	// V_cv2_W
	static JCudaTensor x196 = JTensor.constFloat(0.0f, 50, 20, 5, 5).asJCudaTensor();
	// V_fc1_B
	static JCudaTensor x156 = JTensor.constFloat(0.0f, 500).asJCudaTensor();
	// V_fc1_W
	static JCudaTensor x162 = JTensor.constFloat(0.0f, 500, 800).asJCudaTensor();
	// V_fc2_B
	static JCudaTensor x105 = JTensor.constFloat(0.0f, 10).asJCudaTensor();
	// V_fc2_W
	static JCudaTensor x98 = JTensor.constFloat(0.0f, 10, 500).asJCudaTensor();
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
		x39.free();
		x239.free();
		x98.free();
		x35.free();
		x196.free();
		x203.free();
		x23.free();
		x52.free();
		x48.free();
		x13.free();
		x156.free();
		x105.free();
		x245.free();
		x14.free();
		x162.free();
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

			// V_fc2_W <~~ m6 * m7
			float x99, x100;
			float x101;
			float x102;
			x101 = 1;
			x102 = lrn_rate;
			x99 = x101 * x102;
			x100 = momentum;
			JCudaMatrix x103;
			JCudaMatrix x104;
			x103 = x90;
			x104 = x92;
			x103.times(x104, x98, x99, x100);

			// V_fc2_B <~~ Sum(m6)
			float x106, x107;
			float x108;
			float x109;
			x108 = 1;
			x109 = lrn_rate;
			x106 = x108 * x109;
			x107 = momentum;
			JCudaMatrix x110;
			x110 = x90;
			x110.sum(x105, x106, x107);

			// Dealloc(X77)
			JCudaTensor x111;
			x111 = x83;
			x111.free();

			// val X89 = X87 * d_ReLU()(X134)/d_X133
			JCudaTensor x112;
			JCudaTensor x113, x114;
			x113 = x94;
			x114 = x40;
			x112 = x42.backward(x113, x114);

			// Dealloc(X134)
			JCudaTensor x115;
			x115 = x40;
			x115.free();

			// val m2 = (i93) => fc1_W[@, i93]
			JCudaMatrix x116;
			JCudaTensor x117;
			x117 = x35;
			x116 = x117.asMatrix(1, false);

			// fc2_W <~~ V_fc2_W
			float x118, x119;
			x118 = 1;
			float x120;
			float x121;
			x120 = 1;
			float x122;
			float x123;
			float x124;
			float x125;
			x124 = 1;
			x125 = decay;
			x122 = x124 * x125;
			float x126;
			float x127;
			x126 = 1;
			x127 = lrn_rate;
			x123 = x126 * x127;
			x121 = x122 * x123;
			x119 = x120 + x121;
			JCudaTensor x128;
			x128 = x98;
			x48.update(x128, x118, x119);

			// fc2_B <~~ V_fc2_B
			float x129, x130;
			x129 = 1;
			float x131;
			float x132;
			x131 = 1;
			float x133;
			float x134;
			float x135;
			float x136;
			x135 = 1;
			x136 = decay;
			x133 = x135 * x136;
			float x137;
			float x138;
			x137 = 1;
			x138 = lrn_rate;
			x134 = x137 * x138;
			x132 = x133 * x134;
			x130 = x131 + x132;
			JCudaTensor x139;
			x139 = x105;
			x52.update(x139, x129, x130);

			// val m8 = (i27) => X89[@, i27]
			JCudaMatrix x140;
			JCudaTensor x141;
			x141 = x112;
			x140 = x141.asMatrix(1, false);

			// val X90 = (X89)(i92 | @) * m2
			JCudaTensor x142;
			JCudaMatrix x143;
			JCudaMatrix x144;
			JCudaTensor x145;
			x145 = x112;
			x143 = x145.asMatrix(1, true);
			x144 = x116;
			x142 = x143.times(x144);

			// val m9 = (i28) => X130[1><3][@, i28]
			JCudaMatrix x146;
			JCudaTensor x147;
			JCudaTensor x148;
			x148 = x26;
			x147 = x148.flatten(1, new int[]{50, 4, 4});
			x146 = x147.asMatrix(1, false);

			// val X92 = X90[1<>3] * d_Pooling(2,2,0,true)(X130,X129)/d_X129
			JCudaTensor x149;
			JCudaTensor x150, x151, x152;
			JCudaTensor x153;
			x153 = x142;
			x150 = x153.unflatten(1, new int[]{50, 4, 4});
			x151 = x26;
			x152 = x19;
			x149 = x28.backward(x150, x151, x152);

			// Dealloc(X90)
			JCudaTensor x154;
			x154 = x142;
			x154.free();

			// Dealloc(X129)
			JCudaTensor x155;
			x155 = x19;
			x155.free();

			// V_fc1_B <~~ Sum(m8)
			float x157, x158;
			float x159;
			float x160;
			x159 = 1;
			x160 = lrn_rate;
			x157 = x159 * x160;
			x158 = momentum;
			JCudaMatrix x161;
			x161 = x140;
			x161.sum(x156, x157, x158);

			// V_fc1_W <~~ m8 * m9
			float x163, x164;
			float x165;
			float x166;
			x165 = 1;
			x166 = lrn_rate;
			x163 = x165 * x166;
			x164 = momentum;
			JCudaMatrix x167;
			JCudaMatrix x168;
			x167 = x140;
			x168 = x146;
			x167.times(x168, x162, x163, x164);

			// Dealloc(X89)
			JCudaTensor x169;
			x169 = x112;
			x169.free();

			// Dealloc(X130)
			JCudaTensor x170;
			x170 = x26;
			x170.free();

			// fc1_B <~~ V_fc1_B
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
			x181 = x156;
			x39.update(x181, x171, x172);

			// fc1_W <~~ V_fc1_W
			float x182, x183;
			x182 = 1;
			float x184;
			float x185;
			x184 = 1;
			float x186;
			float x187;
			float x188;
			float x189;
			x188 = 1;
			x189 = decay;
			x186 = x188 * x189;
			float x190;
			float x191;
			x190 = 1;
			x191 = lrn_rate;
			x187 = x190 * x191;
			x185 = x186 * x187;
			x183 = x184 + x185;
			JCudaTensor x192;
			x192 = x162;
			x35.update(x192, x182, x183);

			// val X93 = X92 * d_Convolv(1,0)(cv2_W)/d_X128
			JCudaTensor x193;
			JCudaTensor x194, x195;
			x194 = x149;
			x195 = x23;
			x193 = x25.backward_data(x194, x195);

			// V_cv2_W <~~ X92 * d_Convolv(1,0)(X128)/d_cv2_W
			float x197, x198;
			float x199;
			float x200;
			x199 = 1;
			x200 = lrn_rate;
			x197 = x199 * x200;
			x198 = momentum;
			JCudaTensor x201, x202;
			x201 = x149;
			x202 = x16;
			x25.backward_filter(x201, x202, x196, x197, x198);

			// V_cv2_B <~~ X92 * d_Convolv(1,0)()/d_cv2_B
			float x204, x205;
			float x206;
			float x207;
			x206 = 1;
			x207 = lrn_rate;
			x204 = x206 * x207;
			x205 = momentum;
			JCudaTensor x208;
			x208 = x149;
			x25.backward_bias(x208, x203, x204, x205);

			// Dealloc(X92)
			JCudaTensor x209;
			x209 = x149;
			x209.free();

			// cv2_W <~~ V_cv2_W
			float x210, x211;
			x210 = 1;
			float x212;
			float x213;
			x212 = 1;
			float x214;
			float x215;
			float x216;
			float x217;
			x216 = 1;
			x217 = decay;
			x214 = x216 * x217;
			float x218;
			float x219;
			x218 = 1;
			x219 = lrn_rate;
			x215 = x218 * x219;
			x213 = x214 * x215;
			x211 = x212 + x213;
			JCudaTensor x220;
			x220 = x196;
			x23.update(x220, x210, x211);

			// cv2_B <~~ V_cv2_B
			float x221, x222;
			x221 = 1;
			float x223;
			float x224;
			x223 = 1;
			float x225;
			float x226;
			float x227;
			float x228;
			x227 = 1;
			x228 = decay;
			x225 = x227 * x228;
			float x229;
			float x230;
			x229 = 1;
			x230 = lrn_rate;
			x226 = x229 * x230;
			x224 = x225 * x226;
			x222 = x223 + x224;
			JCudaTensor x231;
			x231 = x203;
			x24.update(x231, x221, x222);

			// val X95 = X93 * d_Pooling(2,2,0,true)(X128,X127)/d_X127
			JCudaTensor x232;
			JCudaTensor x233, x234, x235;
			x233 = x193;
			x234 = x16;
			x235 = x9;
			x232 = x18.backward(x233, x234, x235);

			// Dealloc(X93)
			JCudaTensor x236;
			x236 = x193;
			x236.free();

			// Dealloc(X128)
			JCudaTensor x237;
			x237 = x16;
			x237.free();

			// Dealloc(X127)
			JCudaTensor x238;
			x238 = x9;
			x238.free();

			// V_cv1_B <~~ X95 * d_Convolv(1,0)()/d_cv1_B
			float x240, x241;
			float x242;
			float x243;
			x242 = 1;
			x243 = lrn_rate;
			x240 = x242 * x243;
			x241 = momentum;
			JCudaTensor x244;
			x244 = x232;
			x15.backward_bias(x244, x239, x240, x241);

			// V_cv1_W <~~ X95 * d_Convolv(1,0)(X126)/d_cv1_W
			float x246, x247;
			float x248;
			float x249;
			x248 = 1;
			x249 = lrn_rate;
			x246 = x248 * x249;
			x247 = momentum;
			JCudaTensor x250, x251;
			x250 = x232;
			x251 = x7;
			x15.backward_filter(x250, x251, x245, x246, x247);

			// Dealloc(X95)
			JCudaTensor x252;
			x252 = x232;
			x252.free();

			// Dealloc(X126)
			JCudaTensor x253;
			x253 = x7;
			x253.free();

			// cv1_B <~~ V_cv1_B
			float x254, x255;
			x254 = 1;
			float x256;
			float x257;
			x256 = 1;
			float x258;
			float x259;
			float x260;
			float x261;
			x260 = 1;
			x261 = decay;
			x258 = x260 * x261;
			float x262;
			float x263;
			x262 = 1;
			x263 = lrn_rate;
			x259 = x262 * x263;
			x257 = x258 * x259;
			x255 = x256 + x257;
			JCudaTensor x264;
			x264 = x239;
			x14.update(x264, x254, x255);

			// cv1_W <~~ V_cv1_W
			float x265, x266;
			x265 = 1;
			float x267;
			float x268;
			x267 = 1;
			float x269;
			float x270;
			float x271;
			float x272;
			x271 = 1;
			x272 = decay;
			x269 = x271 * x272;
			float x273;
			float x274;
			x273 = 1;
			x274 = lrn_rate;
			x270 = x273 * x274;
			x268 = x269 * x270;
			x266 = x267 + x268;
			JCudaTensor x275;
			x275 = x245;
			x13.update(x275, x265, x266);

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X141 = Cuda(X)
			JCudaTensor x276;
			JTensorFloat x277;
			x277 = x3;
			x276 = x277.asJCudaTensor();

			// val X142 = Convolv(1,0)(X141,cv1_W,cv1_B)
			JCudaTensor x278;
			JCudaTensor x279, x280, x281;
			x279 = x276;
			x280 = x13;
			x281 = x14;
			x278 = x15.forward(x279, x280, x281);

			// Dealloc(X141)
			JCudaTensor x282;
			x282 = x276;
			x282.free();

			// val X143 = Pooling(2,2,0,true)(X142)
			JCudaTensor x283;
			JCudaTensor x284;
			x284 = x278;
			x283 = x18.forward(x284);

			// Dealloc(X142)
			JCudaTensor x285;
			x285 = x278;
			x285.free();

			// val X144 = Convolv(1,0)(X143,cv2_W,cv2_B)
			JCudaTensor x286;
			JCudaTensor x287, x288, x289;
			x287 = x283;
			x288 = x23;
			x289 = x24;
			x286 = x25.forward(x287, x288, x289);

			// Dealloc(X143)
			JCudaTensor x290;
			x290 = x283;
			x290.free();

			// val X145 = Pooling(2,2,0,true)(X144)
			JCudaTensor x291;
			JCudaTensor x292;
			x292 = x286;
			x291 = x28.forward(x292);

			// Dealloc(X144)
			JCudaTensor x293;
			x293 = x286;
			x293.free();

			// val X146 = (X145[1><3])(i | @) * (fc1_W)(j | @)
			JCudaTensor x294;
			JCudaMatrix x295;
			JCudaMatrix x296;
			JCudaTensor x297;
			JCudaTensor x298;
			x298 = x291;
			x297 = x298.flatten(1, new int[]{50, 4, 4});
			x295 = x297.asMatrix(1, true);
			JCudaTensor x299;
			x299 = x35;
			x296 = x299.asMatrix(1, true);
			x294 = x295.times(x296);

			// Dealloc(X145)
			JCudaTensor x300;
			x300 = x291;
			x300.free();

			// val X148 = (X146 + (i) => fc1_B)
			JCudaTensor x301;
			JCudaTensor x302, x303;
			x302 = x294;
			x303 = x39;
			x301 = x303.copy(500, x302);

			// val X149 = ReLU()(X148)
			JCudaTensor x304;
			JCudaTensor x305;
			x305 = x301;
			x304 = x42.forward(x305);

			// val X150 = (X149)(i | @) * (fc2_W)(j | @)
			JCudaTensor x306;
			JCudaMatrix x307;
			JCudaMatrix x308;
			JCudaTensor x309;
			x309 = x304;
			x307 = x309.asMatrix(1, true);
			JCudaTensor x310;
			x310 = x48;
			x308 = x310.asMatrix(1, true);
			x306 = x307.times(x308);

			// Dealloc(X149)
			JCudaTensor x311;
			x311 = x304;
			x311.free();

			// val X152 = (X150 + (i) => fc2_B)
			JCudaTensor x312;
			JCudaTensor x313, x314;
			x313 = x306;
			x314 = x52;
			x312 = x314.copy(500, x313);

			// Precision(Accuracy(X152, Y, 1))
			float x316;
			JCudaTensor x317;
			JTensorFloat x318;
			x317 = x312;
			x318 = x4;
			x316 = x317.accuracy(x318, 1);
			System.out.println(x5 + " test precision "  + x316);
			x315 += x316;

			// Dealloc(X152)
			JCudaTensor x319;
			x319 = x312;
			x319.free();

		}
		System.out.println();
		System.out.println("average precision: " + x315/10);
		System.out.println(); 
	}

}