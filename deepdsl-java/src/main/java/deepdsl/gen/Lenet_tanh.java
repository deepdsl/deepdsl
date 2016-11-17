package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Lenet_tanh {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// lrn_rate_1
	static float lrn_rate_1 = -0.1f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/lenet_tanh";
	// test_itr
	static int test_itr = 10;
	// train_itr
	static int train_itr = 1000;

	// (Convolv(1,0),List(List(500, 1, 28, 28), List(20, 1, 5, 5), List(20)))
	static JCudnnConvolution x15 = new JCudnnConvolution(new int[]{500,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
	// (Convolv(1,0),List(List(500, 20, 12, 12), List(50, 20, 5, 5), List(50)))
	static JCudnnConvolution x28 = new JCudnnConvolution(new int[]{500,20,12,12},new int[]{50,20,5,5},new int[]{50}, 1, 0);
	// (MNIST,false)
	static MnistFactory x2 = MnistFactory.getFactory(false, new int[]{500, 1, 28, 28});
	// (MNIST,true)
	static MnistFactory x1 = MnistFactory.getFactory(true, new int[]{500, 1, 28, 28});
	// (Pooling(2,2,0,true),List(List(500, 20, 24, 24)))
	static JCudnnPooling x18 = new JCudnnPooling(new int[]{500,20,24,24}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(500, 50, 8, 8)))
	static JCudnnPooling x31 = new JCudnnPooling(new int[]{500,50,8,8}, 2, 2, 0, PoolingType.MAX);
	// (Softmax(),List(List(500, 10)))
	static JCudnnSoftmax x61 = new JCudnnSoftmax(new int[]{500,10}, SoftmaxAlgorithm.ACCURATE);
	// (Tanh(),List(List(500, 20, 12, 12)))
	static JCudnnActivation x21 = new JCudnnActivation(new int[]{500,20,12,12}, ActivationMode.TANH);
	// (Tanh(),List(List(500, 50, 4, 4)))
	static JCudnnActivation x34 = new JCudnnActivation(new int[]{500,50,4,4}, ActivationMode.TANH);
	// (Tanh(),List(List(500, 500)))
	static JCudnnActivation x48 = new JCudnnActivation(new int[]{500,500}, ActivationMode.TANH);
	// B
	static JCudaTensor x45 = JTensor.constFloat(0.0f, 500).load(network_dir + "/B").asJCudaTensor();
	// B1
	static JCudaTensor x14 = JTensor.constFloat(0.0f, 20).load(network_dir + "/B1").asJCudaTensor();
	// B2
	static JCudaTensor x27 = JTensor.constFloat(0.0f, 50).load(network_dir + "/B2").asJCudaTensor();
	// B3
	static JCudaTensor x58 = JTensor.constFloat(0.0f, 10).load(network_dir + "/B3").asJCudaTensor();
	// Precision((Sum(X605) / |500|))
	static float x238;
	// Theta
	static JCudaTensor x54 = JTensor.constFloat(0.0f, 10, 500).load(network_dir + "/Theta").asJCudaTensor();
	// W
	static JCudaTensor x41 = JTensor.randomFloat(-0.06793662f, 0.06793662f, 500, 800).load(network_dir + "/W").asJCudaTensor();
	// W1
	static JCudaTensor x13 = JTensor.randomFloat(-0.2f, 0.2f, 20, 1, 5, 5).load(network_dir + "/W1").asJCudaTensor();
	// W2
	static JCudaTensor x26 = JTensor.randomFloat(-0.08596024f, 0.08596024f, 50, 20, 5, 5).load(network_dir + "/W2").asJCudaTensor();
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;

	public static void main(String[] args){
		double t = System.nanoTime();
		train();
		System.out.println((System.nanoTime() - t) / 1.0E9);
		test();
		x54.save(network_dir + "/Theta");
		x13.save(network_dir + "/W1");
		x14.save(network_dir + "/B1");
		x27.save(network_dir + "/B2");
		x26.save(network_dir + "/W2");
		x41.save(network_dir + "/W");
		x45.save(network_dir + "/B");
		x58.save(network_dir + "/B3");
		x58.free();
		x27.free();
		x54.free();
		x26.free();
		x45.free();
		x41.free();
		x13.free();
		x14.free();
		x61.free();
		x15.free();
		x34.free();
		x18.free();
		x48.free();
		x28.free();
		x21.free();
		x31.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void train() {
		for(int x5=0; x5<train_itr; x5++) {
			JTensorFloatTuple x6 =  x1.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X8 = Cuda(X)
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x3;
			x7 = x8.asJCudaTensor();

			// val X9 = Convolv(1,0)(X8,W1,B1)
			JCudaTensor x9;
			JCudaTensor x10, x11, x12;
			x10 = x7;
			x11 = x13;
			x12 = x14;
			x9 = x15.forward(x10, x11, x12);

			// val X10 = Pooling(2,2,0,true)(X9)
			JCudaTensor x16;
			JCudaTensor x17;
			x17 = x9;
			x16 = x18.forward(x17);

			// val X11 = Tanh()(X10.copy)
			JCudaTensor x19;
			JCudaTensor x20;
			x20 = x16;
			x20 = x20.clone();
			x19 = x21.forward(x20);

			// val X12 = Convolv(1,0)(X11,W2,B2)
			JCudaTensor x22;
			JCudaTensor x23, x24, x25;
			x23 = x19;
			x24 = x26;
			x25 = x27;
			x22 = x28.forward(x23, x24, x25);

			// val X13 = Pooling(2,2,0,true)(X12)
			JCudaTensor x29;
			JCudaTensor x30;
			x30 = x22;
			x29 = x31.forward(x30);

			// val X14 = Tanh()(X13.copy)
			JCudaTensor x32;
			JCudaTensor x33;
			x33 = x29;
			x33 = x33.clone();
			x32 = x34.forward(x33);

			// val X15 = (X14[1><3])(i | @) * (W)(j | @)
			JCudaTensor x35;
			JCudaMatrix x36;
			JCudaMatrix x37;
			JCudaTensor x38;
			JCudaTensor x39;
			x39 = x32;
			x38 = x39.flatten(1, new int[]{50, 4, 4});
			x36 = x38.asMatrix(1, true);
			JCudaTensor x40;
			x40 = x41;
			x37 = x40.asMatrix(1, true);
			x35 = x36.times(x37);

			// val X17 = (X15 + (i) => B)
			JCudaTensor x42;
			JCudaTensor x43, x44;
			x43 = x35;
			x44 = x45;
			x42 = x44.copy(500, x43);

			// val X18 = Tanh()(X17)
			JCudaTensor x46;
			JCudaTensor x47;
			x47 = x42;
			x46 = x48.forward(x47);

			// val X19 = (X18)(i | @) * (Theta)(j | @)
			JCudaTensor x49;
			JCudaMatrix x50;
			JCudaMatrix x51;
			JCudaTensor x52;
			x52 = x46;
			x50 = x52.asMatrix(1, true);
			JCudaTensor x53;
			x53 = x54;
			x51 = x53.asMatrix(1, true);
			x49 = x50.times(x51);

			// val X21 = (X19 + (i) => B3)
			JCudaTensor x55;
			JCudaTensor x56, x57;
			x56 = x49;
			x57 = x58;
			x55 = x57.copy(500, x56);

			// val X22 = Softmax()(X21)
			JCudaTensor x59;
			JCudaTensor x60;
			x60 = x55;
			x59 = x61.forward(x60);

			// Dealloc(X21)
			JCudaTensor x62;
			x62 = x55;
			x62.free();

			// val X23 = Cuda(Indicator(Y, 10))
			JCudaTensor x63;
			JTensorFloat x64;
			x64 = x4.asIndicator(10);
			x63 = x64.asJCudaTensor();

			// val X24 = Log X22.copy
			JCudaTensor x65;
			JCudaTensor x66;
			x66 = x59;
			x66 = x66.clone();
			x65 = x66.log();

			// val X54 = 1/(X22.copy)
			JCudaTensor x67;
			JCudaTensor x68;
			float x69;
			x68 = x59;
			x68 = x68.clone();
			x69 = -1;
			x67 = x68.pow(x69);

			// Cost(((0 - (X23 . X24)) / |500|))
			float x70;
			float x71;
			float x72;
			float x73;
			JCudaTensor x74, x75;
			x74 = x63;
			x75 = x65;
			x73 = x74.dot(x75);
			x71 = - x73;
			x72 = 500;
			x70 = x71 / x72;
			System.out.println(x5 + " " + x70);
			if (Float.isNaN(x70)) { System.exit(-1); }

			// Dealloc(X24)
			JCudaTensor x76;
			x76 = x65;
			x76.free();

			// val X55 = X23.copy .* X54
			JCudaTensor x77;
			JCudaTensor x78, x79;
			x78 = x63;
			x78 = x78.clone();
			x79 = x67;
			x77 = x78.times_i(x79);

			// Dealloc(X54)
			JCudaTensor x80;
			x80 = x67;
			x80.free();

			// Dealloc(X23)
			JCudaTensor x81;
			x81 = x63;
			x81.free();

			// val X56 = - X55
			JCudaTensor x82;
			JCudaTensor x83;
			float x84;
			x83 = x77;
			x84 = -1;
			x82 = x83.times_i(x84);

			// val X57 = (X56 / |500|)
			JCudaTensor x85;
			JCudaTensor x86;
			float x87;
			x86 = x82;
			float x88;
			x88 = 500;
			x87 = 1 / x88;
			x85 = x86.times_i(x87);

			// val X73 = X57 * d_Softmax()(X22)/d_X21
			JCudaTensor x89;
			JCudaTensor x90, x91;
			x90 = x85;
			x91 = x59;
			x89 = x61.backward(x90, x91);

			// Dealloc(X57)
			JCudaTensor x92;
			x92 = x85;
			x92.free();

			// Dealloc(X22)
			JCudaTensor x93;
			x93 = x59;
			x93.free();

			// val m3 = (i30) => Theta[@, i30]
			JCudaMatrix x94;
			JCudaTensor x95;
			x95 = x54;
			x94 = x95.asMatrix(1, false);

			// val X153 = (X73)(i29 | @) * m3
			JCudaTensor x96;
			JCudaMatrix x97;
			JCudaMatrix x98;
			JCudaTensor x99;
			x99 = x89;
			x97 = x99.asMatrix(1, true);
			x98 = x94;
			x96 = x97.times(x98);

			// val m1 = (i19) => X73[@, i19]
			JCudaMatrix x100;
			JCudaTensor x101;
			x101 = x89;
			x100 = x101.asMatrix(1, false);

			// val m2 = (i20) => X18[@, i20]
			JCudaMatrix x102;
			JCudaTensor x103;
			x103 = x46;
			x102 = x103.asMatrix(1, false);

			// Theta <~~ m1 * m2
			float x104, x105;
			x104 = lrn_rate_1;
			x105 = 1;
			JCudaMatrix x106;
			JCudaMatrix x107;
			x106 = x100;
			x107 = x102;
			x106.times(x107, x54, x104, x105);

			// B3 <~~ Sum(m1)
			float x108, x109;
			x108 = lrn_rate_1;
			x109 = 1;
			JCudaMatrix x110;
			x110 = x100;
			x110.sum(x58, x108, x109);

			// Dealloc(X73)
			JCudaTensor x111;
			x111 = x89;
			x111.free();

			// val X155 = X153 * d_Tanh()(X18)/d_X17
			JCudaTensor x112;
			JCudaTensor x113, x114;
			x113 = x96;
			x114 = x46;
			x112 = x48.backward(x113, x114);

			// Dealloc(X18)
			JCudaTensor x115;
			x115 = x46;
			x115.free();

			// val m4 = (i34) => W[@, i34]
			JCudaMatrix x116;
			JCudaTensor x117;
			x117 = x41;
			x116 = x117.asMatrix(1, false);

			// val m12 = (i137) => X155[@, i137]
			JCudaMatrix x118;
			JCudaTensor x119;
			x119 = x112;
			x118 = x119.asMatrix(1, false);

			// val m13 = (i138) => X14[1><3][@, i138]
			JCudaMatrix x120;
			JCudaTensor x121;
			JCudaTensor x122;
			x122 = x32;
			x121 = x122.flatten(1, new int[]{50, 4, 4});
			x120 = x121.asMatrix(1, false);

			// val X156 = (X155)(i33 | @) * m4
			JCudaTensor x123;
			JCudaMatrix x124;
			JCudaMatrix x125;
			JCudaTensor x126;
			x126 = x112;
			x124 = x126.asMatrix(1, true);
			x125 = x116;
			x123 = x124.times(x125);

			// W <~~ m12 * m13
			float x127, x128;
			x127 = lrn_rate_1;
			x128 = 1;
			JCudaMatrix x129;
			JCudaMatrix x130;
			x129 = x118;
			x130 = x120;
			x129.times(x130, x41, x127, x128);

			// B <~~ Sum(m12)
			float x131, x132;
			x131 = lrn_rate_1;
			x132 = 1;
			JCudaMatrix x133;
			x133 = x118;
			x133.sum(x45, x131, x132);

			// Dealloc(X155)
			JCudaTensor x134;
			x134 = x112;
			x134.free();

			// val X158 = X156[1<>3] * d_Tanh()(X14)/d_X13
			JCudaTensor x135;
			JCudaTensor x136, x137;
			JCudaTensor x138;
			x138 = x123;
			x136 = x138.unflatten(1, new int[]{50, 4, 4});
			x137 = x32;
			x135 = x34.backward(x136, x137);

			// Dealloc(X14)
			JCudaTensor x139;
			x139 = x32;
			x139.free();

			// val X160 = X158 * d_Pooling(2,2,0,true)(X13,X12)/d_X12
			JCudaTensor x140;
			JCudaTensor x141, x142, x143;
			x141 = x135;
			x142 = x29;
			x143 = x22;
			x140 = x31.backward(x141, x142, x143);

			// Dealloc(X158)
			JCudaTensor x144;
			x144 = x135;
			x144.free();

			// Dealloc(X13)
			JCudaTensor x145;
			x145 = x29;
			x145.free();

			// Dealloc(X12)
			JCudaTensor x146;
			x146 = x22;
			x146.free();

			// B2 <~~ X160 * d_Convolv(1,0)()/d_B2
			float x147, x148;
			x147 = lrn_rate_1;
			x148 = 1;
			JCudaTensor x149;
			x149 = x140;
			x28.backward_bias(x149, x27, x147, x148);

			// val X161 = X160 * d_Convolv(1,0)(W2)/d_X11
			JCudaTensor x150;
			JCudaTensor x151, x152;
			x151 = x140;
			x152 = x26;
			x150 = x28.backward_data(x151, x152);

			// W2 <~~ X160 * d_Convolv(1,0)(X11)/d_W2
			float x153, x154;
			x153 = lrn_rate_1;
			x154 = 1;
			JCudaTensor x155, x156;
			x155 = x140;
			x156 = x19;
			x28.backward_filter(x155, x156, x26, x153, x154);

			// Dealloc(X160)
			JCudaTensor x157;
			x157 = x140;
			x157.free();

			// val X163 = X161 * d_Tanh()(X11)/d_X10
			JCudaTensor x158;
			JCudaTensor x159, x160;
			x159 = x150;
			x160 = x19;
			x158 = x21.backward(x159, x160);

			// Dealloc(X11)
			JCudaTensor x161;
			x161 = x19;
			x161.free();

			// val X165 = X163 * d_Pooling(2,2,0,true)(X10,X9)/d_X9
			JCudaTensor x162;
			JCudaTensor x163, x164, x165;
			x163 = x158;
			x164 = x16;
			x165 = x9;
			x162 = x18.backward(x163, x164, x165);

			// Dealloc(X163)
			JCudaTensor x166;
			x166 = x158;
			x166.free();

			// Dealloc(X10)
			JCudaTensor x167;
			x167 = x16;
			x167.free();

			// Dealloc(X9)
			JCudaTensor x168;
			x168 = x9;
			x168.free();

			// W1 <~~ X165 * d_Convolv(1,0)(X8)/d_W1
			float x169, x170;
			x169 = lrn_rate_1;
			x170 = 1;
			JCudaTensor x171, x172;
			x171 = x162;
			x172 = x7;
			x15.backward_filter(x171, x172, x13, x169, x170);

			// Dealloc(X8)
			JCudaTensor x173;
			x173 = x7;
			x173.free();

			// B1 <~~ X165 * d_Convolv(1,0)()/d_B1
			float x174, x175;
			x174 = lrn_rate_1;
			x175 = 1;
			JCudaTensor x176;
			x176 = x162;
			x15.backward_bias(x176, x14, x174, x175);

			// Dealloc(X165)
			JCudaTensor x177;
			x177 = x162;
			x177.free();

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X587 = Cuda(X)
			JCudaTensor x178;
			JTensorFloat x179;
			x179 = x3;
			x178 = x179.asJCudaTensor();

			// val X588 = Convolv(1,0)(X587,W1,B1)
			JCudaTensor x180;
			JCudaTensor x181, x182, x183;
			x181 = x178;
			x182 = x13;
			x183 = x14;
			x180 = x15.forward(x181, x182, x183);

			// Dealloc(X587)
			JCudaTensor x184;
			x184 = x178;
			x184.free();

			// val X589 = Pooling(2,2,0,true)(X588)
			JCudaTensor x185;
			JCudaTensor x186;
			x186 = x180;
			x185 = x18.forward(x186);

			// Dealloc(X588)
			JCudaTensor x187;
			x187 = x180;
			x187.free();

			// val X590 = Tanh()(X589)
			JCudaTensor x188;
			JCudaTensor x189;
			x189 = x185;
			x188 = x21.forward(x189);

			// val X591 = Convolv(1,0)(X590,W2,B2)
			JCudaTensor x190;
			JCudaTensor x191, x192, x193;
			x191 = x188;
			x192 = x26;
			x193 = x27;
			x190 = x28.forward(x191, x192, x193);

			// Dealloc(X590)
			JCudaTensor x194;
			x194 = x188;
			x194.free();

			// val X592 = Pooling(2,2,0,true)(X591)
			JCudaTensor x195;
			JCudaTensor x196;
			x196 = x190;
			x195 = x31.forward(x196);

			// Dealloc(X591)
			JCudaTensor x197;
			x197 = x190;
			x197.free();

			// val X593 = Tanh()(X592)
			JCudaTensor x198;
			JCudaTensor x199;
			x199 = x195;
			x198 = x34.forward(x199);

			// val X594 = (X593[1><3])(i | @) * (W)(j | @)
			JCudaTensor x200;
			JCudaMatrix x201;
			JCudaMatrix x202;
			JCudaTensor x203;
			JCudaTensor x204;
			x204 = x198;
			x203 = x204.flatten(1, new int[]{50, 4, 4});
			x201 = x203.asMatrix(1, true);
			JCudaTensor x205;
			x205 = x41;
			x202 = x205.asMatrix(1, true);
			x200 = x201.times(x202);

			// Dealloc(X593)
			JCudaTensor x206;
			x206 = x198;
			x206.free();

			// val X596 = (X594 + (i) => B)
			JCudaTensor x207;
			JCudaTensor x208, x209;
			x208 = x200;
			x209 = x45;
			x207 = x209.copy(500, x208);

			// val X597 = Tanh()(X596)
			JCudaTensor x210;
			JCudaTensor x211;
			x211 = x207;
			x210 = x48.forward(x211);

			// val X598 = (X597)(i | @) * (Theta)(j | @)
			JCudaTensor x212;
			JCudaMatrix x213;
			JCudaMatrix x214;
			JCudaTensor x215;
			x215 = x210;
			x213 = x215.asMatrix(1, true);
			JCudaTensor x216;
			x216 = x54;
			x214 = x216.asMatrix(1, true);
			x212 = x213.times(x214);

			// Dealloc(X597)
			JCudaTensor x217;
			x217 = x210;
			x217.free();

			// val X600 = (X598 + (i) => B3)
			JCudaTensor x218;
			JCudaTensor x219, x220;
			x219 = x212;
			x220 = x58;
			x218 = x220.copy(500, x219);

			// val X601 = Cuda(Indicator(Y, 10))
			JCudaTensor x221;
			JTensorFloat x222;
			x222 = x4.asIndicator(10);
			x221 = x222.asJCudaTensor();

			// val X602 = X601 .* X600
			JCudaTensor x223;
			JCudaTensor x224, x225;
			x224 = x221;
			x225 = x218;
			x223 = x224.times_i(x225);

			// val X603 = Sum((X602)(i14 | @))
			JCudaTensor x226;
			JCudaMatrix x227;
			JCudaTensor x228;
			x228 = x223;
			x227 = x228.asMatrix(1, true);
			x226 = x227.sum();

			// Dealloc(X602)
			JCudaTensor x229;
			x229 = x223;
			x229.free();

			// val X604 = Max((X600)(i14 | @))
			JCudaTensor x230;
			JCudaMatrix x231;
			JCudaTensor x232;
			x232 = x218;
			x231 = x232.asMatrix(1, true);
			x230 = x231.max();

			// Dealloc(X600)
			JCudaTensor x233;
			x233 = x218;
			x233.free();

			// val X605 = 1{X603 == X604}
			JCudaTensor x234;
			JCudaTensor x235, x236;
			x235 = x226;
			x236 = x230;
			x234 = x235.eq(x236);

			// Dealloc(X604)
			JCudaTensor x237;
			x237 = x230;
			x237.free();

			// Precision((Sum(X605) / |500|))
			float x239;
			float x240;
			float x241;
			JCudaTensor x242;
			x242 = x234;
			x240 = x242.sum();
			x241 = 500;
			x239 = x240 / x241;
			System.out.println(x5 + " test precision "  + x239);
			x238 += x239;

			// Dealloc(X605)
			JCudaTensor x243;
			x243 = x234;
			x243.free();

		}
		System.out.println();
		System.out.println("average precision: " + x238/10);
		System.out.println(); 
	}

}