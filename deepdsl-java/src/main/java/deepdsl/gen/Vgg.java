package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.data.imagenet.LmdbUtils;


public class Vgg {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// decay_1
	static float decay_1 = 0.99995f;
	// lrn_rate_1
	static float lrn_rate_1 = -0.1f;
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/vgg";
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

	// (Convolv(1,1),List(List(64, 128, 112, 112), List(128, 128, 3, 3), List(128)))
	static JCudnnConvolution x52 = new JCudnnConvolution(new int[]{64,128,112,112},new int[]{128,128,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(64, 128, 56, 56), List(256, 128, 3, 3), List(256)))
	static JCudnnConvolution x64 = new JCudnnConvolution(new int[]{64,128,56,56},new int[]{256,128,3,3},new int[]{256}, 1, 1);
	// (Convolv(1,1),List(List(64, 256, 28, 28), List(512, 256, 3, 3), List(512)))
	static JCudnnConvolution x94 = new JCudnnConvolution(new int[]{64,256,28,28},new int[]{512,256,3,3},new int[]{512}, 1, 1);
	// (Convolv(1,1),List(List(64, 256, 56, 56), List(256, 256, 3, 3), List(256)))
	static JCudnnConvolution x74 = new JCudnnConvolution(new int[]{64,256,56,56},new int[]{256,256,3,3},new int[]{256}, 1, 1);
	// (Convolv(1,1),List(List(64, 3, 224, 224), List(64, 3, 3, 3), List(64)))
	static JCudnnConvolution x20 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,3,3},new int[]{64}, 1, 1);
	// (Convolv(1,1),List(List(64, 512, 14, 14), List(512, 512, 3, 3), List(512)))
	static JCudnnConvolution x124 = new JCudnnConvolution(new int[]{64,512,14,14},new int[]{512,512,3,3},new int[]{512}, 1, 1);
	// (Convolv(1,1),List(List(64, 512, 28, 28), List(512, 512, 3, 3), List(512)))
	static JCudnnConvolution x104 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{512,512,3,3},new int[]{512}, 1, 1);
	// (Convolv(1,1),List(List(64, 64, 112, 112), List(128, 64, 3, 3), List(128)))
	static JCudnnConvolution x42 = new JCudnnConvolution(new int[]{64,64,112,112},new int[]{128,64,3,3},new int[]{128}, 1, 1);
	// (Convolv(1,1),List(List(64, 64, 224, 224), List(64, 64, 3, 3), List(64)))
	static JCudnnConvolution x30 = new JCudnnConvolution(new int[]{64,64,224,224},new int[]{64,64,3,3},new int[]{64}, 1, 1);
	// (Dropout(0.5),List(List(64, 4096)))
	static JCudnnDropout x163 = new JCudnnDropout(new int[]{64,4096}, 0.5f);
	// (Lmdb(1000000,10000,Win32,1000),false)
	static LmdbFactory x2 = LmdbFactory.getFactory(test_data_path, test_size, new int[]{64, 3, 224, 224}, platform, 1000, true);
	// (Lmdb(1000000,10000,Win32,1000),true)
	static LmdbFactory x1 = LmdbFactory.getFactory(train_data_path, train_size, new int[]{64, 3, 224, 224}, platform, 1000, false);
	// (LogSoftmax(),List(List(64, 1000)))
	static JCudnnSoftmax x190 = new JCudnnSoftmax(new int[]{64,1000}, SoftmaxAlgorithm.LOG);
	// (Pooling(2,2,0,true),List(List(64, 128, 112, 112)))
	static JCudnnPooling x57 = new JCudnnPooling(new int[]{64,128,112,112}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(64, 256, 56, 56)))
	static JCudnnPooling x87 = new JCudnnPooling(new int[]{64,256,56,56}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(64, 512, 14, 14)))
	static JCudnnPooling x146 = new JCudnnPooling(new int[]{64,512,14,14}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(64, 512, 28, 28)))
	static JCudnnPooling x117 = new JCudnnPooling(new int[]{64,512,28,28}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(64, 64, 224, 224)))
	static JCudnnPooling x35 = new JCudnnPooling(new int[]{64,64,224,224}, 2, 2, 0, PoolingType.MAX);
	// (ReLU(),List(List(64, 128, 112, 112)))
	static JCudnnActivation x45 = new JCudnnActivation(new int[]{64,128,112,112}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 256, 56, 56)))
	static JCudnnActivation x67 = new JCudnnActivation(new int[]{64,256,56,56}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 4096)))
	static JCudnnActivation x160 = new JCudnnActivation(new int[]{64,4096}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 512, 14, 14)))
	static JCudnnActivation x127 = new JCudnnActivation(new int[]{64,512,14,14}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 512, 28, 28)))
	static JCudnnActivation x97 = new JCudnnActivation(new int[]{64,512,28,28}, ActivationMode.RELU);
	// (ReLU(),List(List(64, 64, 224, 224)))
	static JCudnnActivation x23 = new JCudnnActivation(new int[]{64,64,224,224}, ActivationMode.RELU);
	// Precision(Accuracy(1))
	static float x649;
	// X
	static JTensorFloat x3;
	// Y
	static JTensorFloat x4;
	// cv11_B
	static JCudaTensor x19 = JTensor.constFloat(0.0f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
	// cv11_W
	static JCudaTensor x18 = JTensor.randomFloat(-0.27216554f, 0.27216554f, 64, 3, 3, 3).load(network_dir + "/cv11_W").asJCudaTensor();
	// cv12_B
	static JCudaTensor x29 = JTensor.constFloat(0.0f, 64).load(network_dir + "/cv12_B").asJCudaTensor();
	// cv12_W
	static JCudaTensor x28 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/cv12_W").asJCudaTensor();
	// cv21_B
	static JCudaTensor x41 = JTensor.constFloat(0.0f, 128).load(network_dir + "/cv21_B").asJCudaTensor();
	// cv21_W
	static JCudaTensor x40 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 128, 64, 3, 3).load(network_dir + "/cv21_W").asJCudaTensor();
	// cv22_B
	static JCudaTensor x51 = JTensor.constFloat(0.0f, 128).load(network_dir + "/cv22_B").asJCudaTensor();
	// cv22_W
	static JCudaTensor x50 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/cv22_W").asJCudaTensor();
	// cv31_B
	static JCudaTensor x63 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv31_B").asJCudaTensor();
	// cv31_W
	static JCudaTensor x62 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 256, 128, 3, 3).load(network_dir + "/cv31_W").asJCudaTensor();
	// cv32_B
	static JCudaTensor x73 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv32_B").asJCudaTensor();
	// cv32_W
	static JCudaTensor x72 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/cv32_W").asJCudaTensor();
	// cv33_B
	static JCudaTensor x82 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv33_B").asJCudaTensor();
	// cv33_W
	static JCudaTensor x81 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
	// cv41_B
	static JCudaTensor x93 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv41_B").asJCudaTensor();
	// cv41_W
	static JCudaTensor x92 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 512, 256, 3, 3).load(network_dir + "/cv41_W").asJCudaTensor();
	// cv42_B
	static JCudaTensor x103 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv42_B").asJCudaTensor();
	// cv42_W
	static JCudaTensor x102 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv42_W").asJCudaTensor();
	// cv43_B
	static JCudaTensor x112 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv43_B").asJCudaTensor();
	// cv43_W
	static JCudaTensor x111 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
	// cv51_B
	static JCudaTensor x123 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv51_B").asJCudaTensor();
	// cv51_W
	static JCudaTensor x122 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv51_W").asJCudaTensor();
	// cv52_B
	static JCudaTensor x133 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv52_B").asJCudaTensor();
	// cv52_W
	static JCudaTensor x132 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv52_W").asJCudaTensor();
	// cv53_B
	static JCudaTensor x141 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv53_B").asJCudaTensor();
	// cv53_W
	static JCudaTensor x140 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
	// fc6_B
	static JCudaTensor x157 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc6_B").asJCudaTensor();
	// fc6_W
	static JCudaTensor x153 = JTensor.randomFloat(-0.008928572f, 0.008928572f, 4096, 25088).load(network_dir + "/fc6_W").asJCudaTensor();
	// fc7_B
	static JCudaTensor x173 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
	// fc7_W
	static JCudaTensor x169 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 4096, 4096).load(network_dir + "/fc7_W").asJCudaTensor();
	// fc8_B
	static JCudaTensor x187 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
	// fc8_W
	static JCudaTensor x183 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

	public static void main(String[] args){
		double t = System.nanoTime();
		train();
		System.out.println((System.nanoTime() - t) / 1.0E9);
		test();
		x19.save(network_dir + "/cv11_B");
		x18.save(network_dir + "/cv11_W");
		x29.save(network_dir + "/cv12_B");
		x28.save(network_dir + "/cv12_W");
		x41.save(network_dir + "/cv21_B");
		x40.save(network_dir + "/cv21_W");
		x51.save(network_dir + "/cv22_B");
		x50.save(network_dir + "/cv22_W");
		x63.save(network_dir + "/cv31_B");
		x62.save(network_dir + "/cv31_W");
		x73.save(network_dir + "/cv32_B");
		x72.save(network_dir + "/cv32_W");
		x82.save(network_dir + "/cv33_B");
		x81.save(network_dir + "/cv33_W");
		x93.save(network_dir + "/cv41_B");
		x92.save(network_dir + "/cv41_W");
		x103.save(network_dir + "/cv42_B");
		x102.save(network_dir + "/cv42_W");
		x112.save(network_dir + "/cv43_B");
		x111.save(network_dir + "/cv43_W");
		x123.save(network_dir + "/cv51_B");
		x122.save(network_dir + "/cv51_W");
		x133.save(network_dir + "/cv52_B");
		x132.save(network_dir + "/cv52_W");
		x141.save(network_dir + "/cv53_B");
		x140.save(network_dir + "/cv53_W");
		x157.save(network_dir + "/fc6_B");
		x153.save(network_dir + "/fc6_W");
		x173.save(network_dir + "/fc7_B");
		x169.save(network_dir + "/fc7_W");
		x187.save(network_dir + "/fc8_B");
		x183.save(network_dir + "/fc8_W");
		x187.free();
		x140.free();
		x103.free();
		x51.free();
		x169.free();
		x18.free();
		x19.free();
		x29.free();
		x112.free();
		x63.free();
		x123.free();
		x40.free();
		x50.free();
		x183.free();
		x141.free();
		x173.free();
		x153.free();
		x92.free();
		x41.free();
		x82.free();
		x133.free();
		x93.free();
		x73.free();
		x28.free();
		x122.free();
		x62.free();
		x72.free();
		x157.free();
		x132.free();
		x111.free();
		x81.free();
		x102.free();
		x127.free();
		x94.free();
		x97.free();
		x64.free();
		x124.free();
		x67.free();
		x35.free();
		x163.free();
		x23.free();
		x74.free();
		x45.free();
		x104.free();
		x52.free();
		x20.free();
		x57.free();
		x30.free();
		x160.free();
		x146.free();
		x87.free();
		x42.free();
		x190.free();
		x117.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void train() {
		for(int x5=0; x5<train_itr; x5++) {
			JTensorFloatTuple x6 =  x1.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X1864 = Cuda(Indicator(Y, 1000))
			JCudaTensor x7;
			JTensorFloat x8;
			x8 = x4.asIndicator(1000);
			x7 = x8.asJCudaTensor();

			// val X1818 = Cuda(X)
			JCudaTensor x9;
			JTensorFloat x10;
			x10 = x3;
			x9 = x10.asJCudaTensor();

			// val X891 = - X1864.copy
			JCudaTensor x11;
			JCudaTensor x12;
			float x13;
			x12 = x7;
			x12 = x12.clone();
			x13 = -1;
			x11 = x12.times_i(x13);

			// val X1819 = Convolv(1,1)(X1818,cv11_W,cv11_B)
			JCudaTensor x14;
			JCudaTensor x15, x16, x17;
			x15 = x9;
			x16 = x18;
			x17 = x19;
			x14 = x20.forward(x15, x16, x17);

			// val X1820 = ReLU()(X1819)
			JCudaTensor x21;
			JCudaTensor x22;
			x22 = x14;
			x21 = x23.forward(x22);

			// val X1821 = Convolv(1,1)(X1820,cv12_W,cv12_B)
			JCudaTensor x24;
			JCudaTensor x25, x26, x27;
			x25 = x21;
			x26 = x28;
			x27 = x29;
			x24 = x30.forward(x25, x26, x27);

			// val X1822 = ReLU()(X1821)
			JCudaTensor x31;
			JCudaTensor x32;
			x32 = x24;
			x31 = x23.forward(x32);

			// val X1823 = Pooling(2,2,0,true)(X1822)
			JCudaTensor x33;
			JCudaTensor x34;
			x34 = x31;
			x33 = x35.forward(x34);

			// val X1824 = Convolv(1,1)(X1823,cv21_W,cv21_B)
			JCudaTensor x36;
			JCudaTensor x37, x38, x39;
			x37 = x33;
			x38 = x40;
			x39 = x41;
			x36 = x42.forward(x37, x38, x39);

			// val X1825 = ReLU()(X1824)
			JCudaTensor x43;
			JCudaTensor x44;
			x44 = x36;
			x43 = x45.forward(x44);

			// val X1826 = Convolv(1,1)(X1825,cv22_W,cv22_B)
			JCudaTensor x46;
			JCudaTensor x47, x48, x49;
			x47 = x43;
			x48 = x50;
			x49 = x51;
			x46 = x52.forward(x47, x48, x49);

			// val X1827 = ReLU()(X1826)
			JCudaTensor x53;
			JCudaTensor x54;
			x54 = x46;
			x53 = x45.forward(x54);

			// val X1828 = Pooling(2,2,0,true)(X1827)
			JCudaTensor x55;
			JCudaTensor x56;
			x56 = x53;
			x55 = x57.forward(x56);

			// val X1829 = Convolv(1,1)(X1828,cv31_W,cv31_B)
			JCudaTensor x58;
			JCudaTensor x59, x60, x61;
			x59 = x55;
			x60 = x62;
			x61 = x63;
			x58 = x64.forward(x59, x60, x61);

			// val X1830 = ReLU()(X1829)
			JCudaTensor x65;
			JCudaTensor x66;
			x66 = x58;
			x65 = x67.forward(x66);

			// val X1831 = Convolv(1,1)(X1830,cv32_W,cv32_B)
			JCudaTensor x68;
			JCudaTensor x69, x70, x71;
			x69 = x65;
			x70 = x72;
			x71 = x73;
			x68 = x74.forward(x69, x70, x71);

			// val X1832 = ReLU()(X1831)
			JCudaTensor x75;
			JCudaTensor x76;
			x76 = x68;
			x75 = x67.forward(x76);

			// val X1833 = Convolv(1,1)(X1832,cv33_W,cv33_B)
			JCudaTensor x77;
			JCudaTensor x78, x79, x80;
			x78 = x75;
			x79 = x81;
			x80 = x82;
			x77 = x74.forward(x78, x79, x80);

			// val X1834 = ReLU()(X1833)
			JCudaTensor x83;
			JCudaTensor x84;
			x84 = x77;
			x83 = x67.forward(x84);

			// val X1835 = Pooling(2,2,0,true)(X1834)
			JCudaTensor x85;
			JCudaTensor x86;
			x86 = x83;
			x85 = x87.forward(x86);

			// val X1836 = Convolv(1,1)(X1835,cv41_W,cv41_B)
			JCudaTensor x88;
			JCudaTensor x89, x90, x91;
			x89 = x85;
			x90 = x92;
			x91 = x93;
			x88 = x94.forward(x89, x90, x91);

			// val X1837 = ReLU()(X1836)
			JCudaTensor x95;
			JCudaTensor x96;
			x96 = x88;
			x95 = x97.forward(x96);

			// val X1838 = Convolv(1,1)(X1837,cv42_W,cv42_B)
			JCudaTensor x98;
			JCudaTensor x99, x100, x101;
			x99 = x95;
			x100 = x102;
			x101 = x103;
			x98 = x104.forward(x99, x100, x101);

			// val X1839 = ReLU()(X1838)
			JCudaTensor x105;
			JCudaTensor x106;
			x106 = x98;
			x105 = x97.forward(x106);

			// val X1840 = Convolv(1,1)(X1839,cv43_W,cv43_B)
			JCudaTensor x107;
			JCudaTensor x108, x109, x110;
			x108 = x105;
			x109 = x111;
			x110 = x112;
			x107 = x104.forward(x108, x109, x110);

			// val X1841 = ReLU()(X1840)
			JCudaTensor x113;
			JCudaTensor x114;
			x114 = x107;
			x113 = x97.forward(x114);

			// val X1842 = Pooling(2,2,0,true)(X1841)
			JCudaTensor x115;
			JCudaTensor x116;
			x116 = x113;
			x115 = x117.forward(x116);

			// val X1843 = Convolv(1,1)(X1842,cv51_W,cv51_B)
			JCudaTensor x118;
			JCudaTensor x119, x120, x121;
			x119 = x115;
			x120 = x122;
			x121 = x123;
			x118 = x124.forward(x119, x120, x121);

			// val X1844 = ReLU()(X1843)
			JCudaTensor x125;
			JCudaTensor x126;
			x126 = x118;
			x125 = x127.forward(x126);

			// val X1845 = Convolv(1,1)(X1844,cv52_W,cv52_B)
			JCudaTensor x128;
			JCudaTensor x129, x130, x131;
			x129 = x125;
			x130 = x132;
			x131 = x133;
			x128 = x124.forward(x129, x130, x131);

			// val X1846 = ReLU()(X1845)
			JCudaTensor x134;
			JCudaTensor x135;
			x135 = x128;
			x134 = x127.forward(x135);

			// val X1847 = Convolv(1,1)(X1846,cv53_W,cv53_B)
			JCudaTensor x136;
			JCudaTensor x137, x138, x139;
			x137 = x134;
			x138 = x140;
			x139 = x141;
			x136 = x124.forward(x137, x138, x139);

			// val X1848 = ReLU()(X1847)
			JCudaTensor x142;
			JCudaTensor x143;
			x143 = x136;
			x142 = x127.forward(x143);

			// val X1849 = Pooling(2,2,0,true)(X1848)
			JCudaTensor x144;
			JCudaTensor x145;
			x145 = x142;
			x144 = x146.forward(x145);

			// val X1850 = (X1849[1><3])(i | @) * (fc6_W)(j | @)
			JCudaTensor x147;
			JCudaMatrix x148;
			JCudaMatrix x149;
			JCudaTensor x150;
			JCudaTensor x151;
			x151 = x144;
			x150 = x151.flatten(1, new int[]{512, 7, 7});
			x148 = x150.asMatrix(1, true);
			JCudaTensor x152;
			x152 = x153;
			x149 = x152.asMatrix(1, true);
			x147 = x148.times(x149);

			// val X1852 = (X1850 + (i) => fc6_B)
			JCudaTensor x154;
			JCudaTensor x155, x156;
			x155 = x147;
			x156 = x157;
			x154 = x156.copy(64, x155);

			// val X1853 = ReLU()(X1852)
			JCudaTensor x158;
			JCudaTensor x159;
			x159 = x154;
			x158 = x160.forward(x159);

			// val X1854 = Dropout(0.5)(X1853)
			JCudaTensor x161;
			JCudaTensor x162;
			x162 = x158;
			x161 = x163.forward(x162);

			// val X1855 = (X1854)(i | @) * (fc7_W)(j | @)
			JCudaTensor x164;
			JCudaMatrix x165;
			JCudaMatrix x166;
			JCudaTensor x167;
			x167 = x161;
			x165 = x167.asMatrix(1, true);
			JCudaTensor x168;
			x168 = x169;
			x166 = x168.asMatrix(1, true);
			x164 = x165.times(x166);

			// val X1857 = (X1855 + (i) => fc7_B)
			JCudaTensor x170;
			JCudaTensor x171, x172;
			x171 = x164;
			x172 = x173;
			x170 = x172.copy(64, x171);

			// val X1858 = ReLU()(X1857)
			JCudaTensor x174;
			JCudaTensor x175;
			x175 = x170;
			x174 = x160.forward(x175);

			// val X1859 = Dropout(0.5)(X1858)
			JCudaTensor x176;
			JCudaTensor x177;
			x177 = x174;
			x176 = x163.forward(x177);

			// val X1860 = (X1859)(i | @) * (fc8_W)(j | @)
			JCudaTensor x178;
			JCudaMatrix x179;
			JCudaMatrix x180;
			JCudaTensor x181;
			x181 = x176;
			x179 = x181.asMatrix(1, true);
			JCudaTensor x182;
			x182 = x183;
			x180 = x182.asMatrix(1, true);
			x178 = x179.times(x180);

			// val X1862 = (X1860 + (i) => fc8_B)
			JCudaTensor x184;
			JCudaTensor x185, x186;
			x185 = x178;
			x186 = x187;
			x184 = x186.copy(64, x185);

			// val X1863 = LogSoftmax()(X1862)
			JCudaTensor x188;
			JCudaTensor x189;
			x189 = x184;
			x188 = x190.forward(x189);

			// Dealloc(X1862)
			JCudaTensor x191;
			x191 = x184;
			x191.free();

			// val X892 = (X891 / |64|)
			JCudaTensor x192;
			JCudaTensor x193;
			float x194;
			x193 = x11;
			float x195;
			x195 = 64;
			x194 = 1 / x195;
			x192 = x193.times_i(x194);

			// Cost(((0 - (X1864 . X1863)) / |64|))
			float x196;
			float x197;
			float x198;
			float x199;
			JCudaTensor x200, x201;
			x200 = x7;
			x201 = x188;
			x199 = x200.dot(x201);
			x197 = - x199;
			x198 = 64;
			x196 = x197 / x198;
			System.out.println(x5 + " " + x196);
			if (Float.isNaN(x196)) { System.exit(-1); }

			// Dealloc(X1864)
			JCudaTensor x202;
			x202 = x7;
			x202.free();

			// val X939 = X892 * d_LogSoftmax()(X1863)/d_X1862
			JCudaTensor x203;
			JCudaTensor x204, x205;
			x204 = x192;
			x205 = x188;
			x203 = x190.backward(x204, x205);

			// Dealloc(X892)
			JCudaTensor x206;
			x206 = x192;
			x206.free();

			// Dealloc(X1863)
			JCudaTensor x207;
			x207 = x188;
			x207.free();

			// val m1 = (i625) => fc8_W[@, i625]
			JCudaMatrix x208;
			JCudaTensor x209;
			x209 = x183;
			x208 = x209.asMatrix(1, false);

			// val X981 = (X939)(i624 | @) * m1
			JCudaTensor x210;
			JCudaMatrix x211;
			JCudaMatrix x212;
			JCudaTensor x213;
			x213 = x203;
			x211 = x213.asMatrix(1, true);
			x212 = x208;
			x210 = x211.times(x212);

			// val m49 = (i17) => X939[@, i17]
			JCudaMatrix x214;
			JCudaTensor x215;
			x215 = x203;
			x214 = x215.asMatrix(1, false);

			// val m51 = (i21) => X1859[@, i21]
			JCudaMatrix x216;
			JCudaTensor x217;
			x217 = x176;
			x216 = x217.asMatrix(1, false);

			// fc8_B <~~ Sum(m49)
			float x218, x219;
			x218 = lrn_rate_1;
			x219 = decay_1;
			JCudaMatrix x220;
			x220 = x214;
			x220.sum(x187, x218, x219);

			// fc8_W <~~ m49 * m51
			float x221, x222;
			x221 = lrn_rate_1;
			x222 = decay_1;
			JCudaMatrix x223;
			JCudaMatrix x224;
			x223 = x214;
			x224 = x216;
			x223.times(x224, x183, x221, x222);

			// Dealloc(X939)
			JCudaTensor x225;
			x225 = x203;
			x225.free();

			// Dealloc(X1859)
			JCudaTensor x226;
			x226 = x176;
			x226.free();

			// val X982 = X981 * d_Dropout(0.5)()/d_X1858
			JCudaTensor x227;
			JCudaTensor x228;
			x228 = x210;
			x227 = x163.backward(x228);

			// Dealloc(X981)
			JCudaTensor x229;
			x229 = x210;
			x229.free();

			// val X984 = X982 * d_ReLU()(X1858)/d_X1857
			JCudaTensor x230;
			JCudaTensor x231, x232;
			x231 = x227;
			x232 = x174;
			x230 = x160.backward(x231, x232);

			// Dealloc(X1858)
			JCudaTensor x233;
			x233 = x174;
			x233.free();

			// val m2 = (i629) => fc7_W[@, i629]
			JCudaMatrix x234;
			JCudaTensor x235;
			x235 = x169;
			x234 = x235.asMatrix(1, false);

			// val X985 = (X984)(i628 | @) * m2
			JCudaTensor x236;
			JCudaMatrix x237;
			JCudaMatrix x238;
			JCudaTensor x239;
			x239 = x230;
			x237 = x239.asMatrix(1, true);
			x238 = x234;
			x236 = x237.times(x238);

			// val m46 = (i30) => X984[@, i30]
			JCudaMatrix x240;
			JCudaTensor x241;
			x241 = x230;
			x240 = x241.asMatrix(1, false);

			// val m48 = (i34) => X1854[@, i34]
			JCudaMatrix x242;
			JCudaTensor x243;
			x243 = x161;
			x242 = x243.asMatrix(1, false);

			// fc7_W <~~ m46 * m48
			float x244, x245;
			x244 = lrn_rate_1;
			x245 = decay_1;
			JCudaMatrix x246;
			JCudaMatrix x247;
			x246 = x240;
			x247 = x242;
			x246.times(x247, x169, x244, x245);

			// Dealloc(X1854)
			JCudaTensor x248;
			x248 = x161;
			x248.free();

			// fc7_B <~~ Sum(m46)
			float x249, x250;
			x249 = lrn_rate_1;
			x250 = decay_1;
			JCudaMatrix x251;
			x251 = x240;
			x251.sum(x173, x249, x250);

			// Dealloc(X984)
			JCudaTensor x252;
			x252 = x230;
			x252.free();

			// val X986 = X985 * d_Dropout(0.5)()/d_X1853
			JCudaTensor x253;
			JCudaTensor x254;
			x254 = x236;
			x253 = x163.backward(x254);

			// Dealloc(X985)
			JCudaTensor x255;
			x255 = x236;
			x255.free();

			// val X988 = X986 * d_ReLU()(X1853)/d_X1852
			JCudaTensor x256;
			JCudaTensor x257, x258;
			x257 = x253;
			x258 = x158;
			x256 = x160.backward(x257, x258);

			// Dealloc(X1853)
			JCudaTensor x259;
			x259 = x158;
			x259.free();

			// val m3 = (i633) => fc6_W[@, i633]
			JCudaMatrix x260;
			JCudaTensor x261;
			x261 = x153;
			x260 = x261.asMatrix(1, false);

			// val m43 = (i57) => X988[@, i57]
			JCudaMatrix x262;
			JCudaTensor x263;
			x263 = x256;
			x262 = x263.asMatrix(1, false);

			// val m45 = (i61) => X1849[1><3][@, i61]
			JCudaMatrix x264;
			JCudaTensor x265;
			JCudaTensor x266;
			x266 = x144;
			x265 = x266.flatten(1, new int[]{512, 7, 7});
			x264 = x265.asMatrix(1, false);

			// val X989 = (X988)(i632 | @) * m3
			JCudaTensor x267;
			JCudaMatrix x268;
			JCudaMatrix x269;
			JCudaTensor x270;
			x270 = x256;
			x268 = x270.asMatrix(1, true);
			x269 = x260;
			x267 = x268.times(x269);

			// fc6_B <~~ Sum(m43)
			float x271, x272;
			x271 = lrn_rate_1;
			x272 = decay_1;
			JCudaMatrix x273;
			x273 = x262;
			x273.sum(x157, x271, x272);

			// fc6_W <~~ m43 * m45
			float x274, x275;
			x274 = lrn_rate_1;
			x275 = decay_1;
			JCudaMatrix x276;
			JCudaMatrix x277;
			x276 = x262;
			x277 = x264;
			x276.times(x277, x153, x274, x275);

			// Dealloc(X988)
			JCudaTensor x278;
			x278 = x256;
			x278.free();

			// val X991 = X989[1<>3] * d_Pooling(2,2,0,true)(X1849,X1848)/d_X1848
			JCudaTensor x279;
			JCudaTensor x280, x281, x282;
			JCudaTensor x283;
			x283 = x267;
			x280 = x283.unflatten(1, new int[]{512, 7, 7});
			x281 = x144;
			x282 = x142;
			x279 = x146.backward(x280, x281, x282);

			// Dealloc(X989)
			JCudaTensor x284;
			x284 = x267;
			x284.free();

			// Dealloc(X1849)
			JCudaTensor x285;
			x285 = x144;
			x285.free();

			// val X993 = X991 * d_ReLU()(X1848)/d_X1847
			JCudaTensor x286;
			JCudaTensor x287, x288;
			x287 = x279;
			x288 = x142;
			x286 = x127.backward(x287, x288);

			// Dealloc(X1848)
			JCudaTensor x289;
			x289 = x142;
			x289.free();

			// cv53_B <~~ X993 * d_Convolv(1,1)()/d_cv53_B
			float x290, x291;
			x290 = lrn_rate_1;
			x291 = decay_1;
			JCudaTensor x292;
			x292 = x286;
			x124.backward_bias(x292, x141, x290, x291);

			// val X994 = X993 * d_Convolv(1,1)(cv53_W)/d_X1846
			JCudaTensor x293;
			JCudaTensor x294, x295;
			x294 = x286;
			x295 = x140;
			x293 = x124.backward_data(x294, x295);

			// cv53_W <~~ X993 * d_Convolv(1,1)(X1846)/d_cv53_W
			float x296, x297;
			x296 = lrn_rate_1;
			x297 = decay_1;
			JCudaTensor x298, x299;
			x298 = x286;
			x299 = x134;
			x124.backward_filter(x298, x299, x140, x296, x297);

			// Dealloc(X993)
			JCudaTensor x300;
			x300 = x286;
			x300.free();

			// val X996 = X994 * d_ReLU()(X1846)/d_X1845
			JCudaTensor x301;
			JCudaTensor x302, x303;
			x302 = x293;
			x303 = x134;
			x301 = x127.backward(x302, x303);

			// Dealloc(X1846)
			JCudaTensor x304;
			x304 = x134;
			x304.free();

			// cv52_B <~~ X996 * d_Convolv(1,1)()/d_cv52_B
			float x305, x306;
			x305 = lrn_rate_1;
			x306 = decay_1;
			JCudaTensor x307;
			x307 = x301;
			x124.backward_bias(x307, x133, x305, x306);

			// val X997 = X996 * d_Convolv(1,1)(cv52_W)/d_X1844
			JCudaTensor x308;
			JCudaTensor x309, x310;
			x309 = x301;
			x310 = x132;
			x308 = x124.backward_data(x309, x310);

			// cv52_W <~~ X996 * d_Convolv(1,1)(X1844)/d_cv52_W
			float x311, x312;
			x311 = lrn_rate_1;
			x312 = decay_1;
			JCudaTensor x313, x314;
			x313 = x301;
			x314 = x125;
			x124.backward_filter(x313, x314, x132, x311, x312);

			// Dealloc(X996)
			JCudaTensor x315;
			x315 = x301;
			x315.free();

			// val X999 = X997 * d_ReLU()(X1844)/d_X1843
			JCudaTensor x316;
			JCudaTensor x317, x318;
			x317 = x308;
			x318 = x125;
			x316 = x127.backward(x317, x318);

			// Dealloc(X1844)
			JCudaTensor x319;
			x319 = x125;
			x319.free();

			// cv51_B <~~ X999 * d_Convolv(1,1)()/d_cv51_B
			float x320, x321;
			x320 = lrn_rate_1;
			x321 = decay_1;
			JCudaTensor x322;
			x322 = x316;
			x124.backward_bias(x322, x123, x320, x321);

			// val X1000 = X999 * d_Convolv(1,1)(cv51_W)/d_X1842
			JCudaTensor x323;
			JCudaTensor x324, x325;
			x324 = x316;
			x325 = x122;
			x323 = x124.backward_data(x324, x325);

			// cv51_W <~~ X999 * d_Convolv(1,1)(X1842)/d_cv51_W
			float x326, x327;
			x326 = lrn_rate_1;
			x327 = decay_1;
			JCudaTensor x328, x329;
			x328 = x316;
			x329 = x115;
			x124.backward_filter(x328, x329, x122, x326, x327);

			// Dealloc(X999)
			JCudaTensor x330;
			x330 = x316;
			x330.free();

			// val X1002 = X1000 * d_Pooling(2,2,0,true)(X1842,X1841)/d_X1841
			JCudaTensor x331;
			JCudaTensor x332, x333, x334;
			x332 = x323;
			x333 = x115;
			x334 = x113;
			x331 = x117.backward(x332, x333, x334);

			// Dealloc(X1000)
			JCudaTensor x335;
			x335 = x323;
			x335.free();

			// Dealloc(X1842)
			JCudaTensor x336;
			x336 = x115;
			x336.free();

			// val X1004 = X1002 * d_ReLU()(X1841)/d_X1840
			JCudaTensor x337;
			JCudaTensor x338, x339;
			x338 = x331;
			x339 = x113;
			x337 = x97.backward(x338, x339);

			// Dealloc(X1841)
			JCudaTensor x340;
			x340 = x113;
			x340.free();

			// cv43_B <~~ X1004 * d_Convolv(1,1)()/d_cv43_B
			float x341, x342;
			x341 = lrn_rate_1;
			x342 = decay_1;
			JCudaTensor x343;
			x343 = x337;
			x104.backward_bias(x343, x112, x341, x342);

			// val X1005 = X1004 * d_Convolv(1,1)(cv43_W)/d_X1839
			JCudaTensor x344;
			JCudaTensor x345, x346;
			x345 = x337;
			x346 = x111;
			x344 = x104.backward_data(x345, x346);

			// cv43_W <~~ X1004 * d_Convolv(1,1)(X1839)/d_cv43_W
			float x347, x348;
			x347 = lrn_rate_1;
			x348 = decay_1;
			JCudaTensor x349, x350;
			x349 = x337;
			x350 = x105;
			x104.backward_filter(x349, x350, x111, x347, x348);

			// Dealloc(X1004)
			JCudaTensor x351;
			x351 = x337;
			x351.free();

			// val X1007 = X1005 * d_ReLU()(X1839)/d_X1838
			JCudaTensor x352;
			JCudaTensor x353, x354;
			x353 = x344;
			x354 = x105;
			x352 = x97.backward(x353, x354);

			// Dealloc(X1839)
			JCudaTensor x355;
			x355 = x105;
			x355.free();

			// cv42_B <~~ X1007 * d_Convolv(1,1)()/d_cv42_B
			float x356, x357;
			x356 = lrn_rate_1;
			x357 = decay_1;
			JCudaTensor x358;
			x358 = x352;
			x104.backward_bias(x358, x103, x356, x357);

			// val X1008 = X1007 * d_Convolv(1,1)(cv42_W)/d_X1837
			JCudaTensor x359;
			JCudaTensor x360, x361;
			x360 = x352;
			x361 = x102;
			x359 = x104.backward_data(x360, x361);

			// cv42_W <~~ X1007 * d_Convolv(1,1)(X1837)/d_cv42_W
			float x362, x363;
			x362 = lrn_rate_1;
			x363 = decay_1;
			JCudaTensor x364, x365;
			x364 = x352;
			x365 = x95;
			x104.backward_filter(x364, x365, x102, x362, x363);

			// Dealloc(X1007)
			JCudaTensor x366;
			x366 = x352;
			x366.free();

			// val X1010 = X1008 * d_ReLU()(X1837)/d_X1836
			JCudaTensor x367;
			JCudaTensor x368, x369;
			x368 = x359;
			x369 = x95;
			x367 = x97.backward(x368, x369);

			// Dealloc(X1837)
			JCudaTensor x370;
			x370 = x95;
			x370.free();

			// cv41_B <~~ X1010 * d_Convolv(1,1)()/d_cv41_B
			float x371, x372;
			x371 = lrn_rate_1;
			x372 = decay_1;
			JCudaTensor x373;
			x373 = x367;
			x94.backward_bias(x373, x93, x371, x372);

			// val X1011 = X1010 * d_Convolv(1,1)(cv41_W)/d_X1835
			JCudaTensor x374;
			JCudaTensor x375, x376;
			x375 = x367;
			x376 = x92;
			x374 = x94.backward_data(x375, x376);

			// cv41_W <~~ X1010 * d_Convolv(1,1)(X1835)/d_cv41_W
			float x377, x378;
			x377 = lrn_rate_1;
			x378 = decay_1;
			JCudaTensor x379, x380;
			x379 = x367;
			x380 = x85;
			x94.backward_filter(x379, x380, x92, x377, x378);

			// Dealloc(X1010)
			JCudaTensor x381;
			x381 = x367;
			x381.free();

			// val X1013 = X1011 * d_Pooling(2,2,0,true)(X1835,X1834)/d_X1834
			JCudaTensor x382;
			JCudaTensor x383, x384, x385;
			x383 = x374;
			x384 = x85;
			x385 = x83;
			x382 = x87.backward(x383, x384, x385);

			// Dealloc(X1011)
			JCudaTensor x386;
			x386 = x374;
			x386.free();

			// Dealloc(X1835)
			JCudaTensor x387;
			x387 = x85;
			x387.free();

			// val X1015 = X1013 * d_ReLU()(X1834)/d_X1833
			JCudaTensor x388;
			JCudaTensor x389, x390;
			x389 = x382;
			x390 = x83;
			x388 = x67.backward(x389, x390);

			// Dealloc(X1834)
			JCudaTensor x391;
			x391 = x83;
			x391.free();

			// cv33_B <~~ X1015 * d_Convolv(1,1)()/d_cv33_B
			float x392, x393;
			x392 = lrn_rate_1;
			x393 = decay_1;
			JCudaTensor x394;
			x394 = x388;
			x74.backward_bias(x394, x82, x392, x393);

			// val X1016 = X1015 * d_Convolv(1,1)(cv33_W)/d_X1832
			JCudaTensor x395;
			JCudaTensor x396, x397;
			x396 = x388;
			x397 = x81;
			x395 = x74.backward_data(x396, x397);

			// cv33_W <~~ X1015 * d_Convolv(1,1)(X1832)/d_cv33_W
			float x398, x399;
			x398 = lrn_rate_1;
			x399 = decay_1;
			JCudaTensor x400, x401;
			x400 = x388;
			x401 = x75;
			x74.backward_filter(x400, x401, x81, x398, x399);

			// Dealloc(X1015)
			JCudaTensor x402;
			x402 = x388;
			x402.free();

			// val X1018 = X1016 * d_ReLU()(X1832)/d_X1831
			JCudaTensor x403;
			JCudaTensor x404, x405;
			x404 = x395;
			x405 = x75;
			x403 = x67.backward(x404, x405);

			// Dealloc(X1832)
			JCudaTensor x406;
			x406 = x75;
			x406.free();

			// cv32_B <~~ X1018 * d_Convolv(1,1)()/d_cv32_B
			float x407, x408;
			x407 = lrn_rate_1;
			x408 = decay_1;
			JCudaTensor x409;
			x409 = x403;
			x74.backward_bias(x409, x73, x407, x408);

			// val X1019 = X1018 * d_Convolv(1,1)(cv32_W)/d_X1830
			JCudaTensor x410;
			JCudaTensor x411, x412;
			x411 = x403;
			x412 = x72;
			x410 = x74.backward_data(x411, x412);

			// cv32_W <~~ X1018 * d_Convolv(1,1)(X1830)/d_cv32_W
			float x413, x414;
			x413 = lrn_rate_1;
			x414 = decay_1;
			JCudaTensor x415, x416;
			x415 = x403;
			x416 = x65;
			x74.backward_filter(x415, x416, x72, x413, x414);

			// Dealloc(X1018)
			JCudaTensor x417;
			x417 = x403;
			x417.free();

			// val X1021 = X1019 * d_ReLU()(X1830)/d_X1829
			JCudaTensor x418;
			JCudaTensor x419, x420;
			x419 = x410;
			x420 = x65;
			x418 = x67.backward(x419, x420);

			// Dealloc(X1830)
			JCudaTensor x421;
			x421 = x65;
			x421.free();

			// cv31_B <~~ X1021 * d_Convolv(1,1)()/d_cv31_B
			float x422, x423;
			x422 = lrn_rate_1;
			x423 = decay_1;
			JCudaTensor x424;
			x424 = x418;
			x64.backward_bias(x424, x63, x422, x423);

			// val X1022 = X1021 * d_Convolv(1,1)(cv31_W)/d_X1828
			JCudaTensor x425;
			JCudaTensor x426, x427;
			x426 = x418;
			x427 = x62;
			x425 = x64.backward_data(x426, x427);

			// cv31_W <~~ X1021 * d_Convolv(1,1)(X1828)/d_cv31_W
			float x428, x429;
			x428 = lrn_rate_1;
			x429 = decay_1;
			JCudaTensor x430, x431;
			x430 = x418;
			x431 = x55;
			x64.backward_filter(x430, x431, x62, x428, x429);

			// Dealloc(X1021)
			JCudaTensor x432;
			x432 = x418;
			x432.free();

			// val X1024 = X1022 * d_Pooling(2,2,0,true)(X1828,X1827)/d_X1827
			JCudaTensor x433;
			JCudaTensor x434, x435, x436;
			x434 = x425;
			x435 = x55;
			x436 = x53;
			x433 = x57.backward(x434, x435, x436);

			// Dealloc(X1022)
			JCudaTensor x437;
			x437 = x425;
			x437.free();

			// Dealloc(X1828)
			JCudaTensor x438;
			x438 = x55;
			x438.free();

			// val X1026 = X1024 * d_ReLU()(X1827)/d_X1826
			JCudaTensor x439;
			JCudaTensor x440, x441;
			x440 = x433;
			x441 = x53;
			x439 = x45.backward(x440, x441);

			// Dealloc(X1827)
			JCudaTensor x442;
			x442 = x53;
			x442.free();

			// cv22_B <~~ X1026 * d_Convolv(1,1)()/d_cv22_B
			float x443, x444;
			x443 = lrn_rate_1;
			x444 = decay_1;
			JCudaTensor x445;
			x445 = x439;
			x52.backward_bias(x445, x51, x443, x444);

			// val X1027 = X1026 * d_Convolv(1,1)(cv22_W)/d_X1825
			JCudaTensor x446;
			JCudaTensor x447, x448;
			x447 = x439;
			x448 = x50;
			x446 = x52.backward_data(x447, x448);

			// cv22_W <~~ X1026 * d_Convolv(1,1)(X1825)/d_cv22_W
			float x449, x450;
			x449 = lrn_rate_1;
			x450 = decay_1;
			JCudaTensor x451, x452;
			x451 = x439;
			x452 = x43;
			x52.backward_filter(x451, x452, x50, x449, x450);

			// Dealloc(X1026)
			JCudaTensor x453;
			x453 = x439;
			x453.free();

			// val X1029 = X1027 * d_ReLU()(X1825)/d_X1824
			JCudaTensor x454;
			JCudaTensor x455, x456;
			x455 = x446;
			x456 = x43;
			x454 = x45.backward(x455, x456);

			// Dealloc(X1825)
			JCudaTensor x457;
			x457 = x43;
			x457.free();

			// cv21_B <~~ X1029 * d_Convolv(1,1)()/d_cv21_B
			float x458, x459;
			x458 = lrn_rate_1;
			x459 = decay_1;
			JCudaTensor x460;
			x460 = x454;
			x42.backward_bias(x460, x41, x458, x459);

			// val X1030 = X1029 * d_Convolv(1,1)(cv21_W)/d_X1823
			JCudaTensor x461;
			JCudaTensor x462, x463;
			x462 = x454;
			x463 = x40;
			x461 = x42.backward_data(x462, x463);

			// cv21_W <~~ X1029 * d_Convolv(1,1)(X1823)/d_cv21_W
			float x464, x465;
			x464 = lrn_rate_1;
			x465 = decay_1;
			JCudaTensor x466, x467;
			x466 = x454;
			x467 = x33;
			x42.backward_filter(x466, x467, x40, x464, x465);

			// Dealloc(X1029)
			JCudaTensor x468;
			x468 = x454;
			x468.free();

			// val X1032 = X1030 * d_Pooling(2,2,0,true)(X1823,X1822)/d_X1822
			JCudaTensor x469;
			JCudaTensor x470, x471, x472;
			x470 = x461;
			x471 = x33;
			x472 = x31;
			x469 = x35.backward(x470, x471, x472);

			// Dealloc(X1030)
			JCudaTensor x473;
			x473 = x461;
			x473.free();

			// Dealloc(X1823)
			JCudaTensor x474;
			x474 = x33;
			x474.free();

			// val X1034 = X1032 * d_ReLU()(X1822)/d_X1821
			JCudaTensor x475;
			JCudaTensor x476, x477;
			x476 = x469;
			x477 = x31;
			x475 = x23.backward(x476, x477);

			// Dealloc(X1822)
			JCudaTensor x478;
			x478 = x31;
			x478.free();

			// cv12_B <~~ X1034 * d_Convolv(1,1)()/d_cv12_B
			float x479, x480;
			x479 = lrn_rate_1;
			x480 = decay_1;
			JCudaTensor x481;
			x481 = x475;
			x30.backward_bias(x481, x29, x479, x480);

			// val X1035 = X1034 * d_Convolv(1,1)(cv12_W)/d_X1820
			JCudaTensor x482;
			JCudaTensor x483, x484;
			x483 = x475;
			x484 = x28;
			x482 = x30.backward_data(x483, x484);

			// cv12_W <~~ X1034 * d_Convolv(1,1)(X1820)/d_cv12_W
			float x485, x486;
			x485 = lrn_rate_1;
			x486 = decay_1;
			JCudaTensor x487, x488;
			x487 = x475;
			x488 = x21;
			x30.backward_filter(x487, x488, x28, x485, x486);

			// Dealloc(X1034)
			JCudaTensor x489;
			x489 = x475;
			x489.free();

			// val X1037 = X1035 * d_ReLU()(X1820)/d_X1819
			JCudaTensor x490;
			JCudaTensor x491, x492;
			x491 = x482;
			x492 = x21;
			x490 = x23.backward(x491, x492);

			// Dealloc(X1820)
			JCudaTensor x493;
			x493 = x21;
			x493.free();

			// cv11_W <~~ X1037 * d_Convolv(1,1)(X1818)/d_cv11_W
			float x494, x495;
			x494 = lrn_rate_1;
			x495 = decay_1;
			JCudaTensor x496, x497;
			x496 = x490;
			x497 = x9;
			x20.backward_filter(x496, x497, x18, x494, x495);

			// Dealloc(X1818)
			JCudaTensor x498;
			x498 = x9;
			x498.free();

			// cv11_B <~~ X1037 * d_Convolv(1,1)()/d_cv11_B
			float x499, x500;
			x499 = lrn_rate_1;
			x500 = decay_1;
			JCudaTensor x501;
			x501 = x490;
			x20.backward_bias(x501, x19, x499, x500);

			// Dealloc(X1037)
			JCudaTensor x502;
			x502 = x490;
			x502.free();

		}

	}

	static void test() {
		for(int x5=0; x5<test_itr; x5++) {
			JTensorFloatTuple x6 =  x2.nextFloat();
			x3 = x6.image;
			x4 = x6.label;

			// val X1865 = Cuda(X)
			JCudaTensor x503;
			JTensorFloat x504;
			x504 = x3;
			x503 = x504.asJCudaTensor();

			// val X1866 = Convolv(1,1)(X1865,cv11_W,cv11_B)
			JCudaTensor x505;
			JCudaTensor x506, x507, x508;
			x506 = x503;
			x507 = x18;
			x508 = x19;
			x505 = x20.forward(x506, x507, x508);

			// Dealloc(X1865)
			JCudaTensor x509;
			x509 = x503;
			x509.free();

			// val X1867 = ReLU()(X1866)
			JCudaTensor x510;
			JCudaTensor x511;
			x511 = x505;
			x510 = x23.forward(x511);

			// val X1868 = Convolv(1,1)(X1867,cv12_W,cv12_B)
			JCudaTensor x512;
			JCudaTensor x513, x514, x515;
			x513 = x510;
			x514 = x28;
			x515 = x29;
			x512 = x30.forward(x513, x514, x515);

			// Dealloc(X1867)
			JCudaTensor x516;
			x516 = x510;
			x516.free();

			// val X1869 = ReLU()(X1868)
			JCudaTensor x517;
			JCudaTensor x518;
			x518 = x512;
			x517 = x23.forward(x518);

			// val X1870 = Pooling(2,2,0,true)(X1869)
			JCudaTensor x519;
			JCudaTensor x520;
			x520 = x517;
			x519 = x35.forward(x520);

			// Dealloc(X1869)
			JCudaTensor x521;
			x521 = x517;
			x521.free();

			// val X1871 = Convolv(1,1)(X1870,cv21_W,cv21_B)
			JCudaTensor x522;
			JCudaTensor x523, x524, x525;
			x523 = x519;
			x524 = x40;
			x525 = x41;
			x522 = x42.forward(x523, x524, x525);

			// Dealloc(X1870)
			JCudaTensor x526;
			x526 = x519;
			x526.free();

			// val X1872 = ReLU()(X1871)
			JCudaTensor x527;
			JCudaTensor x528;
			x528 = x522;
			x527 = x45.forward(x528);

			// val X1873 = Convolv(1,1)(X1872,cv22_W,cv22_B)
			JCudaTensor x529;
			JCudaTensor x530, x531, x532;
			x530 = x527;
			x531 = x50;
			x532 = x51;
			x529 = x52.forward(x530, x531, x532);

			// Dealloc(X1872)
			JCudaTensor x533;
			x533 = x527;
			x533.free();

			// val X1874 = ReLU()(X1873)
			JCudaTensor x534;
			JCudaTensor x535;
			x535 = x529;
			x534 = x45.forward(x535);

			// val X1875 = Pooling(2,2,0,true)(X1874)
			JCudaTensor x536;
			JCudaTensor x537;
			x537 = x534;
			x536 = x57.forward(x537);

			// Dealloc(X1874)
			JCudaTensor x538;
			x538 = x534;
			x538.free();

			// val X1876 = Convolv(1,1)(X1875,cv31_W,cv31_B)
			JCudaTensor x539;
			JCudaTensor x540, x541, x542;
			x540 = x536;
			x541 = x62;
			x542 = x63;
			x539 = x64.forward(x540, x541, x542);

			// Dealloc(X1875)
			JCudaTensor x543;
			x543 = x536;
			x543.free();

			// val X1877 = ReLU()(X1876)
			JCudaTensor x544;
			JCudaTensor x545;
			x545 = x539;
			x544 = x67.forward(x545);

			// val X1878 = Convolv(1,1)(X1877,cv32_W,cv32_B)
			JCudaTensor x546;
			JCudaTensor x547, x548, x549;
			x547 = x544;
			x548 = x72;
			x549 = x73;
			x546 = x74.forward(x547, x548, x549);

			// Dealloc(X1877)
			JCudaTensor x550;
			x550 = x544;
			x550.free();

			// val X1879 = ReLU()(X1878)
			JCudaTensor x551;
			JCudaTensor x552;
			x552 = x546;
			x551 = x67.forward(x552);

			// val X1880 = Convolv(1,1)(X1879,cv33_W,cv33_B)
			JCudaTensor x553;
			JCudaTensor x554, x555, x556;
			x554 = x551;
			x555 = x81;
			x556 = x82;
			x553 = x74.forward(x554, x555, x556);

			// Dealloc(X1879)
			JCudaTensor x557;
			x557 = x551;
			x557.free();

			// val X1881 = ReLU()(X1880)
			JCudaTensor x558;
			JCudaTensor x559;
			x559 = x553;
			x558 = x67.forward(x559);

			// val X1882 = Pooling(2,2,0,true)(X1881)
			JCudaTensor x560;
			JCudaTensor x561;
			x561 = x558;
			x560 = x87.forward(x561);

			// Dealloc(X1881)
			JCudaTensor x562;
			x562 = x558;
			x562.free();

			// val X1883 = Convolv(1,1)(X1882,cv41_W,cv41_B)
			JCudaTensor x563;
			JCudaTensor x564, x565, x566;
			x564 = x560;
			x565 = x92;
			x566 = x93;
			x563 = x94.forward(x564, x565, x566);

			// Dealloc(X1882)
			JCudaTensor x567;
			x567 = x560;
			x567.free();

			// val X1884 = ReLU()(X1883)
			JCudaTensor x568;
			JCudaTensor x569;
			x569 = x563;
			x568 = x97.forward(x569);

			// val X1885 = Convolv(1,1)(X1884,cv42_W,cv42_B)
			JCudaTensor x570;
			JCudaTensor x571, x572, x573;
			x571 = x568;
			x572 = x102;
			x573 = x103;
			x570 = x104.forward(x571, x572, x573);

			// Dealloc(X1884)
			JCudaTensor x574;
			x574 = x568;
			x574.free();

			// val X1886 = ReLU()(X1885)
			JCudaTensor x575;
			JCudaTensor x576;
			x576 = x570;
			x575 = x97.forward(x576);

			// val X1887 = Convolv(1,1)(X1886,cv43_W,cv43_B)
			JCudaTensor x577;
			JCudaTensor x578, x579, x580;
			x578 = x575;
			x579 = x111;
			x580 = x112;
			x577 = x104.forward(x578, x579, x580);

			// Dealloc(X1886)
			JCudaTensor x581;
			x581 = x575;
			x581.free();

			// val X1888 = ReLU()(X1887)
			JCudaTensor x582;
			JCudaTensor x583;
			x583 = x577;
			x582 = x97.forward(x583);

			// val X1889 = Pooling(2,2,0,true)(X1888)
			JCudaTensor x584;
			JCudaTensor x585;
			x585 = x582;
			x584 = x117.forward(x585);

			// Dealloc(X1888)
			JCudaTensor x586;
			x586 = x582;
			x586.free();

			// val X1890 = Convolv(1,1)(X1889,cv51_W,cv51_B)
			JCudaTensor x587;
			JCudaTensor x588, x589, x590;
			x588 = x584;
			x589 = x122;
			x590 = x123;
			x587 = x124.forward(x588, x589, x590);

			// Dealloc(X1889)
			JCudaTensor x591;
			x591 = x584;
			x591.free();

			// val X1891 = ReLU()(X1890)
			JCudaTensor x592;
			JCudaTensor x593;
			x593 = x587;
			x592 = x127.forward(x593);

			// val X1892 = Convolv(1,1)(X1891,cv52_W,cv52_B)
			JCudaTensor x594;
			JCudaTensor x595, x596, x597;
			x595 = x592;
			x596 = x132;
			x597 = x133;
			x594 = x124.forward(x595, x596, x597);

			// Dealloc(X1891)
			JCudaTensor x598;
			x598 = x592;
			x598.free();

			// val X1893 = ReLU()(X1892)
			JCudaTensor x599;
			JCudaTensor x600;
			x600 = x594;
			x599 = x127.forward(x600);

			// val X1894 = Convolv(1,1)(X1893,cv53_W,cv53_B)
			JCudaTensor x601;
			JCudaTensor x602, x603, x604;
			x602 = x599;
			x603 = x140;
			x604 = x141;
			x601 = x124.forward(x602, x603, x604);

			// Dealloc(X1893)
			JCudaTensor x605;
			x605 = x599;
			x605.free();

			// val X1895 = ReLU()(X1894)
			JCudaTensor x606;
			JCudaTensor x607;
			x607 = x601;
			x606 = x127.forward(x607);

			// val X1896 = Pooling(2,2,0,true)(X1895)
			JCudaTensor x608;
			JCudaTensor x609;
			x609 = x606;
			x608 = x146.forward(x609);

			// Dealloc(X1895)
			JCudaTensor x610;
			x610 = x606;
			x610.free();

			// val X1897 = (X1896[1><3])(i | @) * (fc6_W)(j | @)
			JCudaTensor x611;
			JCudaMatrix x612;
			JCudaMatrix x613;
			JCudaTensor x614;
			JCudaTensor x615;
			x615 = x608;
			x614 = x615.flatten(1, new int[]{512, 7, 7});
			x612 = x614.asMatrix(1, true);
			JCudaTensor x616;
			x616 = x153;
			x613 = x616.asMatrix(1, true);
			x611 = x612.times(x613);

			// Dealloc(X1896)
			JCudaTensor x617;
			x617 = x608;
			x617.free();

			// val X1899 = (X1897 + (i) => fc6_B)
			JCudaTensor x618;
			JCudaTensor x619, x620;
			x619 = x611;
			x620 = x157;
			x618 = x620.copy(64, x619);

			// val X1900 = ReLU()(X1899)
			JCudaTensor x621;
			JCudaTensor x622;
			x622 = x618;
			x621 = x160.forward(x622);

			// val X1901 = Dropout(0.5)(X1900)
			JCudaTensor x623;
			JCudaTensor x624;
			x624 = x621;
			x623 = x163.forward(x624);

			// Dealloc(X1900)
			JCudaTensor x625;
			x625 = x621;
			x625.free();

			// val X1902 = (X1901)(i | @) * (fc7_W)(j | @)
			JCudaTensor x626;
			JCudaMatrix x627;
			JCudaMatrix x628;
			JCudaTensor x629;
			x629 = x623;
			x627 = x629.asMatrix(1, true);
			JCudaTensor x630;
			x630 = x169;
			x628 = x630.asMatrix(1, true);
			x626 = x627.times(x628);

			// Dealloc(X1901)
			JCudaTensor x631;
			x631 = x623;
			x631.free();

			// val X1904 = (X1902 + (i) => fc7_B)
			JCudaTensor x632;
			JCudaTensor x633, x634;
			x633 = x626;
			x634 = x173;
			x632 = x634.copy(64, x633);

			// val X1905 = ReLU()(X1904)
			JCudaTensor x635;
			JCudaTensor x636;
			x636 = x632;
			x635 = x160.forward(x636);

			// val X1906 = Dropout(0.5)(X1905)
			JCudaTensor x637;
			JCudaTensor x638;
			x638 = x635;
			x637 = x163.forward(x638);

			// Dealloc(X1905)
			JCudaTensor x639;
			x639 = x635;
			x639.free();

			// val X1907 = (X1906)(i | @) * (fc8_W)(j | @)
			JCudaTensor x640;
			JCudaMatrix x641;
			JCudaMatrix x642;
			JCudaTensor x643;
			x643 = x637;
			x641 = x643.asMatrix(1, true);
			JCudaTensor x644;
			x644 = x183;
			x642 = x644.asMatrix(1, true);
			x640 = x641.times(x642);

			// Dealloc(X1906)
			JCudaTensor x645;
			x645 = x637;
			x645.free();

			// val X1909 = (X1907 + (i) => fc8_B)
			JCudaTensor x646;
			JCudaTensor x647, x648;
			x647 = x640;
			x648 = x187;
			x646 = x648.copy(64, x647);

			// Precision(Accuracy(1))
			float x650;
			JCudaTensor x651;
			JTensorFloat x652;
			x651 = x646;
			x652 = x4;
			x650 = x651.accuracy(x652, 1);
			System.out.println(x5 + " test precision "  + x650);
			x649 += x650;

			// Dealloc(X1909)
			JCudaTensor x653;
			x653 = x646;
			x653.free();

		}
		System.out.println();
		System.out.println("average precision: " + x649/10);
		System.out.println(); 
	}

}