package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Vgg {
static{
// comment the first or both lines below for memory efficient mode
JCudaTensor.enableMemoryCache();
JCudaTensor.enableWorkspaceCache();
}
// decay
static float decay = 5.0E-4f;
// lrn_rate
static float lrn_rate = -0.1f;
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/vgg";
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
static JCudnnConvolution x17 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,3,3},new int[]{64}, 1, 1);
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
// (Imagenet,false)
static ImagenetFactory x2 = ImagenetFactory.getFactory(test_data_path, test_size, new int[]{64, 3, 224, 224}, 1000, true);
// (Imagenet,true)
static ImagenetFactory x1 = ImagenetFactory.getFactory(train_data_path, train_size, new int[]{64, 3, 224, 224}, 1000, false);
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
// Precision(Accuracy(X1913, Y, 1))
static float x969;
// X
static JTensorFloat x3;
// Y
static JTensorFloat x4;
// cv11_B
static JCudaTensor x16 = JTensor.constFloat(0.0f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
// cv11_W
static JCudaTensor x15 = JTensor.randomFloat(-0.27216554f, 0.27216554f, 64, 3, 3, 3).load(network_dir + "/cv11_W").asJCudaTensor();
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
x16.save(network_dir + "/cv11_B");
x15.save(network_dir + "/cv11_W");
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
x15.free();
x187.free();
x140.free();
x103.free();
x51.free();
x169.free();
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
x16.free();
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
x57.free();
x17.free();
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

// val X1822 = Cuda(X)
JCudaTensor x7;
JTensorFloat x8;
x8 = x3;
x7 = x8.asJCudaTensor();

// val X1868 = Cuda(Indicator(Y, 1000))
JCudaTensor x9;
JTensorFloat x10;
x10 = x4.asIndicator(1000);
x9 = x10.asJCudaTensor();

// val X1823 = Convolv(1,1)(X1822,cv11_W,cv11_B)
JCudaTensor x11;
JCudaTensor x12, x13, x14;
x12 = x7;
x13 = x15;
x14 = x16;
x11 = x17.forward(x12, x13, x14);

// val X895 = - X1868.copy
JCudaTensor x18;
JCudaTensor x19;
float x20;
x19 = x9;
x19 = x19.clone();
x20 = -1;
x18 = x19.times_i(x20);

// val X1824 = ReLU()(X1823)
JCudaTensor x21;
JCudaTensor x22;
x22 = x11;
x21 = x23.forward(x22);

// val X1825 = Convolv(1,1)(X1824,cv12_W,cv12_B)
JCudaTensor x24;
JCudaTensor x25, x26, x27;
x25 = x21;
x26 = x28;
x27 = x29;
x24 = x30.forward(x25, x26, x27);

// val X1826 = ReLU()(X1825)
JCudaTensor x31;
JCudaTensor x32;
x32 = x24;
x31 = x23.forward(x32);

// val X1827 = Pooling(2,2,0,true)(X1826)
JCudaTensor x33;
JCudaTensor x34;
x34 = x31;
x33 = x35.forward(x34);

// val X1828 = Convolv(1,1)(X1827,cv21_W,cv21_B)
JCudaTensor x36;
JCudaTensor x37, x38, x39;
x37 = x33;
x38 = x40;
x39 = x41;
x36 = x42.forward(x37, x38, x39);

// val X1829 = ReLU()(X1828)
JCudaTensor x43;
JCudaTensor x44;
x44 = x36;
x43 = x45.forward(x44);

// val X1830 = Convolv(1,1)(X1829,cv22_W,cv22_B)
JCudaTensor x46;
JCudaTensor x47, x48, x49;
x47 = x43;
x48 = x50;
x49 = x51;
x46 = x52.forward(x47, x48, x49);

// val X1831 = ReLU()(X1830)
JCudaTensor x53;
JCudaTensor x54;
x54 = x46;
x53 = x45.forward(x54);

// val X1832 = Pooling(2,2,0,true)(X1831)
JCudaTensor x55;
JCudaTensor x56;
x56 = x53;
x55 = x57.forward(x56);

// val X1833 = Convolv(1,1)(X1832,cv31_W,cv31_B)
JCudaTensor x58;
JCudaTensor x59, x60, x61;
x59 = x55;
x60 = x62;
x61 = x63;
x58 = x64.forward(x59, x60, x61);

// val X1834 = ReLU()(X1833)
JCudaTensor x65;
JCudaTensor x66;
x66 = x58;
x65 = x67.forward(x66);

// val X1835 = Convolv(1,1)(X1834,cv32_W,cv32_B)
JCudaTensor x68;
JCudaTensor x69, x70, x71;
x69 = x65;
x70 = x72;
x71 = x73;
x68 = x74.forward(x69, x70, x71);

// val X1836 = ReLU()(X1835)
JCudaTensor x75;
JCudaTensor x76;
x76 = x68;
x75 = x67.forward(x76);

// val X1837 = Convolv(1,1)(X1836,cv33_W,cv33_B)
JCudaTensor x77;
JCudaTensor x78, x79, x80;
x78 = x75;
x79 = x81;
x80 = x82;
x77 = x74.forward(x78, x79, x80);

// val X1838 = ReLU()(X1837)
JCudaTensor x83;
JCudaTensor x84;
x84 = x77;
x83 = x67.forward(x84);

// val X1839 = Pooling(2,2,0,true)(X1838)
JCudaTensor x85;
JCudaTensor x86;
x86 = x83;
x85 = x87.forward(x86);

// val X1840 = Convolv(1,1)(X1839,cv41_W,cv41_B)
JCudaTensor x88;
JCudaTensor x89, x90, x91;
x89 = x85;
x90 = x92;
x91 = x93;
x88 = x94.forward(x89, x90, x91);

// val X1841 = ReLU()(X1840)
JCudaTensor x95;
JCudaTensor x96;
x96 = x88;
x95 = x97.forward(x96);

// val X1842 = Convolv(1,1)(X1841,cv42_W,cv42_B)
JCudaTensor x98;
JCudaTensor x99, x100, x101;
x99 = x95;
x100 = x102;
x101 = x103;
x98 = x104.forward(x99, x100, x101);

// val X1843 = ReLU()(X1842)
JCudaTensor x105;
JCudaTensor x106;
x106 = x98;
x105 = x97.forward(x106);

// val X1844 = Convolv(1,1)(X1843,cv43_W,cv43_B)
JCudaTensor x107;
JCudaTensor x108, x109, x110;
x108 = x105;
x109 = x111;
x110 = x112;
x107 = x104.forward(x108, x109, x110);

// val X1845 = ReLU()(X1844)
JCudaTensor x113;
JCudaTensor x114;
x114 = x107;
x113 = x97.forward(x114);

// val X1846 = Pooling(2,2,0,true)(X1845)
JCudaTensor x115;
JCudaTensor x116;
x116 = x113;
x115 = x117.forward(x116);

// val X1847 = Convolv(1,1)(X1846,cv51_W,cv51_B)
JCudaTensor x118;
JCudaTensor x119, x120, x121;
x119 = x115;
x120 = x122;
x121 = x123;
x118 = x124.forward(x119, x120, x121);

// val X1848 = ReLU()(X1847)
JCudaTensor x125;
JCudaTensor x126;
x126 = x118;
x125 = x127.forward(x126);

// val X1849 = Convolv(1,1)(X1848,cv52_W,cv52_B)
JCudaTensor x128;
JCudaTensor x129, x130, x131;
x129 = x125;
x130 = x132;
x131 = x133;
x128 = x124.forward(x129, x130, x131);

// val X1850 = ReLU()(X1849)
JCudaTensor x134;
JCudaTensor x135;
x135 = x128;
x134 = x127.forward(x135);

// val X1851 = Convolv(1,1)(X1850,cv53_W,cv53_B)
JCudaTensor x136;
JCudaTensor x137, x138, x139;
x137 = x134;
x138 = x140;
x139 = x141;
x136 = x124.forward(x137, x138, x139);

// val X1852 = ReLU()(X1851)
JCudaTensor x142;
JCudaTensor x143;
x143 = x136;
x142 = x127.forward(x143);

// val X1853 = Pooling(2,2,0,true)(X1852)
JCudaTensor x144;
JCudaTensor x145;
x145 = x142;
x144 = x146.forward(x145);

// val X1854 = (X1853[1><3])(i1 | @) * (fc6_W)(i2 | @)
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

// val X1856 = (X1854 + (i1) => fc6_B)
JCudaTensor x154;
JCudaTensor x155, x156;
x155 = x147;
x156 = x157;
x154 = x156.copy(64, x155);

// val X1857 = ReLU()(X1856)
JCudaTensor x158;
JCudaTensor x159;
x159 = x154;
x158 = x160.forward(x159);

// val X1858 = Dropout(0.5)(X1857)
JCudaTensor x161;
JCudaTensor x162;
x162 = x158;
x161 = x163.forward(x162);

// val X1859 = (X1858)(i4 | @) * (fc7_W)(i5 | @)
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

// val X1861 = (X1859 + (i4) => fc7_B)
JCudaTensor x170;
JCudaTensor x171, x172;
x171 = x164;
x172 = x173;
x170 = x172.copy(64, x171);

// val X1862 = ReLU()(X1861)
JCudaTensor x174;
JCudaTensor x175;
x175 = x170;
x174 = x160.forward(x175);

// val X1863 = Dropout(0.5)(X1862)
JCudaTensor x176;
JCudaTensor x177;
x177 = x174;
x176 = x163.forward(x177);

// val X1864 = (X1863)(i7 | @) * (fc8_W)(i8 | @)
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

// val X1866 = (X1864 + (i7) => fc8_B)
JCudaTensor x184;
JCudaTensor x185, x186;
x185 = x178;
x186 = x187;
x184 = x186.copy(64, x185);

// val X1867 = LogSoftmax()(X1866)
JCudaTensor x188;
JCudaTensor x189;
x189 = x184;
x188 = x190.forward(x189);

// Dealloc(X1866)
JCudaTensor x191;
x191 = x184;
x191.free();

// val X896 = (X895 / |64|)
JCudaTensor x192;
JCudaTensor x193;
float x194;
x193 = x18;
float x195;
x195 = 64;
x194 = 1 / x195;
x192 = x193.times_i(x194);

// Cost(((0 - (X1868 . X1867)) / |64|))
float x196;
float x197;
float x198;
float x199;
JCudaTensor x200, x201;
x200 = x9;
x201 = x188;
x199 = x200.dot(x201);
x197 = - x199;
x198 = 64;
x196 = x197 / x198;
System.out.println(x5 + " " + x196);
if (Float.isNaN(x196)) { System.exit(-1); }

// Dealloc(X1868)
JCudaTensor x202;
x202 = x9;
x202.free();

// val X943 = X896 * d_LogSoftmax()(X1867)/d_X1866
JCudaTensor x203;
JCudaTensor x204, x205;
x204 = x192;
x205 = x188;
x203 = x190.backward(x204, x205);

// Dealloc(X896)
JCudaTensor x206;
x206 = x192;
x206.free();

// Dealloc(X1867)
JCudaTensor x207;
x207 = x188;
x207.free();

// val m1 = (i631) => fc8_W[@, i631]
JCudaMatrix x208;
JCudaTensor x209;
x209 = x183;
x208 = x209.asMatrix(1, false);

// val X985 = (X943)(i630 | @) * m1
JCudaTensor x210;
JCudaMatrix x211;
JCudaMatrix x212;
JCudaTensor x213;
x213 = x203;
x211 = x213.asMatrix(1, true);
x212 = x208;
x210 = x211.times(x212);

// val m49 = (i23) => X943[@, i23]
JCudaMatrix x214;
JCudaTensor x215;
x215 = x203;
x214 = x215.asMatrix(1, false);

// val m51 = (i27) => X1863[@, i27]
JCudaMatrix x216;
JCudaTensor x217;
x217 = x176;
x216 = x217.asMatrix(1, false);

// fc8_W <~~ m49 * m51
float x218, x219;
float x220;
float x221;
x220 = 1;
x221 = lrn_rate;
x218 = x220 * x221;
float x222;
float x223;
x222 = 1;
float x224;
float x225;
float x226;
float x227;
x226 = 1;
x227 = decay;
x224 = x226 * x227;
float x228;
float x229;
x228 = 1;
x229 = lrn_rate;
x225 = x228 * x229;
x223 = x224 * x225;
x219 = x222 + x223;
JCudaMatrix x230;
JCudaMatrix x231;
x230 = x214;
x231 = x216;
x230.times(x231, x183, x218, x219);

// Dealloc(X1863)
JCudaTensor x232;
x232 = x176;
x232.free();

// fc8_B <~~ Sum(m49)
float x233, x234;
float x235;
float x236;
x235 = 1;
x236 = lrn_rate;
x233 = x235 * x236;
float x237;
float x238;
x237 = 1;
float x239;
float x240;
float x241;
float x242;
x241 = 1;
x242 = decay;
x239 = x241 * x242;
float x243;
float x244;
x243 = 1;
x244 = lrn_rate;
x240 = x243 * x244;
x238 = x239 * x240;
x234 = x237 + x238;
JCudaMatrix x245;
x245 = x214;
x245.sum(x187, x233, x234);

// Dealloc(X943)
JCudaTensor x246;
x246 = x203;
x246.free();

// val X986 = X985 * d_Dropout(0.5)()/d_X1862
JCudaTensor x247;
JCudaTensor x248;
x248 = x210;
x247 = x163.backward(x248);

// Dealloc(X985)
JCudaTensor x249;
x249 = x210;
x249.free();

// val X988 = X986 * d_ReLU()(X1862)/d_X1861
JCudaTensor x250;
JCudaTensor x251, x252;
x251 = x247;
x252 = x174;
x250 = x160.backward(x251, x252);

// Dealloc(X1862)
JCudaTensor x253;
x253 = x174;
x253.free();

// val m2 = (i635) => fc7_W[@, i635]
JCudaMatrix x254;
JCudaTensor x255;
x255 = x169;
x254 = x255.asMatrix(1, false);

// val m46 = (i36) => X988[@, i36]
JCudaMatrix x256;
JCudaTensor x257;
x257 = x250;
x256 = x257.asMatrix(1, false);

// val m48 = (i40) => X1858[@, i40]
JCudaMatrix x258;
JCudaTensor x259;
x259 = x161;
x258 = x259.asMatrix(1, false);

// val X989 = (X988)(i634 | @) * m2
JCudaTensor x260;
JCudaMatrix x261;
JCudaMatrix x262;
JCudaTensor x263;
x263 = x250;
x261 = x263.asMatrix(1, true);
x262 = x254;
x260 = x261.times(x262);

// fc7_B <~~ Sum(m46)
float x264, x265;
float x266;
float x267;
x266 = 1;
x267 = lrn_rate;
x264 = x266 * x267;
float x268;
float x269;
x268 = 1;
float x270;
float x271;
float x272;
float x273;
x272 = 1;
x273 = decay;
x270 = x272 * x273;
float x274;
float x275;
x274 = 1;
x275 = lrn_rate;
x271 = x274 * x275;
x269 = x270 * x271;
x265 = x268 + x269;
JCudaMatrix x276;
x276 = x256;
x276.sum(x173, x264, x265);

// fc7_W <~~ m46 * m48
float x277, x278;
float x279;
float x280;
x279 = 1;
x280 = lrn_rate;
x277 = x279 * x280;
float x281;
float x282;
x281 = 1;
float x283;
float x284;
float x285;
float x286;
x285 = 1;
x286 = decay;
x283 = x285 * x286;
float x287;
float x288;
x287 = 1;
x288 = lrn_rate;
x284 = x287 * x288;
x282 = x283 * x284;
x278 = x281 + x282;
JCudaMatrix x289;
JCudaMatrix x290;
x289 = x256;
x290 = x258;
x289.times(x290, x169, x277, x278);

// Dealloc(X988)
JCudaTensor x291;
x291 = x250;
x291.free();

// Dealloc(X1858)
JCudaTensor x292;
x292 = x161;
x292.free();

// val X990 = X989 * d_Dropout(0.5)()/d_X1857
JCudaTensor x293;
JCudaTensor x294;
x294 = x260;
x293 = x163.backward(x294);

// Dealloc(X989)
JCudaTensor x295;
x295 = x260;
x295.free();

// val X992 = X990 * d_ReLU()(X1857)/d_X1856
JCudaTensor x296;
JCudaTensor x297, x298;
x297 = x293;
x298 = x158;
x296 = x160.backward(x297, x298);

// Dealloc(X1857)
JCudaTensor x299;
x299 = x158;
x299.free();

// val m3 = (i639) => fc6_W[@, i639]
JCudaMatrix x300;
JCudaTensor x301;
x301 = x153;
x300 = x301.asMatrix(1, false);

// val m43 = (i63) => X992[@, i63]
JCudaMatrix x302;
JCudaTensor x303;
x303 = x296;
x302 = x303.asMatrix(1, false);

// val X993 = (X992)(i638 | @) * m3
JCudaTensor x304;
JCudaMatrix x305;
JCudaMatrix x306;
JCudaTensor x307;
x307 = x296;
x305 = x307.asMatrix(1, true);
x306 = x300;
x304 = x305.times(x306);

// val m45 = (i67) => X1853[1><3][@, i67]
JCudaMatrix x308;
JCudaTensor x309;
JCudaTensor x310;
x310 = x144;
x309 = x310.flatten(1, new int[]{512, 7, 7});
x308 = x309.asMatrix(1, false);

// fc6_B <~~ Sum(m43)
float x311, x312;
float x313;
float x314;
x313 = 1;
x314 = lrn_rate;
x311 = x313 * x314;
float x315;
float x316;
x315 = 1;
float x317;
float x318;
float x319;
float x320;
x319 = 1;
x320 = decay;
x317 = x319 * x320;
float x321;
float x322;
x321 = 1;
x322 = lrn_rate;
x318 = x321 * x322;
x316 = x317 * x318;
x312 = x315 + x316;
JCudaMatrix x323;
x323 = x302;
x323.sum(x157, x311, x312);

// fc6_W <~~ m43 * m45
float x324, x325;
float x326;
float x327;
x326 = 1;
x327 = lrn_rate;
x324 = x326 * x327;
float x328;
float x329;
x328 = 1;
float x330;
float x331;
float x332;
float x333;
x332 = 1;
x333 = decay;
x330 = x332 * x333;
float x334;
float x335;
x334 = 1;
x335 = lrn_rate;
x331 = x334 * x335;
x329 = x330 * x331;
x325 = x328 + x329;
JCudaMatrix x336;
JCudaMatrix x337;
x336 = x302;
x337 = x308;
x336.times(x337, x153, x324, x325);

// Dealloc(X992)
JCudaTensor x338;
x338 = x296;
x338.free();

// val X995 = X993[1<>3] * d_Pooling(2,2,0,true)(X1853,X1852)/d_X1852
JCudaTensor x339;
JCudaTensor x340, x341, x342;
JCudaTensor x343;
x343 = x304;
x340 = x343.unflatten(1, new int[]{512, 7, 7});
x341 = x144;
x342 = x142;
x339 = x146.backward(x340, x341, x342);

// Dealloc(X993)
JCudaTensor x344;
x344 = x304;
x344.free();

// Dealloc(X1853)
JCudaTensor x345;
x345 = x144;
x345.free();

// val X997 = X995 * d_ReLU()(X1852)/d_X1851
JCudaTensor x346;
JCudaTensor x347, x348;
x347 = x339;
x348 = x142;
x346 = x127.backward(x347, x348);

// Dealloc(X1852)
JCudaTensor x349;
x349 = x142;
x349.free();

// cv53_B <~~ X997 * d_Convolv(1,1)()/d_cv53_B
float x350, x351;
float x352;
float x353;
x352 = 1;
x353 = lrn_rate;
x350 = x352 * x353;
float x354;
float x355;
x354 = 1;
float x356;
float x357;
float x358;
float x359;
x358 = 1;
x359 = decay;
x356 = x358 * x359;
float x360;
float x361;
x360 = 1;
x361 = lrn_rate;
x357 = x360 * x361;
x355 = x356 * x357;
x351 = x354 + x355;
JCudaTensor x362;
x362 = x346;
x124.backward_bias(x362, x141, x350, x351);

// val X998 = X997 * d_Convolv(1,1)(cv53_W)/d_X1850
JCudaTensor x363;
JCudaTensor x364, x365;
x364 = x346;
x365 = x140;
x363 = x124.backward_data(x364, x365);

// cv53_W <~~ X997 * d_Convolv(1,1)(X1850)/d_cv53_W
float x366, x367;
float x368;
float x369;
x368 = 1;
x369 = lrn_rate;
x366 = x368 * x369;
float x370;
float x371;
x370 = 1;
float x372;
float x373;
float x374;
float x375;
x374 = 1;
x375 = decay;
x372 = x374 * x375;
float x376;
float x377;
x376 = 1;
x377 = lrn_rate;
x373 = x376 * x377;
x371 = x372 * x373;
x367 = x370 + x371;
JCudaTensor x378, x379;
x378 = x346;
x379 = x134;
x124.backward_filter(x378, x379, x140, x366, x367);

// Dealloc(X997)
JCudaTensor x380;
x380 = x346;
x380.free();

// val X1000 = X998 * d_ReLU()(X1850)/d_X1849
JCudaTensor x381;
JCudaTensor x382, x383;
x382 = x363;
x383 = x134;
x381 = x127.backward(x382, x383);

// Dealloc(X1850)
JCudaTensor x384;
x384 = x134;
x384.free();

// cv52_B <~~ X1000 * d_Convolv(1,1)()/d_cv52_B
float x385, x386;
float x387;
float x388;
x387 = 1;
x388 = lrn_rate;
x385 = x387 * x388;
float x389;
float x390;
x389 = 1;
float x391;
float x392;
float x393;
float x394;
x393 = 1;
x394 = decay;
x391 = x393 * x394;
float x395;
float x396;
x395 = 1;
x396 = lrn_rate;
x392 = x395 * x396;
x390 = x391 * x392;
x386 = x389 + x390;
JCudaTensor x397;
x397 = x381;
x124.backward_bias(x397, x133, x385, x386);

// val X1001 = X1000 * d_Convolv(1,1)(cv52_W)/d_X1848
JCudaTensor x398;
JCudaTensor x399, x400;
x399 = x381;
x400 = x132;
x398 = x124.backward_data(x399, x400);

// cv52_W <~~ X1000 * d_Convolv(1,1)(X1848)/d_cv52_W
float x401, x402;
float x403;
float x404;
x403 = 1;
x404 = lrn_rate;
x401 = x403 * x404;
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
x402 = x405 + x406;
JCudaTensor x413, x414;
x413 = x381;
x414 = x125;
x124.backward_filter(x413, x414, x132, x401, x402);

// Dealloc(X1000)
JCudaTensor x415;
x415 = x381;
x415.free();

// val X1003 = X1001 * d_ReLU()(X1848)/d_X1847
JCudaTensor x416;
JCudaTensor x417, x418;
x417 = x398;
x418 = x125;
x416 = x127.backward(x417, x418);

// Dealloc(X1848)
JCudaTensor x419;
x419 = x125;
x419.free();

// cv51_B <~~ X1003 * d_Convolv(1,1)()/d_cv51_B
float x420, x421;
float x422;
float x423;
x422 = 1;
x423 = lrn_rate;
x420 = x422 * x423;
float x424;
float x425;
x424 = 1;
float x426;
float x427;
float x428;
float x429;
x428 = 1;
x429 = decay;
x426 = x428 * x429;
float x430;
float x431;
x430 = 1;
x431 = lrn_rate;
x427 = x430 * x431;
x425 = x426 * x427;
x421 = x424 + x425;
JCudaTensor x432;
x432 = x416;
x124.backward_bias(x432, x123, x420, x421);

// val X1004 = X1003 * d_Convolv(1,1)(cv51_W)/d_X1846
JCudaTensor x433;
JCudaTensor x434, x435;
x434 = x416;
x435 = x122;
x433 = x124.backward_data(x434, x435);

// cv51_W <~~ X1003 * d_Convolv(1,1)(X1846)/d_cv51_W
float x436, x437;
float x438;
float x439;
x438 = 1;
x439 = lrn_rate;
x436 = x438 * x439;
float x440;
float x441;
x440 = 1;
float x442;
float x443;
float x444;
float x445;
x444 = 1;
x445 = decay;
x442 = x444 * x445;
float x446;
float x447;
x446 = 1;
x447 = lrn_rate;
x443 = x446 * x447;
x441 = x442 * x443;
x437 = x440 + x441;
JCudaTensor x448, x449;
x448 = x416;
x449 = x115;
x124.backward_filter(x448, x449, x122, x436, x437);

// Dealloc(X1003)
JCudaTensor x450;
x450 = x416;
x450.free();

// val X1006 = X1004 * d_Pooling(2,2,0,true)(X1846,X1845)/d_X1845
JCudaTensor x451;
JCudaTensor x452, x453, x454;
x452 = x433;
x453 = x115;
x454 = x113;
x451 = x117.backward(x452, x453, x454);

// Dealloc(X1004)
JCudaTensor x455;
x455 = x433;
x455.free();

// Dealloc(X1846)
JCudaTensor x456;
x456 = x115;
x456.free();

// val X1008 = X1006 * d_ReLU()(X1845)/d_X1844
JCudaTensor x457;
JCudaTensor x458, x459;
x458 = x451;
x459 = x113;
x457 = x97.backward(x458, x459);

// Dealloc(X1845)
JCudaTensor x460;
x460 = x113;
x460.free();

// cv43_B <~~ X1008 * d_Convolv(1,1)()/d_cv43_B
float x461, x462;
float x463;
float x464;
x463 = 1;
x464 = lrn_rate;
x461 = x463 * x464;
float x465;
float x466;
x465 = 1;
float x467;
float x468;
float x469;
float x470;
x469 = 1;
x470 = decay;
x467 = x469 * x470;
float x471;
float x472;
x471 = 1;
x472 = lrn_rate;
x468 = x471 * x472;
x466 = x467 * x468;
x462 = x465 + x466;
JCudaTensor x473;
x473 = x457;
x104.backward_bias(x473, x112, x461, x462);

// val X1009 = X1008 * d_Convolv(1,1)(cv43_W)/d_X1843
JCudaTensor x474;
JCudaTensor x475, x476;
x475 = x457;
x476 = x111;
x474 = x104.backward_data(x475, x476);

// cv43_W <~~ X1008 * d_Convolv(1,1)(X1843)/d_cv43_W
float x477, x478;
float x479;
float x480;
x479 = 1;
x480 = lrn_rate;
x477 = x479 * x480;
float x481;
float x482;
x481 = 1;
float x483;
float x484;
float x485;
float x486;
x485 = 1;
x486 = decay;
x483 = x485 * x486;
float x487;
float x488;
x487 = 1;
x488 = lrn_rate;
x484 = x487 * x488;
x482 = x483 * x484;
x478 = x481 + x482;
JCudaTensor x489, x490;
x489 = x457;
x490 = x105;
x104.backward_filter(x489, x490, x111, x477, x478);

// Dealloc(X1008)
JCudaTensor x491;
x491 = x457;
x491.free();

// val X1011 = X1009 * d_ReLU()(X1843)/d_X1842
JCudaTensor x492;
JCudaTensor x493, x494;
x493 = x474;
x494 = x105;
x492 = x97.backward(x493, x494);

// Dealloc(X1843)
JCudaTensor x495;
x495 = x105;
x495.free();

// cv42_B <~~ X1011 * d_Convolv(1,1)()/d_cv42_B
float x496, x497;
float x498;
float x499;
x498 = 1;
x499 = lrn_rate;
x496 = x498 * x499;
float x500;
float x501;
x500 = 1;
float x502;
float x503;
float x504;
float x505;
x504 = 1;
x505 = decay;
x502 = x504 * x505;
float x506;
float x507;
x506 = 1;
x507 = lrn_rate;
x503 = x506 * x507;
x501 = x502 * x503;
x497 = x500 + x501;
JCudaTensor x508;
x508 = x492;
x104.backward_bias(x508, x103, x496, x497);

// val X1012 = X1011 * d_Convolv(1,1)(cv42_W)/d_X1841
JCudaTensor x509;
JCudaTensor x510, x511;
x510 = x492;
x511 = x102;
x509 = x104.backward_data(x510, x511);

// cv42_W <~~ X1011 * d_Convolv(1,1)(X1841)/d_cv42_W
float x512, x513;
float x514;
float x515;
x514 = 1;
x515 = lrn_rate;
x512 = x514 * x515;
float x516;
float x517;
x516 = 1;
float x518;
float x519;
float x520;
float x521;
x520 = 1;
x521 = decay;
x518 = x520 * x521;
float x522;
float x523;
x522 = 1;
x523 = lrn_rate;
x519 = x522 * x523;
x517 = x518 * x519;
x513 = x516 + x517;
JCudaTensor x524, x525;
x524 = x492;
x525 = x95;
x104.backward_filter(x524, x525, x102, x512, x513);

// Dealloc(X1011)
JCudaTensor x526;
x526 = x492;
x526.free();

// val X1014 = X1012 * d_ReLU()(X1841)/d_X1840
JCudaTensor x527;
JCudaTensor x528, x529;
x528 = x509;
x529 = x95;
x527 = x97.backward(x528, x529);

// Dealloc(X1841)
JCudaTensor x530;
x530 = x95;
x530.free();

// cv41_B <~~ X1014 * d_Convolv(1,1)()/d_cv41_B
float x531, x532;
float x533;
float x534;
x533 = 1;
x534 = lrn_rate;
x531 = x533 * x534;
float x535;
float x536;
x535 = 1;
float x537;
float x538;
float x539;
float x540;
x539 = 1;
x540 = decay;
x537 = x539 * x540;
float x541;
float x542;
x541 = 1;
x542 = lrn_rate;
x538 = x541 * x542;
x536 = x537 * x538;
x532 = x535 + x536;
JCudaTensor x543;
x543 = x527;
x94.backward_bias(x543, x93, x531, x532);

// val X1015 = X1014 * d_Convolv(1,1)(cv41_W)/d_X1839
JCudaTensor x544;
JCudaTensor x545, x546;
x545 = x527;
x546 = x92;
x544 = x94.backward_data(x545, x546);

// cv41_W <~~ X1014 * d_Convolv(1,1)(X1839)/d_cv41_W
float x547, x548;
float x549;
float x550;
x549 = 1;
x550 = lrn_rate;
x547 = x549 * x550;
float x551;
float x552;
x551 = 1;
float x553;
float x554;
float x555;
float x556;
x555 = 1;
x556 = decay;
x553 = x555 * x556;
float x557;
float x558;
x557 = 1;
x558 = lrn_rate;
x554 = x557 * x558;
x552 = x553 * x554;
x548 = x551 + x552;
JCudaTensor x559, x560;
x559 = x527;
x560 = x85;
x94.backward_filter(x559, x560, x92, x547, x548);

// Dealloc(X1014)
JCudaTensor x561;
x561 = x527;
x561.free();

// val X1017 = X1015 * d_Pooling(2,2,0,true)(X1839,X1838)/d_X1838
JCudaTensor x562;
JCudaTensor x563, x564, x565;
x563 = x544;
x564 = x85;
x565 = x83;
x562 = x87.backward(x563, x564, x565);

// Dealloc(X1015)
JCudaTensor x566;
x566 = x544;
x566.free();

// Dealloc(X1839)
JCudaTensor x567;
x567 = x85;
x567.free();

// val X1019 = X1017 * d_ReLU()(X1838)/d_X1837
JCudaTensor x568;
JCudaTensor x569, x570;
x569 = x562;
x570 = x83;
x568 = x67.backward(x569, x570);

// Dealloc(X1838)
JCudaTensor x571;
x571 = x83;
x571.free();

// cv33_B <~~ X1019 * d_Convolv(1,1)()/d_cv33_B
float x572, x573;
float x574;
float x575;
x574 = 1;
x575 = lrn_rate;
x572 = x574 * x575;
float x576;
float x577;
x576 = 1;
float x578;
float x579;
float x580;
float x581;
x580 = 1;
x581 = decay;
x578 = x580 * x581;
float x582;
float x583;
x582 = 1;
x583 = lrn_rate;
x579 = x582 * x583;
x577 = x578 * x579;
x573 = x576 + x577;
JCudaTensor x584;
x584 = x568;
x74.backward_bias(x584, x82, x572, x573);

// val X1020 = X1019 * d_Convolv(1,1)(cv33_W)/d_X1836
JCudaTensor x585;
JCudaTensor x586, x587;
x586 = x568;
x587 = x81;
x585 = x74.backward_data(x586, x587);

// cv33_W <~~ X1019 * d_Convolv(1,1)(X1836)/d_cv33_W
float x588, x589;
float x590;
float x591;
x590 = 1;
x591 = lrn_rate;
x588 = x590 * x591;
float x592;
float x593;
x592 = 1;
float x594;
float x595;
float x596;
float x597;
x596 = 1;
x597 = decay;
x594 = x596 * x597;
float x598;
float x599;
x598 = 1;
x599 = lrn_rate;
x595 = x598 * x599;
x593 = x594 * x595;
x589 = x592 + x593;
JCudaTensor x600, x601;
x600 = x568;
x601 = x75;
x74.backward_filter(x600, x601, x81, x588, x589);

// Dealloc(X1019)
JCudaTensor x602;
x602 = x568;
x602.free();

// val X1022 = X1020 * d_ReLU()(X1836)/d_X1835
JCudaTensor x603;
JCudaTensor x604, x605;
x604 = x585;
x605 = x75;
x603 = x67.backward(x604, x605);

// Dealloc(X1836)
JCudaTensor x606;
x606 = x75;
x606.free();

// cv32_B <~~ X1022 * d_Convolv(1,1)()/d_cv32_B
float x607, x608;
float x609;
float x610;
x609 = 1;
x610 = lrn_rate;
x607 = x609 * x610;
float x611;
float x612;
x611 = 1;
float x613;
float x614;
float x615;
float x616;
x615 = 1;
x616 = decay;
x613 = x615 * x616;
float x617;
float x618;
x617 = 1;
x618 = lrn_rate;
x614 = x617 * x618;
x612 = x613 * x614;
x608 = x611 + x612;
JCudaTensor x619;
x619 = x603;
x74.backward_bias(x619, x73, x607, x608);

// val X1023 = X1022 * d_Convolv(1,1)(cv32_W)/d_X1834
JCudaTensor x620;
JCudaTensor x621, x622;
x621 = x603;
x622 = x72;
x620 = x74.backward_data(x621, x622);

// cv32_W <~~ X1022 * d_Convolv(1,1)(X1834)/d_cv32_W
float x623, x624;
float x625;
float x626;
x625 = 1;
x626 = lrn_rate;
x623 = x625 * x626;
float x627;
float x628;
x627 = 1;
float x629;
float x630;
float x631;
float x632;
x631 = 1;
x632 = decay;
x629 = x631 * x632;
float x633;
float x634;
x633 = 1;
x634 = lrn_rate;
x630 = x633 * x634;
x628 = x629 * x630;
x624 = x627 + x628;
JCudaTensor x635, x636;
x635 = x603;
x636 = x65;
x74.backward_filter(x635, x636, x72, x623, x624);

// Dealloc(X1022)
JCudaTensor x637;
x637 = x603;
x637.free();

// val X1025 = X1023 * d_ReLU()(X1834)/d_X1833
JCudaTensor x638;
JCudaTensor x639, x640;
x639 = x620;
x640 = x65;
x638 = x67.backward(x639, x640);

// Dealloc(X1834)
JCudaTensor x641;
x641 = x65;
x641.free();

// cv31_B <~~ X1025 * d_Convolv(1,1)()/d_cv31_B
float x642, x643;
float x644;
float x645;
x644 = 1;
x645 = lrn_rate;
x642 = x644 * x645;
float x646;
float x647;
x646 = 1;
float x648;
float x649;
float x650;
float x651;
x650 = 1;
x651 = decay;
x648 = x650 * x651;
float x652;
float x653;
x652 = 1;
x653 = lrn_rate;
x649 = x652 * x653;
x647 = x648 * x649;
x643 = x646 + x647;
JCudaTensor x654;
x654 = x638;
x64.backward_bias(x654, x63, x642, x643);

// val X1026 = X1025 * d_Convolv(1,1)(cv31_W)/d_X1832
JCudaTensor x655;
JCudaTensor x656, x657;
x656 = x638;
x657 = x62;
x655 = x64.backward_data(x656, x657);

// cv31_W <~~ X1025 * d_Convolv(1,1)(X1832)/d_cv31_W
float x658, x659;
float x660;
float x661;
x660 = 1;
x661 = lrn_rate;
x658 = x660 * x661;
float x662;
float x663;
x662 = 1;
float x664;
float x665;
float x666;
float x667;
x666 = 1;
x667 = decay;
x664 = x666 * x667;
float x668;
float x669;
x668 = 1;
x669 = lrn_rate;
x665 = x668 * x669;
x663 = x664 * x665;
x659 = x662 + x663;
JCudaTensor x670, x671;
x670 = x638;
x671 = x55;
x64.backward_filter(x670, x671, x62, x658, x659);

// Dealloc(X1025)
JCudaTensor x672;
x672 = x638;
x672.free();

// val X1028 = X1026 * d_Pooling(2,2,0,true)(X1832,X1831)/d_X1831
JCudaTensor x673;
JCudaTensor x674, x675, x676;
x674 = x655;
x675 = x55;
x676 = x53;
x673 = x57.backward(x674, x675, x676);

// Dealloc(X1026)
JCudaTensor x677;
x677 = x655;
x677.free();

// Dealloc(X1832)
JCudaTensor x678;
x678 = x55;
x678.free();

// val X1030 = X1028 * d_ReLU()(X1831)/d_X1830
JCudaTensor x679;
JCudaTensor x680, x681;
x680 = x673;
x681 = x53;
x679 = x45.backward(x680, x681);

// Dealloc(X1831)
JCudaTensor x682;
x682 = x53;
x682.free();

// cv22_B <~~ X1030 * d_Convolv(1,1)()/d_cv22_B
float x683, x684;
float x685;
float x686;
x685 = 1;
x686 = lrn_rate;
x683 = x685 * x686;
float x687;
float x688;
x687 = 1;
float x689;
float x690;
float x691;
float x692;
x691 = 1;
x692 = decay;
x689 = x691 * x692;
float x693;
float x694;
x693 = 1;
x694 = lrn_rate;
x690 = x693 * x694;
x688 = x689 * x690;
x684 = x687 + x688;
JCudaTensor x695;
x695 = x679;
x52.backward_bias(x695, x51, x683, x684);

// val X1031 = X1030 * d_Convolv(1,1)(cv22_W)/d_X1829
JCudaTensor x696;
JCudaTensor x697, x698;
x697 = x679;
x698 = x50;
x696 = x52.backward_data(x697, x698);

// cv22_W <~~ X1030 * d_Convolv(1,1)(X1829)/d_cv22_W
float x699, x700;
float x701;
float x702;
x701 = 1;
x702 = lrn_rate;
x699 = x701 * x702;
float x703;
float x704;
x703 = 1;
float x705;
float x706;
float x707;
float x708;
x707 = 1;
x708 = decay;
x705 = x707 * x708;
float x709;
float x710;
x709 = 1;
x710 = lrn_rate;
x706 = x709 * x710;
x704 = x705 * x706;
x700 = x703 + x704;
JCudaTensor x711, x712;
x711 = x679;
x712 = x43;
x52.backward_filter(x711, x712, x50, x699, x700);

// Dealloc(X1030)
JCudaTensor x713;
x713 = x679;
x713.free();

// val X1033 = X1031 * d_ReLU()(X1829)/d_X1828
JCudaTensor x714;
JCudaTensor x715, x716;
x715 = x696;
x716 = x43;
x714 = x45.backward(x715, x716);

// Dealloc(X1829)
JCudaTensor x717;
x717 = x43;
x717.free();

// cv21_B <~~ X1033 * d_Convolv(1,1)()/d_cv21_B
float x718, x719;
float x720;
float x721;
x720 = 1;
x721 = lrn_rate;
x718 = x720 * x721;
float x722;
float x723;
x722 = 1;
float x724;
float x725;
float x726;
float x727;
x726 = 1;
x727 = decay;
x724 = x726 * x727;
float x728;
float x729;
x728 = 1;
x729 = lrn_rate;
x725 = x728 * x729;
x723 = x724 * x725;
x719 = x722 + x723;
JCudaTensor x730;
x730 = x714;
x42.backward_bias(x730, x41, x718, x719);

// val X1034 = X1033 * d_Convolv(1,1)(cv21_W)/d_X1827
JCudaTensor x731;
JCudaTensor x732, x733;
x732 = x714;
x733 = x40;
x731 = x42.backward_data(x732, x733);

// cv21_W <~~ X1033 * d_Convolv(1,1)(X1827)/d_cv21_W
float x734, x735;
float x736;
float x737;
x736 = 1;
x737 = lrn_rate;
x734 = x736 * x737;
float x738;
float x739;
x738 = 1;
float x740;
float x741;
float x742;
float x743;
x742 = 1;
x743 = decay;
x740 = x742 * x743;
float x744;
float x745;
x744 = 1;
x745 = lrn_rate;
x741 = x744 * x745;
x739 = x740 * x741;
x735 = x738 + x739;
JCudaTensor x746, x747;
x746 = x714;
x747 = x33;
x42.backward_filter(x746, x747, x40, x734, x735);

// Dealloc(X1033)
JCudaTensor x748;
x748 = x714;
x748.free();

// val X1036 = X1034 * d_Pooling(2,2,0,true)(X1827,X1826)/d_X1826
JCudaTensor x749;
JCudaTensor x750, x751, x752;
x750 = x731;
x751 = x33;
x752 = x31;
x749 = x35.backward(x750, x751, x752);

// Dealloc(X1034)
JCudaTensor x753;
x753 = x731;
x753.free();

// Dealloc(X1827)
JCudaTensor x754;
x754 = x33;
x754.free();

// val X1038 = X1036 * d_ReLU()(X1826)/d_X1825
JCudaTensor x755;
JCudaTensor x756, x757;
x756 = x749;
x757 = x31;
x755 = x23.backward(x756, x757);

// Dealloc(X1826)
JCudaTensor x758;
x758 = x31;
x758.free();

// cv12_B <~~ X1038 * d_Convolv(1,1)()/d_cv12_B
float x759, x760;
float x761;
float x762;
x761 = 1;
x762 = lrn_rate;
x759 = x761 * x762;
float x763;
float x764;
x763 = 1;
float x765;
float x766;
float x767;
float x768;
x767 = 1;
x768 = decay;
x765 = x767 * x768;
float x769;
float x770;
x769 = 1;
x770 = lrn_rate;
x766 = x769 * x770;
x764 = x765 * x766;
x760 = x763 + x764;
JCudaTensor x771;
x771 = x755;
x30.backward_bias(x771, x29, x759, x760);

// val X1039 = X1038 * d_Convolv(1,1)(cv12_W)/d_X1824
JCudaTensor x772;
JCudaTensor x773, x774;
x773 = x755;
x774 = x28;
x772 = x30.backward_data(x773, x774);

// cv12_W <~~ X1038 * d_Convolv(1,1)(X1824)/d_cv12_W
float x775, x776;
float x777;
float x778;
x777 = 1;
x778 = lrn_rate;
x775 = x777 * x778;
float x779;
float x780;
x779 = 1;
float x781;
float x782;
float x783;
float x784;
x783 = 1;
x784 = decay;
x781 = x783 * x784;
float x785;
float x786;
x785 = 1;
x786 = lrn_rate;
x782 = x785 * x786;
x780 = x781 * x782;
x776 = x779 + x780;
JCudaTensor x787, x788;
x787 = x755;
x788 = x21;
x30.backward_filter(x787, x788, x28, x775, x776);

// Dealloc(X1038)
JCudaTensor x789;
x789 = x755;
x789.free();

// val X1041 = X1039 * d_ReLU()(X1824)/d_X1823
JCudaTensor x790;
JCudaTensor x791, x792;
x791 = x772;
x792 = x21;
x790 = x23.backward(x791, x792);

// Dealloc(X1824)
JCudaTensor x793;
x793 = x21;
x793.free();

// cv11_W <~~ X1041 * d_Convolv(1,1)(X1822)/d_cv11_W
float x794, x795;
float x796;
float x797;
x796 = 1;
x797 = lrn_rate;
x794 = x796 * x797;
float x798;
float x799;
x798 = 1;
float x800;
float x801;
float x802;
float x803;
x802 = 1;
x803 = decay;
x800 = x802 * x803;
float x804;
float x805;
x804 = 1;
x805 = lrn_rate;
x801 = x804 * x805;
x799 = x800 * x801;
x795 = x798 + x799;
JCudaTensor x806, x807;
x806 = x790;
x807 = x7;
x17.backward_filter(x806, x807, x15, x794, x795);

// Dealloc(X1822)
JCudaTensor x808;
x808 = x7;
x808.free();

// cv11_B <~~ X1041 * d_Convolv(1,1)()/d_cv11_B
float x809, x810;
float x811;
float x812;
x811 = 1;
x812 = lrn_rate;
x809 = x811 * x812;
float x813;
float x814;
x813 = 1;
float x815;
float x816;
float x817;
float x818;
x817 = 1;
x818 = decay;
x815 = x817 * x818;
float x819;
float x820;
x819 = 1;
x820 = lrn_rate;
x816 = x819 * x820;
x814 = x815 * x816;
x810 = x813 + x814;
JCudaTensor x821;
x821 = x790;
x17.backward_bias(x821, x16, x809, x810);

// Dealloc(X1041)
JCudaTensor x822;
x822 = x790;
x822.free();

}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X1869 = Cuda(X)
JCudaTensor x823;
JTensorFloat x824;
x824 = x3;
x823 = x824.asJCudaTensor();

// val X1870 = Convolv(1,1)(X1869,cv11_W,cv11_B)
JCudaTensor x825;
JCudaTensor x826, x827, x828;
x826 = x823;
x827 = x15;
x828 = x16;
x825 = x17.forward(x826, x827, x828);

// Dealloc(X1869)
JCudaTensor x829;
x829 = x823;
x829.free();

// val X1871 = ReLU()(X1870)
JCudaTensor x830;
JCudaTensor x831;
x831 = x825;
x830 = x23.forward(x831);

// val X1872 = Convolv(1,1)(X1871,cv12_W,cv12_B)
JCudaTensor x832;
JCudaTensor x833, x834, x835;
x833 = x830;
x834 = x28;
x835 = x29;
x832 = x30.forward(x833, x834, x835);

// Dealloc(X1871)
JCudaTensor x836;
x836 = x830;
x836.free();

// val X1873 = ReLU()(X1872)
JCudaTensor x837;
JCudaTensor x838;
x838 = x832;
x837 = x23.forward(x838);

// val X1874 = Pooling(2,2,0,true)(X1873)
JCudaTensor x839;
JCudaTensor x840;
x840 = x837;
x839 = x35.forward(x840);

// Dealloc(X1873)
JCudaTensor x841;
x841 = x837;
x841.free();

// val X1875 = Convolv(1,1)(X1874,cv21_W,cv21_B)
JCudaTensor x842;
JCudaTensor x843, x844, x845;
x843 = x839;
x844 = x40;
x845 = x41;
x842 = x42.forward(x843, x844, x845);

// Dealloc(X1874)
JCudaTensor x846;
x846 = x839;
x846.free();

// val X1876 = ReLU()(X1875)
JCudaTensor x847;
JCudaTensor x848;
x848 = x842;
x847 = x45.forward(x848);

// val X1877 = Convolv(1,1)(X1876,cv22_W,cv22_B)
JCudaTensor x849;
JCudaTensor x850, x851, x852;
x850 = x847;
x851 = x50;
x852 = x51;
x849 = x52.forward(x850, x851, x852);

// Dealloc(X1876)
JCudaTensor x853;
x853 = x847;
x853.free();

// val X1878 = ReLU()(X1877)
JCudaTensor x854;
JCudaTensor x855;
x855 = x849;
x854 = x45.forward(x855);

// val X1879 = Pooling(2,2,0,true)(X1878)
JCudaTensor x856;
JCudaTensor x857;
x857 = x854;
x856 = x57.forward(x857);

// Dealloc(X1878)
JCudaTensor x858;
x858 = x854;
x858.free();

// val X1880 = Convolv(1,1)(X1879,cv31_W,cv31_B)
JCudaTensor x859;
JCudaTensor x860, x861, x862;
x860 = x856;
x861 = x62;
x862 = x63;
x859 = x64.forward(x860, x861, x862);

// Dealloc(X1879)
JCudaTensor x863;
x863 = x856;
x863.free();

// val X1881 = ReLU()(X1880)
JCudaTensor x864;
JCudaTensor x865;
x865 = x859;
x864 = x67.forward(x865);

// val X1882 = Convolv(1,1)(X1881,cv32_W,cv32_B)
JCudaTensor x866;
JCudaTensor x867, x868, x869;
x867 = x864;
x868 = x72;
x869 = x73;
x866 = x74.forward(x867, x868, x869);

// Dealloc(X1881)
JCudaTensor x870;
x870 = x864;
x870.free();

// val X1883 = ReLU()(X1882)
JCudaTensor x871;
JCudaTensor x872;
x872 = x866;
x871 = x67.forward(x872);

// val X1884 = Convolv(1,1)(X1883,cv33_W,cv33_B)
JCudaTensor x873;
JCudaTensor x874, x875, x876;
x874 = x871;
x875 = x81;
x876 = x82;
x873 = x74.forward(x874, x875, x876);

// Dealloc(X1883)
JCudaTensor x877;
x877 = x871;
x877.free();

// val X1885 = ReLU()(X1884)
JCudaTensor x878;
JCudaTensor x879;
x879 = x873;
x878 = x67.forward(x879);

// val X1886 = Pooling(2,2,0,true)(X1885)
JCudaTensor x880;
JCudaTensor x881;
x881 = x878;
x880 = x87.forward(x881);

// Dealloc(X1885)
JCudaTensor x882;
x882 = x878;
x882.free();

// val X1887 = Convolv(1,1)(X1886,cv41_W,cv41_B)
JCudaTensor x883;
JCudaTensor x884, x885, x886;
x884 = x880;
x885 = x92;
x886 = x93;
x883 = x94.forward(x884, x885, x886);

// Dealloc(X1886)
JCudaTensor x887;
x887 = x880;
x887.free();

// val X1888 = ReLU()(X1887)
JCudaTensor x888;
JCudaTensor x889;
x889 = x883;
x888 = x97.forward(x889);

// val X1889 = Convolv(1,1)(X1888,cv42_W,cv42_B)
JCudaTensor x890;
JCudaTensor x891, x892, x893;
x891 = x888;
x892 = x102;
x893 = x103;
x890 = x104.forward(x891, x892, x893);

// Dealloc(X1888)
JCudaTensor x894;
x894 = x888;
x894.free();

// val X1890 = ReLU()(X1889)
JCudaTensor x895;
JCudaTensor x896;
x896 = x890;
x895 = x97.forward(x896);

// val X1891 = Convolv(1,1)(X1890,cv43_W,cv43_B)
JCudaTensor x897;
JCudaTensor x898, x899, x900;
x898 = x895;
x899 = x111;
x900 = x112;
x897 = x104.forward(x898, x899, x900);

// Dealloc(X1890)
JCudaTensor x901;
x901 = x895;
x901.free();

// val X1892 = ReLU()(X1891)
JCudaTensor x902;
JCudaTensor x903;
x903 = x897;
x902 = x97.forward(x903);

// val X1893 = Pooling(2,2,0,true)(X1892)
JCudaTensor x904;
JCudaTensor x905;
x905 = x902;
x904 = x117.forward(x905);

// Dealloc(X1892)
JCudaTensor x906;
x906 = x902;
x906.free();

// val X1894 = Convolv(1,1)(X1893,cv51_W,cv51_B)
JCudaTensor x907;
JCudaTensor x908, x909, x910;
x908 = x904;
x909 = x122;
x910 = x123;
x907 = x124.forward(x908, x909, x910);

// Dealloc(X1893)
JCudaTensor x911;
x911 = x904;
x911.free();

// val X1895 = ReLU()(X1894)
JCudaTensor x912;
JCudaTensor x913;
x913 = x907;
x912 = x127.forward(x913);

// val X1896 = Convolv(1,1)(X1895,cv52_W,cv52_B)
JCudaTensor x914;
JCudaTensor x915, x916, x917;
x915 = x912;
x916 = x132;
x917 = x133;
x914 = x124.forward(x915, x916, x917);

// Dealloc(X1895)
JCudaTensor x918;
x918 = x912;
x918.free();

// val X1897 = ReLU()(X1896)
JCudaTensor x919;
JCudaTensor x920;
x920 = x914;
x919 = x127.forward(x920);

// val X1898 = Convolv(1,1)(X1897,cv53_W,cv53_B)
JCudaTensor x921;
JCudaTensor x922, x923, x924;
x922 = x919;
x923 = x140;
x924 = x141;
x921 = x124.forward(x922, x923, x924);

// Dealloc(X1897)
JCudaTensor x925;
x925 = x919;
x925.free();

// val X1899 = ReLU()(X1898)
JCudaTensor x926;
JCudaTensor x927;
x927 = x921;
x926 = x127.forward(x927);

// val X1900 = Pooling(2,2,0,true)(X1899)
JCudaTensor x928;
JCudaTensor x929;
x929 = x926;
x928 = x146.forward(x929);

// Dealloc(X1899)
JCudaTensor x930;
x930 = x926;
x930.free();

// val X1901 = (X1900[1><3])(i1 | @) * (fc6_W)(i2 | @)
JCudaTensor x931;
JCudaMatrix x932;
JCudaMatrix x933;
JCudaTensor x934;
JCudaTensor x935;
x935 = x928;
x934 = x935.flatten(1, new int[]{512, 7, 7});
x932 = x934.asMatrix(1, true);
JCudaTensor x936;
x936 = x153;
x933 = x936.asMatrix(1, true);
x931 = x932.times(x933);

// Dealloc(X1900)
JCudaTensor x937;
x937 = x928;
x937.free();

// val X1903 = (X1901 + (i1) => fc6_B)
JCudaTensor x938;
JCudaTensor x939, x940;
x939 = x931;
x940 = x157;
x938 = x940.copy(64, x939);

// val X1904 = ReLU()(X1903)
JCudaTensor x941;
JCudaTensor x942;
x942 = x938;
x941 = x160.forward(x942);

// val X1905 = Dropout(0.5)(X1904)
JCudaTensor x943;
JCudaTensor x944;
x944 = x941;
x943 = x163.forward(x944);

// Dealloc(X1904)
JCudaTensor x945;
x945 = x941;
x945.free();

// val X1906 = (X1905)(i4 | @) * (fc7_W)(i5 | @)
JCudaTensor x946;
JCudaMatrix x947;
JCudaMatrix x948;
JCudaTensor x949;
x949 = x943;
x947 = x949.asMatrix(1, true);
JCudaTensor x950;
x950 = x169;
x948 = x950.asMatrix(1, true);
x946 = x947.times(x948);

// Dealloc(X1905)
JCudaTensor x951;
x951 = x943;
x951.free();

// val X1908 = (X1906 + (i4) => fc7_B)
JCudaTensor x952;
JCudaTensor x953, x954;
x953 = x946;
x954 = x173;
x952 = x954.copy(64, x953);

// val X1909 = ReLU()(X1908)
JCudaTensor x955;
JCudaTensor x956;
x956 = x952;
x955 = x160.forward(x956);

// val X1910 = Dropout(0.5)(X1909)
JCudaTensor x957;
JCudaTensor x958;
x958 = x955;
x957 = x163.forward(x958);

// Dealloc(X1909)
JCudaTensor x959;
x959 = x955;
x959.free();

// val X1911 = (X1910)(i7 | @) * (fc8_W)(i8 | @)
JCudaTensor x960;
JCudaMatrix x961;
JCudaMatrix x962;
JCudaTensor x963;
x963 = x957;
x961 = x963.asMatrix(1, true);
JCudaTensor x964;
x964 = x183;
x962 = x964.asMatrix(1, true);
x960 = x961.times(x962);

// Dealloc(X1910)
JCudaTensor x965;
x965 = x957;
x965.free();

// val X1913 = (X1911 + (i7) => fc8_B)
JCudaTensor x966;
JCudaTensor x967, x968;
x967 = x960;
x968 = x187;
x966 = x968.copy(64, x967);

// Precision(Accuracy(X1913, Y, 1))
float x970;
JCudaTensor x971;
JTensorFloat x972;
x971 = x966;
x972 = x4;
x970 = x971.accuracy(x972, 1);
System.out.println(x5 + " test precision "  + x970);
x969 += x970;

// Dealloc(X1913)
JCudaTensor x973;
x973 = x966;
x973.free();

}
System.out.println();
System.out.println("average precision: " + x969/test_itr);
System.out.println(); 
}

}