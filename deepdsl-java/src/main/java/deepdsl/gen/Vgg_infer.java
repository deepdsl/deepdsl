package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;

// This file is for inference only, which needs trained parameters.
public class Vgg_infer {
static{
// comment the first or both lines below for memory efficient mode
JCudaTensor.enableMemoryCache();
JCudaTensor.enableWorkspaceCache();
}
// network_dir
static String network_dir = "src/main/java/deepdsl/gen/vgg";
// test_data_path
static String test_data_path = "dataset/imagenet/ilsvrc12_val_lmdb";
// test_itr
static int test_itr = 10;
// test_size
static int test_size = 10000;

// (Convolv(1,1),List(List(64, 128, 112, 112), List(128, 128, 3, 3), List(128)))
static JCudnnConvolution x49 = new JCudnnConvolution(new int[]{64,128,112,112},new int[]{128,128,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(64, 128, 56, 56), List(256, 128, 3, 3), List(256)))
static JCudnnConvolution x63 = new JCudnnConvolution(new int[]{64,128,56,56},new int[]{256,128,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(List(64, 256, 28, 28), List(512, 256, 3, 3), List(512)))
static JCudnnConvolution x97 = new JCudnnConvolution(new int[]{64,256,28,28},new int[]{512,256,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(List(64, 256, 56, 56), List(256, 256, 3, 3), List(256)))
static JCudnnConvolution x74 = new JCudnnConvolution(new int[]{64,256,56,56},new int[]{256,256,3,3},new int[]{256}, 1, 1);
// (Convolv(1,1),List(List(64, 3, 224, 224), List(64, 3, 3, 3), List(64)))
static JCudnnConvolution x13 = new JCudnnConvolution(new int[]{64,3,224,224},new int[]{64,3,3,3},new int[]{64}, 1, 1);
// (Convolv(1,1),List(List(64, 512, 14, 14), List(512, 512, 3, 3), List(512)))
static JCudnnConvolution x131 = new JCudnnConvolution(new int[]{64,512,14,14},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(List(64, 512, 28, 28), List(512, 512, 3, 3), List(512)))
static JCudnnConvolution x108 = new JCudnnConvolution(new int[]{64,512,28,28},new int[]{512,512,3,3},new int[]{512}, 1, 1);
// (Convolv(1,1),List(List(64, 64, 112, 112), List(128, 64, 3, 3), List(128)))
static JCudnnConvolution x38 = new JCudnnConvolution(new int[]{64,64,112,112},new int[]{128,64,3,3},new int[]{128}, 1, 1);
// (Convolv(1,1),List(List(64, 64, 224, 224), List(64, 64, 3, 3), List(64)))
static JCudnnConvolution x24 = new JCudnnConvolution(new int[]{64,64,224,224},new int[]{64,64,3,3},new int[]{64}, 1, 1);
// (Dropout(0.5),List(List(64, 4096)))
static JCudnnDropout x175 = new JCudnnDropout(new int[]{64,4096}, 0.5f);
// (Imagenet,false)
static ImagenetFactory x1 = ImagenetFactory.getFactory(test_data_path, test_size, new int[]{64, 3, 224, 224}, 1000, true);
// (Pooling(2,2,0,true),List(List(64, 128, 112, 112)))
static JCudnnPooling x55 = new JCudnnPooling(new int[]{64,128,112,112}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(64, 256, 56, 56)))
static JCudnnPooling x89 = new JCudnnPooling(new int[]{64,256,56,56}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(64, 512, 14, 14)))
static JCudnnPooling x156 = new JCudnnPooling(new int[]{64,512,14,14}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(64, 512, 28, 28)))
static JCudnnPooling x123 = new JCudnnPooling(new int[]{64,512,28,28}, 2, 2, 0, PoolingType.MAX);
// (Pooling(2,2,0,true),List(List(64, 64, 224, 224)))
static JCudnnPooling x30 = new JCudnnPooling(new int[]{64,64,224,224}, 2, 2, 0, PoolingType.MAX);
// (ReLU(),List(List(64, 128, 112, 112)))
static JCudnnActivation x42 = new JCudnnActivation(new int[]{64,128,112,112}, ActivationMode.RELU);
// (ReLU(),List(List(64, 256, 56, 56)))
static JCudnnActivation x67 = new JCudnnActivation(new int[]{64,256,56,56}, ActivationMode.RELU);
// (ReLU(),List(List(64, 4096)))
static JCudnnActivation x172 = new JCudnnActivation(new int[]{64,4096}, ActivationMode.RELU);
// (ReLU(),List(List(64, 512, 14, 14)))
static JCudnnActivation x135 = new JCudnnActivation(new int[]{64,512,14,14}, ActivationMode.RELU);
// (ReLU(),List(List(64, 512, 28, 28)))
static JCudnnActivation x101 = new JCudnnActivation(new int[]{64,512,28,28}, ActivationMode.RELU);
// (ReLU(),List(List(64, 64, 224, 224)))
static JCudnnActivation x17 = new JCudnnActivation(new int[]{64,64,224,224}, ActivationMode.RELU);
// X
static JTensorFloat x2;
// cv11_B
static JCudaTensor x12 = JTensor.constFloat(0.0f, 64).load(network_dir + "/cv11_B").asJCudaTensor();
// cv11_W
static JCudaTensor x11 = JTensor.randomFloat(-0.27216554f, 0.27216554f, 64, 3, 3, 3).load(network_dir + "/cv11_W").asJCudaTensor();
// cv12_B
static JCudaTensor x23 = JTensor.constFloat(0.0f, 64).load(network_dir + "/cv12_B").asJCudaTensor();
// cv12_W
static JCudaTensor x22 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 64, 64, 3, 3).load(network_dir + "/cv12_W").asJCudaTensor();
// cv21_B
static JCudaTensor x37 = JTensor.constFloat(0.0f, 128).load(network_dir + "/cv21_B").asJCudaTensor();
// cv21_W
static JCudaTensor x36 = JTensor.randomFloat(-0.058925565f, 0.058925565f, 128, 64, 3, 3).load(network_dir + "/cv21_W").asJCudaTensor();
// cv22_B
static JCudaTensor x48 = JTensor.constFloat(0.0f, 128).load(network_dir + "/cv22_B").asJCudaTensor();
// cv22_W
static JCudaTensor x47 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 128, 128, 3, 3).load(network_dir + "/cv22_W").asJCudaTensor();
// cv31_B
static JCudaTensor x62 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv31_B").asJCudaTensor();
// cv31_W
static JCudaTensor x61 = JTensor.randomFloat(-0.041666668f, 0.041666668f, 256, 128, 3, 3).load(network_dir + "/cv31_W").asJCudaTensor();
// cv32_B
static JCudaTensor x73 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv32_B").asJCudaTensor();
// cv32_W
static JCudaTensor x72 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/cv32_W").asJCudaTensor();
// cv33_B
static JCudaTensor x83 = JTensor.constFloat(0.0f, 256).load(network_dir + "/cv33_B").asJCudaTensor();
// cv33_W
static JCudaTensor x82 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 256, 256, 3, 3).load(network_dir + "/cv33_W").asJCudaTensor();
// cv41_B
static JCudaTensor x96 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv41_B").asJCudaTensor();
// cv41_W
static JCudaTensor x95 = JTensor.randomFloat(-0.029462783f, 0.029462783f, 512, 256, 3, 3).load(network_dir + "/cv41_W").asJCudaTensor();
// cv42_B
static JCudaTensor x107 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv42_B").asJCudaTensor();
// cv42_W
static JCudaTensor x106 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv42_W").asJCudaTensor();
// cv43_B
static JCudaTensor x117 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv43_B").asJCudaTensor();
// cv43_W
static JCudaTensor x116 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv43_W").asJCudaTensor();
// cv51_B
static JCudaTensor x130 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv51_B").asJCudaTensor();
// cv51_W
static JCudaTensor x129 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv51_W").asJCudaTensor();
// cv52_B
static JCudaTensor x141 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv52_B").asJCudaTensor();
// cv52_W
static JCudaTensor x140 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv52_W").asJCudaTensor();
// cv53_B
static JCudaTensor x150 = JTensor.constFloat(0.0f, 512).load(network_dir + "/cv53_B").asJCudaTensor();
// cv53_W
static JCudaTensor x149 = JTensor.randomFloat(-0.020833334f, 0.020833334f, 512, 512, 3, 3).load(network_dir + "/cv53_W").asJCudaTensor();
// fc6_B
static JCudaTensor x169 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc6_B").asJCudaTensor();
// fc6_W
static JCudaTensor x164 = JTensor.randomFloat(-0.008928572f, 0.008928572f, 4096, 25088).load(network_dir + "/fc6_W").asJCudaTensor();
// fc7_B
static JCudaTensor x187 = JTensor.constFloat(0.0f, 4096).load(network_dir + "/fc7_B").asJCudaTensor();
// fc7_W
static JCudaTensor x182 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 4096, 4096).load(network_dir + "/fc7_W").asJCudaTensor();
// fc8_B
static JCudaTensor x203 = JTensor.constFloat(0.0f, 1000).load(network_dir + "/fc8_B").asJCudaTensor();
// fc8_W
static JCudaTensor x198 = JTensor.randomFloat(-0.022097087f, 0.022097087f, 1000, 4096).load(network_dir + "/fc8_W").asJCudaTensor();

public static void main(String[] args){
test();
x61.free();
x187.free();
x116.free();
x140.free();
x164.free();
x169.free();
x106.free();
x130.free();
x107.free();
x96.free();
x36.free();
x22.free();
x141.free();
x203.free();
x23.free();
x12.free();
x83.free();
x82.free();
x48.free();
x73.free();
x62.free();
x149.free();
x72.free();
x129.free();
x182.free();
x37.free();
x150.free();
x198.free();
x95.free();
x11.free();
x47.free();
x117.free();
x131.free();
x24.free();
x97.free();
x175.free();
x63.free();
x123.free();
x49.free();
x67.free();
x172.free();
x74.free();
x89.free();
x55.free();
x108.free();
x17.free();
x13.free();
x38.free();
x30.free();
x156.free();
x101.free();
x42.free();
x135.free();
JCudaTensor.clearMemoryCache();
JCudaFunction.destroy();
}
static void test() {
 for(int x3=0; x3<test_itr; x3++) {
JTensorFloatTuple x4 =  x1.nextFloat();
x2 = x4.image;

// val X1914 = Cuda(X)
JCudaTensor x5;
JTensorFloat x6;
x6 = x2;
x5 = x6.asJCudaTensor();

// val X1915 = Convolv(1,1)(X1914,cv11_W,cv11_B)
JCudaTensor x7;
JCudaTensor x8, x9, x10;
x8 = x5;
x9 = x11;
x10 = x12;
x7 = x13.forward(x8, x9, x10);

// Dealloc(X1914)
JCudaTensor x14;
x14 = x5;
x14.free();

// val X1916 = ReLU()(X1915)
JCudaTensor x15;
JCudaTensor x16;
x16 = x7;
x15 = x17.forward(x16);

// val X1917 = Convolv(1,1)(X1916,cv12_W,cv12_B)
JCudaTensor x18;
JCudaTensor x19, x20, x21;
x19 = x15;
x20 = x22;
x21 = x23;
x18 = x24.forward(x19, x20, x21);

// Dealloc(X1916)
JCudaTensor x25;
x25 = x15;
x25.free();

// val X1918 = ReLU()(X1917)
JCudaTensor x26;
JCudaTensor x27;
x27 = x18;
x26 = x17.forward(x27);

// val X1919 = Pooling(2,2,0,true)(X1918)
JCudaTensor x28;
JCudaTensor x29;
x29 = x26;
x28 = x30.forward(x29);

// Dealloc(X1918)
JCudaTensor x31;
x31 = x26;
x31.free();

// val X1920 = Convolv(1,1)(X1919,cv21_W,cv21_B)
JCudaTensor x32;
JCudaTensor x33, x34, x35;
x33 = x28;
x34 = x36;
x35 = x37;
x32 = x38.forward(x33, x34, x35);

// Dealloc(X1919)
JCudaTensor x39;
x39 = x28;
x39.free();

// val X1921 = ReLU()(X1920)
JCudaTensor x40;
JCudaTensor x41;
x41 = x32;
x40 = x42.forward(x41);

// val X1922 = Convolv(1,1)(X1921,cv22_W,cv22_B)
JCudaTensor x43;
JCudaTensor x44, x45, x46;
x44 = x40;
x45 = x47;
x46 = x48;
x43 = x49.forward(x44, x45, x46);

// Dealloc(X1921)
JCudaTensor x50;
x50 = x40;
x50.free();

// val X1923 = ReLU()(X1922)
JCudaTensor x51;
JCudaTensor x52;
x52 = x43;
x51 = x42.forward(x52);

// val X1924 = Pooling(2,2,0,true)(X1923)
JCudaTensor x53;
JCudaTensor x54;
x54 = x51;
x53 = x55.forward(x54);

// Dealloc(X1923)
JCudaTensor x56;
x56 = x51;
x56.free();

// val X1925 = Convolv(1,1)(X1924,cv31_W,cv31_B)
JCudaTensor x57;
JCudaTensor x58, x59, x60;
x58 = x53;
x59 = x61;
x60 = x62;
x57 = x63.forward(x58, x59, x60);

// Dealloc(X1924)
JCudaTensor x64;
x64 = x53;
x64.free();

// val X1926 = ReLU()(X1925)
JCudaTensor x65;
JCudaTensor x66;
x66 = x57;
x65 = x67.forward(x66);

// val X1927 = Convolv(1,1)(X1926,cv32_W,cv32_B)
JCudaTensor x68;
JCudaTensor x69, x70, x71;
x69 = x65;
x70 = x72;
x71 = x73;
x68 = x74.forward(x69, x70, x71);

// Dealloc(X1926)
JCudaTensor x75;
x75 = x65;
x75.free();

// val X1928 = ReLU()(X1927)
JCudaTensor x76;
JCudaTensor x77;
x77 = x68;
x76 = x67.forward(x77);

// val X1929 = Convolv(1,1)(X1928,cv33_W,cv33_B)
JCudaTensor x78;
JCudaTensor x79, x80, x81;
x79 = x76;
x80 = x82;
x81 = x83;
x78 = x74.forward(x79, x80, x81);

// Dealloc(X1928)
JCudaTensor x84;
x84 = x76;
x84.free();

// val X1930 = ReLU()(X1929)
JCudaTensor x85;
JCudaTensor x86;
x86 = x78;
x85 = x67.forward(x86);

// val X1931 = Pooling(2,2,0,true)(X1930)
JCudaTensor x87;
JCudaTensor x88;
x88 = x85;
x87 = x89.forward(x88);

// Dealloc(X1930)
JCudaTensor x90;
x90 = x85;
x90.free();

// val X1932 = Convolv(1,1)(X1931,cv41_W,cv41_B)
JCudaTensor x91;
JCudaTensor x92, x93, x94;
x92 = x87;
x93 = x95;
x94 = x96;
x91 = x97.forward(x92, x93, x94);

// Dealloc(X1931)
JCudaTensor x98;
x98 = x87;
x98.free();

// val X1933 = ReLU()(X1932)
JCudaTensor x99;
JCudaTensor x100;
x100 = x91;
x99 = x101.forward(x100);

// val X1934 = Convolv(1,1)(X1933,cv42_W,cv42_B)
JCudaTensor x102;
JCudaTensor x103, x104, x105;
x103 = x99;
x104 = x106;
x105 = x107;
x102 = x108.forward(x103, x104, x105);

// Dealloc(X1933)
JCudaTensor x109;
x109 = x99;
x109.free();

// val X1935 = ReLU()(X1934)
JCudaTensor x110;
JCudaTensor x111;
x111 = x102;
x110 = x101.forward(x111);

// val X1936 = Convolv(1,1)(X1935,cv43_W,cv43_B)
JCudaTensor x112;
JCudaTensor x113, x114, x115;
x113 = x110;
x114 = x116;
x115 = x117;
x112 = x108.forward(x113, x114, x115);

// Dealloc(X1935)
JCudaTensor x118;
x118 = x110;
x118.free();

// val X1937 = ReLU()(X1936)
JCudaTensor x119;
JCudaTensor x120;
x120 = x112;
x119 = x101.forward(x120);

// val X1938 = Pooling(2,2,0,true)(X1937)
JCudaTensor x121;
JCudaTensor x122;
x122 = x119;
x121 = x123.forward(x122);

// Dealloc(X1937)
JCudaTensor x124;
x124 = x119;
x124.free();

// val X1939 = Convolv(1,1)(X1938,cv51_W,cv51_B)
JCudaTensor x125;
JCudaTensor x126, x127, x128;
x126 = x121;
x127 = x129;
x128 = x130;
x125 = x131.forward(x126, x127, x128);

// Dealloc(X1938)
JCudaTensor x132;
x132 = x121;
x132.free();

// val X1940 = ReLU()(X1939)
JCudaTensor x133;
JCudaTensor x134;
x134 = x125;
x133 = x135.forward(x134);

// val X1941 = Convolv(1,1)(X1940,cv52_W,cv52_B)
JCudaTensor x136;
JCudaTensor x137, x138, x139;
x137 = x133;
x138 = x140;
x139 = x141;
x136 = x131.forward(x137, x138, x139);

// Dealloc(X1940)
JCudaTensor x142;
x142 = x133;
x142.free();

// val X1942 = ReLU()(X1941)
JCudaTensor x143;
JCudaTensor x144;
x144 = x136;
x143 = x135.forward(x144);

// val X1943 = Convolv(1,1)(X1942,cv53_W,cv53_B)
JCudaTensor x145;
JCudaTensor x146, x147, x148;
x146 = x143;
x147 = x149;
x148 = x150;
x145 = x131.forward(x146, x147, x148);

// Dealloc(X1942)
JCudaTensor x151;
x151 = x143;
x151.free();

// val X1944 = ReLU()(X1943)
JCudaTensor x152;
JCudaTensor x153;
x153 = x145;
x152 = x135.forward(x153);

// val X1945 = Pooling(2,2,0,true)(X1944)
JCudaTensor x154;
JCudaTensor x155;
x155 = x152;
x154 = x156.forward(x155);

// Dealloc(X1944)
JCudaTensor x157;
x157 = x152;
x157.free();

// val X1946 = (X1945[1><3])(i1 | @) * (fc6_W)(i2 | @)
JCudaTensor x158;
JCudaMatrix x159;
JCudaMatrix x160;
JCudaTensor x161;
JCudaTensor x162;
x162 = x154;
x161 = x162.flatten(1, new int[]{512, 7, 7});
x159 = x161.asMatrix(1, true);
JCudaTensor x163;
x163 = x164;
x160 = x163.asMatrix(1, true);
x158 = x159.times(x160);

// Dealloc(X1945)
JCudaTensor x165;
x165 = x154;
x165.free();

// val X1948 = (X1946 + (i1) => fc6_B)
JCudaTensor x166;
JCudaTensor x167, x168;
x167 = x158;
x168 = x169;
x166 = x168.copy(64, x167);

// val X1949 = ReLU()(X1948)
JCudaTensor x170;
JCudaTensor x171;
x171 = x166;
x170 = x172.forward(x171);

// val X1950 = Dropout(0.5)(X1949)
JCudaTensor x173;
JCudaTensor x174;
x174 = x170;
x173 = x175.forward(x174);

// Dealloc(X1949)
JCudaTensor x176;
x176 = x170;
x176.free();

// val X1951 = (X1950)(i4 | @) * (fc7_W)(i5 | @)
JCudaTensor x177;
JCudaMatrix x178;
JCudaMatrix x179;
JCudaTensor x180;
x180 = x173;
x178 = x180.asMatrix(1, true);
JCudaTensor x181;
x181 = x182;
x179 = x181.asMatrix(1, true);
x177 = x178.times(x179);

// Dealloc(X1950)
JCudaTensor x183;
x183 = x173;
x183.free();

// val X1953 = (X1951 + (i4) => fc7_B)
JCudaTensor x184;
JCudaTensor x185, x186;
x185 = x177;
x186 = x187;
x184 = x186.copy(64, x185);

// val X1954 = ReLU()(X1953)
JCudaTensor x188;
JCudaTensor x189;
x189 = x184;
x188 = x172.forward(x189);

// val X1955 = Dropout(0.5)(X1954)
JCudaTensor x190;
JCudaTensor x191;
x191 = x188;
x190 = x175.forward(x191);

// Dealloc(X1954)
JCudaTensor x192;
x192 = x188;
x192.free();

// val X1956 = (X1955)(i7 | @) * (fc8_W)(i8 | @)
JCudaTensor x193;
JCudaMatrix x194;
JCudaMatrix x195;
JCudaTensor x196;
x196 = x190;
x194 = x196.asMatrix(1, true);
JCudaTensor x197;
x197 = x198;
x195 = x197.asMatrix(1, true);
x193 = x194.times(x195);

// Dealloc(X1955)
JCudaTensor x199;
x199 = x190;
x199.free();

// val X1958 = (X1956 + (i7) => fc8_B)
JCudaTensor x200;
JCudaTensor x201, x202;
x201 = x193;
x202 = x203;
x200 = x202.copy(64, x201);

// Prediction(X1958)
JCudaTensor x204;
x204 = x200;
System.out.println(x3 + " inference " + java.util.Arrays.toString(x204.prediction()));

// Dealloc(X1958)
JCudaTensor x205;
x205 = x200;
x205.free();

}
 
}

}