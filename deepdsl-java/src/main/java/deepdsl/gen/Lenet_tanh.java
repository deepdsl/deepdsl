package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;


public class Lenet_tanh {
static{
// comment the first or both lines below for memory efficient mode
JCudaTensor.enableMemoryCache();
JCudaTensor.enableWorkspaceCache();
}
// lrn_rate
static float lrn_rate = -0.1f;
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
// Precision(Accuracy(X205, Y, 1))
static float x237;
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

// val X175 = Cuda(X)
JCudaTensor x7;
JTensorFloat x8;
x8 = x3;
x7 = x8.asJCudaTensor();

// val X176 = Convolv(1,0)(X175,W1,B1)
JCudaTensor x9;
JCudaTensor x10, x11, x12;
x10 = x7;
x11 = x13;
x12 = x14;
x9 = x15.forward(x10, x11, x12);

// val X177 = Pooling(2,2,0,true)(X176)
JCudaTensor x16;
JCudaTensor x17;
x17 = x9;
x16 = x18.forward(x17);

// val X178 = Tanh()(X177.copy)
JCudaTensor x19;
JCudaTensor x20;
x20 = x16;
x20 = x20.clone();
x19 = x21.forward(x20);

// val X179 = Convolv(1,0)(X178,W2,B2)
JCudaTensor x22;
JCudaTensor x23, x24, x25;
x23 = x19;
x24 = x26;
x25 = x27;
x22 = x28.forward(x23, x24, x25);

// val X180 = Pooling(2,2,0,true)(X179)
JCudaTensor x29;
JCudaTensor x30;
x30 = x22;
x29 = x31.forward(x30);

// val X181 = Tanh()(X180.copy)
JCudaTensor x32;
JCudaTensor x33;
x33 = x29;
x33 = x33.clone();
x32 = x34.forward(x33);

// val X182 = (X181[1><3])(i10 | @) * (W)(i11 | @)
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

// val X184 = (X182 + (i10) => B)
JCudaTensor x42;
JCudaTensor x43, x44;
x43 = x35;
x44 = x45;
x42 = x44.copy(500, x43);

// val X185 = Tanh()(X184)
JCudaTensor x46;
JCudaTensor x47;
x47 = x42;
x46 = x48.forward(x47);

// val X186 = (X185)(i13 | @) * (Theta)(i14 | @)
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

// val X188 = (X186 + (i13) => B3)
JCudaTensor x55;
JCudaTensor x56, x57;
x56 = x49;
x57 = x58;
x55 = x57.copy(500, x56);

// val X189 = Softmax()(X188)
JCudaTensor x59;
JCudaTensor x60;
x60 = x55;
x59 = x61.forward(x60);

// Dealloc(X188)
JCudaTensor x62;
x62 = x55;
x62.free();

// val X190 = Cuda(Indicator(Y, 10))
JCudaTensor x63;
JTensorFloat x64;
x64 = x4.asIndicator(10);
x63 = x64.asJCudaTensor();

// val X97 = 1/(X189.copy)
JCudaTensor x65;
JCudaTensor x66;
float x67;
x66 = x59;
x66 = x66.clone();
x67 = -1;
x65 = x66.pow(x67);

// val X191 = Log X189.copy
JCudaTensor x68;
JCudaTensor x69;
x69 = x59;
x69 = x69.clone();
x68 = x69.log();

// Cost(((0 - (X190 . X191)) / |500|))
float x70;
float x71;
float x72;
float x73;
JCudaTensor x74, x75;
x74 = x63;
x75 = x68;
x73 = x74.dot(x75);
x71 = - x73;
x72 = 500;
x70 = x71 / x72;
System.out.println(x5 + " " + x70);
if (Float.isNaN(x70)) { System.exit(-1); }

// Dealloc(X191)
JCudaTensor x76;
x76 = x68;
x76.free();

// val X98 = X190.copy .* X97
JCudaTensor x77;
JCudaTensor x78, x79;
x78 = x63;
x78 = x78.clone();
x79 = x65;
x77 = x78.times_i(x79);

// Dealloc(X97)
JCudaTensor x80;
x80 = x65;
x80.free();

// Dealloc(X190)
JCudaTensor x81;
x81 = x63;
x81.free();

// val X99 = - X98
JCudaTensor x82;
JCudaTensor x83;
float x84;
x83 = x77;
x84 = -1;
x82 = x83.times_i(x84);

// val X100 = (X99 / |500|)
JCudaTensor x85;
JCudaTensor x86;
float x87;
x86 = x82;
float x88;
x88 = 500;
x87 = 1 / x88;
x85 = x86.times_i(x87);

// val X116 = X100 * d_Softmax()(X189)/d_X188
JCudaTensor x89;
JCudaTensor x90, x91;
x90 = x85;
x91 = x59;
x89 = x61.backward(x90, x91);

// Dealloc(X100)
JCudaTensor x92;
x92 = x85;
x92.free();

// Dealloc(X189)
JCudaTensor x93;
x93 = x59;
x93.free();

// val m1 = (i119) => Theta[@, i119]
JCudaMatrix x94;
JCudaTensor x95;
x95 = x54;
x94 = x95.asMatrix(1, false);

// val m6 = (i20) => X116[@, i20]
JCudaMatrix x96;
JCudaTensor x97;
x97 = x89;
x96 = x97.asMatrix(1, false);

// val m7 = (i21) => X185[@, i21]
JCudaMatrix x98;
JCudaTensor x99;
x99 = x46;
x98 = x99.asMatrix(1, false);

// val X127 = (X116)(i118 | @) * m1
JCudaTensor x100;
JCudaMatrix x101;
JCudaMatrix x102;
JCudaTensor x103;
x103 = x89;
x101 = x103.asMatrix(1, true);
x102 = x94;
x100 = x101.times(x102);

// B3 <~~ Sum(m6)
float x104, x105;
float x106;
float x107;
x106 = 1;
x107 = lrn_rate;
x104 = x106 * x107;
x105 = 1;
JCudaMatrix x108;
x108 = x96;
x108.sum(x58, x104, x105);

// Theta <~~ m6 * m7
float x109, x110;
float x111;
float x112;
x111 = 1;
x112 = lrn_rate;
x109 = x111 * x112;
x110 = 1;
JCudaMatrix x113;
JCudaMatrix x114;
x113 = x96;
x114 = x98;
x113.times(x114, x54, x109, x110);

// Dealloc(X116)
JCudaTensor x115;
x115 = x89;
x115.free();

// val X129 = X127 * d_Tanh()(X185)/d_X184
JCudaTensor x116;
JCudaTensor x117, x118;
x117 = x100;
x118 = x46;
x116 = x48.backward(x117, x118);

// Dealloc(X185)
JCudaTensor x119;
x119 = x46;
x119.free();

// val m2 = (i123) => W[@, i123]
JCudaMatrix x120;
JCudaTensor x121;
x121 = x41;
x120 = x121.asMatrix(1, false);

// val m8 = (i29) => X129[@, i29]
JCudaMatrix x122;
JCudaTensor x123;
x123 = x116;
x122 = x123.asMatrix(1, false);

// val m9 = (i30) => X181[1><3][@, i30]
JCudaMatrix x124;
JCudaTensor x125;
JCudaTensor x126;
x126 = x32;
x125 = x126.flatten(1, new int[]{50, 4, 4});
x124 = x125.asMatrix(1, false);

// val X130 = (X129)(i122 | @) * m2
JCudaTensor x127;
JCudaMatrix x128;
JCudaMatrix x129;
JCudaTensor x130;
x130 = x116;
x128 = x130.asMatrix(1, true);
x129 = x120;
x127 = x128.times(x129);

// W <~~ m8 * m9
float x131, x132;
float x133;
float x134;
x133 = 1;
x134 = lrn_rate;
x131 = x133 * x134;
x132 = 1;
JCudaMatrix x135;
JCudaMatrix x136;
x135 = x122;
x136 = x124;
x135.times(x136, x41, x131, x132);

// B <~~ Sum(m8)
float x137, x138;
float x139;
float x140;
x139 = 1;
x140 = lrn_rate;
x137 = x139 * x140;
x138 = 1;
JCudaMatrix x141;
x141 = x122;
x141.sum(x45, x137, x138);

// Dealloc(X129)
JCudaTensor x142;
x142 = x116;
x142.free();

// val X132 = X130[1<>3] * d_Tanh()(X181)/d_X180
JCudaTensor x143;
JCudaTensor x144, x145;
JCudaTensor x146;
x146 = x127;
x144 = x146.unflatten(1, new int[]{50, 4, 4});
x145 = x32;
x143 = x34.backward(x144, x145);

// Dealloc(X181)
JCudaTensor x147;
x147 = x32;
x147.free();

// val X134 = X132 * d_Pooling(2,2,0,true)(X180,X179)/d_X179
JCudaTensor x148;
JCudaTensor x149, x150, x151;
x149 = x143;
x150 = x29;
x151 = x22;
x148 = x31.backward(x149, x150, x151);

// Dealloc(X132)
JCudaTensor x152;
x152 = x143;
x152.free();

// Dealloc(X180)
JCudaTensor x153;
x153 = x29;
x153.free();

// Dealloc(X179)
JCudaTensor x154;
x154 = x22;
x154.free();

// B2 <~~ X134 * d_Convolv(1,0)()/d_B2
float x155, x156;
float x157;
float x158;
x157 = 1;
x158 = lrn_rate;
x155 = x157 * x158;
x156 = 1;
JCudaTensor x159;
x159 = x148;
x28.backward_bias(x159, x27, x155, x156);

// val X135 = X134 * d_Convolv(1,0)(W2)/d_X178
JCudaTensor x160;
JCudaTensor x161, x162;
x161 = x148;
x162 = x26;
x160 = x28.backward_data(x161, x162);

// W2 <~~ X134 * d_Convolv(1,0)(X178)/d_W2
float x163, x164;
float x165;
float x166;
x165 = 1;
x166 = lrn_rate;
x163 = x165 * x166;
x164 = 1;
JCudaTensor x167, x168;
x167 = x148;
x168 = x19;
x28.backward_filter(x167, x168, x26, x163, x164);

// Dealloc(X134)
JCudaTensor x169;
x169 = x148;
x169.free();

// val X137 = X135 * d_Tanh()(X178)/d_X177
JCudaTensor x170;
JCudaTensor x171, x172;
x171 = x160;
x172 = x19;
x170 = x21.backward(x171, x172);

// Dealloc(X178)
JCudaTensor x173;
x173 = x19;
x173.free();

// val X139 = X137 * d_Pooling(2,2,0,true)(X177,X176)/d_X176
JCudaTensor x174;
JCudaTensor x175, x176, x177;
x175 = x170;
x176 = x16;
x177 = x9;
x174 = x18.backward(x175, x176, x177);

// Dealloc(X137)
JCudaTensor x178;
x178 = x170;
x178.free();

// Dealloc(X177)
JCudaTensor x179;
x179 = x16;
x179.free();

// Dealloc(X176)
JCudaTensor x180;
x180 = x9;
x180.free();

// B1 <~~ X139 * d_Convolv(1,0)()/d_B1
float x181, x182;
float x183;
float x184;
x183 = 1;
x184 = lrn_rate;
x181 = x183 * x184;
x182 = 1;
JCudaTensor x185;
x185 = x174;
x15.backward_bias(x185, x14, x181, x182);

// W1 <~~ X139 * d_Convolv(1,0)(X175)/d_W1
float x186, x187;
float x188;
float x189;
x188 = 1;
x189 = lrn_rate;
x186 = x188 * x189;
x187 = 1;
JCudaTensor x190, x191;
x190 = x174;
x191 = x7;
x15.backward_filter(x190, x191, x13, x186, x187);

// Dealloc(X139)
JCudaTensor x192;
x192 = x174;
x192.free();

// Dealloc(X175)
JCudaTensor x193;
x193 = x7;
x193.free();

}
 
}

static void test() {
 for(int x5=0; x5<test_itr; x5++) {
JTensorFloatTuple x6 =  x2.nextFloat();
x3 = x6.image;
x4 = x6.label;

// val X192 = Cuda(X)
JCudaTensor x194;
JTensorFloat x195;
x195 = x3;
x194 = x195.asJCudaTensor();

// val X193 = Convolv(1,0)(X192,W1,B1)
JCudaTensor x196;
JCudaTensor x197, x198, x199;
x197 = x194;
x198 = x13;
x199 = x14;
x196 = x15.forward(x197, x198, x199);

// Dealloc(X192)
JCudaTensor x200;
x200 = x194;
x200.free();

// val X194 = Pooling(2,2,0,true)(X193)
JCudaTensor x201;
JCudaTensor x202;
x202 = x196;
x201 = x18.forward(x202);

// Dealloc(X193)
JCudaTensor x203;
x203 = x196;
x203.free();

// val X195 = Tanh()(X194)
JCudaTensor x204;
JCudaTensor x205;
x205 = x201;
x204 = x21.forward(x205);

// val X196 = Convolv(1,0)(X195,W2,B2)
JCudaTensor x206;
JCudaTensor x207, x208, x209;
x207 = x204;
x208 = x26;
x209 = x27;
x206 = x28.forward(x207, x208, x209);

// Dealloc(X195)
JCudaTensor x210;
x210 = x204;
x210.free();

// val X197 = Pooling(2,2,0,true)(X196)
JCudaTensor x211;
JCudaTensor x212;
x212 = x206;
x211 = x31.forward(x212);

// Dealloc(X196)
JCudaTensor x213;
x213 = x206;
x213.free();

// val X198 = Tanh()(X197)
JCudaTensor x214;
JCudaTensor x215;
x215 = x211;
x214 = x34.forward(x215);

// val X199 = (X198[1><3])(i10 | @) * (W)(i11 | @)
JCudaTensor x216;
JCudaMatrix x217;
JCudaMatrix x218;
JCudaTensor x219;
JCudaTensor x220;
x220 = x214;
x219 = x220.flatten(1, new int[]{50, 4, 4});
x217 = x219.asMatrix(1, true);
JCudaTensor x221;
x221 = x41;
x218 = x221.asMatrix(1, true);
x216 = x217.times(x218);

// Dealloc(X198)
JCudaTensor x222;
x222 = x214;
x222.free();

// val X201 = (X199 + (i10) => B)
JCudaTensor x223;
JCudaTensor x224, x225;
x224 = x216;
x225 = x45;
x223 = x225.copy(500, x224);

// val X202 = Tanh()(X201)
JCudaTensor x226;
JCudaTensor x227;
x227 = x223;
x226 = x48.forward(x227);

// val X203 = (X202)(i13 | @) * (Theta)(i14 | @)
JCudaTensor x228;
JCudaMatrix x229;
JCudaMatrix x230;
JCudaTensor x231;
x231 = x226;
x229 = x231.asMatrix(1, true);
JCudaTensor x232;
x232 = x54;
x230 = x232.asMatrix(1, true);
x228 = x229.times(x230);

// Dealloc(X202)
JCudaTensor x233;
x233 = x226;
x233.free();

// val X205 = (X203 + (i13) => B3)
JCudaTensor x234;
JCudaTensor x235, x236;
x235 = x228;
x236 = x58;
x234 = x236.copy(500, x235);

// Precision(Accuracy(X205, Y, 1))
float x238;
JCudaTensor x239;
JTensorFloat x240;
x239 = x234;
x240 = x4;
x238 = x239.accuracy(x240, 1);
System.out.println(x5 + " test precision "  + x238);
x237 += x238;

// Dealloc(X205)
JCudaTensor x241;
x241 = x234;
x241.free();

}
System.out.println();
System.out.println("average precision: " + x237/test_itr);
System.out.println(); 
}

}