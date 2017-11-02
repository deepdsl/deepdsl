package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class Vgg extends CudaRun {

public static void main(String[] args){
Vgg run = new Vgg();
run.train(1000);
run.test(10);
run.save();
run.free();
}

public Vgg() {
super("src/main/java/deepdsl/gen/vgg");
setTrainData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_train_lmdb", 1000000, new int[]{64, 3, 224, 224}, 1000, false));
setTestData(LmdbFactory.getFactory("dataset/imagenet/ilsvrc12_val_lmdb", 10000, new int[]{64, 3, 224, 224}, 1000, true));
}

float lrn_rate = -0.01f;
float momentum = 0.9f;
float decay = 5.0E-4f;

JCudnnConvolution y18 = addConvolution(new int[]{64,128,112,112},new int[]{128,128,3,3},new int[]{128}, 1, 1);
JCudnnConvolution y15 = addConvolution(new int[]{64,128,56,56},new int[]{256,128,3,3},new int[]{256}, 1, 1);
JCudnnConvolution y11 = addConvolution(new int[]{64,256,28,28},new int[]{512,256,3,3},new int[]{512}, 1, 1);
JCudnnConvolution y14 = addConvolution(new int[]{64,256,56,56},new int[]{256,256,3,3},new int[]{256}, 1, 1);
JCudnnConvolution y23 = addConvolution(new int[]{64,3,224,224},new int[]{64,3,3,3},new int[]{64}, 1, 1);
JCudnnConvolution y7 = addConvolution(new int[]{64,512,14,14},new int[]{512,512,3,3},new int[]{512}, 1, 1);
JCudnnConvolution y10 = addConvolution(new int[]{64,512,28,28},new int[]{512,512,3,3},new int[]{512}, 1, 1);
JCudnnConvolution y19 = addConvolution(new int[]{64,64,112,112},new int[]{128,64,3,3},new int[]{128}, 1, 1);
JCudnnConvolution y22 = addConvolution(new int[]{64,64,224,224},new int[]{64,64,3,3},new int[]{64}, 1, 1);
JCudnnSoftmax y1 = addSoftmax(new int[]{64,1000}, SoftmaxAlgorithm.LOG);
JCudnnPooling y16 = addPooling(new int[]{64,128,112,112}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y12 = addPooling(new int[]{64,256,56,56}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y5 = addPooling(new int[]{64,512,14,14}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y8 = addPooling(new int[]{64,512,28,28}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y20 = addPooling(new int[]{64,64,224,224}, 2, 2, 0, PoolingType.MAX);
JCudnnActivation y17 = addActivation(new int[]{64,128,112,112}, ActivationMode.RELU);
JCudnnActivation y13 = addActivation(new int[]{64,256,56,56}, ActivationMode.RELU);
JCudnnActivation y3 = addActivation(new int[]{64,4096}, ActivationMode.RELU);
JCudnnActivation y6 = addActivation(new int[]{64,512,14,14}, ActivationMode.RELU);
JCudnnActivation y9 = addActivation(new int[]{64,512,28,28}, ActivationMode.RELU);
JCudnnActivation y21 = addActivation(new int[]{64,64,224,224}, ActivationMode.RELU);
JCudnnDropout y4 = addDropout("y4", new int[]{64,4096}, 0.5f);
JCudnnDropout y2 = addDropout("y2", new int[]{64,4096}, 0.5f);
JCudaTensor V_cv11_B = addParam("V_cv11_B", "Constant", 0f, 64);
JCudaTensor V_cv11_W = addParam("V_cv11_W", "Constant", 0f, 64, 3, 3, 3);
JCudaTensor V_cv12_B = addParam("V_cv12_B", "Constant", 0f, 64);
JCudaTensor V_cv12_W = addParam("V_cv12_W", "Constant", 0f, 64, 64, 3, 3);
JCudaTensor V_cv21_B = addParam("V_cv21_B", "Constant", 0f, 128);
JCudaTensor V_cv21_W = addParam("V_cv21_W", "Constant", 0f, 128, 64, 3, 3);
JCudaTensor V_cv22_B = addParam("V_cv22_B", "Constant", 0f, 128);
JCudaTensor V_cv22_W = addParam("V_cv22_W", "Constant", 0f, 128, 128, 3, 3);
JCudaTensor V_cv31_B = addParam("V_cv31_B", "Constant", 0f, 256);
JCudaTensor V_cv31_W = addParam("V_cv31_W", "Constant", 0f, 256, 128, 3, 3);
JCudaTensor V_cv32_B = addParam("V_cv32_B", "Constant", 0f, 256);
JCudaTensor V_cv32_W = addParam("V_cv32_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_cv33_B = addParam("V_cv33_B", "Constant", 0f, 256);
JCudaTensor V_cv33_W = addParam("V_cv33_W", "Constant", 0f, 256, 256, 3, 3);
JCudaTensor V_cv41_B = addParam("V_cv41_B", "Constant", 0f, 512);
JCudaTensor V_cv41_W = addParam("V_cv41_W", "Constant", 0f, 512, 256, 3, 3);
JCudaTensor V_cv42_B = addParam("V_cv42_B", "Constant", 0f, 512);
JCudaTensor V_cv42_W = addParam("V_cv42_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_cv43_B = addParam("V_cv43_B", "Constant", 0f, 512);
JCudaTensor V_cv43_W = addParam("V_cv43_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_cv51_B = addParam("V_cv51_B", "Constant", 0f, 512);
JCudaTensor V_cv51_W = addParam("V_cv51_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_cv52_B = addParam("V_cv52_B", "Constant", 0f, 512);
JCudaTensor V_cv52_W = addParam("V_cv52_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_cv53_B = addParam("V_cv53_B", "Constant", 0f, 512);
JCudaTensor V_cv53_W = addParam("V_cv53_W", "Constant", 0f, 512, 512, 3, 3);
JCudaTensor V_fc6_B = addParam("V_fc6_B", "Constant", 0f, 4096);
JCudaTensor V_fc6_W = addParam("V_fc6_W", "Constant", 0f, 4096, 25088);
JCudaTensor V_fc7_B = addParam("V_fc7_B", "Constant", 0f, 4096);
JCudaTensor V_fc7_W = addParam("V_fc7_W", "Constant", 0f, 4096, 4096);
JCudaTensor V_fc8_B = addParam("V_fc8_B", "Constant", 0f, 1000);
JCudaTensor V_fc8_W = addParam("V_fc8_W", "Constant", 0f, 1000, 4096);
JCudaTensor cv11_B = addParam("cv11_B", "Constant", 0.0f, 64);
JCudaTensor cv11_W = addParam("cv11_W", "Random", 0.27216554f, 64, 3, 3, 3);
JCudaTensor cv12_B = addParam("cv12_B", "Constant", 0.0f, 64);
JCudaTensor cv12_W = addParam("cv12_W", "Random", 0.058925565f, 64, 64, 3, 3);
JCudaTensor cv21_B = addParam("cv21_B", "Constant", 0.0f, 128);
JCudaTensor cv21_W = addParam("cv21_W", "Random", 0.058925565f, 128, 64, 3, 3);
JCudaTensor cv22_B = addParam("cv22_B", "Constant", 0.0f, 128);
JCudaTensor cv22_W = addParam("cv22_W", "Random", 0.041666668f, 128, 128, 3, 3);
JCudaTensor cv31_B = addParam("cv31_B", "Constant", 0.0f, 256);
JCudaTensor cv31_W = addParam("cv31_W", "Random", 0.041666668f, 256, 128, 3, 3);
JCudaTensor cv32_B = addParam("cv32_B", "Constant", 0.0f, 256);
JCudaTensor cv32_W = addParam("cv32_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor cv33_B = addParam("cv33_B", "Constant", 0.0f, 256);
JCudaTensor cv33_W = addParam("cv33_W", "Random", 0.029462783f, 256, 256, 3, 3);
JCudaTensor cv41_B = addParam("cv41_B", "Constant", 0.0f, 512);
JCudaTensor cv41_W = addParam("cv41_W", "Random", 0.029462783f, 512, 256, 3, 3);
JCudaTensor cv42_B = addParam("cv42_B", "Constant", 0.0f, 512);
JCudaTensor cv42_W = addParam("cv42_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor cv43_B = addParam("cv43_B", "Constant", 0.0f, 512);
JCudaTensor cv43_W = addParam("cv43_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor cv51_B = addParam("cv51_B", "Constant", 0.0f, 512);
JCudaTensor cv51_W = addParam("cv51_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor cv52_B = addParam("cv52_B", "Constant", 0.0f, 512);
JCudaTensor cv52_W = addParam("cv52_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor cv53_B = addParam("cv53_B", "Constant", 0.0f, 512);
JCudaTensor cv53_W = addParam("cv53_W", "Random", 0.020833334f, 512, 512, 3, 3);
JCudaTensor fc6_B = addParam("fc6_B", "Constant", 0.0f, 4096);
JCudaTensor fc6_W = addParam("fc6_W", "Random", 0.008928572f, 4096, 25088);
JCudaTensor fc7_B = addParam("fc7_B", "Constant", 0.0f, 4096);
JCudaTensor fc7_W = addParam("fc7_W", "Random", 0.022097087f, 4096, 4096);
JCudaTensor fc8_B = addParam("fc8_B", "Constant", 0.0f, 1000);
JCudaTensor fc8_W = addParam("fc8_W", "Random", 0.022097087f, 1000, 4096);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X257 = Cuda(X)
JCudaTensor X257 = X.asJCudaTensor();
// val X68 = Convolv(1,1)(X257,cv11_W,cv11_B)
JCudaTensor X68 = y23.forward(X257, cv11_W, cv11_B);
// val X69 = ReLU()(X68)
JCudaTensor X69 = y21.forward(X68);
// val X70 = Convolv(1,1)(X69,cv12_W,cv12_B)
JCudaTensor X70 = y22.forward(X69, cv12_W, cv12_B);
// val X71 = ReLU()(X70)
JCudaTensor X71 = y21.forward(X70);
// val X72 = Pooling(2,2,0,true)(X71)
JCudaTensor X72 = y20.forward(X71);
// val X73 = Convolv(1,1)(X72,cv21_W,cv21_B)
JCudaTensor X73 = y19.forward(X72, cv21_W, cv21_B);
// val X74 = ReLU()(X73)
JCudaTensor X74 = y17.forward(X73);
// val X75 = Convolv(1,1)(X74,cv22_W,cv22_B)
JCudaTensor X75 = y18.forward(X74, cv22_W, cv22_B);
// val X76 = ReLU()(X75)
JCudaTensor X76 = y17.forward(X75);
// val X77 = Pooling(2,2,0,true)(X76)
JCudaTensor X77 = y16.forward(X76);
// val X78 = Convolv(1,1)(X77,cv31_W,cv31_B)
JCudaTensor X78 = y15.forward(X77, cv31_W, cv31_B);
// val X79 = ReLU()(X78)
JCudaTensor X79 = y13.forward(X78);
// val X80 = Convolv(1,1)(X79,cv32_W,cv32_B)
JCudaTensor X80 = y14.forward(X79, cv32_W, cv32_B);
// val X81 = ReLU()(X80)
JCudaTensor X81 = y13.forward(X80);
// val X82 = Convolv(1,1)(X81,cv33_W,cv33_B)
JCudaTensor X82 = y14.forward(X81, cv33_W, cv33_B);
// val X83 = ReLU()(X82)
JCudaTensor X83 = y13.forward(X82);
// val X84 = Pooling(2,2,0,true)(X83)
JCudaTensor X84 = y12.forward(X83);
// val X85 = Convolv(1,1)(X84,cv41_W,cv41_B)
JCudaTensor X85 = y11.forward(X84, cv41_W, cv41_B);
// val X86 = ReLU()(X85)
JCudaTensor X86 = y9.forward(X85);
// val X87 = Convolv(1,1)(X86,cv42_W,cv42_B)
JCudaTensor X87 = y10.forward(X86, cv42_W, cv42_B);
// val X88 = ReLU()(X87)
JCudaTensor X88 = y9.forward(X87);
// val X89 = Convolv(1,1)(X88,cv43_W,cv43_B)
JCudaTensor X89 = y10.forward(X88, cv43_W, cv43_B);
// val X90 = ReLU()(X89)
JCudaTensor X90 = y9.forward(X89);
// val X91 = Pooling(2,2,0,true)(X90)
JCudaTensor X91 = y8.forward(X90);
// val X92 = Convolv(1,1)(X91,cv51_W,cv51_B)
JCudaTensor X92 = y7.forward(X91, cv51_W, cv51_B);
// val X93 = ReLU()(X92)
JCudaTensor X93 = y6.forward(X92);
// val X94 = Convolv(1,1)(X93,cv52_W,cv52_B)
JCudaTensor X94 = y7.forward(X93, cv52_W, cv52_B);
// val X95 = ReLU()(X94)
JCudaTensor X95 = y6.forward(X94);
// val X96 = Convolv(1,1)(X95,cv53_W,cv53_B)
JCudaTensor X96 = y7.forward(X95, cv53_W, cv53_B);
// val X97 = ReLU()(X96)
JCudaTensor X97 = y6.forward(X96);
// val X98 = Pooling(2,2,0,true)(X97)
JCudaTensor X98 = y5.forward(X97);
// val X255 = (X98[1><3])(i1 | @) * (fc6_W)(i2 | @)
JCudaTensor X255 = X98.flatten(1, new int[]{512, 7, 7}).asMatrix(1, true).times(fc6_W.asMatrix(1, true));
// val X100 = (X255 + (i1) => fc6_B)
JCudaTensor X100 = fc6_B.copy(64, X255);
// val X101 = ReLU()(X100)
JCudaTensor X101 = y3.forward(X100);
// val X102 = Dropout(0.5)(X101)
JCudaTensor X102 = y4.forward(X101);
// val X251 = (X102)(i4 | @) * (fc7_W)(i5 | @)
JCudaTensor X251 = X102.asMatrix(1, true).times(fc7_W.asMatrix(1, true));
// val X103 = (X251 + (i4) => fc7_B)
JCudaTensor X103 = fc7_B.copy(64, X251);
// val X104 = ReLU()(X103)
JCudaTensor X104 = y3.forward(X103);
// val X105 = Dropout(0.5)(X104)
JCudaTensor X105 = y2.forward(X104);
// val X258 = Cuda(Indicator(Y, 1000))
JCudaTensor X258 = Y.asIndicator(1000).asJCudaTensor();
// val X253 = (X105)(i7 | @) * (fc8_W)(i8 | @)
JCudaTensor X253 = X105.asMatrix(1, true).times(fc8_W.asMatrix(1, true));
// val X106 = (X253 + (i7) => fc8_B)
JCudaTensor X106 = fc8_B.copy(64, X253);
// val X262 = - X258.copy
JCudaTensor X262 = X258.clone().times_i(-1f);;
// val X108 = (X262 / |64|)
JCudaTensor X108 = X262.times_i(1 / 64f);;
// val X107 = LogSoftmax()(X106)
JCudaTensor X107 = y1.forward(X106);
// dealloc X106
X106.free();
// val _loss = ((0 - (X258 . X107)) / |64|)
float _loss = - X258.dot(X107) / 64f;
// dealloc X258
X258.free();
// val m2 = (i54) => fc8_W[@, i54]
JCudaMatrix m2 = fc8_W.asMatrix(1, false);
// val X310 = X108 * d_LogSoftmax()(X107)/d_X106
JCudaTensor X310 = y1.backward(X108, X107);
// dealloc X108
X108.free();
// dealloc X107
X107.free();
// val m4 = (i57) => X310[@, i57]
JCudaMatrix m4 = X310.asMatrix(1, false);
// val X260 = (X310)(i53 | @) * m2
JCudaTensor X260 = X310.asMatrix(1, true).times(m2);
// val m6 = (i22) => X105[@, i22]
JCudaMatrix m6 = X105.asMatrix(1, false);
// val X294 = X260 * d_Dropout(0.5)()/d_X104
JCudaTensor X294 = y2.backward(X260);
// dealloc X260
X260.free();
// V_fc8_B = ((Sum(m4) * -0.01) + (V_fc8_B * 0.9))
m4.sum(V_fc8_B, lrn_rate, momentum);
// V_fc8_W = ((m4 * m6 * -0.01) + (V_fc8_W * 0.9))
m4.times(m6, V_fc8_W, lrn_rate, momentum);
// dealloc X105
X105.free();
// dealloc X310
X310.free();
// fc8_B = (V_fc8_B + (fc8_B * (1 + (5.0E-4 * -0.01))))
fc8_B.update(V_fc8_B, 1f, 1f + decay * lrn_rate);
// val X340 = X294 * d_ReLU()(X104)/d_X103
JCudaTensor X340 = y3.backward(X294, X104);
// dealloc X104
X104.free();
// fc8_W = (V_fc8_W + (fc8_W * (1 + (5.0E-4 * -0.01))))
fc8_W.update(V_fc8_W, 1f, 1f + decay * lrn_rate);
// val m3 = (i47) => fc7_W[@, i47]
JCudaMatrix m3 = fc7_W.asMatrix(1, false);
// val X263 = (X340)(i46 | @) * m3
JCudaTensor X263 = X340.asMatrix(1, true).times(m3);
// val m8 = (i26) => X102[@, i26]
JCudaMatrix m8 = X102.asMatrix(1, false);
// val m1 = (i50) => X340[@, i50]
JCudaMatrix m1 = X340.asMatrix(1, false);
// V_fc7_B = ((Sum(m1) * -0.01) + (V_fc7_B * 0.9))
m1.sum(V_fc7_B, lrn_rate, momentum);
// val X351 = X263 * d_Dropout(0.5)()/d_X101
JCudaTensor X351 = y4.backward(X263);
// dealloc X263
X263.free();
// V_fc7_W = ((m1 * m8 * -0.01) + (V_fc7_W * 0.9))
m1.times(m8, V_fc7_W, lrn_rate, momentum);
// dealloc X102
X102.free();
// dealloc X340
X340.free();
// val m11 = (i40) => fc6_W[@, i40]
JCudaMatrix m11 = fc6_W.asMatrix(1, false);
// fc7_B = (V_fc7_B + (fc7_B * (1 + (5.0E-4 * -0.01))))
fc7_B.update(V_fc7_B, 1f, 1f + decay * lrn_rate);
// fc7_W = (V_fc7_W + (fc7_W * (1 + (5.0E-4 * -0.01))))
fc7_W.update(V_fc7_W, 1f, 1f + decay * lrn_rate);
// val X307 = X351 * d_ReLU()(X101)/d_X100
JCudaTensor X307 = y3.backward(X351, X101);
// dealloc X101
X101.free();
// val m10 = (i30) => X98[1><3][@, i30]
JCudaMatrix m10 = X98.flatten(1, new int[]{512, 7, 7}).asMatrix(1, false);
// val m9 = (i29) => X307[@, i29]
JCudaMatrix m9 = X307.asMatrix(1, false);
// val X349 = (X307)(i39 | @) * m11
JCudaTensor X349 = X307.asMatrix(1, true).times(m11);
// V_fc6_B = ((Sum(m9) * -0.01) + (V_fc6_B * 0.9))
m9.sum(V_fc6_B, lrn_rate, momentum);
// val X326 = X349[1<>3] * d_Pooling(2,2,0,true)(X98,X97)/d_X97
JCudaTensor X326 = y5.backward(X349.unflatten(1, new int[]{512, 7, 7}), X98, X97);
// dealloc X349
X349.free();
// V_fc6_W = ((m9 * m10 * -0.01) + (V_fc6_W * 0.9))
m9.times(m10, V_fc6_W, lrn_rate, momentum);
// dealloc X98
X98.free();
// dealloc X307
X307.free();
// fc6_W = (V_fc6_W + (fc6_W * (1 + (5.0E-4 * -0.01))))
fc6_W.update(V_fc6_W, 1f, 1f + decay * lrn_rate);
// val X305 = X326 * d_ReLU()(X97)/d_X96
JCudaTensor X305 = y6.backward(X326, X97);
// dealloc X97
X97.free();
// fc6_B = (V_fc6_B + (fc6_B * (1 + (5.0E-4 * -0.01))))
fc6_B.update(V_fc6_B, 1f, 1f + decay * lrn_rate);
// val X302 = X305 * d_Convolv(1,1)(cv53_W)/d_X95
JCudaTensor X302 = y7.backward_data(X305, cv53_W);
// V_cv53_B = ((X305 * d_Convolv(1,1)()/d_cv53_B * -0.01) + (V_cv53_B * 0.9))
y7.backward_bias(X305, V_cv53_B, lrn_rate, momentum);
// V_cv53_W = ((X305 * d_Convolv(1,1)(X95)/d_cv53_W * -0.01) + (V_cv53_W * 0.9))
y7.backward_filter(X305, X95, V_cv53_W, lrn_rate, momentum);
// dealloc X305
X305.free();
// val X336 = X302 * d_ReLU()(X95)/d_X94
JCudaTensor X336 = y6.backward(X302, X95);
// dealloc X95
X95.free();
// cv53_B = (V_cv53_B + (cv53_B * (1 + (5.0E-4 * -0.01))))
cv53_B.update(V_cv53_B, 1f, 1f + decay * lrn_rate);
// cv53_W = (V_cv53_W + (cv53_W * (1 + (5.0E-4 * -0.01))))
cv53_W.update(V_cv53_W, 1f, 1f + decay * lrn_rate);
// V_cv52_B = ((X336 * d_Convolv(1,1)()/d_cv52_B * -0.01) + (V_cv52_B * 0.9))
y7.backward_bias(X336, V_cv52_B, lrn_rate, momentum);
// val X275 = X336 * d_Convolv(1,1)(cv52_W)/d_X93
JCudaTensor X275 = y7.backward_data(X336, cv52_W);
// V_cv52_W = ((X336 * d_Convolv(1,1)(X93)/d_cv52_W * -0.01) + (V_cv52_W * 0.9))
y7.backward_filter(X336, X93, V_cv52_W, lrn_rate, momentum);
// dealloc X336
X336.free();
// cv52_W = (V_cv52_W + (cv52_W * (1 + (5.0E-4 * -0.01))))
cv52_W.update(V_cv52_W, 1f, 1f + decay * lrn_rate);
// cv52_B = (V_cv52_B + (cv52_B * (1 + (5.0E-4 * -0.01))))
cv52_B.update(V_cv52_B, 1f, 1f + decay * lrn_rate);
// val X271 = X275 * d_ReLU()(X93)/d_X92
JCudaTensor X271 = y6.backward(X275, X93);
// dealloc X93
X93.free();
// val X342 = X271 * d_Convolv(1,1)(cv51_W)/d_X91
JCudaTensor X342 = y7.backward_data(X271, cv51_W);
// V_cv51_B = ((X271 * d_Convolv(1,1)()/d_cv51_B * -0.01) + (V_cv51_B * 0.9))
y7.backward_bias(X271, V_cv51_B, lrn_rate, momentum);
// V_cv51_W = ((X271 * d_Convolv(1,1)(X91)/d_cv51_W * -0.01) + (V_cv51_W * 0.9))
y7.backward_filter(X271, X91, V_cv51_W, lrn_rate, momentum);
// dealloc X271
X271.free();
// cv51_B = (V_cv51_B + (cv51_B * (1 + (5.0E-4 * -0.01))))
cv51_B.update(V_cv51_B, 1f, 1f + decay * lrn_rate);
// val X291 = X342 * d_Pooling(2,2,0,true)(X91,X90)/d_X90
JCudaTensor X291 = y8.backward(X342, X91, X90);
// dealloc X91
X91.free();
// dealloc X342
X342.free();
// cv51_W = (V_cv51_W + (cv51_W * (1 + (5.0E-4 * -0.01))))
cv51_W.update(V_cv51_W, 1f, 1f + decay * lrn_rate);
// val X278 = X291 * d_ReLU()(X90)/d_X89
JCudaTensor X278 = y9.backward(X291, X90);
// dealloc X90
X90.free();
// V_cv43_W = ((X278 * d_Convolv(1,1)(X88)/d_cv43_W * -0.01) + (V_cv43_W * 0.9))
y10.backward_filter(X278, X88, V_cv43_W, lrn_rate, momentum);
// val X292 = X278 * d_Convolv(1,1)(cv43_W)/d_X88
JCudaTensor X292 = y10.backward_data(X278, cv43_W);
// V_cv43_B = ((X278 * d_Convolv(1,1)()/d_cv43_B * -0.01) + (V_cv43_B * 0.9))
y10.backward_bias(X278, V_cv43_B, lrn_rate, momentum);
// dealloc X278
X278.free();
// cv43_W = (V_cv43_W + (cv43_W * (1 + (5.0E-4 * -0.01))))
cv43_W.update(V_cv43_W, 1f, 1f + decay * lrn_rate);
// val X282 = X292 * d_ReLU()(X88)/d_X87
JCudaTensor X282 = y9.backward(X292, X88);
// dealloc X88
X88.free();
// cv43_B = (V_cv43_B + (cv43_B * (1 + (5.0E-4 * -0.01))))
cv43_B.update(V_cv43_B, 1f, 1f + decay * lrn_rate);
// V_cv42_B = ((X282 * d_Convolv(1,1)()/d_cv42_B * -0.01) + (V_cv42_B * 0.9))
y10.backward_bias(X282, V_cv42_B, lrn_rate, momentum);
// val X344 = X282 * d_Convolv(1,1)(cv42_W)/d_X86
JCudaTensor X344 = y10.backward_data(X282, cv42_W);
// V_cv42_W = ((X282 * d_Convolv(1,1)(X86)/d_cv42_W * -0.01) + (V_cv42_W * 0.9))
y10.backward_filter(X282, X86, V_cv42_W, lrn_rate, momentum);
// dealloc X282
X282.free();
// cv42_W = (V_cv42_W + (cv42_W * (1 + (5.0E-4 * -0.01))))
cv42_W.update(V_cv42_W, 1f, 1f + decay * lrn_rate);
// cv42_B = (V_cv42_B + (cv42_B * (1 + (5.0E-4 * -0.01))))
cv42_B.update(V_cv42_B, 1f, 1f + decay * lrn_rate);
// val X274 = X344 * d_ReLU()(X86)/d_X85
JCudaTensor X274 = y9.backward(X344, X86);
// dealloc X86
X86.free();
// val X329 = X274 * d_Convolv(1,1)(cv41_W)/d_X84
JCudaTensor X329 = y11.backward_data(X274, cv41_W);
// V_cv41_W = ((X274 * d_Convolv(1,1)(X84)/d_cv41_W * -0.01) + (V_cv41_W * 0.9))
y11.backward_filter(X274, X84, V_cv41_W, lrn_rate, momentum);
// V_cv41_B = ((X274 * d_Convolv(1,1)()/d_cv41_B * -0.01) + (V_cv41_B * 0.9))
y11.backward_bias(X274, V_cv41_B, lrn_rate, momentum);
// dealloc X274
X274.free();
// cv41_W = (V_cv41_W + (cv41_W * (1 + (5.0E-4 * -0.01))))
cv41_W.update(V_cv41_W, 1f, 1f + decay * lrn_rate);
// val X286 = X329 * d_Pooling(2,2,0,true)(X84,X83)/d_X83
JCudaTensor X286 = y12.backward(X329, X84, X83);
// dealloc X84
X84.free();
// dealloc X329
X329.free();
// cv41_B = (V_cv41_B + (cv41_B * (1 + (5.0E-4 * -0.01))))
cv41_B.update(V_cv41_B, 1f, 1f + decay * lrn_rate);
// val X301 = X286 * d_ReLU()(X83)/d_X82
JCudaTensor X301 = y13.backward(X286, X83);
// dealloc X83
X83.free();
// val X352 = X301 * d_Convolv(1,1)(cv33_W)/d_X81
JCudaTensor X352 = y14.backward_data(X301, cv33_W);
// V_cv33_B = ((X301 * d_Convolv(1,1)()/d_cv33_B * -0.01) + (V_cv33_B * 0.9))
y14.backward_bias(X301, V_cv33_B, lrn_rate, momentum);
// V_cv33_W = ((X301 * d_Convolv(1,1)(X81)/d_cv33_W * -0.01) + (V_cv33_W * 0.9))
y14.backward_filter(X301, X81, V_cv33_W, lrn_rate, momentum);
// dealloc X301
X301.free();
// cv33_B = (V_cv33_B + (cv33_B * (1 + (5.0E-4 * -0.01))))
cv33_B.update(V_cv33_B, 1f, 1f + decay * lrn_rate);
// cv33_W = (V_cv33_W + (cv33_W * (1 + (5.0E-4 * -0.01))))
cv33_W.update(V_cv33_W, 1f, 1f + decay * lrn_rate);
// val X296 = X352 * d_ReLU()(X81)/d_X80
JCudaTensor X296 = y13.backward(X352, X81);
// dealloc X81
X81.free();
// V_cv32_B = ((X296 * d_Convolv(1,1)()/d_cv32_B * -0.01) + (V_cv32_B * 0.9))
y14.backward_bias(X296, V_cv32_B, lrn_rate, momentum);
// V_cv32_W = ((X296 * d_Convolv(1,1)(X79)/d_cv32_W * -0.01) + (V_cv32_W * 0.9))
y14.backward_filter(X296, X79, V_cv32_W, lrn_rate, momentum);
// val X327 = X296 * d_Convolv(1,1)(cv32_W)/d_X79
JCudaTensor X327 = y14.backward_data(X296, cv32_W);
// dealloc X296
X296.free();
// val X323 = X327 * d_ReLU()(X79)/d_X78
JCudaTensor X323 = y13.backward(X327, X79);
// dealloc X79
X79.free();
// cv32_W = (V_cv32_W + (cv32_W * (1 + (5.0E-4 * -0.01))))
cv32_W.update(V_cv32_W, 1f, 1f + decay * lrn_rate);
// cv32_B = (V_cv32_B + (cv32_B * (1 + (5.0E-4 * -0.01))))
cv32_B.update(V_cv32_B, 1f, 1f + decay * lrn_rate);
// V_cv31_B = ((X323 * d_Convolv(1,1)()/d_cv31_B * -0.01) + (V_cv31_B * 0.9))
y15.backward_bias(X323, V_cv31_B, lrn_rate, momentum);
// V_cv31_W = ((X323 * d_Convolv(1,1)(X77)/d_cv31_W * -0.01) + (V_cv31_W * 0.9))
y15.backward_filter(X323, X77, V_cv31_W, lrn_rate, momentum);
// val X280 = X323 * d_Convolv(1,1)(cv31_W)/d_X77
JCudaTensor X280 = y15.backward_data(X323, cv31_W);
// dealloc X323
X323.free();
// cv31_W = (V_cv31_W + (cv31_W * (1 + (5.0E-4 * -0.01))))
cv31_W.update(V_cv31_W, 1f, 1f + decay * lrn_rate);
// cv31_B = (V_cv31_B + (cv31_B * (1 + (5.0E-4 * -0.01))))
cv31_B.update(V_cv31_B, 1f, 1f + decay * lrn_rate);
// val X318 = X280 * d_Pooling(2,2,0,true)(X77,X76)/d_X76
JCudaTensor X318 = y16.backward(X280, X77, X76);
// dealloc X280
X280.free();
// dealloc X77
X77.free();
// val X269 = X318 * d_ReLU()(X76)/d_X75
JCudaTensor X269 = y17.backward(X318, X76);
// dealloc X76
X76.free();
// val X338 = X269 * d_Convolv(1,1)(cv22_W)/d_X74
JCudaTensor X338 = y18.backward_data(X269, cv22_W);
// V_cv22_W = ((X269 * d_Convolv(1,1)(X74)/d_cv22_W * -0.01) + (V_cv22_W * 0.9))
y18.backward_filter(X269, X74, V_cv22_W, lrn_rate, momentum);
// V_cv22_B = ((X269 * d_Convolv(1,1)()/d_cv22_B * -0.01) + (V_cv22_B * 0.9))
y18.backward_bias(X269, V_cv22_B, lrn_rate, momentum);
// dealloc X269
X269.free();
// cv22_B = (V_cv22_B + (cv22_B * (1 + (5.0E-4 * -0.01))))
cv22_B.update(V_cv22_B, 1f, 1f + decay * lrn_rate);
// cv22_W = (V_cv22_W + (cv22_W * (1 + (5.0E-4 * -0.01))))
cv22_W.update(V_cv22_W, 1f, 1f + decay * lrn_rate);
// val X315 = X338 * d_ReLU()(X74)/d_X73
JCudaTensor X315 = y17.backward(X338, X74);
// dealloc X74
X74.free();
// val X316 = X315 * d_Convolv(1,1)(cv21_W)/d_X72
JCudaTensor X316 = y19.backward_data(X315, cv21_W);
// V_cv21_B = ((X315 * d_Convolv(1,1)()/d_cv21_B * -0.01) + (V_cv21_B * 0.9))
y19.backward_bias(X315, V_cv21_B, lrn_rate, momentum);
// V_cv21_W = ((X315 * d_Convolv(1,1)(X72)/d_cv21_W * -0.01) + (V_cv21_W * 0.9))
y19.backward_filter(X315, X72, V_cv21_W, lrn_rate, momentum);
// dealloc X315
X315.free();
// cv21_B = (V_cv21_B + (cv21_B * (1 + (5.0E-4 * -0.01))))
cv21_B.update(V_cv21_B, 1f, 1f + decay * lrn_rate);
// val X284 = X316 * d_Pooling(2,2,0,true)(X72,X71)/d_X71
JCudaTensor X284 = y20.backward(X316, X72, X71);
// dealloc X316
X316.free();
// dealloc X72
X72.free();
// cv21_W = (V_cv21_W + (cv21_W * (1 + (5.0E-4 * -0.01))))
cv21_W.update(V_cv21_W, 1f, 1f + decay * lrn_rate);
// val X266 = X284 * d_ReLU()(X71)/d_X70
JCudaTensor X266 = y21.backward(X284, X71);
// dealloc X71
X71.free();
// V_cv12_W = ((X266 * d_Convolv(1,1)(X69)/d_cv12_W * -0.01) + (V_cv12_W * 0.9))
y22.backward_filter(X266, X69, V_cv12_W, lrn_rate, momentum);
// V_cv12_B = ((X266 * d_Convolv(1,1)()/d_cv12_B * -0.01) + (V_cv12_B * 0.9))
y22.backward_bias(X266, V_cv12_B, lrn_rate, momentum);
// val X319 = X266 * d_Convolv(1,1)(cv12_W)/d_X69
JCudaTensor X319 = y22.backward_data(X266, cv12_W);
// dealloc X266
X266.free();
// cv12_B = (V_cv12_B + (cv12_B * (1 + (5.0E-4 * -0.01))))
cv12_B.update(V_cv12_B, 1f, 1f + decay * lrn_rate);
// val X347 = X319 * d_ReLU()(X69)/d_X68
JCudaTensor X347 = y21.backward(X319, X69);
// dealloc X69
X69.free();
// cv12_W = (V_cv12_W + (cv12_W * (1 + (5.0E-4 * -0.01))))
cv12_W.update(V_cv12_W, 1f, 1f + decay * lrn_rate);
// V_cv11_W = ((X347 * d_Convolv(1,1)(X257)/d_cv11_W * -0.01) + (V_cv11_W * 0.9))
y23.backward_filter(X347, X257, V_cv11_W, lrn_rate, momentum);
// dealloc X257
X257.free();
// V_cv11_B = ((X347 * d_Convolv(1,1)()/d_cv11_B * -0.01) + (V_cv11_B * 0.9))
y23.backward_bias(X347, V_cv11_B, lrn_rate, momentum);
// dealloc X347
X347.free();
// cv11_B = (V_cv11_B + (cv11_B * (1 + (5.0E-4 * -0.01))))
cv11_B.update(V_cv11_B, 1f, 1f + decay * lrn_rate);
// cv11_W = (V_cv11_W + (cv11_W * (1 + (5.0E-4 * -0.01))))
cv11_W.update(V_cv11_W, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X24 = Cuda(X)
JCudaTensor X24 = X.asJCudaTensor();
// val X25 = Convolv(1,1)(X24,cv11_W,cv11_B)
JCudaTensor X25 = y23.forward(X24, cv11_W, cv11_B);
// dealloc X24
X24.free();
// val X26 = ReLU()(X25)
JCudaTensor X26 = y21.forward(X25);
// val X27 = Convolv(1,1)(X26,cv12_W,cv12_B)
JCudaTensor X27 = y22.forward(X26, cv12_W, cv12_B);
// dealloc X26
X26.free();
// val X28 = ReLU()(X27)
JCudaTensor X28 = y21.forward(X27);
// val X29 = Pooling(2,2,0,true)(X28)
JCudaTensor X29 = y20.forward(X28);
// dealloc X28
X28.free();
// val X30 = Convolv(1,1)(X29,cv21_W,cv21_B)
JCudaTensor X30 = y19.forward(X29, cv21_W, cv21_B);
// dealloc X29
X29.free();
// val X31 = ReLU()(X30)
JCudaTensor X31 = y17.forward(X30);
// val X32 = Convolv(1,1)(X31,cv22_W,cv22_B)
JCudaTensor X32 = y18.forward(X31, cv22_W, cv22_B);
// dealloc X31
X31.free();
// val X33 = ReLU()(X32)
JCudaTensor X33 = y17.forward(X32);
// val X34 = Pooling(2,2,0,true)(X33)
JCudaTensor X34 = y16.forward(X33);
// dealloc X33
X33.free();
// val X35 = Convolv(1,1)(X34,cv31_W,cv31_B)
JCudaTensor X35 = y15.forward(X34, cv31_W, cv31_B);
// dealloc X34
X34.free();
// val X36 = ReLU()(X35)
JCudaTensor X36 = y13.forward(X35);
// val X37 = Convolv(1,1)(X36,cv32_W,cv32_B)
JCudaTensor X37 = y14.forward(X36, cv32_W, cv32_B);
// dealloc X36
X36.free();
// val X38 = ReLU()(X37)
JCudaTensor X38 = y13.forward(X37);
// val X39 = Convolv(1,1)(X38,cv33_W,cv33_B)
JCudaTensor X39 = y14.forward(X38, cv33_W, cv33_B);
// dealloc X38
X38.free();
// val X40 = ReLU()(X39)
JCudaTensor X40 = y13.forward(X39);
// val X41 = Pooling(2,2,0,true)(X40)
JCudaTensor X41 = y12.forward(X40);
// dealloc X40
X40.free();
// val X42 = Convolv(1,1)(X41,cv41_W,cv41_B)
JCudaTensor X42 = y11.forward(X41, cv41_W, cv41_B);
// dealloc X41
X41.free();
// val X43 = ReLU()(X42)
JCudaTensor X43 = y9.forward(X42);
// val X44 = Convolv(1,1)(X43,cv42_W,cv42_B)
JCudaTensor X44 = y10.forward(X43, cv42_W, cv42_B);
// dealloc X43
X43.free();
// val X45 = ReLU()(X44)
JCudaTensor X45 = y9.forward(X44);
// val X46 = Convolv(1,1)(X45,cv43_W,cv43_B)
JCudaTensor X46 = y10.forward(X45, cv43_W, cv43_B);
// dealloc X45
X45.free();
// val X47 = ReLU()(X46)
JCudaTensor X47 = y9.forward(X46);
// val X48 = Pooling(2,2,0,true)(X47)
JCudaTensor X48 = y8.forward(X47);
// dealloc X47
X47.free();
// val X49 = Convolv(1,1)(X48,cv51_W,cv51_B)
JCudaTensor X49 = y7.forward(X48, cv51_W, cv51_B);
// dealloc X48
X48.free();
// val X50 = ReLU()(X49)
JCudaTensor X50 = y6.forward(X49);
// val X51 = Convolv(1,1)(X50,cv52_W,cv52_B)
JCudaTensor X51 = y7.forward(X50, cv52_W, cv52_B);
// dealloc X50
X50.free();
// val X52 = ReLU()(X51)
JCudaTensor X52 = y6.forward(X51);
// val X53 = Convolv(1,1)(X52,cv53_W,cv53_B)
JCudaTensor X53 = y7.forward(X52, cv53_W, cv53_B);
// dealloc X52
X52.free();
// val X54 = ReLU()(X53)
JCudaTensor X54 = y6.forward(X53);
// val X55 = Pooling(2,2,0,true)(X54)
JCudaTensor X55 = y5.forward(X54);
// dealloc X54
X54.free();
// val X62 = (X55[1><3])(i1 | @) * (fc6_W)(i2 | @)
JCudaTensor X62 = X55.flatten(1, new int[]{512, 7, 7}).asMatrix(1, true).times(fc6_W.asMatrix(1, true));
// dealloc X55
X55.free();
// val X57 = (X62 + (i1) => fc6_B)
JCudaTensor X57 = fc6_B.copy(64, X62);
// val X58 = ReLU()(X57)
JCudaTensor X58 = y3.forward(X57);
// val X64 = (X58)(i4 | @) * (fc7_W)(i5 | @)
JCudaTensor X64 = X58.asMatrix(1, true).times(fc7_W.asMatrix(1, true));
// dealloc X58
X58.free();
// val X59 = (X64 + (i4) => fc7_B)
JCudaTensor X59 = fc7_B.copy(64, X64);
// val X60 = ReLU()(X59)
JCudaTensor X60 = y3.forward(X59);
// val X66 = (X60)(i7 | @) * (fc8_W)(i8 | @)
JCudaTensor X66 = X60.asMatrix(1, true).times(fc8_W.asMatrix(1, true));
// dealloc X60
X60.free();
// val X61 = (X66 + (i7) => fc8_B)
JCudaTensor X61 = fc8_B.copy(64, X66);

return X61; 
}

}