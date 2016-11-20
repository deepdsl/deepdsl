package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;

// This file is for inference only, which needs trained parameters.
public class Lenet_infer {
	// comment the line below for memory efficient mode
	static{ JCudaTensor.enableMemoryCache();}
	// network_dir
	static String network_dir = "src/main/java/deepdsl/gen/lenet";
	// test_itr
	static int test_itr = 10;

	// (Convolv(1,0),List(List(500, 1, 28, 28), List(20, 1, 5, 5), List(20)))
	static JCudnnConvolution x13 = new JCudnnConvolution(new int[]{500,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
	// (Convolv(1,0),List(List(500, 20, 12, 12), List(50, 20, 5, 5), List(50)))
	static JCudnnConvolution x25 = new JCudnnConvolution(new int[]{500,20,12,12},new int[]{50,20,5,5},new int[]{50}, 1, 0);
	// (MNIST,false)
	static MnistFactory x1 = MnistFactory.getFactory(false, new int[]{500, 1, 28, 28});
	// (Pooling(2,2,0,true),List(List(500, 20, 24, 24)))
	static JCudnnPooling x17 = new JCudnnPooling(new int[]{500,20,24,24}, 2, 2, 0, PoolingType.MAX);
	// (Pooling(2,2,0,true),List(List(500, 50, 8, 8)))
	static JCudnnPooling x29 = new JCudnnPooling(new int[]{500,50,8,8}, 2, 2, 0, PoolingType.MAX);
	// (ReLU(),List(List(500, 500)))
	static JCudnnActivation x45 = new JCudnnActivation(new int[]{500,500}, ActivationMode.RELU);
	// X
	static JTensorFloat x2;
	// cv1_B
	static JCudaTensor x12 = JTensor.constFloat(0.0f, 20).load(network_dir + "/cv1_B").asJCudaTensor();
	// cv1_W
	static JCudaTensor x11 = JTensor.randomFloat(-0.28284273f, 0.28284273f, 20, 1, 5, 5).load(network_dir + "/cv1_W").asJCudaTensor();
	// cv2_B
	static JCudaTensor x24 = JTensor.constFloat(0.0f, 50).load(network_dir + "/cv2_B").asJCudaTensor();
	// cv2_W
	static JCudaTensor x23 = JTensor.randomFloat(-0.06324555f, 0.06324555f, 50, 20, 5, 5).load(network_dir + "/cv2_W").asJCudaTensor();
	// fc1_B
	static JCudaTensor x42 = JTensor.constFloat(0.0f, 500).load(network_dir + "/fc1_B").asJCudaTensor();
	// fc1_W
	static JCudaTensor x37 = JTensor.randomFloat(-0.05f, 0.05f, 500, 800).load(network_dir + "/fc1_W").asJCudaTensor();
	// fc2_B
	static JCudaTensor x56 = JTensor.constFloat(0.0f, 10).load(network_dir + "/fc2_B").asJCudaTensor();
	// fc2_W
	static JCudaTensor x51 = JTensor.randomFloat(-0.06324555f, 0.06324555f, 10, 500).load(network_dir + "/fc2_W").asJCudaTensor();

	public static void main(String[] args){
		test();
		x24.free();
		x51.free();
		x23.free();
		x12.free();
		x37.free();
		x42.free();
		x11.free();
		x56.free();
		x29.free();
		x25.free();
		x45.free();
		x17.free();
		x13.free();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}
	static void test() {
		for(int x3=0; x3<test_itr; x3++) {
			JTensorFloatTuple x4 =  x1.nextFloat();
			x2 = x4.image;

			// val X153 = Cuda(X)
			JCudaTensor x5;
			JTensorFloat x6;
			x6 = x2;
			x5 = x6.asJCudaTensor();

			// val X154 = Convolv(1,0)(X153,cv1_W,cv1_B)
			JCudaTensor x7;
			JCudaTensor x8, x9, x10;
			x8 = x5;
			x9 = x11;
			x10 = x12;
			x7 = x13.forward(x8, x9, x10);

			// Dealloc(X153)
			JCudaTensor x14;
			x14 = x5;
			x14.free();

			// val X155 = Pooling(2,2,0,true)(X154)
			JCudaTensor x15;
			JCudaTensor x16;
			x16 = x7;
			x15 = x17.forward(x16);

			// Dealloc(X154)
			JCudaTensor x18;
			x18 = x7;
			x18.free();

			// val X156 = Convolv(1,0)(X155,cv2_W,cv2_B)
			JCudaTensor x19;
			JCudaTensor x20, x21, x22;
			x20 = x15;
			x21 = x23;
			x22 = x24;
			x19 = x25.forward(x20, x21, x22);

			// Dealloc(X155)
			JCudaTensor x26;
			x26 = x15;
			x26.free();

			// val X157 = Pooling(2,2,0,true)(X156)
			JCudaTensor x27;
			JCudaTensor x28;
			x28 = x19;
			x27 = x29.forward(x28);

			// Dealloc(X156)
			JCudaTensor x30;
			x30 = x19;
			x30.free();

			// val X158 = (X157[1><3])(i | @) * (fc1_W)(j | @)
			JCudaTensor x31;
			JCudaMatrix x32;
			JCudaMatrix x33;
			JCudaTensor x34;
			JCudaTensor x35;
			x35 = x27;
			x34 = x35.flatten(1, new int[]{50, 4, 4});
			x32 = x34.asMatrix(1, true);
			JCudaTensor x36;
			x36 = x37;
			x33 = x36.asMatrix(1, true);
			x31 = x32.times(x33);

			// Dealloc(X157)
			JCudaTensor x38;
			x38 = x27;
			x38.free();

			// val X160 = (X158 + (i) => fc1_B)
			JCudaTensor x39;
			JCudaTensor x40, x41;
			x40 = x31;
			x41 = x42;
			x39 = x41.copy(500, x40);

			// val X161 = ReLU()(X160)
			JCudaTensor x43;
			JCudaTensor x44;
			x44 = x39;
			x43 = x45.forward(x44);

			// val X162 = (X161)(i | @) * (fc2_W)(j | @)
			JCudaTensor x46;
			JCudaMatrix x47;
			JCudaMatrix x48;
			JCudaTensor x49;
			x49 = x43;
			x47 = x49.asMatrix(1, true);
			JCudaTensor x50;
			x50 = x51;
			x48 = x50.asMatrix(1, true);
			x46 = x47.times(x48);

			// Dealloc(X161)
			JCudaTensor x52;
			x52 = x43;
			x52.free();

			// val X164 = (X162 + (i) => fc2_B)
			JCudaTensor x53;
			JCudaTensor x54, x55;
			x54 = x46;
			x55 = x56;
			x53 = x55.copy(500, x54);

			// Prediction(X164)
			JCudaTensor x57;
			x57 = x53;
			System.out.println(x3 + " inference " + java.util.Arrays.toString(x57.prediction()));

			// Dealloc(X164)
			JCudaTensor x58;
			x58 = x53;
			x58.free();

		}

	}

}