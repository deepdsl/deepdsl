package deepdsl.cudnn;

import java.util.Arrays;

import org.junit.Test;

import deepdsl.cudnn.JCudaTensor;
import deepdsl.cudnn.JCudnnActivation;
import deepdsl.cudnn.config.ActivationMode;
import deepdsl.tensor.JTensorFloat;

public class TestActivation {
	@Test
	public void forward_relu() {
		float[] a = {1,-2,3,-4,5,-6};
		int[] dims = {2, 1, 3, 1};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnActivation activation = new JCudnnActivation(dims, ActivationMode.RELU);
		
		JCudaTensor y = activation.forward(x);
		System.out.println(Arrays.toString(y.asArray()));
		JCudaTensor dy = new JCudaTensor(new JTensorFloat(new float[]{1, 1, 1, 1, 1, 1}, dims));
		
		JCudaTensor dx = activation.backward(dy, y, x);
		
		System.out.println(Arrays.toString(dx.asArray()));
	}
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6};
		int[] dims = {2, 1, 3, 1};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnActivation activation = new JCudnnActivation(dims, ActivationMode.TANH);
		
		JCudaTensor y = activation.forward(x);
		System.out.println(Arrays.toString(y.asArray()));
		
		float[] b = new float[6];
		for(int i=0; i<6; i++) {
			b[i] = (float) Math.tanh(a[i]);
		}
		System.out.println(Arrays.toString(b));
	}
	
	@Test
	public void backward() {
		float[] a = {1,2,3,4,5,6};
		int[] dims = {2, 1, 3, 1};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnActivation activation = new JCudnnActivation(dims, ActivationMode.TANH);
		
		JCudaTensor y = activation.forward(x); 
		JCudaTensor dy = new JCudaTensor(new JTensorFloat(new float[]{1, 1, 1, 1, 1, 1}, dims));
		
		JCudaTensor dx = activation.backward(dy, y, x);
		
		System.out.println(Arrays.toString(dx.asArray()));
		
		float[] b = new float[6];
		for(int i=0; i<6; i++) {
			b[i] = (float) (1 - Math.pow(Math.tanh(a[i]), 2));
		}
		System.out.println(Arrays.toString(b));
	}
}
