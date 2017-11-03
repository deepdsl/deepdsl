package deepdsl.cudnn;

import java.util.Arrays;

import org.junit.Test;

import deepdsl.tensor.JTensorFloat;

public class TestDropout {
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6};
		int[] dims = {2, 3};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnDropout dropout = new JCudnnDropout("dropout", dims, 0.5f);
		
		JCudaTensor y = dropout.forward(x);
		System.out.println(Arrays.toString(y.asArray()));
	}
	
	@Test
	public void backward() {
		float[] a = {1,2,3,4,5,6};
		int[] dims = {2, 3};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnDropout dropout = new JCudnnDropout("dropout", dims, 0.5f);
		
		dropout.forward(x); // don't delete this. There is shared reserve memory between forward and backward
		JCudaTensor dy = new JCudaTensor(new JTensorFloat(new float[]{1, 1, 1, 1, 1, 1}, dims));
		
		JCudaTensor dx = dropout.backward(dy);
		
		System.out.println(Arrays.toString(dx.asArray()));
	}
}
