package deepdsl.cudnn;

import java.util.Arrays;

import org.junit.Test; 
import deepdsl.tensor.JTensorFloat;

public class TestLRN {
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6};
		int[] dims = {2, 1, 3, 1};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnLRN lrn = new JCudnnLRN(dims, 5, 0.0001, 0.75);
		
		JCudaTensor y = lrn.forward(x);
		System.out.println(Arrays.toString(y.asArray()));
	}
	
	@Test
	public void backward() {
		float[] a = {1,2,3,4,5,6};
		int[] dims = {2, 1, 3, 1};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnLRN lrn = new JCudnnLRN(dims, 5, 0.0001, 0.75);
		
		JCudaTensor y = lrn.forward(x); 
		JCudaTensor dy = new JCudaTensor(new JTensorFloat(new float[]{1, 1, 1, 1, 1, 1}, dims));
		
		JCudaTensor dx = lrn.backward(dy, y, x);
		
		System.out.println(Arrays.toString(dx.asArray()));
	}
}
