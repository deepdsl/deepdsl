package deepdsl.cudnn;

import deepdsl.cudnn.config.SoftmaxAlgorithm;
import deepdsl.tensor.JTensorFloat;
import org.junit.Test;

import java.util.Arrays;

public class TestSoftmax {
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6};
		int[] dims = {1, 6, 1, 1};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnSoftmax softmax = new JCudnnSoftmax(dims, SoftmaxAlgorithm.LOG);
		
		JCudaTensor y = softmax.forward(x);
		System.out.println(Arrays.toString(y.asArray()));
		System.out.println(Arrays.toString(log(softmax(a))));
	}
	
	@Test
	public void backward() {
		float[] a = {1,2,3,4,5,6}, c = {0, 0, 0, 0, 0, 1};
		int[] dims = {1, 6, 1, 1};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnSoftmax softmax = new JCudnnSoftmax(dims, SoftmaxAlgorithm.LOG);
		
		JCudaTensor y = softmax.forward(x); 
		JCudaTensor dy = new JCudaTensor(new JTensorFloat(c, dims));
		
		JCudaTensor dx = softmax.backward(dy, y);
		
		System.out.println(Arrays.toString(dx.asArray()));
		
		float[] sa = softmax(a);
		float[] b = new float[6];
		for(int i=0; i<6; i++) {
			b[i] = - sa[i] + c[i];
		}
		System.out.println(Arrays.toString(b));
	}
	
	float[] log(float[] array) {
		float[] ret = new float[array.length]; 
 
		for(int i=0; i<array.length; i++) {
			ret[i] =  (float) Math.log(array[i]);
		}
		return ret;
	}
	
	float[] softmax(float[] array) {
		float[] ret = new float[array.length]; 
		
		for(int i=0; i<array.length; i++) {
			ret[i] = (float) Math.exp(array[i]); 
		}
		float sum = 0;
		for(int i=0; i<array.length; i++) {
			sum += ret[i];
		}
		for(int i=0; i<array.length; i++) {
			ret[i] =  ret[i]/sum;
		}
		return ret;
	}
	
}
