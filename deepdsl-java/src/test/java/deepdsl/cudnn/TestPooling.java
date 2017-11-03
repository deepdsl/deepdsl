package deepdsl.cudnn;

import java.util.Arrays;

import org.junit.Test;

import deepdsl.cudnn.JCudaTensor;
import deepdsl.cudnn.JCudnnPooling;
import deepdsl.tensor.JTensorFloat;

public class TestPooling {
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
		int[] dims = {1, 1, 4, 4};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnPooling pool = new JCudnnPooling(dims);
		
		JCudaTensor y = pool.forward(x);
		System.out.println(Arrays.toString(y.asArray()));
		
		float[] b = new float[4];
		for(int i=0; i<4; i++) {
			b[i] = pool(i, a);
		}
		System.out.println(Arrays.toString(b));
	}
	
	@Test
	public void backward() {
		float[] a = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
		int[] dims = {1, 1, 4, 4}, y_dims = {1, 1, 2, 2};
		JTensorFloat t = new JTensorFloat(a, dims);
		JCudaTensor x = new JCudaTensor(t);
		
		JCudnnPooling pool = new JCudnnPooling(dims);
		
		JCudaTensor y = pool.forward(x);
		
		JCudaTensor dy = new JCudaTensor(new JTensorFloat(new float[]{1, 1, 1, 1}, y_dims));
		
		JCudaTensor dx = pool.backward(y, dy, x);
		
		System.out.println(Arrays.toString(dx.asArray()));
		
		float[] b = new float[4];
		for(int i=0; i<4; i++) {
			b[i] = pool(i, a);
		}
		System.out.println(Arrays.toString(a));
		System.out.println(Arrays.toString(b));
	}
	
	float pool(int i, float[] array) {
		float ret;
		if(i == 0) {
			ret = max(array, new int[] {0, 1, 4, 5});
		}
		else if(i == 1) {
			ret = max(array, new int[] {2, 3, 6, 7});
		}
		else if(i == 2) {
			ret = max(array, new int[] {8, 9, 12, 13});
		}
		else {
			ret = max(array, new int[] {10, 11, 14, 15});
		}
		return ret;
	}
	float max(float[] array, int[] indices) {
		float ret = array[indices[0]];
		for(int i=1;i<4;i++) {
			float x = array[indices[i]];
		
			if(ret < x) {
				ret = x;
			}
		}
		return ret;
	}
}
