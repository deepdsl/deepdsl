package deepdsl.cudnn;

import java.util.Arrays;

import org.junit.Test;

import deepdsl.cudnn.JCudaTensor;
import deepdsl.cudnn.JCudnnConvolution;
import deepdsl.tensor.JTensorFloat;

public class TestConvolution {
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, a_1 = {1, 1, 1, 1}, a_2 = { 0.1f };
		int[] x_dims = {1, 1, 4, 4}, w_dims = {1, 1, 2, 2}, b_dims = {1, 1, 1, 1};
		JTensorFloat t = new JTensorFloat(a, x_dims), t_1 = new JTensorFloat(a_1, w_dims), t_2 = new JTensorFloat(a_2, b_dims);
		JCudaTensor x = new JCudaTensor(t), w = new JCudaTensor(t_1), b = new JCudaTensor(t_2);
		
		JCudnnConvolution convolv = new JCudnnConvolution(x_dims, w_dims, b_dims, 2, 0);
		
		System.out.println(Arrays.toString(convolv.y_dims));
		
		JCudaTensor y = convolv.forward(x, w, b);
		System.out.println(Arrays.toString(y.asArray()));
		
//		float[] b = new float[4];
//		for(int i=0; i<4; i++) {
//			b[i] = pool(i, a);
//		}
//		System.out.println(Arrays.toString(b));
	}
	
	@Test
	public void backward_data() {
		float[] a_1 = {1, 1, 1, 1}; 
		int[] x_dims = {1, 1, 4, 4}, w_dims = {1, 1, 2, 2}, b_dims = {1, 1, 1, 1}, y_dims = {1, 1, 3, 3};
		JTensorFloat t_1 = new JTensorFloat(a_1, w_dims), t_3 = new JTensorFloat(new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1}, y_dims);
		JCudaTensor w = new JCudaTensor(t_1), dy = new JCudaTensor(t_3);
		
		JCudnnConvolution convolv = new JCudnnConvolution(x_dims, w_dims, b_dims);
		 
		JCudaTensor dx = convolv.backward_data(dy, w);
		
		System.out.println(Arrays.toString(dx.asArray()));
	}
	
	@Test
	public void backward_filter() {
		float[] a = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}; 
		int[] x_dims = {1, 1, 4, 4}, w_dims = {1, 1, 2, 2}, b_dims = {1, 1, 1, 1}, y_dims = {1, 1, 3, 3};
		JTensorFloat t = new JTensorFloat(a, x_dims), t_3 = new JTensorFloat(new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1}, y_dims);
		JCudaTensor x = new JCudaTensor(t), dy = new JCudaTensor(t_3);
		
		JCudnnConvolution convolv = new JCudnnConvolution(x_dims, w_dims, b_dims);
		
		JCudaTensor dw = convolv.backward_filter(dy, x);
		
		System.out.println(Arrays.toString(dw.asArray()));
	}
	
	@Test
	public void backward_bias() {
		int[] x_dims = {1, 1, 4, 4}, w_dims = {1, 1, 2, 2}, b_dims = {1, 1, 1, 1}, y_dims = {1, 1, 3, 3};
		JTensorFloat t_3 = new JTensorFloat(new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1}, y_dims);
		JCudaTensor dy = new JCudaTensor(t_3);
		
		JCudnnConvolution convolv = new JCudnnConvolution(x_dims, w_dims, b_dims);
		
		JCudaTensor db = convolv.backward_bias(dy);
		
		System.out.println(Arrays.toString(db.asArray()));
	}
}
