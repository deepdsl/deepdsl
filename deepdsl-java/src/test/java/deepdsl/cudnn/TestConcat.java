package deepdsl.cudnn;

import java.util.Arrays;

import org.junit.Test;

import deepdsl.tensor.JTensorFloat;

public class TestConcat {
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6,7,8}, b = {9,10,11,12,13,14,15,16}, c = {17, 18, 19, 20, 21, 22, 23, 24};
		int[] x_dims = {2, 1, 2, 2}, y_dims = {2, 1, 2, 2}, z_dims = {2, 1, 2, 2};
		
		JTensorFloat t = new JTensorFloat(a, x_dims), t_1 = new JTensorFloat(b, y_dims), t_2 = new JTensorFloat(c, z_dims);
		JCudaTensor x = new JCudaTensor(t), y = new JCudaTensor(t_1), z = new JCudaTensor(t_2);
		
		JCudnnConcat concat = new JCudnnConcat(new int[][] {x_dims, y_dims, z_dims});
		 
		JCudaTensor ret = concat.forward(x, y, z);
		System.out.println(Arrays.toString(ret.asArray()));
		
	}
	
	@Test
	public void backward() {
		float[] a = {1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24};
		int[] r_dims = {2, 3, 2, 2};
		int[] x_dims = {2, 1, 2, 2}, y_dims = {2, 1, 2, 2}, z_dims = {2, 1, 2, 2};
		
		JCudaTensor r = new JTensorFloat(a, r_dims).asJCudaTensor();
		JCudnnConcat concat = new JCudnnConcat(new int[][] {x_dims, y_dims, z_dims});
		
		JCudaTensor[] x = concat.backward(r);
		for(int i=0; i<x.length; i++) {
			System.out.println(Arrays.toString(x[i].asArray()));
		}
	}
}
