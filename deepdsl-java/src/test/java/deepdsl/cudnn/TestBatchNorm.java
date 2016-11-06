package deepdsl.cudnn;

import java.util.Arrays;
import org.junit.Test;

import deepdsl.tensor.JTensorFloat;

public class TestBatchNorm {
	
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6}, a1 = {10, 100, 1000}, a2 = {0.1f, 0.2f, 0.3f};
		int[] dims = {2, 3, 1, 1}, norm_dims = {1, 3, 1, 1};
		JTensorFloat t = new JTensorFloat(a, dims), 
				tscale = new JTensorFloat(a1, norm_dims), 
				tbias = new JTensorFloat(a2, norm_dims);
		JCudaTensor x = t.asJCudaTensor(),
				scale = tscale.asJCudaTensor(),
				bias = tbias.asJCudaTensor();
		
		JCudnnBatchNorm norm = new JCudnnBatchNorm(dims);
		
		JCudaTensor y = norm.forward(x, scale, bias);
		System.out.println(Arrays.toString(y.asArray()));
		System.out.println(Arrays.toString(norm.running_mean.asArray()));
		System.out.println(Arrays.toString(norm.running_variance.asArray()));
		System.out.println(Arrays.toString(norm.saved_mean.asArray()));
		System.out.println(Arrays.toString(norm.saved_inv_variance.asArray()));
		
		y = norm.forward_inference(x, scale, bias);
		System.out.println(Arrays.toString(y.asArray()));
	}
	
	@Test
	public void backward() {
		float[] a = {1, 2, 3, 4, 5 ,6}, a1 = {10, 20, 30}, a2 = {0.1f, 0.2f, 0.3f}, 
				a4 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
		int[] dims = {2, 3, 1, 1}, norm_dims = {1, 3, 1, 1};
		JTensorFloat t = new JTensorFloat(a, dims), 
				tscale = new JTensorFloat(a1, norm_dims), 
				tbias = new JTensorFloat(a2, norm_dims);
		JCudaTensor x = t.asJCudaTensor(),
				scale = tscale.asJCudaTensor(),
				bias = tbias.asJCudaTensor(), 
				dy = new JTensorFloat(a4, dims).asJCudaTensor();
		
		JCudnnBatchNorm norm = new JCudnnBatchNorm(dims);
		
		JCudaTensor y = norm.forward(x, scale, bias);
		 
		JCudaTensor dx[] = norm.backward(dy, x, scale);
		
		System.out.println("y: " + Arrays.toString(y.asArray()));
		System.out.println("dx: " + Arrays.toString(dx[0].asArray()));
		System.out.println("d_scale: " + Arrays.toString(dx[1].asArray()));
		System.out.println("d_bias: " + Arrays.toString(dx[2].asArray()));
		
		y = norm.forward(x, dx[1], dx[2]);
		System.out.println("y: " + Arrays.toString(y.asArray()));
		
		dx = norm.backward(dy, x, dx[1]);
		System.out.println("dx: " + Arrays.toString(dx[0].asArray()));
		System.out.println("d_scale: " + Arrays.toString(dx[1].asArray()));
		System.out.println("d_bias: " + Arrays.toString(dx[2].asArray()));
	}
}
