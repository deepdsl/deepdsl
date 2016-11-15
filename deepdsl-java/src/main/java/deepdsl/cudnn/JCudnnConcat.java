package deepdsl.cudnn;

import java.util.Arrays;

import deepdsl.util.ArithStats;
import jcuda.jcudnn.JCudnn;

// NCHW format
public class JCudnnConcat extends JCudaFunction {
	int[][] dimsList;
	JCudnnDescriptor[] xdptrList, ydptrList;
	int[] ydims;
	
	public JCudnnConcat(int[]... dimsList) {
		this.dimsList = dimsList;
		checkDims(dimsList);
		int channelStride = channelSize(dimsList);
		xdptrList = new JCudnnDescriptor[dimsList.length];
		ydptrList = new JCudnnDescriptor[dimsList.length];
		for(int i=0; i<dimsList.length; i++) {
			xdptrList[i] = new JCudnnDescriptor(dimsList[i]);
			ydptrList[i] = new JCudnnDescriptor(dimsList[i], channelStride);
		}
		int[] xdims = dimsList[0];
		ydims = new int[4];
		ydims[0] = xdims[0]; 
		ydims[1] = channelStride;
		ydims[2] = xdims[2];
		ydims[3] = xdims[3];
	}
	public void free() {
		for(int i=0; i<dimsList.length; i++) {
			xdptrList[i].free();
			ydptrList[i].free();
		}
	}
	// concatenate x tensors along channel axis in y tensor
	public JCudaTensor forward(JCudaTensor... x) { 
		if(x.length != dimsList.length) {
			throw new RuntimeException("tensor list size doesnot match dims list size in tensor concatenation");
		}
		JCudaTensor y = new JCudaTensor(ydims);
		int offset = 0;
		long begin = System.nanoTime();
		
		int ret = 0;
		for(int i=0; i<x.length; i++) {
			int[] xdims = x[i].getDims(); 
			ret = JCudnn.cudnnAddTensor(cudnnHandle, one, xdptrList[i].descriptor, x[i].getData(), zero, ydptrList[i].descriptor, y.getData(offset)); 
			offset += xdims[1] * xdims[2] * xdims[3];
			checkError(ret);
		}
		ArithStats.cuda_timing("concat forward", begin);
		return y;
	}
	// split y into an array of tensors along channel axis -- using predefined descriptors
	public JCudaTensor[] backward(JCudaTensor y) {
		JCudaTensor[] xList = new JCudaTensor[dimsList.length];
		int offset = 0;
		long begin = System.nanoTime();
		
		int ret = 0;
		for(int i=0; i<dimsList.length; i++) {
			int[] xdims = dimsList[i];
			JCudaTensor x = new JCudaTensor(xdims); 
			ret = JCudnn.cudnnAddTensor(cudnnHandle, one, ydptrList[i].descriptor, y.getData(offset), zero, xdptrList[i].descriptor, x.getData());
			checkError(ret);
			offset += xdims[1] * xdims[2] * xdims[3]; 
			xList[i] = x;
		}
		ArithStats.cuda_timing("concat backward", begin);
		return xList;
	}
	
	private void checkDims(int[][] dimsList) {
		int[] xdims = dimsList[0];
		for(int i = 1; i < dimsList.length; i++) {
			int[] ydims = dimsList[i];
			if(xdims[0] != ydims[0] || xdims[2] != ydims[2] || xdims[3] != ydims[3]) {
				throw new RuntimeException("incompatible dimensions for concatenation: " + 
						Arrays.toString(xdims) + " <> " + Arrays.toString(ydims));
			}
		} 
	}

	private int channelSize(int[][] dimsList) {
		int size = 0;
		for(int[] dims: dimsList) {
			size += dims[1];
		}
		return size;
	}

}
