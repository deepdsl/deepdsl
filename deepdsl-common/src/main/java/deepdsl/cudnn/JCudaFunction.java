package deepdsl.cudnn;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.runtime.JCuda;
import jcuda.vec.VecFloat;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public abstract class JCudaFunction {
	abstract void free();
	
	public static final cudnnHandle cudnnHandle = new cudnnHandle();
	public static final cublasHandle cublasHandle = new cublasHandle();
	public static final Pointer one = Pointer.to(new float[]{1.0f});
	public static final Pointer zero = Pointer.to(new float[]{0.0f});
	
	static {
		JCudnn.cudnnCreate(cudnnHandle);
		JCublas2.cublasCreate(cublasHandle);
		VecFloat.init();
	}
	
	public static void destroy() {
		JCudnn.cudnnDestroy(cudnnHandle);
		JCublas2.cublasDestroy(cublasHandle);
		VecFloat.shutdown();
		if(workspaceSize > 0 && workspace != null) {
			cudaFree(workspace);
		}
	} 
	
	public static void allocFloat(Pointer data, int size) {
		allocByte(data, size * Sizeof.FLOAT);  
	}
	public static void allocByte(Pointer data, long size) {
		checkError(cudaMalloc(data, size)); 
	}
	
	public static void free(Pointer data) {
		checkError(cudaFree(data));
	}
	
	public static void copyHostToDevice(float[] source, Pointer target) {
		long size = source.length * Sizeof.FLOAT;
		int ret = cudaMemcpy(target, Pointer.to(source), size, cudaMemcpyHostToDevice); 
		checkError(ret);
	}
	
	public static void copyDeviceToDevice(long size, Pointer source, Pointer target) {
        int ret = cudaMemcpy(target, source, size * Sizeof.FLOAT, cudaMemcpyDeviceToDevice); 
		checkError(ret);
	}
	
	public static void copyDeviceToHost(Pointer source, float[] target) {
		long size = target.length * Sizeof.FLOAT;
		
		int ret = cudaMemcpy(Pointer.to(target), source, size, cudaMemcpyDeviceToHost); 
		checkError(ret);
	}
	
	public static void checkError(int x) {  
		if(x != 0) {
			throw new RuntimeException("error code: " + JCudnn.cudnnGetErrorString(x) + " for jcudnn call");
		}
	}
	
	public static void sync() {
		JCuda.cudaDeviceSynchronize();
	}
	
	private static Pointer workspace;
	static long workspaceSize = 0;
	
	public static void allocateWorkspace(long size) {
		if(workspaceSize < size) { 
			workspaceSize = size; 
			
			if(workspace != null) {
				free(workspace);
				workspace = null;
			}
		} 
	}
	
	public static Pointer getWorkspace() { 
		if(workspace == null) {
			workspace = new Pointer();
			if(workspaceSize > 0) {
				allocByte(workspace, workspaceSize);
			}
		}
		return workspace;
	}
	
	public static long getWorkspaceSize() { return workspaceSize; }
	
	static float current = 1;
	static Pointer currentPointer = one;
	
	public static Pointer pointerTo(float value) { 
		Pointer ret;
		if(value == 1) {
			ret = one;
		}
		else if (value == current){
			ret = currentPointer;
		}
		else {
			current = value;
			cudaFree(currentPointer);
			currentPointer = Pointer.to(new float[]{value}); 
			ret = currentPointer;
		}
		return ret;
	}
}
