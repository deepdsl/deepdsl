package deepdsl.cudnn;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc; 
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;
import jcuda.vec.VecFloat;

public abstract class JCudaFunction { 
	public abstract void free();
	public void save() {}
	
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
		freeWorkspace();
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
	
	public static void copyHostToDevice(byte[] source, Pointer target) {
		long size = source.length;
		int ret = cudaMemcpy(target, Pointer.to(source), size, cudaMemcpyHostToDevice); 
		checkError(ret);
	}
	
	public static void copyDeviceToDevice(long size, Pointer source, Pointer target) {
		size = size * Sizeof.FLOAT;
        int ret = cudaMemcpy(target, source, size, cudaMemcpyDeviceToDevice); 
		checkError(ret);
	}
	
	public static void copyDeviceToHost(Pointer source, float[] target) {
		long size = target.length * Sizeof.FLOAT;
		
		int ret = cudaMemcpy(Pointer.to(target), source, size, cudaMemcpyDeviceToHost); 
		checkError(ret);
	}
	
	public static void copyDeviceToHost(Pointer source, byte[] target) {
		long size = target.length;
		
		int ret = cudaMemcpy(Pointer.to(target), source, size, cudaMemcpyDeviceToHost); 
		checkError(ret);
	}
	
	public static void checkError(int x) {  
		if(x != 0) {
			throw new RuntimeException("error code: " + JCudnn.cudnnGetErrorString(x) + " for jcudnn call");
		}
	}
	
	public static void sync() {
		int ret = JCuda.cudaDeviceSynchronize();
		checkError(ret);
	}
	
	private static Pointer workspace;
	static long workspaceSize = 0;
	
	private static void freeWorkspace() {
		if(workspace != null) {
			if(workspaceSize > 0) {
				free(workspace);
			}
			workspace = null;
		}
	}
	public static void reserveWorkspace(long size) {
		if(workspaceSize < size) { 
			workspaceSize = size; 
			freeWorkspace();
		} 
	}

	// use reserved workspace size (max)
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
	
	public static void enableWorkspaceCache() { cacheWorkspace = true; }
	public static void disableWorkspaceCache() { cacheWorkspace = false; }
	
	private static boolean cacheWorkspace = false;
	
	// use argument size (ignore reserved workspace size) if workspace is not cached
	public static Pointer allocWorkspace(long size) {
		if(! cacheWorkspace) {
			freeWorkspace();
			workspaceSize = size;
		}
		return getWorkspace();
	}
	
	public static void deallocWorkspace() {
		if(! cacheWorkspace) {
			freeWorkspace();
		}
	}
	
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
