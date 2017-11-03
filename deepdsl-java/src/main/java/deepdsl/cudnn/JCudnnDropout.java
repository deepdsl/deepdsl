package deepdsl.cudnn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import deepdsl.util.ArithStats;
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnDropoutDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor; 

class DropoutState implements Serializable {  
	private static final long serialVersionUID = 4800642655547993085L;
	int[] dim;
	float dropout;
	long seed;
	long size; 
	byte[] states;
	
	DropoutState(int[] dim, float dropout, long seed, long size) {
		this(dim, dropout, seed, size, null);
	}
	
	DropoutState(int[] dim, float dropout, long seed, long size, byte[] states) {
		this.dropout = dropout;
		this.dim = dim;
		this.seed = seed;
		this.size = size;
		this.states = states;
	}
	boolean equals(DropoutState t) {
		boolean ret = dropout == t.dropout && dim.length == t.dim.length && seed == t.seed && size == t.size;
		
		if(ret) {
			for(int i=0; i<dim.length; i++) {
				if(dim[i] != t.dim[i]) { ret = false; }
			}
		}
		return ret;
	}
	public DropoutState load(String name) {
		DropoutState ret = this;
		try {
			FileInputStream fileIn = new FileInputStream(name + ".ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			DropoutState t = (DropoutState) in.readObject();
			in.close();
			fileIn.close();
			if(t != null) { 
				if(equals(t)) {
					System.out.printf("Restored %s\n", name);
					ret = t;  
				}
			}
		}
		catch(IOException i) {
		}
		catch(ClassNotFoundException c) { 
		}
		return ret;
	}
	public void save(String name) {
		try {
			FileOutputStream fileOut = new FileOutputStream(name + ".ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(this);
			out.close();
			fileOut.close();
			System.out.printf("Parameter is serialized in %s.ser\n", name);
		}
		catch(IOException i) { 
			System.out.println(i);
		}
	}
}


public class JCudnnDropout extends JCudaFunction {
	cudnnDropoutDescriptor dropoutDesc = new cudnnDropoutDescriptor();
	JCudnnDescriptor dptr;  
	long seed = 0;
	Pointer reserve = new Pointer();
	long[] reserveSize = {0};
	Pointer states = new Pointer();
	DropoutState savedState;
	String path;
	
	public JCudnnDropout(String path, int[] x_dims, float dropout) {
		this.path = path;
		
		this.dptr = new JCudnnDescriptor(x_dims);
		JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
		
		long[] stateSize = {0};
		checkError(JCudnn.cudnnDropoutGetStatesSize(cudnnHandle, stateSize));
		allocByte(states, stateSize[0]);
		checkError(JCudnn.cudnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropout, states, stateSize[0], seed));
 
		savedState = new DropoutState(x_dims, dropout, seed, stateSize[0]).load(path);		
		if(savedState.states != null) {
			JCudaFunction.copyHostToDevice(savedState.states, states); // recover saved states
		}
		else {
			savedState.states = new byte[(int) savedState.size];       
		}
		checkError(JCudnn.cudnnDropoutGetReserveSpaceSize(dptr.descriptor, reserveSize));
		allocByte(reserve, reserveSize[0]);
	}
	
	public void free() {
		dptr.free();
		checkError(JCudnn.cudnnDestroyDropoutDescriptor(dropoutDesc));
		free(states);
		free(reserve);
	}
	
	@Override
	public void save() {
		JCudaFunction.copyDeviceToHost(states, savedState.states);
		savedState.save(path);
	}
	
	public JCudaTensor forward_inference(JCudaTensor x) { return x; }
	
	// y = forward(x)
	public JCudaTensor forward(JCudaTensor x) {
		long begin = System.nanoTime();
		JCudaTensor y = new JCudaTensor(x.getDims()); 
		
		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		 
		int ret = JCudnn.cudnnDropoutForward(cudnnHandle, dropoutDesc,  
				dptr, x.getData(), dptr, y.getData(),
				reserve, reserveSize[0]); 
		 
		checkError(ret);
		
		ArithStats.cuda_timing("Dropout forward", begin);
		return y;
	}
	// dx = backward(dy)
	public JCudaTensor backward(JCudaTensor dy) {
		long begin = System.nanoTime();
		JCudaTensor dx = new JCudaTensor(dy.getDims());

		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		
		int ret = JCudnn.cudnnDropoutBackward(cudnnHandle, dropoutDesc,  
				dptr, dy.getData(), dptr, dx.getData(), 
				reserve, reserveSize[0]);
		 
		checkError(ret);
		
		ArithStats.cuda_timing("Dropout backward", begin);
		return dx;
	} 
}
