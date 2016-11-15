package deepdsl.tensor;

public abstract class JTensorFactory {
	int[] dims;
	int size, batch;
	
	int size() {
		int ret = 1;
		for(int i=0; i<dims.length; i++) {
			ret = ret * dims[i];
		}
		return ret;
	}
	
	JTensorFactory(int[] dims) {
		this.dims = dims;
		this.size = size();
		this.batch = dims[0];
	}
	public abstract JTensorFloatTuple nextFloat();
}
