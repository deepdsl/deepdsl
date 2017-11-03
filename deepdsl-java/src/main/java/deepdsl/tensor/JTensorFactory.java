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
	
	// retrieve image and label simultaneously and nullify the image or label after retrieval
	private JTensorFloat imageCache = null, labelCache = null;
	
	public JTensorFloat image() {
		if (imageCache == null) {
			JTensorFloatTuple tuple = nextFloat();
			imageCache = tuple.image;
			labelCache = tuple.label;
		}
		JTensorFloat ret = imageCache;
		imageCache = null;
		return ret;
	}
	public JTensorFloat label() {
		if (labelCache == null) {
			JTensorFloatTuple tuple = nextFloat();
			imageCache = tuple.image;
			labelCache = tuple.label;
		}
		JTensorFloat ret = labelCache;
		labelCache = null;
		return ret;
	}
	
	public abstract JTensorFloatTuple nextFloat();
}
