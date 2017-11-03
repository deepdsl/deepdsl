package deepdsl.tensor;

import deepdsl.data.mnist.JDataSetLoader;
import deepdsl.data.mnist.JDataSetMeta;
import deepdsl.data.mnist.JImgDataSet;
import deepdsl.data.mnist.JLabelDataSet;
import deepdsl.util.ArithStats;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MnistFactory extends JTensorFactory {
	public static final int max_train_size = 50000, max_test_size = 10000;

	byte[] dataByteArray, labelByteArray; 
	int counter;
	int max_count; 

	private MnistFactory(byte[] dataByteArray, byte[] labelByteArray, int[] dims) {
		super(dims);
		if(dataByteArray.length/size != labelByteArray.length/dims[0]) { 
			throw new RuntimeException("image and label array have different sizes");
		}
		this.dataByteArray = dataByteArray; 
		this.labelByteArray = labelByteArray;
		counter = 0; 
		max_count = dataByteArray.length / size;
	}

	public JTensorFloatTuple nextFloat() {
		long begin = System.nanoTime(); 
		int start = counter * size;

		float[] array = new float[size];
		for(int i=0; i<size; i++) {
			array[i] = ((int) (dataByteArray[i+start] & 0xFF) - 127) / 128.0f; 
		}
		JTensorFloat image = new JTensorFloat(array, dims);
		
		array = new float[batch];
		start = counter * batch;
		for(int i=0; i<batch; i++) {
			array[i] = labelByteArray[i+start] & 0xFF;
		}
		JTensorFloat label = new JTensorFloat(array, new int[]{batch});

		counter = (counter + 1) % max_count;
		ArithStats.timing("sliceFloat", begin);
		
		return new JTensorFloatTuple(image, label);
	}

	/**
	 * The method to load actual dataset (e.g. mnist) into a byte array.
	 * Please see note in the if block that explains why this method can be used universally.
	 *
	 * @param isTrain   (train | test)
	 * @param dataType  (img | label)
	 * @param numSamples the # of samples to load
	 * @param dims		the dimensions of the tensors
	 * @return a tensor factory to return tensors 
	 * @throws IOException
	 */
	public static MnistFactory getFactory(boolean isTrain, int[] dims) {
		if(dims.length != 4) {
			throw new RuntimeException("incorrect dims for Mnist factory: " + Arrays.toString(dims));
		}
		
		byte[] imageByteArray, labelByteArray;
		int i = 0;
		int numSamples = isTrain? max_train_size : max_test_size;

		try {
			Map<String, JDataSetMeta> map = JDataSetLoader.loadJson();
			// the 1st dim is the size of imgs/labels 
			// Since the isFloat parameter here makes no difference when we return byte array, this loadDataSetInByteArray
			// function can be us in all scenarios (i.e. no need to put a similar function in TensorFloat class)
			JImgDataSet imgDataset = JDataSetLoader.getJImgDataSet(map, isTrain, numSamples, false, false);
			List<byte[]> imgs = imgDataset.getImgs();
			imageByteArray = new byte[imgs.size() * imgs.get(0).length];
			for (byte[] img : imgs) { 
				System.arraycopy(img, 0, imageByteArray, i  * img.length, img.length);
				++i;
			}
			i = 0;
			JLabelDataSet labelDataSet = JDataSetLoader.getJLabelDataSet(map, isTrain, numSamples, false);
			List<Byte> labels = labelDataSet.getLabels();
			labelByteArray = new byte[labels.size()];
			for (byte label : labels) {
				labelByteArray[i++] = label;
			} 
		} catch (IOException e) {
			//Wrap this as a RuntimeException such that we don't leak out exception handling to interface
			throw new RuntimeException(e.getMessage());
		}
		return new MnistFactory(imageByteArray, labelByteArray, dims);
	}

}
