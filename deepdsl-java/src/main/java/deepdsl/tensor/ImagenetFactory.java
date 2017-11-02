package deepdsl.tensor;

import deepdsl.data.imagenet.ImagenetDataHandler;
import deepdsl.data.imagenet.Tuple;
import deepdsl.util.ArithStats;

import java.util.Arrays;
import java.util.Random;

public class ImagenetFactory extends JTensorFactory {
    int max_size;
    int count = 0;
    int[] mean;
    boolean centerCropping = false;
    final ImagenetDataHandler handler;

    private ImagenetFactory(int[] dims, String path, int max_size, int numOfClasses, boolean centerCropping) {
        super(dims);

        handler = new ImagenetDataHandler(path, "dataset/imagenet/train.txt", 128);

        this.centerCropping = centerCropping;
        mean = (dims[1] == 3) ? new int[]{104, 117, 123} : new int[dims[1]]; // use fixed mean if channel size is 3

        this.max_size = max_size;

        if(max_size < batch) {
            throw new RuntimeException("not enough training/test samples for even one batch");
        }
        if(_k == 0) {
            _k = numOfClasses;
            if(labelMap == null || labelMap.length != _k) {
                labelMap = new int[numOfClasses];
                Arrays.fill(labelMap, -1);
            }
        }
        else if(_k != numOfClasses){
            throw new RuntimeException("Different number of classes for training/test samples");
        }
    }

    // The lookup method is for testing only to force the class label to have a number of classes less than 1000
    static final int maxNumOfClasses = 1000;
    static int _k = 0;
    static int[] labelMap; // = {365, 981, 863, 342, 819, 337, 374, 608, 895, 263}; // hard-coded class label translation for internal testing

    static int lookup(int x) {
        if(_k >= maxNumOfClasses) { // don't translate labels if there are already 1000 classes
            return x;
        }
        for(int i=0; i<_k; i++) {
            if(labelMap[i] == x) {
                return i;
            }
            else if (labelMap[i] == -1) {
                labelMap[i] = x;
                return i;
            }
        }
        throw new RuntimeException(String.format("Data has more than %d classes of labels", _k));
    }

    public JTensorFloatTuple nextFloat() {
        long begin = System.nanoTime();

        count = count + 1;

        if (count * batch > max_size) {
            count = 1;
        }

        int batch = dims[0], channel = dims[1], height = dims[2], width = dims[3];
        float[] images = new float[size];
        float[] labels = new float[batch];

        int x = 0;

        for (int i = 0; i < batch; i ++) {
            Tuple tuple = handler.nextTuple(count);
            byte[] bytes = tuple.image;
            int label = tuple.label;
            int[] image_dims = tuple.dims;

            if(image_dims[0] != channel || image_dims[1] < height || image_dims[2] < width) {
                throw new RuntimeException(String.format("image dimension %s is not compatible with specified dimensions (last 3) %s",
                        Arrays.toString(image_dims), Arrays.toString(dims)));
            }

            int stride = image_dims[1] * image_dims[2];

            if(image_dims[1] == height && image_dims[2] == width) {
                for(int k = 0; k < channel; k++) {
                    int channel_offset = k * stride;

                    for(int j = 0; j < stride; j++) {
                        images[x++] = ((int) (bytes[channel_offset + j] & 0xFF) - mean[k])/128.0f;
                    }
                }
            }
            else {
                int height_index = centerCropping ? (image_dims[1] - height)/2 : new Random().nextInt(image_dims[1] - height + 1);
                int width_index = centerCropping? (image_dims[2] - width)/2 : new Random().nextInt(image_dims[2] - width + 1);

                for(int k = 0; k < channel; k++) {
                    int channel_offset = k * stride;

                    for(int h = 0; h < height; h++) {
                        int height_offset = channel_offset + (height_index + h) * image_dims[2] + width_index;

                        for(int w = 0; w < width; w++) {
                            images[x++] = ((int) (bytes[height_offset + w] & 0xFF) - mean[k])/128.0f;
                        }

                    }
                }
            }
            labels[i] = lookup(label); // FIXME: Users can replace "lookup(label)" with just "label" for actual training with 1000 classes
        }

        JTensorFloatTuple pair = new JTensorFloatTuple(new JTensorFloat(images, dims),
                new JTensorFloat(labels, new int[]{batch}));
        ArithStats.timing("sliceImagenetFloat", begin);

        return pair;
    }

    /**
     * The method to set up Factory for stream loading from Imagenet
     *
     * @param dims
     * @param path the path of the Imagenet data
     * @return
     */
    public static ImagenetFactory getFactory(String path, int max_size, int[] dims, int numOfClasses, boolean centerCropping) {
        return new ImagenetFactory(dims, path, max_size, numOfClasses, centerCropping);
    }
    public static ImagenetFactory getFactory(String path, int max_size, int[] dims, int numOfClasses) {
        return getFactory(path, max_size, dims, numOfClasses, false);
    }

    public static ImagenetFactory getFactory(String path, int max_size, int[] dims) {
        return getFactory(path, max_size, dims, 10);
    }


    public String toString() { return "Imagenet " + Arrays.toString(dims); }
}
