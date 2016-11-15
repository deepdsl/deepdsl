package deepdsl.data.mnist;

import java.util.ArrayList;
import java.util.List;

public abstract class JImgDataSet {

    private final List<byte[]> imgs = new ArrayList<>();

    private final List<float[]> imgList = new ArrayList<>();

    private final List<double[]> imgDoubleList = new ArrayList<>();

    public void convertByteArrayListToFloatArrayList() {
        if ((imgList.size() == 0) && imgs.size() > 0) {
            for (byte[] img : imgs) {
                float[] floatArray = new float[img.length];
                for (int i = 0; i <  img.length; i++) {
                    floatArray[i] = new Integer((img[i] & 0xFF)).floatValue();
                }
                imgList.add(floatArray);
            }
            imgs.clear();
        }
    }

    public void convertByteArrayListToDoubleArrayList() {
        if ((imgDoubleList.size() == 0) && imgs.size() > 0) {
            for (byte[] img : imgs) {
                double[] doubleArray = new double[img.length];
                for (int i = 0; i <  img.length; i++) {
                    doubleArray[i] = new Integer((img[i] & 0xFF)).doubleValue();
                }
                imgDoubleList.add(doubleArray);
            }
            imgs.clear();
        }
    }

    public List<byte[]> getImgs() {
        return imgs;
    }

    public List<float[]> getFloatArrayImgs() {
        return imgList;
    }

    public List<double[]> getDoubleArrayImgs() {
        return imgDoubleList;
    }

    public void addImg(byte[] img) {
        imgs.add(img);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("" + getHeader() + ", img.number=" + imgList.size() + ", leading 0-5 img data=\n");
        for (int i = 0; i < (5 <= imgList.size() ? 5 : imgList.size()); i++) {
            float[] img = imgList.get(i);
            int returnCheck = 0;
            for (float b : img) {
                sb.append(b + " ");
                if (++returnCheck % 16 == 0) {returnCheck = 0; sb.append("\n");}
            }
            sb.append(getSeparate(100));
        }
        return sb.toString();
    }

    private static String getSeparate(int len) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; i++) {
            sb.append("~");
        }
        return sb.append("\n").toString();
    }

    public abstract IHeader getHeader();

    public abstract void setHeader(Integer... values);

}
