package deepdsl.data.mnist;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

public class JDataSetLoader {

    public static void main(String[] args) throws IOException {
        Map<String, JDataSetMeta> map = loadJson();
        System.out.println(map.get(getDataSetRsourceName()));
        JImgDataSet imgTrainDataset = getJImgDataSet(map, true, 50, true, true);
        JImgDataSet imgTestDataset = getJImgDataSet(map, false, 50, true, true);
        JLabelDataSet labelTrainDataset = getJLabelDataSet(map, true, 50, true);
        JLabelDataSet labelTestDataset = getJLabelDataSet(map, false, 50, true);

        System.out.println(imgTrainDataset);
        System.out.println(labelTrainDataset);
        System.out.println(imgTestDataset);
        System.out.println(labelTestDataset);
    }

    public static String dataSet = "";

    public static String getDataSetRsourceName() throws IOException {
        if (!"".equals(dataSet)) {
            return dataSet;
        }

        Properties props = new Properties();
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(JDataSetLoader.class.getResourceAsStream(Constants.RESOURCE_NAME), "UTF-8"))) {
            props.load(reader);
        }
        //For now, we only support one line in the conf file
        dataSet = props.getProperty(Constants.DATASET_NAME);
        return dataSet;
    }

    public static Map<String, JDataSetMeta> loadJson() throws IOException {
        String dataSet = getDataSetRsourceName();
        try (Reader reader = new InputStreamReader(JDataSetLoader.class.getResourceAsStream(Constants.LEADING_FORWARD_SLASH + dataSet + Constants.JSON), "UTF-8")) {
            Gson gson = new GsonBuilder().create();
            return new HashMap<String, JDataSetMeta>(){
				private static final long serialVersionUID = -93224170573308086L;
				{put(dataSet, gson.fromJson(reader, JDataSetMeta.class));}};
        }
    }

    public static JImgDataSet getJImgDataSet(Map<String, JDataSetMeta> map, Boolean isTrain, Integer numLoading, Boolean isFloat, Boolean isConvert) throws IOException {
        String dataSet = getDataSetRsourceName();
        File file1 = getFile(map.get(dataSet), true, isTrain);
        FileInputStream imgStream = getStream(file1);
        MappedByteBuffer imgBuf = loadMnistBuffer(imgStream, file1.length());
        JImgDataSet imgDataset = (Constants.MNIST_DATASET.equals(dataSet)) ? loadMnistImgDataSet(imgBuf, map.get(dataSet), isTrain, numLoading, isFloat, isConvert) : null;
        closeStream(imgStream);
        return imgDataset;
    }

    public static JLabelDataSet getJLabelDataSet(Map<String, JDataSetMeta> map, Boolean isTrain, Integer numLoading, Boolean isConvert) throws IOException {
        String dataSet = getDataSetRsourceName();
        File file1 = getFile(map.get(dataSet), false, isTrain);
        FileInputStream imgStream = getStream(file1);
        MappedByteBuffer labelBuf = loadMnistBuffer(imgStream, file1.length());
        JLabelDataSet labelDataset = (Constants.MNIST_DATASET.equals(dataSet)) ? loadMnistLabelDataSet(labelBuf, map.get(dataSet), isTrain, numLoading, isConvert) : null;
        closeStream(imgStream);
        return labelDataset;
    }

    private static JLabelDataSet loadMnistLabelDataSet(MappedByteBuffer buffer, JDataSetMeta meta, Boolean isTrain, Integer numLoading, Boolean isConvert) throws IOException {
        int magicValue = 0, numItems = 0;

        for (JDataSetMeta.JTrainOrTest conf : (isTrain ? meta.getTrain() : meta.getTest())) {
            if (Constants.IMAGE.equals(conf.getDataType())) {
                magicValue = convertToInt(loadBufferOfLength(buffer, conf.getHeader().getMagic()));
                numItems = convertToInt(loadBufferOfLength(buffer, conf.getHeader().getItem()));
            }
        }

        JLabelDataSet dataSet = new JMnistLabelDataSet();
        dataSet.setHeader(magicValue, numItems);

        for (int i = 0; i < numLoading; i++) {
            byte[] label = loadBufferOfLength(buffer, 1);
            dataSet.addLabel(label[0]);
        }

        if (isConvert) {
            dataSet.convertByteListToIntList();
        }

        return dataSet;
    }

    private static JImgDataSet loadMnistImgDataSet(MappedByteBuffer buffer, JDataSetMeta meta, Boolean isTrain, Integer numLoading, Boolean isFloat, Boolean isConvert) throws IOException {
        int magicValue = 0, numItems = 0, numRows = 0, numColumns = 0;

        for (JDataSetMeta.JTrainOrTest conf : (isTrain ? meta.getTrain() : meta.getTest())) {
            if (Constants.IMAGE.equals(conf.getDataType())) {
                magicValue = convertToInt(loadBufferOfLength(buffer, conf.getHeader().getMagic()));
                numItems = convertToInt(loadBufferOfLength(buffer, conf.getHeader().getItem()));
                numRows = convertToInt(loadBufferOfLength(buffer, conf.getHeader().getRow()));
                numColumns = convertToInt(loadBufferOfLength(buffer, conf.getHeader().getColumn()));
            }
        }

        JImgDataSet dataSet = new JMnistImgDataSet();
        dataSet.setHeader(magicValue, numItems, numRows, numColumns);

        Integer imgSize = numRows * numColumns;

        for (int i = 0; i < numLoading; i++) {
            byte[] img = loadBufferOfLength(buffer, imgSize);
            dataSet.addImg(img);
        }

        if (isConvert) {
            if (isFloat) {
                dataSet.convertByteArrayListToFloatArrayList();
            } else {
                dataSet.convertByteArrayListToDoubleArrayList();
            }
        }
        return dataSet;
    }

    private static MappedByteBuffer loadMnistBuffer(FileInputStream stream, long fileLength) throws IOException {
        return getBuffer(stream, fileLength);
    }


    private static void closeStream(FileInputStream stream) throws IOException {
        if (stream != null) stream.close();
    }

    private static File getFile(JDataSetMeta meta, Boolean isImg, Boolean isTrain) throws IOException { 
        return new File(meta.getLocation() + getMnistFilename(meta, isTrain, isImg));
    }

    private static MappedByteBuffer getBuffer(FileInputStream stream, long fileLength) throws IOException {
        return stream.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, fileLength);
    }

    private static FileInputStream getStream(File file) throws FileNotFoundException {
        return new FileInputStream(file);
    }

    private static Integer convertToInt(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getInt();
    }

    /**
     * The file pointer is always moving forward in this case
     */
    private static byte[] loadBufferOfLength(MappedByteBuffer buffer, /*Integer offset,*/ Integer length) throws IOException {
        if (buffer == null) throw new IOException("Byte buffer has not been initialized yet!");
        byte[] bytes = new byte[length];
        buffer.get(bytes, 0/*offset*/, length);
        return bytes;
    }

//    private static JDataSetMeta.Header getMnistDataHeader(JDataSetMeta meta, Boolean isTrain) throws IOException {
//        for (JDataSetMeta.JTrainOrTest conf : (isTrain ? meta.getTrain() : meta.getTest())) {
//            return conf.getHeader();
//        }
//        return null;
//    }

    private static String getMnistFilename(JDataSetMeta meta, Boolean isTrain, Boolean isImg) throws IOException {
        for (JDataSetMeta.JTrainOrTest conf : (isTrain ? meta.getTrain() : meta.getTest())) {
            if (isImg) {
                if (Constants.IMAGE.equals(conf.getDataType())) {
                    return conf.getName();
                }
            } else {
                if (Constants.LABEL.equals(conf.getDataType())) {
                    return conf.getName();
                }
            }
        }
        throw new IOException("no image type found in train -> [ { type......");
    }
}
