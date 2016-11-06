package deepdsl.cudnn;

  
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.*;
import jcuda.runtime.JCuda;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
import static jcuda.jcudnn.cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnLRNMode.CUDNN_LRN_CROSS_CHANNEL_DIM1;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE;
import static jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
 
/**
 * Porting of the JCudnn sample running against Mnist dataset from http://www.jcuda.org/samples/MnistJCudnn.java<br>
 * <br>
 * This sample expects the Mnist dataset to be present in the "dataset/mnist/jcudnn" subdirectory.
 */
public class MnistJCudnn
{
    private static final int IMAGE_H = 28;
    private static final int IMAGE_W = 28;
 
    private static final String first_image = "one_28x28.pgm";
    private static final String second_image = "three_28x28.pgm";
    private static final String third_image = "five_28x28.pgm";
    private static final String dataDirectory = "dataset/mnist/jcudnn";
 
    private static final String conv1_bin = "conv1.bin";
    private static final String conv1_bias_bin = "conv1.bias.bin";
    private static final String conv2_bin = "conv2.bin";
    private static final String conv2_bias_bin = "conv2.bias.bin";
    private static final String ip1_bin = "ip1.bin";
    private static final String ip1_bias_bin = "ip1.bias.bin";
    private static final String ip2_bin = "ip2.bin";
    private static final String ip2_bias_bin = "ip2.bias.bin";
 
    public static void main(String args[])
    {
        JCuda.setExceptionsEnabled(true);
        JCudnn.setExceptionsEnabled(true);
        JCublas2.setExceptionsEnabled(true);
 
        int version = (int) cudnnGetVersion();
        System.out.printf("cudnnGetVersion() : %d , " +
            "CUDNN_VERSION from cudnn.h : %d\n",
            version, CUDNN_VERSION);
 
        System.out.println("Creating network and layers...");
        Network mnist = new Network();
       
        System.out.println("Classifying...");
        int i1 = mnist.classifyExample(dataDirectory + first_image);
        int i2 = mnist.classifyExample(dataDirectory + second_image);
 
        mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
        int i3 = mnist.classifyExample(dataDirectory + third_image);
       
        System.out.println(
            "\nResult of classification: " + i1 + " " + i2 + " " + i3);
        if (i1 != 1 || i2 != 3 || i3 != 5)
        {
            System.out.println("\nTest failed!\n");
        }
        else
        {
            System.out.println("\nTest passed!\n");
        }
        mnist.destroy();
    }
 
   
    // The CUDNN_TENSOR_NCHW tensor format specifies that the
    // data is laid out in the following order:
    // image, features map, rows, columns.
    private static class TensorLayout
    {
        int n;
        int c;
        int h;
        int w;
    }
   
    private static class Layer
    {
        int inputs;
        int outputs;
        int kernel_dim;
        Pointer data_d;
        Pointer bias_d;
 
        Layer(int inputs, int outputs, int kernelDim,
            String weightsFileName, String biasFileName)
        {
            this.inputs = inputs;
            this.outputs = outputs;
            this.kernel_dim = kernelDim;
 
            String weightsPath = dataDirectory + weightsFileName;
            String biasPath = dataDirectory + biasFileName;
 
            float weights[] = readBinaryFileUnchecked(weightsPath);
            data_d = createDevicePointer(weights);
 
            float bias[] = readBinaryFileUnchecked(biasPath);
            bias_d = createDevicePointer(bias);
        }
 
        void destroy()
        {
            cudaFree(data_d);
            cudaFree(bias_d);
        }
    };
   
    private static class Network
    {
        private int convAlgorithm;
        private cudnnHandle cudnnHandle;
        private cudnnTensorDescriptor srcTensorDesc;
        private cudnnTensorDescriptor dstTensorDesc;
        private cudnnTensorDescriptor biasTensorDesc;
        private cudnnFilterDescriptor filterDesc;
        private cudnnConvolutionDescriptor convDesc;
        private cudnnPoolingDescriptor poolingDesc;
        private cudnnActivationDescriptor activDesc;
        private cudnnLRNDescriptor normDesc;
        private cublasHandle cublasHandle;
       
        private final Layer conv1;
        private final Layer conv2;
        private final Layer ip1;
        private final Layer ip2;
 
        Network()
        {
            convAlgorithm = -1;
            createHandles();
           
            conv1 = new Layer(1, 20, 5, conv1_bin, conv1_bias_bin);
            conv2 = new Layer(20, 50, 5, conv2_bin, conv2_bias_bin);
            ip1 = new Layer(800, 500, 1, ip1_bin, ip1_bias_bin);
            ip2 = new Layer(500, 10, 1, ip2_bin, ip2_bias_bin);
        }
 
        void createHandles()
        {
            cudnnHandle = new cudnnHandle();
            srcTensorDesc = new cudnnTensorDescriptor();
            dstTensorDesc = new cudnnTensorDescriptor();
            biasTensorDesc = new cudnnTensorDescriptor();
            filterDesc = new cudnnFilterDescriptor();
            convDesc = new cudnnConvolutionDescriptor();
            poolingDesc = new cudnnPoolingDescriptor();
            activDesc = new cudnnActivationDescriptor();
            normDesc = new cudnnLRNDescriptor();
 
            cudnnCreate(cudnnHandle);
            cudnnCreateTensorDescriptor(srcTensorDesc);
            cudnnCreateTensorDescriptor(dstTensorDesc);
            cudnnCreateTensorDescriptor(biasTensorDesc);
            cudnnCreateFilterDescriptor(filterDesc);
            cudnnCreateConvolutionDescriptor(convDesc);
            cudnnCreatePoolingDescriptor(poolingDesc);
            cudnnCreateActivationDescriptor(activDesc);
            cudnnCreateLRNDescriptor(normDesc);
 
            cublasHandle = new cublasHandle();
            cublasCreate(cublasHandle);
        }
 
        void destroy()
        {
            cudnnDestroyLRNDescriptor(normDesc);
            cudnnDestroyPoolingDescriptor(poolingDesc);
            cudnnDestroyActivationDescriptor(activDesc);
            cudnnDestroyConvolutionDescriptor(convDesc);
            cudnnDestroyFilterDescriptor(filterDesc);
            cudnnDestroyTensorDescriptor(srcTensorDesc);
            cudnnDestroyTensorDescriptor(dstTensorDesc);
            cudnnDestroyTensorDescriptor(biasTensorDesc);
            cudnnDestroy(cudnnHandle);
 
            cublasDestroy(cublasHandle);
           
            conv1.destroy();
            conv2.destroy();
            ip1.destroy();
            ip2.destroy();
        }
 
 
        void setConvolutionAlgorithm(int algo)
        {
            convAlgorithm = algo;
        }
 
        void addBias(cudnnTensorDescriptor dstTensorDesc,
            Layer layer, int c, Pointer data)
        {
            cudnnSetTensor4dDescriptor(biasTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, c, 1, 1);
            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(1.0f);
            cudnnAddTensor(cudnnHandle, alpha,
                biasTensorDesc, layer.bias_d, beta, dstTensorDesc, data);
        }
 
        void fullyConnectedForward(Layer ip, TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            if (t.n != 1)
            {
                System.out.println("Not Implemented");
                return;
            }
            int dim_x = t.c * t.h * t.w;
            int dim_y = ip.outputs;
            resize(dim_y, dstData);
 
            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(1.0f);
 
            // place bias into dstData
            cudaMemcpy(dstData, ip.bias_d, dim_y * Sizeof.FLOAT,
                cudaMemcpyDeviceToDevice);
 
            cublasSgemv(cublasHandle, CUBLAS_OP_T,
                dim_x, dim_y, alpha, ip.data_d,
                dim_x, srcData, 1, beta, dstData, 1);
 
            t.h = 1;
            t.w = 1;
            t.c = dim_y;
        }
 
        void convoluteForward(Layer conv, TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            int algo = 0; // cudnnConvolutionFwdAlgo_t
 
            cudnnSetTensor4dDescriptor(srcTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
 
            int tensorDims = 4;
            int tensorOuputDimA[] = { t.n, t.c, t.h, t.w };
            int filterDimA[] = {
                conv.outputs, conv.inputs,
                conv.kernel_dim, conv.kernel_dim };
           
            cudnnSetFilterNdDescriptor(filterDesc,
                CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, tensorDims, filterDimA);
 
            int convDims = 2;
            int padA[] = { 0, 0 };
            int filterStrideA[] = { 1, 1 };
            int upscaleA[] = { 1, 1 };
            cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA,
                filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
 
            // find dimension of convolution output
            cudnnGetConvolutionNdForwardOutputDim(convDesc,
                srcTensorDesc, filterDesc,
                tensorDims, tensorOuputDimA);
            t.n = tensorOuputDimA[0];
            t.c = tensorOuputDimA[1];
            t.h = tensorOuputDimA[2];
            t.w = tensorOuputDimA[3];
 
            cudnnSetTensor4dDescriptor(dstTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
 
            if (convAlgorithm < 0)
            {
                int algoArray[] = { -1 };
               
                // Choose the best according to the preference
                System.out.println(
                    "Testing cudnnGetConvolutionForwardAlgorithm ...");
                cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc,
                    filterDesc, convDesc, dstTensorDesc,
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algoArray);
                algo = algoArray[0];
 
                System.out.println("Fastest algorithm is Algo " + algo);
                convAlgorithm = algo;
 
                // New way of finding the fastest config
                // Setup for findFastest call
                System.out.println(
                    "Testing cudnnFindConvolutionForwardAlgorithm ...");
                int requestedAlgoCount = 5;
                int returnedAlgoCount[] = new int[1];
                cudnnConvolutionFwdAlgoPerf results[] =
                    new cudnnConvolutionFwdAlgoPerf[requestedAlgoCount];
                cudnnFindConvolutionForwardAlgorithm(cudnnHandle,
                    srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                    requestedAlgoCount, returnedAlgoCount, results);
                for (int algoIndex = 0; algoIndex < returnedAlgoCount[0]; ++algoIndex)
                {
                    System.out.printf(
                        "    %s for Algo %d (%s): %f time requiring %d memory\n",
                        cudnnGetErrorString(results[algoIndex].status),
                        results[algoIndex].algo,
                        cudnnConvolutionFwdAlgo.stringFor(results[algoIndex].algo),
                        results[algoIndex].time, results[algoIndex].memory);
                }
            }
            else
            {
                algo = convAlgorithm;
                if (algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
                {
                    System.out.println("Using FFT for convolution");
                }
            }
 
            resize(t.n * t.c * t.h * t.w, dstData);
            long sizeInBytesArray[] = { 0 };
            Pointer workSpace = new Pointer();
            cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                algo, sizeInBytesArray);
            long sizeInBytes = sizeInBytesArray[0];
            if (sizeInBytes != 0)
            {
                cudaMalloc(workSpace, sizeInBytes);
            }
           
            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(0.0f);
            cudnnConvolutionForward(cudnnHandle, alpha, srcTensorDesc,
                srcData, filterDesc, conv.data_d, convDesc, algo,
                workSpace, sizeInBytes, beta, dstTensorDesc, dstData);
            addBias(dstTensorDesc, conv, t.c, dstData);
            if (sizeInBytes != 0)
            {
                cudaFree(workSpace);
            }
        }
 
        void poolForward(TensorLayout t, Pointer srcData,
            Pointer dstData)
        {
            int poolDims = 2;
            int windowDimA[] = { 2, 2 };
            int paddingA[] = { 0, 0 };
            int strideA[] = { 2, 2 };
            cudnnSetPoolingNdDescriptor(poolingDesc,
                CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, poolDims, windowDimA,
                paddingA, strideA);
 
            cudnnSetTensor4dDescriptor(srcTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
 
            int tensorDims = 4;
            int tensorOuputDimA[] = { t.n, t.c, t.h, t.w };
            cudnnGetPoolingNdForwardOutputDim(
                poolingDesc, srcTensorDesc,
                tensorDims, tensorOuputDimA);
            t.n = tensorOuputDimA[0];
            t.c = tensorOuputDimA[1];
            t.h = tensorOuputDimA[2];
            t.w = tensorOuputDimA[3];
 
            cudnnSetTensor4dDescriptor(dstTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
 
            resize(t.n * t.c * t.h * t.w, dstData);
            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(0.0f);
            cudnnPoolingForward(cudnnHandle, poolingDesc,
                alpha, srcTensorDesc, srcData, beta,
                dstTensorDesc, dstData);
        }
 
        void softmaxForward(TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            resize(t.n * t.c * t.h * t.w, dstData);
 
            cudnnSetTensor4dDescriptor(srcTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
            cudnnSetTensor4dDescriptor(dstTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
 
            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(0.0f);
            cudnnSoftmaxForward(cudnnHandle,
                CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                alpha, srcTensorDesc, srcData,
                beta, dstTensorDesc, dstData);
        }
 
        void lrnForward(TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            int lrnN = 5;
            double lrnAlpha, lrnBeta, lrnK;
            lrnAlpha = 0.0001;
            lrnBeta = 0.75;
            lrnK = 1.0;
            cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
 
            resize(t.n * t.c * t.h * t.w, dstData);
 
            cudnnSetTensor4dDescriptor(srcTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
            cudnnSetTensor4dDescriptor(dstTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
 
            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(0.0f);
            cudnnLRNCrossChannelForward(cudnnHandle, normDesc,
                CUDNN_LRN_CROSS_CHANNEL_DIM1,
                alpha, srcTensorDesc, srcData,
                beta, dstTensorDesc, dstData);
        }
 
       
        void activationForward(TensorLayout t,
            Pointer srcData, Pointer dstData)
        {
            cudnnSetActivationDescriptor(activDesc,
                CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
           
            resize(t.n * t.c * t.h * t.w, dstData);
 
            cudnnSetTensor4dDescriptor(srcTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
            cudnnSetTensor4dDescriptor(dstTensorDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                t.n, t.c, t.h, t.w);
 
            Pointer alpha = pointerTo(1.0f);
            Pointer beta = pointerTo(0.0f);
            cudnnActivationForward(cudnnHandle, activDesc,
                alpha, srcTensorDesc, srcData,
                beta, dstTensorDesc, dstData);
        }
 
       
        int classifyExample(String imageFileName)
        {
            TensorLayout t = new TensorLayout();
            Pointer srcData = new Pointer();
            Pointer dstData = new Pointer();
 
            float imgData_h[] = readImageDataUnchecked(imageFileName);
 
            System.out.println("Performing forward propagation ...");
 
            cudaMalloc(srcData, IMAGE_H * IMAGE_W * Sizeof.FLOAT);
            cudaMemcpy(srcData, Pointer.to(imgData_h), IMAGE_H * IMAGE_W
                * Sizeof.FLOAT, cudaMemcpyHostToDevice);
 
            t.n = 1;
            t.c = 1;
            t.h = IMAGE_H;
            t.w = IMAGE_W;
            convoluteForward(conv1, t, srcData, dstData);
            poolForward(t, dstData, srcData);
 
            convoluteForward(conv2, t, srcData, dstData);
            poolForward(t, dstData, srcData);
 
            fullyConnectedForward(ip1, t, srcData, dstData);
            activationForward(t, dstData, srcData);
            lrnForward(t, srcData, dstData);
 
            fullyConnectedForward(ip2, t, dstData, srcData);
            softmaxForward(t, srcData, dstData);
 
            int max_digits = 10;
            float result[] = new float[max_digits];
            cudaMemcpy(Pointer.to(result), dstData,
                max_digits * Sizeof.FLOAT,
                cudaMemcpyDeviceToHost);
            int id = 0;
            for (int i = 1; i < max_digits; i++)
            {
                if (result[id] < result[i])
                    id = i;
            }
 
            System.out.println("Resulting weights from Softmax:");
            printDeviceVector(t.n * t.c * t.h * t.w, dstData);
 
            cudaFree(srcData);
            cudaFree(dstData);
            return id;
        }
    }
 
 
   
    //========================================================================
    // I/O utility methods
   
    private static float[] readBinaryFile(String fileName) throws IOException
    {
        FileInputStream fis = new FileInputStream(new File(fileName));
        byte data[] = readFully(fis);
        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        float result[] = new float[fb.capacity()];
        fb.get(result);
        return result;
    }
 
    private static float[] readBinaryFileUnchecked(String fileName)
    {
        try
        {
            return readBinaryFile(fileName);
        }
        catch (IOException e)
        {
            cudaDeviceReset();
            e.printStackTrace();
            System.exit(-1);
            return null;
        }
    }
 
    private static byte[] readFully(InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[1024];
        while (true)
        {
            int n = inputStream.read(buffer);
            if (n < 0)
            {
                break;
            }
            baos.write(buffer, 0, n);
        }
        byte data[] = baos.toByteArray();
        return data;
    }
 
    @SuppressWarnings("deprecation")
    private static byte[] readBinaryPortableGraymap8bitData(
        InputStream inputStream) throws IOException
    {
        DataInputStream dis = new DataInputStream(inputStream);
        String line = null;
        boolean firstLine = true;
        Integer width = null;
        Integer maxBrightness = null;
        while (true)
        {
            // The DataInputStream#readLine is deprecated,
            // but for ASCII input, it is safe to use it
            line = dis.readLine();
            if (line == null)
            {
                break;
            }
            line = line.trim();
            if (line.startsWith("#"))
            {
                continue;
            }
            if (firstLine)
            {
                firstLine = false;
                if (!line.equals("P5"))
                {
                    throw new IOException(
                        "Data is not a binary portable " +
                        "graymap (P5), but " + line);
                }
                else
                {
                    continue;
                }
            }
            if (width == null)
            {
                String tokens[] = line.split(" ");
                if (tokens.length < 2)
                {
                    throw new IOException(
                        "Expected dimensions, found " + line);
                }
                width = parseInt(tokens[0]);
            }
            else if (maxBrightness == null)
            {
                maxBrightness = parseInt(line);
                if (maxBrightness > 255)
                {
                    throw new IOException(
                        "Only 8 bit values supported. " +
                        "Maximum value is " + maxBrightness);
                }
                break;
            }
        }
        byte data[] = readFully(inputStream);
        return data;
    }
 
    private static Integer parseInt(String s) throws IOException
    {
        try
        {
            return Integer.parseInt(s);
        }
        catch (NumberFormatException e)
        {
            throw new IOException(e);
        }
    }
 
    private static float[] readImageData(String fileName) throws IOException
    {
        InputStream is = new FileInputStream(new File(fileName));
        byte data[] = readBinaryPortableGraymap8bitData(is);
        float imageData[] = new float[data.length];
        for (int i = 0; i < data.length; i++)
        {
            imageData[i] = (((int) data[i]) & 0xff) / 255.0f;
        }
        return imageData;
    }
 
    private static float[] readImageDataUnchecked(String fileName)
    {
        try
        {
            return readImageData(fileName);
        }
        catch (IOException e)
        {
            cudaDeviceReset();
            e.printStackTrace();
            System.exit(-1);
            return null;
        }
    }
   
    //========================================================================
    // utility methods
   
    private static Pointer createDevicePointer(float data[])
    {
        int size = data.length * Sizeof.FLOAT;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, size);
        cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice);
        return deviceData;
    }
 
    private static void resize(int numberOfFloatElements, Pointer data)
    {
        cudaFree(data);
        cudaMalloc(data, numberOfFloatElements * Sizeof.FLOAT);
    }
   
    private static Pointer pointerTo(float value)
    {
        return Pointer.to(new float[] { value });
    }
 
   
   
    //========================================================================
    // debugging utility methods
   
    private static void printDeviceVector(int size, Pointer d)
    {
        float h[] = new float[size];
        cudaDeviceSynchronize();
        cudaMemcpy(Pointer.to(h), d, size * Sizeof.FLOAT,
            cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++)
        {
            System.out.print(h[i] + " ");
        }
        System.out.println();
    }
   
   
}