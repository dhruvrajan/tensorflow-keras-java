package org.tensorflow.keras.datasets;

import org.tensorflow.data.NioTensorFrame;
import org.tensorflow.keras.utils.DataUtils;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.nd.FloatNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.utils.Tuple2;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;

import static org.tensorflow.nio.StaticApi.bufferOfFloats;
import static org.tensorflow.nio.StaticApi.ndArrayOf;

/**
 * Code based on example found at:
 * https://github.com/karllessard/models/tree/master/samples/languages/java/mnist/src/main/java/org/tensorflow/model/sample/mnist
 * <p>
 * Utility for downloading and using MNIST data with a local keras installation.
 */
public class NioMNIST {
    private static final int IMAGE_MAGIC = 2051;
    private static final int LABELS_MAGIC = 2049;
    private static final int OUTPUT_CLASSES = 10;
    private static final int IMG_WIDTH = 28;
    private static final int IMG_HEIGHT = 28;

    private static final String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    private static final String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";

    private static final String ORIGIN_BASE = "http://yann.lecun.com/exdb/mnist/";

    private static final String LOCAL_PREFIX = "datasets/mnist/";

    /**
     * Download MNIST dataset files to the local .keras/ directory.
     *
     * @throws IOException when the download fails
     */
    public static void download() throws IOException {
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES,
                "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609", DataUtils.Checksum.sha256);
        DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS,
                "fcdfeedb53b53c99384b2cd314206a08fdf6aa97070e19921427a179ea123d19", DataUtils.Checksum.sha256);
        DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES,
                "beb4b4806386107117295b2e3e08b4c16a6dfb4f001bfeb97bf25425ba1e08e4", DataUtils.Checksum.sha256);
        DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS,
                "986c5b8cbc6074861436f5581f7798be35c7c0025262d33b4df4c9ef668ec773", DataUtils.Checksum.sha256);
    }

    public static void main(String[] args) throws IOException {
        Tuple2<NioTensorFrame<Float>> nio =  NioMNIST.nioTensorFrames();

        System.out.println();
    }

    public static Tuple2<NioTensorFrame<Float>> nioTensorFrames() throws IOException {
        // Download MNIST files if they don't exist.
        NioMNIST.download();

        // Read data files into arrays
        FloatNdArray trainImages = readImagesToFloatBuffer(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES));
        FloatNdArray trainLabels = readLabelsToFloatBuffer(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS));
        FloatNdArray testImages = readImagesToFloatBuffer(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES));
        FloatNdArray testLabels = readLabelsToFloatBuffer(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS));

        // Create tensor frame
        NioTensorFrame<Float> train = new NioTensorFrame<>(trainImages, trainLabels);
        NioTensorFrame<Float> test = new NioTensorFrame<>(testImages, testLabels);

        // Return a pair of graph loaders; train and test sets
        return new Tuple2<>(train, test);
    }

    public static FloatNdArray readImagesToFloatBuffer(Path imagesPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath.toString())))) {

            if (inputStream.readInt() != IMAGE_MAGIC) {
                throw new IllegalArgumentException("Invalid Image Data File");
            }

            int numImages = inputStream.readInt();
            int rows = inputStream.readInt();
            int cols = inputStream.readInt();

            FloatDataBuffer images = bufferOfFloats(numImages * rows * cols);
            readImagesToFloatBuffer(inputStream, images, numImages);
            return ndArrayOf(images, Shape.make(numImages, rows, cols));
        }
    }

    public static FloatNdArray readLabelsToFloatBuffer(Path labelsPath) throws IOException {
        try (DataInputStream inputStream =
                     new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath.toString())))) {

            if (inputStream.readInt() != LABELS_MAGIC) {
                throw new IllegalArgumentException("Invalid Label Data File");
            }

            int numLabels = inputStream.readInt();

            FloatDataBuffer labels = bufferOfFloats(numLabels * OUTPUT_CLASSES);
            readLabelBytesToFloatBuffer(inputStream, labels, numLabels);
            return ndArrayOf(labels, Shape.make(numLabels, OUTPUT_CLASSES));
        }
    }

    public static void readLabelBytesToFloatBuffer(DataInputStream inputStream, FloatDataBuffer fb, int numLabels) throws IOException {
        // Read Bytes
        byte[] bytes = new byte[numLabels];
        inputStream.readFully(bytes);

        // Convert Bytes to Float Labels
        float[] floats = new float[numLabels * OUTPUT_CLASSES];
        for (int i = 0; i < numLabels; i++) {
            int label = bytes[i] & 0xFF;

            float[] labelOneHot = labelToOneHotVector(label);
            for (int j = 0; j < OUTPUT_CLASSES; j++) {
                floats[i + j] = labelOneHot[j];
            }
        }
        // Write floats to buffer
        fb.put(floats);
    }

    public static void readImagesToFloatBuffer(DataInputStream inputStream, FloatDataBuffer fb, int numImages) throws IOException {
        // Read Bytes
        byte[] bytes = new byte[numImages * IMG_WIDTH * IMG_HEIGHT];
        inputStream.readFully(bytes);

        // Convert Bytes to Floats
        float[] floats = new float[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            floats[i] = bytes[i] & 0xFF;
        }

        // Write floats to buffer
        fb.put(floats);
    }
    private static float[] labelToOneHotVector(int label) {
        float[] oneHot = new float[OUTPUT_CLASSES];
        oneHot[label] = 1.0f;
        return oneHot;
    }
}
