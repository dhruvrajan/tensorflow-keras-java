//package org.tensorflow.keras.datasets;
//
//import org.tensorflow.Graph;
//import org.tensorflow.Tensors;
//import org.tensorflow.data.GraphLoader;
//import org.tensorflow.data.GraphModeTensorFrame;
//import org.tensorflow.keras.utils.DataUtils;
//import org.tensorflow.keras.utils.Keras;
//import org.tensorflow.utils.Pair;
//
//import java.io.IOException;
//
//public class FashionMNIST {
//    private static String ORIGIN_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
//    private static String TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
//    private static String TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
//    private static String TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
//    private static String TEST_LABELS = "t10k-labels-idx1-ubyte.gz";
//
//    private static final String LOCAL_PREFIX = "datasets/fasion_mnist/";
//
//    /**
//     * Download MNIST dataset files to the local .keras/ directory.
//     *
//     * @throws IOException when the download fails
//     */
//    public static void download() throws IOException {
//        DataUtils.getFile(LOCAL_PREFIX + TRAIN_IMAGES, ORIGIN_BASE + TRAIN_IMAGES,
//                "8d4fb7e6c68d591d4c3dfef9ec88bf0d", DataUtils.Checksum.md5);
//        DataUtils.getFile(LOCAL_PREFIX + TRAIN_LABELS, ORIGIN_BASE + TRAIN_LABELS,
//                "25c81989df183df01b3e8a0aad5dffbe", DataUtils.Checksum.md5);
//        DataUtils.getFile(LOCAL_PREFIX + TEST_IMAGES, ORIGIN_BASE + TEST_IMAGES,
//                "bef4ecab320f06d8554ea6380940ec79", DataUtils.Checksum.md5);
//        DataUtils.getFile(LOCAL_PREFIX + TEST_LABELS, ORIGIN_BASE + TEST_LABELS,
//                "bb300cfdad3c16e7a12a480ee83cd310", DataUtils.Checksum.md5);
//    }
//
//    public static Pair<GraphModeTensorFrame<Float>, GraphModeTensorFrame<Float>> graphLoaders(Graph graph) throws IOException {
//        // Download MNIST files if they don't exist.
//        FashionMNIST.download();
//
//        // Read data files into arrays
////        float[][] trainImages = MNIST.readImages(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString());
//        float[][] trainLabels = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString());
//        float[][] testImages = MNIST.readImages(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString());
//        float[][] testLabels = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString());
//
//        // Return a pair of graph loaders; train and test sets
//        return new Pair<>(
//                new GraphModeTensorFrame<>(
//                        graph, Float.class, Tensors.create(trainImages), Tensors.create(trainLabels)),
//                new GraphModeTensorFrame<>(
//                        graph, Float.class, Tensors.create(testImages), Tensors.create(testLabels)));
//    }
//
//    public static Pair<GraphModeTensorFrame<Float>, GraphModeTensorFrame<Float>> graphLoaders2D(Graph graph) throws IOException {
//        // Download MNIST files if they don't exist.
//        FashionMNIST.download();
//
//        // Read data files into arrays
//        float[][][] trainImages = MNIST.readImages2D(Keras.kerasPath(LOCAL_PREFIX, TRAIN_IMAGES).toString());
//        float[][] trainLabels = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX, TRAIN_LABELS).toString());
//        float[][][] testImages = MNIST.readImages2D(Keras.kerasPath(LOCAL_PREFIX, TEST_IMAGES).toString());
//        float[][] testLabels = MNIST.readLabelsOneHot(Keras.kerasPath(LOCAL_PREFIX + TEST_LABELS).toString());
//
//        // Return a pair of graph loaders; train and test sets
//        return new Pair<>(
//                new GraphModeTensorFrame<>(
//                        graph, Float.class, Tensors.create(trainImages), Tensors.create(trainLabels)),
//                new GraphModeTensorFrame<>(
//                        graph, Float.class, Tensors.create(testImages), Tensors.create(testLabels)));
//    }
//
//}
