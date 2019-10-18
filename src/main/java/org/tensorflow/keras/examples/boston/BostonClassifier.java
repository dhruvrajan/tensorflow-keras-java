// package io.gitlab.keras.examples.mnist.naive;
//
//
// import io.gitlab.keras.datasets.BostonHousing;
// import io.gitlab.keras.data.Dataset;
// import org.tensorflow.Shape;
// import org.tensorflow.*;
// import org.tensorflow.op.Ops;
// import org.tensorflow.op.core.*;
// import java.io.IOException;
// import java.util.List;
//

//
// public class BostonClassifier implements Runnable {
//    private static final int FEATURES = 13;
//    private static final int BATCH_SIZE = 3;
//    private static final float LEARNING_RATE = 0.2f;
//
//
//
//    public static void main(String[] args) {
//        try (Graph graph = new Graph()) {
//            BostonClassifier mnist = new BostonClassifier();
//            mnist.run();
//        }
//    }
//
//
//    public void run() {
//        try(Graph graph = new Graph()) {
//            Ops tf = Ops.create(graph);
//            Placeholder<Float> X = tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1,
// FEATURES)));
//            Placeholder<Float> y = tf.placeholder(Float.class);
//
//            Variable<Float> theta = tf.variable(Shape.make(FEATURES, BATCH_SIZE), Float.class);
//            Assign<Float> thetaInit = tf.assign(theta, tf.zeros(constArray(tf, FEATURES,
// BATCH_SIZE), Float.class));
//
//            MatMul<Float> yPred = tf.matMul(X, theta);
//            Sub<Float> error = tf.sub(yPred, y);
//            //tf.print(yPred, Arrays.asList(tf.shape(yPred), tf.shape(y)));
//
//
//
//            tf.reduceMean(tf.square(error), tf.constant(0));
//            Mul<Float> gradients = tf.mul(tf.constant((float) 2.0/FEATURES),
// tf.matMul(tf.transpose(X, tf.constant(new int[] {1, 0})), error));
//            ApplyGradientDescent<Float> applyGradientDescent =
// tf.applyGradientDescent(theta,tf.constant((float) 1), gradients);
//
//            Dataset batches;
//            try {
//                batches = BostonHousing.loadData((float) 0.2);
//            } catch (IOException e) {
//                throw new IllegalArgumentException("Could not load MNIST dataset.");
//            }
//
//            List<float[][]> TRAIN_IMAGE_PATH = batches.XTrain;
//            List<float[]> TRAIN_LABEL_PATH = batches.YTrain;
//            List<float[][]> TEST_IMAGE_PATH = batches.XTrain;
//            List<float[]> TEST_LABEL_PATH = batches.YTest;
//
//            try (Session session = new Session(graph)) {
//                session.runner()
//                        .addTarget(thetaInit)
//                        .run();
//
//
//                for (int i = 0; i < TRAIN_IMAGE_PATH.size(); i ++) {
//                    try (Tensor<Float> imageBatch = Tensors.create(TRAIN_IMAGE_PATH.get(i));
//                         Tensor<Float> labelBatch = Tensors.create(TRAIN_LABEL_PATH.get(i));
//                         Tensor<?> value = session.runner()
//                                 .addTarget(applyGradientDescent)
//                                 .fetch(error)
//                                 .feed(X.asOutput(), imageBatch)
//                                 .feed(y.asOutput(), labelBatch)
//                                 .run()
//                                 .get(0)) {
//                        //System.out.println("Accuracy " + i + ": " + va);
//                        System.out.println("hey");
//
//                    }
//                }
//            }
//        }
//    }
//
//    private static Operand<Integer> constArray(Ops tf, int... i) {
//        return tf.constant(i);
//    }
// }
