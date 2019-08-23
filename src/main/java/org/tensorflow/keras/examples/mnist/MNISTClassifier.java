package org.tensorflow.keras.examples.mnist;

import org.tensorflow.Shape;
import org.tensorflow.*;
import org.tensorflow.keras.data.Dataset;
import org.tensorflow.keras.datasets.MNISTLoader;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

public class MNISTClassifier implements Runnable {
  private static final int INPUT_SIZE = 28 * 28;

  private static final float LEARNING_RATE = 0.2f;
  private static final int FEATURES = 10;
  private static final int BATCH_SIZE = 100;

  public static void main(String[] args) {
    try (Graph graph = new Graph()) {
      MNISTClassifier mnist = new MNISTClassifier();
      mnist.run();
      //            BostonClassifier mnist = new BostonClassifier();
      //            mnist.run();
    }
  }

  public void run() {
    try (Graph graph = new Graph()) {

      Ops tf = Ops.create(graph);

      Placeholder<Float> images =
          tf.placeholder(Float.class, Placeholder.shape(Shape.make(-1, INPUT_SIZE)));
      Placeholder<Float> labels = tf.placeholder(Float.class);

      Variable<Float> weights = tf.variable(Shape.make(INPUT_SIZE, FEATURES), Float.class);
      Assign<Float> weightsInit =
          tf.assign(weights, tf.zeros(constArray(tf, INPUT_SIZE, FEATURES), Float.class));

      Variable<Float> biases = tf.variable(Shape.make(FEATURES), Float.class);
      Assign<Float> biasesInit =
          tf.assign(biases, tf.zeros(tf.constant(new int[] {FEATURES}), Float.class));

      Softmax<Float> softmax = tf.softmax(tf.add(tf.matMul(images, weights), biases));
      Mean<Float> crossEntropy =
          tf.mean(
              tf.neg(tf.reduceSum(tf.mul(labels, tf.log(softmax)), constArray(tf, 1))),
              constArray(tf, 0));

      Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
      Constant<Float> alpha = tf.constant(LEARNING_RATE);
      ApplyGradientDescent<Float> weightGradientDescent =
          tf.applyGradientDescent(weights, alpha, gradients.dy(0));
      ApplyGradientDescent<Float> biasGradientDescent =
          tf.applyGradientDescent(biases, alpha, gradients.dy(1));

      Operand<Long> predicted = tf.argMax(softmax, tf.constant(1));
      Operand<Long> expected = tf.argMax(labels, tf.constant(1));
      Operand<Float> accuracy =
          tf.mean(tf.cast(tf.equal(predicted, expected), Float.class), tf.constant(0));

      Operand first = tf.print(accuracy, Collections.singletonList(accuracy));
      Dataset data;
      try {
        data = MNISTLoader.loadData();
      } catch (IOException e) {
        throw new IllegalArgumentException("Could not load MNIST dataset");
      }
      try (Session session = new Session(graph)) {
        // Initialize weights

        session.runner().addTarget(weightsInit).addTarget(biasesInit).run();
        for (int epoch = 0; epoch < 10; epoch++) {

          // Train the graph
          int trainCount = 0;
          for (Dataset.Split batch : data.trainBatchIterator(BATCH_SIZE)) {
            try (Tensor<Float> imageBatch = Tensors.create(batch.X);
                Tensor<Float> labelBatch = Tensors.create(batch.y)) {
              //                        System.out.println("Train batch " + trainCount);
              session
                  .runner()
                  .addTarget(weightGradientDescent)
                  .addTarget(biasGradientDescent)
                  .feed(images.asOutput(), imageBatch)
                  .feed(labels.asOutput(), labelBatch)
                  .run();
            }

            trainCount++;
          }

          int testCount = 0;

          double acc = 0;
          //   Test the graph
          for (Dataset.Split batch : data.testBatchIterator(BATCH_SIZE)) {
            try (Tensor<Float> imageBatch = Tensors.create(batch.X);
                Tensor<Float> labelBatch = Tensors.create(batch.y);
                Tensor<?> value =
                    session
                        .runner()
                        .fetch(accuracy)
                        .feed(images.asOutput(), imageBatch)
                        .feed(labels.asOutput(), labelBatch)
                        .run()
                        .get(0)) {

              //                            System.out.println("Accuracy " + testCount + ": " +
              // value.floatValue());
              acc += value.floatValue();
            }

            testCount++;
          }

          System.out.println("Accuracy " + epoch + " " + acc / testCount);
        }
      }
    }
  }

  private static Operand<Integer> constArray(Ops tf, int... i) {
    return tf.constant(i);
  }
}
