package org.tensorflow.keras.examples.mnist;

import org.tensorflow.*;
import org.tensorflow.data.GraphModeTensorFrame;
import org.tensorflow.data.TensorFrame;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.MNISTLoader;
import org.tensorflow.keras.layers.Dense;
import org.tensorflow.keras.layers.InputLayer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.utils.Pair;

import java.io.IOException;

public class MNISTTensorFrameClassifier implements Runnable {
  private static final int INPUT_SIZE = 28 * 28;

  private static final float LEARNING_RATE = 0.2f;
  private static final int FEATURES = 10;
  private static final int BATCH_SIZE = 100;
  private static final int EPOCHS = 10;

  public static void main(String[] args) {
    try (Graph graph = new Graph()) {
      MNISTTensorFrameClassifier mnist = new MNISTTensorFrameClassifier();
      mnist.run();
    }
  }

  public void run() {
    try (Graph graph = new Graph()) {

      Ops tf = Ops.create(graph);

      // Load MNIST Dataset
      Pair<TensorFrame<Float>, TensorFrame<Float>> data;
      try {
        data = MNISTLoader.loadData();
      } catch (IOException e) {
        throw new IllegalArgumentException("Could not load MNIST dataset.");
      }

      GraphModeTensorFrame<Float> train = (GraphModeTensorFrame<Float>) data.first();
      GraphModeTensorFrame<Float> test = (GraphModeTensorFrame<Float>) data.second();

      InputLayer inputLayer = InputLayer.create(INPUT_SIZE);
      Dense denseLayer = Dense.options().setActivation(Activations.softmax).create(FEATURES);

      Loss loss = Losses.select(Losses.softmax_crossentropy);
      Metric accuracy = Metrics.select(Metrics.accuracy);
      Optimizer<Float> optimizer = Optimizers.select(Optimizers.sgd);

      // Compile Model
      train.build(tf);
      test.build(tf);

      inputLayer.build(tf);
      denseLayer.build(tf, inputLayer.computeOutputShape());
      loss.build(tf, Shape.make(-1, FEATURES));

      // Fit Model
      try (Session session = new Session(graph)) {
        Session.Runner runner = session.runner();

        // Run initializer ops
        for (Operand<Float> op : denseLayer.initializerOps()) {
          runner.addTarget(op);
        }

        runner.run();

        Placeholder<Float>[] trainPlaceholders = train.getPlaceholders();

        // Run Training Loop
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
          for (Pair<Tensor<Float>[], Operand<Float>[]> batch :
              (Iterable<Pair<Tensor<Float>[], Operand<Float>[]>>)
                  () -> train.getBatchTensorsAndOps(tf)) {

            runner = session.runner();

            Tensor<Float>[] tensors = batch.first();
            Operand<Float>[] operands = batch.second();

            Operand<Float> XOp = operands[0];
            Operand<Float> yOp = operands[1];

            // Get Batches
            for (int i = 0; i < tensors.length; i++) {
              runner.feed(trainPlaceholders[i].asOutput(), tensors[i]);
            }

            // Compute Output / Loss / Accuracy
            Operand<Float> yTrue = yOp;
            Operand<Float> yPred = denseLayer.apply(tf, XOp);

            Operand<Float> batchLoss = loss.apply(tf, yPred, yTrue);
            Operand<Float> batchAccuracy = accuracy.apply(tf, yPred, yTrue);

            for (Operand<Float> op :
                optimizer.minimize(tf, batchLoss, denseLayer.trainableWeights())) {
              runner.addTarget(op);
            }

            runner.feed(XOp.asOutput(), tensors[0]);
            runner.feed(yOp.asOutput(), tensors[1]);

            try (Tensor<?> value = runner.fetch(batchAccuracy).run().get(0)) {
              System.out.println("Batch Accuracy is " + value);
            }
          }
        }
      }
    }
  }
}