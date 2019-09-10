package org.tensorflow.keras.examples.mnist;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
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
import org.tensorflow.keras.optimizers.GradientDescentOptimizer;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.utils.Pair;

import java.io.IOException;

public class MNISTTensorFrameClassifier implements Runnable {
  private static final int INPUT_SIZE = 28 * 28;

  private static final float LEARNING_RATE = 0.2f;
  private static final int FEATURES = 10;
  private static final int BATCH_SIZE = 600;
  private static final int EPOCHS = 10;

  public static void main(String[] args) {
    MNISTTensorFrameClassifier mnist = new MNISTTensorFrameClassifier();
    mnist.run();

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
      Optimizer<Float> optimizer = new GradientDescentOptimizer(LEARNING_RATE);

      // Compile Model
      train.batch(BATCH_SIZE);
      train.build(tf);

      test.batch(BATCH_SIZE);
      test.build(tf);

      inputLayer.build(tf);
      denseLayer.build(tf, inputLayer.computeOutputShape());


      // Fit Model
      try (Session session = new Session(graph)) {
        Session.Runner runner = session.runner();

        // Run initializer ops
        for (Operand<Float> op : denseLayer.initializerOps()) {
          runner.addTarget(op);
        }

        runner.run();

        Placeholder<Float>[] trainPlaceholders = train.getPlaceholders();
        Placeholder<Float>[] testPlaceholders = test.getPlaceholders();

        Tensor<Float>[] trainTensors = train.getTensors();
        Tensor<Float>[] testTensors = test.getTensors();

        try (Tensor<Float> trainXTensor = trainTensors[0];
             Tensor<Float> trainYTensor = trainTensors[1];
             Tensor<Float> testXTensor = testTensors[0];
             Tensor<Float> testYTensor = testTensors[1]) {

          // Run Training Loop
          for (int epoch = 0; epoch < EPOCHS; epoch++) {
            double trainEpochAccuracy = 0;

            int count = 0;
            for (Operand<Float>[] operands :
                (Iterable<Operand<Float>[]>) () -> train.getBatchOps(tf)) {

              runner = session.runner();

              Operand<Float> XOp = operands[0];
              Operand<Float> yOp = operands[1];

              runner.feed(trainPlaceholders[0].asOutput(), trainXTensor);
              runner.feed(trainPlaceholders[1].asOutput(), trainYTensor);

              // Compute Output / Loss / Accuracy
              Operand<Float> yTrue = yOp;
              Operand<Float> yPred = denseLayer.apply(tf, XOp);

              Operand<Float> batchLoss = loss.apply(tf, yTrue, yPred);
              Operand<Float> batchAccuracy = accuracy.apply(tf, yTrue, yPred);

              for (Operand<Float> op :
                  optimizer.minimize(tf, batchLoss, denseLayer.trainableWeights())) {
                runner.addTarget(op);
              }

              try (Tensor<?> value = runner.fetch(batchAccuracy).run().get(0)) {
                trainEpochAccuracy += value.floatValue() / train.numBatches();
                //              System.out.println("Train Batch Accuracy is " + value.floatValue());
              }

              count += 1;
              System.out.println("Count = " + count);
            }

            System.out.println(">>> Train Epoch Accuracy is" + trainEpochAccuracy);
          }


          // Get Test Accuracy
          double testEpochAccuracy = 0;
          for (Operand<Float>[] operands : (Iterable<Operand<Float>[]>) () -> test.getBatchOps(tf)) {
            runner = session.runner();

            Operand<Float> XOp = operands[0];
            Operand<Float> yOp = operands[1];


            runner.feed(testPlaceholders[0].asOutput(), testXTensor);
            runner.feed(testPlaceholders[1].asOutput(), testYTensor);

            // Compute Output / Loss / Accurcy
            Operand<Float> yTrue = yOp;
            Operand<Float> yPred = denseLayer.apply(tf, XOp);

            Operand<Float> batchAccuracy = accuracy.apply(tf, yPred, yTrue);

            try (Tensor<?> value = runner.fetch(batchAccuracy).run().get(0)) {
              testEpochAccuracy += value.floatValue() / test.numBatches();
            }
          }

          System.out.println(">>> Test Epoch Accurcy is " + testEpochAccuracy);
        }
      }
    }
  }
}