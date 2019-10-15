package org.tensorflow.keras.models;

import org.tensorflow.*;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.layers.Input;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Sequential<T extends Number> extends Model<T> {
    private Input<T> firstLayer;
    private Optimizer<T> optimizer;
    private List<Layer<T>> layers;

    private Loss<T> loss;
    private List<Metric<T>> metrics;
;

    private List<Variable<T>> trainableVars;
    private List<Operand<T>> initializerOps;

    @SafeVarargs
    public Sequential(Class<T> dtype, Input<T> firstLayer, Layer<T>... layers) {
        super(dtype);
        this.firstLayer = firstLayer;
        this.layers = Arrays.asList(layers);
    }

    @SafeVarargs
    public static <T extends Number> Sequential of(Class<T> dtype, Input firstLayer, Layer... layers) {
        return new Sequential<>(dtype, firstLayer, layers);
    }

    public Sequential addLayer(Layer<T> layer) {
        layers.add(layer);
        return this;
    }

    @Override
    public Shape computeOutputShape(Shape inputShape) {
        throw new UnsupportedOperationException("Can't call computeOutputShape on Model");
    }

    @Override
    @SafeVarargs
    public final Operand<T> call(Ops tf, Operand<T>... inputs) {
        return this.call(tf, inputs[0]);
    }

    public Operand<T> call(Ops tf, Operand<T> in) {
        Operand<T> out = in;
        for (Layer<T> layer : this.layers) {
            out = layer.apply(tf, out);
        }
        return out;
    }

    @Override
    public void compile(Ops tf, Optimizer<T> optimizer, Loss<T> loss, List<Metric<T>> metrics) {
        this.loss = loss;
        this.metrics = metrics;
        this.optimizer = optimizer;

        // Targets for training loop
        this.trainableVars = new ArrayList<>();
        this.initializerOps = new ArrayList<>();

        // Build layers
        this.firstLayer.doBuild(tf, dtype);
        Shape inputShape = firstLayer.computeOutputShape();

        for (Layer<T> layer : layers) {
            layer.doBuild(tf, inputShape, dtype);
            this.trainableVars.addAll(layer.trainableWeights());
            this.initializerOps.addAll(layer.initializerOps());
            inputShape = layer.computeOutputShape(inputShape);
        }

        for (Metric<T> metric : metrics) {
            metric.doBuild(tf, null, dtype);
        }

        this.optimizer.build(tf);
        this.built = true;
    }

    @Override
    public void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, int epochs, int batchSize) {
        try (Session session = new Session(tf.scope().graph())) {
            runTrainingLoop(tf, session, train, epochs, batchSize, true);
            runPredictionLoop(tf, session, test, batchSize);
        }
    }
    private void runPredictionLoop(Ops tf, Session session, GraphLoader<T> data, int batchSize) {
        runTrainingLoop(tf, session, data, 1, batchSize, false);
    }

    private void runTrainingLoop(Ops tf, Session session, GraphLoader<T> data, int epochs, int batchSize, boolean training) {
        data.batch(batchSize);
        data.build(tf);
        Operand<T>[] dataOps = data.getBatchOperands();

        Session.Runner runner;
        Operand<T> XOp = dataOps[0];
        Operand<T> yOp = dataOps[1];

        // Compute Output / Loss / Accuracy
        Operand<T> yTrue = yOp;
        Operand<T> yPred = this.apply(tf, XOp);

        Operand<T> batchLoss = loss.apply(tf, yTrue, yPred);
        Operand<T> batchAccuracy = this.metrics.get(0).apply(tf, yTrue, yPred);

        List<Operand<T>> minimize = training ? optimizer.minimize(tf, batchLoss, this.trainableVars) : null;

        if (training) {
            runner = session.runner();

            // Run initializer ops
            for (Operand<T> op : this.initializerOps) {
                runner.addTarget(op);
            }

            runner.run();
        }


        for (int epoch = 0; epoch < epochs; epoch++) {
            float trainEpochAccuracy = 0;
            float trainEpochLoss = 0;

            // Load Batches
            for (int i = 0; i < data.numBatches(); i++) {
                runner = session.runner();
                data.feedSessionRunner(runner, i);

                if (training) {
                    for (Operand<T> op : minimize) {
                        runner.addTarget(op);
                    }
                }

                runner.fetch(batchLoss);
                runner.fetch(batchAccuracy);

                List<Tensor<?>> values = runner.run();
                try (Tensor<?> lossTensor = values.get(0);
                     Tensor<?> accuracyTensor = values.get(1)) {
                    trainEpochAccuracy += accuracyTensor.floatValue() / data.numBatches();
                    trainEpochLoss += lossTensor.floatValue() / data.numBatches();
                }
            }

            if (training) {
                System.out.println("Epoch " + epoch + " train accuracy: " + trainEpochAccuracy + "  loss: " + trainEpochLoss);
            } else {
                System.out.println("Test accuracy: " + trainEpochAccuracy + " loss: " + trainEpochLoss);
            }
        }

    }
}
