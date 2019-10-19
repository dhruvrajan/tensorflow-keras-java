package org.tensorflow.keras.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.callbacks.Callback;
import org.tensorflow.keras.layers.Input;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.logs.BatchBeginLogs;
import org.tensorflow.keras.logs.BatchEndLogs;
import org.tensorflow.keras.logs.EpochBeginLogs;
import org.tensorflow.keras.logs.EpochEndLogs;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

public class Sequential<T extends Number> extends Model<T> {
    private Input<T> firstLayer;
    private Optimizer<T> optimizer;
    private List<Layer<T>> layers;

    private Loss loss;
    private List<Metric> metrics;;

    private List<Variable<T>> trainableVars;
    private List<Operand<T>> initializerOps;

    @SafeVarargs
    public Sequential(Class<T> dtype, Input<T> firstLayer, Layer<T>... layers) {
        super(dtype);
        this.firstLayer = firstLayer;
        this.layers = Arrays.asList(layers);
    }

    @SafeVarargs
    public static <T extends Number> Sequential<T> of(Class<T> dtype, Input<T> firstLayer, Layer<T>... layers) {
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
    protected void build(Ops tf, Shape inputShape) {
        throw new UnsupportedOperationException("Cannot build a sequential model");
    }

    @Override
    public void compile(Ops tf, Optimizer<T> optimizer, Loss loss, List<Metric> metrics) {
        this.loss = loss;
        this.metrics = metrics;
        this.optimizer = optimizer;

        // Targets for training loop
        this.trainableVars = new ArrayList<>();
        this.initializerOps = new ArrayList<>();

        // Build layers
        this.firstLayer.build(tf, dtype);
        Shape inputShape = firstLayer.computeOutputShape();

        for (Layer<T> layer : layers) {
            layer.build(tf, inputShape, dtype);
            this.trainableVars.addAll(layer.trainableWeights());
            this.initializerOps.addAll(layer.initializerOps());
            inputShape = layer.computeOutputShape(inputShape);
        }

        this.optimizer.build(tf, dtype);
    }

    @Override
    public void fit(Ops tf, GraphLoader<T> train, GraphLoader<T> test, int epochs, int batchSize,
            List<Callback> callbacks) {
        try (Session session = new Session(tf.scope().graph())) {
            runTrainingLoop(tf, session, train, epochs, batchSize, callbacks, true);
            runPredictionLoop(tf, session, test, batchSize, callbacks);
        }
    }

    private void runPredictionLoop(Ops tf, Session session, GraphLoader<T> data, int batchSize, List<Callback> callbacks) {
        runTrainingLoop(tf, session, data, 1, batchSize, callbacks, false);
    }

    private void runTrainingLoop(Ops tf, Session session, GraphLoader<T> data, int epochs, int batchSize,
            List<Callback> callbacks, boolean training) {
        data.batch(batchSize);
        data.build(tf);
        Operand<T>[] dataOps = data.getBatchOperands();

        Session.Runner runner;
        Operand<T> XOp = dataOps[0];
        Operand<T> yOp = dataOps[1];

        // Compute Output / Loss / Accuracy
        Operand<T> yTrue = yOp;
        Operand<T> yPred = this.apply(tf, XOp);

        Operand<T> batchLoss = loss.apply(tf, getDtype(), yTrue, yPred);
        Operand<T> batchAccuracy = this.metrics.get(0).apply(tf, getDtype(), yTrue, yPred);

        List<Operand<T>> minimize = training ? optimizer.minimize(tf, batchLoss, this.trainableVars) : null;

        for (Callback clbk : callbacks) {
            clbk.onTrainBegin();
        }

        if (training) {
            runner = session.runner();

            // Run initializer ops
            for (Operand<T> op : this.initializerOps) {
                runner.addTarget(op);
            }

            runner.run();
        }

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (Callback clbk : callbacks) {
                clbk.onEpochBegin(epoch, new EpochBeginLogs());
            }

            float trainEpochAccuracy = 0;
            float trainEpochLoss = 0;

            // Load Batches
            for (int i = 0; i < data.numBatches(); i++) {

                for (Callback clbk : callbacks) {
                    BatchBeginLogs logs = new BatchBeginLogs();
                    logs.batchSize = batchSize;
                    clbk.onBatchBegin(i, logs);
                }

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
                try (Tensor<?> lossTensor = values.get(0); Tensor<?> accuracyTensor = values.get(1)) {
                    trainEpochAccuracy += accuracyTensor.floatValue() / data.numBatches();
                    trainEpochLoss += lossTensor.floatValue() / data.numBatches();

                    for (Callback clbk : callbacks) {
                        BatchEndLogs logs = new BatchEndLogs();
                        logs.batchAccuracy = accuracyTensor.floatValue();
                        logs.batchLoss = lossTensor.floatValue();
                        clbk.onBatchEnd(i, logs);
                    }
                }
            }

            if (training) {
                for (Callback clbk : callbacks) {
                    EpochEndLogs logs = new EpochEndLogs();
                    logs.trainAccuracy = trainEpochAccuracy;
                    logs.trainLoss = trainEpochLoss;
                    clbk.onEpochEnd(epoch, logs);
                }
            } else {
                for (Callback clbk : callbacks) {
                    EpochEndLogs logs = new EpochEndLogs();
                    logs.valAccuracy = trainEpochAccuracy;
                    logs.valLoss = trainEpochLoss;
                    clbk.onEpochEnd(epoch, logs);
                }
            }

            for (Callback clbk : callbacks) {
                clbk.onTrainEnd();
            }
        }

    }
}
