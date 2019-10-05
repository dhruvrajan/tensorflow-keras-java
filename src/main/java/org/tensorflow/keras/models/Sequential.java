package org.tensorflow.keras.models;

import org.tensorflow.*;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.layers.Input;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Sequential extends Model<Float> {
    private Input firstLayer;
    private Optimizer<Float> optimizer;
    private List<Layer<Float>> layers;

    private Loss loss;
    private List<MetricFunction> metrics;
;

    private List<Variable<Float>> trainableVars;
    private List<Operand<Float>> initializerOps;

    @SafeVarargs
    public Sequential(Input firstLayer, Layer<Float>... layers) {
        this.firstLayer = firstLayer;
        this.layers = Arrays.asList(layers);
    }

    public Sequential addLayer(Layer<Float> layer) {
        layers.add(layer);
        return this;
    }

    @Override
    public Shape computeOutputShape(Shape inputShape) {
        throw new UnsupportedOperationException("Can't call computeOutputShape on Model");
    }

    @Override
    @SafeVarargs
    public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
        return this.call(tf, inputs[0]);
    }

    public Operand<Float> call(Ops tf, Operand<Float> in) {
        Operand<Float> out = in;
        for (Layer<Float> layer : this.layers) {
            out = layer.apply(tf, out);
        }
        return out;
    }

    public void compile(Ops tf, Optimizer optimizer, Loss loss, List<MetricFunction> metrics) {
        this.loss = loss;
        this.metrics = metrics;
        this.optimizer = optimizer;

        // Targets for training loop
        this.trainableVars = new ArrayList<>();
        this.initializerOps = new ArrayList<>();

        // Build layers
        this.firstLayer.build(tf);
        Shape inputShape = firstLayer.computeOutputShape();

        for (Layer<Float> layer : layers) {
            layer.build(tf, inputShape);
            this.trainableVars.addAll(layer.trainableWeights());
            this.initializerOps.addAll(layer.initializerOps());
            inputShape = layer.computeOutputShape(inputShape);
        }

        this.optimizer.build(tf);
        this.built = true;
    }

    @Override
    public void fit(Ops tf, GraphLoader<Float> train, GraphLoader<Float> test, int epochs, int batchSize) {
        try (Session session = new Session(tf.scope().graph())) {
            runTrainingLoop(tf, session, train, epochs, batchSize, true);
            runPredictionLoop(tf, session, test, batchSize);
        }
    }
    private void runPredictionLoop(Ops tf, Session session, GraphLoader<Float> data, int batchSize) {
        runTrainingLoop(tf, session, data, 1, batchSize, false);
    }

    private void runTrainingLoop(Ops tf, Session session, GraphLoader<Float> data, int epochs, int batchSize, boolean training) {
        data.batch(batchSize);
        data.build(tf);
        Operand<Float>[] dataOps = data.getBatchOperands();

        Session.Runner runner;
        Operand<Float> XOp = dataOps[0];
        Operand<Float> yOp = dataOps[1];

        // Compute Output / Loss / Accuracy
        Operand<Float> yTrue = yOp;
        Operand<Float> yPred = this.apply(tf, XOp);

        Operand<Float> batchLoss = loss.apply(tf, yTrue, yPred);
        Operand<Float> batchAccuracy = this.metrics.get(0).apply(tf, yTrue, yPred);

        List<Operand<Float>> minimize = training ? optimizer.minimize(tf, batchLoss, this.trainableVars) : null;

        if (training) {
            runner = session.runner();

            // Run initializer ops
            for (Operand<Float> op : this.initializerOps) {
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
                    for (Operand<Float> op : minimize) {
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
