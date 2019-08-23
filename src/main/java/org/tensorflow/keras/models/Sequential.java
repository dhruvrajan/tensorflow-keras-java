package org.tensorflow.keras.models;

import org.tensorflow.keras.data.Dataset;
import org.tensorflow.keras.layers.InputLayer;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.losses.Loss;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Sequential extends Model<Float> {
    private InputLayer firstLayer;
    private Placeholder<Float> labels;
    private Optimizer<Float> optimizer;
    private List<Layer<Float>> layers;

    private Loss loss;
    private List<MetricFunction> metrics;
    private Operand<Float> lossOp;
    private Operand<Float> metricOp;

    private List<Variable<Float>> trainableVars;
    private List<Operand<Float>> initializerOps;

    @SafeVarargs
    public Sequential(InputLayer firstLayer, Layer<Float>... layers) {
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
        for (Layer<Float> layer: this.layers) {
            out = layer.apply(tf, out);
        }
        return out;
    }

    public void compile(Ops tf, Optimizer optimizer, Loss loss, List<MetricFunction> metrics) throws Exception {
        this.loss = loss;
        this.metrics = metrics;
        this.optimizer = optimizer;
        this.labels = tf.placeholder(Float.class);

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
    }


    /**
     * Basically, need to collect targets for session.run.
     */
//    public void fit(Ops tf, Graph graph, TensorDataset<Float> data, int epochs, int batchSize) {
//        try (var session = new Session(graph)) {
//
//            // Initialize
//            new SessionRunner(session.runner())
//                    .addTargets(this.initializerOps)
//                    .run();
//
//            // Train
//            var train = data.getTrain();
//            train.create(tf, batchSize);
//
//            double epochAccuracy = 0;
//            double epochLoss = 0;
//
//            for (int epoch = 0; epoch < epochs; epoch++) {
//                for (int i = 0; i < train.numBatches(); i++) {
//                    // Training Loop
//                    var trainBatch = train.loadBatch(tf, i);
//                    var inputData = trainBatch.get(0);
//                    var actual = trainBatch.get(1);
//
//                    var predicted = this.apply(tf, inputData);
//                    var loss = this.loss.apply(tf, predicted, actual);
//                    var metrics = this.metrics
//                            .stream()
//                            .map(m -> m.apply(tf, predicted, actual))
//                            .collect(Collectors.toList());
//
//                    // Run training loop
//                    List<Tensor<?>> trainOutputs =
//                            new SessionRunner(session.runner())
//                                    .addTargets(inputData, actual)
//                                    .addTargets(optimizer.minimize(tf, loss, this.trainableVars))
//                                    .fetch(metrics)
//                                    .run();
//
//                    double accuracy = trainOutputs.get(0).floatValue();
//                    double batchLoss = trainOutputs.get(1).floatValue();
//
//
//                    epochAccuracy += accuracy / train.numBatches();
//                    epochLoss += batchLoss / train.numBatches();
//                }
//
//                System.out.println("Loss on epoch " + epoch + " is: loss=" + epochLoss + "    accuracy=" + epochAccuracy);
//            }
//
//            // Validate
//            var val = data.getVal();
//            val.create(tf, batchSize);
//
//            double valLoss = 0;
//            double valAccuracy = 0;
//            for (int i = 0; i < val.numBatches(); i++) {
//                // Training Loop
//                var valBatch = val.loadBatch(tf, i);
//                var inputData = valBatch.get(0);
//                var actual = valBatch.get(1);
//
//                var predicted = this.call(tf, inputData);
//                var metrics = this.metrics
//                        .stream()
//                        .map(m -> m.apply(tf, predicted, actual))
//                        .collect(Collectors.toList());
//
//                // Collect validation metrics
//                List<Tensor<?>> valOutputs =
//                        new SessionRunner(session.runner())
//                                .addTargets(inputData, actual)
//                                .fetch(metrics)
//                                .run();
//
//                double batchAccuracy = valOutputs.get(0).floatValue();
//                double batchLoss = valOutputs.get(1).floatValue();
//
//                valLoss += batchLoss / val.numBatches();
//                valAccuracy += batchAccuracy / val.numBatches();
//            }
//
//            System.out.println("Val loss is loss=" + valLoss + "accuracy=" + valAccuracy);
//        }
//    }




    public void fit(Ops tf, Dataset data, int epochs, int batchSize) {
        Graph graph = tf.scope().graph();

        List<Dataset.Split> trainBatches = data.trainBatches(batchSize);
        List<Dataset.Split> testBatches = data.testBatches(batchSize);

        try (Session session = new Session(graph)) {

            // initialize weights
            Session.Runner initRunner = session.runner();
            addTargets(initRunner, new ArrayList<Operand<Float>>(firstLayer.initializers.values().stream()
                    .map(i -> i.getInitializerOp()).collect(Collectors.toList())));
            for (Layer<Float> layer : this.layers) {
                addTargets(initRunner, new ArrayList<Operand<Float>>(layer.initializers.values().stream().map(i -> i.getInitializerOp()).collect(Collectors.toList())));
            }
            initRunner.run();

            for (int e = 0; e < epochs; e++) {

                double epochAccuracy = 0;
                double epochLoss = 0;

                // train batches
                for (int i = 0; i < trainBatches.size(); i++) {
                    Session.Runner batchRunner = session.runner();

                    // Run Gradient Descent Ops
                    try (Tensor<Float> XBatch = Tensors.create(trainBatches.get(i).X);
                         Tensor<Float> yBatch = Tensors.create(trainBatches.get(i).y)) {



                        List<Tensor<?>> values = addTargets(batchRunner, optimizer.getTargets())
                                .fetch(metricOp)
                                .fetch(lossOp)
                                .feed(firstLayer.input.asOutput(), XBatch)
                                .feed(labels.asOutput(), yBatch)
                                .run();

                        double accuracy = values.get(0).floatValue();
                        double loss = values.get(1).floatValue();

                        epochAccuracy += accuracy / trainBatches.size();
                        epochLoss += loss / trainBatches.size();
                    }
                }

                // Run Gradient Descent Ops
                epochAccuracy = 0;
                epochLoss = 0;

                for (int i = 0; i < testBatches.size(); i++) {
                    Session.Runner testRunner = session.runner();

                    try (Tensor<Float> XBatch = Tensors.create(testBatches.get(i).X);
                         Tensor<Float> yBatch = Tensors.create(testBatches.get(i).y)) {

                        List<Tensor<?>> values = testRunner
                                .fetch(metricOp)
                                .fetch(lossOp)
                                .feed(firstLayer.input.asOutput(), XBatch)
                                .feed(labels.asOutput(), yBatch)
                                .run();

                        double accuracy = values.get(0).floatValue();
                        double loss = values.get(1).floatValue();

                        epochAccuracy += accuracy / testBatches.size();
                        epochLoss += loss / testBatches.size();

                    }
                }

                System.out.println("(Test) Epoch " + e + " accuracy: " + epochAccuracy + "loss: " + epochLoss);
            }
        }
    }



    private Session.Runner addTargets(Session.Runner runner, List<Operand<Float>> targets) {
        for (Operand target : targets) {
            runner.addTarget(target);
        }
        return runner;
    }

    private Session.Runner fetchOutputs(Session.Runner runner, List<Operand<Float>> outputs) {
        for (Operand<Float> output : outputs) {
            runner.fetch(output.asOutput());
        }
        return runner;
    }
}