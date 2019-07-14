package io.gitlab.keras.models;

import io.gitlab.keras.data.Dataset;
import io.gitlab.keras.data.TensorDataset;
import io.gitlab.keras.layers.InputLayer;
import io.gitlab.keras.layers.Layer;
import io.gitlab.keras.losses.Loss;
import io.gitlab.keras.mixin.MetricFunction;
import io.gitlab.keras.optimizers.Optimizer;
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
    private Operand<Float> lossOp;
    private Operand<Float> metricOp;
    private Optimizer<Float> optimizer;
    private List<Layer<Float>> layers;

    private Loss loss;
    private List<MetricFunction> metrics;

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
        return null;
    }

    @Override
    @SafeVarargs
    public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
        return this.call(tf, inputs[0]);
    }

    public Operand<Float> call(Ops tf, Operand<Float> in) {
        throw new UnsupportedOperationException("Cannot call a sequential model.");
    }

    @Override
    public List<Operand<Float>> initializerOps() {
        return this.layers.stream().flatMap(l -> l.initializerOps().stream()).collect(Collectors.toList());
    }

    public void compile(Ops tf, Optimizer optimizer, Loss loss, List<MetricFunction> metrics) throws Exception {
        this.loss = loss;
        this.metrics = metrics;
        this.optimizer = optimizer;
        this.labels = tf.placeholder(Float.class);

        // Build layers
        this.firstLayer.build(tf);
        Shape inputShape = firstLayer.computeOutputShape();

        for (Layer layer : layers) {
            layer.build(tf, inputShape);
            inputShape = layer.computeOutputShape(inputShape);
        }

        // Build optimizer
        for (Layer layer : layers) {
            optimizer.build(tf, new ArrayList<Variable<Float>>(layer.weights.values()), lossOp);
        }
    }


    public void compile2(Ops tf, Optimizer optimizer, Loss loss, List<MetricFunction> metrics) throws Exception {
        Operand out = firstLayer.build(tf);
        this.loss = loss;
        this.metrics = metrics;
        labels = tf.placeholder(Float.class);
        this.optimizer = optimizer;

        for (Layer layer : layers) {
            layer.build(tf);
            out = layer.call(tf, out);
        }

        lossOp = loss.build(tf, out, labels);

        for (Layer layer : layers) {
            optimizer.build(tf, new ArrayList<Variable<Float>>(layer.weights.values()), lossOp);
        }

        for (MetricFunction metric : this.metrics) {
            metricOp = metric.apply(tf, out, labels);
        }
    }


    /**
     * Basically, need to collect targets for session.run.
     */
    public void fit(Ops tf, TensorDataset<Float> data, int epochs, int batchSize) {

        try (var session = new Session(tf.scope().graph())) {
            // Initialize
            var initializerOps = this.initializerOps();
            addTargets(session.runner(), initializerOps).run();

            // Train
            var train = data.getTrain();
            train.build(tf, batchSize);

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < train.numBatches(); i++) {
                    // Load train batch
                    Session.Runner runner = session.runner();
                    addTargets(runner, train.loadBatch(tf, i));

                    // Run training ops on train batch
                    var trainingOps = optimizer.trainingOps();
                    addTargets(session.runner(), trainingOps);
                    fetchOutputs(runner, metrics.stream()
                            .flatMap(m -> m.metricOps().stream())
                            .collect(Collectors.toList()));

                    // Collect metric output
                    List<Tensor<?>> trainOutputs = runner.run();
                }
            }

            // Evaluate
            var val = data.getVal();
            val.build(tf, batchSize);

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < val.numBatches(); i++) {
                    // Load val batch
                    Session.Runner runner = session.runner();
                    addTargets(runner, val.loadBatch(tf, i));

                    // Fetch metrics on val batch
                    fetchOutputs(runner, metrics.stream()
                            .flatMap(m -> m.metricOps().stream())
                            .collect(Collectors.toList()));

                    // Collect metric output
                    List<Tensor<?>> valOutputs = runner.run();
                }
            }
        }
    }




    public void fit(Graph graph, Dataset data, int epochs, int batchSize) {

        List<Dataset.Split> trainBatches = data.trainBatches(batchSize);
        List<Dataset.Split> testBatches = data.testBatches(batchSize);

        try (Session session = new Session(graph)) {

            // initialize weights
            Session.Runner initRunner = session.runner();
            addTargets(initRunner, new ArrayList<Operand<Float>>(firstLayer.initializers.values()));
            for (Layer layer : this.layers) {
                addTargets(initRunner, new ArrayList<Operand<Float>>(layer.initializers.values()));
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