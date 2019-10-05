package org.tensorflow.keras.examples.mnist;

import org.tensorflow.Graph;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.MNIST;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.layers.Dense;
import org.tensorflow.keras.layers.Input;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.models.Model;
import org.tensorflow.keras.models.Sequential;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

public class MNISTKeras {
    private static Model<Float> model;
    private static Model.CompileOptions compileOptions;
    private static Model.FitOptions fitOptions;

    static {
        // Define Neural Network Model
        model = new Sequential(
                new Input(28 * 28),
                new Dense(128, Dense.Options.builder()
                        .setActivation(Activations.relu)
                        .setKernelInitializer(Initializers.randomNormal)
                        .setBiasInitializer(Initializers.zeros)
                        .build()),
                new Dense(10, Dense.Options.builder()
                        .setActivation(Activations.softmax)
                        .setKernelInitializer(Initializers.randomNormal)
                        .setBiasInitializer(Initializers.zeros)
                        .build())
        );

        // Model Compile Configuration
        compileOptions = Model.CompileOptions.builder()
                .setOptimizer(Optimizers.sgd)
                .setLoss(Losses.softmax_crossentropy)
                .addMetric(Metrics.accuracy)
                .build();

        // Model Training Loop Configuratoin
        fitOptions = Model.FitOptions.builder()
                .setEpochs(10)
                .setBatchSize(100)
                .build();
    }


    public static void main(String[] args) throws Exception {
        try (Graph graph = new Graph()) {
            // Create Tensorflow Ops Accessor
            Ops tf = Ops.create(graph);

            // Compile Model
            model.compile(tf, compileOptions);

            Pair<GraphLoader<Float>, GraphLoader<Float>> loaders = MNIST.graphLoaders();
            // GraphLoader objects contain AutoCloseable `Tensor` objects.
            try (GraphLoader<Float> train = loaders.first();
                 GraphLoader<Float> test = loaders.second()) {
                // Fit model
                model.fit(tf, train, test, fitOptions);
            }
        }
    }
}

