package org.tensorflow.keras.examples.mnist;

import org.tensorflow.Graph;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.MNISTLoader;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.layers.Dense;
import org.tensorflow.keras.layers.InputLayer;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.models.Model;
import org.tensorflow.keras.models.Sequential;
import org.tensorflow.keras.optimizers.GradientDescentOptimizer;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

public class MNISTKeras {

    public static void main(String[] args) throws Exception {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Load MNIST Train / Test Data
            Pair<GraphLoader<Float>, GraphLoader<Float>> data = MNISTLoader.graphDataLoader();
            try (GraphLoader<Float> train = data.first();
                 GraphLoader<Float> test = data.second()) {
                Model model = new Sequential(
                        InputLayer.create(28 * 28),
                        Dense.options()
                                .setActivation(Activations.relu)
                                .setKernelInitializer(Initializers.randomNormal)
                                .setBiasInitializer(Initializers.zeros)
                                .create(128),
                        Dense.options()
                                .setActivation(Activations.relu)
                                .setKernelInitializer(Initializers.randomNormal)
                                .setBiasInitializer(Initializers.zeros)
                                .create(64),
                        Dense.options()
                                .setActivation(Activations.softmax)
                                .setKernelInitializer(Initializers.randomNormal)
                                .setBiasInitializer(Initializers.zeros)
                                .create(10)
                );

                // Compile Model
                model.compile(tf, model.compilerOptions()
                        .setOptimizer(new GradientDescentOptimizer(0.2f))
                        .setLoss(Losses.softmax_crossentropy)
                        .addMetric(Metrics.accuracy)
                        .create(graph));

                // Fit and Evaluate Model
                model.fit(tf, model.fitOptions()
                        .setEpochs(10)
                        .setBatchSize(100)
                        .create(train, test));
            }
        }
    }
}
