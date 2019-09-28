package org.tensorflow.keras.examples.mnist;

import org.tensorflow.Graph;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.data.TensorFrame;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.MNISTLoader;
import org.tensorflow.keras.layers.Dense;
import org.tensorflow.keras.layers.InputLayer;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.models.Model;
import org.tensorflow.keras.models.Sequential;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

public class MNISTKeras {
    private static final int INPUT_SIZE = 28 * 28;
    private static final int HIDDEN = 200;
    private static final int FEATURES = 10;
    private static final int BATCH_SIZE = 100;
    private static final int EPOCHS = 20;

    public static void main(String[] args) throws Exception {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Load MNIST Train / Test Data
            Pair<GraphLoader<Float>, GraphLoader<Float>> data = MNISTLoader.graphDataLoader();
            try (GraphLoader<Float> train = data.first();
                 GraphLoader<Float> test = data.second()) {

                Model model = new Sequential(
                        InputLayer.create(INPUT_SIZE),
                        Dense.options()
                                .setActivation(Activations.softmax)
                                .create(FEATURES)
                );

                model.compile(tf, model.compilerOptions()
                        .setOptimizer(Optimizers.sgd)
                        .setLoss(Losses.softmax_crossentropy)
                        .addMetric(Metrics.accuracy)
                        .create(graph));

                model.fit(tf, train, test, EPOCHS, BATCH_SIZE);
            }
        }
    }
}
