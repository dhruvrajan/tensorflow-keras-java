package org.tensorflow.keras.examples.iris;

import org.tensorflow.Graph;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.Iris;
import org.tensorflow.keras.layers.Dense;
import org.tensorflow.keras.layers.InputLayer;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.models.Model;
import org.tensorflow.keras.models.Sequential;
import org.tensorflow.keras.optimizers.GradientDescentOptimizer;
import org.tensorflow.keras.optimizers.Optimizer;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

import java.awt.*;

public class IrisClassifier {
    private static final int INPUT_SIZE = 4;
    private static final int FEATURES = 3;
    final static int BATCH_SIZE = 5;
    final static int EPOCHS = 10;


    public static void main(String[] args) throws Exception {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            Pair<GraphLoader<Float>, GraphLoader<Float>> data = Iris.loadData(0.3);

            Model model = new Sequential(
                    InputLayer.create(INPUT_SIZE),
                    Dense.options()
                            .setActivation(Activations.softmax)
                            .create(FEATURES)
            );

            model.compile(tf, model.compilerOptions()
                    .setOptimizer(new GradientDescentOptimizer(0.01f))
                    .setLoss(Losses.softmax_crossentropy)
                    .addMetric(Metrics.accuracy)
                    .create(graph));

            model.fit(tf, data.first(), data.second(), EPOCHS, BATCH_SIZE);
        }
    }
}
