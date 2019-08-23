package io.gitlab.tensorflow.keras.examples.mnist;

import io.gitlab.tensorflow.keras.activations.Activations;
import io.gitlab.tensorflow.keras.data.Dataset;
import io.gitlab.tensorflow.keras.datasets.MNISTLoader;
import io.gitlab.tensorflow.keras.layers.Dense;
import io.gitlab.tensorflow.keras.layers.Layers;
import io.gitlab.tensorflow.keras.losses.Losses;
import io.gitlab.tensorflow.keras.metrics.Metrics;
import io.gitlab.tensorflow.keras.models.Model;
import io.gitlab.tensorflow.keras.models.Sequential;
import io.gitlab.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.Graph;
import org.tensorflow.op.Ops;

public class MNISTKeras {

    public static void main(String[] args) throws Exception {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);
            Dataset data = MNISTLoader.loadDataset();

            Sequential model = new Sequential(
                    Layers.inputLayer(28 * 28),

                    // Layers.dense(required1, required2, new Dense.OptionsBuilder().what(22).which(11).build())
                    new Dense(10, Dense.options().setActivation(Activations.logsoftmax)),
                    Dense.create(10),
                    Dense.options()
                        .setActivation(Activations.logsoftmax)
                        .create(10),

                    Layers.dense(10, Activations.relu),
                    Layers.dense(10, Activations.softmax)
            );

            // Build Graph
            model.compile(tf, new Model.CompilerOptions(graph)
                    .setOptimizer(Optimizers.sgd)
                    .setLoss(Losses.softmax_crossentropy)
                    .addMetric(Metrics.accuracy));


            model.fit(tf, data, 100, 100);
        }
    }
}
