package org.tensorflow.keras.examples.fashionMnist;

import org.tensorflow.Graph;
import org.tensorflow.data.GraphLoader;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.datasets.FashionMNIST;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.losses.Losses;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.keras.models.Model;
import org.tensorflow.keras.models.Sequential;
import org.tensorflow.keras.optimizers.Optimizers;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

import static org.tensorflow.keras.layers.Layers.*;

public class FashionMNISTKeras {
    private static Model<Float> model;

    static {
        // Define Neural Network Model
        model = Sequential.of(Float.class,
            input(28, 28),
            flatten(),
            dense(128, Activations.relu, Initializers.randomNormal, Initializers.zeros),
            dense(10, Activations.softmax, Initializers.randomNormal, Initializers.zeros)
        );
    }

    public static Model<Float> train(Model<Float> model) throws Exception {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);
            model.compile(tf, Optimizers.sgd, Losses.sparseCategoricalCrossentropy, Metrics.accuracy);

            Pair<GraphLoader<Float>, GraphLoader<Float>> loaders = FashionMNIST.graphLoaders2D(graph);
            // GraphLoader objects contain AutoCloseable `Tensor` objects.
            try (GraphLoader<Float> train = loaders.first(); GraphLoader<Float> test = loaders.second()) {
                model.fit(tf, graph, train, test, 10, 100);
            }
        }

        return model;
    }

    public static void main(String[] args) throws Exception {
        train(model);
    }
}
