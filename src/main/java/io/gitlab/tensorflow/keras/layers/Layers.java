package io.gitlab.tensorflow.keras.layers;

import io.gitlab.tensorflow.keras.activations.Activations;

public class Layers {

    public static InputLayer inputLayer(int units) {
        return InputLayer.create(units);
    }

    public static Dense dense(int units) {
        return Dense.create(units);
    }

    public static Dense dense(int units, Activations activation) {
        return Dense.options().setActivation(activation).create(units);
    }
}
