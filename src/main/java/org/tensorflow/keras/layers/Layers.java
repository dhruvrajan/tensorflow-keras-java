package org.tensorflow.keras.layers;

import org.tensorflow.keras.activations.Activation;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.utils.Keras;

public class Layers {
    // Builders for Input Layer
    public static Input input(long firstDim, long... units) {
        return new Input(Keras.concatenate(firstDim,units));
    }

    // Builders for Dense Layer
    public static Dense dense(int units) {
        return new Dense(units, Dense.options());
    }

    public static Dense dense(int units, Dense.Options options) {
        return new Dense(units, options);
    }

    public static  Dense dense(int units, Activation<Float> activation) {
        return new Dense(units, Dense.Options.builder().setActivation(activation).build());
    }

    public static  Dense dense(int units, Activations activation) {
        return new Dense(units, Dense.Options.builder().setActivation(activation).build());
    }

    public static  Dense dense(int units, Activations activation, Initializers kernelInitializer, Initializers biasInitializer) {
        return new Dense(units, Dense.Options.builder()
                .setActivation(activation)
                .setKernelInitializer(kernelInitializer)
                .setBiasInitializer(biasInitializer)
                .build());
    }

    // Builders for Flatten Layer
    public static Flatten flatten(int units) {
        return new Flatten(units);
    }
}
