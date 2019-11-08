package org.tensorflow.keras.layers;

import org.tensorflow.keras.activations.Activation;
import org.tensorflow.keras.activations.Activations;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.utils.Keras;

public class Layers {
    // Builders for Input Layer
    public static <T extends Number> Input<T> input(long firstDim, long... units) {
        return new Input<>(Keras.concatenate(firstDim, units));
    }

    // Builders for Dense Layer
    public static <T extends Number> Dense<T> dense(int units) {
        return new Dense<>(units, Dense.Options.defaults());
    }

    public static <T extends Number> Dense<T> dense(int units, Dense.Options options) {
        return new Dense<>(units, options);
    }

    public static <T extends Number> Dense<T> dense(int units, Activation<T> activation) {
        return new Dense<>(units, Dense.Options.builder()
                .setActivation(activation)
                .build());
    }

    public static <T extends Number> Dense<T> dense(int units, Activations activation) {
        return new Dense<>(units, Dense.Options.builder()
                .setActivation(activation)
                .build());
    }

    public static <T extends Number> Dense<T> dense(int units, Activations activation, Initializers kernelInitializer,
            Initializers biasInitializer) {
        return new Dense<>(units, Dense.Options.builder()
                .setActivation(activation)
                .setKernelInitializer(kernelInitializer)
                .setBiasInitializer(biasInitializer)
                .build());
    }

    public static <T extends Number> Dense<T> dense(int units, Activation<T> activation, Initializer kernelInitializer,
            Initializer biasInitializer) {
        return new Dense<>(units, Dense.Options.builder().setActivation(activation)
                .setKernelInitializer(kernelInitializer).setBiasInitializer(biasInitializer).build());
    }

    // Builders for Flatten Layer
    public static <T extends Number> Flatten<T> flatten() {
        return new Flatten<>();
    }
}
