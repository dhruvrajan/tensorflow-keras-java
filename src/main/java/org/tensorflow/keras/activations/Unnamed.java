package org.tensorflow.keras.activations;

import org.tensorflow.Operand;
import org.tensorflow.keras.mixin.ActivationFunction;
import org.tensorflow.op.Ops;

/**
 * Creates an `Activation` from an unnamed function.
 * @param <T>
 */
public class Unnamed<T extends Number> extends Activation<T> {
    private ActivationFunction<T> activation;

    /**
     * Creates an Activation function.
     * @param unnamedActivation An activation function.
     */
    public Unnamed(ActivationFunction<T> unnamedActivation) {
        super();
        this.activation = unnamedActivation;
    }

    /**
     * Applies the given activation.
     */
    @Override
    protected Operand<T> call(Ops tf, Operand<T> inputs) {
        return activation.apply(tf, inputs);
    }
}
