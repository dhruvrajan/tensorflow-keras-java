package io.gitlab.keras.activations;

import io.gitlab.keras.layers.Layer;
import io.gitlab.keras.mixin.ActivationFunction;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;



public class Activation<T extends Number> extends Layer<T> implements ActivationFunction {
    private ActivationFunction<T> activation;

    public Activation(String activation) {
        this.activation = Activations.select(activation);
    }

    public Activation(ActivationFunction<T> activation) {
        this.activation = activation;
    }

    public Operand<T> build(Ops tf, Operand<T> in) {
        return null;
    }

    public Operand<T> call(Ops tf, Operand<T> inputs) {
        return this.activation.apply(tf, inputs);
    }

    @Override
    public Operand apply(Ops tf, Operand features) {
        return call(tf, features);
    }

}