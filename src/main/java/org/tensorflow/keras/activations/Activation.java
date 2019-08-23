package org.tensorflow.keras.activations;

import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.ActivationFunction;
import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;



public class Activation<T extends Number> extends Layer<T> {
    private ActivationFunction<T> activation;

    public Activation(ActivationFunction<T> activation) {
        super(1);
        this.activation = activation;
    }

    public Activation<T> create(Activations activation) {
        return options().create(activation);
    }

    public static Options options() {
        return new Options();
    }

    public static class Options {
        public Options() {

        }

        public <T extends Number> Activation<T> create(Activations activation) {
            return Activations.select(activation);
        }
    }

    @Override
    public void build(Ops tf, Shape in) {
        throw new UnsupportedOperationException("no create");
    }

    @Override
    public Shape computeOutputShape(Shape inputShape) {
        return inputShape;
    }

    private Operand<T> call(Ops tf, Operand<T> inputs) {
        return this.activation.apply(tf, inputs);
    }

    @Override
    public Operand<T> call(Ops tf, Operand<T>... inputs) {
        return call(tf, inputs[0]);
    }



}