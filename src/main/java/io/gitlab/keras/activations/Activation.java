package io.gitlab.keras.activations;

import io.gitlab.keras.layers.Layer;
import io.gitlab.keras.mixin.ActivationFunction;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;



public class Activation<T extends Number> extends Layer<T> implements ActivationFunction {
    private ActivationFunction<T> activation;

    public Activation(String activation) {
        this.activation = ActivationType.select(activation);
    }

    public Activation(ActivationFunction<T> activation) {
        this.activation = activation;
    }

    public Operand<T> build(Ops tf, Operand<T> inputs) {
        return this.activation.apply(tf, inputs);
    }


    public static Activation select(String activationName) {
        return new Activation<>(ActivationType.select(activationName));
    }

    @Override
    public Operand apply(Ops tf, Operand features) {
        return build(tf, features);
    }

    public enum ActivationType {
        sigmoid, tanh, relu, elu, selu, softmax, logsoftmax;

        public static <T extends Number> ActivationFunction<T> select(String activation) {
            return ActivationType.select(ActivationType.valueOf(activation));
        }

        public static <T extends Number> ActivationFunction<T> select(ActivationType type) {
            switch (type) {
                case sigmoid:
                    return Activations::sigmoid;
                case tanh:
                    return Activations::tanh;
                case relu:
                    return Activations::relu;
                case elu:
                    return Activations::elu;
                case selu:
                    return Activations::selu;
                case softmax:
                    return Activations::softmax;
                case logsoftmax:
                    return Activations::logSoftmax;
                default:
                    throw new IllegalArgumentException("Invalid ActivationType");
            }
        }
    }
}