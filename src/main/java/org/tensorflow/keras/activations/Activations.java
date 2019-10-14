package org.tensorflow.keras.activations;

import org.tensorflow.Operand;
import org.tensorflow.keras.mixin.ActivationFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;

/**
 * Helper functions to compute activations using a TF Ops object.
 */
public enum Activations {
    linear,
    sigmoid,
    tanh,
    relu,
    elu,
    selu,
    softmax,
    logsoftmax;

    public static <T extends Number> Activation<T> select(Activations type) {
        return new Activation<>(getActivationFunction(type));
    }

    private static <T extends Number> ActivationFunction<T> getActivationFunction(Activations type) {
        switch (type) {
            case linear:
                return Activations::linear;
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

    /**
     * Linear activation function.
     */
    public static <T> Operand<T> linear(Ops tf, Operand<T> x) {
        return x;
    }

    /**
     * Sigmoid activation function.
     */
    public static <T> Sigmoid<T> sigmoid(Ops tf, Operand<T> x) {
        return tf.sigmoid(x);
    }

    /**
     * Tanh activation function.
     */
    public static <T> Tanh<T> tanh(Ops tf, Operand<T> x) {
        return tf.tanh(x);
    }

    /**
     * Rectified Linear Unit.
     */
    public static <T> Relu<T> relu(Ops tf, Operand<T> features) {
        return tf.relu(features);
    }

    /**
     * Exponential Linear Unit.
     */
    public static <T extends Number> Elu<T> elu(Ops tf, Operand<T> features) {
        return tf.elu(features);
    }

    /**
     * Scaled Exponential Linear Unit.
     */
    public static <T extends Number> Selu<T> selu(Ops tf, Operand<T> features) {
        return tf.selu(features);
    }

    /**
     * Softmax activation function.
     */
    public static <T extends Number> Softmax<T> softmax(Ops tf, Operand<T> logits) {
        return tf.softmax(logits);
    }

    /**
     * Log-Softmax activation function.
     */
    public static <T extends Number> LogSoftmax<T> logSoftmax(Ops tf, Operand<T> logits) {
        return tf.logSoftmax(logits);
    }
}
