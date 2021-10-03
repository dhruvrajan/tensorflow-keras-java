package org.tensorflow.keras.activations;

import org.tensorflow.Operand;
import org.tensorflow.keras.mixin.ActivationFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.Sigmoid;
import org.tensorflow.op.math.Tanh;
import org.tensorflow.op.nn.*;
import org.tensorflow.types.family.TFloating;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 * Helper functions to compute activations using a TF Ops object.
 */
public enum Activations {
    // All standard activations
    linear, sigmoid, tanh, relu, elu, selu, softmax, logSoftmax;

    /**
     * Create an `Activation` object given a type from the `Activations` enumeration.
     */
    public static <T extends TNumber> Activation<T> select(Activations type) {
        return new Lambda<>(getActivationFunction(type));
    }

    /**
     * Map from `Activations` enumeration to respective activation functions.
     */
    private static <T extends TNumber> ActivationFunction<T> getActivationFunction(Activations type) {
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
            case logSoftmax:
                return Activations::logSoftmax;
            default:
                throw new IllegalArgumentException("Invalid Activation Type");
        }
    }

    /**
     * Linear activation function (no op).
     */
    public static <T extends TType> Operand<T> linear(Ops tf, Operand<T> x) {
        return x;
    }

    /**
     * Sigmoid activation function.
     */
    public static <T extends TType> Sigmoid<T> sigmoid(Ops tf, Operand<T> x) {
        return tf.math.sigmoid(x);
    }

    /**
     * Tanh activation function.
     */
    public static <T extends TType> Tanh<T> tanh(Ops tf, Operand<T> x) {
        return tf.math.tanh(x);
    }

    /**
     * Rectified linear unit.
     */
    public static <T extends TType> Relu<T> relu(Ops tf, Operand<T> x) {
        return tf.nn.relu(x);
    }

    /**
     * Exponential linear unit.
     */
    public static <T extends TNumber> Elu<T> elu(Ops tf, Operand<T> x) {
        return tf.nn.elu(x);
    }

    /**
     * Scaled exponential linear Unit.
     */
    public static <T extends TNumber> Selu<T> selu(Ops tf, Operand<T> x) {
        return tf.nn.selu(x);
    }

    /**
     * Softmax activation function.
     */
    public static <T extends TNumber> Softmax<T> softmax(Ops tf, Operand<T> x) {
        return tf.nn.softmax(x);
    }

    /**
     * Log Softmax activation function.
     */
    public static <T extends TNumber> LogSoftmax<T> logSoftmax(Ops tf, Operand<T> x) {
        return tf.nn.logSoftmax(x);
    }
}
