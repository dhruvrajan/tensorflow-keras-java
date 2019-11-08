package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.keras.mixin.LossOrMetric;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;

public enum Losses {
    sparseCategoricalCrossentropy;

    public static Loss select(Losses lossType) {
        return new Lambda(getLossFunction(lossType));
    }

    public static LossOrMetric getLossFunction(Losses lossType) {
        switch (lossType) {
            case sparseCategoricalCrossentropy:
                return Losses::sparseCategoricalCrossentropyLoss;
            default:
                throw new IllegalArgumentException("Invalid loss type.");
        }
    }

    public static <T extends Number> Operand<T> sparseCategoricalCrossentropyLoss(Ops tf, Class<T> dtype, Operand<T> actual, Operand<T> labels) {
        return tf.math.mean(tf.math.neg(tf.reduceSum(tf.math.mul(actual, tf.math.log(labels)), Keras.constArray(tf, 1))), Keras.constArray(tf, 0));
    }
}