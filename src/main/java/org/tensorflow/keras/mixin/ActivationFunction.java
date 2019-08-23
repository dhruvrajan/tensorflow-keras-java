package org.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;


public interface ActivationFunction<T> {
    Operand<T> apply(Ops tf, Operand<T> features);
}