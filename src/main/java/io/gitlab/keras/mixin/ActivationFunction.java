package io.gitlab.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

@Function
public interface ActivationFunction<T> {
    Operand<T> apply(Ops tf, Operand<T> features);
}