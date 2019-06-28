package io.gitlab.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

@FunctionalInterface
public interface ActivationFunction<T> {
    Operand<T> apply(Ops tf, Operand<T> features);
}