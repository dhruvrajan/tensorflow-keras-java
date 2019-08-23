package io.gitlab.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

@FunctionalInterface
public interface LayerFunction<T> {
    @SuppressWarnings("unchecked")
    Operand<T> apply(Ops tf, Operand<T>... inputs);
}