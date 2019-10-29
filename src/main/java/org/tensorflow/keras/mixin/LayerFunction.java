package org.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

@FunctionalInterface
public interface LayerFunction<T extends Number> {
  @SuppressWarnings("unchecked")
  Operand<T> apply(Ops tf, Operand<T>... inputs);
}
