package org.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

@FunctionalInterface
public interface LayerFunction<T extends TType> {
  @SuppressWarnings("unchecked")
  Operand<T> apply(Ops tf, Operand<T>... inputs);
}
