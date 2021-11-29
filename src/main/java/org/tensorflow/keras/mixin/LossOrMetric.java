package org.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

@FunctionalInterface
public interface LossOrMetric {
  <T extends TNumber> Operand<T> apply(Ops tf, Class<T> dtype, Operand<T> yTrue, Operand<T> yPred);
}
