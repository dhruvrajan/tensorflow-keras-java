package org.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

@FunctionalInterface
public interface LossOrMetric {
  <T extends Number> Operand<T> apply(Ops tf, Class<T> dtype, Operand<T> yTrue, Operand<T> yPred);
}
