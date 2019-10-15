package org.tensorflow.keras.mixin;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

@FunctionalInterface
public interface MetricFunction<T extends Number> {
  Operand<T> apply(Ops tf, Operand<T> yTrue, Operand<T> yPred);
}
