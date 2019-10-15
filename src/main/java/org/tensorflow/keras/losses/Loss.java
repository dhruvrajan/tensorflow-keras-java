package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

/**
 * Base class to represent Losses.
 */
public abstract class Loss<T extends Number> implements MetricFunction<T> {
  /**
   * Subclasses should override this method.
   */
  protected abstract Operand<T> call(Ops tf, Operand<T> yTrue, Operand<T> yPred);

  @SafeVarargs
  public final Operand<T> call(Ops tf, Operand<T>... inputs) {
    return this.call(tf, inputs[0], inputs[1]);
  }

  @Override
  public Operand<T> apply(Ops tf, Operand<T> yTrue, Operand<T> yPred) {
    return this.call(tf, yTrue, yPred);
  }
}
