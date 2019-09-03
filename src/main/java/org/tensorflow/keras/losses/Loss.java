package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

/**
 * Base class to represent Losses.
 */
public abstract class Loss implements MetricFunction {
  /**
   * Subclasses should override this method.
   */
  protected abstract Operand<Float> call(Ops tf, Operand<Float> yTrue, Operand<Float> yPred);

  @SafeVarargs
  public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
    return this.call(tf, inputs[0], inputs[1]);
  }

  @Override
  public Operand<Float> apply(Ops tf, Operand<Float> yTrue, Operand<Float> yPred) {
    return this.call(tf, yTrue, yPred);
  }
}
