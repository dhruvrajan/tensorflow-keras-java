package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public abstract class Metric extends Layer<Float> implements MetricFunction {
  private Operand<Float> outputOp;

  public Metric() {
    super(2);
  }

  public static Metric create(Metrics metricType) {
    return options().create(metricType);
  }

  public static Options options() {
    return new Options();
  }

  static class Options {
    public Metric create(Metrics metricType) {
      return Metrics.select(metricType);
    }
  }

  @Override
  public Operand<Float> apply(Ops tf, Operand<Float> yTrue, Operand<Float> yPred) {
    return this.call(tf, yTrue, yPred);
  }

  @Override
  @SafeVarargs
  public final Operand<Float> call(Ops tf, Operand<Float>... ops) {
    return call(tf, ops[0], ops[1]);
  }

  public abstract Operand<Float> call(Ops tf, Operand<Float> output, Operand<Float> label);
}
