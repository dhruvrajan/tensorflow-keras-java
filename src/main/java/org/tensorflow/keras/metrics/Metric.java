package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.keras.layers.Layer;
import org.tensorflow.keras.mixin.MetricFunction;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public abstract class Metric<T extends Number> extends Layer<T> implements MetricFunction<T> {
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
  public Operand<T> apply(Ops tf, Operand<T> yTrue, Operand<T> yPred) {
    return this.call(tf, yTrue, yPred);
  }

  @Override
  @SafeVarargs
  public final Operand<T> call(Ops tf, Operand<T>... ops) {
    return call(tf, ops[0], ops[1]);
  }

  public abstract Operand<T> call(Ops tf, Operand<T> output, Operand<T> label);
}
