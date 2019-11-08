package org.tensorflow.keras.metrics;

public enum Metrics {
  accuracy;
  public static Metric select(Metrics metricType) {
    switch (metricType) {
      case accuracy:
        return new Accuracy();
      default:
        throw new IllegalArgumentException("Invalid metric type");
    }
  }
}
