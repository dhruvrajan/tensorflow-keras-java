package org.tensorflow.keras.optimizers;

import org.tensorflow.types.family.TNumber;

public enum Optimizers {
  sgd;

  public static <T extends TNumber> Optimizer<T> select(Optimizers optimizerType) {
    switch (optimizerType) {
      case sgd:
        return new GradientDescentOptimizer<>(0.2f);
      default:
        throw new IllegalArgumentException("Invalid Optimizer Type.");
    }
  }
}
