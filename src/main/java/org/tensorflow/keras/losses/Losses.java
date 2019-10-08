package org.tensorflow.keras.losses;

public enum Losses {
  mean_squared_error,
    sparseCategoricalCrossentropy;

  public static Loss select(Losses lossType) {
    switch (lossType) {
//      case mean_squared_error:
//        return new MeanSquaredError();
      case sparseCategoricalCrossentropy:
        return new SoftmaxCrossEntropyLoss();
      default:
        throw new IllegalArgumentException("Invalid loss type.");
    }
  }
}
