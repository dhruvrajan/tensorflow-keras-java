package org.tensorflow.keras.callbacks;

import org.tensorflow.keras.logs.*;
public abstract class Callback {
  /**
   * Perform an operation at the start of each epoch.
   * @param epoch Starting Epoch.
   */
  public abstract void onEpochBegin(int epoch, EpochBeginLogs logs);

  public abstract void onEpochEnd(int epoch, EpochEndLogs logs);

  public abstract void onBatchBegin(int batch, BatchBeginLogs logs);

  public abstract void onBatchEnd(int batch, BatchEndLogs logs);

  public abstract void onTrainBegin();

  public abstract void onTrainEnd();
}
