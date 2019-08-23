package io.gitlab.tensorflow.keras.callbacks;

public abstract class Callback {
    abstract void onBatchBegin(int batch);
    abstract void onBatchEnd(int batch);
    abstract void onEpochEnd(int epoch);
}
