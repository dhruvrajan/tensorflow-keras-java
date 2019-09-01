package org.tensorflow.data;

public interface BatchLoader<T> {
  long size();

  long batchSize();

  default long numBatches() {
    return size() / batchSize();
  }
}
