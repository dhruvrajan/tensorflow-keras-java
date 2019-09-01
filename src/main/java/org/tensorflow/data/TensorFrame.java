package org.tensorflow.data;

public abstract class TensorFrame<T> implements Dataset<T>, BatchLoader<T> {
  protected long batchSize = 1;
  protected boolean dropRemainder = false;

  /* Override functions from Dataset<T> */
  @Override
  public Dataset<T> batch(long batchSize, boolean dropRemainder) {
    this.batchSize = batchSize;
    this.dropRemainder = dropRemainder;
    return this;
  }

  @Override
  public long batchSize() {
    return this.batchSize;
  }

  @Override
  public long numBatches() {
    return size() / batchSize;
  }
}
