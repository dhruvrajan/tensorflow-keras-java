package org.tensorflow.data;

public interface Dataset<T> extends AutoCloseable {
  /**
   * Combines consecutive elements of this dataset into batches. Does not drop the last batch, even
   * if it has fewer than batchSize elements.
   *
   * @param batchSize The number of consecutive elements of this dataset to combine in a single
   *     batch.
   * @return A `Dataset`
   */
  default Dataset<T> batch(long batchSize) {
    return this.batch(batchSize, false);
  }

  /**
   * Combines consecutive elements of this dataset into batches.
   *
   * @param batchSize The number of consecutive elements of this dataset to combine in a single
   *     batch.
   * @param dropRemainder A boolean representing whether the last batch should be dropped in the
   *     case that it has fewer than `batchSize` elements.
   * @return A `Dataset`
   */
  Dataset<T> batch(long batchSize, boolean dropRemainder);

  @Override
  void close();
}
