package org.tensorflow.data;

public abstract interface Dataset<T> {

  /**
   * Applies a transformation function to this dataset.
   *
   * @param transformation A function that takes one `Dataset` argument and returns a `Dataset`.
   * @return The `Dataset` returned by applying `transformation` to this dataset.
   */
  //  public abstract Dataset<T> apply(Function<Dataset<T>, Dataset<T>> transformation);

  /**
   * Combines consecutive elements of this dataset into batches. Does not drop the last batch, even
   * if it has fewer than batchSize elements.
   *
   * @param batchSize The number of consecutive elements of this dataset to combine in a single
   *     batch.
   * @return A `Dataset`
   */
  public default Dataset<T> batch(long batchSize) {
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
  public abstract Dataset<T> batch(long batchSize, boolean dropRemainder);

  /**
   * Enumerates the elements of this dataset as pairs of tensors and integers
   *
   * @param start The start value for enumeration
   * @return A `Dataset`
   */
  //  public abstract Iterator<Pair<Integer, Collection<T>>> enumerate(int start);

  //  public Iterator<Operand<T>[]> iterator(Ops tf);
}
