package org.tensorflow.data

trait Dataset[T] extends AutoCloseable {
  /**
    * Combines consecutive elements of this dataset into batches. Does not drop the last batch, even
    * if it has fewer than batchSize elements.
    *
    * @param batchSize The number of consecutive elements of this dataset to combine in a single
    *                  batch.
    * @return A `Dataset`
    */
  def batch(batchSize: Long): Dataset[T] = this.batch(batchSize, dropRemainder = false)

  /**
    * Combines consecutive elements of this dataset into batches.
    *
    * @param batchSize     The number of consecutive elements of this dataset to combine in a single
    *                      batch.
    * @param dropRemainder A boolean representing whether the last batch should be dropped in the
    *                      case that it has fewer than `batchSize` elements.
    * @return A `Dataset`
    */
  def batch(batchSize: Long, dropRemainder: Boolean): Dataset[T]

  def batchSize: Long

  def numBatches: Long

  override def close(): Unit
}
