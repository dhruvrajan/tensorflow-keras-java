package org.tensorflow.data

import org.tensorflow.types.family.TType

abstract class TensorFrame[T <: TType] extends Dataset[T] with GraphLoader[T] {
  protected var _batchSize = 1L
  protected var dropRemainder = false

  /* Override functions from Dataset<T> */ override def batch(batchSize: Long, dropRemainder: Boolean): Dataset[T] = {
    _batchSize = batchSize
    this.dropRemainder = dropRemainder
    this
  }

  override def batchSize: Long = _batchSize

  override def numBatches: Long = size / _batchSize
}
