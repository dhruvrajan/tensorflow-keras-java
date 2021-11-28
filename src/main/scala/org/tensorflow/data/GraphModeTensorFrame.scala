package org.tensorflow.data

import org.tensorflow.{Operand, Session, Tensor}
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.TInt64
import org.tensorflow.types.family.TType

import java.{util => ju}

object GraphModeTensorFrame {
  /**
    * Utility to construct a Shape from a long[]
    */
  private def getShape(dims: Long*) = {
    assert(dims.length > 0)
    Shape.of(dims: _*)
  }
}

class GraphModeTensorFrame[T <: TType](val dtype: Class[T], val firstTensor: Tensor, val tensors: Tensor*) //    @SafeVarargs
  extends TensorFrame[T] with GraphLoader[T] with AutoCloseable { // Check first dimension matches

  val matchDim: Long = firstTensor.shape.size(0)
  for (t <- tensors) {
    if (t.shape.size(0) != matchDim) throw new IllegalArgumentException("All dataTensors in a tensor frame must have equal first dimension.")
  }

  // Record Tensor Objects
  final private val dataTensors = {
    val res = new Array[Tensor](tensors.length + 1)
    res(0) = firstTensor
    System.arraycopy(tensors, 0, res, 1, tensors.length)
    res
  }

  private var dataPlaceholders: Array[Placeholder [T]]      = null
  private var batchOperands   : Array[Operand     [T]]      = null
  private var batchStart      : Array[Placeholder[TInt64]]  = null
  private var batchSizeArr    : Array[Placeholder[TInt64]]  = null
  private var built = false

  def length: Int = this.dataTensors.length

  override def size: Long = this.dataTensors(0).shape.size(0)

  override def build(tf: Ops): Unit = { // Create Placeholders (will be filled by dataTensors before graph is run)
    dataPlaceholders = new Array[Placeholder[T]](this.length)
    for (i <- 0 until this.length) {
      dataPlaceholders(i) = tf.placeholder(this.dtype, Placeholder.shape(dataTensors(i).shape))
    }
    // Placeholder representing batch start and size selectors.
    batchStart    = new Array[Placeholder[TInt64]](this.length)
    batchSizeArr  = new Array[Placeholder[TInt64]](this.length)
    for (i <- 0 until this.length) {
      batchStart  (i) = tf.placeholder(classOf[TInt64], Placeholder.shape(Shape.of(dataTensors(i).shape.numDimensions)))
      batchSizeArr(i) = tf.placeholder(classOf[TInt64], Placeholder.shape(Shape.of(dataTensors(i).shape.numDimensions)))
    }
    // Create batch slice operands
    this.batchOperands = new Array[Operand[T]](this.length)
    for (i <- 0 until this.length) {
      batchOperands(i) = tf.slice(dataPlaceholders(i), batchStart(i), batchSizeArr(i))
    }
    this.built = true
  }

  override def feedSessionRunner(runner: Session#Runner, batch: Long): Session#Runner = { // Feed Data Tensors
    for (i <- 0 until dataPlaceholders.length) {
      runner.feed(dataPlaceholders(i).asOutput, dataTensors(i))
    }
    // Feed Batch Selectors
    for (i <- 0 until this.length) {
      val start = new Array[Long](dataTensors(i).shape.numDimensions)
      ju.Arrays.fill(start, 0)
      start(0) = batch * this.batchSize
      val size = new Array[Long](dataTensors(i).shape.numDimensions)
      ju.Arrays.fill(size, -1)
      size(0) = this.batchSize
      runner.feed(batchStart   (i).asOutput, TInt64.vectorOf(start: _*))
      runner.feed(batchSizeArr (i).asOutput, TInt64.vectorOf(size : _*))
    }
    runner
  }

  def isBuilt: Boolean = built

  def getDataTensors: Array[Tensor] = dataTensors

  def getDataPlaceholders: Array[Placeholder[T]] = dataPlaceholders

  override def getBatchOperands: Array[Operand[T]] = batchOperands

  override def close(): Unit =
    for (tensor <- this.dataTensors) {
      tensor.close()
    }
}
