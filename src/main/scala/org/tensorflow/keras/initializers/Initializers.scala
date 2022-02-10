package org.tensorflow.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.framework.initializers.VarianceScaling.Distribution
import org.tensorflow.framework.initializers.{Glorot, Initializer, Ones, RandomNormal, Zeros}
import org.tensorflow.keras.utils.Keras
import org.tensorflow.op.{Ops => TF}
import org.tensorflow.op.core.Assign
import org.tensorflow.types.family.{TFloating, TNumber, TType}

object Initializers extends Enumeration {
  type Initializers = Value

  val zeros, ones, randomNormal, glorotUniform = Value

  implicit class Ops[T <: TType](private val i: Initializer[T]) extends AnyVal {
    def apply(tf: TF, in: Operand[T], dtype: Class[T]): Assign[T] =
      tf.assign(in, i.call(tf, Keras.shapeOperandL(tf, in.asOutput.shape), dtype))
  }

  def select[T <: TNumber](initializer: Initializers): Initializer[T] = initializer match {
    case `zeros` =>
      new Zeros[T]
    case `ones` =>
      new Ones[T]
    case `randomNormal` =>
      val seed = scala.util.Random.nextLong()
      val res = new RandomNormal[TFloating](0.0f, 0.1f, seed) // XXX TODO: -0.2f, 0.2f)
      res.asInstanceOf[Initializer[T]]
    case `glorotUniform` =>
      val res = new Glorot[TFloating](Distribution.UNIFORM, util.Random.nextLong()) // XXX TODO ok seed?
      res.asInstanceOf[Initializer[T]]
    case _ =>
      throw new IllegalArgumentException("invalid initializer type")
  }
}
