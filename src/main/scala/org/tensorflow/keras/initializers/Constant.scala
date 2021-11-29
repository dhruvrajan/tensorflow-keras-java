//package org.tensorflow.keras.initializers
//
//import org.tensorflow.Operand
//import org.tensorflow.op.Ops
//import org.tensorflow.types.TInt32
//import org.tensorflow.types.family.TNumber
//
//object Constant {
//  def apply(value: Int    ): Constant = IntImpl   (value)
//  def apply(value: Long   ): Constant = LongImpl  (value)
//  def apply(value: Float  ): Constant = FloatImpl (value)
//  def apply(value: Double ): Constant = DoubleImpl(value)
//
//  private final case class IntImpl(value: Int) extends Constant {
//    override def initialize[T <: TNumber](tf: Ops, shape: Operand[TInt32], dtype: Class[T]): Operand[T] = {
//      val c = tf.constant(value)
//      tf.fill(shape, tf.dtypes.cast(c, dtype))
//    }
//  }
//
//  private final case class LongImpl(value: Long) extends Constant {
//    override def initialize[T <: TNumber](tf: Ops, shape: Operand[TInt32], dtype: Class[T]): Operand[T] = {
//      val c = tf.constant(value)
//      tf.fill(shape, tf.dtypes.cast(c, dtype))
//    }
//  }
//
//  private final case class FloatImpl(value: Float) extends Constant {
//    override def initialize[T <: TNumber](tf: Ops, shape: Operand[TInt32], dtype: Class[T]): Operand[T] = {
//      val c = tf.constant(value)
//      tf.fill(shape, tf.dtypes.cast(c, dtype))
//    }
//  }
//
//  private final case class DoubleImpl(value: Double) extends Constant {
//    override def initialize[T <: TNumber](tf: Ops, shape: Operand[TInt32], dtype: Class[T]): Operand[T] = {
//      val c = tf.constant(value)
//      tf.fill(shape, tf.dtypes.cast(c, dtype))
//    }
//  }
//}
//abstract class Constant extends Initializer with Product {
//  override def productPrefix: String = "Constant"
//}
