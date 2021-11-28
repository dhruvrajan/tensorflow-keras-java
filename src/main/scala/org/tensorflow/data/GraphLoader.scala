package org.tensorflow.data

import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TType

trait GraphLoader[T <: TType] extends Dataset[T] {
  def getBatchOperands: Array[Operand[T]]

  def build(tf: Ops): Unit

  def size: Long

  def feedSessionRunner(runner: Session#Runner, batch: Long): Session#Runner
}
