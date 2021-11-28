package org.tensorflow.utils

import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.types.family.TType

import java.{util => ju}

class SessionRunner(val runner: Session#Runner) {
  def addTarget(operation: String): SessionRunner = {
    this.runner.addTarget(operation)
    this
  }

  def addTarget(operand: Operand[_]): SessionRunner = {
    this.runner.addTarget(operand)
    this
  }

  def addTargets(operands: Operand[_]*): SessionRunner = {
    for (operand <- operands) {
      this.runner.addTarget(operand)
    }
    this
  }

  def addTargets[T <: TType](operands: ju.List[Operand[T]]): SessionRunner = {
    operands.forEach { operand =>
      this.runner.addTarget(operand)
    }
    this
  }

  def addTargetsStr(targets: String*): SessionRunner = {
    for (target <- targets) {
      this.runner.addTarget(target)
    }
    this
  }

  def feed(tensors: Array[Tensor], ops: Array[Operand[_]]): SessionRunner = {
    for (pairs <- tensors zip ops) {
      this.runner.feed(pairs._2.asOutput, pairs._1)
    }
    this
  }

  def fetch(operation: String): SessionRunner = {
    this.runner.fetch(operation)
    this
  }

  def fetch(operand: Operand[_]): SessionRunner = {
    this.runner.fetch(operand)
    this
  }

  def fetchSeqStr(operations: String*): SessionRunner = {
    for (operation <- operations) {
      this.runner.fetch(operation)
    }
    this
  }

  def fetchSeq(operands: Operand[_]*): SessionRunner = {
    for (operand <- operands) {
      this.runner.fetch(operand)
    }
    this
  }

  def fetch[T <: TType](operands: ju.List[Operand[T]]): SessionRunner = {
    operands.forEach { operand =>
      this.runner.fetch(operand)
    }
    this
  }

  def run: ju.List[Tensor] = this.runner.run
}
