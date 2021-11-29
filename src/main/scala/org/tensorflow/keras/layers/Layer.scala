package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.framework.initializers.Initializer
import org.tensorflow.keras.initializers.Initializers
import org.tensorflow.keras.initializers.Initializers.Ops
import org.tensorflow.keras.mixin.LayerFunction
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.{Ops => TF}
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Variable
import org.tensorflow.types.family.TNumber

import java.{util => ju}

/**
  * Base layer class.
  *
  * <p>A layer implements common neural network operations, such as convolution, batch norm, etc.
  * These operations require managing weights, losses, updates, and inter-layer connectivity.
  *
  * @param < T> Numeric type of the output (Float, Double)
  */
abstract class Layer[T <: TNumber](val INPUTS_LENGTH: Int) extends LayerFunction[T] {
  protected var built = false
  final private val weights           = new ju.HashMap[String, Variable[T]]
  final private val initializerOpMap  = new ju.HashMap[String, Assign[T]]
  // Input() layer needs to access dtype and built.
  protected var dtype: Class[T] = null

  /**
    * Overrides create(Ops) to add variables (weight tensors) to the layer.
    *
    * The addWeight function and some tf ops require passing a Class<T> "dtype" object
    *
    * To get the dtype of this layer in the build function, use Layer.getDtype()
    *
    * @param tf         Tensorflow Ops accessor
    * @param inputShape Shape of the layer's input tensor
    *
    */
  protected def build(tf: TF, inputShape: Shape): Unit

  final def build(tf: TF, inputShape: Shape, dtype: Class[T]): Unit = {
    this.dtype = dtype
    build(tf, inputShape)
    this.built = true
  }

  /**
    * Computes the output shape of the tensor returned by a Layer from the input tensor's shape
    *
    * @param inputShape Shape of an input tensor to this layer
    * @return Shape of the tensor that would be returned by `apply`.
    */
  def computeOutputShape(inputShape: Shape): Shape

  /**
    * Defines the layer's logic, in terms of input operands, and variables.
    *
    * @param tf     Tensorflow Ops accessor.
    * @param inputs A sequence of TF Operands
    * @return The transformed input tensors, according to the layer's logic.
    */
  @SuppressWarnings(Array("unchecked")) protected def call(tf: TF, inputs: Operand[T]*): Operand[T]

  /**
    * Internal wrapper for Layer.call
    */
  override final def apply(tf: TF, inputs: Operand[T]*): Operand[T] = {
    if (!this.built) throw new IllegalStateException("Layer.call() cannot be called before the layer is built (Layer.build())")
    if (inputs.length != INPUTS_LENGTH) throw new IllegalArgumentException("Layer call() expected " + INPUTS_LENGTH + "inputs; received " + inputs.length + ".")
    this.call(tf, inputs: _*)
  }

  /**
    * Adds a new weight tensor to the layer
    *
    * @param name     variable name
    * @param variable variable to add
    * @return the created variable.
    */
  final protected def addWeight(name: String, variable: Variable[T]): Variable[T] = {
    this.weights.put(name, variable)
    variable
  }

//  final protected def addWeight(tf: Ops, name: String, variable: Variable[T], initializerName: String, initializer: Nothing): Variable[T] = addWeight(tf, name, variable, initializerName, Initializers.select(initializer))

  final protected def addWeight(tf: TF, name: String, variable: Variable[T], initializerName: String, initializer: Initializer[T]): Variable[T] = {
    this.weights.put(name, variable)
    initializerOpMap.put(initializerName, initializer.apply(tf, variable, dtype))
    variable
  }

  def initializerOps: ju.List[Operand[T]] = new ju.ArrayList[Operand[T]](initializerOpMap.values)

  def trainableWeights: ju.List[Variable[T]] = new ju.ArrayList[Variable[T]](this.weights.values)

  def isBuilt: Boolean = this.built

  def hasDtype: Boolean = this.dtype != null

  def getDtype: Class[T] = this.dtype
}
