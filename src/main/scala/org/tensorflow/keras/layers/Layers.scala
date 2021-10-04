package org.tensorflow.keras.layers

import org.tensorflow.keras.activations.Activation
import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.Initializer
import org.tensorflow.keras.initializers.Initializers
import org.tensorflow.keras.utils.Keras
import org.tensorflow.types.family.TNumber

object Layers {
  // Builders for Input Layer
  def input[T <: TNumber](firstDim: Long, units: Long*) =
    new Input[T](Keras.concatenate(firstDim, units: _*): _*)

  // Builders for Dense Layer
  def dense[T <: TNumber](units: Int) = new Dense[T](units, Dense.Options.defaults)

  def dense[T <: TNumber](units: Int, options: Dense.Options[T]) = new Dense[T](units, options)

  def dense[T <: TNumber](units: Int, activation: Activation[T]) =
    new Dense[T](units, Dense.Options.builder[T].setActivation(activation).build)

  def dense[T <: TNumber](units: Int, activation: Activations) =
    new Dense[T](units, Dense.Options.builder[T].setActivation(activation).build)

  def dense[T <: TNumber](units: Int, activation: Activations, kernelInitializer: Initializers,
                          biasInitializer: Initializers) =
    new Dense[T](units, Dense.Options.builder[T]
      .setActivation(activation)
      .setKernelInitializer(kernelInitializer)
      .setBiasInitializer(biasInitializer).build
    )

  def dense[T <: TNumber](units: Int, activation: Activation[T], kernelInitializer: Initializer,
                          biasInitializer: Initializer) =
    new Dense[T](units, Dense.Options.builder[T]
      .setActivation(activation)
      .setKernelInitializer(kernelInitializer)
      .setBiasInitializer(biasInitializer).build
    )

  // Builders for Flatten Layer
  def flatten[T <: TNumber] = new Flatten[T]

  // Builders for Conv2D Layer
  def conv2D(filters: Int, kernelSize: Seq[Int], options: Conv2D.Options = Conv2D.Options()) =
    new Conv2D(filters = filters, kernelSize = kernelSize, options = options)
}
