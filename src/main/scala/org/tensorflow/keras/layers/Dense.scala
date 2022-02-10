// converted and adapted from TensorFlow; originally published under Apache 2.0 license
// Scala code published under LGPL v2.1+

package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.framework.initializers.Initializer
import org.tensorflow.keras.activations.Activation
import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.Initializers
import org.tensorflow.keras.utils.TensorShape
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.types.family.TNumber

object Dense {
  private val DENSE_INPUT_LENGTH  = 1
  private val KERNEL              = "kernel"
  private val KERNEL_INIT         = "kernelInit"
  private val BIAS                = "bias"
  private val BIAS_INIT           = "biasInit"
}

class Dense[T <: TNumber](
                           units              : Int,
                           activation         : Option[Activation[T]] = None,
                           useBias            : Boolean = true,
                           kernelInitializer  : Initializer[T] = Initializers.select(Initializers.glorotUniform),
                           biasInitializer    : Initializer[T] = Initializers.select(Initializers.zeros),
                           kernelRegularizer  : Option[Nothing] = None,
                           biasRegularizer    : Option[Nothing] = None,
                           activityRegularizer: Option[Nothing] = None,
                           kernelConstraint   : Option[Nothing] = None,
                           biasConstraint     : Option[Nothing] = None,
)
  extends Layer[T](Dense.DENSE_INPUT_LENGTH) {

  // weight tensors
  private var kernel: Variable[T] = null
  private var bias  : Variable[T] = null

  override def build(tf: Ops, inputShape: Shape): Unit = { // Check that final dimension is known
    new TensorShape(inputShape).assertKnown(inputShape.numDimensions - 1)
    // Retrieve Layer's dtype
    val dtype = this.getDtype
    // Compute shapes of kernel and bias matrices
    val kernelShape = Shape.of(inputShape.size(inputShape.numDimensions - 1), this.units)
    val biasShape = Shape.of(this.units)
    // Create dense kernel tensor
    this.kernel = this.addWeight(tf, Dense.KERNEL, tf.variable(kernelShape, dtype), Dense.KERNEL_INIT, this.kernelInitializer)
    // Create bias tensor
    this.bias = this.addWeight(tf, Dense.BIAS, tf.variable(biasShape, dtype), Dense.BIAS_INIT, this.biasInitializer)
    // Create Activation
    this.activation.foreach(_.build(tf, computeOutputShape(inputShape), dtype))
  }

  override def computeOutputShape(inputShape: Shape): Shape = // leaves unknown dimensions unknown
    new TensorShape(inputShape).replaceLast(this.units).toShape

  override final def call(tf: Ops, inputs: Seq[Operand[T]], training: Option[Boolean]): Operand[T] =
    this.callOne(tf, inputs.head)

  private def callOne(tf: Ops, input: Operand[T]): Operand[T] = {
    val signal = tf.math.add(tf.linalg.matMul(input, this.kernel), this.bias)
    this.activation.fold[Operand[T]](signal)(_.apply(tf, signal))
  }

  //    public static class Options<T extends Number> {
  //        // Default parameters
  //        private Activation<T> activation;
  //        private Initializer<T> kernelInitializer;
  //        private Initializer<T> biasInitializer;
  //
  //        public static <T extends Number> Options<T> defaults() {
  //            return new Builder<T>(new Options<>())
  //                    .setActivation(Activations.linear)
  //                    .setKernelInitializer(Initializers.randomNormal)
  //                    .setBiasInitializer(Initializers.zeros)
  //                    .build();
  //        }
  //        public static <T extends Number> Builder<T> builder() {
  //            return new Builder<>(defaults());
  //        public static class Builder<T extends Number> {
  //            private Options<T> options;
  //            public Builder(Options<T> options) {
  //                this.options = options;
  //            }
  //            public Builder<T> setActivation(Activations activation) {
  //                return setActivation(Activations.select(activation));
  //            public Builder<T> setActivation(Activation<T> activation) {
  //                this.options.activation = activation;
  //                return this;
  //            public Builder<T> setKernelInitializer(Initializers kernelInitializer) {
  //                return setKernelInitializer(Initializers.select(kernelInitializer));
  //            public Builder<T> setKernelInitializer(Initializer<T> kernelInitializer) {
  //                this.options.kernelInitializer = kernelInitializer;
  //            public Builder<T> setBiasInitializer(Initializers biasInitializer) {
  //                return setBiasInitializer(Initializers.select(biasInitializer));
  //            public Builder<T> setBiasInitializer(Initializer<T> biasInitializer) {
  //                this.options.biasInitializer = biasInitializer;
  //            public Options<T> build() {
  //                return this.options;
  //    }
}