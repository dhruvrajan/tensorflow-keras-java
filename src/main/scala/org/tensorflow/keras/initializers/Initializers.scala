package org.tensorflow.keras.initializers

object Initializers extends Enumeration {
  type Initializers = Value

  val zeros, ones, randomNormal, glorotUniform = Value

  def select(initializer: Initializers): Initializer = initializer match {
    case `zeros` =>
      new Zeros
    case `ones` =>
      new Ones
    case `randomNormal` =>
      new RandomNormal(0.0f, 0.1f, -0.2f, 0.2f)
    case `glorotUniform` =>
      throw new UnsupportedOperationException("Glorot Uniform does not yet exist")
    case _ =>
      throw new IllegalArgumentException("invalid initializer type")
  }
}
