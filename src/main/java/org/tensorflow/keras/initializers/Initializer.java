package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;

public abstract class Initializer<T> {
  Class<T> dtype;
  protected boolean built = false;

  public void build(Class<T> dtype) {
    this.dtype = dtype;
    this.built = true;
  }

  /**
   * Adds an `Assign` Op to the graph to initialize
   * a tensorflow variable as specified by the initializer.
   * @param tf Tensorflow Ops Accessor
   * @param in Variable to initialize
   * @return Assign Operand created
   */
  public Assign<T> apply(Ops tf, Operand<T> in) {
    return tf.assign(in, this.call(tf, Keras.shapeOperand(tf, in.asOutput().shape())));
  }

  /**
   * Returns a Tensor object initialized as
   * specified by the initializer.
   * @param tf Tensorflow Ops Handle
   * @param shape Shape of the tensor
   */
  public abstract Operand<T> call(Ops tf, Operand<Integer> shape);


  public boolean isBuilt() {
    return this.built;
  }
}
