package org.tensorflow.utils

import org.tensorflow.keras.layers.{Conv2D, Layers}
import org.tensorflow.ndarray.Shape

object Implicits {
  implicit class ShapeOps(private val shape: Shape) extends AnyVal {
    /** Aka `shape[-n:]` */
    def takeRight(n: Int): Shape = {
      val sz = shape.numDimensions()
      shape.subShape(sz - n, sz)
    }

    /** Aka `shape[:-n]` */
    def dropRight(n: Int): Shape = {
      val sz = shape.numDimensions()
      shape.subShape(0, sz - n)
    }

    /** Returns True iff `self` is fully defined in every dimension. */
    def isFullyDefined: Boolean = !shape.hasUnknownDimension
  }
}
