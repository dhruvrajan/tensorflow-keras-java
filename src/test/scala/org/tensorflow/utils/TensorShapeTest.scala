package org.tensorflow.utils

import org.junit.jupiter.api.Assertions._
import org.junit.jupiter.api.Test
import org.tensorflow.keras.utils.{Keras, TensorShape}
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.TInt32
import org.tensorflow.{Graph, Session}

object TensorShapeTest {
  private def getShape(dims: Long*) = {
    assert(dims.length > 0)
    Shape.of(dims: _*)
  }
}

class TensorShapeTest {
  @Test private[utils] def testBatch(): Unit = {
    try {
      val graph = new Graph
      try {
        val tf = Ops.create(graph)
        // env is an ExecutionEnvironment, such as a Graph instance.
        try {
          val session = new Session(graph)
          val t1 = Tensors.create(Array[Array[Int]](Array(2, 4), Array(6, 8)))
          try {
            val p1      = tf.placeholder(classOf[TInt32], Placeholder.shape(t1.shape))
            val op      = tf.reshape(p1, tf.constant(Array[Int](-1)))
            val tensors = session.runner.feed(p1.asOutput, t1).fetch(op).run
            val shape   = Keras.shapeFromDims(tensors.get(0).shape.asArray(): _*)
            System.out.println("done")
          } finally {
            if (session != null) session.close()
            if (t1 != null) t1.close()
          }
        }
      } finally if (graph != null) graph.close()
    }
  }

  @Test private[utils] def testTensorShape(): Unit = {
    val first = new TensorShape(1, 2, 3, 4)
    // Test for correct values
    assertEquals(4, first.numDimensions())
    for (i <- 0 until 4) {
      first.assertKnown(i)
      assertTrue(first.isKnown(i))
      assertEquals(i + 1, first.size(i))
    }
    // Test concatenate
    first.concatenate(5, 6, 7, 8)
    assertEquals(8, first.numDimensions)
    for (i <- 0 until 8) {
      first.assertKnown(i)
      assertTrue(first.isKnown(i))
      assertEquals(i + 1, first.size(i))
    }
    // Test setters
    first.replace(3, -1)
    assertFalse(first.isKnown(3))
    assertThrows(classOf[IllegalStateException], () => first.assertKnown(3))
    // Test fromShape
    val shape = Shape.of(5, 6, 7, 8)
    val second = new TensorShape(shape)
    assertEquals(shape, second.toShape)
    // Test Equals
    assertEquals    (new TensorShape(4, 5, 6, 7), new TensorShape(4, 5, 6, 7))
    assertNotEquals (new TensorShape(1, 2, 3, 4), new TensorShape(4, 5, 6, 7))
  }
}
