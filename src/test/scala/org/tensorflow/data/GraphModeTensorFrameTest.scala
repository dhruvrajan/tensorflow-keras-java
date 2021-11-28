package org.tensorflow.data

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.tensorflow.{Graph, Session}
import org.tensorflow.ndarray.{FloatNdArray, StdArrays}
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import org.tensorflow.utils.Tensors

class GraphModeTensorFrameTest {
  @Test private[data] def testGraphModeSession(): Unit = {
    val batchSize = 2
    try {
      val graph = new Graph
      val XTensor = Tensors.create(Array[Array[Float]](
        Array( 1,  2,  3),
        Array( 4,  5,  6),
        Array( 7,  8,  9),
        Array(10, 11, 12),
        Array(13, 14, 15),
        Array(16, 17, 18),
        Array(19, 20, 21),
        Array(22, 23, 24),
        Array(25, 26, 27)
      ))
      val yTensor = Tensors.create(Array[Float](0, 1, 2, 3, 4, 5, 6, 7, 8))
      val graphModeTensorFrame = new GraphModeTensorFrame(classOf[TFloat32], XTensor, yTensor)
      try {
        val tf = Ops.create(graph)
        graphModeTensorFrame.build(tf)
        graphModeTensorFrame.batch(batchSize)
        val operands = graphModeTensorFrame.getBatchOperands
        val XOp = operands(0)
        val yOp = operands(1)
        try {
          val session = new Session(graph)
          try {
            var b = 0
            while ( {
              b < graphModeTensorFrame.numBatches
            }) {
              val batches = graphModeTensorFrame.feedSessionRunner(session.runner, b).fetch(XOp).fetch(yOp).run
              try {
                val XBatch = batches.get(0)
                val yBatch = batches.get(1)
                try { // System.out.println(XBatch.rank());
                  // float[] xarray = StdArrays.array1dCopyOf((FloatNdArray) XBatch);
                  // XXX TODO: this currently fails
                  val xarray = StdArrays.array2dCopyOf(XBatch.asInstanceOf[FloatNdArray])
                  var i = 0
                  while ( {
                    i < XBatch.shape.size(0)
                  }) {
                    var j = 0
                    while ({
                      j < XBatch.shape.size(1)
                    }) { // float real = xarray[i * batchSize + j];
                      val real = xarray(i)(j)
                      assertEquals(b * graphModeTensorFrame.batchSize * XBatch.shape.size(1) + (i.toLong * batchSize) + j + 1, real)

                      j += 1
                    }

                    i += 1
                  }
                } finally {
                  if (XBatch != null) XBatch.close()
                  if (yBatch != null) yBatch.close()
                }
              }

              b += 1
            }
          }
          finally {
            if (session != null) session.close()
          }
        }
      } finally {
        if (graph   != null) graph  .close()
        if (XTensor != null) XTensor.close()
        if (yTensor != null) yTensor.close()
        if (graphModeTensorFrame != null) graphModeTensorFrame.close()
      }
    }
  }
}