package org.tensorflow.keras.data;

import org.junit.jupiter.api.Test;
import org.tensorflow.*;
import org.tensorflow.op.Ops;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.stream.StreamSupport;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorSplitTest {
  TensorSplit<Float> split;

  @Test
  public void setUp() {
    int batchSize = 2;
    float[][] X = {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9},
      {10, 11, 12},
      {13, 14, 15},
      {17, 18, 19},
      {20, 21, 22},
      {23, 24, 25},
      {27, 28, 29},
      {30, 31, 32},
      {33, 34, 35},
      {36, 37, 38}
    };

    float[][] y = {
      {0, 2, 1, 0},
      {1, 2, 0, 0},
      {0, 2, 1, 0},
      {0, 2, 0, 1},
      {0, 2, 1, 0},
      {1, 2, 0, 0},
      {0, 2, 1, 0},
      {0, 2, 0, 1},
      {0, 2, 1, 0},
      {1, 2, 0, 0},
      {0, 2, 1, 0},
      {0, 2, 0, 1},
    };

    Tensor<Float> XTensor = Tensors.create(X);
    Tensor<Float> yTensor = Tensors.create(y);
    split = new TensorSplit<>(XTensor, yTensor, Float.class);

    try (Graph graph = new Graph()) {
      Ops tf = Ops.create(graph);

      split.build(tf, batchSize);
      Iterable<Operand<Float>> iterable = () -> split.XBatchIterator();
      long count = StreamSupport.stream(iterable.spliterator(), false).count();

      try (Session sess = new Session(graph)) {
        Iterable<Operand<Float>> batches = () -> split.XBatchIterator();
        int xcount = 0;
        for (Operand<Float> batch : batches) {
          List<Tensor<?>> t =
              sess.runner()
                  .feed(split.getXOp().asOutput(), split.getX())
                  .feed(split.getyOp().asOutput(), split.getY())
                  .fetch(batch)
                  .run();

          assertEquals(batchSize * 3, t.get(0).numElements());
          assertArrayEquals(new long[] {batchSize, 3}, t.get(0).shape());

          FloatBuffer fb = FloatBuffer.allocate(t.get(0).numElements());
          t.get(0).writeTo(fb);
          fb.position(0);

          for (int i = 0; i < batchSize; i++) {
            float[] arr = new float[3];
            fb.get(arr);
            assertArrayEquals(X[xcount * batchSize + i], arr);
          }
          xcount++;
        }

        int ycount = 0;
        Iterable<Operand<Float>> yBatches = () -> split.yBatchIterator();
        for (Operand<Float> batch : yBatches) {
          List<Tensor<?>> t =
              sess.runner()
                  .feed(split.getXOp().asOutput(), split.getX())
                  .feed(split.getyOp().asOutput(), split.getY())
                  .fetch(batch)
                  .run();

          assertEquals(batchSize * 4, t.get(0).numElements());
          assertArrayEquals(new long[] {batchSize, 4}, t.get(0).shape());

          FloatBuffer fb = FloatBuffer.allocate(t.get(0).numElements());
          t.get(0).writeTo(fb);
          fb.position(0);

          for (int i = 0; i < batchSize; i++) {
            float[] arr = new float[4];
            fb.get(arr);
            assertArrayEquals(y[ycount * batchSize + i], arr);
          }
          ycount++;
        }
      }
    }
  }
}
