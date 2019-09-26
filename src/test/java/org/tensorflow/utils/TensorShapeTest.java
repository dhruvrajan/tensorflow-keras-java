package org.tensorflow.utils;

import org.junit.jupiter.api.Test;
import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Batch;
import org.tensorflow.op.core.ConcatenateDataset;
import org.tensorflow.op.core.Placeholder;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TensorShapeTest {

  @Test
  void testBatch() {
    Tensor<Integer> t1 = Tensors.create(
            new int[]{ 2, 4, 6, 8 }
    );


    try (Graph graph = new Graph()) {
      Ops tf = Ops.create(graph);

      Placeholder<Integer> p1 = tf.placeholder(Integer.class, Placeholder.shape(getShape(t1.shape())));

      Operand<Integer> op = tf.constant(134);
      ConcatenateDataset
      Operand<Integer> list = tf.emptyTensorList(tf.constant(-1), tf.constant(2), Integer.class);
      list = (Operand<Integer>) tf.tensorListPushBack(list, op);


      Operand<Integer> op = tf.roll(p1, tf.constant(1), tf.constant(0));

      try (Session session = new Session(graph)) {
        try (Tensor<?> out = session.runner()
                .feed(p1.asOutput(), t1)
                .fetch(op)
                .run().get(0)) {

          IntBuffer intBuffer = IntBuffer.allocate(4);
          out.writeTo(intBuffer);

          System.out.println("ROLLED: " + Arrays.toString(intBuffer.array()));
        }
      }
    }

  }

  private static Shape getShape(long... dims) {
    assert dims.length > 0;

    long head = dims[0];
    long[] tail = new long[dims.length - 1];
    System.arraycopy(dims, 1, tail, 0, dims.length - 1);

    return Shape.make(head, tail);
  }
  @Test
  void testTensorShape() {
    TensorShape first = new TensorShape(1, 2, 3, 4);

    // Test for correct values
    assertEquals(4, first.rank());
    for (int i = 0; i < 4; i++) {
      first.assertKnown(i);
      assertTrue(first.isKnown(i));
      assertEquals(i + 1, first.get(i));
    }

    // Test concatenate
    first.concatenate(5, 6, 7, 8);

    // Test for correct values
    assertEquals(8, first.rank());
    for (int i = 0; i < 8; i++) {
      first.assertKnown(i);
      assertTrue(first.isKnown(i));
      assertEquals(i + 1, first.get(i));
    }

    // Test setters
    first.replace(3, -1);
    assertFalse(first.isKnown(3));
    assertThrows(IllegalStateException.class, () -> first.assertKnown(3));

    // Test fromShape
    Shape shape = Shape.make(5, 6, 7, 8);
    TensorShape second = new TensorShape(shape);
    assertEquals(shape, second.toShape());

    // Test Equals
    assertEquals(new TensorShape(4, 5, 6, 7), new TensorShape(4, 5, 6, 7));
    assertNotEquals(new TensorShape(1, 2, 3, 4), new TensorShape(4, 5, 6, 7));
  }
}
