package org.tensorflow.utils;

import org.junit.jupiter.api.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.TInt32;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TensorShapeTest {

    @Test
    void testBatch() {

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // env is an ExecutionEnvironment, such as a Graph instance.
            try (Session session = new Session(graph);
                 Tensor t1 = Tensors.create(new int[][] {{2, 4}, {6, 8}})) {
                Placeholder<TInt32> p1 = tf.placeholder(TInt32.class, Placeholder.shape(t1.shape()));
                Operand<TInt32> op = tf.reshape(p1, tf.constant(new int[] {-1}));
                List<Tensor> tensors =
                        session.runner().feed(p1.asOutput(), t1).fetch(op).run();

                Shape shape = Keras.shapeFromDims(tensors.get(0).shape().asArray());
                System.out.println("done");
            }
        }
    }

    private static Shape getShape(long... dims) {
        assert dims.length > 0;
        return Shape.of(dims);
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
        Shape shape = Shape.of(5, 6, 7, 8);
        TensorShape second = new TensorShape(shape);
        assertEquals(shape, second.toShape());

        // Test Equals
        assertEquals(new TensorShape(4, 5, 6, 7), new TensorShape(4, 5, 6, 7));
        assertNotEquals(new TensorShape(1, 2, 3, 4), new TensorShape(4, 5, 6, 7));
    }
}
