package org.tensorflow.data;

import org.junit.jupiter.api.Test;
import org.tensorflow.*;
import org.tensorflow.op.Ops;

import java.nio.FloatBuffer;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GraphModeTensorFrameTest {
    @Test
    void testGraphModeSession() {
        int batchSize = 2;
        try (Graph graph = new Graph();
             Tensor<Float> XTensor = Tensors.create(
                     new float[][]{
                             {1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9},
                             {10, 11, 12},
                             {13, 14, 15},
                             {16, 17, 18},
                             {19, 20, 21},
                             {22, 23, 24},
                             {25, 26, 27}
                     }
             );

             Tensor<Float> yTensor = Tensors.create(new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8});
             GraphModeTensorFrame<Float> graphModeTensorFrame
                     = new GraphModeTensorFrame<>(Float.class, XTensor, yTensor)) {

            Ops tf = Ops.create(graph);

            graphModeTensorFrame.build(tf);
            graphModeTensorFrame.batch(batchSize);

            Operand<Float>[] operands = graphModeTensorFrame.getBatchOperands();
            Operand<Float> XOp = operands[0];
            Operand<Float> yOp = operands[1];
            try (Session session = new Session(graph)) {
                for (int b = 0; b < graphModeTensorFrame.numBatches(); b++) {
                    List<Tensor<?>> batches = graphModeTensorFrame
                            .feedSessionRunner(session.runner(), b)
                            .fetch(XOp).fetch(yOp).run();

                    try (Tensor<?> XBatch = batches.get(0);
                         Tensor<?> yBatch = batches.get(1)) {
                        FloatBuffer XBuffer = FloatBuffer.allocate(3 * 2);
                        FloatBuffer yBuffer = FloatBuffer.allocate(2);

                        XBatch.writeTo(XBuffer);
                        yBatch.writeTo(yBuffer);

                        float[] xarray = XBuffer.array();

                        for (int i = 0; i < XBatch.shape()[0]; i++) {
                            for (int j = 0; j < XBatch.shape()[1]; j++) {
                                float real = xarray[i * batchSize + j];
                                assertEquals(b * graphModeTensorFrame.batchSize() * XBatch.shape()[1] + i * batchSize + j + 1, real);
                            }
                        }

                    }
                }
            }
        }
    }
}