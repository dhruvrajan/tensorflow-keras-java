package org.tensorflow.data;

import org.junit.jupiter.api.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.utils.Tensors;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GraphModeTensorFrameTest {
    @Test
    void testGraphModeSession() {
        int batchSize = 2;
        try (Graph graph = new Graph();
             Tensor XTensor = Tensors.create(
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

             Tensor yTensor = Tensors.create(new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8});
             GraphModeTensorFrame<TFloat32> graphModeTensorFrame
                     = new GraphModeTensorFrame<>(TFloat32.class, XTensor, yTensor)) {

            Ops tf = Ops.create(graph);

            graphModeTensorFrame.build(tf);
            graphModeTensorFrame.batch(batchSize);

            Operand<TFloat32>[] operands = graphModeTensorFrame.getBatchOperands();
            Operand<TFloat32> XOp = operands[0];
            Operand<TFloat32> yOp = operands[1];
            try (Session session = new Session(graph)) {
                for (int b = 0; b < graphModeTensorFrame.numBatches(); b++) {
                    List<Tensor> batches = graphModeTensorFrame
                            .feedSessionRunner(session.runner(), b)
                            .fetch(XOp).fetch(yOp).run();

                    try (Tensor XBatch = batches.get(0);
                         Tensor yBatch = batches.get(1)) {
                        // System.out.println(XBatch.rank());
                        // float[] xarray = StdArrays.array1dCopyOf((FloatNdArray) XBatch);
                        // XXX TODO: this currently fails
                        float[][] xarray = StdArrays.array2dCopyOf((FloatNdArray) XBatch);

                        for (int i = 0; i < XBatch.shape().size(0); i++) {
                            for (int j = 0; j < XBatch.shape().size(1); j++) {
                                // float real = xarray[i * batchSize + j];
                                float real = xarray[i][j];
                                assertEquals(b * graphModeTensorFrame.batchSize() * XBatch.shape().size(1) + ((long) i * batchSize) + j + 1, real);
                            }
                        }

                    }
                }
            }
        }
    }
}