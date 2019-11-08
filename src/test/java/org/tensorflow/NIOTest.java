package org.tensorflow;

import org.junit.jupiter.api.Test;
import org.tensorflow.data.NioTensorFrame;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.nd.IntNdArray;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Indices;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;
import static org.tensorflow.nio.StaticApi.*;
import static org.tensorflow.nio.nd.index.Indices.all;

public class NIOTest {
    @Test
    void test2() {
        // Allocating a 3D matrix of 2x3x2
        IntNdArray matrix3d = ndArrayOfInts(shape(4, 3, 2));
        assertEquals(3, matrix3d.rank());

        // Initializing 3D matrix data with vectors from the first dimension (index 0)
        matrix3d.elements(0).forEachIdx((idx, matrix) -> {
            assertEquals(2, matrix.rank());
            assertEquals(shape(3, 2), matrix.shape());
//            matrix.set(vector(1, 2, 3, 4), 0)
//                    .set(vector(5, 6, 7, 8), 1)
//                    .set(vector(9, 10, 11, 12), 2);
            int s = (int) idx[0];
            matrix.set(vector(1 + s, 2 + s), 0)
                    .set(vector(3 + s, 4 + s), 1)
                    .set(vector(5 + s, 6 + s), 2);
        });


        NioTensorFrame<Integer> tf = new NioTensorFrame<>(matrix3d);

        // Visit all scalars of 3D matrix, printing their coordinates and value
        tf.getBatch(3)[0].scalars().forEachIdx((coords, scalar) ->
                System.out.println("Scalar at " + Arrays.toString(coords) + " has value " + scalar.getValue())
        );
    }

    @Test
    void test() {
        // Allocating a 3D matrix of 2x3x2
        IntNdArray matrix3d = ndArrayOfInts(shape(2, 3, 2));
        assertEquals(3, matrix3d.rank());

        // Initializing 3D matrix data with vectors from the first dimension (index 0)
        matrix3d.elements(0).forEach(matrix -> {
            assertEquals(2, matrix.rank());
            assertEquals(shape(3, 2), matrix.shape());
            matrix.set(vector(1, 2), 0).set(vector(3, 4), 1).set(vector(5, 6), 2);
        });

//        matrix3d.slice(Indices.range())

        // Visit all scalars of 3D matrix, printing their coordinates and value
        matrix3d.scalars().forEachIdx((coords, scalar) ->
                System.out.println("Scalar at " + Arrays.toString(coords) + " has value " + scalar.getInt())
        );

        // Retrieving the second vector of the first matrix
        IntNdArray vector = matrix3d.get(0, 1);
        assertEquals(1, vector.rank());

        // Rewriting the values of the vector using a primitive array
        vector.write(new int[]{7, 8});
        assertEquals(7, matrix3d.getInt(0, 1, 0));
        assertEquals(8, matrix3d.getInt(0, 1, 1));

        // Slicing the 3D matrix so we only keep the second element of the second dimension
        IntNdArray slice = matrix3d.slice(all(), at(1));
        assertEquals(2, slice.rank());
//        assertEquals(shape(2, 2), slice.shape());
        assertEquals(7, slice.getInt(0, 0));  // (0, 1, 0) in the original matrix
        assertEquals(3, slice.getInt(1, 0));  // (1, 1, 0) in the original matrix
    }

    @Test
    void test3() {
        FloatDataBuffer buffer = bufferOfFloats(10);
//        buffer.put()
    }

}
