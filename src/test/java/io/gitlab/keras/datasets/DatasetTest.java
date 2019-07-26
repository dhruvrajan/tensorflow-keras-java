package io.gitlab.keras.datasets;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Iterator;

import static org.junit.jupiter.api.Assertions.*;

class DatasetTest {

    @BeforeEach
    void setUp() {
    }

    @Test
    void testDataset() {
        float[][] X = {
                {0f, 1f},
                {2f, 3f},
                {4f, 5f},
                {6f, 7f},
                {8f, 9f},
                {10f, 11f}
        };

        float[][] y = {
                {0f, 2f},
                {4f, 6f},
                {8f, 10f},
                {12f, 14f},
                {16f, 18f},
                {20f, 22f}
        };

        Dataset.Split split = new Dataset.Split(X, y);

        for (Dataset.Split batch : split.batchIterator(2)) {
            float[][] xbatch = batch.X;
            float[][] ybatch = batch.y;
            assertEquals(2, batch.size());
        }

    }

    @AfterEach
    void tearDown() {
    }
}