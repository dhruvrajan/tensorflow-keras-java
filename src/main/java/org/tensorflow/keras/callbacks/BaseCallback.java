package org.tensorflow.keras.callbacks;

import org.tensorflow.keras.logs.*;

public class BaseCallback extends Callback {
    private int epochInterval;
    private int batchInterval;

    public BaseCallback(int epochInterval, int batchInterval) {
        this.epochInterval = epochInterval;
        this.batchInterval = batchInterval;
    }

    @Override
    public void onEpochBegin(int epoch, EpochBeginLogs logs) {
        if (epoch % epochInterval == 0) {
            System.out.println("Epoch #" + epoch + " ");
        }
    }

    @Override
    public void onEpochEnd(int epoch, EpochEndLogs logs) {
        if (epoch % epochInterval == 0) {
            System.out.println("acc = " + logs.trainAccuracy + " loss = " + logs.trainLoss);
        }
    }

    @Override
    public void onBatchBegin(int batch, BatchBeginLogs logs) {
        if (batch % batchInterval == 0) {
            System.out.print("Training batch #" + batch + ", size = " + logs.batchSize );
        }
    }

    @Override
    public void onBatchEnd(int batch, BatchEndLogs logs) {
        if (batch % batchInterval == 0) {
            System.out.println(" acc = " + logs.batchAccuracy + " loss = " + logs.batchLoss);
        }
    }

    @Override
    public void onTrainBegin() {}

    @Override
    public void onTrainEnd() {}
}