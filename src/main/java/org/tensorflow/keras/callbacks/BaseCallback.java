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
        if (epochInterval > 0  && epoch % epochInterval == 0) {
            System.out.println("Epoch #" + epoch + " ");
        }
    }

    @Override
    public void onEpochEnd(int epoch, EpochEndLogs logs) {
        if (epochInterval > 0 && epoch % epochInterval == 0) {
            if (logs.trainAccuracy != null || logs.trainLoss != null)
                System.out.println("acc = " + logs.trainAccuracy + " loss = " + logs.trainLoss);
            if (logs.valAccuracy != null || logs.valLoss != null)
                System.out.println("val acc = " + logs.valAccuracy + " val loss = " + logs.valLoss);
        }
    }

    @Override
    public void onBatchBegin(int batch, BatchBeginLogs logs) {
        if (batchInterval > 0 && batch % batchInterval == 0) {
            System.out.print("Training batch #" + batch + ", size = " + logs.batchSize );
        }
    }

    @Override
    public void onBatchEnd(int batch, BatchEndLogs logs) {
        if (batchInterval > 0 && batch % batchInterval == 0) {
            System.out.println(" acc = " + logs.batchAccuracy + " loss = " + logs.batchLoss);
        }
    }

    @Override
    public void onTrainBegin() {}

    @Override
    public void onTrainEnd() {}
}