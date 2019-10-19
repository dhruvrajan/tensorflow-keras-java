package org.tensorflow.keras.callbacks;

import org.tensorflow.keras.logs.*;

public class BaseCallback extends Callback {
    @Override
    public void onEpochBegin(int epoch, EpochBeginLogs logs) {
        //System.out.println("Begin Epoch: " + epoch);
    }

    @Override
    public void onEpochEnd(int epoch, EpochEndLogs logs) {
        System.out.println("Finished Epoch " + epoch + " acc = " + logs.trainAccuracy + " loss = " + logs.trainLoss);
    }

    @Override
    public void onBatchBegin(int batch, BatchBeginLogs logs) {
        if (batch % 100 == 0) {
//            System.out.print("Training batch #" + batch + ", size = " + logs.batchSize);
        }
    }

    @Override
    public void onBatchEnd(int batch, BatchEndLogs logs) {
        if (batch % 100 == 0) {
//            System.out.println("acc = " + logs.batchAccuracy + " loss = " + logs.batchLoss);
        }
    }

    @Override
    public void onTrainBegin() {}

    @Override
    public void onTrainEnd() {}
}