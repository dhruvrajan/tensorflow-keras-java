package org.tensorflow.keras.logs;

public class EpochEndLogs extends Logs {
    public Float trainAccuracy;
    public Float trainLoss;
    
    // Validation Logs
    public Float valAccuracy;
    public Float valLoss;
}