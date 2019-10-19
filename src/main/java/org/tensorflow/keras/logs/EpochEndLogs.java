package org.tensorflow.keras.logs;

public class EpochEndLogs extends Logs {
    public float trainAccuracy;
    public float trainLoss;
    
    // Validation Logs
    public float valAccuracy;
    public float valLoss;
}