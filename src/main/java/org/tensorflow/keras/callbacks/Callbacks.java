package org.tensorflow.keras.callbacks;

public enum Callbacks {
    baseCallback;

    public static Callback select(Callbacks callbackType) {
        switch(callbackType) {
            case baseCallback:
                return new BaseCallback();
            default:
                throw new IllegalArgumentException("Invalid callback type");
        }
    }
}