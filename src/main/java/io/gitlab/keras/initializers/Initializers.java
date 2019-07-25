package io.gitlab.keras.initializers;

import java.nio.FloatBuffer;

public enum Initializers {
    zeros;

    public static Initializer<Float> select(Initializers initializer) {
        switch (initializer) {
            case zeros:
                return new Zeros<Float>().dtype(Float.class);
            default:
                throw new IllegalArgumentException("invalid initializer type");
        }
    }
}
