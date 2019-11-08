package org.tensorflow.utils;

public class Tuple3<T> {
    private T first;
    private T second;
    private T third;

    public Tuple3(T first, T second, T third) {
        this.first = first;
        this.second = second;
        this.third = third;
    }

    public T first() {
        return first;
    }

    public T second() {
        return second;
    }

    public T third() {
        return third;
    }
}
