package org.tensorflow.keras.data;


import java.util.ArrayList;
import java.util.List;

/**
 * Batched Da
 */
public class Dataset {
    public static class Split {
        public float[][] X;
        public float[][] y;

        public Split(float[][] X, float[][] y) {
            assert X.length == y.length;

            this.X = X;
            this.y = y;
        }

        List<Split> batches(int batchSize) {
            int numBatches = X.length / batchSize;

            List<Split> batches = new ArrayList<>();

            for (int i = 0; i < numBatches; i++) {
                Split batch = getBatch(batchSize, i);
                batches.add(batch);
            }

            return batches;
        }

        Iterable<Split> batchIterator(int batchSize) {
            return () -> batches(batchSize).iterator();
        }

        /**
         *  Retrieve a particular batch given a batch size
         */
        Split getBatch(int batchSize, int i) {
            assert i < X.length / batchSize;
            return extractRange(batchSize * i, batchSize);
        }

        int size() {
            return X.length;
        }

        Split extractRange(int start, int length) {
            float[][] XRange = new float[length][X[0].length];
            float[][] YRange = new float[length][X[0].length];

            for (int j = start; j < start + length; j++) {
                XRange[j - start] = X[j];
                YRange[j - start] = y[j];
            }

            return new Split(XRange, YRange);
        }

        Dataset divide(double splitFactor) {
            int headSize = (int) splitFactor * size();
            int tailSize = size() - headSize;

            Split train = extractRange(0, headSize);
            Split test = extractRange(headSize, tailSize);

            return new Dataset(train, test);
        }
    }

    Split train;
    Split test;

    public Dataset(float[][] XTrain, float[][] YTrain, float[][] XTest, float[][] YTest) {
        train = new Split(XTrain, YTrain);
        test = new Split(XTest, YTest);
    }

    public Dataset(Split train, Split test) {
        this.train = train;
        this.test = test;
    }

    public List<Split> trainBatches(int batchSize) {
        return train.batches(batchSize);
    }

    public Iterable<Split> trainBatchIterator(int batchSize) {
        return train.batchIterator(batchSize);
    }

    public List<Split> testBatches(int batchSize) {
        return test.batches(batchSize);
    }

    public Iterable<Split> testBatchIterator(int batchSize) {
        return test.batchIterator(batchSize);
    }
}


//public class Dataset<X, Y> {
//    public static class Split<X, Y> {
//        public X[] X;
//        public Y[] y;
//
//        public Split(X[] X, Y[] y) {
//            assert X.length == y.length;
//
//            this.X = X;
//            this.y = y;
//        }
//
//        List<Split<X, Y>> batches(int batchSize) {
//            int numBatches = X.length / batchSize;
//
//            List<Split<X, Y>> batches = new ArrayList<>();
//
//            for (int i = 0; i < numBatches; i++) {
//                Split<X, Y> batch = getBatch(batchSize, i);
//                batches.add(batch);
//            }
//
//            return batches;
//        }
//
//        Iterable<Split<X, Y>> batchIterator(int batchSize) {
//            return () -> batches(batchSize).iterator();
//        }
//
//        /**
//         *  Retrieve a particular batch given a batch size
//         */
//        Split<X, Y> getBatch(int batchSize, int i) {
//            assert i < X.length / batchSize;
//            return extractRange(batchSize * i, batchSize);
//        }
//
//        int size() {
//            return X.length;
//        }
//
//        Split<X, Y> extractRange(int start, int length) {
//            List<X> XRange = new ArrayList<>();
//            List<Y> YRange = new ArrayList<>();
//
//            for (int j = start; j < start + length; j++) {
//                XRange.add(X[j]);
//                YRange.add(y[j]);
//            }
//
//            return new Split<>((X[]) XRange.toArray(), (Y[]) YRange.toArray());
//        }
//
//        Dataset<X, Y> divide(double splitFactor) {
//            int headSize = (int) splitFactor * size();
//            int tailSize = size() - headSize;
//
//            Split<X, Y> train = extractRange(0, headSize);
//            Split<X, Y> test = extractRange(headSize, tailSize);
//
//            return new Dataset<>(train, test);
//        }
//    }
//
//    Split<X, Y> train;
//    Split<X, Y> test;
//
//    public Dataset(X[] XTrain, Y[] YTrain, X[] XTest, Y[] YTest) {
//        train = new Split<>(XTrain, YTrain);
//        test = new Split<>(XTest, YTest);
//    }
//
//    public Dataset(Split<X, Y> train, Split<X, Y> test) {
//        this.train = train;
//        this.test = test;
//    }
//
//    public List<Split<X, Y>> trainBatches(int batchSize) {
//        return train.batches(batchSize);
//    }
//
//    public Iterable<Split<X, Y>> trainBatchIterator(int batchSize) {
//        return train.batchIterator(batchSize);
//    }
//
//    public List<Split<X, Y>> testBatches(int batchSize) {
//        return test.batches(batchSize);
//    }
//
//    public Iterable<Split<X, Y>> testBatchIterator(int batchSize) {
//        return test.batchIterator(batchSize);
//    }
//}