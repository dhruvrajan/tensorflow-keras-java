//package io.gitlab.keras.examples.iris;
//
//import io.gitlab.keras.datasets.Dataset;
//import io.gitlab.keras.datasets.Iris;
//import io.gitlab.keras.layers.Dense;
//import io.gitlab.keras.layers.InputLayer;
//import io.gitlab.keras.models.Sequential;
//import io.gitlab.keras.losses.SoftmaxCrossEntropyLoss;
//import io.gitlab.keras.metrics.Accuracy;
//import io.gitlab.keras.optimizers.GradientDescentOptimizer;
//import org.tensorflow.Graph;
//import org.tensorflow.Shape;
//import org.tensorflow.op.Ops;
//
//import java.io.IOException;
//import java.util.List;
//
//public class IrisClassifier {
//    final static int BATCH_SIZE = 5;
//
//    public static void main(String[] args) throws IOException {
//        try (Graph graph = new Graph()) {
//            Ops tf = Ops.create(graph);
//
//            Dataset data = Iris.loadData(BATCH_SIZE, 0.3);
//
//            Sequential model = new Sequential(
//                    new InputLayer(4, BATCH_SIZE),
//                    new Dense(3, Shape.make(BATCH_SIZE, 4))
//            );
//
//            model.compile(tf, new GradientDescentOptimizer((float) 0.06), new SoftmaxCrossEntropyLoss(), new Accuracy());
//            model.fit(graph, data.XTrain, data.YTrain, data.XTest, data.YTest, 100);
//        }
//    }
//}
