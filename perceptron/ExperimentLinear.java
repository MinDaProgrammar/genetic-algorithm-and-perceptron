/**
 * Trains a perceptron to classify 2D points as above or below the line y = 2x + 1.
 * Input is an (x, y) coordinate pair; output 1 means above the line, 0 means below.
 * Weights are initialised randomly in [-0.5, 0.5].
 *
 * <p>Usage:
 * <pre>
 *   javac Perceptron.java ExperimentLinear.java
 *   java ExperimentLinear [--verbose]
 * </pre>
 */
public class ExperimentLinear {

    /**
     * Entry point. Trains the perceptron on a set of labelled (x, y) points
     * and prints the final weights and a verification table.
     *
     * @param args command-line arguments; pass {@code --verbose} to print
     *             per-pass training details
     */
    public static void main(String[] args) {
        boolean verbose = (args.length > 0 && args[0].equals("--verbose"));

        // is the coordinate (x,y) above the y=2x+1? True / False
        // above (y > 2x+1) -> target 1,  below (y < 2x+1) -> target 0
        double[][] inputs  = { {0,0}, {0,2}, {2,0}, {2,6} };
        int[]      targets = {    0,     1,     0,     1  };

        // Random initial weights subtracted by 0.5 to make sure to have some negative values too.
        double[] initialWeights = {
            Math.random() - 0.5,   // W0 bias
            Math.random() - 0.5,   // W1
            Math.random() - 0.5    // W2
        };
        double learningRate = 0.1;

        System.out.printf("Initial weights: W0=%.4f  W1=%.4f  W2=%.4f%n",
                initialWeights[0], initialWeights[1], initialWeights[2]);

        Perceptron p      = new Perceptron(initialWeights, learningRate, verbose);
        int        passes = p.train(inputs, targets);

        double[] w = p.getWeights();
        System.out.printf("%nDone — passes: %d  |  W0=%.4f  W1=%.4f  W2=%.4f%n",
                passes, w[0], w[1], w[2]);

        // Verify all outputs are correct
        System.out.println("\nVerification:");
        System.out.println("  x      y      predicted    expected");
        for (int i = 0; i < inputs.length; i++) {
            double net    = w[0] + w[1]*inputs[i][0] + w[2]*inputs[i][1];
            int    output = (Math.round(net * 1e10) / 1e10 >= 0) ? 1 : 0;
            String predicted = (output == 1) ? "above" : "below";
            String expected  = (targets[i] == 1) ? "above" : "below";
            System.out.printf("  %.1f    %.1f    %-12s %s%n",
                    inputs[i][0], inputs[i][1], predicted, expected);
        }
    }
}
