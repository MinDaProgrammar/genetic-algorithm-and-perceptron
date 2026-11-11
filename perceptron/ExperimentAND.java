/**
 * Trains a perceptron to learn the AND function.
 * Starting weights and learning rate match the lecture slides exactly
 * (W0_bias=-0.3, W1=0.1, W2=0.3, learningRate=0.1).
 *
 * <p>Usage:
 * <pre>
 *   javac Perceptron.java ExperimentAND.java
 *   java ExperimentAND [--verbose]
 * </pre>
 */
public class ExperimentAND {

    /**
     * Entry point. Trains the perceptron on the AND truth table and prints
     * the final weights and a verification table.
     *
     * @param args command-line arguments; pass {@code --verbose} to print
     *             per-pass training details
     */
    public static void main(String[] args) {
        boolean verbose = (args.length > 0 && args[0].equals("--verbose"));

        // AND truth table
        double[][] inputs  = { {0,0}, {0,1}, {1,0}, {1,1} };
        int[]      targets = {    0,     0,     0,     1  };

        // Starting weights from lecture slides (W0_bias=-0.3, W1=0.1, W2=0.3)
        double[] initialWeights = { -0.3, 0.1, 0.3 };
        double   learningRate   = 0.1;

        Perceptron p      = new Perceptron(initialWeights, learningRate, true);
        int        passes = p.train(inputs, targets);

        double[] w = p.getWeights();
        System.out.printf("%nDone — passes: %d  |  W0=%.4f  W1=%.4f  W2=%.4f%n",
                passes, w[0], w[1], w[2]);

        // Verify all outputs are correct
        System.out.println("\nVerification:");
        System.out.println("  x1    x2    output  target");
        for (int i = 0; i < inputs.length; i++) {
            double net    = w[0] + w[1]*inputs[i][0] + w[2]*inputs[i][1];
            int    output = (Math.round(net * 1e10) / 1e10 >= 0) ? 1 : 0;
            System.out.printf("  %.1f   %.1f   %d       %d%n",
                    inputs[i][0], inputs[i][1], output, targets[i]);
        }
    }
}
