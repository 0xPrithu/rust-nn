# rust-nn
# Prerequisites

Before diving into this project, it's helpful to have some basic
understanding of a few key topics:

-   **Rust programming**: Familiarity with Rust syntax, variables,
    functions, slices, and basic data structures will make the code
    easier to follow.

-   **Logic gates**: Knowing how simple logic gates like AND, OR, and
    NOT work will help you understand the learning examples used here.

-   **Basic math concepts**: Understanding simple algebra and functions
    will help you grasp how neurons combine inputs, weights, and biases.

-   **General programming concepts**: Loops, conditionals, and iterators
    will be used in the Rust code.

If you're new to any of these areas, I recommend reviewing beginner
tutorials on Rust and digital logic before proceeding. This will make
the learning process smoother and more enjoyable!

# Introduction

Neural networks sound complex, but at their core, they are just
mathematical functions that learn patterns. In this project, we build a
simple neural network from scratch in Rust to answer the question:
*"Should I wear a jacket today?"*

**Why Rust?** Rust gives us low-level control, safety, and performance
--- making it an exciting language for exploring machine learning
fundamentals without relying on high-level libraries.

# Understanding a Neuron

In biology, a neuron is a special cell in the brain and nervous system
that sends and receives signals. It gets messages from other neurons
through its branches called dendrites, adds them up in its body, and if
the message is strong enough, sends a signal out through its axon to
other neurons. This helps living things react, make decisions, and
learn.

In computers, people copied this idea to make artificial neurons.
Instead of branches, an artificial neuron takes inputs (numbers),
multiplies them by weights (which show how important each input is), and
adds a bias (which changes how sensitive the neuron is).\
**A single neuron does three things:**

1.  Takes inputs ($x_1, x_2, \dots, x_n$).

2.  Multiplies them by weights ($w_1, w_2, \dots, w_n$).

3.  Adds a bias $b$ and passes the result through an activation
    function: $$y = \sigma \Big( \sum_{i=1}^{n} w_i x_i + b \Big)$$

# Why Multiply Inputs by Weights?

Not all inputs are equally important when making a decision. Weights
tell the neuron how much attention to give each input.

For example, when deciding whether to wear a jacket:

-   \- Temperature might be very important (higher temperature usually
    means no jacket),

-   \- Wind might be somewhat important,

-   \- Rain might be very important (rain almost always means wear a
    jacket).

By multiplying inputs by weights, the neuron learns to \"care more\"
about some factors and \"care less\" about others, making its decision
more accurate.\

``` {style="ruststyle" caption="Example: Should I wear a jacket?"}
let inputs = [temperature, wind, rain];
let weights = [-0.5, 0.3, 0.9];  // Importance of each factor

// Without weights (all inputs treated equally)
let bad_decision = temperature + wind + rain;
// 20 + 5 + 0 = 25 → Always wear a jacket??? That's not right!

// With weights (importance considered)
let good_decision = (temperature * -0.5) + (wind * 0.3) + (rain * 0.9);
// (20 * -0.5) + (5 * 0.3) + (0 * 0.9) = -10 + 1.5 + 0 = -8.5
// Negative means probably don't wear a jacket on a warm day
```

# Why Add Bias?

Even after multiplying inputs by their weights, a neuron might still not
give the right output. This is where the **bias** comes in --- it shifts
the neuron's output up or down, changing its default tendency.

Think of it like a personal preference:

-   A person who easily feels cold might need very little convincing to
    wear a jacket (high positive bias).

-   A person who prefers being warm might need much stronger reasons
    before deciding to wear one (negative bias).

Adding bias makes the neuron more flexible. Without it, the neuron
always starts at zero --- but with bias, it can adjust to different
starting points.

``` {style="ruststyle" caption="Effect of bias"}
let result = (inputs × weights) + bias;

// Example: You're naturally cold-sensitive (bias = +2.0)
// Even on mild days, you tend to wear a jacket.

// Or, you're naturally warm (bias = -2.0)
// You need more convincing to wear a jacket.
```

# Adding Non-Linearity: Sigmoid Activation

After multiplying inputs by weights and adding a bias, the neuron's raw
output can be *any* number --- very large, very small, or even negative.
But in most cases, we want the output to stay within a fixed range (for
example, between 0 and 1). This is where an **activation function**
comes in.

The **sigmoid function** squashes any number into a range between 0 and
1, which makes it:

-   Easier to interpret (0 = "don't activate", 1 = "fully activate").

-   Useful for probabilities (like: "How likely is it that I need a
    jacket?").

-   Smoother for learning --- small changes in inputs lead to small
    changes in output.

``` {style="ruststyle" caption="Sigmoid activation function"}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Example: Raw neuron output = 10  → Sigmoid = 0.999
// Example: Raw neuron output = -5  → Sigmoid ≈ 0.007
// Example: Raw neuron output = 0   → Sigmoid = 0.5
```

Using the sigmoid function makes the neuron behave more like a
"decision-maker" rather than just a calculator. To constrain our output
between 0 and 1, we use the **sigmoid function**:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

# Building a Jacket Predictor Neuron

As mentioned in the introduction, our goal is to make a simple predictor
that decides whether you should wear a jacket. This decision depends on
factors such as:

-   **Temperature** (colder temperatures make jackets more likely)

-   **Wind speed** (wind increases the need for a jacket)

-   **Rain** (rain almost always means wearing a jacket)

We can represent this as a single neuron:
$$y = \sigma \Big( (w_1 \times \text{temperature}) + (w_2 \times \text{wind}) + (w_3 \times \text{rain}) + b \Big)$$
Where:

-   $w_1, w_2, w_3$ are the weights (how important each factor is),

-   $b$ is the bias (our default tendency toward wearing a jacket),

-   $\sigma$ is the sigmoid activation (Sigmoid Function/Activator).

``` {style="ruststyle" caption="A complete neuron for jacket prediction."}
fn complete_neuron(inputs: &[f64], weights: &[f64], bias: f64) -> f64 {
    let weighted_sum = inputs.iter() // iterating through the input parameter
        .zip(weights.iter()) // grouping inputs alongside their corresponding weights into a tuple like structure 
        .map(|(input, weight)| input * weight) // once grouped, '.map' multiplies them (ex: (2.0 x 0.5))
        .sum::<f64>() + bias; // simple AND gate logic to sum up all values and add a bias to them
    sigmoid(weighted_sum) // sigmoid function to limit values from 0 and 1 only
}
```

Now this neuron outputs a value between 0 and 1:

-   Values closer to 1 mean "yes, wear a jacket".

-   Values closer to 0 mean "no, don't wear a jacket".

By tweaking the weights and bias, we can make this neuron behave like a
basic decision‑maker.

# Testing the Jacket Predictor Neuron

To see our neuron in action, we test it with real weather values. Each
input (temperature, wind, rain) is multiplied by its corresponding
weight, then a bias is added. This produces a *decision score* between 0
and 1 using the sigmoid function --- values closer to 1 mean "wear a
jacket."\
\
\

``` {style="ruststyle" caption="Testing the jacket predictor neuron."}
fn test_single_neuron() {
    println!("TESTING A SINGLE NEURON");

    // WEATHER TEST SET 1 
    
    let weather_set_1 = [15.0, 5.0, 0.0]; // 15°C, 5mph wind, no rain
    let importance = [-0.1, 0.3, 0.8];    // Weight of each factor
    let bias = 2.0;                        // Default tendency toward wearing a jacket
    let decision = complete_neuron(&weather_today, &importance, bias);

    let weather_set_2 = [5.0, 10.0, 1.0]; // Cold, windy, rainy
    let decision2 = complete_neuron(&cold_rainy, &importance, bias);
    
}
```

By adjusting the weights and bias, we can make the neuron prioritize
different factors (e.g., caring more about rain than wind), resulting in
more realistic decisions.

# Connecting Multiple Neurons: Building a Layer

A single neuron can only learn one simple pattern. To learn more complex
patterns, we connect multiple neurons together in a **layer**. Each
neuron in the layer processes the same inputs independently, but with
its own unique weights and bias.

``` {style="ruststyle" caption="A simple layer of neurons with explanations."}
struct SimpleLayer {
    neuron_weights: Vec<Vec<f64>>, // Each neuron has its own vector of weights
    neuron_biases: Vec<f64>,       // Each neuron has its own bias
}

impl SimpleLayer {
    // Creates a new layer with `num_neurons` neurons, each having `num_inputs` inputs
    fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let mut neuron_weights = Vec::new();
        let mut neuron_biases = Vec::new();
        
        // For each neuron, generate random weights and a bias
        for _ in 0..num_neurons {
            // Create a vector of weights (one per input), random between -1 and 1
            let weights: Vec<f64> = (0..num_inputs)
                .map(|_| fastrand::f64() * 2.0 - 1.0)
                .collect();
            neuron_weights.push(weights);
            
            // Generate a random bias for this neuron
            neuron_biases.push(fastrand::f64() * 2.0 - 1.0);
        }
        
        // Return the layer with all neuron weights and biases initialized
        SimpleLayer { neuron_weights, neuron_biases }
    }
    
    // Passes inputs through the layer, returning outputs from each neuron
    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut outputs = Vec::new();
        
        // Each neuron processes the same inputs with its own weights and bias
        for i in 0..self.neuron_weights.len() {
            let output = complete_neuron(inputs, &self.neuron_weights[i], self.neuron_biases[i]);
            outputs.push(output);
        }
        
        outputs  // Return outputs of all neurons as a vector
    }
}
```

Each neuron applies its unique weights and bias to the inputs, then
produces one output. Collectively, the layer outputs a vector of values
--- one per neuron --- allowing the network to detect multiple features
or patterns simultaneously.

# Stacking Layers: Building a Simple Network

A single layer can detect multiple features, but often we want to
**combine** those features into more complex decisions. We do this by
**stacking layers**, where the output of one layer becomes the input of
the next. This forms a small **feedforward network**.

In this example, the network has:

-   **Layer 1** (hidden layer): 3 neurons, each receiving 2 inputs.

-   **Layer 2** (output layer): 1 neuron, receiving the 3 outputs from
    the hidden layer.

``` {style="ruststyle" caption="A simple two-layer neural network."}
struct SuperSimpleNetwork {
    layer1: SimpleLayer, // First layer (hidden layer)
    layer2: SimpleLayer, // Second layer (output layer)
}

impl SuperSimpleNetwork {
    // Creates a new network with 2 inputs, 3 hidden neurons, and 1 output neuron
    fn new() -> Self {
        // Hidden layer: 3 neurons, each with 2 inputs
        let layer1 = SimpleLayer::new(3, 2);
        
        // Output layer: 1 neuron, taking 3 inputs (outputs of hidden layer)
        let layer2 = SimpleLayer::new(1, 3);
        
        SuperSimpleNetwork { layer1, layer2 }
    }
    
    // Passes data through the network and returns the final prediction
    fn predict(&self, inputs: &[f64]) -> f64 {
        // Step 1: Feed inputs into the first layer (hidden layer)
        let hidden_outputs = self.layer1.forward(inputs);
        
        // Step 2: Feed hidden layer outputs into the second layer (output layer)
        let final_outputs = self.layer2.forward(&hidden_outputs);
        
        // Step 3: Since output layer has only one neuron, return its value
        final_outputs[0]
    }
}
```

The data flows in one direction:
$$\text{Inputs} \rightarrow \text{Hidden Layer} \rightarrow \text{Output Layer} \rightarrow \text{Prediction}$$
By stacking layers, the network can learn to combine basic features
detected in the hidden layer into more abstract and complex patterns in
the output layer.

# Making the Network Learn

So far, our networks have random weights and biases --- meaning they
make random predictions. To make them useful, we need to **teach** them
using a process called **training**. In its simplest form, training
means:

1.  Show the network an example input and the correct answer.

2.  Compare its prediction to the correct answer (find the *error*).

3.  Adjust the weights and bias to reduce that error.

4.  Repeat many times.

Here is a minimal network that can **learn** a simple pattern using a
very basic learning rule:

``` {style="ruststyle" caption="A tiny network that learns from examples."}
struct LearningNetwork {
    weights: Vec<f64>,  // Connection strengths
    bias: f64,          // Threshold offset
    learning_rate: f64, // Controls how big each adjustment is
}

impl LearningNetwork {
    // Create with random weights and bias
    fn new() -> Self {
        LearningNetwork {
            weights: vec![
                fastrand::f64() * 2.0 - 1.0, // Random between -1 and 1
                fastrand::f64() * 2.0 - 1.0
            ],
            bias: fastrand::f64() * 2.0 - 1.0,
            learning_rate: 0.1,
        }
    }
    
    // Make a prediction for given inputs
    fn predict(&self, inputs: &[f64]) -> f64 {
        complete_neuron(inputs, &self.weights, self.bias)
    }
    
    // Update weights and bias based on one example
    fn learn_from_example(&mut self, inputs: &[f64], correct_answer: f64) {
        // 1. Predict output
        let prediction = self.predict(inputs);
        
        // 2. Calculate error (difference from correct answer)
        let error = correct_answer - prediction;
        
        // 3. Adjust each weight
        for i in 0..self.weights.len() {
            self.weights[i] += self.learning_rate * error * inputs[i];
        }
        
        // 4. Adjust the bias
        self.bias += self.learning_rate * error;
    }
}
```

The key idea is that **error** tells the network *which direction* to
adjust each weight:

-   If the prediction is too low, increase weights and bias.

-   If the prediction is too high, decrease them.

By repeating this process over many examples, the network gradually
improves its predictions --- this is the essence of **learning**.

# Results

We trained the network to learn the AND gate. Before training, outputs
were random and did not match the correct answers. After 1000 training
epochs, the outputs closely matched the expected truth table.

``` {style="ruststyle" caption="Training progress for the AND gate."}
BEFORE TRAINING:
  [0.0, 0.0] -> 0.442  (should be 0.0)
  [0.0, 1.0] -> 0.517  (should be 0.0)
  [1.0, 0.0] -> 0.391  (should be 0.0)
  [1.0, 1.0] -> 0.631  (should be 1.0)

AFTER 1000 EPOCHS:
  [0.0, 0.0] -> 0.031  (close to 0.0)
  [0.0, 1.0] -> 0.056  (close to 0.0)
  [1.0, 0.0] -> 0.048  (close to 0.0)
  [1.0, 1.0] -> 0.945  (close to 1.0)
```

We can see that:

-   Outputs for false cases moved close to **0**.

-   Output for the true case moved close to **1**.

# Reflection

Through this project, I learned that:

1.  At their core, neural networks are nothing more than **weights**,
    **biases**, and a bit of simple math --- but arranged in a powerful
    way.

2.  Activation functions such as the sigmoid are essential, as they
    allow networks to learn non-linear patterns.

3.  It is entirely possible to implement a working machine learning
    model from scratch in Rust, without relying on external libraries.

Looking ahead, there are several ways this project could be improved:

-   Implement **backpropagation** to train multi-layer networks more
    effectively.

-   Expand the dataset to handle more complex, real-world problems.

-   Build a small CLI tool to make interactive predictions directly from
    the terminal.

# Conclusion

In this project, we designed, implemented, and trained a simple neural
network in Rust entirely from scratch. While intentionally minimal, it
captures the **fundamental ideas** behind artificial intelligence:
learning from examples, adjusting parameters, and improving over time.

This work serves as a solid foundation for exploring more advanced
architectures, optimization techniques, and applications --- proving
that even complex ideas can start from simple, hand-built code.
