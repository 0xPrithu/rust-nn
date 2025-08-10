use std::f64;

fn simple_neuron(inputs: &[f64]) -> f64 {
    inputs.iter().sum()
}

fn weighted_neuron(inputs: &[f64], weights: &[f64]) -> f64 {
    inputs.iter()
        .zip(weights.iter())
        .map(|(input, weight)| input * weight)
        .sum()
}

fn neuron_with_bias(inputs: &[f64], weights: &[f64], bias: f64) -> f64 {
    let weighted_sum = inputs.iter()
        .zip(weights.iter())
        .map(|(input, weight)| input * weight)
        .sum::<f64>();
    
    weighted_sum + bias
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn complete_neuron(inputs: &[f64], weights: &[f64], bias: f64) -> f64 {
    let weighted_sum = inputs.iter()
        .zip(weights.iter())
        .map(|(input, weight)| input * weight)
        .sum::<f64>() + bias;
    
    sigmoid(weighted_sum)
}

fn test_single_neuron() {
    println!("🧪 TESTING A SINGLE NEURON");
    println!("==========================");
    
    let weather_today = [15.0, 5.0, 0.0];
    let importance = [
        -0.1,
         0.3,
         0.8,
    ];
    
    let bias = 2.0;
    
    let decision = complete_neuron(&weather_today, &importance, bias);
    
    println!("Weather: {}°C, {}mph wind, {} rain", 
             weather_today[0], weather_today[1], weather_today[2]);
    println!("Decision score: {:.3}", decision);
    println!("Wear jacket? {}", if decision > 0.5 { "YES" } else { "NO" });
    
    println!("\nTrying different weather...");
    let cold_rainy = [5.0, 10.0, 1.0];
    let decision2 = complete_neuron(&cold_rainy, &importance, bias);
    println!("Cold & rainy decision: {:.3} -> {}", 
             decision2, if decision2 > 0.5 { "YES" } else { "NO" });
}

struct SimpleLayer {
    neuron_weights: Vec<Vec<f64>>,
    neuron_biases: Vec<f64>,
}

impl SimpleLayer {
    fn new(num_neurons: usize, num_inputs: usize) -> Self {
        let mut neuron_weights = Vec::new();
        let mut neuron_biases = Vec::new();
        
        for _ in 0..num_neurons {
            let weights: Vec<f64> = (0..num_inputs)
                .map(|_| fastrand::f64() * 2.0 - 1.0)
                .collect();
            neuron_weights.push(weights);
            neuron_biases.push(fastrand::f64() * 2.0 - 1.0);
        }
        
        SimpleLayer { neuron_weights, neuron_biases }
    }
    
    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut outputs = Vec::new();
        
        for i in 0..self.neuron_weights.len() {
            let output = complete_neuron(inputs, &self.neuron_weights[i], self.neuron_biases[i]);
            outputs.push(output);
        }
        
        outputs
    }
}

struct SuperSimpleNetwork {
    layer1: SimpleLayer,
    layer2: SimpleLayer,
}

impl SuperSimpleNetwork {
    fn new() -> Self {
        let layer1 = SimpleLayer::new(3, 2);
        let layer2 = SimpleLayer::new(1, 3);
        
        SuperSimpleNetwork { layer1, layer2 }
    }
    
    fn predict(&self, inputs: &[f64]) -> f64 {
        let hidden_outputs = self.layer1.forward(inputs);
        let final_outputs = self.layer2.forward(&hidden_outputs);
        final_outputs[0]
    }
}

fn test_network() {
    println!("\n🕸️ TESTING A SIMPLE NETWORK");
    println!("============================");
    
    let network = SuperSimpleNetwork::new();
    let test_inputs = [0.5, 0.8];
    let result = network.predict(&test_inputs);
    
    println!("Input: {:?}", test_inputs);
    println!("Network output: {:.3}", result);
    println!("(This is random right now because we haven't trained it!)");
}

struct LearningNetwork {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl LearningNetwork {
    fn new() -> Self {
        LearningNetwork {
            weights: vec![fastrand::f64() * 2.0 - 1.0, fastrand::f64() * 2.0 - 1.0],
            bias: fastrand::f64() * 2.0 - 1.0,
            learning_rate: 0.1,
        }
    }
    
    fn predict(&self, inputs: &[f64]) -> f64 {
        complete_neuron(inputs, &self.weights, self.bias)
    }
    
    fn learn_from_example(&mut self, inputs: &[f64], correct_answer: f64) {
        let prediction = self.predict(inputs);
        let error = correct_answer - prediction;
        
        for i in 0..self.weights.len() {
            self.weights[i] += self.learning_rate * error * inputs[i];
        }
        
        self.bias += self.learning_rate * error;
    }
}

fn watch_learning() {
    println!("\n🎓 WATCHING THE NETWORK LEARN");
    println!("==============================");
    
    let mut network = LearningNetwork::new();
    
    let examples = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 0.0),
        ([1.0, 0.0], 0.0),
        ([1.1, 1.0], 1.0),
    ];
    
    println!("Teaching it the AND gate...");
    println!("(Both inputs must be 1 to output 1)\n");
    
    println!("BEFORE TRAINING:");
    for (inputs, expected) in &examples {
        let prediction = network.predict(inputs);
        println!("  {:?} -> {:.3} (should be {:.1})", inputs, prediction, expected);
    }
    
    for epoch in 0..1000 {
        for (inputs, correct_answer) in &examples {
            network.learn_from_example(inputs, *correct_answer);
        }
        
        if epoch % 200 == 0 {
            println!("\nAfter {} training steps:", epoch);
            for (inputs, expected) in &examples {
                let prediction = network.predict(inputs);
                println!("  {:?} -> {:.3} (should be {:.1})", inputs, prediction, expected);
            }
        }
    }
    
    println!("\n🎉 Final results:");
    for (inputs, expected) in &examples {
        let prediction = network.predict(inputs);
        let close_enough = (prediction - expected).abs() < 0.1;
        let status = if close_enough { "✅" } else { "❌" };
        println!("  {} {:?} -> {:.3} (should be {:.1})", status, inputs, prediction, expected);
    }
}

fn main() {
    println!("🧠 NEURAL NETWORKS FROM ZERO");
    println!("=============================\n");
    
    test_single_neuron();
    test_network();
    watch_learning();
    
    println!("\n💡 WHAT YOU JUST SAW:");
    println!("====================");
    println!("1. A neuron is just: inputs × weights + bias, then squash");
    println!("2. A network is just: connect multiple neurons together");
    println!("3. Learning is just: adjust weights when you're wrong");
    println!("4. That's literally it! Everything else is just scaling this up.");
    
    println!("\n🚀 TO UNDERSTAND MORE:");
    println!("======================");
    println!("- Play with the weights and see what happens");
    println!("- Try teaching it different patterns (OR gate, etc.)");
    println!("- Change the learning rate and see how it affects training");
    println!("- Add more neurons and see if it can learn harder problems");
}
