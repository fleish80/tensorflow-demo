// Simple TensorFlow.js example
// Create a simple model
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd'
});

// Generate some synthetic data for training
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

// Train the model
async function trainModel() {
  const outputElement = document.getElementById('micro-out-div');
  outputElement.innerText = 'Training...';
  
  await model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 10 === 0) {
          console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(5)}`);
        }
      }
    }
  });
  
  // Use the model to make predictions
  const testInput = tf.tensor2d([5, 10], [2, 1]);
  const prediction = model.predict(testInput);
  
  // Print the results
  outputElement.innerText = `Training complete!\n\n` +
    `Predicted values:\n` +
    `Input of 5: ${prediction.dataSync()[0].toFixed(2)}\n` +
    `Input of 10: ${prediction.dataSync()[1].toFixed(2)}\n\n` +
    `Actual function is y = 2x - 1`;
}

// Run the training when the page loads
window.onload = trainModel;