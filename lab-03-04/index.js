// Wait for TensorFlow.js to load and then display the number of tensors
tf.ready().then(() => {
    console.log('TensorFlow.js loaded');
    const tensorCountSpan = document.getElementById('tensor-count');
    tensorCountSpan.textContent = tf.memory().numTensors;
}).catch(error => {
    const tensorCountSpan = document.getElementById('tensor-count');
    tensorCountSpan.textContent = 'Error';
    tensorCountSpan.style.color = 'red';
});

/**
 * Creates multiple tensors to demonstrate memory usage
 * This function creates 100 tensors and performs operations on them
 */
function createTensors() {
    for (let i = 0; i < 100; i++) {
        // Create a 1D tensor with values [1, 2, 3]
        const tensor1 = tf.tensor([1, 2, 3]);
        // Create a scalar tensor with the current loop value
        const tensor2 = tf.scalar(i);
        // Multiply tensors and print the result
        tensor1.mul(tensor2).print();
    }
}

/**
 * Wraps tensor creation in tf.tidy() to automatically clean up memory
 * tf.tidy() automatically disposes of tensors when the function completes
 * This prevents memory leaks and keeps the tensor count low
 */
tf.tidy(() => {
    createTensors();
});