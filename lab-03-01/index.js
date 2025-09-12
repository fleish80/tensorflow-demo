/*
  TensorFlow.js Lab 03-01
  - Demonstrates basic tensor creation and math operations.
  - Shows current number of tensors in memory once TF.js is ready.
  - Wraps demo calls in tf.tidy to automatically free intermediate tensors.

  Note: Tensors whose references are kept (e.g., assigned to DOM for display)
  will persist outside tidy. Dispose them manually if you no longer need them.
*/

tf.ready().then(() => {
    console.log('TensorFlow.js loaded');
    const tensorCountSpan = document.getElementById('tensor-count');
    tensorCountSpan.textContent = tf.memory().numTensors;
}).catch(error => {
    const tensorCountSpan = document.getElementById('tensor-count');
    tensorCountSpan.textContent = 'Error';
    tensorCountSpan.style.color = 'red';
});

// Example 1: 1D tensor → scalar multiply
// xs: [1, 2, 3] (shape [3])
// ys = xs * 5
function createTensors1() {
    const xs = tf.tensor1d([1, 2, 3]);
    const ys = xs.mul(tf.scalar(5));
    const tensor1Span = document.getElementById('tensor-1');
    // tf.Tensor has a human-readable toString(); assigning to textContent renders it
    tensor1Span.textContent = ys;
}

// Example 2: 2D tensor → scalar multiply
// xs: [[1, 2], [3, 4], [5, 6]] (shape [3, 2])
// ys = xs * 5
function createTensors2() {
    const xs = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const ys = xs.mul(tf.scalar(5));
    const tensor2Span1 = document.getElementById('tensor-2-1');
    tensor2Span1.textContent = xs;
    const tensor2Span2 = document.getElementById('tensor-2-2');
    tensor2Span2.textContent = ys;
}

// Example 3: elementwise add two 2D tensors (same shape)
// xs20, xs21: [[1, 2], [3, 4], [5, 6]] (shape [3, 2])
// ys = xs20 + xs21
function createTensors3() {
    const xs20 = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const xs21 = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);


    const ys = xs20.add(xs21);
    const tensor3Span1 = document.getElementById('tensor-3-1');
    tensor3Span1.textContent = xs20;
    const tensor3Span2 = document.getElementById('tensor-3-2');
    tensor3Span2.textContent = xs21;
    const tensor3Span3 = document.getElementById('tensor-3-3');
    tensor3Span3.textContent = ys;
}

// Run all examples inside a tidy to release intermediate tensors automatically.
// Keep in mind: any tensors you store globally or attach to the DOM will persist.
tf.tidy(() => {
    createTensors1();
    createTensors2();
    createTensors3();
});