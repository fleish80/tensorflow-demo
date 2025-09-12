tf.ready().then(() => {
    console.log('TensorFlow.js loaded');
    const tensorCountSpan = document.getElementById('tensor-count');
    tensorCountSpan.textContent = tf.memory().numTensors;
}).catch(error => {
    const tensorCountSpan = document.getElementById('tensor-count');
    tensorCountSpan.textContent = 'Error';
    tensorCountSpan.style.color = 'red';
});

function createTensors1() {
    const xs = tf.tensor1d([1, 2, 3]);
    const ys = xs.mul(tf.scalar(5));
    const tensor1Span = document.getElementById('tensor-1');
    tensor1Span.textContent = ys;
}

function createTensors2() {
    const xs = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    const ys = xs.mul(tf.scalar(5));
    const tensor2Span1 = document.getElementById('tensor-2-1');
    tensor2Span1.textContent = xs;
    const tensor2Span2 = document.getElementById('tensor-2-2');
    tensor2Span2.textContent = ys;
}

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

tf.tidy(() => {
    createTensors1();
    createTensors2();
    createTensors3();
});