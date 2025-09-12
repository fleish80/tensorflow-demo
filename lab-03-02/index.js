/*
  Lab 03-02 remarks
  - IMPORTANT: getYs(xs, m, c) is the core linear model y = m·x + c.
              It uses vectorized TensorFlow.js ops and has no side effects.
  - DISPLAY: Everything inside tf.ready().then(...) prepares demo inputs
             and renders values to the page for visualization.
*/

// IMPORTANT: Core computation (pure function). Uses vectorized ops; no loops needed.
function getYs(xs, m, c) {
    return xs.mul(m).add(c);
}

/*
  IMPORTANT: computeLinearDemo()
  Returns demo inputs and outputs for y = m·x + c.
  This function is REALLY what matters (the core math). The rest below renders
  the values to the page for visualization only.
*/
function computeLinearDemo() {
    const xs = tf.tensor1d([1, 5, 10]);
    const slopeM = 2;
    const interceptC = 1;
    const ys = getYs(xs, slopeM, interceptC);
    return { xs, ys, slopeM, interceptC };
}

tf.ready().then(() => {
    // DISPLAY: Cache UI elements for rendering results.
    const tensorStatusSpan = document.getElementById('status-tensors');
    const mSpan = document.getElementById('param-m');
    const cSpan = document.getElementById('param-c');
    const equationBadge = document.getElementById('equation');
    const xsText = document.getElementById('xs-text');
    const ysText = document.getElementById('ys-text');
    const xsShapeBadge = document.getElementById('xs-shape');
    const ysShapeBadge = document.getElementById('ys-shape');
    const pairsBody = document.getElementById('pairs-body');

    // IMPORTANT: Core demo values (inputs and result) computed separately.
    const { xs, ys, slopeM, interceptC } = computeLinearDemo();

    // DISPLAY: Show parameters and formatted equation.
    mSpan.textContent = String(slopeM);
    cSpan.textContent = String(interceptC);
    equationBadge.textContent = `y = ${slopeM}·x + ${interceptC}`;

    // DISPLAY: Render tensors as human-readable strings and show their shapes.
    xsText.textContent = xs.toString();
    ysText.textContent = ys.toString();

    xsShapeBadge.textContent = `shape [${xs.shape.join(', ')}]`;
    ysShapeBadge.textContent = `shape [${ys.shape.join(', ')}]`;

    // DISPLAY: Build a small table of x → y pairs for clarity.
    //          arraySync() materializes values on CPU; fine for small demos,
    //          avoid for very large tensors.
    pairsBody.innerHTML = '';
    const xsArray = xs.arraySync();
    const ysArray = ys.arraySync();
    for (let index = 0; index < xsArray.length; index++) {
        const tr = document.createElement('tr');
        const tdIdx = document.createElement('td');
        const tdX = document.createElement('td');
        const tdY = document.createElement('td');
        tdIdx.textContent = String(index + 1);
        tdX.textContent = String(xsArray[index]);
        tdY.textContent = String(ysArray[index]);
        tr.appendChild(tdIdx);
        tr.appendChild(tdX);
        tr.appendChild(tdY);
        pairsBody.appendChild(tr);
    }

    // DISPLAY: Show current tensor count once after rendering.
    tensorStatusSpan.textContent = String(tf.memory().numTensors);

    // DISPLAY: Print to console for reference.
    ys.print();
}).catch(() => {
    // DISPLAY: Basic error fallback for the status line.
    const tensorStatusSpan = document.getElementById('status-tensors');
    if (tensorStatusSpan) {
        tensorStatusSpan.textContent = 'Error';
    }
});