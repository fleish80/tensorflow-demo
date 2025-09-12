/*
  Lab 03-03 remarks
  - IMPORTANT (core logic): minMaxNormalize(xs) computes y = (x - min) / (max - min).
    It is a pure function on tensors and has no UI/DOM awareness.
  - DISPLAY (rendering): Everything inside tf.ready().then(...) is only for showing
    results on the page: parameters, shapes, x→y pairs, and memory status.
*/

// IMPORTANT: Core computation (pure). Returns both ys and the min/max used.
function minMaxNormalize(xs) {
    const min = xs.min();
    const max = xs.max();
    const range = max.sub(min);
    const ys = xs.sub(min).div(range);
    return { ys, min, max };
}

/*
  IMPORTANT: prepareDemoData()
  Creates demo input tensor and runs normalization. Separated from rendering so 
  it can be reused or unit-tested independently.
*/
function prepareDemoData() {
    const xs = tf.tensor1d([25, 76, 4, 23, -5, 22]);
    const { ys, min, max } = minMaxNormalize(xs);
    return { xs, ys, min, max };
}

// DISPLAY: Render everything to the page when TF.js is ready.
tf.ready().then(() => {
    // Cache UI elements.
    const tensorStatusSpan = document.getElementById('status-tensors');
    const minSpan = document.getElementById('param-min');
    const maxSpan = document.getElementById('param-max');
    const equationBadge = document.getElementById('equation');
    const xsText = document.getElementById('xs-text');
    const ysText = document.getElementById('ys-text');
    const xsShapeBadge = document.getElementById('xs-shape');
    const ysShapeBadge = document.getElementById('ys-shape');
    const pairsBody = document.getElementById('pairs-body');

    // IMPORTANT: Compute demo data using pure logic functions.
    const { xs, ys, min, max } = prepareDemoData();

    // DISPLAY: Show parameters and equation with concrete values.
    const minValue = min.dataSync()[0];
    const maxValue = max.dataSync()[0];
    minSpan.textContent = String(minValue);
    maxSpan.textContent = String(maxValue);
    equationBadge.textContent = `y = (x − ${minValue}) / (${maxValue} − ${minValue})`;

    // DISPLAY: Render tensors and shapes.
    xsText.textContent = xs.toString();
    ysText.textContent = ys.toString();
    xsShapeBadge.textContent = `shape [${xs.shape.join(', ')}]`;
    ysShapeBadge.textContent = `shape [${ys.shape.join(', ')}]`;

    // DISPLAY: Build pairs table x → y (materialize small arrays on CPU for clarity).
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

    // DISPLAY: Show tensor count once.
    tensorStatusSpan.textContent = String(tf.memory().numTensors);

    // DISPLAY: Console output for reference during development.
    ys.print();
}).catch(() => {
    // DISPLAY: Basic error fallback for the status line.
    const tensorStatusSpan = document.getElementById('status-tensors');
    if (tensorStatusSpan) {
        tensorStatusSpan.textContent = 'Error';
    }
});