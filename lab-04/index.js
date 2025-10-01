/*
  Lab 04 — Linear relationship exploration with TensorFlow.js

  What this file does:
  - Loads a housing dataset (King County) via tf.data.csv.
  - Extracts a single feature (sqft_living) as X and price as Y.
  - Visualizes the raw scatter of X vs Y with tfjs-vis.
  - Converts arrays to 2D tensors shaped [numExamples, 1].
  - Performs Min-Max normalization on features and labels:
      normalized = (value - min) / (max - min)
  - Visualizes normalized data as a scatter plot.
  - Demonstrates how to de-normalize values back to the original scale.

  Notes on data handling:
  - tf.data.Dataset is lazy/streaming; using toArray() materializes all data into memory.
  - dataSync() synchronously blocks to read Tensor values; for non-blocking use data().
  - This example favors clarity over performance; for large datasets, prefer batching/streaming.
*/



tf.ready().then(async () => {
    // Load the CSV as a streaming dataset. Each element is an object keyed by column names.
    // Example element: { id: 7129300520, date: '20141013T000000', price: 221900, sqft_living: 1180, ... }
    // We only use numeric columns directly. Non-numeric columns may need parsing.
    const houseSalesDataset = tf.data.csv('/lab-04/kc_house_data.csv');

    // Peek at a few rows for sanity. take(10) limits to the first 10 rows.
    // toArray() collects the dataset into memory; fine for preview, but avoid on huge data.
    console.log(await houseSalesDataset.take(10).toArray());

    // Map each row into a simple point { x, y } where
    //  - x := sqft_living (feature)
    //  - y := price (label)
    // This remains a Dataset of points until we call toArray().
    const points = houseSalesDataset.map(record => ({
       x: record.sqft_living,
       y: record.price
    }));

    const pointsArray = await points.toArray();
    // Ensure an even number of examples so tf.split(..., 2) yields equal halves.
    if (pointsArray.length % 2 !== 0) {
        pointsArray.pop();
    }
    // Shuffle to randomize train/test halves before splitting.
    tf.util.shuffle(pointsArray);

    // Visualize raw (unnormalized) data. We materialize the dataset to an array for plotting.
    plotData(pointsArray, 'Square feet');

    // Features (inputs)
    // Collect just the X values. Note: variable name has a small typo kept intentionally.
    const feauturesValues = pointsArray.map(record => record.x);
    // Create a 2D tensor with shape [numExamples, 1] for model-friendly shape.
    const featureTensor =  tf.tensor2d(feauturesValues, [feauturesValues.length, 1]);

    // Labels (outputs)
    const labelsValues = pointsArray.map(record => record.y);
    const labelsTensor =  tf.tensor2d(labelsValues, [labelsValues.length, 1]);

    // Normalize features and labels using Min-Max scaling.
    // Returns an object { tensor, min, max } so we can later de-normalize results.
    const normalizedFeatures = minMaxNormalize(featureTensor);
    const normalizedLabels = minMaxNormalize(labelsTensor);

    const [trainFeatures, testFeatures] = tf.split(normalizedFeatures.tensor, 2);
    const [trainLabels, testLabels] = tf.split(normalizedLabels.tensor, 2);
    // Inspect train/test splits in console (debug). Pass true to print values.
    trainFeatures.print(true);
    testFeatures.print(true);
    trainLabels.print(true);
    testLabels.print(true);


    // Build normalized scatter points as plain numbers
    // dataSync(): synchronously reads tensor values into a TypedArray (blocks until
    // any pending GPU work finishes). Returns a 1D, flattened view of the data.
    // Note: prefer data() for non-blocking reads; use dataSync() sparingly in perf-critical code.
    const xNorm = normalizedFeatures.tensor.dataSync();
    const yNorm = normalizedLabels.tensor.dataSync();
    const normalizedPoints = Array.from({ length: xNorm.length }, (_, i) => ({
        x: xNorm[i],
        y: yNorm[i]
    }));

    // Visualize normalized data. The absolute scales are removed, patterns remain.
    plotData(normalizedPoints, 'Normalized Square feet');

    // Demonstrate how to map normalized values back to original scale.
    const deNormalizedFeatures = deMinMaxNormalize(normalizedFeatures.tensor, normalizedFeatures.min, normalizedFeatures.max);
    const deNormalizedLabels = deMinMaxNormalize(normalizedLabels.tensor, normalizedLabels.min, normalizedLabels.max);


}).catch(error => {
    // Basic error handling for dataset loading or tensor operations
    console.error('Error loading dataset:', error);
});

async function plotData(pointsArray, featureName) {
    // Render a scatter plot using tfjs-vis. The library expects:
    //  - values: an array of series, where each series is an array of {x, y} points.
    //  - series: optional names for each series to show in the legend.
    tfvis.render.scatterplot(
        {name: `${featureName} vs Housing Price`},
        {values: [pointsArray], series: ['original']},
        {
            xLabel: featureName,
            yLabel: 'Price',
        });
}

function minMaxNormalize(tensor) {
    // Compute columnwise min and max: for a [N, 1] tensor, these are scalars.
    // For higher-rank tensors, broadcasting applies when subtracting/dividing.
    const min = tensor.min();
    const max = tensor.max();
    const range = max.sub(min);
    // Apply Min-Max scaling: (x - min) / (max - min).
    // We return the min and max so that predictions made in normalized space can be
    // mapped back to the original scale later (de-normalization).
    return {
        tensor: tensor.sub(min).div(range),
        min,
        max
    }
}

function deMinMaxNormalize(tensor, min, max) {
    // Inverse of Min-Max scaling: x = normalized * (max - min) + min
    const range = max.sub(min);
    return tensor.mul(range).add(min);
}