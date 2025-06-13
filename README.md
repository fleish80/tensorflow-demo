# TensorFlow.js Demo

A simple demonstration of TensorFlow.js for machine learning in the browser.

## Overview

This project showcases a basic TensorFlow.js implementation that trains a simple linear regression model. The model learns to predict values based on a linear relationship (y = 2x - 1).

## Features

- **Simple Linear Regression**: Trains a neural network to learn a linear function
- **Browser-based**: Runs entirely in the browser using TensorFlow.js
- **Real-time Training**: Shows training progress with epoch updates
- **Prediction Demo**: Makes predictions on new data after training

## Files

- `tensorflow.html` - Main HTML file with TensorFlow.js CDN and UI
- `index.js` - JavaScript code containing the TensorFlow.js model and training logic

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/fleish80/tensorflow-demo.git
   cd tensorflow-demo
   ```

2. Open `tensorflow.html` in your web browser

3. Watch the training process and see the predictions!

## What the Model Does

The model is trained on synthetic data points:
- Input: [-1, 0, 1, 2, 3, 4]
- Output: [-3, -1, 1, 3, 5, 7]

The model learns the relationship y = 2x - 1 and can then predict values for new inputs.

## Training Details

- **Model Type**: Sequential neural network with one dense layer
- **Loss Function**: Mean Squared Error
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Epochs**: 100
- **Training Data**: 6 data points

## Technologies Used

- [TensorFlow.js](https://www.tensorflow.org/js) - Machine learning library for JavaScript
- HTML5 - Web page structure
- JavaScript - Programming logic

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to fork this repository and submit pull requests to improve the demo! 