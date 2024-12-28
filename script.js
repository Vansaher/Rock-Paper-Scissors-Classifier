let model;
const webcamElement = document.getElementById('webcam');
const startWebcamButton = document.getElementById('startWebcam');
const outputElement = document.getElementById('output');

// Load the model
async function loadModel() {
  try {
    model = await tf.loadGraphModel('./model/model.json'); // Path to your model.json
    console.log('Model Loaded!');
  } catch (error) {
    console.error('Error loading model:', error);
    alert('Failed to load the model. Please check the path and model files.');
  }
}

// Setup the webcam
async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamElement.srcObject = stream;
    console.log('Webcam is ready.');
  } catch (error) {
    console.error('Error accessing webcam:', error);
    alert('Could not access the webcam. Please check your permissions.');
  }
}

// Run predictions
async function predict() {
  if (!model) {
    console.error('Model is not loaded yet.');
    return;
  }

  const webcamCapture = tf.browser.fromPixels(webcamElement);
  const resizedCapture = tf.image.resizeBilinear(webcamCapture, [224, 224]); // Match the model's input size (Teachable Machine)
  const normalizedCapture = resizedCapture.div(255).expandDims(0); // Normalize and add batch dimension

  const predictions = await model.predict(normalizedCapture).data();
  const classes = ['Rock', 'Paper', 'Scissors']; // Replace with your own class names

  // Find the class with the highest probability
  const maxIndex = predictions.indexOf(Math.max(...predictions));
  outputElement.innerText = `Prediction: ${classes[maxIndex]}`;

  // Dispose tensors to free up memory
  webcamCapture.dispose();
  resizedCapture.dispose();
  normalizedCapture.dispose();
}

// Main function to start everything
startWebcamButton.addEventListener('click', async () => {
  await loadModel();  // Load the model first
  await setupWebcam();  // Setup webcam
  setInterval(predict, 100);  // Run predictions every 100ms
});
