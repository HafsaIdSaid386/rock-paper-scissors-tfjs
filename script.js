// ===== Rock • Paper • Scissors Classifier – script.js =====

// Global model variable
let model = null;

// IMPORTANT: labels must be in the SAME ORDER as training
// In Colab we saw: ['paper', 'rock', 'scissors']
const LABELS = ["paper", "rock", "scissors"];

// -------- 1. Load the TFJS model --------
async function loadModel() {
  try {
    model = await tf.loadLayersModel("rps_tfjs_model/model.json");
    console.log("✅ Model loaded");
    const resultEl = document.getElementById("result");
    if (resultEl && !resultEl.innerText) {
      resultEl.innerText = "Model loaded. Please upload an image.";
    }
  } catch (err) {
    console.error("❌ Error loading model:", err);
    const resultEl = document.getElementById("result");
    if (resultEl) {
      resultEl.innerText = "Error loading model. Check console.";
    }
  }
}

loadModel();

// -------- 2. Handle image upload --------
const fileInput = document.getElementById("imageUpload");
const previewImg = document.getElementById("preview");
const resultText = document.getElementById("result");

if (fileInput) {
  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Show preview
    const imgURL = URL.createObjectURL(file);
    previewImg.src = imgURL;
    previewImg.style.display = "block";

    // When image is loaded, run prediction
    previewImg.onload = () => {
      URL.revokeObjectURL(imgURL); // free memory
      predict(previewImg);
    };
  });
} else {
  console.warn("⚠️ No element with id 'imageUpload' found in HTML.");
}

// -------- 3. Predict function --------
async function predict(imgElement) {
  if (!model) {
    if (resultText) resultText.innerText = "Model not loaded yet...";
    console.warn("Model not loaded yet");
    return;
  }

  // Convert image to tensor
  let tensor = tf.browser.fromPixels(imgElement)
    .resizeNearestNeighbor([224, 224])  // same size as training
    .toFloat()
    .div(255.0)                         // normalize 0–1
    .expandDims();                      // shape [1, 224, 224, 3]

  // Run prediction
  const prediction = model.predict(tensor);
  const probs = prediction.dataSync();  // JS array of 3 numbers

  // Find index of max probability
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }

  const label = LABELS[maxIdx];

  if (resultText) {
    resultText.innerHTML = `Prediction: <b>${label.toUpperCase()}</b>`;
  }
  console.log("Predictions:", probs, "=>", label);
}
