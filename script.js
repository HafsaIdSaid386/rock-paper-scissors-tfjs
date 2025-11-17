// ===== Rock • Paper • Scissors Classifier – script.js =====

// Global model
let model = null;

// Order of classes from training: ['paper', 'rock', 'scissors']
const LABELS = ["paper", "rock", "scissors"];

// ---------- 1. Load the TFJS model ----------
async function loadModel() {
  const resultEl = document.getElementById("result");

  try {
    // Path is RELATIVE to index.html
    model = await tf.loadLayersModel("rps_tfjs_model/model.json");
    console.log("✅ Model loaded");
    if (resultEl) {
      resultEl.innerText = "Model loaded. Please upload an image.";
    }
  } catch (err) {
    console.error("❌ Error loading model:", err);
    if (resultEl) {
      resultEl.innerText = "Error loading model (see console).";
    }
  }
}

loadModel();

// ---------- 2. Handle image upload ----------
const fileInput  = document.getElementById("imageUpload");
const previewImg = document.getElementById("preview");
const resultText = document.getElementById("result");

if (fileInput) {
  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.style.display = "block";

    previewImg.onload = () => {
      URL.revokeObjectURL(url);  // free memory
      predict(previewImg);
    };
  });
} else {
  console.warn("⚠️ No element with id='imageUpload' found.");
}

// ---------- 3. Predict function ----------
async function predict(imgElement) {
  if (!model) {
    if (resultText) resultText.innerText = "Model not loaded yet...";
    console.warn("Model not loaded yet");
    return;
  }

  // Convert image to tensor
  let tensor = tf.browser.fromPixels(imgElement)
    .resizeNearestNeighbor([224, 224]) // same as training
    .toFloat()
    // NOTE: your model already has a Rescaling(1/255) layer,
    // so this extra div(255) is optional. You can comment it out if needed.
    //.div(255.0)                      
    .expandDims();                     // [1, 224, 224, 3]

  const prediction = model.predict(tensor);
  const probs = prediction.dataSync(); // array of 3 numbers

  // Find max index
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }

  const label = LABELS[maxIdx];

  if (resultText) {
    resultText.innerHTML = `Prediction: <b>${label.toUpperCase()}</b>`;
  }

  console.log("🔎 probs:", probs, "→", label);
}
