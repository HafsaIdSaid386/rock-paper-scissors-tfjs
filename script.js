let model;

// Load model
async function loadModel() {
    model = await tf.loadLayersModel("rps_tfjs_model/model.json");
    console.log("Model loaded!");
}

loadModel();

// Handle image upload
document.getElementById("imageUpload").addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);

    img.onload = () => predict(img);
});

// Model prediction
async function predict(img) {
    if (!model) return;

    let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .expandDims()
        .toFloat()
        .div(255.0);

    let prediction = model.predict(tensor);
    let index = prediction.argMax(1).dataSync()[0];

    const labels = ["paper", "rock", "scissors"];

    document.getElementById("result").innerText =
        "Prediction: " + labels[index];
}
