import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

ml5.setBackend("webgl");

const nn = ml5.neuralNetwork({task: 'classification', debug: true});
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
};
nn.load(modelDetails, () => console.log("Model loaded successfully!"));

const enableWebcamButton = document.getElementById("webcamButton");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);

let handLandmarker = undefined;
let webcamRunning = false;

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    enableWebcamButton.addEventListener("click", enableCam);
};

async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

async function predictWebcam() {
    const results = await handLandmarker.detectForVideo(video, performance.now());
    const predictionDiv = document.getElementById('predictionDiv');

    if (results.landmarks.length === 0) {
        predictionDiv.innerHTML = '';
    } else {
        const hand = results.landmarks[0];
        if (hand) {
            const handData = hand.map((point) => [point.x, point.y, point.z]).flat();

            const prediction = await nn.classify(handData);

            if (prediction && prediction.length > 0) {
                const topPrediction = prediction.reduce((max, p) => p.confidence > max.confidence ? p : max);

                console.log("Top prediction:", topPrediction);

                switch (topPrediction.label) {
                    case "Schild":
                        predictionDiv.innerHTML = 'Schild 🛡️';
                        console.log('Set predictionDiv to Schild');
                        break;
                    case "Magie":
                        predictionDiv.innerHTML = 'Magie 🔥';
                        console.log('Set predictionDiv to Magie');
                        break;
                    case "Zwaard":
                        predictionDiv.innerHTML = 'Zwaard 🗡️';
                        console.log('Set predictionDiv to Zwaard');
                        break;
                    default:
                        predictionDiv.innerHTML = '';
                        console.log('Set predictionDiv to empty (default)');
                        break;
                }
            }
        }

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        for (const hand of results.landmarks) {
            drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {color: "#00FF00", lineWidth: 5});
            drawUtils.drawLandmarks(hand, {radius: 4, color: "#FF0000", lineWidth: 2});
        }
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

if (navigator.mediaDevices?.getUserMedia) {
    nn.load(modelDetails, () => {
        console.log("Model loaded successfully!");
        createHandLandmarker();
    });
}

