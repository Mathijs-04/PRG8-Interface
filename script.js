import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

ml5.setBackend("webgl");

const nn = ml5.neuralNetwork({ task: 'classification', debug: true });
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
};

let gameState = {
    isPlaying: false,
    countdown: 3,
    playerChoice: null,
    computerChoice: null,
    timerId: null
};

const elements = {
    webcamButton: document.getElementById("webcamButton"),
    startGameButton: document.getElementById("startGame"),
    predictionDiv: document.getElementById("predictionDiv"),
    countdown: document.getElementById("countdown"),
    computerChoice: document.getElementById("computerChoice"),
    result: document.getElementById("result")
};

const choices = ['Schild', 'Magie', 'Zwaard'];
const outcomes = {
    Schild: { beats: 'Zwaard', emoji: '🛡️' },
    Zwaard: { beats: 'Magie', emoji: '🗡️' },
    Magie: { beats: 'Schild', emoji: '🔥' }
};

function getComputerChoice() {
    return choices[Math.floor(Math.random() * choices.length)];
}

function determineWinner(player, computer) {
    if (player === computer) return "Gelijkspel!";
    return outcomes[player].beats === computer ? "Jij wint! 🎉" : "Computer wint! 😢";
}

async function updateGameDisplay() {
    elements.countdown.textContent = gameState.countdown;
    elements.computerChoice.textContent = gameState.computerChoice
        ? `${gameState.computerChoice} ${outcomes[gameState.computerChoice].emoji}`
        : "";
}

function gameLoop() {
    if (gameState.countdown > 0) {
        gameState.countdown--;
        elements.countdown.textContent = `Countdown: ${gameState.countdown}`;
        elements.computerChoice.textContent = "Computer kiest...";
    } else {
        clearInterval(gameState.timerId);
        const result = determineWinner(gameState.playerChoice, gameState.computerChoice);
        elements.result.textContent = result;
        elements.result.style.color = result.includes("wint")
            ? (result.includes("Jij") ? "#4CAF50" : "#ff4444")
            : "#ffd700";

        elements.computerChoice.textContent =
            `Computer: ${gameState.computerChoice} ${outcomes[gameState.computerChoice].emoji}`;

        setTimeout(() => {
            gameState.isPlaying = false;
            gameState.countdown = 3;
            gameState.playerChoice = null;
            gameState.computerChoice = null;
            elements.startGameButton.disabled = false;
            elements.countdown.textContent = "";
            elements.computerChoice.textContent = "";
            elements.result.textContent = "";
        }, 3000);
    }
}

let handLandmarker, webcamRunning = false;
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawUtils = new DrawingUtils(canvasCtx);

async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
}

async function enableCam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            webcamRunning = true;
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

async function predictWebcam() {
    if (!webcamRunning) return;

    const results = await handLandmarker.detectForVideo(video, performance.now());
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.landmarks.length > 0) {
        const hand = results.landmarks[0];
        const handData = hand.map((point) => [point.x, point.y, point.z]).flat();

        const prediction = await nn.classify(handData);
        if (prediction.length > 0) {
            const topPrediction = prediction.reduce((max, p) => p.confidence > max.confidence ? p : max);
            elements.predictionDiv.textContent = `${topPrediction.label} ${outcomes[topPrediction.label]?.emoji || ''}`;
            if (gameState.isPlaying) gameState.playerChoice = topPrediction.label;
        }

        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    requestAnimationFrame(predictWebcam);
}

elements.webcamButton.addEventListener("click", enableCam);
elements.startGameButton.addEventListener("click", () => {
    if (!gameState.isPlaying) {
        gameState.isPlaying = true;
        gameState.computerChoice = getComputerChoice();
        elements.startGameButton.disabled = true;
        gameState.timerId = setInterval(gameLoop, 1000);
    }
});

nn.load(modelDetails, () => {
    console.log("Model loaded successfully!");
    createHandLandmarker();
});
