ml5.setBackend("webgl");
const nn = ml5.neuralNetwork({ task: 'classification', debug: true })
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
}
nn.load(modelDetails, () => console.log("het model is geladen!"))
