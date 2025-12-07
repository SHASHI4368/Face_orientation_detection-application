import * as ort from "onnxruntime-web";

let session = null;

export async function loadCheatModel() {
  if (!session) {
    session = await ort.InferenceSession.create("/model/cheat_detector.onnx");
  }
  return session;
}

export async function predictCheating(deltaRoll, deltaPitch, deltaYaw) {
  const session = await loadCheatModel();

  // ONNX expects Float32 tensor of shape [1,3]
  const input = new ort.Tensor(
    "float32",
    Float32Array.from([deltaRoll, deltaPitch, deltaYaw]),
    [1, 3]
  );

  const results = await session.run({ input });

  // classifiers use "probabilities" or "output_label" depending on opset
  const prob = results.probabilities
    ? results.probabilities.data[1]
    : results.output_label
    ? results.output_label.data[0]
    : 0;

  return prob; // probability of cheating (0 to 1)
}
