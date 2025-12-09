import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

export async function loadCheatModel() {
  if (!session) {
    session = await ort.InferenceSession.create(
      "/new-model/cheat_detector.onnx",
      {
        executionProviders: ["wasm"], // explicit provider
      }
    );
  }
  return session;
}

export async function predictNewCheating(roll:number, pitch:number, yaw:number) {
  const session = await loadCheatModel();

  // ONNX expects Float32 tensor of shape [1,3]
  const input = new ort.Tensor(
    "float32",
    Float32Array.from([roll, pitch, yaw]),
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
