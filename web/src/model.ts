import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
  HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

export class Model {
  handLandmarker?: HandLandmarker;
  private drawingUtils: DrawingUtils;

  constructor(ctx: CanvasRenderingContext2D) {
    this.load();
    this.drawingUtils = new DrawingUtils(ctx);
  }

  private async load() {
    const vision = await FilesetResolver.forVisionTasks(
      // "src/wasm",
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
    );
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        // modelAssetPath: "src/models/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    // demosSection.classList.remove("invisible");
  }

  drawPredictions(results?: HandLandmarkerResult) {
    for (const landmarks of results?.landmarks || []) {
      this.drawingUtils.drawConnectors(
        landmarks,
        HandLandmarker.HAND_CONNECTIONS,
        {
          color: "#00FF00",
          lineWidth: 5,
        },
      );
      this.drawingUtils.drawLandmarks(landmarks, {
        color: "#000000",
        lineWidth: 2,
      });
    }
  }
}
