import { HandDetector } from "@tensorflow-models/hand-pose-detection";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

export class Model {
  // private readonly modelURL: string;
  private detector?: HandDetector;

  // private model?: GraphModel<string>;
  handLandmarker?: HandLandmarker;

  constructor(modelURL = "models/sign_language_model.json") {
    console.log("heya, model construction");
    this.load();
    // this.modelURL = modelURL;
  }

  private async load() {
    const createHandLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "node_modules/@mediapipe/tasks-vision/wasm",
        // "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
      );
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          // modelAssetPath:
          // `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          modelAssetPath: `src/models/hand_landmarker.task`,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
      });
      // demosSection.classList.remove("invisible");
    };
    await createHandLandmarker();

    // this.model = await loadGraphModel(this.modelURL);
    // const model = handPoseDetection.SupportedModels.MediaPipeHands;
    // const detectorConfig = {
    //   runtime: "mediapipe", // or 'tfjs',
    //   solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
    //   modelType: "full",
    // } as MediaPipeHandsModelConfig;
    // this.detector = await handPoseDetection.createDetector(
    //   model,
    //   detectorConfig,
    // );
  }

  predict() {
    const image = new Image();
    image.src = "src/images/letter_b.jpg";
    console.log({ image });
    const handLandmarkerResult = this.handLandmarker?.detect(image);
    console.log(handLandmarkerResult);
    // const hands = this.detector?.estimateHands(image);
    // console.log(hands);
  }
}
