import {
  DrawingUtils,
  FilesetResolver,
  HandLandmarker,
  HandLandmarkerResult,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import a from "../src/letter_keypoints/a.json";
import b from "../src/letter_keypoints/b.json";
import c from "../src/letter_keypoints/c.json";
import d from "../src/letter_keypoints/d.json";
import e from "../src/letter_keypoints/e.json";
import f from "../src/letter_keypoints/f.json";
import h from "../src/letter_keypoints/h.json";
import i from "../src/letter_keypoints/i.json";
import k from "../src/letter_keypoints/k.json";
import l from "../src/letter_keypoints/l.json";
import m from "../src/letter_keypoints/m.json";

type KeyPoint = {
  x: number;
  y: number;
  z: number;
};

export class Model {
  handLandmarker?: HandLandmarker;
  private drawingUtils: DrawingUtils;
  private readonly keypoints: Record<string, KeyPoint[]>;

  constructor(ctx: CanvasRenderingContext2D) {
    this.load();
    this.drawingUtils = new DrawingUtils(ctx);
    this.keypoints = { a, b, c, d, e, f, h, i, k, l, m };
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

  drawResults(results?: HandLandmarkerResult) {
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

  findClosestLetter(results?: HandLandmarkerResult): string | undefined {
    const landmarks = results?.landmarks[0];
    if (!landmarks) return;

    this.scaleKeypointResults(landmarks);
    let closestLetter;
    let minimumDistance = Infinity;
    for (const letter in this.keypoints) {
      const keypoints = this.keypoints[letter];
      if (!keypoints) continue;

      const distance = this.findDistance(keypoints, landmarks);
      if (distance < minimumDistance) {
        closestLetter = letter;
        minimumDistance = distance;
      }
    }
    return closestLetter;
  }

  private scaleKeypointResults(results: NormalizedLandmark[]) {
    let x_min = Infinity;
    let y_min = Infinity;
    let z_min = Infinity;

    let x_max = -Infinity;
    let y_max = -Infinity;
    let z_max = -Infinity;

    for (const landmark of results) {
      if (landmark.x < x_min) x_min = landmark.x;
      if (landmark.y < y_min) y_min = landmark.y;
      if (landmark.z < z_min) z_min = landmark.z;

      if (landmark.x >= x_max) x_max = landmark.x;
      if (landmark.y >= y_max) y_max = landmark.y;
      if (landmark.z >= z_max) z_max = landmark.z;
    }

    for (const keypoint of results) {
      keypoint.x = (keypoint.x - x_min) / (x_max - x_min);
      keypoint.y = (keypoint.y - y_min) / (y_max - y_min);
      keypoint.z = (keypoint.z - z_min) / (z_max - z_min);
    }
  }

  private findDistance(
    first: KeyPoint[],
    second: KeyPoint[],
    numKeyPoints = 21,
  ): number {
    let distance = 0;
    for (let i = 0; i < numKeyPoints; i++) {
      const x_diff = Math.pow((first[i]?.x || 0) - (second[i]?.x || 0), 2);
      const y_diff = Math.pow((first[i]?.y || 0) - (second[i]?.y || 0), 2);
      const z_diff = Math.pow((first[i]?.z || 0) - (second[i]?.z || 0), 2);
      // distances.push(x_diff + y_diff + z_diff);
      distance += x_diff + y_diff + z_diff;
    }

    return distance / (numKeyPoints * 3);
  }
}
