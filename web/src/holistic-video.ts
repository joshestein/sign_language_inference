import { HAND_CONNECTIONS, Holistic, Results } from "@mediapipe/holistic";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { Camera } from "@mediapipe/camera_utils";
import { PretrainedModel } from "./pretrained-model";

(() => {
  const video = document.getElementById("video")! as HTMLVideoElement;
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  if (!canvas) return;
  const ctx = canvas.getContext("2d")!;

  const model = new PretrainedModel();

  function onResults(results: Results) {
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (results.segmentationMask) ctx.drawImage(results.segmentationMask, 0, 0, canvas.width, canvas.height);

    // Only overwrite existing pixels.
    ctx.globalCompositeOperation = "source-in";
    ctx.fillStyle = "#00FF00";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Only overwrite missing pixels.
    ctx.globalCompositeOperation = "destination-atop";
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    ctx.globalCompositeOperation = "source-over";
    // drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#8fc9ad", lineWidth: 4 });
    // drawLandmarks(ctx, results.poseLandmarks, { color: "#8fc9ad", lineWidth: 2 });
    // drawConnectors(ctx, results.faceLandmarks, FACEMESH_TESSELATION, { color: "#eeeeee", lineWidth: 1 });
    drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: "#eeeeee", lineWidth: 5 });
    drawLandmarks(ctx, results.leftHandLandmarks, { color: "#8fc9ad", lineWidth: 2 });
    drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: "#eeeeee", lineWidth: 5 });
    drawLandmarks(ctx, results.rightHandLandmarks, { color: "#8fc9ad", lineWidth: 2 });

    model.predict(results);

    ctx.restore();
  }

  const holistic = new Holistic({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    },
  });
  holistic.setOptions({
    modelComplexity: 0,
    smoothLandmarks: false,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  holistic.onResults(onResults);

  const camera = new Camera(video, {
    onFrame: async () => {
      await holistic.send({ image: video });
    },
    width: 1280,
    height: 720,
  });
  camera.start();
})();
