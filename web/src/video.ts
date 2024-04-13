import { Model } from "./model.js";

(() => {
  let video: HTMLVideoElement | null = null;
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d");
  const model = new Model(ctx!);

  function runInference() {
    if (!video) return;
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    let startTimeMs = performance.now();
    const results = model.handLandmarker?.detectForVideo(video, startTimeMs);
    model.drawPredictions(results);

    window.requestAnimationFrame(runInference);
  }

  function startup() {
    video = document.getElementById("video") as HTMLVideoElement;

    navigator.mediaDevices
      // facingMode refers to front-facing "user" or backwards-facing "environment" camera
      .getUserMedia({ video: { facingMode: "user" }, audio: false })
      .then((stream) => {
        if (!video) return;
        video.srcObject = stream;
        video.play();
        // video.addEventListener("loadeddata", runInference);
        runInference();
      })
      .catch((err) => {
        console.error(`Error occurred getting video stream ${err}`);
      });

    // video?.addEventListener("canplay", (event) => runInference());
  }

  window?.addEventListener("load", startup, false);
})();
