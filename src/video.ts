(() => {
  let video: HTMLVideoElement | null = null;

  function startup() {
    video = document.getElementById("video") as HTMLVideoElement;

    navigator.mediaDevices
      // facingMode refers to front-facing "user" or backwards-facing "environment" camera
      .getUserMedia({ video: { facingMode: "user" }, audio: false })
      .then((stream) => {
        if (!video) return;

        video.srcObject = stream;
        video.play();
      })
      .catch((err) => {
        console.error(`Error occurred getting video stream ${err}`);
      });
  }

  window.addEventListener("load", startup, false);
})();
