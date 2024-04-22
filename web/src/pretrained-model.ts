import * as tflite from "@tensorflow/tfjs-tflite";
import "@tensorflow/tfjs-backend-cpu";

export class PretrainedModel {
  private model?: tflite.TFLiteModel;

  constructor() {
    this.loadModel();
  }

  async loadModel() {
    this.model = await tflite.loadTFLiteModel("pretrained.tflite");
    console.log(this.model);
  }

  predict(input: any) {
    this.model?.predict(input);
  }
}
