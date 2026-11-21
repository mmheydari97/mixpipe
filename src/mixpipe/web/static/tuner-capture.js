"use strict";
// AudioWorklet for the tuner: the microphone in 2048-sample Float32 chunks,
// posted to the page (tuner.js), which sends them to the server.

const CHUNK = 2048;

class TunerCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(CHUNK);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === CHUNK) {
          this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
          this.buf = new Float32Array(CHUNK);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("tuner-capture", TunerCapture);
