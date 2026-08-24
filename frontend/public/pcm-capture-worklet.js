/* global AudioWorkletProcessor, registerProcessor */

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.offset = 0;
    this.disposed = false;
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") {
        this.flush();
        this.port.postMessage({ type: "flush-complete" });
      } else if (event.data?.type === "dispose") {
        this.buffer.fill(0);
        this.offset = 0;
        this.disposed = true;
      }
    };
  }

  flush() {
    if (this.offset === 0) return;
    const samples = this.buffer.slice(0, this.offset);
    this.buffer.fill(0, 0, this.offset);
    this.offset = 0;
    this.port.postMessage(
      { type: "samples", buffer: samples.buffer },
      [samples.buffer],
    );
  }

  process(inputs) {
    if (this.disposed) return false;
    const channels = inputs[0];
    if (!channels?.length || !channels[0]?.length) return true;

    const frameLength = channels[0].length;
    for (let index = 0; index < frameLength; index += 1) {
      let sample = 0;
      for (const channel of channels) sample += channel[index] || 0;
      this.buffer[this.offset] = sample / channels.length;
      this.offset += 1;
      if (this.offset === this.buffer.length) this.flush();
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
