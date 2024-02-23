const recorder = new Librosa.MonoRecorder(44100);
const audioContext = new AudioContext();
const microphone = audioContext.createMediaStreamSource(navigator.mediaDevices.getUserMedia({ audio: true }));
const processor = audioContext.createScriptProcessor(4096, 1, 1);

microphone.connect(processor);
processor.connect(audioContext.destination);

const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const audioPlaybackElement = document.getElementById('audio-playback');

let recording = false;
let audioData = [];

recordButton.addEventListener('click', () => {
  if (!recording) {
    navigator.mediaDevices.getUserMedia({ audio: true }, () => {
      recording = true;
      recordButton.textContent = 'Stop Recording';
      stopButton.classList.remove('inactive');

      processor.onaudioprocess = (event) => {
        audioData.push(event.inputBuffer.getChannelData(0));
      };
    }, (error) => {
      console.error('Failed to access microphone:', error);
    });
  }
});

stopButton.addEventListener('click', () => {
  if (recording) {
    recording = false;
    recordButton.textContent = 'Start Recording';
    stopButton.classList.add('inactive');

    processor.onaudioprocess = null;

    const audioBlob = new Blob(audioData, { type: 'audio/wav' });
    const audioURL = URL.createObjectURL(audioBlob);

    audioPlaybackElement.src = audioURL;
    audioPlaybackElement.classList.remove('hidden');
    audioPlaybackElement.play();

    audioData = [];
  }
});