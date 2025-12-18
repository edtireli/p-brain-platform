const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

function playTone(frequency: number, duration: number, type: OscillatorType = 'sine', volume: number = 0.3) {
  const oscillator = audioContext.createOscillator();
  const gainNode = audioContext.createGain();
  
  oscillator.connect(gainNode);
  gainNode.connect(audioContext.destination);
  
  oscillator.type = type;
  oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
  
  gainNode.gain.setValueAtTime(volume, audioContext.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + duration);
  
  oscillator.start(audioContext.currentTime);
  oscillator.stop(audioContext.currentTime + duration);
}

export function playSuccessSound() {
  playTone(523.25, 0.1, 'sine', 0.2);
  setTimeout(() => playTone(659.25, 0.1, 'sine', 0.2), 100);
  setTimeout(() => playTone(783.99, 0.15, 'sine', 0.25), 200);
}

export function playErrorSound() {
  playTone(311.13, 0.15, 'square', 0.15);
  setTimeout(() => playTone(233.08, 0.25, 'square', 0.12), 150);
}

export function resumeAudioContext() {
  if (audioContext.state === 'suspended') {
    audioContext.resume();
  }
}
