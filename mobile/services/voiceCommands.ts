/**
 * ═══════════════════════════════════
 * mobile/services/voiceCommands.ts
 * ═══════════════════════════════════
 *
 * BrailleVision AI — Global Voice Command Engine
 * Provides always-on, hands-free control for blind users via Web Speech API.
 * Continuously listens for spoken commands and dispatches them to registered handlers.
 */

// ─────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────

/** A map of spoken phrase patterns → handler functions */
export type CommandMap = Record<string, () => void>;

/** Listener that fires whenever listening state changes */
export type ListeningStateListener = (isListening: boolean, lastCommand?: string) => void;

// ─────────────────────────────────────────────────────────────
// GLOBAL NAVIGATION CALLBACK
// Set from _layout.tsx after router is available
// ─────────────────────────────────────────────────────────────
let _navigateTo: ((tab: string) => void) | null = null;
let _captureCallback: (() => void) | null = null;
let _pickImageCallback: (() => void) | null = null;
let _readResultCallback: (() => void) | null = null;
let _stopAudioCallback: (() => void) | null = null;
let _flashlightCallback: ((on: boolean) => void) | null = null;

// ─────────────────────────────────────────────────────────────
// VOICE COMMAND ENGINE SINGLETON
// ─────────────────────────────────────────────────────────────

class VoiceCommandEngineClass {
  private recognition: any = null;
  private isListening = false;
  private isSupported = false;
  private stateListeners: ListeningStateListener[] = [];
  private extraCommands: CommandMap = {};
  private restartTimer: any = null;
  private lastCommandText = '';
  private isSuppressed = false; // suppress while TTS speaks so it doesn't hear itself
  private isSuppressedExternal = false; // suppress while external TTS player speaks

  constructor() {
    if (typeof window !== 'undefined') {
      const SpeechRecognition =
        (window as any).SpeechRecognition ||
        (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        this.isSupported = true;
        this.recognition = new SpeechRecognition();
        this._configure();
      }
    }
  }

  // ─────────────────────────────────────────────────────────────
  // CONFIGURE RECOGNITION ENGINE
  // ─────────────────────────────────────────────────────────────
  private _configure() {
    if (!this.recognition) return;

    this.recognition.continuous = true;
    this.recognition.interimResults = false;
    this.recognition.lang = 'en-US';
    this.recognition.maxAlternatives = 3;

    this.recognition.onresult = (event: any) => {
      if (this.isSuppressed || this.isSuppressedExternal) return;

      // Suppress if Zustand store says we are currently playing audio
      try {
        const store = require('../store/useAppStore').useAppStore.getState();
        if (store.isPlayingAudio) return;
      } catch (_) {}

      if (typeof window !== 'undefined' && (window as any).speechSynthesis?.speaking) return;

      const results = event.results;
      for (let i = event.resultIndex; i < results.length; i++) {
        if (results[i].isFinal) {
          // Check all alternatives for best match
          for (let alt = 0; alt < results[i].length; alt++) {
            const phrase = results[i][alt].transcript.trim().toLowerCase();
            if (this._handlePhrase(phrase)) break;
          }
        }
      }
    };

    this.recognition.onerror = (event: any) => {
      // 'no-speech' is common and harmless — auto-restart
      if (event.error === 'no-speech' || event.error === 'audio-capture') {
        this._scheduleRestart();
        return;
      }
      // 'not-allowed' means user denied mic
      if (event.error === 'not-allowed') {
        this.isListening = false;
        this._notifyListeners();
        return;
      }
      // All other errors: attempt restart
      this._scheduleRestart();
    };

    this.recognition.onend = () => {
      // Auto-restart if we're supposed to be listening
      if (this.isListening) {
        this._scheduleRestart();
      }
    };
  }

  // ─────────────────────────────────────────────────────────────
  // HANDLE A RECOGNIZED PHRASE
  // Returns true if a command was matched
  // ─────────────────────────────────────────────────────────────
  private _handlePhrase(phrase: string): boolean {
    this.lastCommandText = phrase;

    // ── NAVIGATION COMMANDS ──
    if (this._match(phrase, ['go to scanner', 'open camera', 'open scanner', 'scanner', 'camera'])) {
      this._confirm('Opening scanner.');
      _navigateTo?.('scanner');
      return true;
    }

    if (this._match(phrase, ['go to upload', 'open upload', 'open gallery', 'gallery', 'upload'])) {
      this._confirm('Opening upload screen.');
      _navigateTo?.('upload');
      return true;
    }

    if (this._match(phrase, ['go to history', 'open history', 'history', 'my history'])) {
      this._confirm('Opening scan history.');
      _navigateTo?.('history');
      return true;
    }

    if (this._match(phrase, ['go to settings', 'open settings', 'settings'])) {
      this._confirm('Opening settings.');
      _navigateTo?.('settings');
      return true;
    }

    // ── SCANNER COMMANDS ──
    if (this._match(phrase, ['scan', 'capture', 'read', 'take photo', 'snap', 'decode', 'read braille'])) {
      this._confirm('Scanning.');
      _captureCallback?.();
      return true;
    }

    // ── UPLOAD COMMANDS ──
    if (this._match(phrase, ['open gallery', 'upload photo', 'upload image', 'pick image', 'pick photo', 'choose photo', 'choose image'])) {
      this._confirm('Opening photo gallery.');
      _pickImageCallback?.();
      return true;
    }

    // ── AUDIO COMMANDS ──
    if (this._match(phrase, ['read result', 'speak', 'play', 'read it', 'play audio', 'read out', 'say it'])) {
      this._confirm('Reading result.');
      _readResultCallback?.();
      return true;
    }

    if (this._match(phrase, ['stop', 'quiet', 'mute', 'silence', 'stop speaking', 'shut up'])) {
      this._confirm('Stopped.');
      _stopAudioCallback?.();
      return true;
    }

    // ── FLASHLIGHT COMMANDS ──
    if (this._match(phrase, ['flashlight on', 'torch on', 'turn on torch', 'turn on flashlight', 'light on'])) {
      this._confirm('Flashlight on.');
      _flashlightCallback?.(true);
      return true;
    }

    if (this._match(phrase, ['flashlight off', 'torch off', 'turn off torch', 'turn off flashlight', 'light off'])) {
      this._confirm('Flashlight off.');
      _flashlightCallback?.(false);
      return true;
    }

    // ── HELP ──
    if (this._match(phrase, ['help', 'commands', 'what can i say', 'what can i do', 'options'])) {
      this._speakHelp();
      return true;
    }

    // ── SCREEN-SPECIFIC EXTRA COMMANDS ──
    for (const [pattern, handler] of Object.entries(this.extraCommands)) {
      if (phrase.includes(pattern.toLowerCase())) {
        handler();
        return true;
      }
    }

    return false;
  }

  // ─────────────────────────────────────────────────────────────
  // HELPERS
  // ─────────────────────────────────────────────────────────────
  private _match(phrase: string, patterns: string[]): boolean {
    return patterns.some((p) => phrase.includes(p.toLowerCase()));
  }

  private _confirm(message: string) {
    // Suppress recognition while TTS speaks to avoid feedback loop
    this.isSuppressed = true;
    this._notifyListeners(message);

    const speak = (text: string) => {
      if (typeof window === 'undefined') return;
      const utterance = new (window as any).SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US';
      utterance.rate = 1.1;
      utterance.onend = () => {
        setTimeout(() => { this.isSuppressed = false; }, 500);
      };
      utterance.onerror = () => { this.isSuppressed = false; };
      (window as any).speechSynthesis?.cancel();
      (window as any).speechSynthesis?.speak(utterance);
    };

    speak(message);
  }

  private _speakHelp() {
    const helpText = [
      'Available voice commands:',
      'Say "scan" or "capture" to read Braille with the camera.',
      'Say "open gallery" or "upload photo" to upload an image.',
      'Say "read result" or "play" to hear the decoded text.',
      'Say "stop" to stop audio.',
      'Say "flashlight on" or "flashlight off" to control the torch.',
      'Say "go to history" to view past scans.',
      'Say "go to settings" to open preferences.',
    ].join(' ');

    this._confirm(helpText);
  }

  private _scheduleRestart(delayMs = 300) {
    if (this.restartTimer) clearTimeout(this.restartTimer);
    this.restartTimer = setTimeout(() => {
      if (this.isListening && this.recognition) {
        try { this.recognition.start(); } catch (_) {}
      }
    }, delayMs);
  }

  private _notifyListeners(lastCommand?: string) {
    for (const listener of this.stateListeners) {
      listener(this.isListening, lastCommand);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // PUBLIC API
  // ─────────────────────────────────────────────────────────────

  get supported() { return this.isSupported; }
  get listening() { return this.isListening; }
  get lastCommand() { return this.lastCommandText; }

  /** Suppress recognition temporarily (e.g. while speaking) */
  setSuppressed(suppressed: boolean) {
    this.isSuppressedExternal = suppressed;
  }

  /** Start continuous listening */
  start() {
    if (!this.isSupported || this.isListening) return;
    this.isListening = true;
    try {
      this.recognition.start();
    } catch (_) {
      // Already started — ignore
    }
    this._notifyListeners();
    // Greet user
    this._confirm('Voice commands active. Say help for available commands.');
  }

  /** Stop listening */
  stop() {
    if (!this.isSupported || !this.isListening) return;
    this.isListening = false;
    if (this.restartTimer) clearTimeout(this.restartTimer);
    try { this.recognition.stop(); } catch (_) {}
    this._notifyListeners();
  }

  /** Toggle on/off */
  toggle() {
    if (this.isListening) this.stop();
    else this.start();
  }

  /** Register a callback when tab navigation is needed */
  setNavigateCallback(fn: (tab: string) => void) { _navigateTo = fn; }

  /** Register screen-specific action callbacks */
  setCaptureCallback(fn: () => void) { _captureCallback = fn; }
  setPickImageCallback(fn: () => void) { _pickImageCallback = fn; }
  setReadResultCallback(fn: () => void) { _readResultCallback = fn; }
  setStopAudioCallback(fn: () => void) { _stopAudioCallback = fn; }
  setFlashlightCallback(fn: (on: boolean) => void) { _flashlightCallback = fn; }

  /** Clear a specific callback (call on screen blur) */
  clearCaptureCallback() { _captureCallback = null; }
  clearPickImageCallback() { _pickImageCallback = null; }
  clearReadResultCallback() { _readResultCallback = null; }
  clearFlashlightCallback() { _flashlightCallback = null; }

  /** Subscribe to listening state changes (for UI indicator) */
  addStateListener(fn: ListeningStateListener) {
    this.stateListeners.push(fn);
  }

  removeStateListener(fn: ListeningStateListener) {
    this.stateListeners = this.stateListeners.filter((l) => l !== fn);
  }
}

// Export singleton
export const VoiceCommandEngine = new VoiceCommandEngineClass();
export default VoiceCommandEngine;
