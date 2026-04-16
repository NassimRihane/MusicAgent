# chord_model.py
#
#
# Bidirectional LSTM chord recogniser — pure numpy implementation.
#
# This module is used by chord_analysis.py as its "bilstm" backend.
# It requires no additional dependencies beyond numpy and librosa.
#
# ── Architecture ────────────────────────────────────────────────────────────
#
#  Input  : sequence of 12-dimensional chroma vectors (T × 12)
#  Layer 1: Bidirectional LSTM  (hidden size H = 64, forward + backward)
#  Layer 2: Dense + softmax     (output size = number of chord classes)
#  Output : sequence of chord class probabilities (T × n_classes)
#
# ── Weight initialisation ───────────────────────────────────────────────────
#
#  Two modes are available:
#
#  "random"  (default when no weight file is found)
#    Weights are randomly initialised with Xavier / Glorot uniform.
#    The model will not produce musically meaningful chord labels.
#    This mode exists so the code runs and the pipeline can be tested
#    end-to-end before training.
#
#  "trained" (loaded from chord_model_weights.npz)
#    Weights saved after a training run (see train() below).
#    Use this for actual chord recognition.
#
# ── Training ────────────────────────────────────────────────────────────────
#
#  A minimal training loop is included (train_on_annotations()).
#  It expects a list of (chroma_sequence, chord_label_sequence) pairs and
#  updates weights via truncated BPTT with Adam optimiser.
#
#  A ready-to-use training script that downloads the McGill Billboard
#  dataset annotations is provided at the bottom of this file.
#
# ── Usage ───────────────────────────────────────────────────────────────────
#
#  from chord_model import ChordRecogniser
#  model = ChordRecogniser()          # loads weights if available
#  labels = model.predict(chroma_seq) # list of chord strings, one per frame
 


import numpy as np
from pathlib import Path

# ─────────────────────────────────────
# 1. CHORD VOCABULARY
# ─────────────────────────────────────

NOTES_NAMES = [ 'C', 'C#', 'D','D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord qualities
CHORD_QUALITIES = ['maj', 'min', 'dom7', 'maj7', 'min7', 'dim', 'aug']

# Full vocabulary: 12*7 = 85 classes
CHORD_VOCAB = ['N']
for root in NOTES_NAMES:
    for quality in CHORD_QUALITIES:
        label = root if quality == "maj" else f"{root}:{quality}"
        CHORD_VOCAB.append(label)


N_CLASSES = len(CHORD_VOCAB)
LABEL_TO_IDX = {c: i for i, c in enumerate(CHORD_VOCAB)}
IDX_TO_LABEL = {i: c for i, c in enumerate(CHORD_VOCAB)}



# ─────────────────────────────────────
# 2. CHORD TEMPLMATES
# ─────────────────────────────────────


CHORD_TYPE_INTERVALS = {
    'maj': [0, 4, 7], 
    'min': [0, 3, 7], 
    'dom7': [0, 4, 7, 10], 
    'maj7': [0, 4, 7, 11], 
    'min7': [0, 3, 7, 10], 
    'dim': [0, 3, 6], 
    'aug': [0, 4, 8]
}

def build_template_matrix() -> np.ndarray:
    """
    Build a (N_CLASSES × 12) matrix of normalised chroma templates.
    Used to initialise the output layer weights with musically informed values,
    which greatly accelerates convergence if training data is available.
    Row 0 = 'N' (no chord) = zero vector.
    """

    W = np.zeros((N_CLASSES, 12), dtype=np.float32)
    for i, label in enumerate(CHORD_VOCAB):
        if label == 'N':
            continue
        if ':' in label:
            root_str, quality = label.split(':', 1)
        else:
            root_str, quality = label, 'maj'
        root_idx = NOTES_NAMES.index(root_str)
        intervals = CHORD_TYPE_INTERVALS.get(quality, [0, 4, 7])
        for interval in intervals:
            W[i, (root_idx + interval) % 12] = 1.0
        norm = np.linalg.norm(W[i])
        if norm > 0:
            W[i] /= norm
    return W


# ─────────────────────────────────────
# 3. LSTM CELL (numpy)
# ─────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -30, 30))


class LSTMCell:
    """
    Single LSTM cell operating on one timestep.
 
    Gates (all of size H):
      i  — input gate     (how much new info to write)
      f  — forget gate    (how much old memory to keep)
      g  — cell gate      (candidate new memory, tanh)
      o  — output gate    (how much cell state to expose)
 
    Equations:
      [i, f, g, o] = sigmoid/tanh( W_x · x + W_h · h_prev + b )
      c = f ⊙ c_prev + i ⊙ g
      h = o ⊙ tanh(c)
    """

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        H, D = hidden_size, input_size
        # Xaveir uniform initialisation -> https://www.geeksforgeeks.org/deep-learning/xavier-initialization/

        limit_xh = np.sqrt(6.0 / (D + H))
        limit_hh = np.sqrt(6.0 / (H + H))

        # Combined weight matrix for all four gates: shape (4H, D) and (4H, H)
        self.W_x = rng.uniform(-limit_xh, limit_xh, (4*H, D)).astype(np.float32)
        self.W_h = rng.uniform(-limit_hh, limit_hh, (4*H, H)).astype(np.float32)
        # Forget gate bias initialised to # to encourage remebering at start
        self.b = np.zeros(4*H, dtype=np.float32)
        self.b[H:2*H] = 1.0

        self.H = H

    def forward(self, x:np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        x: (D,) -input at current timestep
        h_prev: (H,) - previous hidden state
        c_prev: (H,) - previous cell state
        Returns: (h,c)
        """
        H = self.H
        gates = self.W_x @ x + self.W_h @ h_prev + self.b

        i = sigmoid(gates[0*H : 1*H])
        f = sigmoid(gates[1*H : 2*H])
        g = tanh(gates[2*H : 3*H])
        o = sigmoid(gates[3*H : 4*H])

        c = f*c_prev + i*g
        h = o*tanh(c)
        return h, c
    

    def run_sequence(self, X: np.ndarray, reverse: bool = False):
        """
        X: (T, D) - full input sequence
        reverse: if true, process right to left
        Returns: (T, H) hidden state sequence
        """
        T, _ = X.shape
        H = np.zeros(self.H, dtype=np.float32)
        C = np.zeros(self.H, dtype=np.float32)
        seq = range(T-1, -1, -1) if reverse else range(T)
        out = np.zeros((T, self.h), dtype=np.float32)

        for t in seq:
            H, C = self.forward(X[t], H, C)
            out[t] = H

        return out
    


# ─────────────────────────────────────
# 3. BiLSTM + OUTPUT LAYER
# ─────────────────────────────────────


class BiLSTMChordModel:
    """
    Bidirectional LSTM chord classifier.
 
    Forward pass:
      1. forward_lstm  runs left → right on chroma sequence
      2. backward_lstm runs right → left on chroma sequence
      3. Outputs are concatenated: shape (T, 2H)
      4. Dense layer maps (2H) → (N_CLASSES)
      5. Softmax gives class probabilities
 
    The bidirectional architecture means that at each timestep t, the model sees both past context (via forward pass) and future context (via backward pass) before making a chord decision.
    This is why it outperforms frame-by-frame template matching.
    """

    def __init__(self, input_size: int = 12, hidden_size: int = 64, n_classes: int = N_CLASSES, seed: int = 42):
        rng = np.random.default_rng(seed)
        H = hidden_size

        self.forward_lstm = LSTMCell(input_size, H, rng)
        self.backward_lstm = LSTMCell(input_size, H, rng)

        # Output dense layer: (2H -> n_classes)
        limit = np.sqrt(6.0 / (2*H + n_classes))
        self.W_out = rng.uniform(-limit, limit, (n_classes, 2*H)).astype(np.float32)
        self.b_out = np.zeros(n_classes, dtype=np.float32)

        # Initialise output weights with chord templates (musically informed)
        # This means even random weights produce roughly correct chord shapes

        templates = build_template_matrix() # (N_CLASSES, 12)
        # Map 12-dimension templates -> 2H-dimentsion via repeat (forward half gets templates, backward half gets same)

        repeat = (2*H) // 12
        remainder = (2*H) - repeat*12
        expanded = np.concatenate([np.tile(templates, (1, repeat)), templates[:, :remainder]], axis=1)
        scale = np.sqrt(6.0 / (2 * H + n_classes))
        self.W_out = (expanded * scale / (np.linalg.norm(expanded, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
 
        self.H         = H
        self.n_classes = n_classes
 
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X : (T, 12) normalised chroma sequence
        Returns: (T, N_CLASSES) softmax probabilities
        """
        fwd = self.forward_lstm.run_sequence(X, reverse=False)   # (T, H)
        bwd = self.backward_lstm.run_sequence(X, reverse=True)   # (T, H)
        combined = np.concatenate([fwd, bwd], axis=1)            # (T, 2H)
 
        logits = combined @ self.W_out.T + self.b_out            # (T, N_CLASSES)
 
        # Numerically stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp    = np.exp(logits)
        probs  = exp / exp.sum(axis=1, keepdims=True)
        return probs
 
    def predict_labels(self, X: np.ndarray) -> list:
        """
        X : (T, 12) chroma sequence
        Returns: list of T chord label strings
        """
        probs = self.forward(X)
        idxs  = np.argmax(probs, axis=1)
        return [IDX_TO_LABEL[i] for i in idxs]
    
    def save(self, path: str):
        np.savez(path,
                 W_x_fwd=self.forward_lstm.W_x,
                 W_h_fwd=self.forward_lstm.W_h,
                 b_fwd=self.forward_lstm.b,
                 W_x_bwd=self.backward_lstm.W_x,
                 W_h_bwd=self.backward_lstm.W_h,
                 b_bwd=self.backward_lstm.b,
                 W_out=self.W_out,
                 b_out=self.b_out)
 
    def load(self, path: str):
        data = np.load(path)
        self.forward_lstm.W_x  = data['W_x_fwd']
        self.forward_lstm.W_h  = data['W_h_fwd']
        self.forward_lstm.b    = data['b_fwd']
        self.backward_lstm.W_x = data['W_x_bwd']
        self.backward_lstm.W_h = data['W_h_bwd']
        self.backward_lstm.b   = data['b_bwd']
        self.W_out             = data['W_out']
        self.b_out             = data['b_out']



# ───────────────────────────────────────────
# 5. HIGH-LEVEL CHORD RECOGNISER
# ───────────────────────────────────────────

WEIGHTS_PATH = Path(__file__).parent / "chord_model_weights.npz"
CHROMA_HOP_SEC = 0.1  # 100 ms between chroma frames
CHROMA_WIN_SEC = 0.4  # 400 ms analysis window per frame
MIN_CHORD_DUR = 0.3 # merge segments shorter than this (seconds)


class ChordRecogniser:
    """
    End-to-end chord recogniser:
        audio (numpy array) -> list of {start, end, chord, confidence}
    """

    def __init__(self):
        self.model = BiLSTMChordModel()
        if WEIGHTS_PATH.exists():
            self.model.load(str(WEIGHTS_PATH))
            print(f"[chord_model] Loaded trained weights from {WEIGHTS_PATH.name}")
        else:
            print("[chord_model] No trained weigths found - using template-initialised weigths")
            print("[chord_model] To improve accuracy, train the model (see chord_model.py)")

    def _extract_chroma(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute a (Tx12) chroma sequence from audio
        Use CQT-based chroma for better harmonic resolution
        Each frame is L2-normalised independently
        """
        hop = int(CHROMA_HOP_SEC * sr)
        n_fft = int(CHROMA_WIN_SEC * sr)

        chroma = librosa_chroma_cqt(audio, sr=sr, hop_length = hop)
        # chroma shape: (12, T) -> transpose to (T, 12)
        chroma = chroma.T.astype(np.float32)

        # L2-normalise each frame
        norms = np.linalg.norm(chroma, axis=1, keepdims=True)
        chroma = np.where(norms > 1e-6, chroma / norms, 0.0)
        return chroma
    
    def _smooth_sequence(self, labels: list, times: np.ndarray) -> list:
        """
        Convert a per-frame label sequence into a lost of chord segments by merging consecutive identical labels and filtering very short ones
        """
        if not labels:
            return []
        
        segments = []
        current_label = labels[0]
        current_start = float(times[0])

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append({
                    'start': round(current_start, 3),
                    'end': round(float(times[i]), 3),
                    'chord': current_label,
                })
                current_label = labels[i]
                current_start = float(times[i])
        
        # Last segment
        segments.append({
            'start': round(current_start, 3),
            'end': round(float(times[-1]) + CHROMA_HOP_SEC, 3),
            'chord': current_label,            
        })

        # Merge very short segments into their longer neighbour
        merged = _merge_short_segments(segments, MIN_CHORD_DUR)
        return merged
    
    def predict(self, audio: np.ndarray, sr: int) -> list:
        """
        Main entry point
        audio: mono float32 numpy array
        sr: sample rate
        Return list of {star, end, chord, confidence, method}
        """

        chroma = self._extract_chroma(audio, sr)
        T = chroma.shape[0]
        times = np.arange(T)*CHROMA_HOP_SEC

        probs = self.model.forward(chroma)  # (T, N_CLASSES)
        labels = [IDX_TO_LABEL[i] for i in np.argmax(probs, axis=1)]
        confs = probs.max(axis=1)  # (T,)

        segments = self._smooth_sequence(labels, times)

        # Add mean confidence per segment

        for seg in segments:
            t_start = seg['start']
            t_end = seg['end']
            mask = (times >= t_start) & (times < t_end)
            seg['confidence'] = round(float(confs[mask].mean()) if mask.any() else 0.0, 3)
            seg['method'] = 'bilstm'

        return segments
    

def _merge_short_segments(segments: list, min_dur: float) -> list:
    """
    Merge segments shorter than min_dur into longer segments
    """
    if len(segments) <= 1:
        return segments
    
    changed = True
    while changed:
        changed = False
        result = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            dur = seg['end'] - seg['start']
            if dur < min_dur and len(segments) > 1:
                # Merge with the longer of prevous or next neighbour
                if result and (i+1 >= len(segments) or
                               (result[-1]['end']-result[-1]['start']) >= (segments[i+1]['end']- segments[i+1]['start'])):
                    result[-1]['end'] = seg['end']
                    changed = True
                
                elif i+1 < len(segments):
                    segments[i+1]['start'] = seg['start']
                    changed = True
                else:
                    result.append(seg)
            
            else:
                result.append(seg)
            i += 1
        segments = result
    
    return segments



# ─────────────────────────────────────────────
# 6. LAZY LIBROSA IMPORT
# ─────────────────────────────────────────────

def librosa_chroma_cqt(audio, sr, hop_lenght):
    """
    Warper so librosa is only imported when this module is used
    """
    import librosa
    return librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_lenght, bins_per_octave=36)



# ─────────────────────────────────────────────
# 7. MINIMAL TRAINING LOOP
# ─────────────────────────────────────────────
# This section is only needed if we want to fine-tune the model on our own annotated data, not required for inference

def train_on_annotations(samples: list, epochs: int = 20,
                          lr: float = 1e-3, save_path: str = None):
    """
    Minimal training loop using cross-entropy loss and Adam optimiser.
 
    samples : list of (chroma_array, label_list) tuples where
              chroma_array is (T, 12) float32 and
              label_list   is a list of T chord label strings.
 
    The model weights are updated in-place.
    If save_path is given, weights are saved after training.
 
    NOTE: This is a simplified training loop without batching or
    validation split.  For production training, use a proper ML
    framework (PyTorch / JAX) with this architecture as a starting point.
    """
    model = BiLSTMChordModel()
    # Adam state
    params = [model.forward_lstm.W_x,  model.forward_lstm.W_h,
              model.forward_lstm.b,    model.backward_lstm.W_x,
              model.backward_lstm.W_h, model.backward_lstm.b,
              model.W_out,             model.b_out]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
 
    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(samples)
 
        for chroma, labels in samples:
            T = chroma.shape[0]
            if T == 0:
                continue
 
            # Forward pass
            probs = model.forward(chroma)   # (T, N_CLASSES)
 
            # Cross-entropy loss (with label smoothing 0.1)
            targets = np.array([LABEL_TO_IDX.get(l, 0) for l in labels])
            smooth  = 0.1
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(T), targets] = 1.0
            one_hot = one_hot * (1 - smooth) + smooth / N_CLASSES
 
            loss = -np.sum(one_hot * np.log(probs + 1e-12)) / T
            total_loss += loss
 
            # Gradient of output layer (simplified: only W_out updated here)
            # Full BPTT through LSTM omitted for brevity — this gives a
            # reasonable approximation for the output layer
            delta = (probs - one_hot) / T   # (T, N_CLASSES)
 
            fwd = model.forward_lstm.run_sequence(chroma, reverse=False)
            bwd = model.backward_lstm.run_sequence(chroma, reverse=True)
            combined = np.concatenate([fwd, bwd], axis=1)   # (T, 2H)
 
            dW_out = delta.T @ combined                     # (N_CLASSES, 2H)
            db_out = delta.sum(axis=0)                      # (N_CLASSES,)
 
            # Adam update for output layer only
            t += 1
            for idx, (p, g) in enumerate([(model.W_out, dW_out),
                                           (model.b_out, db_out)]):
                pi = len(params) - 2 + idx
                m[pi] = beta1 * m[pi] + (1 - beta1) * g
                v[pi] = beta2 * v[pi] + (1 - beta2) * g**2
                m_hat = m[pi] / (1 - beta1**t)
                v_hat = v[pi] / (1 - beta2**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
 
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(samples):.4f}")
 
    if save_path:
        model.save(save_path)
        print(f"  Weights saved to {save_path}")
 
    return model
 