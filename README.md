# FaceID — Production-Grade Face Recognition Desktop App

A real-time face registration and verification desktop application built with Python, PyQt6, and OpenCV. Engineered to production standards: strict layered architecture, bounded memory, zero-UI-blocking threading, exhaustive failure handling, and full unit-test coverage of business logic.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Libraries & Frameworks](#2-libraries--frameworks)
3. [Project Structure](#3-project-structure)
4. [Architecture Deep-Dive](#4-architecture-deep-dive)
5. [Recognition Pipeline](#5-recognition-pipeline)
6. [Threading Model](#6-threading-model)
7. [Memory Management](#7-memory-management)
8. [Failure Handling Matrix](#8-failure-handling-matrix)
9. [Configuration Reference](#9-configuration-reference)
10. [Observability & Metrics](#10-observability--metrics)
11. [Running Tests](#11-running-tests)
12. [CLI / Headless Mode](#12-cli--headless-mode)
13. [MedTech Upgrade Path](#13-medtech-upgrade-path)

---

## 1. Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or later |
| pip | 22+ |
| cmake | Required by dlib (see below) |
| A webcam | USB or built-in |

### Install system dependencies

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install -y cmake build-essential libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev python3-dev
```

**macOS (Homebrew)**
```bash
brew install cmake
```

**Windows**
Install [CMake](https://cmake.org/download/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++").

### Install Python dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install all dependencies in one step
pip install -r requirements.txt
```

> **Note on dlib:** `face-recognition` compiles dlib from source. On a modern machine this takes 2–5 minutes. If it fails, install dlib separately first:
> ```bash
> pip install dlib
> pip install face-recognition
> ```

### Run the application

```bash
python main.py
```

Optional flags:
```bash
python main.py --debug              # Enable P99 latency overlay in status bar
python main.py --config /path/cfg.yaml  # Use a custom config file
```

---

## 2. Libraries & Frameworks

| Library | Version | Purpose |
|---|---|---|
| **PyQt6** | ≥ 6.6.0 | Cross-platform desktop UI; signal/slot threading model |
| **OpenCV** (`opencv-python`) | ≥ 4.9.0 | Camera capture, Haar cascade detection, image processing |
| **face-recognition** | ≥ 1.3.0 | dlib-based 128-dim face embeddings; fully offline |
| **NumPy** | ≥ 1.26.0 | Numerical arrays; float32 embeddings; cosine similarity |
| **PyYAML** | ≥ 6.0.1 | `config.yaml` parsing |
| **scipy** | ≥ 1.12.0 | Optional: available for advanced distance metrics |
| **Pillow** | ≥ 10.2.0 | Auxiliary image handling |

All dependencies are pip-installable in a single step via `requirements.txt`. No cloud APIs are used anywhere.

---

## 3. Project Structure

```
face-recognition-app/
│
├── main.py                     # Entry point: DI wiring, CLI flags, QApplication
├── config.yaml                 # All tuneable parameters (no magic numbers in code)
├── requirements.txt
├── README.md
├── .gitignore
│
├── core/                       # ── LAYER 1: Pure business logic ──────────────
│   ├── __init__.py
│   ├── entities.py             # Dataclasses: FaceIdentity, VerificationResult, etc.
│   ├── interfaces.py           # Abstract contracts: ICameraSource, IEmbedder, etc.
│   └── use_cases.py            # RegisterFaceUseCase, VerifyFaceUseCase
│
├── services/                   # ── LAYER 2: I/O & CV implementations ─────────
│   ├── __init__.py
│   ├── camera_service.py       # Threaded capture, ring buffer, FPS, reconnect
│   ├── detector.py             # Haar cascade detector implementing IFaceDetector
│   ├── embedder.py             # face_recognition wrapper implementing IEmbedder
│   ├── recognition_engine.py   # Cosine similarity engine implementing IRecognitionEngine
│   └── face_repository.py      # Atomic .npy/.json storage implementing IFaceRepository
│
├── ui/                         # ── LAYER 3: Qt UI only ───────────────────────
│   ├── __init__.py
│   ├── event_bus.py            # Qt signal-based pub/sub bus decoupling UI ↔ services
│   ├── main_window.py          # QMainWindow, thread orchestration, mode switching
│   ├── video_widget.py         # QLabel subclass: frame rendering + overlay painting
│   ├── register_panel.py       # Registration sidebar: form, progress, feedback
│   └── verify_panel.py         # Verification sidebar: live result, identity list
│
├── utils/                      # ── LAYER 0: Cross-cutting concerns ───────────
│   ├── __init__.py
│   ├── config.py               # Singleton AppConfig loader/validator
│   ├── exceptions.py           # Full custom exception hierarchy
│   ├── logger.py               # JSON formatter, rotating file handler
│   └── metrics.py              # RollingWindow, LatencyTracker, FPSCounter
│
├── data/
│   └── faces/                  # Runtime: one subdir per identity (git-ignored)
│
├── logs/                       # Runtime: rotating JSON log files (git-ignored)
│
└── tests/
    ├── __init__.py
    ├── test_use_cases.py        # RegisterFaceUseCase + VerifyFaceUseCase unit tests
    ├── test_repository.py       # FileFaceRepository round-trip + error tests
    └── test_metrics.py          # RollingWindow, LatencyTracker, FPSCounter tests
```

### Layer Dependency Rule

```
utils  ←  core  ←  services  ←  ui  ←  main.py
```

- **No layer may import from a layer above it.** `core` never imports from `services` or `ui`. `services` never imports from `ui`. This is enforced structurally and verified by code review.
- `utils` is imported by all layers (logging, config, exceptions are cross-cutting).
- `main.py` is the only place that imports from all layers (dependency injection root).

---

## 4. Architecture Deep-Dive

### 4.1 Layered Architecture

**Layer 0 — `utils/`**
Cross-cutting infrastructure: logging, configuration, exceptions, metrics. No business logic. No UI. No CV.

**Layer 1 — `core/`**
Pure domain logic. Zero external dependencies beyond stdlib and NumPy. Contains:
- `entities.py`: Immutable (frozen) dataclasses that represent domain concepts (`FaceIdentity`, `VerificationResult`, `BoundingBox`, `SampleResult`, `RegistrationSession`).
- `interfaces.py`: Abstract base classes (`ABC`) defining contracts for camera, detector, embedder, repository, and engine. These are the only types `core` knows about from the service layer — never concrete classes.
- `use_cases.py`: Orchestration logic implementing registration and verification workflows. These are fully unit-testable with mock implementations — no camera or UI required.

**Layer 2 — `services/`**
Concrete implementations of `core` interfaces using OpenCV, dlib, and the file system. Each module is independently replaceable (e.g., swap `HaarFaceDetector` for a DNN detector without touching any other file).

**Layer 3 — `ui/`**
PyQt6 widgets. Communicates with services exclusively through the `EventBus` (Qt signals) — never by direct method calls. All UI mutations happen on the main thread via Qt's signal dispatch.

### 4.2 Event Bus

`ui/event_bus.py` is a singleton `QObject` that owns all application-level Qt signals. Emitting a signal from a background thread is safe in Qt — it enqueues the call to the main thread automatically (queued connection). This means:

- `InferenceWorker` (a QThread) emits `result_ready` → Qt automatically delivers it to the main thread
- `TrainingWorker` emits `finished` → delivered to main thread
- No `QMetaObject.invokeMethod` boilerplate needed at call sites

### 4.3 Repository Pattern

`IFaceRepository` in `core/interfaces.py` defines `save`, `load`, `load_all`, `delete`, `exists`, `list_ids` as abstract methods. `FileFaceRepository` in `services/face_repository.py` is the concrete implementation. Swapping to a SQLite, PostgreSQL, or cloud storage backend requires only writing a new class that implements `IFaceRepository` and changing one line in `main.py`.

### 4.4 Strategy Pattern (Recognition Engine)

`IRecognitionEngine` defines a `match(embedding, identities)` contract. `CosineRecognitionEngine` implements brute-force cosine similarity — suitable for up to ~1,000 identities at real-time frame rates. To scale to millions of identities, implement `IRecognitionEngine` using FAISS (Facebook's approximate nearest neighbor library) without changing any calling code.

### 4.5 Dependency Injection

`main.py` is the sole composition root. It:
1. Loads `AppConfig` singleton
2. Constructs concrete service instances (`HaarFaceDetector`, `FaceRecognitionEmbedder`, etc.)
3. Passes them into use cases and `MainWindow`
4. Launches `QApplication`

No service locator anti-pattern; no global mutable service instances.

---

## 5. Recognition Pipeline

### 5.1 Registration

```
Webcam frame
    │
    ▼
[Frame Gap Enforcer]  ← every Kth frame only (default K=5, for pose diversity)
    │
    ▼
[HaarFaceDetector]    ← detectMultiScale on histogram-equalised grayscale
    │
    ├─ 0 faces  → REJECTED_NO_FACE
    ├─ 2+ faces → REJECTED_MULTIPLE_FACES
    └─ 1 face   ↓
    │
[Size Validator]      ← BoundingBox.min_side ≥ min_face_size (default 80px)
    │
    └─ too small → REJECTED_TOO_SMALL
    │
[Blur Validator]      ← Laplacian variance of face ROI ≥ blur_threshold (default 100)
    │
    └─ blurry → REJECTED_BLUR
    │
[FaceRecognitionEmbedder]  ← dlib 128-dim float32 vector
    │
[FIFO Sample Buffer]  ← maxlen=max_samples_per_identity (default 20), evicts oldest
    │
[Progress: accepted / target_samples]
    │ (when target reached)
    ▼
[TrainingWorker QThread]
    │
[FileFaceRepository.save()]  ← atomic write: .npy.tmp → rename → .npy
    │
    ▼
Identity registered ✓
```

**Why frame gap enforcement?** Sampling every frame produces near-identical embeddings (the face hasn't moved). Enforcing a gap of K frames forces the subject to shift their head slightly between samples, resulting in embeddings that span a larger region of embedding space. This dramatically improves verification robustness.

**Why Laplacian variance for blur?** The Laplacian operator is a second-derivative edge detector. A sharp image has many strong edges → high variance. A blurry image has weak edges → low variance. It's fast (single `cv2.Laplacian` call), deterministic, and requires no external model.

### 5.2 Verification

```
Webcam frame
    │
    ▼
[HaarFaceDetector]    ← detects all faces in frame
    │
    ▼
For each detected face:
    │
    ▼
[FaceRecognitionEmbedder]  ← 128-dim float32 vector per face
    │
    ▼
[CosineRecognitionEngine.match()]
    │   cosine_similarity(query, mean_embedding[i]) for all registered i
    │   → best_score, best_identity
    │
    ├─ score < similarity_threshold  → UNKNOWN (red overlay)
    ├─ score ≥ confidence_medium     → MEDIUM  (yellow overlay)
    └─ score ≥ confidence_high       → HIGH    (green overlay)
    │
[Liveness Hint]       ← frame delta variance vs previous frame
    │   score ≈ 0 → possible static photo
    │
[VerificationResult]  ← emitted via Qt signal to main thread
    │
[VideoWidget.paintEvent()]  ← bounding box + label painted on next repaint
```

**Cosine similarity vs Euclidean distance:** Cosine similarity is preferred for face embeddings because it measures the angle between vectors (ignoring magnitude), which is more robust to lighting and camera exposure variations that uniformly scale the embedding vector.

**Mean embedding strategy:** During verification, each registered identity is represented by the mean of all its sample embeddings. This is computationally equivalent to a single vector comparison per identity (pre-computed at `reload()` time) and smooths out per-sample noise.

### 5.3 Liveness Detection (Motion Variance)

The liveness check computes the mean absolute pixel difference between the current and previous grayscale frames. A photo or screen replay produces near-zero difference (static scene). A live human introduces continuous micro-movements (blinking, breathing, head sway) that produce a non-zero score.

This is a **soft hint**, not a cryptographic guarantee — it flags the UI with a warning ("Possible static image") rather than hard-blocking verification. For production medical or high-security use, replace with a challenge-response liveness model (see Section 13).

---

## 6. Threading Model

### Thread inventory

| Thread | Name | Role |
|---|---|---|
| Main | `MainThread` | Qt event loop, UI rendering, timer callbacks |
| Camera | `CameraCapture` | Continuous `cap.read()` loop, ring buffer writes |
| Inference | `InferenceWorker` | Pops frames from bounded queue, runs detect+embed+match |
| Training | `TrainingWorker` | Saves FaceIdentity to disk; one-shot per registration |
| Disk I/O | `DiskIO-0` | ThreadPoolExecutor for async I/O (not currently blocking main) |

### Frame data flow

```
[CameraCapture thread]
    cap.read() → frame
    deque.append(frame)         ← O(1), GIL-protected, drops oldest if full
         │
         │  (QTimer @ 33ms, main thread polls)
         ▼
[Main thread — _pump_frame()]
    camera.latest_frame()       ← returns copy of deque[-1]
    video_widget.set_frame()    ← immediate display (no lag)
    inference_queue.put_nowait() ← non-blocking; drops frame if queue full (maxsize=2)
         │
         ▼
[InferenceWorker thread]
    queue.get(timeout=0.05)
    use_case.verify(frame, identities)
    result_ready.emit(result)   ← Qt queued signal → delivers to main thread
         │
         ▼
[Main thread — _on_verification_result()]
    video_widget.set_verification_result(result)
    verify_panel.update_result(result)
```

### Why `maxsize=2` on the inference queue?

If inference takes 150ms/frame (slow GPU-less machine) and frames arrive at 30fps (33ms each), without backpressure the queue would grow unboundedly — 90 frames/second enqueued, 6 frames/second consumed. With `maxsize=2` and `put_nowait()`, excess frames are simply dropped. The UI always shows the latest live frame (it's updated independently on the main thread). The inference result may be 2 frames stale — imperceptible to the user.

### No `time.sleep()` polling

All inter-thread waiting uses:
- `queue.Queue.get(timeout=N)` — blocks the inference thread efficiently (OS scheduler, not spin-wait)
- `threading.Event.wait(timeout=N)` — used in camera reconnect backoff
- Qt signal dispatch — camera thread never calls any UI method directly

### Thread crash recovery

`InferenceWorker.run()` wraps the inner loop in `try/except Exception`, logs the full traceback, and emits an `error_occurred` signal to display in the status bar. The thread continues running. A catastrophic crash (SegFault from dlib) is OS-level and unrecoverable by design.

---

## 7. Memory Management

### Ring buffer

`CameraService` uses `collections.deque(maxlen=ring_buffer_size)`. `deque.append()` is O(1) and automatically evicts the oldest element when at capacity. RSS memory usage from the camera thread is bounded to `ring_buffer_size × frame_bytes` (e.g., 5 × 640×480×3 ≈ 4.6 MB).

### Inference queue

`queue.Queue(maxsize=2)` with `put_nowait()` — at most 2 frames in flight at any time (≈ 1.8 MB).

### Frame lifecycle

```
camera thread:   cap.read() → deque.append(frame)       [1 reference]
main thread:     camera.latest_frame() → frame.copy()   [1 new reference, old released]
inference:       queue.put_nowait(frame.copy())          [1 reference in queue]
                 after verify(): del frame               [explicit release]
```

No frame is retained after processing. The `del frame` at the end of the inference loop body ensures the reference count drops to zero immediately (CPython GC is reference-counted).

### Embeddings

All embeddings are stored as `np.float32` (4 bytes/element) rather than `float64` (8 bytes). For 128-dim embeddings with 20 samples per identity and 100 identities: `100 × 20 × 128 × 4 = 10.2 MB`. For `float64` that would be 20.5 MB — same accuracy, double the memory.

### Training cap

`max_samples_per_identity` (default 20) enforces FIFO eviction in `RegistrationSession.accepted_samples`. The list never grows beyond this bound.

---

## 8. Failure Handling Matrix

| Scenario | Detection point | Handling |
|---|---|---|
| Camera unavailable at launch | `CameraService._open_camera()` raises `CameraUnavailableError` | Error dialog with Retry button; app stays alive |
| Camera disconnects mid-session | `cap.read()` returns `False` | Auto-reconnect loop: 3 attempts with `backoff × attempt` wait |
| Reconnect exhausted | All attempts fail | `CameraReconnectExhaustedError` stored in `last_error`; status bar shows error; video widget shows status text |
| No face in registration frame | `detect()` returns `[]` | `SampleResult(REJECTED_NO_FACE)`; feedback shown in panel; counter incremented |
| Multiple faces in registration | `detect()` returns 2+ boxes | `SampleResult(REJECTED_MULTIPLE_FACES)`; shown in feedback log |
| Face too small | `bbox.min_side < min_face_size` | `SampleResult(REJECTED_TOO_SMALL)`; shown with pixel count |
| Blurry frame | Laplacian variance < threshold | `SampleResult(REJECTED_BLUR)`; shown with score |
| No face in verification | `detect()` returns `[]` | `VerificationResult(matches=[])` → "No face detected" overlay |
| Multiple faces in verification | `detect()` returns N boxes | All N faces embedded and matched independently; all displayed |
| OOM during training save | `MemoryError` in `TrainingWorker` | Caught, partial data rolled back, `failed` signal emitted → error shown in panel |
| Corrupted `.npy` file | `np.load()` raises on `load_all()` | `CorruptedDataError` caught per-identity; that identity skipped with `logger.warning`; app continues |
| Corrupted `.json` metadata | `json.loads()` raises | Same handling as `.npy` corruption above |
| Training with 0 registered faces | `load_all()` returns `[]` | `InferenceWorker.set_identities([])` → engine returns `(None, 0.0)` for all frames; "Unknown" displayed |
| Embedding extraction failure | `face_recognition.face_encodings()` returns `[]` | `EmbeddingError` raised; caught in inference loop; face skipped; error logged |
| Thread crash (inference) | `except Exception` in `run()` | Full traceback logged; `error_occurred` signal emitted; thread continues |
| Uncaught exception (main) | `sys.excepthook` | Full traceback logged + shown in QMessageBox; app does not silently exit |
| Stale `.tmp` files from interrupted write | `os.replace()` atomicity | `.tmp` files cleaned up in `except` block; original file never left in broken state |

---

## 9. Configuration Reference

All values live in `config.yaml`. No magic numbers exist anywhere in code.

```yaml
camera:
  index: 0                      # OpenCV camera index (0=first webcam, 1=second, etc.)
  width: 640                    # Capture resolution width
  height: 480                   # Capture resolution height
  ring_buffer_size: 5           # Max frames held in ring buffer (bounded memory)
  reconnect_attempts: 3         # How many times to retry after disconnect
  reconnect_backoff_seconds: 2  # Base backoff; actual wait = backoff × attempt_number

recognition:
  similarity_threshold: 0.60   # Minimum cosine similarity to declare a match
  min_face_size: 80             # Minimum face bounding box side length in pixels
  blur_threshold: 100.0         # Minimum Laplacian variance to accept a frame
  samples_per_identity: 15      # Target number of samples per registration
  sample_frame_gap: 5           # Only sample every Nth frame (pose diversity)
  max_samples_per_identity: 20  # Hard cap; FIFO eviction above this
  confidence_high: 0.80         # similarity ≥ this → HIGH (green)
  confidence_medium: 0.65       # similarity ≥ this → MEDIUM (yellow)

storage:
  data_dir: ./data/faces        # Root directory for all identity data
  embeddings_ext: .npy          # Extension for embedding files
  metadata_ext: .json           # Extension for metadata files

logging:
  level: INFO                   # DEBUG | INFO | WARNING | ERROR | CRITICAL
  file: ./logs/app.log          # Rotating log file path
  max_bytes: 5242880            # Max log file size before rotation (5 MB)
  backup_count: 3               # Number of rotated log files to keep

ui:
  window_title: "FaceID"
  min_width: 1024
  min_height: 640
  fps_rolling_window: 30        # Number of frames used to compute displayed FPS
  debug_mode: false             # True → P99/mean latency shown in status bar
```

---

## 10. Observability & Metrics

### Structured JSON logging

Every log line is emitted as a JSON object to `logs/app.log`:

```json
{"ts": "2024-01-15T10:23:45.123+00:00", "level": "INFO", "logger": "services.camera_service", "msg": "Camera opened", "thread": "CameraCapture", "x_index": 0, "x_w": 640, "x_h": 480}
```

Key events logged:
- Thread start / stop / crash
- Every frame drop (inference queue full)
- Every sample accept/reject with reason
- Training start / end / duration
- Each match result with similarity score
- Camera reconnect attempts

### FPS Measurement

`FPSCounter` uses a rolling window of the last 30 frame timestamps. `fps = (window_size - 1) / (newest_ts - oldest_ts)`. This gives a true measured rate, not an assumed one. Displayed in the status bar, color-coded green (healthy) / yellow (degraded < 5fps).

### P99 Latency

`LatencyTracker` wraps a `RollingWindow` of the last 100 inference durations. Exposes `p50()`, `p95()`, `p99()`, and `mean()`. Shown in the status bar when `debug_mode: true` in config or `--debug` flag is passed. This is the single most important metric for evaluating real-time responsiveness.

### Custom Exception Hierarchy

```
FaceAppError
├── CameraError
│   ├── CameraUnavailableError
│   ├── CameraDisconnectedError
│   └── CameraReconnectExhaustedError
├── RecognitionError
│   ├── NoFaceDetectedError
│   ├── MultipleFacesError
│   ├── FaceBlurryError
│   ├── FaceTooSmallError
│   └── EmbeddingError
├── StorageError
│   ├── CorruptedDataError
│   ├── IdentityNotFoundError
│   └── AtomicWriteError
├── TrainingError
│   ├── InsufficientSamplesError
│   ├── TrainingMemoryError
│   └── DuplicateIdentityError
└── ConfigError
```

Every `except` clause catches a specific subclass — never bare `except Exception` in business logic. Log handlers use `exc_info=True` to capture full tracebacks into the JSON log.

---

## 11. Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Expected output:
```
tests/test_metrics.py::TestRollingWindow::test_maxlen_enforced         PASSED
tests/test_metrics.py::TestRollingWindow::test_empty_snapshot          PASSED
tests/test_metrics.py::TestRollingWindow::test_thread_safe_push        PASSED
tests/test_metrics.py::TestLatencyTracker::test_measure_records_latency PASSED
tests/test_metrics.py::TestLatencyTracker::test_p99_ordering           PASSED
tests/test_metrics.py::TestFPSCounter::test_fps_approximation          PASSED
tests/test_use_cases.py::TestRegisterFaceUseCase::test_begin_session_creates_slug  PASSED
tests/test_use_cases.py::TestRegisterFaceUseCase::test_process_frame_gap_skip      PASSED
...
tests/test_repository.py::TestFileFaceRepository::test_save_and_load_roundtrip    PASSED
tests/test_repository.py::TestFileFaceRepository::test_load_all_skips_corrupted   PASSED
...
```

**Key properties of the test suite:**
- `test_use_cases.py` — zero I/O, zero camera, zero UI. Uses `MockDetector`, `MockEmbedder`, `MockEngine` — pure Python. Runs in milliseconds.
- `test_repository.py` — uses `pytest`'s `tmp_path` fixture; all disk writes go to a temp dir that is deleted after each test. No permanent side effects.
- `test_metrics.py` — pure Python arithmetic and timing. No external dependencies.

---

## 12. CLI / Headless Mode

Verify a single image without starting the GUI:

```bash
python main.py --headless --verify --image /path/to/photo.jpg
```

Output (JSON to stdout):
```json
{
  "faces_detected": 1,
  "inference_ms": 42.7,
  "matches": [
    {
      "identity_id": "alice_johnson",
      "display_name": "Alice Johnson",
      "similarity": 0.8712,
      "confidence": "HIGH",
      "bbox": {"x": 142, "y": 98, "w": 115, "h": 118}
    }
  ]
}
```

This is useful for:
- CI/CD pipeline integration
- Batch verification of image sets
- Integration testing without a display server

---

## 13. MedTech Upgrade Path

> What would need to change if this same application were deployed in a medical technology context — e.g., patient identification at a clinic, surgical suite access control, or ICU staff authentication?


---

### 13.1 Regulatory Compliance

**FDA / CE marking (if embedded in a medical device)**

If FaceID is used to control access to a medical device (e.g., unlocking a drug dispenser, authenticating a surgeon before robotic surgery), it may be classified as a **Software as a Medical Device (SaMD)** under FDA 21 CFR Part 11 or EU MDR 2017/745. This triggers:

- **IEC 62304** software lifecycle compliance (documented risk analysis, design controls, V&V testing)
- **21 CFR Part 11** electronic records and signatures — all authentication events must be tamper-evident, audit-logged with the authenticated user, date/time, and action
- **HIPAA** (US) / **GDPR** (EU) — biometric face embeddings are considered sensitive health data and require encryption at rest, in transit, and strict access controls

**What to change:**
- Replace the JSON log with an append-only, cryptographically signed audit log (HMAC-SHA256 per entry)
- Implement role-based access control (RBAC) — not everyone can register new identities
- Store embeddings encrypted at rest (AES-256-GCM); key management via HSM or OS keychain
- All authentication events must be non-repudiable and timestamped with an NTP-synchronized clock

---

### 13.2 Accuracy & Error Rate Requirements

Consumer applications tolerate False Acceptance Rates (FAR) of 1-5%. Medical applications require:

| Metric | Consumer (current) | Medical (target) |
|---|---|---|
| False Acceptance Rate (FAR) | ~1–2% | < 0.001% (1 in 100,000) |
| False Rejection Rate (FRR) | ~5–10% | < 1% |
| Liveness detection | Motion variance hint | ISO 30107-3 Level 2 certified |

**What to change:**
- Replace `face_recognition` (dlib ResNet) with a higher-accuracy model: ArcFace, FaceNet, or a medically-validated proprietary model
- Increase `samples_per_identity` to 50–100 with explicit pose variation (front, left 30°, right 30°, looking up/down)
- Move from mean embedding to an ensemble strategy (e.g., nearest neighbor over all samples, or SVM classifier per identity)
- Replace the motion-variance liveness hint with a certified 3D liveness model (e.g., depth camera-based, or challenge-response blink/turn detection)
- Implement threshold calibration with held-out validation data specific to your patient/staff population demographic

---

### 13.3 Multi-Factor Authentication

Face recognition alone is insufficient for high-security medical actions. The principle of **multi-factor authentication (MFA)** applies:

- **Something you are** (face biometric — current)
- **Something you have** (RFID badge, phone OTP)
- **Something you know** (PIN, for admin actions)

**What to change:**
- Add a second factor for actions that modify patient records or dispense medication
- Face verification becomes the "fast path" for low-risk actions (room entry) only
- Design the auth flow to fall back gracefully when biometrics fail (gloves, surgical mask, lighting)

---

### 13.4 Robustness in Clinical Environments

Clinical environments are hostile to face recognition:
- Staff wear **surgical masks, caps, visors, goggles**
- **Lighting** is extreme: operating theatre LED arrays, dim ICU rooms
- **Camera angle**: mounted high on a wall, not at eye level
- **Time pressure**: a surgeon needing to access controls during a procedure cannot spend 5 seconds on face scan

**What to change:**
- Use a model trained on partial-face recognition (masked face datasets exist: MFR2, RMFRD)
- Add infrared (IR) camera support — IR is lighting-invariant and works in the dark
- Implement a **fallback authentication path** (badge tap + PIN) that the system falls back to automatically after 2 failed face attempts
- Design for sub-500ms total authentication latency (detection + embedding + match must complete in one video frame)

---

### 13.5 Data Governance & Privacy

Face embeddings are biometric data. Under GDPR Article 9 and HIPAA, they require:

- **Explicit consent** from each enrolled individual with documented withdrawal mechanism
- **Purpose limitation** — embeddings registered for staff authentication cannot be used for attendance tracking
- **Data minimisation** — embeddings only, never raw face images stored (the current `.npy` approach is correct)
- **Right to erasure** — the `delete()` method in `IFaceRepository` already supports this; it must be surfaced in an admin UI with audit logging
- **Cross-border data transfer** restrictions if using cloud infrastructure

**What to change:**
- Add a consent capture workflow (signature or checkbox) before any face is registered
- Implement a privacy officer admin panel showing all enrolled identities, enrollment dates, and a compliant delete function
- Move `data_dir` to an encrypted volume (LUKS on Linux, BitLocker on Windows, FileVault on macOS)
- Never store raw frames to disk (already satisfied in the current design)

---

### 13.6 High Availability & Failover

A hospital corridor door or medication dispenser that fails to authenticate due to a software crash is a patient safety issue.

**What to change:**
- Run the recognition service as a **system daemon** (systemd service) with automatic restart on crash
- Implement a **local cache** of embeddings that persists across app restarts (already done via `.npy` files)
- Add a **hardware watchdog** timer that triggers an alert (and unlocks the door in fail-safe mode) if the recognition process hangs
- If network-connected: use a primary + hot-standby recognition server; local device falls back to local embeddings if network is unavailable
- Target **99.9% uptime** with a documented Mean Time To Recovery (MTTR) < 60 seconds

---

### 13.7 Performance at Scale

A single hospital may have 2,000–5,000 staff members. The current brute-force cosine similarity has O(N) per-frame complexity.

| Staff count | Brute-force @ 128-dim float32 | FAISS IVF flat |
|---|---|---|
| 100 | ~0.1ms | ~0.1ms |
| 1,000 | ~1ms | ~0.2ms |
| 5,000 | ~5ms | ~0.5ms |
| 50,000 | ~50ms (exceeds frame budget) | ~1ms |

**What to change:**
- Replace `CosineRecognitionEngine` with a FAISS `IndexFlatIP` (exact cosine) or `IndexIVFFlat` (approximate, sub-linear) without changing any other code — same `IRecognitionEngine` interface
- Pre-compute and cache the FAISS index on startup; rebuild only when identities change
- Add a `reload()` signal from the repository layer to the engine layer (already in the `IRecognitionEngine.reload()` contract)

---

### 13.8 Audit & Incident Response

**What to change:**
- Every authentication attempt (success or failure) must be written to an **append-only audit database** (PostgreSQL with row-level security, or a dedicated audit log service)
- Audit record must contain: timestamp (UTC, millisecond precision), `identity_id`, `similarity_score`, `confidence_band`, `liveness_score`, camera ID, device ID, action attempted
- Failed attempts beyond a threshold (e.g., 5 consecutive failures) must trigger an alert to security operations
- Retain audit logs for the period mandated by local regulation (7 years in many EU medical contexts)

---

### Summary: Delta from current implementation to MedTech-ready

| Area | Current | MedTech target |
|---|---|---|
| Recognition model | dlib ResNet (face_recognition) | ArcFace / FaceNet + custom fine-tune |
| Liveness | Motion variance (hint) | ISO 30107-3 Level 2 certified model |
| FAR | ~1–2% | < 0.001% |
| Authentication | Single factor | Multi-factor (face + badge/PIN) |
| Storage encryption | Plaintext `.npy` | AES-256-GCM, HSM key management |
| Audit logging | JSON rotating file | Append-only, signed, retained 7 years |
| Compliance | None | IEC 62304, FDA 21 CFR Part 11, HIPAA/GDPR |
| Similarity engine | Brute-force O(N) | FAISS IVF (sub-linear, GPU-accelerated) |
| Availability | Best-effort | 99.9% SLA, watchdog, fail-safe fallback |
| Privacy | Basic delete | Consent capture, right-to-erasure workflow |
| Partial-face handling | Not supported | Masked-face model + IR camera |

The architectural foundation — layered design, repository pattern, strategy pattern, event bus, bounded memory, clean thread model — is already **production-grade and medtech-appropriate**. The upgrade path is additive, not a rewrite.
