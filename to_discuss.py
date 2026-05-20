Solved problems
embedder.py
Fixed face_recognition embedding crash by making the RGB frame contiguous and ensuring bbox coordinates are Python ints.


camera_service.py
Made camera reconnect less aggressive by only reconnecting after multiple consecutive read failures.
Added a safer CAP_PROP_BUFFERSIZE set guarded by try/except.


use_cases.py
Added temporal liveness smoothing to avoid noisy single-frame motion checks.
Fixed liveness logic so a valid identity match is not incorrectly forced to UNKNOWN.

main_window.py
Added “process every Nth frame” behavior to reduce inference load while keeping the preview smooth.
Added debug-style inference logging and queue backpressure handling.



main.py
Added a proper signal handler for SIGINT/SIGTERM so Ctrl+C exits the Qt app cleanly.


config.yaml / config.py
Added config options for verify_frame_gap, liveness_threshold, and liveness_window.
Added validation for those new settings.


####Problems that should be solved with team discussion

1) Detector robustness and pose tolerance
Current Haar frontal-face detector is weak for head-up, sideways, tilted, or partially occluded faces.
Team should decide whether to:
add profile cascades,
switch to a more modern detector (DNN, MTCNN, RetinaFace),
or adopt a hybrid multi-detector strategy.
2)Spoofing / liveness security
Current motion-based liveness is a weak heuristic.
Team discussion needed to choose an appropriate anti-spoofing strategy for the product:
challenge-response,
ML-based spoof detector,
depth/IR support,
or a policy-based safety threshold.
3) User experience vs accuracy tradeoff
How aggressive should frame skipping be?
Should the system prioritize responsiveness or recognition accuracy?
This affects UI behavior and acceptable false reject rates.
4) Camera/hardware reliability
The repeated camera reconnect behavior suggests backend/hardware issues rather than pure app logic.
Team should decide on supported camera platforms and whether to add hardware diagnostics or fallback modes.
5)Scalability and identity management
If this is used in med tech,
how many identities will be supported,
whether cosine matching is enough,
whether a more structured identity database is needed.

###Improvements for med-tech use
Reliability
Use a stronger face detector that handles pose, lighting, and partial occlusion.
Add multi-frame stability checks before accepting a verification result.

Safety
Add anti-spoofing / liveness detection beyond simple frame differencing.
Log failed attempts and support audit trails.

Accuracy
Collect multiple registration samples from each user under different angles and lighting.
Tune similarity thresholds and confidence bands for clinical-grade consistency.

Usability
Improve feedback when detection fails: “Please look at the camera” or “Head too tilted”.
Reduce latency by processing fewer frames for inference and keeping preview fluid.

Compliance
Ensure identity data and embeddings are stored securely.
Consider access controls and data retention policies for medical environments.
