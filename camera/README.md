# Camera Integration

The Version 2 operator dashboard uses the single-owner camera lifecycle service in
`src/api/camera.py`. Python/OpenCV opens the selected laptop-connected webcam and
passes each frame through YOLOv8 plus Ultralytics ByteTrack once. The service draws
persistent fish IDs and the configured inspection line, then publishes the result
as an MJPEG stream. The browser does not request direct camera access.

`capture.py` remains reserved for future offline acquisition utilities. Camera,
conveyor, motor, servo, and other hardware control remain out of scope.
