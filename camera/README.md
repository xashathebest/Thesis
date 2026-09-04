# Camera Integration

The Version 1 operator dashboard uses the single-owner camera lifecycle service in
`src/api/camera.py`. Python/OpenCV opens the selected laptop-connected webcam,
passes each frame through the loaded YOLO model once, and publishes the annotated
result as an MJPEG stream. The browser does not request direct camera access.

`capture.py` remains reserved for future offline acquisition utilities. Camera,
conveyor, motor, servo, and other hardware control remain out of scope.
