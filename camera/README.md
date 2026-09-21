# Camera integration

The canonical camera implementation is `src/api/camera.py`; it is the sole owner
of the OpenCV capture handle. It runs Model 1 whole-fish detection/tracking, then
the Model 2 Head/Body/Tail evidence pipeline, and publishes an annotated MJPEG
stream. The browser never opens the camera directly.

`GET /api/camera/settings` reports only properties that the active OpenCV/UVC
backend can actually probe. During explicit Calibration Mode, supported controls
such as exposure, gain, white balance, focus, resolution, and FPS can be requested
through `POST /api/camera/settings`; each accepted value is read back and stored in
the durable session profile. Unsupported driver properties remain unavailable and
never crash inspection.

Inspection policy and acquisition changes clear active track evidence and create a
new durable configuration/session snapshot. The repository does not assume a
specific EMEET/UVC property range or device name, because OpenCV does not expose
those reliably across Windows camera backends.

## Formal research calibration

A saved profile is not automatically a calibrated profile. For an independent
study, an operator must explicitly complete this sequence while the camera is
running, unlocked, and in Calibration Mode:

1. Record the physical device identity, camera height/angle, conveyor position,
   and lighting position.
2. Adjust only controls the active driver exposes; the profile stores both actual
   hardware readbacks and unsupported properties.
3. Record an `empty_conveyor` reference through `POST /api/camera/reference`.
   A `representative_fish` reference is optional. Reference statistics include
   brightness mean/median/standard deviation, fish/background contrast when a
   fish is present, and informational HSV means. They never affect a grade.
4. Save the profile, then call `POST /api/camera/profile/confirm` with
   `operator_confirmed: true` and the recorded installation metadata.
5. Lock the inspection camera with `POST /api/camera/lock`.

Profile confirmation requires an empty-conveyor reference and all physical setup
fields. A locked profile reports camera disconnection, backend/resolution/control
drift, and reference-brightness warnings; it never silently alters hardware or
grades to compensate. A formal validation session must treat such warnings as
acquisition evidence and either document or restart the study.

Saving changed hardware/acquisition evidence or replacing a reference scene clears
the prior operator confirmation, requiring a fresh explicit confirmation. Lock and
unlock transitions alone preserve an otherwise unchanged confirmation; neither
operation adjusts the camera or the grading policy.
