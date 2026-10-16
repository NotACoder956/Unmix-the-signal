# Place physical multi-device recordings here

Capture the same sound scene simultaneously on at least two separate devices; three devices are recommended for the three-source example. The files must overlap in time. See [`docs/REAL_WORLD_RECORDING_GUIDE.md`](../../docs/REAL_WORLD_RECORDING_GUIDE.md) for the recording procedure.

Expected example filenames (do not add fake or procedural audio here and call it real-world):

- `device_1.wav`
- `device_2.wav`
- `device_3.wav`

Run the experiment with:

```bash
python scripts/run_device_recordings.py --inputs data/real_world/device_1.wav data/real_world/device_2.wav data/real_world/device_3.wav --output-dir outputs/real_world
```

This directory intentionally contains no fabricated multi-device recording. Real recordings must be captured on actual devices.
