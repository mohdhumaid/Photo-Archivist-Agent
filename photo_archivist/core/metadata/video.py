"""Step 3d: video/audio via ffprobe + keyframe extraction (with PyAV / OpenCV fallbacks)."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile


def ffprobe(path: str) -> dict:
    """Probe video/audio metadata: ffprobe binary first, then PyAV, then OpenCV, then audio fallbacks."""
    # 1. Try ffprobe binary
    exe = shutil.which("ffprobe")
    if exe:
        try:
            out = subprocess.run(
                [exe, "-v", "quiet", "-print_format", "json", "-show_format",
                 "-show_streams", path],
                capture_output=True, text=True,
            )
            data = json.loads(out.stdout or "{}")
            if data and ("streams" in data or "format" in data):
                return data
        except Exception:
            pass

    # 2. Try PyAV
    pyav_res = _pyav_probe(path)
    if "_error" not in pyav_res:
        return pyav_res

    # 3. Try OpenCV
    cv2_res = _cv2_probe(path)
    if "_error" not in cv2_res:
        return cv2_res

    # 4. Try audio-specific probe
    audio_res = _audio_probe(path)
    if "_error" not in audio_res:
        return audio_res

    # All failed, return best available error
    if "_error" in audio_res:
        return audio_res
    elif "_error" in cv2_res:
        return cv2_res
    elif "_error" in pyav_res:
        return pyav_res
    else:
        return {"_error": "Could not probe video/audio metadata with any available tool.", "_via": "fallback_chain"}



def _pyav_probe(path: str) -> dict:
    """Video/audio format & streams via PyAV (bundled FFmpeg C libraries in Python)."""
    try:
        import av
    except ImportError:
        return {"_error": "PyAV not installed", "_via": "av"}
    try:
        container = av.open(path)
    except Exception as e:
        return {"_error": f"av cannot open file: {e}", "_via": "av"}

    try:
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        dur = float(container.duration / av.time_base) if container.duration else 0.0
        fmt = {
            "filename": os.path.abspath(path),
            "nb_streams": len(container.streams),
            "format_name": container.format.name,
            "duration": str(dur),
            "size": str(sz),
            "bit_rate": str(container.bit_rate or 0),
            "tags": dict(container.metadata or {}),
        }
        streams = []
        for s in container.streams:
            st_info = {
                "index": s.index,
                "codec_type": s.type,
                "codec_name": getattr(s.codec_context, "name", "unknown"),
            }
            if s.type == "video":
                st_info["width"] = s.codec_context.width if s.codec_context else 0
                st_info["height"] = s.codec_context.height if s.codec_context else 0
                fps = float(s.average_rate) if s.average_rate else 0.0
                st_info["avg_frame_rate"] = f"{fps}/1"
                st_info["nb_frames"] = str(s.frames or 0)
                st_info["duration"] = str(dur)
            elif s.type == "audio":
                st_info["channels"] = getattr(s.codec_context, "channels", 0)
                st_info["sample_rate"] = str(getattr(s.codec_context, "sample_rate", 0))
            if getattr(s, "metadata", None):
                st_info["tags"] = dict(s.metadata)
            streams.append(st_info)

        return {"format": fmt, "streams": streams, "_via": "av"}
    except Exception as e:
        return {"_error": f"av probe failed: {e}", "_via": "av"}
    finally:
        try:
            container.close()
        except Exception:
            pass


def _cv2_probe(path: str) -> dict:
    """Video format/streams via OpenCV (no ffprobe needed)."""
    try:
        import cv2
    except ImportError:
        return {"_error": "ffprobe missing and opencv not installed", "_via": "cv2"}
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return {"_error": "cv2 cannot open this video", "_via": "cv2"}
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if w <= 0 or h <= 0:
            return {"_error": "cv2 cannot decode stream dimensions", "_via": "cv2"}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        try:
            codec = "".join(chr(int((fourcc >> (8 * i)) & 255)) for i in range(4))
        except Exception:
            codec = ""
        dur = (n / fps) if (fps > 0 and n > 0) else 0.0
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        return {
            "format": {
                "filename": os.path.abspath(path),
                "duration": str(dur),
                "size": str(sz),
            },
            "streams": [{
                "codec_type": "video",
                "width": w,
                "height": h,
                "avg_frame_rate": f"{fps}/1",
                "codec_name": codec.strip("\x00") or "unknown",
                "duration": str(dur),
                "nb_frames": str(n)
            }],
            "_via": "cv2"
        }
    except Exception as e:
        return {"_error": f"cv2 probe failed: {e}", "_via": "cv2"}
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _audio_probe(path: str) -> dict:
    """Basic audio container probing via standard library wave."""
    ext = os.path.splitext(path)[1].lower()
    sz = os.path.getsize(path) if os.path.exists(path) else 0
    if ext == ".wav":
        try:
            import wave
            with wave.open(path, "rb") as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                frames = wf.getnframes()
                dur = frames / float(rate) if rate > 0 else 0.0
                return {
                    "format": {"filename": os.path.abspath(path), "duration": str(dur), "size": str(sz)},
                    "streams": [{
                        "codec_type": "audio",
                        "codec_name": "pcm_s16le",
                        "channels": channels,
                        "sample_rate": str(rate),
                        "duration": str(dur)
                    }],
                    "_via": "wave"
                }
        except Exception:
            pass
    return {"_error": f"unsupported audio container: {ext}", "_via": "audio"}


def keyframe(path: str, at_seconds: float = 1.0) -> str | None:
    """Extract a single JPEG keyframe to a temp file for vision. Returns temp path.

    ffmpeg first; PyAV -> OpenCV VideoCapture (seek + imwrite) when absent.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        try:
            subprocess.run(
                [exe, "-y", "-v", "error", "-ss", str(at_seconds), "-i", path,
                 "-frames:v", "1", "-q:v", "3", tmp.name],
                capture_output=True, timeout=60,
            )
            if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                return tmp.name
        except Exception:
            pass
        if os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except Exception:
                pass

    kf_pyav = _pyav_keyframe(path, at_seconds)
    if kf_pyav:
        return kf_pyav

    return _cv2_keyframe(path, at_seconds)


def _pyav_keyframe(path: str, at_seconds: float = 1.0) -> str | None:
    """Extract keyframe via PyAV (Python-FFmpeg bindings)."""
    try:
        import av
        from PIL import Image
    except ImportError:
        return None
    try:
        container = av.open(path)
        if not container.streams.video:
            container.close()
            return None
        stream = container.streams.video[0]
        target_pts = int(max(0.0, at_seconds) / float(stream.time_base)) if stream.time_base else 0
        try:
            container.seek(target_pts, any_frame=False, stream=stream)
        except Exception:
            pass
        frame_img = None
        for frame in container.decode(stream):
            frame_img = frame.to_image()
            break
        container.close()
        if frame_img:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            frame_img.save(tmp.name, "JPEG", quality=88)
            return tmp.name
    except Exception:
        pass
    return None


def _cv2_keyframe(path: str, at_seconds: float = 1.0) -> str | None:
    """Keyframe via OpenCV seek + imwrite (no ffmpeg needed)."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, at_seconds) * 1000.0)
        except Exception:
            pass
        ok, frame = cap.read()
        if not ok or frame is None:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass
            for _ in range(30):
                try:
                    ok, frame = cap.read()
                except Exception:
                    break
                if ok and frame is not None:
                    break
        if not ok or frame is None:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        if cv2.imwrite(tmp.name, frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            return tmp.name
        return None
    except Exception:
        return None
    finally:
        try:
            cap.release()
        except Exception:
            pass
