"""Tiny dependency-free MJPEG server: lets you watch the annotated video in a
browser on your laptop while the script runs headless over SSH."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"--frame"


class MJPEGStreamer:
    def __init__(self, port=8080):
        self.port = port
        self._jpeg = None
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)

    def update(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._new_frame:
            self._jpeg = buf.tobytes()
            self._new_frame.notify_all()

    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
                )
                self.end_headers()
                try:
                    while True:
                        with streamer._new_frame:
                            streamer._new_frame.wait(timeout=5)
                            jpeg = streamer._jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(BOUNDARY + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # viewer closed the tab

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"MJPEG stream: http://<jetson-ip>:{self.port}/")
