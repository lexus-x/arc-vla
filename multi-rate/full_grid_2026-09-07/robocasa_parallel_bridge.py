"""One isolated RoboCasa simulator process per client; existing wire protocol.

Linux only. Run in robocasa_uv. The original single-client bridge cannot serve
multiple harness workers concurrently. This changes scheduling, not simulation.
"""
import argparse
import socketserver
from robocasa_bridge import handle


class Client(socketserver.BaseRequestHandler):
    def handle(self):
        handle(self.request, {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    with socketserver.ForkingTCPServer(("127.0.0.1", args.port), Client) as server:
        print(f"PORT={server.server_address[1]}", flush=True)
        server.serve_forever()
