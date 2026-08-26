"""Traffic capture module using tcpdump inside the Docker target container."""

import subprocess
import time
from pathlib import Path
from typing import Optional


class TrafficCapture:
    def __init__(self, container_name: str = "target-server", output_dir: str = "data/raw/adversarial"):
        self.container_name = container_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Optional[Path] = None

    def start_capture(self, filename: str = "capture.pcap") -> Path:
        self._current_file = self.output_dir / filename

        # Kill any existing tcpdump first
        subprocess.run(
            ["docker", "exec", self.container_name, "pkill", "tcpdump"],
            capture_output=True, timeout=5,
        )
        time.sleep(0.3)

        # Start tcpdump detached inside the container
        subprocess.run(
            [
                "docker", "exec", "-d", self.container_name,
                "tcpdump", "-i", "eth0", "-w", f"/tmp/{filename}", "-U",
                "not", "port", "2222",
            ],
            capture_output=True, timeout=10,
        )

        # Wait for tcpdump to initialize
        time.sleep(2)
        return self._current_file

    def stop_capture(self) -> Optional[Path]:
        if not self._current_file:
            return None

        # Send SIGINT to tcpdump so it flushes and writes the pcap header
        subprocess.run(
            ["docker", "exec", self.container_name, "pkill", "-SIGINT", "tcpdump"],
            capture_output=True, timeout=10,
        )
        time.sleep(2)

        container_path = f"/tmp/{self._current_file.name}"
        local_path = self._current_file

        # Copy pcap from container to host
        result = subprocess.run(
            ["docker", "cp", f"{self.container_name}:{container_path}", str(local_path)],
            capture_output=True, timeout=30,
        )

        # Cleanup container file
        subprocess.run(
            ["docker", "exec", self.container_name, "rm", "-f", container_path],
            capture_output=True,
        )

        self._current_file = None
        return local_path

    def capture_session(self, attack_fn, filename: str = "capture.pcap"):
        self.start_capture(filename)
        try:
            result = attack_fn()
        finally:
            time.sleep(1)
            pcap_path = self.stop_capture()
        return result, pcap_path
