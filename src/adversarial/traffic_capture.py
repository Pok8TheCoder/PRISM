"""Traffic capture module using tcpdump inside the Docker target container."""

import subprocess
import time
from pathlib import Path
from typing import Optional

from src.adversarial.lab_config import SAVE_DIR, TARGET_CONTAINER


class TrafficCapture:
    def __init__(
        self,
        container_name: str = TARGET_CONTAINER,
        output_dir: str | Path = SAVE_DIR,
    ):
        self.container_name = container_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Optional[Path] = None

    def start_capture(self, filename: str = "capture.pcap") -> Path:
        self._current_file = self.output_dir / filename

        subprocess.run(
            ["docker", "exec", self.container_name, "pkill", "tcpdump"],
            capture_output=True,
            timeout=5,
        )
        time.sleep(0.3)

        subprocess.run(
            [
                "docker", "exec", "-d", self.container_name,
                "tcpdump", "-i", "eth0", "-w", f"/tmp/{filename}", "-U",
                "not", "port", "2222",
            ],
            capture_output=True,
            timeout=10,
        )
        time.sleep(2)
        return self._current_file

    def stop_capture(self) -> Optional[Path]:
        if not self._current_file:
            return None

        subprocess.run(
            ["docker", "exec", self.container_name, "pkill", "-SIGINT", "tcpdump"],
            capture_output=True,
            timeout=10,
        )
        time.sleep(2)

        container_path = f"/tmp/{self._current_file.name}"
        local_path = self._current_file

        subprocess.run(
            ["docker", "cp", f"{self.container_name}:{container_path}", str(local_path)],
            capture_output=True,
            timeout=30,
        )

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
