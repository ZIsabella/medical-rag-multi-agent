from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a dataset file to be downloaded."""

    name: str
    url: str
    output_path: Path
    sha256: str | None = None


class DatasetDownloader:
    """
    Ensures required dataset files exist locally.

    Behavior:
    - Does not re-download an existing non-empty valid file.
    - Downloads to a temporary file first.
    - Optionally validates SHA-256.
    - Moves the file to its target path only after successful download.
    """

    def __init__(self, timeout_seconds: int = 120) -> None:
        self._timeout_seconds = timeout_seconds

    def ensure(self, spec: DatasetSpec) -> Path:
        """
        Returns the local dataset path.

        If a valid file already exists, download is skipped.
        Otherwise, the file is downloaded safely and then moved
        to spec.output_path.
        """
        if self._is_valid_existing_file(
            file_path=spec.output_path,
            expected_sha256=spec.sha256,
        ):
            logger.info(
                "Dataset already exists; skipping download: name=%s path=%s",
                spec.name,
                spec.output_path,
            )
            return spec.output_path

        if not spec.url or not spec.url.strip():
            raise ValueError(
                f"Dataset URL is empty for '{spec.name}'. "
                "Set the corresponding dataset URL environment variable."
            )

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading dataset: name=%s url=%s target=%s",
            spec.name,
            spec.url,
            spec.output_path,
        )

        temp_path = self._download_to_temp_file(
            url=spec.url,
            destination_dir=spec.output_path.parent,
        )

        try:
            if spec.sha256 is not None:
                self._validate_sha256(
                    file_path=temp_path,
                    expected_sha256=spec.sha256,
                )

            shutil.move(str(temp_path), str(spec.output_path))

            logger.info(
                "Dataset download completed successfully: name=%s path=%s",
                spec.name,
                spec.output_path,
            )

            return spec.output_path

        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _download_to_temp_file(
        self,
        url: str,
        destination_dir: Path,
    ) -> Path:
        """
        Downloads a URL to a temporary file in destination_dir.
        Returns the temporary file path after successful download.
        """
        destination_dir.mkdir(parents=True, exist_ok=True)

        import tempfile

        # ... داخل متد _download_to_temp_file ...

        temp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=tempfile.gettempdir(), 
            prefix=".download_",
            suffix=".tmp",
        )

        temp_path = Path(temp_file.name)

        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Medical-RAG-Dataset-Downloader/1.0"
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                shutil.copyfileobj(response, temp_file)

            temp_file.close()

            if temp_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Downloaded file is empty. URL: {url}"
                )

            return temp_path

        except urllib.error.HTTPError as error:
            temp_file.close()
            temp_path.unlink(missing_ok=True)

            raise RuntimeError(
                "HTTP error while downloading dataset. "
                f"status={error.code}, url={url}"
            ) from error

        except urllib.error.URLError as error:
            temp_file.close()
            temp_path.unlink(missing_ok=True)

            raise RuntimeError(
                f"Network error while downloading dataset. url={url}"
            ) from error

        except Exception:
            temp_file.close()
            temp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _is_valid_existing_file(
        file_path: Path,
        expected_sha256: str | None,
    ) -> bool:
        """Checks whether an existing file is non-empty and valid."""
        if not file_path.is_file() or file_path.stat().st_size == 0:
            return False

        if expected_sha256 is None:
            return True

        actual_sha256 = DatasetDownloader._calculate_sha256(file_path)

        return actual_sha256.lower() == expected_sha256.lower()

    @staticmethod
    def _validate_sha256(
        file_path: Path,
        expected_sha256: str,
    ) -> None:
        """Raises RuntimeError when a file checksum does not match."""
        actual_sha256 = DatasetDownloader._calculate_sha256(file_path)

        if actual_sha256.lower() != expected_sha256.lower():
            raise RuntimeError(
                "SHA-256 validation failed for downloaded dataset. "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )

    @staticmethod
    def _calculate_sha256(file_path: Path) -> str:
        """Calculates SHA-256 incrementally to avoid loading a large file."""
        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()
