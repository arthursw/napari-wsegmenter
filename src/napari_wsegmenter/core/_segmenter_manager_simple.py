from typing import cast

import numpy as np

from wetlands.ndarray import NDArray

from ._segmenter_manager_base import SegmenterManagerBase


class SegmenterManager(SegmenterManagerBase):

    _shared_image: NDArray | None = None
    _shared_segmentation: NDArray | None = None

    def _initialize_shared_memory(self, image: np.ndarray):
        segmentation_shape = image.shape[:2]
        if (
            self._shared_image is not None
            and self._shared_segmentation is not None
        ):
            if (
                self._shared_image.array.dtype == image.dtype
                and self._shared_image.array.shape == image.shape
            ):
                self._shared_image.array[:] = image[:]
                self._shared_segmentation.array[:] = np.zeros(
                    segmentation_shape, "uint8"
                )
                return
            else:
                self.release_shared_memory()
        self._shared_image = NDArray(image)
        self._shared_segmentation = NDArray(
            np.zeros(segmentation_shape, "uint8")
        )

    def perform_segmentation(
        self, image: np.ndarray, segmenter: str, shared_memory: bool = True
    ):
        if not shared_memory:
            return super().perform_segmentation(image, segmenter)
        segmenter_module = self._initialize_environment(segmenter)
        self._initialize_shared_memory(image)

        segmenter_module.segment(
            self._shared_image,
            self.config[segmenter]["default_parameters"],
            self._shared_segmentation,
        )
        return self._shared_segmentation

    def release_shared_memory(self):
        if self._shm_image:
            self._shm_image.unlink()
            self._shm_image.unregister()
            self._shm_image = None
        if self._shm_segmentation:
            self._shm_segmentation.unlink()
            self._shm_segmentation.unregister()
            self._shm_segmentation = None

    def exit(self):
        self.release_shared_memory()
        self.exit_environments()


if __name__ == "__main__":

    segmenter_manager = SegmenterManager()
    result = cast(
        np.ndarray,
        segmenter_manager.perform_segmentation(
            np.random.random((100, 100)), "StarDist", True
        ),
    )
    print(result.shape)
    segmenter_manager.exit()
