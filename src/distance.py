"""Monocular distance estimation with the pinhole camera model.

    Z = f_px * H_real / h_px

Similar triangles: an object of real height H at distance Z projects to
h = f * H / Z pixels on the image plane, so Z = f * H / h.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "camera.yaml"


@dataclass
class DistanceEstimator:
    focal_px: float  # focal length in pixels at ref_width
    ref_width: int
    objects: dict  # class name -> {"height_m": float, "use_max_side": bool}

    @classmethod
    def from_config(cls, path: Path = DEFAULT_CONFIG) -> "DistanceEstimator":
        cfg = yaml.safe_load(Path(path).read_text())
        cam = cfg["camera"]
        focal = cam.get("focal_px") or focal_from_fov(cam["ref_width"], cam["hfov_deg"])
        return cls(focal_px=float(focal), ref_width=int(cam["ref_width"]), objects=cfg["objects"])

    def focal_for(self, img_width: int) -> float:
        """Focal length in pixels scales linearly with image resolution."""
        return self.focal_px * img_width / self.ref_width

    def estimate(self, cls_name: str, box: tuple, img_width: int) -> float | None:
        """Distance in metres for an (x1, y1, x2, y2) pixel box, or None for unknown classes."""
        spec = self.objects.get(cls_name)
        if spec is None:
            return None
        x1, y1, x2, y2 = box
        h_px = y2 - y1
        if spec.get("use_max_side"):
            h_px = max(h_px, x2 - x1)
        if h_px <= 0:
            return None
        return self.focal_for(img_width) * spec["height_m"] / h_px


def focal_from_fov(width_px: int, hfov_deg: float) -> float:
    """f_px = (W / 2) / tan(HFOV / 2)."""
    return (width_px / 2) / math.tan(math.radians(hfov_deg) / 2)


def calibrate_focal(distance_m: float, h_px: float, height_m: float) -> float:
    """Recover f_px from one object at a measured distance: f = Z * h / H."""
    return distance_m * h_px / height_m


if __name__ == "__main__":
    est = DistanceEstimator.from_config()
    print(f"focal length: {est.focal_px:.1f} px at width {est.ref_width}")
    for name, h_px in [("cone", 70), ("stop_sign", 150), ("barrier", 200)]:
        z = est.estimate(name, (0, 0, h_px / 2, h_px), est.ref_width)
        print(f"{name:10s} box height {h_px:4d} px -> {z:.2f} m")
