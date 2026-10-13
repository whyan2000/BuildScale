"""Crop a perspective view out of a Mapillary equirectangular panorama,
aimed at a given compass bearing (degrees clockwise from true north).

Convention assumed (to be empirically verified against real images, see
verify_crops.py): the equirectangular image is north-referenced --
column x=0 (and x=width, wrapping) corresponds to true north (bearing 0),
and column increases with bearing going clockwise (east). This is the
standard convention for Mapillary spherical panoramas; compass_angle
(the camera's own heading at capture time) is NOT needed for this mapping
-- it's purely informational about which way the vehicle was pointed.
"""

import math

import numpy as np
from PIL import Image


def crop_perspective(equirect: Image.Image, yaw_deg: float, pitch_deg: float = 0.0,
                      hfov_deg: float = 90.0, out_w: int = 640, out_h: int = 640,
                      mode: str = "perspective", vfov_deg: float = None) -> Image.Image:
    """mode="perspective": standard gnomonic reprojection (rectilinear,
    straight lines stay straight). Breaks down near +-90 deg elevation --
    close, tall buildings need near-vertical look-up angles that blow up
    under this projection (a close 6-14 story building from ~8-10m away
    needs ~50-80deg elevation, which requires an enormous/singular hfov
    here). Kept for the original validated horizon-level use case.

    mode="cylindrical": elevation maps linearly to source row (no tangent,
    no pole singularity), so a much larger vfov_deg (e.g. 100-140) can be
    used to capture full building height at close range without the
    extreme stretching/blow-up perspective mode has near the zenith.
    Vertical lines curve slightly at the edges (standard architecture-
    photography tradeoff) but nothing near/far never inflates without
    bound. Azimuth is still linear in angle (same as the source
    equirect's own column mapping), so horizontal appearance is close to
    perspective for moderate hfov.
    """
    src = np.asarray(equirect.convert("RGB"), dtype=np.uint8)
    src_h, src_w = src.shape[:2]

    hfov = math.radians(hfov_deg)
    if vfov_deg is None:
        aspect = out_h / out_w
        vfov_deg = math.degrees(2 * math.atan(math.tan(hfov / 2) * aspect)) if mode == "perspective" else hfov_deg * (out_h / out_w)
    vfov = math.radians(vfov_deg)

    # Pixel grid in normalized camera coords, x right in [-1,1], y up in [-1,1].
    xs = np.linspace(-1, 1, out_w)
    ys = np.linspace(1, -1, out_h)  # row 0 = top = +1 (up)
    xg, yg = np.meshgrid(xs, ys)

    if mode == "cylindrical":
        azimuth = (yaw_deg + xg * math.degrees(hfov / 2)) % 360
        elevation = pitch_deg + yg * math.degrees(vfov / 2)
        elevation = np.clip(elevation, -89.9, 89.9)
    else:
        x_cam = xg * math.tan(hfov / 2)
        y_cam = yg * math.tan(vfov / 2)
        z_cam = np.ones_like(x_cam)

        # Camera looks down +z, x right, y up. Rotate by pitch around X axis,
        # then by yaw around Y (vertical) axis. Yaw increases clockwise (toward
        # +x / east) to match compass bearing convention.
        pitch = math.radians(pitch_deg)
        cp, sp = math.cos(pitch), math.sin(pitch)
        y1 = y_cam * cp - z_cam * sp
        z1 = y_cam * sp + z_cam * cp
        x1 = x_cam

        yaw = math.radians(yaw_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        x2 = x1 * cy + z1 * sy
        z2 = -x1 * sy + z1 * cy
        y2 = y1

        norm = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
        x2, y2, z2 = x2 / norm, y2 / norm, z2 / norm

        azimuth = (np.degrees(np.arctan2(x2, z2)) + 360) % 360  # 0=north, clockwise
        elevation = np.degrees(np.arcsin(np.clip(y2, -1, 1)))    # +90=up, -90=down

    src_x = (azimuth / 360.0) * src_w
    src_y = ((90.0 - elevation) / 180.0) * src_h

    src_x0 = np.clip(np.floor(src_x).astype(int), 0, src_w - 1)
    src_x1 = (src_x0 + 1) % src_w
    src_y0 = np.clip(np.floor(src_y).astype(int), 0, src_h - 1)
    src_y1 = np.clip(src_y0 + 1, 0, src_h - 1)

    fx = src_x - src_x0
    fy = src_y - src_y0

    def gather(yy, xx):
        return src[yy, xx].astype(np.float32)

    top = gather(src_y0, src_x0) * (1 - fx)[..., None] + gather(src_y0, src_x1) * fx[..., None]
    bot = gather(src_y1, src_x0) * (1 - fx)[..., None] + gather(src_y1, src_x1) * fx[..., None]
    out = top * (1 - fy)[..., None] + bot * fy[..., None]

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--yaw", type=float, required=True)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--vfov", type=float, default=None)
    p.add_argument("--mode", choices=["perspective", "cylindrical"], default="perspective")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    im = Image.open(args.image)
    crop = crop_perspective(im, args.yaw, args.pitch, args.hfov, mode=args.mode, vfov_deg=args.vfov)
    crop.save(args.out)
    print(f"wrote {args.out}")
