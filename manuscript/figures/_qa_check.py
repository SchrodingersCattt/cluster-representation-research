"""Programmatic figure-QA helpers (see FIGURE_QA.md for the binding rules).

The plot scripts import these helpers and call them right after they
finish drawing a panel.  Any rule violation is reported as a structured
:class:`QAError`; the calling script can then either ``raise`` (CI mode)
or print and continue (interactive iteration).

Design notes
------------
* Detection is image-based so it stays honest -- if an annotation is
  drawn over an atom, no amount of axes-fraction bookkeeping will hide
  it from a colour-match on the rendered RGBA buffer.
* Coordinates are in **image pixel space** throughout (matching what
  ``ax.imshow(image)`` puts on the data axes).  The plot scripts must
  therefore convert any display- or axes-fraction coordinates to image
  pixels before calling the helpers.
* The helpers are intentionally cheap (numpy + scipy.ndimage only) so
  they run on every render without slowing the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


@dataclass
class QAError:
    """A single rule violation, surfaced by the panel-level validator."""

    rule: str
    panel: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.panel}] {self.rule}: {self.detail}"


# ---------------------------------------------------------------------------
# Image / mask helpers
# ---------------------------------------------------------------------------


def build_atom_mask(
    image: np.ndarray,
    atom_color_hexes: Sequence[str] | None = None,
    *,
    color_tol: float = 70.0,
    dilate_px: int = 1,
) -> np.ndarray:
    """Return a boolean mask of every atom-coloured pixel in ``image``.

    If ``atom_color_hexes`` is provided we only flag pixels close to one
    of those colours (sphere-shaded balls have a wide gradient so the
    tolerance is generous).  If it is ``None`` we treat *any* non-near-
    white opaque pixel as part of an atom -- this is the right default for
    the panel-b "is the label sitting on top of any atom?" check.

    A small binary dilation (``dilate_px``) closes the anti-aliased seam
    around each sphere so a label that lands one pixel outside an atom's
    silhouette still counts as "touching the atom".
    """
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("image must be HxWx3 or HxWx4")
    rgb = image[..., :3].astype(np.int32)
    if image.shape[-1] == 4:
        alpha = image[..., 3]
    else:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)

    if atom_color_hexes:
        masks = []
        for hex_str in atom_color_hexes:
            hex_str = hex_str.lstrip("#")
            target = np.array(
                [int(hex_str[i:i + 2], 16) for i in (0, 2, 4)],
                dtype=np.int32,
            )
            diff = np.linalg.norm(rgb - target, axis=2)
            masks.append(diff < color_tol)
        mask = np.logical_or.reduce(masks)
    else:
        near_white = (rgb >= 235).all(axis=2)
        mask = (alpha > 200) & (~near_white)

    if dilate_px > 0:
        try:
            from scipy import ndimage as ndi
            mask = ndi.binary_dilation(mask, iterations=int(dilate_px))
        except Exception:  # scipy missing -- skip dilation
            pass
    return mask


def whitespace_distance_map(atom_mask: np.ndarray) -> np.ndarray:
    """Pixel-wise Euclidean distance to the nearest atom pixel (in px).

    The placement algorithm uses this as its primary score: candidates
    with a larger value sit deeper in the panel's whitespace.
    """
    try:
        from scipy import ndimage as ndi
    except Exception as exc:  # pragma: no cover - scipy is a hard dep here
        raise RuntimeError("scipy.ndimage is required for whitespace QA") from exc
    return ndi.distance_transform_edt(~atom_mask).astype(np.float32)


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------


def estimate_text_bbox_px(
    text: str,
    fontsize_pt: float,
    center_xy: tuple[float, float],
    *,
    dpi: float = 220.0,
    char_width_factor: float = 0.55,
    pad_factor: float = 0.06,
) -> tuple[float, float, float, float]:
    """Cheap matplotlib-free estimate of a horizontal text bbox in pixels.

    The plot scripts feed candidate label positions through this helper so
    they can reason about overlaps without forcing a render pass for every
    candidate.  ``char_width_factor=0.55`` is calibrated against
    matplotlib's default sans-serif: 'N1', 'Cl1' come out ~ correct width
    while not over-claiming whitespace.  ``pad_factor=0.06`` matches the
    actual ``bbox=round,pad=0.10`` patch (0.10 fontsize units of padding
    on each side, normalised by the text height).
    """
    h_pt = float(fontsize_pt)
    w_pt = h_pt * char_width_factor * max(len(text), 1)
    h_px = h_pt * dpi / 72.0
    w_px = w_pt * dpi / 72.0
    h_px *= 1.0 + pad_factor
    w_px *= 1.0 + pad_factor
    cx, cy = center_xy
    return (cx - w_px / 2.0, cy - h_px / 2.0, cx + w_px / 2.0, cy + h_px / 2.0)


def bbox_intersects_mask(bbox: tuple[float, float, float, float],
                         mask: np.ndarray) -> int:
    """Number of mask pixels the bbox overlaps; 0 means clean."""
    height, width = mask.shape
    x0 = int(max(0, np.floor(bbox[0])))
    y0 = int(max(0, np.floor(bbox[1])))
    x1 = int(min(width, np.ceil(bbox[2])))
    y1 = int(min(height, np.ceil(bbox[3])))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(mask[y0:y1, x0:x1].sum())


def bboxes_overlap(a: tuple[float, float, float, float],
                   b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def bbox_edge_distance(a: tuple[float, float, float, float],
                       b: tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


# ---------------------------------------------------------------------------
# Geometry checks
# ---------------------------------------------------------------------------


def arrow_is_uturn(atom_xy: tuple[float, float],
                   label_center: tuple[float, float],
                   cluster_center: tuple[float, float],
                   *,
                   min_cosine: float = -0.3) -> bool:
    """True if the label sits on the *far* inner side of the cluster.

    Encodes "arrows must not pull the atom back across the entire
    cluster".  We compare the cosine of ``(atom - cluster_centre)`` and
    ``(label_center - atom_xy)``: a label placed radially outward gives
    ~+1, a label placed strictly opposite gives -1.  ``min_cosine = -0.3``
    only flags the genuinely bad u-turns (~107 degrees off radial); small
    tangential or slightly-inward placements are allowed because the
    user explicitly objected to "must be 180-degree outward".
    """
    a = np.asarray(atom_xy, dtype=float) - np.asarray(cluster_center, dtype=float)
    b = np.asarray(label_center, dtype=float) - np.asarray(atom_xy, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return False
    cos_ab = float(np.dot(a, b) / (na * nb))
    return cos_ab < float(min_cosine)


def segments_intersect(p1, p2, p3, p4) -> bool:
    """Strict intersection between two open segments (shared endpoints OK)."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    p4 = np.asarray(p4, dtype=float)
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    s = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    eps = 1e-3
    return (eps < t < 1.0 - eps) and (eps < s < 1.0 - eps)


def segment_enters_bbox(p1: tuple[float, float],
                         p2: tuple[float, float],
                         bbox: tuple[float, float, float, float],
                         *,
                         margin: float = 0.0) -> bool:
    """True if the open segment p1-p2 crosses the rectangle ``bbox``.

    Used to detect "leader line passes through a sibling label's bbox"
    -- the case where another label's gray leader visually appears to be
    pointing at the wrong text.  The endpoints themselves are allowed to
    sit on or inside the bbox (the segment's own label endpoint is
    obviously inside its own bbox), so we only flag the case where the
    *open* segment intersects one of the four bbox edges.

    A small ``margin`` shrinks the bbox slightly before testing so we
    don't fire on segments that just graze a corner due to rounding.
    """
    x0, y0, x1, y1 = bbox
    if margin > 0.0:
        x0 += margin
        y0 += margin
        x1 -= margin
        y1 -= margin
    if x1 <= x0 or y1 <= y0:
        return False
    # Quick reject by axis-aligned extent.
    if max(p1[0], p2[0]) < x0 or min(p1[0], p2[0]) > x1:
        return False
    if max(p1[1], p2[1]) < y0 or min(p1[1], p2[1]) > y1:
        return False
    edges = (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    )
    for ea, eb in edges:
        if segments_intersect(p1, p2, ea, eb):
            return True
    return False


def segment_atom_hits(atom_xy: tuple[float, float],
                       label_xy: tuple[float, float],
                       atom_mask: np.ndarray,
                       *,
                       target_radius_px: float = 14.0,
                       label_radius_px: float = 10.0,
                       sample_step_px: float = 1.5) -> int:
    """Count atom-mask pixels the open leader segment passes through.

    The two endpoints (the target atom itself, and the label's bbox) are
    excluded by skipping samples within ``target_radius_px`` of
    ``atom_xy`` and ``label_radius_px`` of ``label_xy`` respectively.
    Anything else the segment touches is a *non-target* atom that the
    leader is visually pointing through -- which is what we want to
    block.
    """
    p0 = np.asarray(atom_xy, dtype=float)
    p1 = np.asarray(label_xy, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-6:
        return 0
    n = max(2, int(np.ceil(length / sample_step_px)))
    ts = np.linspace(0.0, 1.0, n + 1)
    pts = p0[None, :] * (1 - ts[:, None]) + p1[None, :] * ts[:, None]
    height, width = atom_mask.shape
    hits = 0
    for x, y in pts:
        if (np.hypot(x - p0[0], y - p0[1]) < target_radius_px
                or np.hypot(x - p1[0], y - p1[1]) < label_radius_px):
            continue
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        if 0 <= ix < width and 0 <= iy < height and atom_mask[iy, ix]:
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Top-level validators
# ---------------------------------------------------------------------------


def validate_panel_boundary(
    *,
    panel: str,
    figure_image: np.ndarray,
    rect_px: tuple[float, float, float, float],
    background_threshold: int = 235,
    max_data_pixels: int = 0,
) -> list[QAError]:
    """Verify a no-data zone (e.g. inter-panel gap) is empty.

    The rectangle ``rect_px = (x0, y0, x1, y1)`` is in figure-image pixel
    coordinates. Any pixel with all RGB channels >= ``background_threshold``
    counts as background; anything else counts as panel content that has
    invaded the safety band. If the count exceeds ``max_data_pixels`` the
    band is reported as failing.

    This is the programmatic enforcement of FIGURE_QA.md rule 1.6
    (panel-boundary clearance): every figure that uses a "row gap" or a
    legend safety band can call this helper to make sure no axis label,
    legend, or data point has crept across the boundary.
    """
    if figure_image.ndim != 3 or figure_image.shape[-1] not in (3, 4):
        raise ValueError("figure_image must be HxWx3 or HxWx4")
    height, width = figure_image.shape[:2]
    x0_f, y0_f, x1_f, y1_f = rect_px
    x0 = int(max(0, np.floor(x0_f)))
    y0 = int(max(0, np.floor(y0_f)))
    x1 = int(min(width, np.ceil(x1_f)))
    y1 = int(min(height, np.ceil(y1_f)))
    if x1 <= x0 or y1 <= y0:
        return []
    region = figure_image[y0:y1, x0:x1, :3].astype(np.int32)
    is_data = ~(region >= background_threshold).all(axis=2)
    n_invading = int(is_data.sum())
    if n_invading > int(max_data_pixels):
        return [QAError(
            rule="panel-boundary-clearance",
            panel=panel,
            detail=(f"safety band ({x0},{y0})-({x1},{y1}) has "
                    f"{n_invading} non-background pixels "
                    f"(allowed {int(max_data_pixels)})"),
        )]
    return []


@dataclass
class LabelRecord:
    """One annotation slated for QA validation."""

    label: str
    atom_xy_px: tuple[float, float]
    text_xy_px: tuple[float, float]
    fontsize_pt: float

    def bbox(self, dpi: float = 220.0) -> tuple[float, float, float, float]:
        return estimate_text_bbox_px(
            self.label, self.fontsize_pt, self.text_xy_px, dpi=dpi,
        )


def validate_atom_label_panel(
    *,
    panel: str,
    image: np.ndarray,
    records: Sequence[LabelRecord],
    cluster_center_px: tuple[float, float],
    atom_color_hexes: Sequence[str] | None = None,
    min_atom_clearance_px: float = 8.0,
    min_label_edge_distance_factor: float = 1.0,
    dpi: float = 220.0,
    color_tol: float = 70.0,
    dilate_px: int = 1,
    check_uturn: bool = True,
    max_arrow_factor: float | None = 4.0,
    target_atom_radius_px: float = 14.0,
) -> list[QAError]:
    """Run the full panel-b atom-label checklist; return all violations.

    Rules enforced (matching ``FIGURE_QA.md`` section 2.1):
        * label bbox must not intersect any atom pixel;
        * sibling label bboxes must not overlap and their edge-to-edge
          distance must be ``>= min_label_edge_distance_factor * text_height``;
        * the label-center distance to the nearest atom pixel must be
          ``>= min_atom_clearance_px``;
        * the arrow from label to atom must not u-turn;
        * arrow line segments must not cross each other;
        * an arrow line must not cross a sibling label's bbox;
        * an arrow line must not pass through any non-target atom;
        * arrow length must be ``<= max_arrow_factor * text_height_px``
          (set ``max_arrow_factor=None`` to disable the cap).
    """
    errors: list[QAError] = []

    atom_mask = build_atom_mask(
        image,
        atom_color_hexes=atom_color_hexes,
        color_tol=color_tol,
        dilate_px=dilate_px,
    )
    distmap = whitespace_distance_map(atom_mask)
    height, width = atom_mask.shape

    bboxes = [rec.bbox(dpi=dpi) for rec in records]

    # 1) bbox vs atom mask
    for rec, bb in zip(records, bboxes):
        hits = bbox_intersects_mask(bb, atom_mask)
        if hits > 0:
            errors.append(QAError(
                rule="label-bbox-on-atom",
                panel=panel,
                detail=(f"{rec.label}: bbox covers {hits} atom pixels at "
                        f"({rec.text_xy_px[0]:.0f},{rec.text_xy_px[1]:.0f})"),
            ))

    # 2) center-to-nearest-atom clearance
    for rec, bb in zip(records, bboxes):
        cx, cy = bbox_center(bb)
        ix = int(np.clip(round(cx), 0, width - 1))
        iy = int(np.clip(round(cy), 0, height - 1))
        d = float(distmap[iy, ix])
        if d < min_atom_clearance_px:
            errors.append(QAError(
                rule="label-too-close-to-atom",
                panel=panel,
                detail=(f"{rec.label}: bbox center {d:.1f}px from nearest atom "
                        f"(min {min_atom_clearance_px:.1f})"),
            ))

    # 3) pairwise sibling overlap and spacing
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            ri, rj = records[i], records[j]
            bi, bj = bboxes[i], bboxes[j]
            if bboxes_overlap(bi, bj):
                errors.append(QAError(
                    rule="label-bbox-overlap",
                    panel=panel,
                    detail=f"{ri.label} & {rj.label} bboxes overlap",
                ))
                continue
            edge_d = bbox_edge_distance(bi, bj)
            text_h_px = ri.fontsize_pt * dpi / 72.0
            min_edge = min_label_edge_distance_factor * text_h_px
            if edge_d < min_edge:
                errors.append(QAError(
                    rule="label-bbox-too-close",
                    panel=panel,
                    detail=(f"{ri.label} & {rj.label} edge-to-edge {edge_d:.1f}px "
                            f"< min {min_edge:.1f}"),
                ))

    # 4) arrow u-turn
    if check_uturn:
        for rec, bb in zip(records, bboxes):
            center = bbox_center(bb)
            if arrow_is_uturn(rec.atom_xy_px, center, cluster_center_px):
                errors.append(QAError(
                    rule="arrow-u-turn",
                    panel=panel,
                    detail=(f"{rec.label}: label sits inside the cluster "
                            f"(arrow points back toward centre)"),
                ))

    # 5) arrow crossings (only between distinct atoms)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            ri, rj = records[i], records[j]
            if ri.atom_xy_px == rj.atom_xy_px:
                continue
            ci = bbox_center(bboxes[i])
            cj = bbox_center(bboxes[j])
            if segments_intersect(ri.atom_xy_px, ci, rj.atom_xy_px, cj):
                errors.append(QAError(
                    rule="arrow-crossing",
                    panel=panel,
                    detail=f"{ri.label} arrow crosses {rj.label} arrow",
                ))

    # 6) leader line crossing into a sibling's bbox -- the case where one
    #    label's gray leader visually appears to be pointing at another
    #    label's text instead of its own atom.
    for i, ri in enumerate(records):
        leader = (ri.atom_xy_px, bbox_center(bboxes[i]))
        for j, rj in enumerate(records):
            if i == j:
                continue
            if segment_enters_bbox(leader[0], leader[1], bboxes[j], margin=1.0):
                errors.append(QAError(
                    rule="leader-crosses-label",
                    panel=panel,
                    detail=(f"{ri.label} leader passes through "
                            f"{rj.label}'s bbox"),
                ))

    # 7) leader line passing through a non-target atom -- the leader
    #    visually overlaps a different atom and the arrow becomes
    #    ambiguous about what it is pointing at.
    for rec, bb in zip(records, bboxes):
        bw = float(bb[2] - bb[0])
        bh = float(bb[3] - bb[1])
        label_skip_px = 0.5 * float(np.hypot(bw, bh)) + 2.0
        hits = segment_atom_hits(
            rec.atom_xy_px,
            bbox_center(bb),
            atom_mask,
            target_radius_px=target_atom_radius_px,
            label_radius_px=label_skip_px,
        )
        if hits > 0:
            errors.append(QAError(
                rule="leader-through-atom",
                panel=panel,
                detail=(f"{rec.label}: leader passes through {hits} "
                        f"non-target atom px"),
            ))

    # 8) arrow length cap -- enforces "label adjacent to atom" rather
    #    than "label parked in deep whitespace".
    if max_arrow_factor is not None:
        for rec, bb in zip(records, bboxes):
            text_h_px = rec.fontsize_pt * dpi / 72.0
            cx, cy = bbox_center(bb)
            arrow_len = float(np.hypot(
                cx - rec.atom_xy_px[0],
                cy - rec.atom_xy_px[1],
            ))
            max_arrow = float(max_arrow_factor) * text_h_px
            if arrow_len > max_arrow:
                errors.append(QAError(
                    rule="arrow-too-long",
                    panel=panel,
                    detail=(f"{rec.label}: arrow {arrow_len:.0f}px > "
                            f"max {max_arrow:.0f}px "
                            f"(={max_arrow_factor:.1f} x text height)"),
                ))

    return errors


# ---------------------------------------------------------------------------
# Whitespace-aware placement (used by plot_fig5.panel_b)
# ---------------------------------------------------------------------------


def whitespace_label_placement(
    *,
    image: np.ndarray,
    atoms: Sequence[tuple[str, tuple[float, float]]],
    cluster_center_px: tuple[float, float],
    fontsize_pt: float,
    forbidden_rects_px: Sequence[tuple[float, float, float, float]] = (),
    atom_color_hexes: Sequence[str] | None = None,
    min_atom_clearance_px: float = 6.0,
    min_label_edge_distance_factor: float = 0.4,
    dpi: float = 220.0,
    radius_search_factors: Sequence[float] = (
        0.10, 0.14, 0.18, 0.23, 0.28, 0.34, 0.42, 0.52,
    ),
    angle_step_deg: float = 9.0,
    color_tol: float = 70.0,
    max_arrow_factor: float = 4.0,
    target_atom_radius_px: float = 14.0,
) -> list[LabelRecord]:
    """Greedy "label adjacent to atom" placement.

    For each labelled atom (placed in "most peripheral first" order) we
    sample candidate positions on a polar grid centred at ``cluster_center``
    along ``atoms[i].radial direction``.  Candidates are filtered by a
    set of hard rules and then ranked by **shortest arrow first**; the
    whitespace clearance acts only as a hard floor (``>=
    min_atom_clearance_px``) rather than a primary objective.

    Hard rules (all matching the validator in
    :func:`validate_atom_label_panel`):

    * bbox does not intersect ``atom_mask``;
    * bbox does not intersect any forbidden rectangle;
    * bbox does not overlap any previously placed bbox;
    * sibling-bbox edge distance >= ``min_label_edge_distance_factor *
      text_height``;
    * arrow does not u-turn (cosine threshold tunable inside
      :func:`arrow_is_uturn`);
    * candidate arrow does not cross any previously placed arrow;
    * candidate arrow does not enter any previously placed bbox;
    * candidate arrow does not pass through a non-target atom;
    * arrow length <= ``max_arrow_factor * text_height_px``.

    The default ``radius_search_factors`` are tight (0.10-0.52 x the
    smaller axes-box dimension) so the algorithm always tries
    "immediately adjacent" positions before reaching for far whitespace.
    """
    height, width = image.shape[:2]
    # Match the validator's dilation so placement and validate
    # never disagree by a single anti-aliased edge pixel.
    atom_mask = build_atom_mask(
        image,
        atom_color_hexes=atom_color_hexes,
        color_tol=color_tol,
        dilate_px=1,
    )
    distmap = whitespace_distance_map(atom_mask)
    cx_c, cy_c = cluster_center_px
    text_h_px = fontsize_pt * dpi / 72.0
    min_edge = min_label_edge_distance_factor * text_h_px
    max_arrow_px = float(max_arrow_factor) * text_h_px

    base_radius = max(60.0, 0.30 * min(width, height))

    placed: list[LabelRecord] = []
    placed_bboxes: list[tuple[float, float, float, float]] = []
    placed_arrows: list[tuple[tuple[float, float], tuple[float, float]]] = []

    # Order: most peripheral atoms first (they have the most freedom).
    order = sorted(
        range(len(atoms)),
        key=lambda i: -float(np.hypot(
            atoms[i][1][0] - cx_c, atoms[i][1][1] - cy_c)),
    )

    for i in order:
        label, atom_xy = atoms[i]
        ax_, ay_ = float(atom_xy[0]), float(atom_xy[1])
        radial = np.array([ax_ - cx_c, ay_ - cy_c], dtype=float)
        norm = float(np.linalg.norm(radial))
        if norm < 1e-3:
            radial = np.array([0.0, -1.0])
            norm = 1.0
        radial /= norm

        candidates: list[tuple[float, tuple[float, float]]] = []
        # Track per-rule rejection counts so the error message can point
        # at the actual blocker if no slot is found.
        reject_counts = {
            "atom_mask": 0,
            "forbidden": 0,
            "sibling_bbox_overlap": 0,
            "sibling_bbox_too_close": 0,
            "uturn": 0,
            "arrow_crossing": 0,
            "leader_into_sibling_bbox": 0,
            "leader_through_atom": 0,
            "arrow_too_long": 0,
            "low_clearance": 0,
        }
        for r_mul in radius_search_factors:
            r = base_radius * r_mul
            for step in range(0, int(360.0 / angle_step_deg) + 1):
                # alternate sides so we explore symmetrically
                sign = 1 if step % 2 == 0 else -1
                ang = np.deg2rad(angle_step_deg * (step // 2) * sign)
                cos_a, sin_a = np.cos(ang), np.sin(ang)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rot_dir = rot @ radial
                tx = ax_ + rot_dir[0] * r
                ty = ay_ + rot_dir[1] * r
                tx = float(np.clip(tx, 12.0, width - 12.0))
                ty = float(np.clip(ty, 12.0, height - 12.0))
                bb = estimate_text_bbox_px(
                    label, fontsize_pt, (tx, ty), dpi=dpi,
                )

                # Hard filters first - cheap rejections.
                if bbox_intersects_mask(bb, atom_mask) > 0:
                    reject_counts["atom_mask"] += 1
                    continue
                hit_forbidden = False
                for fx0, fy0, fx1, fy1 in forbidden_rects_px:
                    if bboxes_overlap(bb, (fx0, fy0, fx1, fy1)):
                        hit_forbidden = True
                        break
                if hit_forbidden:
                    reject_counts["forbidden"] += 1
                    continue
                if any(bboxes_overlap(bb, pb) for pb in placed_bboxes):
                    reject_counts["sibling_bbox_overlap"] += 1
                    continue
                if any(bbox_edge_distance(bb, pb) < min_edge
                       for pb in placed_bboxes):
                    reject_counts["sibling_bbox_too_close"] += 1
                    continue
                if arrow_is_uturn((ax_, ay_), bbox_center(bb),
                                  (cx_c, cy_c)):
                    reject_counts["uturn"] += 1
                    continue

                cand_center = bbox_center(bb)

                # Reject arrow crossings against previously placed arrows.
                arrow_xings = False
                for pa, pb_ctr in placed_arrows:
                    if segments_intersect((ax_, ay_), cand_center, pa, pb_ctr):
                        arrow_xings = True
                        break
                if arrow_xings:
                    reject_counts["arrow_crossing"] += 1
                    continue

                # Candidate arrow must not enter any previously placed
                # label's bbox (else the leader visually looks as if it
                # is pointing at the wrong text).
                if any(segment_enters_bbox(
                        (ax_, ay_), cand_center, pb, margin=1.0)
                       for pb in placed_bboxes):
                    reject_counts["leader_into_sibling_bbox"] += 1
                    continue

                # Candidate arrow must not pass through a non-target
                # atom -- the leader would otherwise look like it is
                # pointing at the wrong sphere.
                bw = float(bb[2] - bb[0])
                bh = float(bb[3] - bb[1])
                label_skip_px = 0.5 * float(np.hypot(bw, bh)) + 2.0
                if segment_atom_hits(
                        (ax_, ay_), cand_center, atom_mask,
                        target_radius_px=target_atom_radius_px,
                        label_radius_px=label_skip_px) > 0:
                    reject_counts["leader_through_atom"] += 1
                    continue

                # Arrow length cap -- "label adjacent to atom".
                arrow_len = float(np.hypot(tx - ax_, ty - ay_))
                if arrow_len > max_arrow_px:
                    reject_counts["arrow_too_long"] += 1
                    continue

                # Whitespace clearance acts only as a hard floor.
                cx, cy = cand_center
                ix = int(np.clip(round(cx), 0, width - 1))
                iy = int(np.clip(round(cy), 0, height - 1))
                clearance = float(distmap[iy, ix])
                if clearance < min_atom_clearance_px:
                    reject_counts["low_clearance"] += 1
                    continue

                # Score: shortest arrow wins; clearance breaks ties so
                # equally-short candidates prefer the airier one.
                score = -arrow_len + 0.02 * (clearance - min_atom_clearance_px)
                candidates.append((score, (tx, ty)))

        if not candidates:
            top_rejects = sorted(
                reject_counts.items(), key=lambda kv: -kv[1]
            )
            top_str = ", ".join(f"{k}={v}" for k, v in top_rejects if v)
            raise RuntimeError(
                f"whitespace_label_placement: no valid slot for label "
                f"{label!r} at atom ({ax_:.0f},{ay_:.0f}); "
                f"the panel is over-packed -- enlarge the tile, drop "
                f"labels, raise max_arrow_factor, or relax "
                f"min_atom_clearance_px. "
                f"Rejection breakdown: {top_str}"
            )

        # Pick the best by score.
        candidates.sort(key=lambda t: -t[0])
        _, (tx, ty) = candidates[0]
        rec = LabelRecord(
            label=label,
            atom_xy_px=(ax_, ay_),
            text_xy_px=(tx, ty),
            fontsize_pt=fontsize_pt,
        )
        placed.append(rec)
        bb = rec.bbox(dpi=dpi)
        placed_bboxes.append(bb)
        placed_arrows.append(((ax_, ay_), bbox_center(bb)))

    # Restore original input order so the caller's label numbering is
    # preserved when emitting annotations.
    by_label = {rec.label: rec for rec in placed}
    return [by_label[label] for label, _ in atoms]
