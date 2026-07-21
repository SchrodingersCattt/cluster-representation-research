from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import plotly.graph_objects as go


def _normalize(vec: Iterable[float], fallback: Iterable[float]) -> np.ndarray:
    arr = np.array(list(vec), dtype=float)
    if arr.shape != (3,) or np.linalg.norm(arr) < 1e-8:
        arr = np.array(list(fallback), dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _plotly_camera_from_scene(scene: dict) -> dict:
    eye = _normalize(scene.get("view_direction", [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]) * 1.8
    up = _normalize(scene.get("up", [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
    return {
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": float(up[0]), "y": float(up[1]), "z": float(up[2])},
    }


def _visible_atoms(scene: dict, style: dict):
    atoms = scene["draw_atoms"]
    if style.get("show_minor_only", False):
        atoms = [atom for atom in atoms if atom["is_minor"]]
    return atoms or scene["draw_atoms"]


def _scene_ranges(scene: dict, style: dict, topology_data: dict | None = None):
    """Compute ``[xr, yr, zr]`` axis ranges for the Plotly scene.

    A scene-level ``viewport`` override (set by :func:`uniform_viewport`) wins
    unconditionally; this is how caller code pins several scenes to a shared
    world cube so they render at identical screen scale.

    Otherwise the bounds are inflated by each atom's **visual radius** (rather
    than a blanket 18 % fractional pad) so spheres — especially large halides
    like Cl, Br, I — are never clipped at the panel edge. Unit-cell corners and
    topology markers expand the box but do not contribute radii.
    """
    override = scene.get("viewport")
    if override:
        return [
            [float(override["x"][0]), float(override["x"][1])],
            [float(override["y"][0]), float(override["y"][1])],
            [float(override["z"][0]), float(override["z"][1])],
        ]

    atoms = _visible_atoms(scene, style)

    atom_mins = None
    atom_maxs = None
    if atoms:
        carts = np.array([atom["cart"] for atom in atoms], dtype=float)
        radii = np.array(
            [_effective_atom_radius(atom, style) for atom in atoms],
            dtype=float,
        )
        atom_mins = (carts - radii[:, None]).min(axis=0)
        atom_maxs = (carts + radii[:, None]).max(axis=0)

    extras = []
    if style.get("show_unit_cell", False):
        a = np.array(scene["M"][:, 0], dtype=float)
        b = np.array(scene["M"][:, 1], dtype=float)
        c = np.array(scene["M"][:, 2], dtype=float)
        for corner in (
            np.zeros(3, dtype=float),
            a, b, c, a + b, a + c, b + c, a + b + c,
        ):
            extras.append(corner)
    if topology_data:
        center = topology_data.get("center_coords")
        if center is not None:
            extras.append(np.array(center, dtype=float))
        for point in topology_data.get("shell_coords") or []:
            extras.append(np.array(point, dtype=float))
    if extras:
        extras_arr = np.array(extras, dtype=float)
        extras_min = extras_arr.min(axis=0)
        extras_max = extras_arr.max(axis=0)
        if atom_mins is None:
            atom_mins, atom_maxs = extras_min, extras_max
        else:
            atom_mins = np.minimum(atom_mins, extras_min)
            atom_maxs = np.maximum(atom_maxs, extras_max)

    if atom_mins is None:
        return [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

    span = np.maximum(atom_maxs - atom_mins, 0.8)
    # Small breathing-room pad layered on top of radius-aware bounds.
    pad = np.maximum(span * 0.06, 0.25)
    mins = atom_mins - pad
    maxs = atom_maxs + pad
    return [
        [float(mins[0]), float(maxs[0])],
        [float(mins[1]), float(maxs[1])],
        [float(mins[2]), float(maxs[2])],
    ]


def uniform_viewport(scenes, *, style=None, padding=0.0):
    """Stamp a shared world-cube viewport on each scene so ``build_figure``
    renders them at identical screen scale.

    For every scene the viewport becomes a cube centred on that scene's own
    atom-bounding centroid. The cube side length equals the largest
    radius-aware axis-aligned span across **all** input scenes (+ ``padding``
    in Å on every side). Callers that later draw the scenes in a grid get
    panels with a single physical length scale — no more "small molecule
    ballooning to fill the panel while the big one shrinks to pinheads".

    The ``viewport`` key is written in-place on each scene dict. Subsequent
    calls to :func:`_scene_ranges` (and therefore :func:`build_figure`) honour
    it and skip their own bounds calculation.

    Parameters
    ----------
    scenes
        Iterable of scene dicts (as returned by ``build_scene_from_cif`` /
        ``build_scene_from_atoms``).
    style
        Optional style dict used to infer ``atom_scale``. When omitted, each
        scene's own ``scene["style"]`` is consulted with a default of 1.0.
    padding
        Extra padding in Å added symmetrically to every face of the cube.

    Returns
    -------
    list[dict]
        The stamped ``viewport`` dicts, one per scene, in the order the
        scenes were provided.
    """
    scenes = list(scenes)
    if not scenes:
        return []

    radius_spans = []
    centroids = []
    for scene in scenes:
        scn_style = style if style is not None else scene.get("style") or {}
        atoms = scene.get("draw_atoms") or []
        if not atoms:
            radius_spans.append(1.0)
            centroids.append(np.zeros(3, dtype=float))
            continue
        carts = np.array([atom["cart"] for atom in atoms], dtype=float)
        # Use the *displayed* half-extent of each atom (sphere OR ORTEP
        # ellipsoid max principal axis) so ORTEP mode doesn't clip the
        # outer vertices of thermal ellipsoids that extend beyond the
        # covalent-sphere radius.
        radii = np.array(
            [_effective_atom_radius(atom, scn_style) for atom in atoms],
            dtype=float,
        )
        mins = (carts - radii[:, None]).min(axis=0)
        maxs = (carts + radii[:, None]).max(axis=0)
        radius_spans.append(float((maxs - mins).max()))
        centroids.append(0.5 * (mins + maxs))

    half = 0.5 * max(radius_spans) + float(padding)
    viewports = []
    for scene, center in zip(scenes, centroids):
        viewport = {
            "x": [float(center[0] - half), float(center[0] + half)],
            "y": [float(center[1] - half), float(center[1] + half)],
            "z": [float(center[2] - half), float(center[2] + half)],
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "half_span": float(half),
        }
        scene["viewport"] = viewport
        viewports.append(viewport)
    return viewports


def _style_bool(style: dict, key: str, default: bool = False) -> bool:
    return bool(style.get(key, default))


def style_from_controls(atom_scale, bond_radius, minor_opacity, axis_scale, options) -> dict:
    options = set(options or [])
    return {
        "atom_scale": float(atom_scale),
        "bond_radius": float(bond_radius),
        "minor_opacity": float(minor_opacity),
        "axis_scale": float(axis_scale),
        "show_labels": "labels" in options,
        "show_axes": "axes" in options,
        "show_minor_only": "minor_only" in options,
        "minor_wireframe": "minor_wireframe" in options,
        "show_hydrogen": "hydrogens" in options,
        "show_unit_cell": "unit_cell_box" in options,
        "fast_rendering": "fast_rendering" in options,
        "topology_enabled": "topology" in options,
    }


def _unit_sphere(lat_steps: int = 9, lon_steps: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    vertices = []
    for lat_idx in range(lat_steps + 1):
        theta = math.pi * lat_idx / lat_steps
        for lon_idx in range(lon_steps):
            phi = 2.0 * math.pi * lon_idx / lon_steps
            vertices.append(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ]
            )
    triangles = []
    for lat_idx in range(lat_steps):
        for lon_idx in range(lon_steps):
            next_lon = (lon_idx + 1) % lon_steps
            a = lat_idx * lon_steps + lon_idx
            b = lat_idx * lon_steps + next_lon
            c = (lat_idx + 1) * lon_steps + lon_idx
            d = (lat_idx + 1) * lon_steps + next_lon
            triangles.append([a, c, b])
            triangles.append([b, c, d])
    return np.array(vertices, dtype=float), np.array(triangles, dtype=int)


def _append_mesh(mesh: dict, vertices: np.ndarray, triangles: np.ndarray):
    base = len(mesh["x"])
    mesh["x"].extend(vertices[:, 0].tolist())
    mesh["y"].extend(vertices[:, 1].tolist())
    mesh["z"].extend(vertices[:, 2].tolist())
    mesh["i"].extend((triangles[:, 0] + base).tolist())
    mesh["j"].extend((triangles[:, 1] + base).tolist())
    mesh["k"].extend((triangles[:, 2] + base).tolist())


def _sphere_mesh(center: Iterable[float], radius: float, lat_steps: int = 9, lon_steps: int = 14):
    unit_vertices, unit_triangles = _unit_sphere(lat_steps=lat_steps, lon_steps=lon_steps)
    center = np.array(center, dtype=float)
    vertices = unit_vertices * float(radius) + center[None, :]
    return vertices, unit_triangles


# Chi-square inverse CDF values for 3 degrees of freedom (ORTEP probability
# surfaces). The displacement-probability ellipsoid at level ``p`` satisfies
# x^T U^{-1} x = c^2 where ``c^2`` is :data:`_CHI2_INV_3DOF[p]`. Standard
# choices in crystallography are 0.30, 0.50, 0.90 — 0.50 is the de-facto
# default in Mercury / SHELXL ORTEP figures and the value the legacy
# pipeline used.
_CHI2_INV_3DOF: Dict[float, float] = {
    0.30: 1.4237,
    0.50: 2.3660,
    0.70: 3.6648,
    0.90: 6.2514,
    0.99: 11.3449,
}


def _ortep_mesh(
    center: Iterable[float],
    U_cart: np.ndarray,
    *,
    probability: float = 0.50,
    iso_radius_floor: float = 0.05,
    iso_radius_ceil: float = 0.40,
    lat_steps: int = 12,
    lon_steps: int = 18,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an ORTEP-style anisotropic displacement ellipsoid.

    The ellipsoid surface encloses ``probability`` of the trivariate-normal
    nuclear-density distribution implied by the Cartesian-frame mean-square
    displacement matrix ``U_cart`` (units Å²). Its principal half-axes are
    ``sqrt(c² · λ_k)`` where ``λ_k`` are the eigenvalues of ``U_cart`` and
    ``c² = χ²_inv(3, probability)`` (see :data:`_CHI2_INV_3DOF`).

    Negative or NaN eigenvalues — which crop up for ill-refined ADPs — are
    clamped to a small positive floor so the ellipsoid stays renderable
    rather than collapsing into a degenerate disc/line. Likewise the
    half-axes are clipped to ``iso_radius_ceil`` so a single pathological
    atom cannot blow out the viewport.
    """
    unit_vertices, unit_triangles = _unit_sphere(lat_steps=lat_steps, lon_steps=lon_steps)
    center = np.array(center, dtype=float)
    U = np.asarray(U_cart, dtype=float)
    if U.shape != (3, 3) or not np.all(np.isfinite(U)):
        radius = max(iso_radius_floor, 0.18)
        return unit_vertices * radius + center[None, :], unit_triangles
    # Symmetrise to suppress round-off asymmetry.
    U = 0.5 * (U + U.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(U)
    except np.linalg.LinAlgError:
        radius = max(iso_radius_floor, 0.18)
        return unit_vertices * radius + center[None, :], unit_triangles
    eigvals = np.clip(eigvals, iso_radius_floor**2, np.inf)
    c2 = _CHI2_INV_3DOF.get(round(float(probability), 2), _CHI2_INV_3DOF[0.50])
    half_axes = np.sqrt(c2 * eigvals)
    half_axes = np.clip(half_axes, iso_radius_floor, iso_radius_ceil)
    # Transform the unit sphere by R · diag(half_axes), then translate.
    M = eigvecs @ np.diag(half_axes)
    vertices = unit_vertices @ M.T + center[None, :]
    return vertices, unit_triangles


def _iso_ellipsoid_radius(uiso: float, probability: float = 0.50,
                           floor: float = 0.10, ceil: float = 0.40) -> float:
    """Equivalent isotropic ellipsoid radius for atoms with only Uiso."""
    if uiso is None or not np.isfinite(uiso) or uiso <= 0.0:
        return max(floor, 0.18)
    c2 = _CHI2_INV_3DOF.get(round(float(probability), 2), _CHI2_INV_3DOF[0.50])
    return float(np.clip(math.sqrt(c2 * float(uiso)), floor, ceil))


def _effective_atom_radius(atom: dict, style: dict) -> float:
    """Return the largest half-extent of the atom's drawn primitive (Å).

    Covers all three render-mode branches in :func:`_atom_mesh_traces` so
    callers that need a bounding radius (viewport, camera-framing) never
    under-estimate and clip ORTEP ellipsoids:

    * ORTEP + anisotropic U → ``sqrt(c² · λ_max)`` with ``λ_max`` the
      largest eigenvalue of ``U``.
    * ORTEP + Uiso only → the isotropic-equivalent radius.
    * Everything else → ``atom_radius * atom_scale`` (the classical sphere).

    Hydrogens in ORTEP mode are forced to ``ortep_hydrogen_radius`` (a
    small fixed sphere) rather than their inflated Uiso ellipsoid —
    SHELX refinement typically pins H Uiso at 1.2× its parent C/N/O Uiso,
    which would otherwise render H larger than the heavy atom it hangs
    off.
    """
    render_mode = str(style.get("atom_render", "sphere")).lower()
    use_ortep = render_mode == "ortep"
    probability = float(style.get("ortep_probability", 0.50))
    atom_scale = float(style.get("atom_scale", 1.0))
    elem = str(atom.get("elem", "")).strip()
    is_hydrogen = elem == "H"

    if use_ortep and is_hydrogen:
        radius = float(style.get("ortep_hydrogen_radius", 0.20)) * atom_scale
    elif use_ortep and atom.get("U") is not None:
        U = np.asarray(atom["U"], dtype=float)
        if U.shape == (3, 3) and np.all(np.isfinite(U)):
            U = 0.5 * (U + U.T) * (atom_scale ** 2)
            try:
                eigvals = np.linalg.eigvalsh(U)
                lam_max = float(np.clip(eigvals.max(), 1.0e-4, None))
                c2 = _CHI2_INV_3DOF.get(round(probability, 2), _CHI2_INV_3DOF[0.50])
                radius = math.sqrt(c2 * lam_max)
            except np.linalg.LinAlgError:
                radius = float(atom.get("atom_radius", 0.18)) * atom_scale
        else:
            radius = float(atom.get("atom_radius", 0.18)) * atom_scale
    elif use_ortep and atom.get("uiso") is not None and float(atom["uiso"]) > 0.0:
        radius = _iso_ellipsoid_radius(float(atom["uiso"]), probability=probability) * atom_scale
    else:
        radius = float(atom.get("atom_radius", 0.18)) * atom_scale
    return max(float(radius), 0.05)


def _cylinder_mesh(p0: Iterable[float], p1: Iterable[float], radius: float, sides: int = 8):
    start = np.array(p0, dtype=float)
    end = np.array(p1, dtype=float)
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    axis /= length
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(axis, ref)) > 0.92:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    ring0 = []
    ring1 = []
    for idx in range(sides):
        ang = 2.0 * math.pi * idx / sides
        offset = math.cos(ang) * u * radius + math.sin(ang) * v * radius
        ring0.append(start + offset)
        ring1.append(end + offset)
    vertices = np.array(ring0 + ring1 + [start, end], dtype=float)
    cap0 = len(vertices) - 2
    cap1 = len(vertices) - 1
    triangles = []
    for idx in range(sides):
        nxt = (idx + 1) % sides
        a0 = idx
        a1 = nxt
        b0 = idx + sides
        b1 = nxt + sides
        triangles.extend([[a0, b0, a1], [a1, b0, b1], [cap0, a1, a0], [cap1, b0, b1]])
    return vertices, np.array(triangles, dtype=int)


def _atom_selection_trace(scene: dict, style: dict):
    xs, ys, zs, sizes, labels, customdata = [], [], [], [], [], []
    for idx, atom in enumerate(scene["draw_atoms"]):
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        xs.append(float(atom["cart"][0]))
        ys.append(float(atom["cart"][1]))
        zs.append(float(atom["cart"][2]))
        sizes.append(max(6.0, 48.0 * atom["atom_radius"] * float(style["atom_scale"])))
        labels.append(atom["label"])
        customdata.append([idx, atom["label"], atom["elem"], int(atom["is_minor"])])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=sizes, color="rgba(0,0,0,0)", opacity=0.02),
        customdata=customdata,
        hovertemplate="%{customdata[1]} (%{customdata[2]})<extra></extra>",
        showlegend=False,
        name="atom-selection",
    )


def _bond_segments(scene: dict, style: dict):
    for bond in scene["bonds"]:
        if style.get("show_minor_only", False) and not bond["is_minor"]:
            continue
        start = np.array(bond["start"], dtype=float)
        end = np.array(bond["end"], dtype=float)
        mid = (start + end) / 2.0
        yield bond["color_i"], bond["is_minor"], start, mid
        yield bond["color_j"], bond["is_minor"], mid, end


def _bond_mesh_traces(scene: dict, style: dict):
    groups: Dict[Tuple[str, bool], dict] = {}
    radius = max(0.04, float(style["bond_radius"]))
    for color, is_minor, start, end in _bond_segments(scene, style):
        key = (color, is_minor)
        groups.setdefault(key, {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []})
        vertices, triangles = _cylinder_mesh(
            start,
            end,
            radius * (float(style["minor_bond_scale"]) if is_minor else 1.0),
            sides=7,
        )
        if len(vertices):
            _append_mesh(groups[key], vertices, triangles)

    traces = []
    for (color, is_minor), payload in groups.items():
        traces.append(
            go.Mesh3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                i=payload["i"],
                j=payload["j"],
                k=payload["k"],
                color=color,
                opacity=float(style["minor_opacity"]) if is_minor else 1.0,
                hoverinfo="skip",
                showlegend=False,
                flatshading=False,
            )
        )
    return traces


def _atom_mesh_traces(scene: dict, style: dict):
    """Build per-atom Mesh3d traces.

    Two rendering modes are supported:

    * ``style["atom_render"] == "sphere"`` (default) — isotropic spheres
      sized by the atom's covalent/display radius scaled by
      ``style["atom_scale"]``. This is the original ball-and-stick look.
    * ``style["atom_render"] == "ortep"`` — ORTEP-style anisotropic
      thermal ellipsoids at the probability level
      ``style["ortep_probability"]`` (default 0.50). Atoms with anisotropic
      ADPs (``atom["U"]`` 3×3 in Å²) get a true thermal ellipsoid; atoms
      with only ``Uiso`` get an isotropic equivalent sized from
      ``Uiso``; atoms with neither fall back to the sphere radius. The
      resulting figure is publication-quality crystallography rather than
      generic ball-and-stick.

    Disorder opacity floor: previously the minor-site opacity was clamped
    to ``max(0.48, minor_opacity)`` to keep faded atoms visible. That
    floor washed out the major-vs-minor contrast, so it's relaxed to
    ``style["minor_opacity_floor"]`` (default 0.18) which lets the user
    push minor sites to near-transparent for true ORTEP-style disorder.
    """
    render_mode = str(style.get("atom_render", "sphere")).lower()
    use_ortep = render_mode == "ortep"
    probability = float(style.get("ortep_probability", 0.50))
    minor_opacity_floor = float(style.get("minor_opacity_floor", 0.18))
    sphere_lat = int(style.get("sphere_lat_steps", 8))
    sphere_lon = int(style.get("sphere_lon_steps", 12))
    ortep_lat = int(style.get("ortep_lat_steps", 18))
    ortep_lon = int(style.get("ortep_lon_steps", 28))
    h_radius = float(style.get("ortep_hydrogen_radius", 0.20))
    # Plotly's 3D Mesh3d lighting defaults (ambient 0.8 / diffuse 0.8 /
    # specular 0.05 / roughness 0.5) produce a washed-out flat look with
    # harsh terminator bands on ellipsoids. Tune toward a softer matte
    # surface: lower ambient so shape reads, moderate diffuse, a touch
    # of specular for "crystallographic polish", high roughness to kill
    # the hotspot, and two balanced light positions.
    lighting = dict(
        ambient=float(style.get("mesh_ambient", 0.52)),
        diffuse=float(style.get("mesh_diffuse", 0.90)),
        specular=float(style.get("mesh_specular", 0.18)),
        roughness=float(style.get("mesh_roughness", 0.82)),
        fresnel=float(style.get("mesh_fresnel", 0.06)),
    )
    lightposition = dict(
        x=float(style.get("mesh_lightposition_x", 140)),
        y=float(style.get("mesh_lightposition_y", 220)),
        z=float(style.get("mesh_lightposition_z", 260)),
    )

    groups: Dict[Tuple[str, bool], dict] = {}
    for atom in scene["draw_atoms"]:
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        key = (atom["color"], atom["is_minor"])
        groups.setdefault(key, {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []})

        elem = str(atom.get("elem", "")).strip()
        is_hydrogen = elem == "H"
        u_cart = atom.get("U")
        uiso = atom.get("uiso")
        atom_scale = float(style["atom_scale"])
        # Hydrogens never get thermal ellipsoids: SHELX rides H Uiso at
        # 1.2× its parent, so an ORTEP-sized H would dwarf the heavy
        # atom it hangs off. Force a small fixed display sphere instead.
        if use_ortep and is_hydrogen:
            radius = h_radius * atom_scale
            if atom["is_minor"]:
                radius *= 1.08
            vertices, triangles = _sphere_mesh(atom["cart"], radius,
                                                lat_steps=ortep_lat, lon_steps=ortep_lon)
        elif use_ortep and u_cart is not None and np.asarray(u_cart, dtype=float).shape == (3, 3) \
                and np.all(np.isfinite(np.asarray(u_cart, dtype=float))):
            scaled_U = np.asarray(u_cart, dtype=float) * (atom_scale ** 2)
            vertices, triangles = _ortep_mesh(
                atom["cart"], scaled_U,
                probability=probability,
                lat_steps=ortep_lat, lon_steps=ortep_lon,
            )
        elif use_ortep and uiso is not None and float(uiso) > 0.0:
            radius = _iso_ellipsoid_radius(float(uiso), probability=probability) * atom_scale
            if atom["is_minor"]:
                radius *= 1.12
            vertices, triangles = _sphere_mesh(atom["cart"], radius,
                                                lat_steps=ortep_lat, lon_steps=ortep_lon)
        else:
            radius = float(atom["atom_radius"]) * atom_scale
            if atom["is_minor"]:
                radius *= 1.12
            vertices, triangles = _sphere_mesh(atom["cart"], radius,
                                                lat_steps=sphere_lat, lon_steps=sphere_lon)
        _append_mesh(groups[key], vertices, triangles)

    traces = []
    major_opacity = float(style.get("major_opacity", 1.0))
    minor_opacity = float(style.get("minor_opacity", 0.35))
    for (color, is_minor), payload in groups.items():
        opacity = max(minor_opacity_floor, minor_opacity) if is_minor else major_opacity
        traces.append(
            go.Mesh3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                i=payload["i"],
                j=payload["j"],
                k=payload["k"],
                color=color,
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
                flatshading=False,
                lighting=lighting,
                lightposition=lightposition,
            )
        )
    return traces


def _bond_scatter_traces(scene: dict, style: dict):
    groups: Dict[Tuple[str, bool], list[list[float]]] = {}
    for color, is_minor, start, end in _bond_segments(scene, style):
        groups.setdefault((color, is_minor), []).append([start, end])

    traces = []
    base_width = max(4.0, 24.0 * float(style["bond_radius"]))
    for (color, is_minor), segments in groups.items():
        xs, ys, zs = [], [], []
        for start, end in segments:
            xs.extend([float(start[0]), float(end[0]), None])
            ys.extend([float(start[1]), float(end[1]), None])
            zs.extend([float(start[2]), float(end[2]), None])
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=base_width * (float(style["minor_bond_scale"]) if is_minor else 1.0)),
                opacity=float(style["minor_opacity"]) if is_minor else 1.0,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _atom_scatter_traces(scene: dict, style: dict):
    groups: Dict[Tuple[str, bool], dict] = {}
    for idx, atom in enumerate(scene["draw_atoms"]):
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        key = (atom["elem"], atom["is_minor"])
        groups.setdefault(
            key,
            {"x": [], "y": [], "z": [], "size": [], "text": [], "color": atom["color"], "customdata": []},
        )
        base_size = max(10.0, 95.0 * atom["atom_radius"] * float(style["atom_scale"]))
        groups[key]["x"].append(float(atom["cart"][0]))
        groups[key]["y"].append(float(atom["cart"][1]))
        groups[key]["z"].append(float(atom["cart"][2]))
        groups[key]["size"].append(base_size * (1.12 if atom["is_minor"] else 1.0))
        groups[key]["text"].append(atom["label"])
        groups[key]["customdata"].append([idx, atom["label"], atom["elem"], int(atom["is_minor"])])

    traces = []
    for (elem, is_minor), payload in groups.items():
        traces.append(
            go.Scatter3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                mode="markers",
                text=payload["text"],
                customdata=payload["customdata"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=payload["size"],
                    color=payload["color"],
                    opacity=(
                        max(float(style.get("minor_opacity_floor", 0.18)),
                            float(style["minor_opacity"]))
                        if is_minor else float(style.get("major_opacity", 1.0))
                    ),
                    line=dict(color="#444444" if is_minor else payload["color"], width=3.5 if is_minor else 0),
                ),
                showlegend=False,
                name=f"{elem}{' minor' if is_minor else ''}",
            )
        )
    return traces


def _minor_bond_wireframe_traces(scene: dict, style: dict):
    if not style.get("minor_wireframe", False):
        return []
    xs, ys, zs = [], [], []
    for bond in scene["bonds"]:
        if not bond["is_minor"]:
            continue
        start = np.array(bond["start"], dtype=float)
        end = np.array(bond["end"], dtype=float)
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    if not xs:
        return []
    base_width = max(3.0, 22.0 * float(style["bond_radius"]))
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#202020", width=base_width),
            opacity=0.9,
            hoverinfo="skip",
            showlegend=False,
        )
    ]


def _minor_outline_traces(scene: dict, style: dict):
    payload = {"x": [], "y": [], "z": [], "size": []}
    for atom in scene["draw_atoms"]:
        if not atom["is_minor"]:
            continue
        if style.get("show_minor_only", False) and not atom["is_minor"]:
            continue
        base_size = max(10.0, 95.0 * atom["atom_radius"] * float(style["atom_scale"]))
        ring_scale = 1.34 if style.get("minor_wireframe", False) else 1.20
        payload["x"].append(float(atom["cart"][0]))
        payload["y"].append(float(atom["cart"][1]))
        payload["z"].append(float(atom["cart"][2]))
        payload["size"].append(base_size * ring_scale)
    if not payload["x"]:
        return []
    line_color = "#111111" if style.get("minor_wireframe", False) else "#555555"
    line_width = 7.0 if style.get("minor_wireframe", False) else 4.5
    return [
        go.Scatter3d(
            x=payload["x"],
            y=payload["y"],
            z=payload["z"],
            mode="markers",
            marker=dict(
                size=payload["size"],
                color="rgba(255,255,255,0.0)",
                opacity=1.0,
                line=dict(color=line_color, width=line_width),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    ]


def _highlight_traces(scene: dict, style: dict):
    if style.get("show_minor_only", False):
        return []
    light_dir = (
        -0.28 * np.array(scene["view_x"], dtype=float)
        + 0.34 * np.array(scene["view_y"], dtype=float)
        + 0.72 * np.array(scene["view_z"], dtype=float)
    )
    norm = np.linalg.norm(light_dir)
    if norm < 1e-8:
        return []
    light_dir /= norm

    groups: Dict[str, dict] = {}
    for atom in scene["draw_atoms"]:
        if atom["is_minor"] or atom["elem"] == "H":
            continue
        size = max(5.0, 55.0 * atom["atom_radius"] * float(style["atom_scale"]))
        center = np.array(atom["cart"], dtype=float) + light_dir * (atom["atom_radius"] * float(style["atom_scale"]) * 0.25)
        key = atom["color_light"]
        groups.setdefault(key, {"x": [], "y": [], "z": [], "size": []})
        groups[key]["x"].append(float(center[0]))
        groups[key]["y"].append(float(center[1]))
        groups[key]["z"].append(float(center[2]))
        groups[key]["size"].append(size)

    traces = []
    for color, payload in groups.items():
        traces.append(
            go.Scatter3d(
                x=payload["x"],
                y=payload["y"],
                z=payload["z"],
                mode="markers",
                marker=dict(
                    size=payload["size"],
                    color=color,
                    opacity=0.65,
                    line=dict(color="rgba(255,255,255,0.6)", width=1.5),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _label_traces(scene: dict, style: dict):
    if not style.get("show_labels", True):
        return []
    buckets = {
        False: {"x": [], "y": [], "z": [], "text": [], "color": "#111111"},
        True: {"x": [], "y": [], "z": [], "text": [], "color": "#777777"},
    }
    for item in scene["label_items"]:
        if style.get("show_minor_only", False) and not item["is_minor"]:
            continue
        bucket = buckets[item["is_minor"]]
        bucket["x"].append(float(item["label_cart"][0]))
        bucket["y"].append(float(item["label_cart"][1]))
        bucket["z"].append(float(item["label_cart"][2]))
        bucket["text"].append(item["text"])

    traces = []
    for is_minor, bucket in buckets.items():
        if not bucket["x"]:
            continue
        traces.append(
            go.Scatter3d(
                x=bucket["x"],
                y=bucket["y"],
                z=bucket["z"],
                mode="text",
                text=bucket["text"],
                textfont=dict(size=10 if is_minor else 11, color=bucket["color"]),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _axis_traces(scene: dict, style: dict):
    if not style.get("show_axes", True):
        return []
    mins = np.array(scene["bounds"]["mins"], dtype=float)
    screen_span = max(scene["bounds"]["screen_ranges"])
    offset = 0.10 * screen_span
    origin = mins - offset * np.array(scene["view_x"], dtype=float)
    origin -= offset * np.array(scene["view_y"], dtype=float)
    scale = float(style["axis_scale"]) * screen_span
    color = style.get("axis_color", "#666666")
    opacity = float(style.get("axis_opacity", 0.72))
    labels = style.get("axes_labels") or ["a", "b", "c"]
    labels = list(labels) + ["", "", ""]  # pad defensively

    traces = []
    for vec, label in zip(
        [scene["M"][:, 0], scene["M"][:, 1], scene["M"][:, 2]],
        labels[:3],
    ):
        v = _normalize(vec, [1.0, 0.0, 0.0])
        end = origin + v * scale
        traces.append(
            go.Scatter3d(
                x=[float(origin[0]), float(end[0])],
                y=[float(origin[1]), float(end[1])],
                z=[float(origin[2]), float(end[2])],
                mode="lines",
                line=dict(color=color, width=5),
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter3d(
                x=[float(end[0])],
                y=[float(end[1])],
                z=[float(end[2])],
                mode="text",
                text=[label],
                textfont=dict(size=12, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def axis_key_overlay(scene: dict, style: dict) -> tuple[list[dict], list[dict]]:
    """Build Plotly paper-coord annotations + shapes for a corner axis triad.

    The triad is rendered in **screen space** (paper coordinates) rather than
    inside the 3D scene, so labels and arrows live in a stable figure corner
    and cannot be clipped by the 3D viewport cube or a caller's outer
    matplotlib axes. Labels stack in a left-aligned vertical column (one per
    crystallographic axis, default order c → b → a top-to-bottom), with each
    label followed by a short arrow pointing in the *projected* direction of
    that axis. Arrow lengths are normalised so the longest projection fills
    ``axis_key_arrow_len`` while shorter axes preserve their relative length.

    The arrow body is drawn as a Plotly line ``shape`` and the arrowhead as a
    filled triangular path — both of which honour ``xref='paper'``. Labels
    are separate ``annotations`` objects. Returns ``(annotations, shapes)``.

    Set ``style["show_axis_key"] = True`` to include the triad; when off this
    helper returns empty lists. The projections are read from
    ``scene["projected_axes"]`` (populated by :func:`scene.build_scene_from_atoms`)
    and the label strings come from ``style["axes_labels"]`` with stacking
    order controlled by ``style["axis_key_label_order"]``.
    """
    if not style.get("show_axis_key", False):
        return [], []
    projections = scene.get("projected_axes")
    if not projections or len(projections) < 3:
        return [], []

    axes_labels = list(style.get("axes_labels") or scene.get("axis_labels") or ["a", "b", "c"])[:3]
    label_to_proj = {axes_labels[i]: projections[i] for i in range(min(3, len(axes_labels)))}

    order = list(style.get("axis_key_label_order") or ["c", "b", "a"])
    order = [label for label in order if label in label_to_proj]
    if not order:
        return [], []

    anchor = style.get("axis_key_anchor") or [0.05, 0.07]
    anchor_x = float(anchor[0])
    anchor_y = float(anchor[1])
    row_gap = float(style.get("axis_key_row_gap", 0.095))
    arrow_len = float(style.get("axis_key_arrow_len", 0.085))
    label_pad = float(style.get("axis_key_label_pad", 0.045))
    font_size = float(style.get("axis_key_font_size", 13))
    line_width = float(style.get("axis_key_line_width", 1.6))
    head_len = float(style.get("axis_key_head_len", 0.025))
    head_width = float(style.get("axis_key_head_width", 0.018))
    color = style.get("axis_key_color", "#2F2F2F")
    italic = bool(style.get("axis_key_italic", True))

    norms = [math.hypot(float(label_to_proj[label][0]), float(label_to_proj[label][1])) for label in order]
    max_norm = max(norms) if norms else 0.0
    if max_norm < 1e-8:
        return [], []

    # Cap arrow_len so the arrow's **vertical** extent (arrow_len * |dy/norm|)
    # can never exceed half the row gap. Without this clamp a steeply-
    # projecting axis on one row can shoot into the neighbouring row and
    # collide with that row's label, producing the "fragmented triad" look.
    # Share a single scale factor across all rows so relative lengths are
    # preserved.
    max_abs_uy = max(
        abs(float(label_to_proj[label][1]) / norm) if norm > 1e-8 else 0.0
        for label, norm in zip(order, norms)
    )
    if max_abs_uy > 1e-8:
        y_budget = 0.42 * row_gap
        arrow_len = min(arrow_len, y_budget / max_abs_uy)

    annotations: list[dict] = []
    shapes: list[dict] = []
    n_rows = len(order)
    for row_idx, label in enumerate(order):
        row_y = anchor_y + (n_rows - 1 - row_idx) * row_gap
        text = f"<i>{label}</i>" if italic else label
        annotations.append(dict(
            x=anchor_x, y=row_y,
            xref="paper", yref="paper",
            text=text,
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=font_size, color=color),
        ))
        dx, dy = label_to_proj[label]
        norm = math.hypot(float(dx), float(dy))
        if norm < 1e-8:
            continue
        ux = float(dx) / norm
        uy = float(dy) / norm
        # Scale arrow length by the axis's 2D projection magnitude so near-
        # perpendicular axes render as shorter arrows. Impose a minimum so
        # (a) near-perpendicular axes never collapse to an invisible speck
        # (the user would read that as a rendering bug) and (b) the shaft is
        # always longer than the arrowhead — otherwise the head's base
        # falls behind the arrow's own origin and the triad visibly
        # fragments into detached triangles.
        min_scale = 0.65
        rel = max(norm / max_norm, min_scale)
        length = max(arrow_len * rel, 1.35 * head_len)
        x0 = anchor_x + label_pad
        y0 = row_y
        x1 = x0 + length * ux
        y1 = y0 + length * uy
        # Arrow shaft (stops just short of the tip to avoid the arrowhead
        # line-width bleeding past the triangle on retina renders).
        shaft_end_x = x1 - 0.55 * head_len * ux
        shaft_end_y = y1 - 0.55 * head_len * uy
        shapes.append(dict(
            type="line",
            xref="paper", yref="paper",
            x0=x0, y0=y0,
            x1=shaft_end_x, y1=shaft_end_y,
            line=dict(color=color, width=line_width),
            layer="above",
        ))
        # Filled triangular arrowhead tip — points from (x1, y1) backward
        # along (-ux, -uy), with left/right base points straddling the
        # perpendicular (-uy, ux).
        base_cx = x1 - head_len * ux
        base_cy = y1 - head_len * uy
        px = -uy
        py = ux
        base_left_x = base_cx + 0.5 * head_width * px
        base_left_y = base_cy + 0.5 * head_width * py
        base_right_x = base_cx - 0.5 * head_width * px
        base_right_y = base_cy - 0.5 * head_width * py
        shapes.append(dict(
            type="path",
            xref="paper", yref="paper",
            path=(
                f"M {x1},{y1} "
                f"L {base_left_x},{base_left_y} "
                f"L {base_right_x},{base_right_y} Z"
            ),
            fillcolor=color,
            line=dict(color=color, width=0),
            layer="above",
        ))
    return annotations, shapes


def axis_key_annotations(scene: dict, style: dict) -> list[dict]:
    """Backwards-compatible wrapper returning only the annotations list.

    Prefer :func:`axis_key_overlay` which also returns paper-coord shapes for
    the arrow shafts and arrowheads.
    """
    annotations, _ = axis_key_overlay(scene, style)
    return annotations


def _unit_cell_traces(scene: dict, style: dict):
    if not style.get("show_unit_cell", False):
        return []
    origin = np.zeros(3, dtype=float)
    a = np.array(scene["M"][:, 0], dtype=float)
    b = np.array(scene["M"][:, 1], dtype=float)
    c = np.array(scene["M"][:, 2], dtype=float)
    corners = {
        "000": origin,
        "100": a,
        "010": b,
        "001": c,
        "110": a + b,
        "101": a + c,
        "011": b + c,
        "111": a + b + c,
    }
    edges = [
        ("000", "100"), ("000", "010"), ("000", "001"),
        ("100", "110"), ("100", "101"),
        ("010", "110"), ("010", "011"),
        ("001", "101"), ("001", "011"),
        ("110", "111"), ("101", "111"), ("011", "111"),
    ]
    xs, ys, zs = [], [], []
    for start_key, end_key in edges:
        start = corners[start_key]
        end = corners[end_key]
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#777777", width=4),
            opacity=0.8,
            hoverinfo="skip",
            showlegend=False,
            name="unit-cell-box",
        )
    ]


def hull_mesh_trace(shell_coords, color: str, opacity: float = 0.15):
    coords = np.array(shell_coords, dtype=float)
    if len(coords) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
    except Exception:  # pragma: no cover - optional dependency
        return None
    hull = ConvexHull(coords)
    return go.Mesh3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        i=hull.simplices[:, 0],
        j=hull.simplices[:, 1],
        k=hull.simplices[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
        name="coordination-hull",
    )


def hull_edge_traces(shell_coords, color: str):
    coords = np.array(shell_coords, dtype=float)
    if len(coords) < 4:
        return []
    try:
        from scipy.spatial import ConvexHull
    except Exception:  # pragma: no cover - optional dependency
        return []
    hull = ConvexHull(coords)
    edges = set()
    for simplex in hull.simplices:
        a, b, c = simplex
        edges.add(tuple(sorted((int(a), int(b)))))
        edges.add(tuple(sorted((int(b), int(c)))))
        edges.add(tuple(sorted((int(a), int(c)))))

    xs, ys, zs = [], [], []
    for i, j in sorted(edges):
        p0 = coords[i]
        p1 = coords[j]
        xs.extend([float(p0[0]), float(p1[0]), None])
        ys.extend([float(p0[1]), float(p1[1]), None])
        zs.extend([float(p0[2]), float(p1[2]), None])
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=6),
            opacity=0.95,
            hoverinfo="skip",
            showlegend=False,
            name="coordination-edges",
        )
    ]


def shell_center_lines(center, shell_coords):
    center = np.array(center, dtype=float)
    coords = np.array(shell_coords, dtype=float)
    if len(coords) == 0:
        return []
    xs, ys, zs = [], [], []
    for point in coords:
        xs.extend([float(center[0]), float(point[0]), None])
        ys.extend([float(center[1]), float(point[1]), None])
        zs.extend([float(center[2]), float(point[2]), None])
    return [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#6A5ACD", width=4, dash="dash"),
            opacity=0.85,
            hoverinfo="skip",
            showlegend=False,
            name="coordination-lines",
        )
    ]


def shell_atom_traces(shell_coords, distances, color="#7C5CBF"):
    coords = np.array(shell_coords, dtype=float)
    if len(coords) == 0:
        return []
    dists = np.array(distances, dtype=float)
    if len(dists) == 0:
        dists = np.ones(len(coords))
    size = 12.0 + (dists.max() - dists + 0.1) * 5.0
    return [
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=size.tolist(), color=color, opacity=0.9, line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate="d=%{text:.3f} Å<extra></extra>",
            text=dists.tolist(),
            showlegend=False,
            name="coordination-shell",
        )
    ]


def topology_traces(topology_data: dict | None):
    if not topology_data:
        return []
    traces = []
    shell_coords = topology_data.get("shell_coords") or []
    center = topology_data.get("center_coords")
    distances = topology_data.get("distances") or []
    hull_trace = hull_mesh_trace(shell_coords, color="#7C5CBF", opacity=0.16)
    if hull_trace is not None:
        traces.append(hull_trace)
    traces.extend(hull_edge_traces(shell_coords, color="#7C5CBF"))
    if center is not None:
        traces.extend(shell_center_lines(center, shell_coords))
        traces.append(
            go.Scatter3d(
                x=[float(center[0])],
                y=[float(center[1])],
                z=[float(center[2])],
                mode="markers",
                marker=dict(size=14, color="#E07C24", opacity=0.95, line=dict(color="#FFFFFF", width=1.5)),
                hovertemplate=f"{topology_data.get('center_label', 'center')}<extra></extra>",
                showlegend=False,
            )
        )
    traces.extend(shell_atom_traces(shell_coords, distances))
    return traces


def topology_histogram_figure(topology_data: dict | None) -> go.Figure:
    fig = go.Figure()
    distances = (topology_data or {}).get("all_distances", [])
    shell = set((topology_data or {}).get("distances", []))
    if distances:
        colors = ["#7C5CBF" if dist in shell else "#C9C9E8" for dist in distances]
        fig.add_trace(go.Bar(x=list(range(1, len(distances) + 1)), y=distances, marker_color=colors))
    fig.update_layout(
        margin=dict(l=18, r=18, t=28, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Neighbor rank",
        yaxis_title="Distance (Å)",
        showlegend=False,
        title="Distance Histogram",
    )
    return fig


def topology_results_markdown(topology_data: dict | None) -> str:
    if not topology_data:
        return "Topology analysis inactive."
    angular = topology_data.get("angular", {})
    best = angular.get("best_match")
    planarity = topology_data.get("planarity", {})
    prism = topology_data.get("prism_analysis", {})
    lines = [
        f"Center: {topology_data.get('center_label', '?')} ({topology_data.get('center_type', '?')})",
        f"CN: {topology_data.get('coordination_number', 0)}",
    ]
    if best:
        lines.append(f"Best ideal: {best['name']} (angular RMSD {best['angular_rmsd']:.2f}°)")
    if planarity.get("best_rms") is not None:
        lines.append(f"Best planarity RMS: {planarity['best_rms']:.3f} Å")
    if prism.get("classification"):
        lines.append(f"Prism test: {prism['classification']} ({prism['twist_deg']:.1f}°)")
    return "\n".join(lines)


def build_figure(scene: dict, style: dict, topology_data: dict | None = None) -> go.Figure:
    fig = go.Figure()
    xr, yr, zr = _scene_ranges(scene, style, topology_data=topology_data if style.get("topology_enabled", True) else None)
    use_fast = bool(style.get("fast_rendering", False)) or len(scene.get("draw_atoms", [])) > 200

    bond_traces = _bond_scatter_traces(scene, style) if use_fast else _bond_mesh_traces(scene, style)
    atom_traces = _atom_scatter_traces(scene, style) if use_fast else _atom_mesh_traces(scene, style)

    for trace in bond_traces:
        fig.add_trace(trace)
    for trace in _minor_bond_wireframe_traces(scene, style):
        fig.add_trace(trace)
    for trace in atom_traces:
        fig.add_trace(trace)
    for trace in _minor_outline_traces(scene, style):
        fig.add_trace(trace)
    for trace in _highlight_traces(scene, style):
        fig.add_trace(trace)
    for trace in _label_traces(scene, style):
        fig.add_trace(trace)
    for trace in _axis_traces(scene, style):
        fig.add_trace(trace)
    for trace in _unit_cell_traces(scene, style):
        fig.add_trace(trace)
    if style.get("topology_enabled", True):
        for trace in topology_traces(topology_data):
            fig.add_trace(trace)
    fig.add_trace(_atom_selection_trace(scene, style))

    show_title = bool(style.get("show_title", True))
    title_arg = dict(text=scene["title"], x=0.5) if show_title else None
    top_margin = 50 if show_title else 0

    # If all three axis ranges share a side (i.e. a caller stamped a cube via
    # uniform_viewport), lock the aspect ratio to ``cube`` so the camera does
    # not stretch when Plotly renders to a non-square viewport.
    xr_span = xr[1] - xr[0]
    yr_span = yr[1] - yr[0]
    zr_span = zr[1] - zr[0]
    is_cube = max(
        abs(xr_span - yr_span),
        abs(yr_span - zr_span),
        abs(xr_span - zr_span),
    ) < 1e-6
    aspectmode = "cube" if is_cube else "data"

    layout_kwargs = dict(
        title=title_arg,
        showlegend=False,
        paper_bgcolor=style.get("background", "#FFFFFF"),
        plot_bgcolor=style.get("background", "#FFFFFF"),
        margin=dict(l=0, r=0, t=top_margin, b=0),
        scene=dict(
            xaxis=dict(visible=False, range=xr),
            yaxis=dict(visible=False, range=yr),
            zaxis=dict(visible=False, range=zr),
            aspectmode=aspectmode,
            camera=_plotly_camera_from_scene(scene),
            bgcolor=style.get("background", "#FFFFFF"),
        ),
    )
    key_annotations, key_shapes = axis_key_overlay(scene, style)
    if key_annotations:
        layout_kwargs["annotations"] = key_annotations
    if key_shapes:
        layout_kwargs["shapes"] = key_shapes
    fig.update_layout(**layout_kwargs)
    return fig
