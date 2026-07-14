import logging

from itertools import permutations
from pathlib import Path

import numpy as np
import trimesh

from mathutils import Vector

console_logger = logging.getLogger(__name__)


DIMENSION_CONTRACT_SCHEMA_VERSION = 1
DIMENSION_CONTRACT_RTOL = 1.0e-5
DIMENSION_CONTRACT_ATOL_METERS = 1.0e-6
DIMENSION_CONTRACT_MIN_EXTENT_METERS = 1.0e-9
DIMENSION_CONTRACT_MAX_UNIFORM_SCALE = 1.0e6
DIMENSION_CONTRACT_MAJOR_AXIS_MIN_OCCUPANCY = 0.5


def load_mesh_as_trimesh(mesh_path: Path, force_merge: bool = True) -> trimesh.Trimesh:
    """Load a mesh file and ensure it's a single Trimesh object.

    Handles Scene objects (files containing multiple meshes) by concatenating
    all Trimesh components into a single mesh. This is commonly needed when
    loading GLTF files that may contain multiple geometry objects.

    Args:
        mesh_path: Path to mesh file (GLTF, GLB, OBJ, STL, etc.). Must exist.
        force_merge: If True, merge Scene objects into single Trimesh. If False,
            raise ValueError if a Scene is encountered. Default: True.

    Returns:
        Single Trimesh object containing the loaded geometry.

    Raises:
        ValueError: If file cannot be loaded, contains no valid geometry, or
            contains a Scene when force_merge=False.
        FileNotFoundError: If mesh_path does not exist.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {mesh_path}: {e}")

    if isinstance(mesh, trimesh.Scene):
        if not force_merge:
            raise ValueError(
                f"Expected single Trimesh, got Scene with multiple meshes: {mesh_path}"
            )

        # Extract all valid Trimesh objects from the Scene.
        meshes = [
            geom
            for geom in mesh.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]

        if not meshes:
            raise ValueError(f"Scene contains no valid meshes: {mesh_path}")

        # Concatenate all meshes into a single Trimesh.
        mesh = trimesh.util.concatenate(meshes)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Could not load valid Trimesh from {mesh_path}. "
            f"Got type: {type(mesh)}. File may be corrupted or contain no mesh geometry."
        )

    return mesh


def validate_dimension_vector(
    dimensions: list[float] | tuple[float, float, float] | np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Return a finite, strictly-positive three-axis dimension vector."""
    try:
        values = np.asarray(dimensions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly three finite numbers") from exc
    if values.shape != (3,):
        raise ValueError(f"{label} must contain exactly three values, got {dimensions}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{label} must contain finite positive values, got {dimensions}")
    return values


def scene_to_gltf_dimensions(
    scene_dimensions: list[float] | tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Map SceneSmith Z-up ``[width, depth, height]`` to GLTF Y-up axes."""
    width, depth, height = validate_dimension_vector(
        scene_dimensions, label="requested SceneSmith dimensions"
    )
    return np.array([width, height, depth], dtype=float)


def gltf_to_scene_dimensions(
    gltf_dimensions: list[float] | tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Map GLTF Y-up ``[x, y, z]`` extents to SceneSmith Z-up dimensions."""
    x_extent, y_extent, z_extent = validate_dimension_vector(
        gltf_dimensions, label="measured GLTF dimensions"
    )
    return np.array([x_extent, z_extent, y_extent], dtype=float)


def measure_mesh_dimensions(mesh_path: Path) -> np.ndarray:
    """Load a mesh and return validated native-axis AABB extents.

    Source meshes have arbitrary absolute scale, and legitimate paper-like assets can
    be far thinner than one millimeter.  The dimension contract therefore rejects only
    non-finite or numerically zero extents here; semantic aspect compatibility is
    checked separately.
    """
    mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)
    extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)):
        raise ValueError(f"Mesh has invalid bounds at {mesh_path}: {extents}")
    if np.any(extents <= DIMENSION_CONTRACT_MIN_EXTENT_METERS):
        raise ValueError(
            "Mesh has numerically degenerate dimensions at "
            f"{mesh_path}: {extents}; every axis must exceed "
            f"{DIMENSION_CONTRACT_MIN_EXTENT_METERS}m"
        )
    return extents


def uniform_dimension_fit_plan(
    current_dimensions: list[float] | tuple[float, float, float] | np.ndarray,
    target_dimensions: list[float] | tuple[float, float, float] | np.ndarray,
    *,
    orientation_invariant: bool = False,
    major_axis_min_occupancy: float = DIMENSION_CONTRACT_MAJOR_AXIS_MIN_OCCUPANCY,
) -> dict[str, object]:
    """Plan a proportion-preserving fit and reject aspect-incompatible candidates.

    A single scale is the largest value that fits all requested bounds.  The two
    largest requested axes must each remain substantially occupied.  Thickness may
    underfill, which permits real sheets of paper, but a box/trophy cannot be shrunk
    into a tiny speck merely to satisfy a paper-thickness bound.
    """
    current = validate_dimension_vector(current_dimensions, label="current dimensions")
    target = validate_dimension_vector(target_dimensions, label="target dimensions")
    if orientation_invariant:
        current = np.sort(current)
        target = np.sort(target)
    if (
        not np.isfinite(major_axis_min_occupancy)
        or major_axis_min_occupancy <= 0.0
        or major_axis_min_occupancy > 1.0
    ):
        raise ValueError("major_axis_min_occupancy must be in (0, 1]")

    ratios = target / current
    uniform_scale = float(np.min(ratios))
    if (
        not np.isfinite(uniform_scale)
        or uniform_scale <= 0.0
        or uniform_scale > DIMENSION_CONTRACT_MAX_UNIFORM_SCALE
    ):
        raise ValueError(f"Unsafe uniform dimension scale: {uniform_scale}")
    fitted = current * uniform_scale
    occupancy = fitted / target
    major_axes = np.argsort(target)[-2:]
    major_occupancies = occupancy[major_axes]
    compatible = bool(
        np.all(fitted <= target + DIMENSION_CONTRACT_ATOL_METERS)
        and np.all(major_occupancies >= major_axis_min_occupancy)
    )
    return {
        "current_dimensions": current,
        "target_dimensions": target,
        "uniform_scale": uniform_scale,
        "expected_dimensions": fitted,
        "occupancy": occupancy,
        "major_axes": major_axes,
        "major_axis_occupancies": major_occupancies,
        "major_axis_min_occupancy": float(major_axis_min_occupancy),
        "compatible": compatible,
    }


def mesh_dimension_candidate_compatibility(
    mesh_path: Path,
    requested_scene_dimensions: list[float],
) -> tuple[bool, dict[str, object]]:
    """Prefilter an arbitrarily oriented candidate before semantic validation."""
    current = measure_mesh_dimensions(mesh_path)
    target = scene_to_gltf_dimensions(requested_scene_dimensions)
    plan = uniform_dimension_fit_plan(
        current, target, orientation_invariant=True
    )
    return bool(plan["compatible"]), plan


def _proper_axis_permutation_transform(
    permutation: tuple[int, int, int],
) -> np.ndarray:
    """Return a proper right-angle rotation with ``new[i] = old[permutation[i]]``.

    AABB extents do not encode axis signs, but the visual mesh still needs a
    rotation rather than a reflection.  Flip one row when necessary so the
    resulting signed permutation has determinant +1.
    """
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = 0.0
    for new_axis, old_axis in enumerate(permutation):
        transform[new_axis, old_axis] = 1.0
    if np.linalg.det(transform[:3, :3]) < 0.0:
        transform[2, permutation[2]] = -1.0
    return transform


def scale_mesh_uniformly_to_scene_dimension_contract(
    mesh_path: Path,
    requested_scene_dimensions: list[float],
    output_path: Path | None = None,
) -> tuple[Path, float, dict[str, object]]:
    """Uniformly fit a canonical Y-up mesh to an explicit SceneSmith request.

    The exported file is reloaded and must exactly match the planned uniform result,
    remain inside every requested bound, and satisfy the major-axis occupancy policy.
    No median/average scale is permitted.
    """
    current = measure_mesh_dimensions(mesh_path)
    target = scene_to_gltf_dimensions(requested_scene_dimensions)

    # Canonicalization normally establishes the vertical (GLTF Y) axis, but
    # its inferred front direction can yaw an asset by 90 degrees.  Preserve
    # that cheap correction first.  Some thin assets have an ambiguous VLM
    # front/up prediction; if neither horizontal orientation fits, recover a
    # proper right-angle axis rotation from the explicit requested dimensions.
    # This is not an unconstrained fit: the requested scene height still has
    # to occupy GLTF Y, so an upright bottle cannot be silently accepted lying
    # on its side.
    fits: list[tuple[str, np.ndarray, dict[str, object], np.ndarray]] = []
    for orientation, candidate_dimensions, transform in (
        ("direct", current, np.eye(4, dtype=float)),
        (
            "yaw",
            current[[2, 1, 0]],
            trimesh.transformations.rotation_matrix(
                np.pi / 2.0, [0.0, 1.0, 0.0]
            ),
        ),
    ):
        candidate_plan = uniform_dimension_fit_plan(candidate_dimensions, target)
        if candidate_plan["compatible"]:
            fits.append((orientation, candidate_dimensions, candidate_plan, transform))

    if not fits:
        # Recover only a discrete, rigid axis permutation.  The direct and yaw
        # cases above stay preferred so assets with an unambiguous VLM
        # orientation retain their established visual direction.
        for permutation in permutations(range(3)):
            if permutation in ((0, 1, 2), (2, 1, 0)):
                continue
            candidate_dimensions = current[list(permutation)]
            candidate_plan = uniform_dimension_fit_plan(candidate_dimensions, target)
            if candidate_plan["compatible"]:
                fits.append(
                    (
                        "axis_recovery",
                        candidate_dimensions,
                        candidate_plan,
                        _proper_axis_permutation_transform(permutation),
                    )
                )

    if not fits:
        direct_plan = uniform_dimension_fit_plan(current, target)
        raise ValueError(
            "Mesh aspect ratio is incompatible with requested dimensions: "
            f"current_gltf={current.tolist()}, target_gltf={target.tolist()}, "
            "major_axis_occupancies="
            f"{np.asarray(direct_plan['major_axis_occupancies']).tolist()}, required>="
            f"{direct_plan['major_axis_min_occupancy']}"
        )

    # Prefer the direct orientation when it fits.  Otherwise select the
    # largest valid uniform fit, preserving as much usable geometry as possible
    # without violating any requested bound.
    orientation, fitted_current, plan, transform = max(
        fits,
        key=lambda fit: (
            float(fit[2]["uniform_scale"]),
            fit[0] == "direct",
            fit[0] == "yaw",
        ),
    )

    mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)
    if orientation != "direct":
        mesh.apply_transform(transform)
    uniform_scale = float(plan["uniform_scale"])
    mesh.apply_scale(uniform_scale)
    final_output_path = output_path if output_path is not None else mesh_path
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(final_output_path)

    reloaded = measure_mesh_dimensions(final_output_path)
    expected = np.asarray(plan["expected_dimensions"], dtype=float)
    if not np.allclose(
        reloaded,
        expected,
        rtol=DIMENSION_CONTRACT_RTOL,
        atol=DIMENSION_CONTRACT_ATOL_METERS,
    ):
        raise ValueError(
            "Reloaded mesh violates exact dimension postcondition: "
            f"expected_gltf={expected.tolist()}, measured_gltf={reloaded.tolist()}"
        )
    if np.any(reloaded > target + DIMENSION_CONTRACT_ATOL_METERS):
        raise ValueError(
            "Reloaded mesh exceeds requested bounds: "
            f"target_gltf={target.tolist()}, measured_gltf={reloaded.tolist()}"
        )

    receipt = {
        "schema_version": DIMENSION_CONTRACT_SCHEMA_VERSION,
        "policy": "proportion_preserving_uniform_fit",
        "requested_scene_dimensions_m": validate_dimension_vector(
            requested_scene_dimensions, label="requested SceneSmith dimensions"
        ).tolist(),
        "requested_gltf_dimensions_m": target.tolist(),
        "source_gltf_dimensions_m": current.tolist(),
        "yaw_rotation_degrees": 90 if orientation == "yaw" else 0,
        "orientation_recovery": orientation,
        "oriented_source_gltf_dimensions_m": fitted_current.tolist(),
        "planned_final_gltf_dimensions_m": expected.tolist(),
        "measured_final_gltf_dimensions_m": reloaded.tolist(),
        "measured_final_scene_dimensions_m": gltf_to_scene_dimensions(reloaded).tolist(),
        "uniform_scale": uniform_scale,
        "major_axis_occupancies": np.asarray(
            plan["major_axis_occupancies"], dtype=float
        ).tolist(),
        "major_axis_min_occupancy": float(plan["major_axis_min_occupancy"]),
        "rtol": DIMENSION_CONTRACT_RTOL,
        "atol_m": DIMENSION_CONTRACT_ATOL_METERS,
        "status": "pass",
    }
    return final_output_path, uniform_scale, receipt


def convert_glb_to_gltf(
    input_path: Path, output_path: Path, export_yup: bool = False
) -> Path:
    """Convert GLB file to GLTF with separate texture files using Blender.

    Drake requires GLTF files with separate textures rather than GLB files
    with embedded textures. This function uses Blender to import a GLB file
    and export it as GLTF_SEPARATE format, which creates separate files for
    textures and binary data.

    Coordinate System Handling:
    - export_yup=True: Converts Blender's Z-up to GLTF's Y-up standard
      (used for initial conversion before canonicalization)
    - export_yup=False: Preserves Blender's Z-up orientation
      (used after canonicalization for Drake)

    Pipeline workflow:
    1. Initial GLB→GLTF conversion uses export_yup=True (creates Y-up GLTF)
    2. VLM analyzes the Y-up GLTF (Blender imports and converts to Z-up)
    3. Canonicalization processes mesh in Blender's Z-up space
    4. Final export uses export_yup=False to preserve Z-up for Drake

    Args:
        input_path: Path to input GLB or GLTF file. Must exist.
        output_path: Path for output GLTF file. Textures and .bin files will
            be saved in the same directory with related names.
        export_yup: If True, converts to Y-up GLTF standard. If False, keeps
            Blender's Z-up orientation. Default False for Drake compatibility.

    Returns:
        Path to the converted GLTF file.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If Blender conversion fails.
    """
    # NOTE: bpy is imported inside the function to avoid import errors in test
    # environments. The bpy library can fail to load due to missing system
    # dependencies, and tests/unit/__init__.py provides a mock fallback. Importing
    # at module level would trigger the import error before the mock is set up.
    import bpy

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    console_logger.info(f"Converting {input_path.suffix} to GLTF: {output_path}")

    # Clear existing scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import GLB/GLTF file.
    bpy.ops.import_scene.gltf(filepath=str(input_path))

    # Select all imported objects.
    bpy.ops.object.select_all(action="SELECT")

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export as GLTF with separate files (textures, bin data).
    # export_yup parameter controls coordinate system:
    # - True: Convert Z-up (Blender) to Y-up (GLTF standard) for pre-canonicalization
    # - False: Keep Z-up (Blender) for post-canonicalization Drake assets
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLTF_SEPARATE",  # Separate .gltf, .bin, textures.
        use_selection=True,
        export_yup=export_yup,
    )

    console_logger.info(
        f"Converted to GLTF with separate textures (Drake compatible): {output_path}"
    )

    return output_path


def convert_gltf_to_glb(input_path: Path, output_path: Path) -> Path:
    """Convert GLTF file (with external .bin buffers) to self-contained GLB.

    This is primarily used when sending meshes via HTTP to BlenderServer, where
    the server cannot access external .bin files referenced by GLTF. GLB format
    embeds all buffers into a single binary file.

    Uses trimesh for conversion, which doesn't require bpy and can be called
    safely from forked worker processes.

    Args:
        input_path: Path to input GLTF or GLB file. Must exist.
        output_path: Path for output GLB file.

    Returns:
        Path to the converted GLB file.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If mesh cannot be loaded or conversion fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    console_logger.debug(f"Converting {input_path} to GLB: {output_path}")

    try:
        # Load GLTF (trimesh will resolve external .bin buffers).
        scene = trimesh.load(str(input_path))

        # Ensure output directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export as GLB (single binary file with embedded buffers).
        scene.export(str(output_path), file_type="glb")

        console_logger.debug(f"Converted to GLB: {output_path}")
        return output_path

    except Exception as e:
        raise ValueError(f"Failed to convert {input_path} to GLB: {e}")


def convert_obj_to_gltf(
    input_path: Path, output_path: Path, export_yup: bool = True
) -> Path:
    """Convert OBJ file to GLTF with embedded textures using Blender.

    OBJ files with MTL materials and texture references are converted to GLTF
    format which Drake's Meshcat can render with textures.

    Coordinate System Handling:
    - OBJ files are typically Z-up (same as Blender's native format)
    - export_yup=True: Converts to GLTF's Y-up standard (recommended)
    - export_yup=False: Keeps Z-up orientation

    Args:
        input_path: Path to input OBJ file. Must exist.
        output_path: Path for output GLTF file.
        export_yup: If True, converts to Y-up GLTF standard. Default True.

    Returns:
        Path to the converted GLTF file.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If Blender conversion fails.
    """
    import bpy

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    console_logger.info(f"Converting OBJ to GLTF: {input_path} -> {output_path}")

    # Clear existing scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import OBJ file with +Y forward, +Z up (Drake/URDF convention).
    # Blender handles MTL and textures automatically.
    bpy.ops.wm.obj_import(filepath=str(input_path), forward_axis="Y", up_axis="Z")

    # Select all imported objects.
    bpy.ops.object.select_all(action="SELECT")

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export as GLTF with separate textures (Drake compatible).
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLTF_SEPARATE",  # Separate .gltf, .bin, textures.
        export_yup=export_yup,
    )

    console_logger.info(f"Converted OBJ to GLTF: {output_path}")

    return output_path


def convert_objs_to_gltf(directory: Path, export_yup: bool = True) -> list[Path]:
    """Convert all OBJ files in a directory to GLTF.

    Each OBJ file is converted to a GLTF file with the same base name.
    This is useful for SDF conversion where each visual mesh needs to be
    converted individually.

    Args:
        directory: Directory containing OBJ files.
        export_yup: If True, converts to Y-up GLTF standard. Default True.

    Returns:
        List of paths to converted GLTF files.
    """
    converted = []
    obj_files = sorted(directory.glob("*.obj"))

    for obj_path in obj_files:
        gltf_path = obj_path.with_suffix(".gltf")
        try:
            convert_obj_to_gltf(obj_path, gltf_path, export_yup=export_yup)
            converted.append(gltf_path)
        except Exception as e:
            console_logger.warning(f"Failed to convert {obj_path}: {e}")

    console_logger.info(
        f"Converted {len(converted)}/{len(obj_files)} OBJ files to GLTF"
    )
    return converted


def merge_objs_to_gltf(
    obj_paths_with_offsets: list[tuple[Path, tuple[float, float, float]]],
    output_path: Path,
) -> Path:
    """Merge multiple OBJ files into a single GLTF with transform offsets applied.

    Each OBJ file is imported, positioned according to its offset, and then all
    meshes are joined into a single object before exporting. Materials and textures
    are preserved.

    This is useful for combining multiple visual mesh files for a URDF link into
    a single GLTF file, which reduces draw calls and simplifies the file structure.

    Args:
        obj_paths_with_offsets: List of (obj_path, (x, y, z)) tuples where the
            offset is the position to apply to the mesh vertices.
        output_path: Path for output GLTF file.

    Returns:
        Path to the merged GLTF file.

    Raises:
        ValueError: If no valid meshes were imported.
    """
    import bpy

    # Clear existing scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    imported_objects = []
    for obj_path, offset in obj_paths_with_offsets:
        if not obj_path.exists():
            console_logger.warning(f"OBJ file not found, skipping: {obj_path}")
            continue

        # Import OBJ with +Y forward, +Z up (Drake/URDF convention).
        bpy.ops.wm.obj_import(
            filepath=str(obj_path),
            forward_axis="Y",
            up_axis="Z",
        )

        # Apply offset to newly imported mesh objects.
        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                obj.location += Vector(offset)
                imported_objects.append(obj)

    if not imported_objects:
        raise ValueError("No valid OBJ files were imported")

    # Select all imported objects.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_objects[0]

    # Apply transforms to bake offsets into vertex data.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Join all meshes into one.
    bpy.ops.object.join()

    merged = bpy.context.active_object
    merged.name = output_path.stem

    console_logger.info(
        f"Merged {len(obj_paths_with_offsets)} OBJ files: "
        f"{len(merged.data.vertices)} vertices, "
        f"{len(merged.data.materials)} materials"
    )

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export as GLTF with default Y-up (standard GLTF convention).
    # Drake handles Y-up GLTF correctly.
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLTF_SEPARATE",
        export_yup=True,  # Standard GLTF Y-up convention.
        export_materials="EXPORT",
    )

    console_logger.info(f"Exported merged GLTF: {output_path}")

    return output_path


def scale_mesh_uniformly_to_dimensions(
    mesh_path: Path,
    desired_dimensions: list[float],
    output_path: Path | None = None,
    min_dimension_meters: float = 0.001,
    relative_threshold: float = 0.01,
) -> tuple[Path, float]:
    """Scale a 3D mesh uniformly to match desired dimensions.

    Uses the median scale factor across all axes to preserve the mesh's
    original proportions while scaling to match the target dimensions. This
    is appropriate for image-to-3D generated meshes where the relative
    proportions are likely correct but the absolute scale is unknown.

    Validates mesh dimensions to reject degenerate geometries that would
    produce incorrect results when uniformly scaled.

    Args:
        mesh_path: Path to input mesh file (GLB, OBJ, STL, etc.). Must exist.
        desired_dimensions: Target (width, depth, height) in meters to fit
            within. Must be positive values. Width corresponds to X-axis, depth
            to Y-axis, and height to Z-axis in the mesh's local coordinate
            system.
        output_path: Optional output path for the scaled mesh. If None, the
            input mesh will be overwritten. The format is inferred from the file
            extension.
        min_dimension_meters: Minimum acceptable dimension (meters). Meshes with
            any dimension below this are rejected as degenerate. Default: 0.001 (1mm).
        relative_threshold: Minimum ratio between smallest and largest dimension.
            Meshes where min_dim/max_dim < this threshold are rejected. Default:
            0.01 (1%, meaning aspect ratios worse than 100:1 are rejected).

    Returns:
        Tuple of (path to scaled mesh file, uniform scale factor applied).
        The scale factor is needed to correctly scale HSSD pre-computed support
        surfaces which are stored at original mesh dimensions.

    Raises:
        FileNotFoundError: If the input mesh file does not exist.
        ValueError: If desired dimensions contain non-positive values, if the
            mesh cannot be loaded, or if mesh has degenerate dimensions.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    # Validate dimensions.
    if len(desired_dimensions) != 3:
        raise ValueError(
            f"desired_dimensions must contain exactly 3 values (width, depth, height), "
            f"got {len(desired_dimensions)}: {desired_dimensions}"
        )
    if any(dim <= 0 for dim in desired_dimensions):
        raise ValueError(f"All dimensions must be positive, got: {desired_dimensions}")

    # Load mesh and ensure it's a single Trimesh object.
    mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)

    # Get current bounding box.
    bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    current_dimensions = bounds[1] - bounds[0]  # [width, depth, height]

    # Check for degenerate dimensions (completely flat meshes).
    if np.any(current_dimensions < min_dimension_meters):
        degenerate_axes = [
            f"{axis}={dim:.6f}m"
            for axis, dim in zip(["X", "Y", "Z"], current_dimensions)
            if dim < min_dimension_meters
        ]
        raise ValueError(
            f"Mesh has degenerate dimension(s) below {min_dimension_meters}m "
            f"threshold: {', '.join(degenerate_axes)}. Current dimensions: "
            f"{current_dimensions}. Cannot scale flat or degenerate mesh from "
            f"{mesh_path}. This likely indicates a mesh generation failure - "
            f"please regenerate the asset."
        )

    # Check for relative degenerate dimensions (one dimension much smaller than others).
    # This catches cases where a dimension passes the absolute threshold but would still
    # cause extreme scaling artifacts due to disproportionate geometry.
    min_dim = np.min(current_dimensions)
    max_dim = np.max(current_dimensions)
    relative_ratio = min_dim / max_dim

    if relative_ratio < relative_threshold:
        min_axis_idx = np.argmin(current_dimensions)
        axis_names = ["X", "Y", "Z"]
        raise ValueError(
            f"Degenerate dimension detected - {axis_names[min_axis_idx]}-axis "
            f"({min_dim:.6f}m) is only {relative_ratio:.1%} of largest dimension "
            f"({max_dim:.6f}m). Current dimensions: {current_dimensions}. "
            f"Cannot uniformly scale mesh with such extreme proportions (threshold: "
            f"{relative_threshold:.0%}). This likely indicates a mesh generation failure "
            f"where the model produced near-2D geometry. Please regenerate the asset."
        )

    # Calculate uniform scale factor (median to match target dimensions).
    # Use median instead of mean for robustness to near-degenerate dimensions.
    desired_array = np.array(desired_dimensions)
    scale_factors = desired_array / current_dimensions
    uniform_scale = np.median(scale_factors)

    # Calculate actual resulting dimensions.
    actual_dimensions = current_dimensions * uniform_scale

    console_logger.info(
        f"Uniformly scaling mesh from {current_dimensions} to "
        f"{actual_dimensions} (requested: {desired_dimensions}, "
        f"scale factor: {uniform_scale:.3f})"
    )

    # Apply uniform scaling.
    mesh.apply_scale(uniform_scale)

    # Determine output path.
    final_output_path = output_path if output_path is not None else mesh_path

    # Ensure output directory exists.
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export scaled mesh.
    mesh.export(final_output_path)

    console_logger.info(f"Uniformly scaled mesh saved to {final_output_path}")

    return final_output_path, uniform_scale


def _compute_bbox_min_distance(bounds1: np.ndarray, bounds2: np.ndarray) -> float:
    """Compute minimum distance between two axis-aligned bounding boxes.

    Args:
        bounds1: First bounding box as [min, max] with shape (2, 3).
        bounds2: Second bounding box as [min, max] with shape (2, 3).

    Returns:
        Minimum distance between the two bounding boxes. Returns 0 if boxes
        overlap or touch.
    """
    # For each axis, compute the gap between the boxes.
    # If boxes overlap on an axis, gap is 0.
    gaps = np.zeros(3)
    for i in range(3):
        # Gap is the distance between the closest edges on this axis.
        gap = max(0, max(bounds1[0, i] - bounds2[1, i], bounds2[0, i] - bounds1[1, i]))
        gaps[i] = gap

    # Minimum distance is the Euclidean distance of the gaps.
    return np.linalg.norm(gaps)


def remove_mesh_floaters(
    mesh_path: Path, output_path: Path | None = None, distance_threshold: float = 0.05
) -> Path:
    """Remove disconnected mesh components (floaters) based on spatial distance.

    Splits the mesh into connected components and removes floaters that are
    spatially separated from the main mesh using a distance-based clustering
    algorithm. This approach correctly preserves small legitimate parts (handles,
    knobs) that are close to the main mesh while removing actual floaters that
    are far away, regardless of their size.

    Algorithm:
    1. Split mesh into connected components
    2. Find largest component by volume (seed for main cluster)
    3. Iteratively add components within distance_threshold to main cluster
    4. Remove all components not in the main cluster

    Args:
        mesh_path: Path to input mesh file (GLB, GLTF, OBJ, STL, etc.). Must exist.
        output_path: Optional output path for the cleaned mesh. If None, the
            input mesh will be overwritten. The format is inferred from the file
            extension.
        distance_threshold: Maximum distance (in meters) between bounding boxes
            for a component to be considered part of the main cluster. Components
            further than this distance from any component in the main cluster will
            be removed as floaters. Default is 0.05 (5cm). Set to very large value
            (e.g., 1000.0) to keep all components.

    Returns:
        Path to the cleaned mesh file.

    Raises:
        FileNotFoundError: If the input mesh file does not exist.
        ValueError: If the mesh cannot be loaded or contains no valid geometry.
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    console_logger.info(
        f"Removing mesh floaters (distance threshold={distance_threshold:.3f}m)"
    )

    # Load mesh and ensure it's a single Trimesh object.
    mesh = load_mesh_as_trimesh(mesh_path, force_merge=True)

    # Split mesh into connected components.
    components = mesh.split()

    console_logger.info(f"Found {len(components)} connected component(s)")

    # If only one component, no floaters to remove.
    if len(components) <= 1:
        console_logger.info("Single component mesh, no floaters to remove")
        final_output_path = output_path if output_path is not None else mesh_path
        if output_path is not None:
            mesh.export(final_output_path)
        return final_output_path

    # Calculate volumes for all components to find the largest (seed).
    volumes = np.array([comp.volume for comp in components])
    largest_idx = np.argmax(volumes)

    console_logger.info(
        f"Starting spatial clustering from largest component "
        f"(volume: {volumes[largest_idx]:.6f})"
    )

    # Initialize main cluster with largest component.
    main_cluster_indices = {largest_idx}
    remaining_indices = set(range(len(components))) - main_cluster_indices

    # Iteratively add components within distance threshold.
    changed = True
    while changed and remaining_indices:
        changed = False
        for idx in list(remaining_indices):
            comp_bounds = components[idx].bounds

            # Check distance to any component in main cluster.
            min_dist_to_cluster = float("inf")
            for cluster_idx in main_cluster_indices:
                cluster_bounds = components[cluster_idx].bounds
                dist = _compute_bbox_min_distance(comp_bounds, cluster_bounds)
                min_dist_to_cluster = min(min_dist_to_cluster, dist)

            # Add to cluster if within threshold.
            if min_dist_to_cluster <= distance_threshold:
                main_cluster_indices.add(idx)
                remaining_indices.remove(idx)
                changed = True
                console_logger.debug(
                    f"Added component {idx} to cluster "
                    f"(distance: {min_dist_to_cluster:.3f}m, "
                    f"volume: {volumes[idx]:.6f})"
                )

    # Build kept and removed component lists.
    kept_components = [components[i] for i in sorted(main_cluster_indices)]
    removed_indices = remaining_indices
    removed_count = len(removed_indices)
    removed_volume = sum(volumes[i] for i in removed_indices)

    console_logger.info(
        f"Keeping {len(kept_components)} component(s), "
        f"removing {removed_count} floater(s) "
        f"(total removed volume: {removed_volume:.6f})"
    )

    # Log details of removed floaters.
    for idx in sorted(removed_indices):
        console_logger.info(f"Removed floater {idx}: volume={volumes[idx]:.6f}")

    # Combine kept components.
    if len(kept_components) == 1:
        cleaned_mesh = kept_components[0]
    else:
        cleaned_mesh = trimesh.util.concatenate(kept_components)

    # Determine output path.
    final_output_path = output_path if output_path is not None else mesh_path

    # Ensure output directory exists.
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export cleaned mesh.
    cleaned_mesh.export(final_output_path)

    console_logger.info(f"Cleaned mesh saved to {final_output_path}")

    return final_output_path
