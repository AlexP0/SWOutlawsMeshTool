# Changelog

## 0.0.10

### Fixed

#### Mesh-local uint8 bone index layouts

Some skeletal meshes use mesh-local bone indices stored as `uint8`, even when the
full skeleton contains more than 255 bones.

The importer previously inferred `uint16` bone indices from the skeleton size,
which caused certain meshes (for example `chain_mesh`) to be decoded incorrectly,
producing invalid bone references and unusable imported geometry.

The importer now detects when a declared `uint16` index layout cannot physically
fit within the vertex stride and automatically falls back to a packed:

- `uint8_norm` weights
- `uint8` mesh-local bone indices

layout when appropriate.

#### Export round-trip support

The exporter now writes bone indices using the physical storage layout returned by
`get_vertex_weight_storage_layout()` rather than relying solely on the global mesh
index type.

This fixes round-trip import/export for meshes using mesh-local `uint8` indices
and prevents vertex-buffer corruption caused by writing `uint16` indices into
`uint8` layouts.

#### Vertex stride validation

Added export-time validation to detect vertex-buffer layout mismatches.

The exporter now raises an error if a vertex write exceeds the declared vertex
stride, preventing silent corruption of subsequent mesh data.

## 0.0.9

### Fixed

- Fixed triangle index over-read on tiny/proxy `*_CLOTH_RENDER` meshes.
  - These meshes can declare `Index Count: 3` but also have a `size_a / 4` fallback value that previously caused the importer to read too many triangles.
  - The importer now prefers the declared index count unless the fallback is proven to fit available face data.
- Fixed `struct.error: unpack requires a buffer of 2 bytes` when importing tiny `*_CLOTH_RENDER` proxy meshes.

### Changed

- Added an informational popup when importing extremely small/proxy meshes with only up to 3 vertices and 3 indices.
- The popup explains that the mesh is likely a proxy/helper/render marker mesh rather than a normal editable character or clothing mesh.
- The same warning is also printed to the console.

### Tested

- `jacket_CLOTH_RENDER` LODs now import without crashing.
- The imported result is a tiny triangle, which matches the declared 3-vertex / 3-index proxy mesh data.

## 0.0.8

### Fixed

- Fixed import of rigid/accessory meshes that use a compact 16-byte vertex layout with `float3` positions and 4 trailing bytes.
  - Detected layout:
    ```text
    vertex_stride == 16
    weight_count == 1
    binding_count == 1
    normal_type == float
    ```
  - These meshes are now treated as:
    ```text
    offset 00-11: float3 position
    offset 12-15: trailing/unknown bytes
    rigid binding: mesh-local bone 0, weight 1.0
    ```
- Fixed rigid/accessory meshes importing as an unusable dot/blob in Blender due to being misclassified as `int16_norm + scale` position layouts.
- Fixed noisy and misleading `Bone index out of MeshBone range` messages for rigid single-binding meshes by no longer reading arbitrary non-weight vertex bytes as bone indices.
- Suppressed misleading `uint16_norm` / 3-byte-per-weight-slot warnings for known rigid single-binding float3 layouts.

### Changed

- Added clearer console output separators for mesh Load and Import operations.
- Load output now starts each mesh block with the mesh name and mesh index before offset/stride details.
- Import output now prints a clear header/footer and key mesh/LOD layout information before import processing.
- Rigid single-binding layouts now report their physical weight storage type as `rigid_single_binding`.

### Tested

The following rigid/accessory meshes were verified to import correctly and round-trip with no-change export in-game:

- `metal_mesh`
- `wirepouch_mesh`
- `led1_mesh`
- `led2_mesh`
- `screen_mesh`

## 0.0.7

### Fixed

- Fixed import/export of meshes where the declared logical weight count is smaller than the physical weight/index slot count stored in the vertex stride.
  - Example observed layout: declared 6 weights, but physically stored 8 `uint8_norm` weights and 8 `uint8` indices.
- Fixed import/export of packed `uint8_norm` weight layouts that could previously be misdetected as `uint16_norm` due to ambiguous stride math.
  - Example observed layout: declared/guessed 2 `uint16_norm` weights, but physically stored 4 `uint8_norm` weights and 4 `uint8` indices.
- Preserved zero-weight slots while reading weight/index data so later non-zero weights are not paired with the wrong bone index.
- Made unsigned 16-bit normalized read/write handling consistent with the full `0..65535` range.
- Fixed importing meshes containing duplicate triangle records by constructing the initial Blender mesh with `Mesh.from_pydata()` instead of `bmesh.faces.new()`.
- Avoided fresh-scene `AssetPath` polling errors before an asset path has been selected.
- Improved add-on unregistration cleanup by removing the custom `Scene.SWOMT` property.

### Changed

- Mesh load output now reports physical weight storage details in addition to the declared weight count.

### Tested

The following observed round-trip failures were used to validate the fixes:

- `shoes_mesh`: declared 6 weights, physically stored 8 `uint8_norm` weight/index slots.
- `nails_mesh`: guessed as 2 `uint16_norm` weights, physically stored 4 `uint8_norm` weight/index slots.
- `pants_mesh`: duplicate triangle import case.
- `body_mesh`: regression test.

## 0.0.6

- Upstream baseline version before the compatibility fixes above.
