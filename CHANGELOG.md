# Changelog

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

## 0.0.6

- Upstream baseline version before these compatibility fixes.
