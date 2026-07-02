# Pull Request Summary

This PR fixes several Star Wars Outlaws `.mmb` mesh round-trip issues found while testing imported/exported assets in Blender and in-game.

## Fixes

- Detects physical weight/index slot counts from vertex stride instead of relying only on the declared logical weight count.
- Supports meshes that declare fewer weights than their physical storage layout provides.
- Detects packed `uint8_norm` weight layouts that were previously misdetected as `uint16_norm`.
- Preserves fixed weight/index slot alignment, including zero-weight slots.
- Fixes unsigned 16-bit normalized scaling to use the full `0..65535` range.
- Preserves duplicate triangle records during import by using `Mesh.from_pydata()` for initial mesh creation.
- Avoids `AssetPath` polling errors in fresh Blender scenes.

## Tested cases

- Mesh declaring 6 weights but physically storing 8 `uint8_norm` weight/index slots: now imports, exports, and works in-game.
- Mesh guessed as 2 `uint16_norm` weights but physically storing 4 `uint8_norm` weight/index slots: now imports, exports, and works in-game.
- Previously working body mesh still imports, exports, and works in-game.
- Mesh with duplicate triangle records now imports instead of failing with `faces.new(...): face already exists`.
