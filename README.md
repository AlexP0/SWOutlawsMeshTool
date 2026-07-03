# Star Wars Outlaws Mesh Tool

Blender add-on for importing and exporting skeletal mesh data from Star Wars Outlaws `.mmb` files.

The tool is intended for modding workflows where `.mmb` files have been extracted/exported with tools such as Hunter and then edited in Blender.

## Features

* Load Star Wars Outlaws skeletal mesh `.mmb` assets.
* Import individual mesh LODs into Blender.
* Export edited LODs back into the `.mmb` asset.
* Preserve unedited meshes and LODs while rebuilding the edited asset.
* Import/export bone weights, UVs, vertex colors, normals, tangents, and triangle indices for supported layouts.

## Installation

1. Click the green **Code** button on GitHub and choose **Download ZIP**, or download a release ZIP if one is available.
2. In Blender, open **Edit > Preferences > Add-ons**.
3. Click the install button / arrow menu and choose **Install from Disk...**.
4. Select the downloaded ZIP file.
5. Enable **Star Wars Outlaws Mesh Tool**.

## Usage

1. Open Blender.
2. Go to the **Scene Properties** tab.
3. Find the **Star Wars: Outlaws Mesh Tool** panel.
4. Select an `.mmb` asset. Split assets are supported by selecting the `.mmb` or `.mmb\_0` file, depending on the extracted file set.
5. Click **Load**.
6. Click **Import** on the mesh/LOD you want to edit.
7. Edit the imported mesh in Blender.
8. Click **Export** on the corresponding mesh/LOD to write the edited data back.

## Notes and limitations

This add-on is based on reverse-engineering and format heuristics. Star Wars Outlaws mesh assets can use several vertex-layout variants, and not all variants may be fully supported.



