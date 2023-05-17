import aspose.threed as a3d

scene = a3d.Scene.from_file("mesh.ply")
scene.save("mesh.glb")