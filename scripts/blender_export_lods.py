"""Run inside Blender 4.5+: import textured OBJ and export UV-preserving LODs.

Blender uses its bundled Python; do not import this into the desktop app.
"""
import json
import sys
from pathlib import Path
import bpy


def main():
    source, destination = map(Path, sys.argv[sys.argv.index('--') + 1:])
    source, destination = source.resolve(), destination.resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Preserve the source coordinate frame (our viewer uses Z up).
    bpy.ops.wm.obj_import(filepath=str(source), forward_axis='Y', up_axis='Z')
    originals = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not originals or not sum(len(obj.data.polygons) for obj in originals):
        raise RuntimeError('OBJ contains no mesh faces')
    images = [image for image in bpy.data.images if image.source == 'FILE']
    if not images:
        raise RuntimeError('OBJ has no atlas images')
    for image in images:
        if not Path(bpy.path.abspath(image.filepath)).is_file():
            raise RuntimeError('Missing atlas: ' + image.filepath)
        image.pack()
    for obj in originals:
        if not obj.data.uv_layers:
            raise RuntimeError('Mesh has no UV coordinates')
        bpy.context.view_layer.objects.active = obj
        modifier = obj.modifiers.new('Triangulate', 'TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    total = sum(len(obj.data.polygons) for obj in originals)
    report = {'source': str(source), 'source_faces': total, 'lods': {}}
    for level, target in [('high', total), ('medium', min(total, max(1, total // 2), 100000)), ('low', min(total, 10000))]:
        bpy.ops.object.select_all(action='DESELECT')
        copies = []
        for original in originals:
            obj = original.copy(); obj.data = original.data.copy()
            bpy.context.collection.objects.link(obj)
            obj.select_set(True); bpy.context.view_layer.objects.active = obj
            if target < total:
                modifier = obj.modifiers.new('LOD reduction', 'DECIMATE')
                modifier.ratio = target / total
                modifier.use_collapse_triangulate = True
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            copies.append(obj)
        count = sum(len(obj.data.polygons) for obj in copies)
        if count == 0:
            raise RuntimeError('LOD simplification removed all geometry')
        path = destination / f'scene_{level}.glb'
        bpy.ops.export_scene.gltf(filepath=str(path), export_format='GLB', use_selection=True,
                                  export_yup=False, export_texcoords=True, export_materials='EXPORT')
        report['lods'][level] = {'path': str(path), 'faces': count, 'bytes': path.stat().st_size}
        for obj in copies:
            bpy.data.objects.remove(obj, do_unlink=True)
    (destination / 'lod-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
