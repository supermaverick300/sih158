"""Reuse an existing stereo workspace to build the new open visible surface."""
import argparse
from pathlib import Path
import signal
import sys
import threading
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from drone3d_studio.domain.models import PipelineConfig
from drone3d_studio.persistence import store
from drone3d_studio.reconstruction.colmap import detect,run_command
from drone3d_studio.reconstruction.visible_surface import from_stereo
from drone3d_studio.services.meshes import model_record,transfer_vertex_colors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("dense_workspace",type=Path)
    parser.add_argument("--colmap",default="")
    parser.add_argument("--stride",type=int,default=4)
    args=parser.parse_args()
    root=args.project.resolve()
    project=store.load(root)
    dense=args.dense_workspace.resolve()
    work=root/"reconstruction"/("visible-"+uuid4().hex[:12])
    work.mkdir(parents=True)
    text=work/"text-model"
    text.mkdir()
    cancel=threading.Event()
    signal.signal(signal.SIGINT,lambda *_:cancel.set())
    config=PipelineConfig(capture_mode="Single pass",depth_stride=args.stride)
    exe=detect(args.colmap or project.settings.executable)
    def log(message):
        print(message,flush=True)
    run_command([exe,"model_converter","--input_path",str(dense/"sparse"),"--output_path",str(text),"--output_type","TXT"],cancel,log,work)
    output=from_stereo(dense,text,config,cancel,lambda _,s:log(s),work/"visible-surface.ply")
    record,mesh=model_record(root,output,"COLMAP")
    full_faces=record.faces
    if full_faces>12000:
        preview=work/"inspection-mesh.ply"
        run_command([exe,"mesh_simplifier","--input_path",str(output),"--output_path",str(preview),"--MeshSimplification.target_face_ratio",str(12000/full_faces)],cancel,log,work)
        preview_record,preview_mesh=model_record(root,preview,"COLMAP")
        transfer_vertex_colors(mesh,preview_mesh,cancel)
        preview_mesh.export(str(preview))
        record=preview_record
    record.name="Visible surface from existing pass (preview)"
    for old in project.models:
        old.visible=False
    project.models.append(record)
    project.pipeline=config
    project.status="Scene ready"
    project.reconstruction_status="Visible surface rebuilt from existing camera poses/depth; georeference unverified"
    store.save(root,project)
    log(f"Full surface: {output}; {full_faces:,} faces")


if __name__=="__main__":
    main()
