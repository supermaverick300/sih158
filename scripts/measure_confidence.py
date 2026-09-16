"""Measure existing reconstruction evidence without rerunning reconstruction."""
import argparse
import json
from pathlib import Path
import sys
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from drone3d_studio.domain.models import Frame
from drone3d_studio.persistence import store
from drone3d_studio.reconstruction.confidence import attach_run, ConfidenceConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path)
    parser.add_argument('--config',type=Path,help='JSON ConfidenceConfig overrides; effective configuration is saved in each report')
    args=parser.parse_args()
    config=ConfidenceConfig(**json.loads(args.config.read_text(encoding='utf-8'))) if args.config else ConfidenceConfig()
    root=args.project.resolve(); project=store.load(root); groups={}
    for model in project.models:
        if model.origin not in ('COLMAP','OpenMVS','Depth Anything V2','MiDaS estimated relief'):
            continue
        try:
            relative=store.resolve(root,model.path).relative_to(root/'reconstruction')
        except ValueError:
            continue
        if len(relative.parts)>=2:
            groups.setdefault(root/'reconstruction'/relative.parts[0],[]).append((model,None))
    if not groups:
        raise ValueError('No reconstruction workspace associated with project models')
    for work,result in groups.items():
        snapshot=work/'metadata/frame-evidence.json'
        frames=[Frame.model_validate(f) for f in json.loads(snapshot.read_text(encoding='utf-8'))] if snapshot.is_file() else []
        attach_run(root,work,frames,result,threading.Event(),print,config)
    store.save(root,project)


if __name__=='__main__':
    main()
