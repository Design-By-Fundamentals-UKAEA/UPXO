"""Run the bundled sample: python -m upxo.meshing.gbconformant.d3v2p0."""
import argparse
from pathlib import Path
import json
import numpy as np
from .mesh import mesh_voxels
from .paths import sample_path, run_directory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', nargs='?', type=Path, default=sample_path())
parser.add_argument('--output', type=Path, default=run_directory('cli')/'blk_mesh')
parser.add_argument('--iterations', type=int, default=20)
parser.add_argument('--spacing', type=float, default=1.)
parser.add_argument('--min-quality', type=float, default=0.5)
parser.add_argument('--quality-iterations', type=int, default=60)
args = parser.parse_args()
mesh = mesh_voxels(np.load(args.input, allow_pickle=False), spacing=args.spacing,
                   iterations=args.iterations, min_quality=args.min_quality,
                   quality_iterations=args.quality_iterations)
print(json.dumps(mesh.save(args.output), indent=2))
