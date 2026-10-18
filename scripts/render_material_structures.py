"""Render the existing material examples and saved outputs in one visual style."""
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.structure_rendering import render_structure
from unifiedgroupgen.representation_audit import compiled_chart, expand_numpy
from unifiedgroupgen.symmetry import Descriptor


def main():
    from pymatgen.core import Structure
    output = ROOT/'assets/structures'; output.mkdir(exist_ok=True)
    examples = ROOT/'assets/paper/source/material_examples.json'
    records = []
    for item in json.loads(examples.read_text(encoding='utf-8'))['examples']:
        chart = compiled_chart(Descriptor.from_dict(item['descriptor']))
        lattice, coordinates, _ = expand_numpy(chart, item['state'])
        records.append((item['id'], examples, lattice.T, coordinates, chart.elements,
                        tuple(chart.periodic), 'Real encoded dataset example'))
    for source in sorted((ROOT/'assets/videos/model_walkthrough/outputs').glob('*.cif')):
        structure = Structure.from_file(source)
        periodic = (True, True, source.stem.endswith('space'))
        records.append((source.stem, source, structure.lattice.matrix, structure.frac_coords,
                        structure.atomic_numbers, periodic, 'Saved model output; no new sampling'))
    manifest = {'style': 'User reference matched / transparent PNG / VESTA / matte spheres / coordination hulls / axis compass',
                'reference': 'https://github.com/materialsproject/crystaltoolkit/blob/main/crystal_toolkit/components/structure.py',
                'bond_display': 'Current-geometry distance cutoff: 1.15 times covalent-radius sum; omit metal-metal contacts in mixed metal/nonmetal compounds; visual contacts, not chemical certification',
                'renderer_sha256': hashlib.sha256((ROOT/'src/unifiedgroupgen/structure_rendering.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                'figures': []}
    for name, source, lattice, coordinates, elements, periodic, scope in records:
        layer = not periodic[2]; repeat = (2, 2, 1) if layer else (1, 1, 1)
        for view in ('isometric', 'top', 'side') if layer else ('isometric',):
            image = render_structure(lattice, coordinates, elements, periodic=periodic,
                                     repeat=repeat, size=(900, 900), view=view, background=None)
            filename = f'{name}_{view}.png'; image.save(output/filename)
            manifest['figures'].append({'file': filename, 'scope': scope,
                'source': source.relative_to(ROOT).as_posix(),
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'periodic': [bool(v) for v in periodic], 'display_repeat': repeat,
                'independent_cell_atoms': len(elements), 'view': view})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Rendered {len(manifest["figures"])} PNG views from unchanged structures.')


if __name__ == '__main__':
    main()
