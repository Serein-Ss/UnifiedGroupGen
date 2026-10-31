"""Demonstrate one MP20 encoding with the current compiler; no training is run."""
from unifiedgroupgen.common.configuration import project_root, resolve_path
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment

from unifiedgroupgen.pl_data.encoding import convert_row, parse_cif
from unifiedgroupgen.pl_modules.orbitflow import OrbitFlow
from unifiedgroupgen.pl_modules.objectives.flow_matching import conditional_path
from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor, group_table


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    out = Path(__file__).resolve().parent
    prepared = resolve_path(ROOT / "artifacts/full/mp_20/train.jsonl")
    with prepared.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record["id"] == "mp-2657":
                break
        else:
            raise ValueError("mp-2657 is not in the prepared MP20 training set")
    source = resolve_path(ROOT.parent / "data/mp_20/train.csv")
    with source.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index == record["provenance"]["source_row"]:
                break
    assert row["material_id"] == record["id"]
    source_hash = hashlib.sha256(row["cif"].encode()).hexdigest()
    assert source_hash == record["provenance"]["source_cif_sha256"]
    config = yaml.safe_load((resolve_path(ROOT / "configs/mp20_local.yaml")).read_text(encoding="utf-8"))
    reconverted = convert_row((index, row, config, str(source)))
    assert "record" in reconverted, reconverted
    new_record = json.loads(json.dumps(reconverted["record"]))
    assert new_record["descriptor"] == record["descriptor"]

    c = compile_descriptor(Descriptor.from_dict(record["descriptor"]))
    state = torch.tensor(record["state"], dtype=torch.float64)
    expanded = c.expand(state)
    coords = expanded["coordinates"].numpy()
    lattice = expanded["lattice"].numpy()
    metric = lattice.T @ lattice
    raw = parse_cif(row["cif"])
    rebuilt = Structure(Lattice(lattice.T), c.elements, coords)
    operations = group_table("space", c.descriptor.number)[0].ops
    constraint = c.constraint_matrix(coords)
    basis = c.coordinate_basis.reshape(3*c.num_atoms, c.coordinate_dof)
    kernel = null_space(constraint)
    basis_projector = basis @ np.linalg.pinv(basis)

    def closure(frac):
        maximum = 0.0
        for op in operations:
            mapped = frac @ op.rotation_matrix.T + op.translation_vector
            for element in np.unique(c.elements):
                ids = np.flatnonzero(c.elements == element)
                delta = mapped[ids, None] - frac[None, ids]
                delta -= np.round(delta)
                costs = np.linalg.norm(delta @ lattice.T, axis=-1)
                ii, jj = linear_sum_assignment(costs)
                maximum = max(maximum, float(costs[ii, jj].max()))
        return maximum

    changed_state = state.clone()
    changed_state[-1] += 0.02
    changed = c.expand(changed_state)["coordinates"].numpy()
    broken = coords.copy()
    broken[2, 0] += 0.02
    matcher = StructureMatcher(ltol=0.01, stol=0.01, angle_tol=1, scale=False)
    orbit_reports = []
    for number, (spec, a) in enumerate(zip(c.descriptor.orbits, c.blocks)):
        wp = next(w for w in group_table("space", c.descriptor.number) if w.letter == spec.letter)
        ids = np.flatnonzero(c.atom_orbit == number)
        b = c.b[ids]
        orbit_reports.append({
            "element": str(rebuilt[ids[0]].specie), "wyckoff": f"{len(ids)}{spec.letter}",
            "dof": int(a.shape[-1]), "atom_indices": ids.tolist(),
            "representative": coords[ids[0]].tolist(),
            "stabilizer_order": int(sum(
                np.linalg.norm((op.operate(coords[ids[0]])-coords[ids[0]]) -
                               np.round(op.operate(coords[ids[0]])-coords[ids[0]])) < 1e-8
                for op in operations)),
            "affine_maps": [op.as_xyz_str() for op in wp.ops],
            "basis": a.tolist(), "offsets": b.tolist(),
        })

    generator = torch.Generator().manual_seed(42)
    prior = c.prior(dtype=torch.float64, generator=generator)
    path_state, target = conditional_path(
        c, state, 0.4, generator=torch.Generator().manual_seed(42))
    path_checks = []
    for time in (0.0, 0.25, 0.5, 0.75, 1.0):
        zt, vt = conditional_path(c, state, time, generator=torch.Generator().manual_seed(42))
        geom = c.expand(zt)
        path_checks.append({
            "t": time,
            "group_residual_fractional": float(np.max(np.abs(
                c.constraint_matrix(geom["coordinates"].numpy()) @
                c.coordinate_velocity(vt[c.lattice_dof:]).numpy().reshape(-1)))),
            "minimum_metric_eigenvalue": float(torch.linalg.eigvalsh(geom["lattice"].T @ geom["lattice"]).min()),
        })

    # Same physical cell under the weaker P1 requirement: an explanatory second
    # description, not a relabeled training example and not a new data point.
    p1 = compile_descriptor(Descriptor("space", 1, tuple(OrbitSpec("a", int(z)) for z in c.elements)))
    p1_state = torch.tensor(np.r_[p1.encode_metric(metric), coords.reshape(-1)], dtype=torch.float64)
    model_cfg = config["model"]
    model = OrbitFlow(config["properties"], hidden=model_cfg["flow_hidden"],
                      layers=model_cfg["flow_layers"], frequencies=model_cfg["frequencies"],
                      cond_dropout=config["cond_dropout"], checkpoint_layers=False,
                      edge_chunk_size=model_cfg["edge_chunk_size"]).double().eval()
    captured = {}
    hook = model.coordinate_head.register_forward_hook(
        lambda module, inputs, output: captured.update(candidates=output.detach()))
    with torch.no_grad():
        predictions = model.forward_batch([c, p1], [state, p1_state], [0.4, 0.4],
                                          [record["properties"], record["properties"]])
    hook.remove()
    candidate = captured["candidates"][:c.num_atoms]
    projected = c.coordinate_velocity(predictions[0][c.lattice_dof:])
    p1_match = torch.max(torch.abs(p1.expand(p1_state)["coordinates"]-expanded["coordinates"])).item()

    report = {
        "scope": "Local MP20 dataset encoding and symmetry demonstration; random-weight forward pass, not trained generation.",
        "material_id": record["id"], "formula": rebuilt.composition.reduced_formula,
        "source_csv": source.relative_to(ROOT.parent).as_posix(), "source_row_zero_based": index,
        "prepared_file": prepared.relative_to(ROOT).as_posix(), "prepared_line_one_based": line_number,
        "source_cif_sha256": source_hash,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "source_record_properties": record["properties"],
        "raw_lattice_rows": raw.lattice.matrix.tolist(),
        "raw_fractional_coordinates": raw.frac_coords.tolist(), "raw_elements": [str(s.specie) for s in raw],
        "lattice_columns": lattice.tolist(), "abc_angstrom": list(rebuilt.lattice.abc),
        "angles_degrees": list(rebuilt.lattice.angles), "volume_angstrom3": rebuilt.volume,
        "expanded_coordinates": coords.tolist(), "expanded_elements": [str(s.specie) for s in rebuilt],
        "space_group": {"number": SpacegroupAnalyzer(rebuilt, symprec=0.001).get_space_group_number(),
                        "symbol": SpacegroupAnalyzer(rebuilt, symprec=0.001).get_space_group_symbol()},
        "operation_count_modulo_cell_translations": len(operations),
        "operations": [{"xyz": op.as_xyz_str(), "R": op.rotation_matrix.tolist(),
                        "t": op.translation_vector.tolist()} for op in operations],
        "orbits": orbit_reports,
        "lattice_dof": c.lattice_dof, "coordinate_dof": c.coordinate_dof,
        "total_continuous_dof": c.dof, "state": state.tolist(),
        "metric_reference": (c.root @ c.root).tolist(), "metric_basis": c.metric_basis.tolist(),
        "coordinate_basis": c.coordinate_basis.tolist(), "coordinate_offsets": c.b.tolist(),
        "constraint_matrix_shape": list(constraint.shape), "constraint_matrix_rank": int(np.linalg.matrix_rank(constraint)),
        "constraint_nullity": int(kernel.shape[1]),
        "coordinate_basis_rank": int(np.linalg.matrix_rank(basis)),
        "constraint_times_basis_max": float(np.max(np.abs(constraint @ basis))),
        "nullspace_projector_difference_max": float(np.max(np.abs(basis_projector-kernel@kernel.T))),
        "metric_symmetry_residual_max": max(float(np.max(np.abs(op.rotation_matrix.T@metric@op.rotation_matrix-metric))) for op in operations),
        "operation_closure_residual_angstrom": closure(coords),
        "reconversion_state_max_abs_difference": float(np.max(np.abs(np.array(new_record["state"])-state.numpy()))),
        "source_reconstruction_structure_match": bool(matcher.fit(raw, rebuilt)),
        "u_plus_0p02": {"coordinates": changed.tolist(), "closure_residual_angstrom": closure(changed),
                       "reidentified_group": SpacegroupAnalyzer(
                           Structure(Lattice(lattice.T), c.elements, changed), symprec=0.001).get_space_group_number()},
        "one_oxygen_x_plus_0p02": {"closure_residual_angstrom": closure(broken),
                                 "reidentified_group": SpacegroupAnalyzer(
                                     Structure(Lattice(lattice.T), c.elements, broken), symprec=0.001).get_space_group_number()},
        "training_path_example": {"seed": 42, "t": 0.4, "prior": prior.tolist(),
                                  "state_at_t": path_state.tolist(), "target_velocity": target.tolist(),
                                  "checks": path_checks},
        "network_demo": {"trained": False, "dtype": "float64", "device": "cpu",
                         "parameters": sum(p.numel() for p in model.parameters()),
                         "hidden": model_cfg["flow_hidden"], "layers": model_cfg["flow_layers"],
                         "frequencies": model_cfg["frequencies"], "edge_feature_dimension": 6*model_cfg["frequencies"]+12,
                         "nodes_per_structure": c.num_atoms, "directed_edges_per_structure": c.num_atoms*(c.num_atoms-1),
                         "input_state_dimensions": [c.dof, p1.dof], "output_velocity_dimensions": [len(v) for v in predictions],
                         "candidate_constraint_residual": float(np.max(np.abs(constraint@candidate.numpy().reshape(-1)))),
                         "projected_constraint_residual": float(np.max(np.abs(constraint@projected.numpy().reshape(-1)))),
                         "p1_same_cell_coordinate_difference": p1_match},
    }
    assert report["space_group"]["number"] == 136
    assert report["constraint_nullity"] == c.coordinate_dof == 1
    assert report["source_reconstruction_structure_match"]
    assert report["operation_closure_residual_angstrom"] < 1e-10
    assert report["nullspace_projector_difference_max"] < 1e-10
    (out / "source.cif").write_text(row["cif"], encoding="utf-8")
    (out / "reconstructed.cif").write_text(rebuilt.to(fmt="cif"), encoding="utf-8")
    (out / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    keys = ("material_id", "space_group", "abc_angstrom", "expanded_coordinates",
            "orbits", "state", "metric_basis", "constraint_matrix_shape", "constraint_matrix_rank",
            "constraint_nullity", "nullspace_projector_difference_max",
            "metric_symmetry_residual_max", "operation_closure_residual_angstrom",
            "reconversion_state_max_abs_difference", "source_reconstruction_structure_match",
            "u_plus_0p02", "one_oxygen_x_plus_0p02", "training_path_example", "network_demo")
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
