"""Audit real C2DB MoS2 and a shared random-weight SG/LG forward pass; no training."""
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
import spglib
import torch
import yaml
from pymatgen.core import Lattice, Structure
from scipy.linalg import null_space

from unifiedgroupgen.pl_data.encoding import convert_row, identified_layer_record, operation_residual, parse_cif
from unifiedgroupgen.pl_modules.orbitflow import OrbitFlow
from unifiedgroupgen.pl_modules.objectives.flow_matching import conditional_path
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor, group_table


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    out = Path(__file__).resolve().parent
    prepared = resolve_path(ROOT / "artifacts/full_official_tol0p1_v7/c2db_51/val.jsonl")
    with prepared.open(encoding="utf-8") as handle:
        line_number, record = next((i, json.loads(line)) for i, line in enumerate(handle, 1)
                                   if json.loads(line)["id"] == "1MoS2-1")
    source = resolve_path(ROOT / "artifacts/source_recovery/c2db_official/val.csv")
    with source.open(encoding="utf-8", newline="") as handle:
        index, row = next((i, row) for i, row in enumerate(csv.DictReader(handle))
                          if i == record["provenance"]["source_row"])
    assert hashlib.sha256(row["cif"].encode()).hexdigest() == record["provenance"]["source_cif_sha256"]
    config = yaml.safe_load((resolve_path(ROOT / "configs/c2db_official_local.yaml")).read_text(encoding="utf-8"))
    converted = convert_row((index, row, config, str(source)))
    assert "record" in converted, converted
    converted = json.loads(json.dumps(converted["record"]))
    assert converted["descriptor"] == record["descriptor"]
    assert np.allclose(converted["state"], record["state"], atol=1e-10, rtol=0)

    raw = parse_cif(row["cif"], preserve_fractional=True)
    c = compile_descriptor(Descriptor.from_dict(record["descriptor"]))
    state = torch.tensor(record["state"], dtype=torch.float64)
    expanded = c.expand(state)
    coords, lattice = expanded["coordinates"].numpy(), expanded["lattice"].numpy()
    group = group_table("layer", c.descriptor.number)
    operations = group[0].ops
    constraint = c.constraint_matrix(coords)
    basis = c.coordinate_basis.reshape(-1, c.coordinate_dof)
    kernel = null_space(constraint)

    def closure(positions):
        return float(operation_residual(positions, c.elements, operations, c.periodic, lattice))

    def export_structure(positions, box_height=20.0):
        rows = lattice.T.copy()
        rows[2] = [0, 0, box_height]
        cart = positions @ lattice.T
        cart[:, 2] += box_height / 2
        return Structure(Lattice(rows, pbc=(True, True, False)), c.elements,
                         cart, coords_are_cartesian=True, to_unit_cell=False)

    def identify(positions):
        structure = export_structure(positions)
        found = spglib.get_layergroup((structure.lattice.matrix, structure.frac_coords,
                                      structure.atomic_numbers), aperiodic_dir=2, symprec=1e-5)
        return int(found.number) if found is not None else None

    changed = state.clone()
    changed[-1] += 0.2
    changed_coords = c.expand(changed)["coordinates"].numpy()
    broken_coords = coords.copy()
    broken_coords[1, 2] += 0.2
    vacuum_checks = []
    for height in (20.0, 30.0):
        descriptor, reencoded, provenance = identified_layer_record(export_structure(coords, height), 0.1, 78)
        assert descriptor.to_dict() == c.descriptor.to_dict()
        difference = float(np.max(np.abs(reencoded - state.numpy())))
        assert difference < 1e-10
        vacuum_checks.append({"box_height_angstrom": height, "state_max_difference": difference})

    orbit_reports = []
    for orbit_index, spec in enumerate(c.descriptor.orbits):
        ids = np.flatnonzero(c.atom_orbit == orbit_index)
        wp = next(w for w in group if w.letter == spec.letter)
        representative = coords[ids[0]]
        stabilizer = []
        for operation in operations:
            delta = operation.operate(representative) - representative
            delta[c.periodic] -= np.round(delta[c.periodic])
            if np.linalg.norm(delta @ lattice.T) < 1e-8:
                stabilizer.append(operation.as_xyz_str())
        orbit_reports.append({"atomic_number": spec.atomic_number, "wyckoff": f"{len(ids)}{spec.letter}",
                              "maps": [op.as_xyz_str() for op in wp.ops],
                              "stabilizer": stabilizer, "dof": c.blocks[orbit_index].shape[-1]})

    mp = json.loads((resolve_path(ROOT / "examples/mp20_walkthrough/record.json")).read_text(encoding="utf-8"))
    mc = compile_descriptor(Descriptor.from_dict(mp["descriptor"]))
    ms = torch.tensor(mp["state"], dtype=torch.float64)
    joint = yaml.safe_load((resolve_path(ROOT / "configs/joint_space_layer_large.yaml")).read_text(encoding="utf-8"))
    settings = joint["model"]
    model = OrbitFlow(joint["properties"], hidden=settings["flow_hidden"],
                      layers=settings["flow_layers"], frequencies=settings["frequencies"],
                      cond_dropout=joint["cond_dropout"], checkpoint_layers=False,
                      edge_chunk_size=settings["edge_chunk_size"]).double().eval()
    conditions = [dict(mp["properties"], source_domain=0), dict(record["properties"], source_domain=2)]
    candidates = {}
    hook = model.coordinate_head.register_forward_hook(
        lambda module, inputs, output: candidates.update(value=output.detach()))
    with torch.no_grad():
        predictions = model.forward_batch([mc, c], [ms, state], [0.4, 0.4], conditions)
    hook.remove()
    network_checks = []
    offset = 0
    for compiled, z, prediction in zip((mc, c), (ms, state), predictions):
        matrix = compiled.constraint_matrix(compiled.expand(z)["coordinates"].numpy())
        raw_velocity = candidates["value"][offset:offset+compiled.num_atoms].numpy().reshape(-1)
        legal_velocity = compiled.coordinate_velocity(prediction[compiled.lattice_dof:]).numpy().reshape(-1)
        residual = float(np.max(np.abs(matrix @ legal_velocity)))
        assert residual < 1e-10
        network_checks.append({"kind": compiled.descriptor.kind, "group": compiled.descriptor.number,
                               "input_dimension": len(z), "output_dimension": len(prediction),
                               "group_embedding_id": model.group_id(compiled.descriptor),
                               "nodes": compiled.num_atoms, "edges": compiled.edge_index.shape[1],
                               "candidate_residual": float(np.max(np.abs(matrix @ raw_velocity))),
                               "projected_residual": residual})
        offset += compiled.num_atoms
    path, velocity = conditional_path(c, state, 0.4, generator=torch.Generator().manual_seed(42))
    report = {
        "scope": "Real local C2DB encoding and shared random-weight forward audit; not trained generation or stability evidence.",
        "id": record["id"], "split": "val", "prepared_line_one_based": line_number,
        "source_row_zero_based": index, "provenance": record["provenance"],
        "layer_group": {"number": c.descriptor.number, "symbol": group.symbol},
        "operations": [op.as_xyz_str() for op in operations], "orbits": orbit_reports,
        "raw_lattice_rows": raw.lattice.matrix.tolist(), "raw_fractional_coordinates": raw.frac_coords.tolist(),
        "coordinates_xy_fractional_z_angstrom": coords.tolist(), "physical_conversion_matrix": lattice.tolist(),
        "inplane_metric": (lattice.T @ lattice)[:2, :2].tolist(),
        "metric_reference": (c.root @ c.root).tolist(), "metric_basis": c.metric_basis.tolist(),
        "coordinate_basis": c.coordinate_basis.tolist(), "coordinate_offsets": c.b.tolist(),
        "lattice_dof": c.lattice_dof, "coordinate_dof": c.coordinate_dof,
        "state": state.tolist(), "coordinate_periodic_mask": c.periodic.tolist(),
        "state_periodic_mask": c.state_periodic.tolist(), "thickness_angstrom": float(np.ptp(coords[:, 2])),
        "constraint_shape": list(constraint.shape), "constraint_rank": int(np.linalg.matrix_rank(constraint)),
        "constraint_nullity": kernel.shape[1], "constraint_times_basis_max": float(np.max(np.abs(constraint @ basis))),
        "nullspace_projector_difference": float(np.max(np.abs(basis @ np.linalg.pinv(basis)-kernel @ kernel.T))),
        "closure_residual_angstrom": closure(coords), "reidentified_layer_group": identify(coords),
        "reconversion_state_max_difference": float(np.max(np.abs(np.array(converted["state"])-state.numpy()))),
        "h_plus_0p2": {"coordinates": changed_coords.tolist(), "closure_residual_angstrom": closure(changed_coords),
                        "layer_group": identify(changed_coords)},
        "one_sulfur_up_0p2": {"closure_residual_angstrom": closure(broken_coords), "layer_group": identify(broken_coords)},
        "vacuum_invariance": vacuum_checks,
        "path_at_t0p4_seed42": {"state": path.tolist(), "target_velocity": velocity.tolist()},
        "joint_network": {"trained": False, "parameters": sum(p.numel() for p in model.parameters()),
                          "hidden": settings["flow_hidden"], "layers": settings["flow_layers"],
                          "checks": network_checks},
    }
    assert report["constraint_nullity"] == 1
    assert report["closure_residual_angstrom"] < 1e-10
    assert report["nullspace_projector_difference"] < 1e-10
    assert report["reidentified_layer_group"] == report["h_plus_0p2"]["layer_group"] == 78
    (out / "source.cif").write_text(row["cif"], encoding="utf-8")
    (out / "reconstructed_vacuum20A.cif").write_text(export_structure(coords).to(fmt="cif"), encoding="utf-8")
    (out / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("provenance", "raw_lattice_rows", "coordinate_basis")},
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
