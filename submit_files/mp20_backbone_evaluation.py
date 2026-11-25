"""Submit and collect the MP-20 backbone and all-space-group evaluations on TH-HPC6."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from types import SimpleNamespace

import yaml

MODEL_BY_BACKBONE = {"eqv2": "equiformer_v2", "cspnet": "cspnet"}
MODES = ("number", "operations")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/"configs/evaluation/mp20_backbones_space230.yaml"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def model_name(backbone, task):
    return f"full_mp20_{backbone}_{task}_improved"


def group_requests(records, settings, prepare_manifest):
    from unifiedgroupgen.group_constraints.compiler import catalog_operations
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.common.csp_cases import parse_csp_manifest

    by_group = defaultdict(list)
    for row in records:
        d = row["descriptor"]
        if d["kind"] != "space" or not 1 <= d["number"] <= 230:
            raise ValueError(f"Not a standard space group: {row['id']}")
        by_group[d["number"]].append(row)
    groups = {}
    sym = settings["specified_symmetry"]
    for n in range(sym["first_group"], sym["last_group"]+1):
        operations = [
            dict(rotation=r.tolist(), translation=t.tolist())
            for r, t in reversed(list(catalog_operations("space", n)))
        ]
        size = sym["attempts_per_group"]
        if by_group[n]:
            row = random.Random(settings["seed"]+n).choice(by_group[n])
            manifest = prepare_manifest([row], size)
            source = "train_reference"
        else:
            # A single orbit gives an achievable exact inventory even for a group absent in train.
            spec = catalog_spec("space", n)
            multiplicity = min(template.multiplicity for template in spec.templates)
            manifest = parse_csp_manifest(dict(schema="ugg_csp_cases_v1", candidates_per_case=size,
                cases=[dict(dataset="mp_20", case_id=f"inventory_probe_sg{n:03d}", kind="space",
                            counts={str(sym["synthetic_element"]):multiplicity})]))
            source = "wyckoff_inventory_probe"
        groups[str(n)] = dict(operations=operations, csp_manifest=manifest.to_dict(),
                              csp_reference=source)
    return groups


def jobs_for(groups, shardsize):
    group_ids = list(groups)
    jobs = {}
    for backbone in MODEL_BY_BACKBONE:
        for task in ("dng", "csp"):
            jobs[f"{backbone}_{task}_standard"] = dict(backbone=backbone, task=task,
                                                       mode="standard", groups=[])
            for mode in MODES:
                for start in range(0,len(group_ids),shardsize):
                    shard = group_ids[start:start+shardsize]
                    label = f"{backbone}_{task}_{mode}_{shard[0]}-{shard[-1]}"
                    jobs[label] = dict(backbone=backbone, task=task, mode=mode, groups=shard)
    return jobs


def prepare(root, campaign_dir, settings, campaign):
    from unifiedgroupgen.common.source import file_hash
    from unifiedgroupgen.evaluation.joint_csp import prepare_csp_manifest
    from unifiedgroupgen.pl_data.cache import records_from_cache

    expected_counts = {"train":27136, "test":9046}
    expected_models = {model_name(b,t) for b in MODEL_BY_BACKBONE for t in ("dng","csp")}
    if set(campaign["models"]) != expected_models or len(campaign["models"]) != 4:
        raise ValueError("Unexpected frozen training campaign models")
    audits = [read(campaign_dir/"audits"/f"{name}.json") for name in campaign["models"]]
    caches = {}
    for split, count in expected_counts.items():
        inputs = audits[0]["inputs"][split]
        if len(inputs) != 1 or any(not item["passed"] or item["inputs"][split] != inputs or
                                   item["records"][split] != count for item in audits):
            raise ValueError(f"Inconsistent audited {split} split")
        if file_hash(Path(inputs[0]["path"])) != inputs[0]["sha256"]:
            raise ValueError(f"Changed {split} cache bytes")
        caches[split] = inputs[0]

    checkpoints = {}
    for name in sorted(expected_models):
        path = root/"outputs"/name/campaign_dir.name/"best.ckpt"
        if path.stat().st_size == 0 or read(path.parent/"model.status.json")["state"] != "complete":
            raise ValueError(f"Missing completed checkpoint: {path}")
        checkpoints[name] = dict(path=str(path), sha256=file_hash(path))

    records = sorted(records_from_cache(Path(caches["train"]["path"])), key=lambda row: row["id"])
    if (len(records) != expected_counts["train"] or len({r["id"] for r in records}) != len(records)
            or any(r["dataset"] != "mp_20" for r in records)):
        raise ValueError("Unexpected training records")
    csp = settings["csp"]
    selected = sorted(random.Random(settings["seed"]).sample(records,csp["cases"]), key=lambda row:row["id"])
    manifest = prepare_csp_manifest(selected, max(csp["ks"]))
    groups = group_requests(records, settings, prepare_csp_manifest)
    jobs = jobs_for(groups, settings["specified_symmetry"]["groups_per_job"])
    launcher = Path(__file__).resolve()
    sbatch = launcher.with_suffix(".sbatch")
    if not sbatch.is_file():
        raise FileNotFoundError(sbatch)
    return dict(settings=settings, root=str(root), source=campaign["source"],
                launcher_sha256=file_hash(launcher), sbatch_sha256=file_hash(sbatch),
                caches=caches, checkpoints=checkpoints, csp_manifest=manifest.to_dict(),
                csp_manifest_sha256=manifest.digest, groups=groups, jobs=jobs)


def submit(run, plan):
    from fcntl import flock, LOCK_EX, LOCK_NB

    resources = plan["settings"]["slurm"]
    launcher = Path(__file__).resolve()
    logs = run/"logs"
    logs.mkdir(exist_ok=True)
    with (run/"submission.jsonl").open("a+",encoding="utf-8",buffering=1) as ledger:
        flock(ledger,LOCK_EX | LOCK_NB)
        ledger.seek(0)
        queued = {r["label"]:r["job_id"] for r in map(json.loads,ledger)}
        for label in plan["jobs"]:
            if label in queued:
                print("ALREADY_QUEUED",label,queued[label],flush=True)
                continue
            command = ["yhbatch","--parsable","--export=ALL",f"--job-name=ugg_eval_{label}",
                "--partition="+resources["partition"],"-N","1","-n","1","--gres=gpu:1",
                f"--cpus-per-task={resources['cpus']}","--mem="+resources["memory"],
                "--time="+resources["time"],"--chdir="+plan["source"],
                "--output="+str(logs/"%x-%j.out"),str(launcher.with_suffix(".sbatch")),
                str(Path(sys.executable)),str(launcher),str(run/"evaluation_plan.json"),label]
            reply = subprocess.check_output(command,text=True).strip()
            job = reply.split(";",1)[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected scheduler reply for {label}: {reply!r}")
            ledger.write(json.dumps(dict(label=label,job_id=job))+"\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            print("QUEUED",label,job,flush=True)
    print("ALL_EVALUATIONS_QUEUED",len(plan["jobs"]),run,flush=True)


def summarize_group(path, number, operations, expected, expected_inventory=None):
    from unifiedgroupgen.common.structures import iter_records, load_structure
    from unifiedgroupgen.evaluation.common import check_symmetry

    fields = ("structure_valid", "target_operation_containment", "valid_operation_containment",
              "exact_target_group", "strict_supergroup")
    counts, stats, failures = Counter(), Counter(), Counter()
    for row in iter_records(path):
        val = row.get("validation", {})
        valid = bool(val.get("structure_valid"))
        counts["structure_valid"] += valid
        counts["target_operation_containment"] += bool(val.get("cif_operation_check", {}).get("target_operation_containment"))
        counts["valid_operation_containment"] += bool(valid and val.get("cif_operation_check", {}).get("target_operation_containment"))
        if expected_inventory is not None:
            inventory_ok = Counter(row.get("atomic_numbers") or []) == expected_inventory
            counts["composition_match"] += inventory_ok
            counts["valid_operations_and_composition"] += bool(
                inventory_ok and valid and val.get("cif_operation_check", {}).get("target_operation_containment"))
        stats[row["status"]] += 1
        failures.update(row.get("rejection_reasons") or ([row["reason"]] if row.get("reason") else []))
        if "cif" not in row or not val.get("cif_operation_check", {}).get("target_operation_containment"):
            continue
        report = dict(target_group=number, basic_valid=valid)
        check_symmetry(report, row, {"operations": operations}, load_structure(row, path.parent), "space", True)
        counts["exact_target_group"] += report.get("exact_target_group") is True
        counts["strict_supergroup"] += report.get("strict_supergroup") is True
        counts["relation_unresolved"] += report.get("operation_relation") == "unresolved"
    if sum(stats.values()) != expected:
        raise ValueError(f"Incomplete group ledger: {path}")
    if expected_inventory is not None:
        fields += ("composition_match", "valid_operations_and_composition")
    return dict(group=number, attempts=expected, outcomes=dict(stats), reasons=dict(failures),
                counts={key: counts[key] for key in (*fields, "relation_unresolved")},
                rates={key: counts[key]/expected for key in fields})



def worker(run, plan, label):
    import torch
    from unifiedgroupgen.common.csp_cases import parse_csp_manifest
    from unifiedgroupgen.common.source import file_hash
    from unifiedgroupgen.common.structures import iter_records
    from unifiedgroupgen.conditional.tasks import generation_mode
    from unifiedgroupgen.evaluation.joint_csp import evaluate_csp
    from unifiedgroupgen.generation.unified.runner import generate
    from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
    from unifiedgroupgen.group_constraints.operations import compile_request
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.pl_data.cache import records_from_cache
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.pl_modules.training_utils import strict_fp32
    from unifiedgroupgen.pl_modules.unified.module import load_joint_checkpoint

    torch.set_num_threads(1)
    strict_fp32(dict(strict_fp32=True,precision="fp32"))
    job = plan["jobs"][label]
    backbone, task, mode = (job[key] for key in ("backbone","task","mode"))
    saved = plan["checkpoints"][model_name(backbone,task)]
    checkpoint_path = Path(saved["path"])
    if file_hash(checkpoint_path) != saved["sha256"]:
        raise ValueError("Checkpoint changed after submission")
    checkpoint = load_joint_checkpoint(checkpoint_path,"cpu")
    if (generation_mode(checkpoint["config"]) != task
            or checkpoint["config"]["model"]["backbone"] != MODEL_BY_BACKBONE[backbone]):
        raise ValueError("Checkpoint task or backbone mismatch")
    model = build_joint_model(checkpoint["config"],"cuda:0").eval()
    model.load_state_dict(checkpoint["model"],strict=True)
    output = run/label
    output.mkdir()
    train, test = (Path(plan["caches"][split]["path"]) for split in ("train","test"))
    references = {row["id"]:row for row in records_from_cache(train)} if task == "csp" else {}
    settings = plan["settings"]
    sampling = settings["sampling"]

    def sample(directory, attempts, seed, manifest=None, number=None, operations=None):
        spec = None
        if number is not None:
            spec = (compile_request(dict(kind="space",operations=operations)) if mode == "operations"
                    else catalog_spec("space",number))
        if manifest:
            for case in manifest.cases:
                case.chemistry.domain(model.allowed_elements)
                if sum(dict(case.chemistry.counts).values()) > model.max_atoms:
                    raise ValueError(f"Inventory exceeds checkpoint max_atoms: {case.case_id}")
        path = directory/"samples.jsonl"
        args = SimpleNamespace(checkpoint=str(checkpoint_path),output=str(path),steps=sampling["steps"],
            batch_size=sampling["batch_size"],count=attempts,seed=seed,device="cuda:0",kind="space",
            group=number if mode == "number" else None,symmetry=True if mode == "operations" else None,
            case_manifest=manifest,case_id=None,case_dataset=None,exact_group=False,
            csp_group_policy=settings["csp"]["group_policy"] if task == "csp" else "prior")
        torch.cuda.reset_peak_memory_stats()
        report = generate(args,checkpoint,model,spec,ChemicalControl.from_inputs(),{},path)
        accepted = report["accepted"]
        save(directory/"sampling_cost.json",dict(
            attempts=attempts, accepted=accepted, sampling_seconds=report["seconds"],
            seconds_per_attempt=report["seconds"]/attempts,
            seconds_per_accepted=report["seconds"]/accepted if accepted else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        return path

    def match(path,manifest,ks):
        refs = [references[case.case_id] for case in manifest.cases]
        metrics = evaluate_csp(iter_records(path),refs,tuple(ks),manifest)
        save(path.parent/"csp_metrics.json",metrics)
        return metrics["summary"]

    if mode == "standard":
        if task == "csp":
            manifest = parse_csp_manifest(plan["csp_manifest"])
            if manifest.digest != plan["csp_manifest_sha256"]:
                raise ValueError("Shared CSP manifest changed")
            path = sample(output,len(manifest.cases)*manifest.candidates_per_case,settings["seed"],manifest)
            print("CSP_TRAIN_RECONSTRUCTION",match(path,manifest,settings["csp"]["ks"]),flush=True)
        else:
            count = settings["dng"]["attempts"]
            path = sample(output,count,settings["seed"])
            subprocess.run([sys.executable,"-B","-m","unifiedgroupgen.evaluate",
                f"evaluation.samples={path}","evaluation.kind=space","evaluation.dataset=mp_20",
                f"evaluation.reference_files=[{test}]",f"evaluation.novelty_files=[{train}]",
                f"evaluation.expected_attempts={count}",f"evaluation.output={output/'metrics'}",
                f"evaluation.cache={output/'reference_cache'}"],check=True)
    else:
        reports = []
        count = settings["specified_symmetry"]["attempts_per_group"]
        for group in job["groups"]:
            number = int(group)
            request = plan["groups"][group]
            manifest = parse_csp_manifest(request["csp_manifest"]) if task == "csp" else None
            path = sample(output/f"group_{number:03d}",count,settings["seed"]+1000*number,
                          manifest,number,request["operations"])
            inventory = dict(manifest.cases[0].chemistry.counts) if manifest else None
            metrics = summarize_group(path,number,request["operations"],count,inventory)
            if task == "csp":
                metrics["reference"] = request["csp_reference"]
                metrics["csp"] = (match(path,manifest,(1,5,10,count))
                                  if metrics["reference"] == "train_reference" else None)
            reports.append(metrics)
            print("GROUP_EVALUATED",label,number,count,flush=True)
        save(output/"group_results.json",reports)
    save(output/"complete.json",dict(job=label,task=task,mode=mode))
    print("EVALUATION_COMPLETE",label,output,flush=True)


def pool(rows):
    total = sum(row["attempts"] for row in rows)
    return dict(groups=len(rows),attempts=total,rates={
        key:sum(row["counts"][key] for row in rows)/total for key in rows[0]["rates"]} if total else {})


def collect(run, plan):
    absent = [label for label in plan["jobs"] if not (run/label/"complete.json").is_file()]
    if absent:
        raise SystemExit("INCOMPLETE_JOBS: "+", ".join(absent))
    by_mode = defaultdict(list)
    for label, job in plan["jobs"].items():
        if job["mode"] != "standard":
            rows = read(run/label/"group_results.json")
            if {str(row["group"]) for row in rows} != set(job["groups"]) or len(rows) != len(job["groups"]):
                raise ValueError(f"Incomplete group shard: {label}")
            by_mode[job["backbone"],job["task"],job["mode"]].extend(rows)
    summary = {}
    for (backbone,task,mode), rows in by_mode.items():
        rows.sort(key=lambda row:row["group"])
        if [str(row["group"]) for row in rows] != list(plan["groups"]):
            raise ValueError(f"Missing or duplicate group: {backbone}/{task}/{mode}")
        if any(row["attempts"] != plan["settings"]["specified_symmetry"]["attempts_per_group"] for row in rows):
            raise ValueError("Incorrect group denominator")
        entry = dict(total=pool(rows),per_group=rows)
        if task == "csp":
            real = [r for r in rows if r["reference"] == "train_reference"]
            probes = [r for r in rows if r["reference"] == "wyckoff_inventory_probe"]
            entry.update(train_reference=pool(real),inventory_probe=pool(probes),
                match_over_train_reference_groups={
                    k:sum(row["csp"][k]["match_rate"] for row in real)/len(real)
                    for k in real[0]["csp"]} if real else {})
        summary[f"{backbone}_{task}_{mode}"] = entry
    save(run/"all_groups_summary.json",summary)
    print("ALL_SPACE_GROUPS_COLLECTED",run/"all_groups_summary.json",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("submit","worker","collect"))
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    root = Path(os.environ.get("UGG_PROJECT_ROOT",Path.cwd())).resolve()
    if args.action == "submit":
        settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not all(re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]+",settings[key]) for key in ("name","campaign")):
            parser.error("Invalid campaign or evaluation name")
        campaign_dir = root/"intermediate/hpc_runs"/settings["campaign"]
        campaign = read(campaign_dir/"campaign.json")
        source = (campaign_dir/"source").resolve()
        if campaign["revision"] != settings["frozen_revision"] or Path(campaign["source"]).resolve() != source:
            raise ValueError("Unexpected frozen source or revision")
        run = campaign_dir/"evaluation"/settings["name"]
        plan_path = run/"evaluation_plan.json"
    else:
        if args.plan is None or (args.action == "worker" and not args.job):
            parser.error("worker/collect require --plan; worker also requires --job")
        plan_path = args.plan.resolve()
        run = plan_path.parent
        plan = read(plan_path)
        root,source = Path(plan["root"]),Path(plan["source"])
    os.environ["UGG_PROJECT_ROOT"] = str(root)
    os.chdir(source)
    sys.path.insert(0,str(source))
    if args.action == "submit":
        from unifiedgroupgen.common.source import file_hash
        launcher = Path(__file__).resolve()
        if plan_path.is_file():
            plan = read(plan_path)
            if (plan["settings"] != settings or plan["launcher_sha256"] != file_hash(launcher)
                    or plan["sbatch_sha256"] != file_hash(launcher.with_suffix(".sbatch"))):
                raise ValueError("Existing evaluation plan has different settings or launcher")
        else:
            plan = prepare(root,campaign_dir,settings,campaign)
            run.mkdir(parents=True)
            save(plan_path,plan)
        print("SPACE_GROUP_PLAN",len(plan["groups"]),dict(Counter(
            group["csp_reference"] for group in plan["groups"].values())),flush=True)
        submit(run,plan)
    elif args.action == "worker":
        worker(run,plan,args.job)
    else:
        collect(run,plan)


if __name__ == "__main__":
    main()

