#!/usr/bin/env python3
import argparse
import json
import os
import resource
import tarfile
import time
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def iter_imagefolder_samples(split_dir, class_to_idx=None):
    split_dir = Path(split_dir)
    classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if class_to_idx is None:
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    samples = []
    for class_name in classes:
        if class_name not in class_to_idx:
            continue
        class_dir = split_dir / class_name
        label = class_to_idx[class_name]
        for path in sorted(class_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(path), label))
    return samples, class_to_idx


def collect_imagefolder_by_class(split_dir, class_to_idx=None):
    split_dir = Path(split_dir)
    classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if class_to_idx is None:
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    samples_by_label = {}
    for class_name in classes:
        if class_name not in class_to_idx:
            continue
        class_dir = split_dir / class_name
        label = class_to_idx[class_name]
        samples_by_label[label] = [
            str(path)
            for path in sorted(class_dir.rglob('*'))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return samples_by_label, class_to_idx


def add_file_to_tar(tar, src_path, member_name):
    src_path = Path(src_path)
    info = tarfile.TarInfo(member_name)
    stat = src_path.stat()
    info.size = stat.st_size
    info.mtime = int(stat.st_mtime)
    with src_path.open('rb') as f:
        tar.addfile(info, f)


def write_split(samples, output_dir, split, shard_size, start_index=0):
    output_dir = Path(output_dir)
    split_meta = {
        'samples': len(samples),
        'full_shard_size': shard_size,
        'shards': [],
    }

    t0 = time.time()
    for shard_idx, start in enumerate(range(0, len(samples), shard_size)):
        shard_samples = samples[start:start + shard_size]
        shard_name = f'{split}-{shard_idx:06d}.tar'
        final_path = output_dir / shard_name
        tmp_path = output_dir / f'.{shard_name}.tmp'
        if final_path.exists():
            final_path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()

        with tarfile.open(tmp_path, 'w') as tar:
            for local_idx, (src_path, label) in enumerate(shard_samples):
                sample_idx = start_index + start + local_idx
                ext = Path(src_path).suffix.lower()
                if ext == '.jpeg':
                    ext = '.jpg'
                member_name = f'{sample_idx:09d}_{label:04d}{ext}'
                add_file_to_tar(tar, src_path, member_name)

        os.replace(tmp_path, final_path)
        split_meta['shards'].append({'name': shard_name, 'samples': len(shard_samples)})
        elapsed = max(1e-6, time.time() - t0)
        done = min(start + len(shard_samples), len(samples))
        print(
            f'{split}: wrote {shard_name} ({len(shard_samples)} samples), '
            f'{done}/{len(samples)} done, {done / elapsed:.1f} samples/s',
            flush=True,
        )

    return split_meta


def _raise_file_limit_for_shards(num_shards):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = num_shards + 64
    if soft >= required:
        return
    if hard < required:
        raise RuntimeError(
            f'Need at least {required} open files to write {num_shards} shards, '
            f'but RLIMIT_NOFILE is soft={soft}, hard={hard}. Increase ulimit or use a larger --shard-size.'
        )
    resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))


def _build_interleaved_position_maps(samples_by_label):
    labels = sorted(samples_by_label)
    lengths = {label: len(samples_by_label[label]) for label in labels}
    max_len = max(lengths.values(), default=0)
    round_offsets = []
    ranks_by_round = []
    offset = 0
    for sample_idx in range(max_len):
        active_labels = [label for label in labels if lengths[label] > sample_idx]
        round_offsets.append(offset)
        ranks_by_round.append({label: rank for rank, label in enumerate(active_labels)})
        offset += len(active_labels)
    return round_offsets, ranks_by_round, offset


def write_split_class_mixed(samples_by_label, output_dir, split, shard_size, start_index=0):
    output_dir = Path(output_dir)
    total_samples = sum(len(paths) for paths in samples_by_label.values())
    num_shards = (total_samples + shard_size - 1) // shard_size
    _raise_file_limit_for_shards(num_shards)

    round_offsets, ranks_by_round, mapped_total = _build_interleaved_position_maps(samples_by_label)
    if mapped_total != total_samples:
        raise RuntimeError(f'Internal sample accounting mismatch: mapped={mapped_total}, total={total_samples}')

    shard_specs = []
    for shard_idx in range(num_shards):
        shard_name = f'{split}-{shard_idx:06d}.tar'
        final_path = output_dir / shard_name
        tmp_path = output_dir / f'.{shard_name}.tmp'
        if final_path.exists():
            final_path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()
        shard_specs.append((shard_name, final_path, tmp_path))

    tar_files = []
    shard_counts = [0 for _ in range(num_shards)]
    split_meta = {
        'samples': total_samples,
        'full_shard_size': shard_size,
        'shards': [{'name': shard_name, 'samples': 0} for shard_name, _, _ in shard_specs],
    }

    t0 = time.time()
    done = 0
    next_report = 20000
    try:
        tar_files = [tarfile.open(tmp_path, 'w') for _, _, tmp_path in shard_specs]
        for label in sorted(samples_by_label):
            for class_sample_idx, src_path in enumerate(samples_by_label[label]):
                position = round_offsets[class_sample_idx] + ranks_by_round[class_sample_idx][label]
                shard_idx = position // shard_size
                ext = Path(src_path).suffix.lower()
                if ext == '.jpeg':
                    ext = '.jpg'
                member_name = f'{start_index + position:09d}_{label:04d}{ext}'
                add_file_to_tar(tar_files[shard_idx], src_path, member_name)
                shard_counts[shard_idx] += 1
                done += 1
                if done >= next_report or done == total_samples:
                    elapsed = max(1e-6, time.time() - t0)
                    print(
                        f'{split}: wrote {done}/{total_samples} samples across {num_shards} shards, '
                        f'{done / elapsed:.1f} samples/s',
                        flush=True,
                    )
                    next_report += 20000
    finally:
        for tar in tar_files:
            tar.close()

    for shard_idx, (shard_name, final_path, tmp_path) in enumerate(shard_specs):
        os.replace(tmp_path, final_path)
        split_meta['shards'][shard_idx]['samples'] = shard_counts[shard_idx]

    full_shards = sum(1 for count in shard_counts if count == shard_size)
    print(
        f'{split}: finalized {num_shards} shards ({full_shards} full, {total_samples} samples)',
        flush=True,
    )
    return split_meta


def atomic_write_json(path, data):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description='Convert ImageNet ImageFolder data to tar shards.')
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--shard-size', type=int, default=2048)
    parser.add_argument('--format', choices=['class-mixed', 'global-shuffle'], default='class-mixed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f'Output dir is not empty: {output_dir}. Pass --overwrite to replace shard files.')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading train samples from {args.train_dir}', flush=True)
    train_by_label, class_to_idx = collect_imagefolder_by_class(args.train_dir)
    print(f'Reading val samples from {args.val_dir}', flush=True)
    val_by_label, _ = collect_imagefolder_by_class(args.val_dir, class_to_idx=class_to_idx)
    train_samples = sum(len(paths) for paths in train_by_label.values())
    val_samples = sum(len(paths) for paths in val_by_label.values())

    if args.format == 'global-shuffle':
        import random
        train_flat = [(src_path, label) for label, paths in train_by_label.items() for src_path in paths]
        val_flat = [(src_path, label) for label, paths in val_by_label.items() for src_path in paths]
        rng = random.Random(args.seed)
        rng.shuffle(train_flat)

    metadata = {
        'format': 'dnasnet-imagenet-tar-shards-v1',
        'conversion_format': args.format,
        'seed': args.seed,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source': {
            'train_dir': os.path.abspath(args.train_dir),
            'val_dir': os.path.abspath(args.val_dir),
        },
        'shard_size': args.shard_size,
        'class_to_idx': class_to_idx,
        'idx_to_class': [class_name for class_name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])],
        'splits': {},
    }

    print(f'Found train={train_samples} val={val_samples} classes={len(class_to_idx)}', flush=True)
    if args.format == 'global-shuffle':
        metadata['splits']['train'] = write_split(train_flat, output_dir, 'train', args.shard_size, start_index=0)
        metadata['splits']['val'] = write_split(val_flat, output_dir, 'val', args.shard_size, start_index=0)
    else:
        metadata['splits']['train'] = write_split_class_mixed(
            train_by_label, output_dir, 'train', args.shard_size, start_index=0
        )
        metadata['splits']['val'] = write_split_class_mixed(
            val_by_label, output_dir, 'val', args.shard_size, start_index=0
        )
    atomic_write_json(output_dir / 'metadata.json', metadata)
    print(f'Wrote metadata: {output_dir / "metadata.json"}', flush=True)


if __name__ == '__main__':
    main()
