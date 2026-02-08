# merge_dna/data/data.py
import gzip
import logging
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Tuple

from Bio import SeqIO
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
PAD_ID = 5
VOCAB_SIZE = 6


def seq_to_ids(seq: str):
    return [DNA_MAP.get(ch, DNA_MAP["N"]) for ch in seq]


def filter_fn(char: str) -> str:
    """Normalize non-ACGT -> N"""
    return char if char in {"A", "C", "G", "T"} else "N"


def clean_sequence(seq: str) -> str:
    """Uppercase and map invalid bases to 'N'"""
    seq = seq.upper().replace("\n", "").replace("\r", "")
    # Replace invalid characters
    return "".join(filter_fn(ch) for ch in seq)


def open_fasta(path: Path):
    """Return a text-mode file handle for FASTA or gz FASTA files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # handle .fna.gz .fa.gz .fasta.gz etc
    if len(path.suffixes) >= 1 and path.suffixes[-1] == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        return open(path, "rt", encoding="utf-8", errors="replace")


def _generate_examples(files: Iterable[Path], chunk_length: int, overlap: int) -> Iterator[Tuple[int, dict]]:
    """
    Iterate over a list of Path objects, stream FASTA records, and yield windows.

    Yields (key:int, example:dict)
    """
    key = 0
    files = list(files)
    logger.info(f"_generate_examples: starting with {len(files)} files chunk_length={chunk_length} overlap={overlap}")
    for file in files:
        try:
            logger.debug(f"_generate_examples: opening {file}")
            with open_fasta(file) as fh:
                for rec in SeqIO.parse(fh, "fasta"):
                    if rec is None:
                        continue
                    seq = str(rec.seq)
                    if not seq:
                        continue
                    seq = clean_sequence(seq)
                    seq_length = len(seq)

                    # compute number of full chunks we can get when allowing 2*overlap padding
                    if seq_length < chunk_length:
                        logger.debug(f"Skipping short contig {rec.id} in {file} len={seq_length}")
                        continue

                    # compute number of windows: ensure non-negative
                    # We want non-overlapping core chunk_length with overlap padding on both sides:
                    # Equivalent to sliding by chunk_length (stride==chunk_length) but include overlap on both sides
                    # number of chunks = floor((seq_length - 2*overlap) / chunk_length)
                    num_chunks = (seq_length - 2 * overlap) // chunk_length
                    if num_chunks < 1:
                        logger.debug(f"No chunks for contig {rec.id} (len={seq_length}) after accounting for overlap")
                        continue

                    usable_len = chunk_length * num_chunks + 2 * overlap
                    seq = seq[:usable_len]
                    seq_length = len(seq)

                    for i in range(num_chunks):
                        start_pos = i * chunk_length
                        end_pos = min(seq_length, (i + 1) * chunk_length + 2 * overlap)
                        chunk_sequence = seq[start_pos:end_pos]
                        example = {
                            "sequence": chunk_sequence,
                            "description": rec.description,
                            "start_pos": int(start_pos),
                            "end_pos": int(end_pos),
                            "file": str(file),
                            "record_id": rec.id,
                        }
                        yield key, example
                        key += 1
        except Exception as e:
            logger.exception("Error reading file %s: %s", file, e)
            continue
    logger.info("_generate_examples: finished streaming files")



def collate_genomic(batch: list, pad_id: int = PAD_ID):
    """
    Collate a batch from the generator: batch is list of (key, example_dict)
    Returns tensor: batch_ids
    """
    _, examples = zip(*batch)
    seqs = [ex["sequence"] for ex in examples]
    id_lists = [seq_to_ids(s) for s in seqs]
    lengths = [len(x) for x in id_lists]
    if len(lengths) == 0:
        raise ValueError("collate_genomic received empty batch")
    L = max(lengths)

    batch_ids = torch.full((len(id_lists), L), fill_value=pad_id, dtype=torch.long)
    for i, arr in enumerate(id_lists):
        batch_ids[i, :len(arr)] = torch.tensor(arr, dtype=torch.long)

    return batch_ids


class ShardedGenomicDataset(IterableDataset):
    def __init__(self, all_files: List[Path], generator_factory: Callable, chunk_length: int = 6000, overlap: int = 100, infinite: bool = False):
        super().__init__()
        self.all_files = list(all_files)
        self.generator_factory = generator_factory
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.infinite = infinite

    def _get_sharded_files(self, worker_id: int, num_workers: int) -> List[Path]:
        if num_workers <= 1:
            return self.all_files
        return [p for idx, p in enumerate(self.all_files) if (idx % num_workers) == worker_id]

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            wid, nw = 0, 1
        else:
            wid, nw = worker.id, worker.num_workers

        files_shard = self._get_sharded_files(wid, nw)
        logger.info(f"ShardedGenomicDataset.__iter__ worker={wid}/{nw} files={len(files_shard)} infinite={self.infinite}")
        # create fresh generator for this worker
        gen = self.generator_factory(files_shard, chunk_length=self.chunk_length, overlap=self.overlap)

        if not self.infinite:
            for item in gen:
                yield item
        else:
            while True:
                for item in gen:
                    yield item
                # recreate the generator for next pass
                gen = self.generator_factory(files_shard, chunk_length=self.chunk_length, overlap=self.overlap)


def make_generator_factory(base_generator_fn: Callable):
    """
    base_generator_fn(files_subset, chunk_length, overlap) -> generator
    Returns factory(files_subset, chunk_length, overlap) -> generator (fresh each call)
    """
    def factory(files_subset: List[Path], chunk_length: int = 6000, overlap: int = 100):
        return base_generator_fn(files_subset, chunk_length=chunk_length, overlap=overlap)
    return factory


HERE = Path(__file__).resolve()
GENOMES_DIR = HERE.parent / "genomes"
if not GENOMES_DIR.exists():
    raise RuntimeError(f"Genomes dir not found: {GENOMES_DIR}")

# Accept .fa, .fna, with or without .gz
files = sorted([p for p in GENOMES_DIR.glob("*") if p.is_file() and any(suffix in "".join(p.suffixes) for suffix in [".fa", ".fna"])])

chunk_length = 6000
overlap = 100


def make_data_loader(batch_size: int, infinite: bool = False, num_workers: int = 0):
    factory = make_generator_factory(_generate_examples)   # pass the function, not an instance
    ds = ShardedGenomicDataset(files, generator_factory=factory, chunk_length=chunk_length, overlap=overlap, infinite=infinite)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_genomic,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader


def smoke_test_loader(n_epochs: int = 3, batches_per_epoch: int = 4, batch_size: int = 2):
    print("Running smoke test: n_epochs", n_epochs, "batches_per_epoch", batches_per_epoch, "batch_size", batch_size)
    loader = make_data_loader(batch_size=batch_size, infinite=True, num_workers=0)
    for epoch in range(n_epochs):
        print("=== EPOCH", epoch)
        it = iter(loader)
        for i in range(batches_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                print("StopIteration on epoch", epoch, "batch", i)
                break
            print(" epoch", epoch, "batch", i, "len_input", batch["input_ids"].shape, "first_keys", batch["keys"][:2])
    print("smoke test done")


# Run smoke test when executed directly
if __name__ == "__main__":
    smoke_test_loader()
