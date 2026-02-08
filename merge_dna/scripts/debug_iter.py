import time
from pathlib import Path
from torch.utils.data import DataLoader
from merge_dna.data.multi_species_corpus import make_generator_factory, ShardedGenomicDataset

# dummy generator function: yields 100 items per file
def dummy_gen(files, chunk_length=10, overlap=1):
    for f in files:
        for i in range(10):
            yield (f.name, {"sequence": f"SEQ_{f.name}_{i}", "start_pos": i, "end_pos": i+1})

# create fake files list
files = [Path(f"file_{i}.fna.gz") for i in range(5)]
factory = make_generator_factory(dummy_gen)
ds = ShardedGenomicDataset(files, generator_factory=factory, chunk_length=10, overlap=1, infinite=False)
loader = DataLoader(ds, batch_size=4, collate_fn=lambda b: b, num_workers=0)

for epoch in range(3):
    print("=== EPOCH", epoch)
    it = iter(loader)
    cnt = 0
    for batch in it:
        print(" batch:", batch[:1])
        cnt += 1
        if cnt >= 3:
            break
    print(" epoch", epoch, "consumed", cnt, "batches")
