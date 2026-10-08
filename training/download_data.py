"""Step 1: download the raw public datasets from the Hugging Face Hub.

Sources (all permissively licensed):
  * HuggingFaceTB/smollm-corpus  cosmopedia-v2      synthetic textbooks/stories (Apache-2.0)
  * HuggingFaceTB/smollm-corpus  fineweb-edu-dedup  educational web pages (ODC-By)
  * HuggingFaceTB/smol-smoltalk                      chat conversations (Apache-2.0)
  * HuggingFaceTB/smoltalk  everyday-conversations   short chit-chat (Apache-2.0)
"""
import os
from huggingface_hub import hf_hub_download
from common import RAW

FILES = [
    ("HuggingFaceTB/smoltalk", "data/everyday-conversations/train-00000-of-00001.parquet"),
    ("HuggingFaceTB/smol-smoltalk", "data/train-00000-of-00004.parquet"),
    ("HuggingFaceTB/smol-smoltalk", "data/train-00001-of-00004.parquet"),
] + [("HuggingFaceTB/smollm-corpus", f"cosmopedia-v2/train-{i:05d}-of-00104.parquet") for i in range(3)] \
  + [("HuggingFaceTB/smollm-corpus", f"fineweb-edu-dedup/train-{i:05d}-of-00234.parquet") for i in range(2)]

if __name__ == "__main__":
    for repo, f in FILES:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(RAW, repo.split("/")[1]))
        print("ok", p, flush=True)
