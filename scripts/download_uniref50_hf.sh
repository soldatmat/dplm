pip install huggingface_hub
mkdir -p data-bin
# download uniref50 dataset from huggingface hub
huggingface-cli download airkingbd/uniref50 --repo-type dataset --local-dir ./data-bin/uniref50_hf
