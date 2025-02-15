from datasets import load_dataset

dataset = load_dataset(
    "Metacreation/GigaMIDI", trust_remote_code=True, streaming=True
)

train_set = dataset['train']
for x in train_set:
    if type(x["music"]) is not bytes:
        print(type(x["music"]))