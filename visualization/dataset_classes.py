import argparse, glob, numpy as np, pyarrow.parquet as pq, torchvision

from setup import dataset_class_names


def train_targets(dataset_type: str):
    # y de todas as imagens de treino; no ImageNet lê só a coluna de rótulos dos Parquet (sem carregar as imagens)
    if dataset_type == "Imagenet":
        return np.concatenate([pq.read_table(file, columns=["label"])["label"].to_numpy() for file in sorted(glob.glob("datasets/Imagenet/data/train-*.parquet"))])

    return torchvision.datasets.MNIST("datasets", train=True, download=False).targets.numpy()


def main(dataset_type: str = "Imagenet", classes: list = None):
    names = dataset_class_names(dataset_type)

    ids, counts = np.unique(train_targets(dataset_type), return_counts=True)

    # com --classes só essas aparecem, e "label" é a posição delas ordenadas: o valor a usar em --labels (ablação) e nas linhas dos gráficos
    positions = {} if classes is None else {c: p for p, c in enumerate(np.sort(classes))}

    print(f"{dataset_type}: {len(ids)} classes no treino, {counts.sum()} imagens")

    for class_id, count in zip(ids, counts):
        if classes is None or class_id in positions:
            print(f"y = {class_id:>4} | {names[class_id]:<30} | {count:>5} imagens de treino" + (f" | label = {positions[class_id]}" if positions else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset_type", default="Imagenet", choices=["Mnist", "Imagenet"])
    parser.add_argument("--classes", type=int, nargs="+", default=None)

    main(**vars(parser.parse_args()))
