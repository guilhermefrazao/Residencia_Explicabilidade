from supervized_learning import main as train_supervized
from contribution.contributions import main as interpret_contributions
from patching.neural_ablation import main as ablate_neurons


def main(learning_type: str="supervized", checkpoints="checkpoint/ImagenetTrain", dataset="Imagenet", model="ResNet50", loss_type="CrossEntropy", optimizer_type="Adam", layer_names=None, reduction=None, n_epochs=100, interpret=True):
    if learning_type == "supervized":
        train_supervized(checkpoints=checkpoints, dataset=dataset, model=model, loss_type=loss_type, optimizer_type=optimizer_type, n_epochs=n_epochs)
    
    if learning_type == "unsupervized":
        pass

    if learning_type == "reinforcement":
        pass


    if interpret:
        interpret_contributions(model_path=checkpoints, dataset_type=dataset, model_type=model, layer_names=layer_names, reduction=reduction)

        ablate_neurons(model_path=checkpoints, dataset_type=dataset, model_type=model, label=[1, 6, 7], layer_names=layer_names, ablated_layers=None)


if __name__ == "__main__":
    main()