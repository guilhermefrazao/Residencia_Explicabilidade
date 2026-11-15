from supervized_learning import main as train_supervized
from contribution.contributions import main as interpret_contributions

def main(learning_type: str="supervized", checkpoints="checkpoint/test", dataset="Mnist", model="MLP", loss_type="CrossEntropy", optimizer_type="Adam"):
    if learning_type == "supervized":
        train_supervized(checkpoints=checkpoints, dataset=dataset, model=model, loss_type=loss_type, optimizer_type=optimizer_type)
    
    if learning_type == "unsupervized":
        pass

    if learning_type == "reinforcement":
        pass


    interpret_contributions(model_path=checkpoints, dataset_type=dataset, model_type=model)


if __name__ == "__main__":
    main()