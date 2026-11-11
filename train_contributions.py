from supervized_learning import main as train_supervized
from contribution.contributions import main as interpret_contributions

def main(learning_type: str="supervized"):
    if learning_type == "supervized":
        train_supervized()
    
    if learning_type == "unsupervized":
        pass

    if learning_type == "reinforcement":
        pass


    interpret_contributions()


if __name__ == "__main__":

    main()