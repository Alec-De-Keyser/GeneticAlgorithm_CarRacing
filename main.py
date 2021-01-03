from GymTrainer import GymTrainer

if __name__ == '__main__':
    gym_trainer = GymTrainer()
    #gym_trainer.show_gym([0, 5, 10, 20, 30])
    #gym_trainer.train_gym(75)
    gym_trainer.continue_training("/home/alec/PycharmProjects/GeneticAlgorithm_CarRacing/weights39.csv", 40, 10)
