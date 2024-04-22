from finetune import *



models = [
    # ("alexnet", lambda : torchvision.models.alexnet(pretrained=True)),
    # ("regnet_y_400mf", lambda: torchvision.models.regnet_y_400mf(pretrained=True)),
    # ("regnet_x_400mf", lambda: torchvision.models.regnet_x_400mf(pretrained=True)),
    # ("regnet_x_800mf", lambda: torchvision.models.regnet_x_800mf(pretrained=True)),
    # ("regnet_y_800mf", lambda: torchvision.models.regnet_y_800mf(pretrained=True)),
    # ("regnet_x_1_6gf", lambda: torchvision.models.regnet_x_1_6gf(pretrained=True)),
    # ("googlenet", lambda: torchvision.models.googlenet(pretrained=True)),
    ("resnet18", lambda: torchvision.models.resnet18(pretrained=True)),
]

if __name__ == "__main__":
    # start = int(sys.argv[1])
    # print("Training the ", "odd" if start == 1 else "even", " models")
    # models = models[start::2]
    print([x[0] for x in models])

    for name, builder in models:
        print("=" * 100)
        print("=" * 100)
        print("Training model", name)
        print("=" * 100)
        print("=" * 100)
        try:
            def prepare_builder():
                model = builder()
                prepare_pretrained_model(model)
                model.to(device)
                return model
            # print(summary(model, (3, 300, 400)))

            # Create a study object and optimize it
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, name, prepare_builder), n_trials=100)

            best_params = study.best_params
            print(f"Best hyperparameters: {best_params}")

            # train_fine_tuning(name, model, 0.001, param_group=True)
        except KeyboardInterrupt:
            cmd = input("If you want to exit, type q. Otherwise, hit enter.")
            if cmd == "q":
                exit(0)
        except Exception:
            print("Error during building model:")
            print(traceback.format_exc())
            print("Skipping")