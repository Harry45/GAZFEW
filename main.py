import src.processing as sp

# sp.generate_random_set(['spiral', 'ring'], 1000, save=True)

# d = sp.split_data(['spiral', 'ring'], val_size=0.35, save=True)

sp.images_train_validate(['spiral', 'ring'])
