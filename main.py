import src.processing as sp

sp.generate_random_set(['spiral', 'elliptical'], 900, save=True)

d = sp.split_data(['spiral', 'elliptical'], val_size=0.35, save=True)

sp.images_train_validate(['spiral', 'elliptical'])
