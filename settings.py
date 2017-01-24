CONFIG = {
        'nz': 100, # the number of dim.
        'train': {
            'batchsize': 100,
            'n_epoch': 10000, # the number of epoch
            'n_train': 200000, # the number of train per epoch
            'image_save_interval': 50000,
            'paths': {
                'dataset': './images', # input dataset
                'out_image_dir': './out_images',
                'out_model_dir': './out_models'
                }
            },
        'visualize': {
            'out_height_num': 10, # the number of tiled images vertically
            'out_width_num': 10, # the number of tiled images horizontally
            'out_height': 22.0, # output image height (*100px)
            'out_width': 22.0, # output image width (*100px)
            'paths': {
                'model_file': 'generator_model.h5', # path to generator model
                'out_file': 'output.png'
                }
            }
        }
