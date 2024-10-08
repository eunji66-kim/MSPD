from models.MSPD import MSPD


def create_model(model_name, opts):
    if model_name == 'mspd':
        model = MSPD(in_channels=1, out_channels=1, num_module1=opts.num_module1, num_module2=opts.num_module2, num_module3=opts.num_module3)
    
    return model
