import yaml

def parse(opt_path, is_tain=True):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    #logger.info('Reading .yml file .......')
    with open(opt_path,mode='r') as f:
        opt = yaml.load(f,Loader=yaml.FullLoader)
    # Export CUDA_VISIBLE_DEVICES
    #gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #logger.info('Export CUDA_VISIBLE_DEVICES = {}'.format(gpu_list))

    # is_train into option
    #opt['is_train'] = is_tain
    return opt