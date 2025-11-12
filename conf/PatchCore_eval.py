import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'PatchCore'
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.split_size = 64
  eval.thr_start = 0.6   
  eval.thr_end = 0.9     
  eval.thr_step = 0.01
  eval.fpr = [0.00027, 0.01] #T2    #[0.00042, 0.01] T1
  eval.calc_thr = False
  eval.thr_fpr = [0.87, 0.698] #T2  #[0.84, 0.665] T1
  eval.opt_thr = 0.87            #0.84 T1
  eval.data_is_healthy = False
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 1
  data.num_channels = 1
  data.workers = 1
  data.data_type = 'T2w'

  
  
  config.model = model = ml_collections.ConfigDict()
  model.num_neighbors = 1
  model.patchsize = 5
  model.patchstride = 1
  model.input_shape = (3, 224, 224)
  model.pretrain_embed_dim = 1024
  model.target_embed_dim = 1024
  model.backbone = 'WideResNet50'
  model.layers = ['layer2', 'layer3']
  model.normalize = False
  model.index_file = 'models/PatchCore/my.index'
  model.sample_size = 0.01
  model.faiss_gpu = True


  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4

  config.seed = 42

  return config
