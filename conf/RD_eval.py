import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'RD'
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.split_size = 155
  eval.thr_start = 0.01
  eval.thr_end = 0.3
  eval.thr_step = 0.01
  eval.fpr = [0.01, 0.019] #T2   #[0.01, 0.01767] T1
  eval.calc_thr = False
  eval.thr_fpr = [0.183, 0.17] #T2  #[0.147, 0.14] T1
  eval.opt_thr = 0.17 #T2     #0.14 T1
  eval.data_is_healthy = False
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 3   #T2    #4 T1


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 1
  data.num_channels = 1
  data.workers = 1
  data.data_type = 'T2w'

  
  
  config.model = model = ml_collections.ConfigDict()
  model.input_shape = (3,224,224)
  model.layers = ['layer1', 'layer2', 'layer3']
  model.normalize = False


  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4

  
  config.seed = 42

  return config
