import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'IterMaskAE'
  eval.first_model = 'IterMaskFirst'
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.split_size = 155
  eval.thr_start = 0.001
  eval.thr_end = 0.1
  eval.thr_step = 0.001
  eval.fpr = [0.00055, 0.01]      #[0.01, 0.0215] T1
  eval.calc_thr = False
  eval.thr_fpr = [0.14, 0.04] #T2     #[0.1991, 0.13] T1
  eval.opt_thr = 0.14  #0.13
  eval.data_is_healthy = False
  eval.validation_thr = 0.0040 #T2   #0.0022T1
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint_1 = 20
  accelerator.checkpoint_2 = 20


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""  # validation dataset for estimating the thr or the test set
  data.image_size = 224
  data.batch_size = 1
  data.num_channels = 1
  data.workers = 1
  data.data_type = 'T2w'

  
  
  config.model = model = ml_collections.ConfigDict()
  model.ema_rate = 0.9999
  model.condition = True
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 1, 1, 2, 3, 4)
  model.num_res_blocks = (1, 1, 1, 2, 4, 2)
  model.attn_resolutions = (14,7)
  model.attn_emb_dim = 32
  model.rescale = False
  model.dropout = 0.1
  model.dropout_at = 14


  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4

  
  config.seed = 42

  return config
