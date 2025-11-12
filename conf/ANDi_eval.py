import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'ANDi'
  eval.split_size = 155
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.thr_start = 0.00001 
  eval.thr_end = 0.00005  
  eval.thr_step = 0.000001
  eval.fpr = [0.01, 0.01177]   #[0.0083, 0.01] T2
  eval.calc_thr = False
  eval.thr_fpr = [2.9e-05, 2.7e-05] #[2.1e-05, 1.9327198970131576e-05]T2
  eval.opt_thr = 2.7e-05    #2.1e-05T2
  eval.data_is_healthy = False
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 6


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.data_type = 'T1w'
  data.image_size = 224
  data.batch_size = 1
  data.num_channels = 1
  data.workers = 1


  config.model = model = ml_collections.ConfigDict()
  model.ema_rate = 0.9999
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 1, 1, 2, 3, 4)
  model.num_res_blocks = (1, 1, 1, 2, 4, 2)
  model.attn_resolutions = (14,7)
  model.attn_emb_dim = 32
  model.rescale = True
  model.embedding_type = 'positional'
  model.dropout = 0.1
  model.dropout_at = 14
  model.pyramid = False
  model.discount = 0.9
  model.vpred = True


  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4


  config.diffusion = diffusion = ml_collections.ConfigDict()
  diffusion.num_latents = 1000
  diffusion.schedule = 'cosine'
  diffusion.logsnr_min = -15
  diffusion.logsnr_max = 15
  diffusion.shift = True


  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.aggregation = 'gmean'
  sampling.start = 125
  sampling.stop = 25


  config.seed = 42


  return config
