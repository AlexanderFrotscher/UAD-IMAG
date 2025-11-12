import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'Disyre'
  eval.split_size = 155
  eval.sw_batch_size = 310
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.thr_start = 0.001
  eval.thr_end = 0.1
  eval.thr_step = 0.001
  eval.fpr = [0.00108, 0.01] #T2     #[0.00169, 0.01] T1
  eval.calc_thr = False
  eval.thr_fpr = [0.032, 0.0098] #T2   #[0.021, 0.008] T1
  eval.opt_thr = 0.032 #T2  #0.021 T1
  eval.data_is_healthy = False
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False
  


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 20


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ''
  data.data_type = 'T1w'
  data.image_size = 128
  data.load_size = 224
  data.batch_size = 1
  data.num_channels = 1
  data.workers = 1


  config.model = model = ml_collections.ConfigDict()
  model.ema_rate = 0.9999
  model.nonlinearity = 'swish'
  model.nf = 64
  model.ch_mult = (1, 2, 2, 4, 4)
  model.num_res_blocks = (1, 1, 2, 4, 2)
  model.attn_resolutions = (16,8)
  model.attn_emb_dim = 64
  model.rescale = True
  model.embedding_type = 'positional'
  model.dropout = 0.1
  model.dropout_at = 8


  config.diffusion = diffusion = ml_collections.ConfigDict()
  diffusion.num_latents = 100
  diffusion.beta_min = 0.1
  diffusion.beta_max = 20.

  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4


  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps = 4


  config.seed = 42

  return config
