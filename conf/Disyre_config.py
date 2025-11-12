import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.run_name = 'Disyre'
  training.n_iters = 100000
  training.snapshot_freq = 5000
  training.log_freq = 50
  


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.train_continue = False
  accelerator.checkpoint = 0


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.shape_dataset = "......shape_data.tsv" # path to the shape dataset
  data.image_size = 128
  data.num_channels = 1
  data.batch_size = 64
  data.workers = 4
  data.anom_type = 'dag'
  data.anom_patch_size = [64,64]
  data.no_anom_in_background = True
  data.p_anomaly = 1.0
  data.codebook = [0.6079509, 0.07953554, 0.4649098, 0.78062236, 0.31114596] # codebook T1
  #data.codebook = [0.04920235, 0.5074527, 0.2573119, 0.7149557, 0.37794548] # T2


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


  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.steps = 5
  sampling.num_up = 3


  config.optim = optim = ml_collections.ConfigDict()
  optim.lr = 1e-4



  config.seed = 42

  return config
