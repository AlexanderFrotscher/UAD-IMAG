import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # eval
  config.eval = eval = ml_collections.ConfigDict()
  eval.run_name = 'RIAD'
  eval.output = "/results/"
  eval.output_mf = "/results/"
  eval.split_size = 155
  eval.thr_start = 0
  eval.thr_end = 0.03
  eval.thr_step = 0.001
  eval.fpr = [0.01, 0.086]      #[0.01, 0.146] T1
  eval.calc_thr = False
  eval.thr_fpr = [0.013, 0.008]  #[0.012, 0.006] T1
  eval.opt_thr = 0.008   #0.006 T1
  eval.data_is_healthy = False
  eval.k = [8,16,28,32]  # size rectangular regions
  eval.n = 5  # number disjoint subsets
  if eval.calc_thr:
    assert eval.data_is_healthy
  eval.slice = False
  eval.distribution = False


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.checkpoint = 20

  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.data_type = 'T2w'
  data.image_size = 224
  data.batch_size = 1
  data.num_channels = 1
  data.workers = 1
  data.num_channels = 1


  config.model = model = ml_collections.ConfigDict()
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 1, 1, 2, 3, 4)
  model.num_res_blocks = (1, 1, 1, 2, 3, 2)
  model.attn_resolutions = (14,7)
  model.attn_emb_dim = 32
  model.rescale = False
  model.embedding_type = None
  model.dropout = 0.1
  model.dropout_at = 14

  config.postprocessing = postprocessing = ml_collections.ConfigDict()
  postprocessing.kernel_size = 4


  config.seed = 42

  return config
