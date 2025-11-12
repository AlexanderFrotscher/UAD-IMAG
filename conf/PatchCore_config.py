import ml_collections


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.run_name = 'PatchCore'
  


  config.accelerator = accelerator = ml_collections.ConfigDict()
  accelerator.train_continue = False
  accelerator.checkpoint = 0


  config.data = data = ml_collections.ConfigDict()
  data.dataset = ""
  data.image_size = 224
  data.batch_size = 64
  data.num_channels = 1
  data.workers = 4

  
  
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
  model.faiss_gpu = False
  model.sample_size = 0.01
  model.index_file = 'models/PatchCore/my.index'


  config.seed = 42

  return config
