class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/dingzhaodong/project/second_work'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/dingzhaodong/project/second_work/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/dingzhaodong/project/second_work/pretrained_networks'
        self.got10k_val_dir = '/data/dingzhaodong/project/second_work/data/got10k/val'
        self.lasot_lmdb_dir = '/data/dingzhaodong/project/second_work/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/dingzhaodong/project/second_work/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/dingzhaodong/project/second_work/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/dingzhaodong/project/second_work/data/coco_lmdb'
        self.coco_dir = '/data/dingzhaodong/project/second_work/data/coco'
        self.lasot_dir = '/data/dingzhaodong/project/second_work/data/lasot'
        self.got10k_dir = '/data/dingzhaodong/project/second_work/data/got10k/train'
        self.trackingnet_dir = '/data/dingzhaodong/project/second_work/data/trackingnet'
        self.depthtrack_dir = '/data/dingzhaodong/project/second_work/data/depthtrack/train'
        self.lasher_dir = '/data/dingzhaodong/project/second_work/data/LasHeR'
        self.visevent_dir = '/data/dingzhaodong/project/second_work/data/visevent/train'
