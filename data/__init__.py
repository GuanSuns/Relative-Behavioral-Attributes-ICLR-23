import os

from omegaconf import OmegaConf


def get_attribute_data(cfg, read_test_data=False):
    # read attribute info
    domain_description = OmegaConf.load(cfg.domain_description_file)
    OmegaConf.update(cfg, 'attr_info',
                     domain_description.attr_info,
                     force_add=True)

    # determine whether to use language attr representation
    use_language_attr = False
    if 'use_language_attr' in cfg and cfg.use_language_attr:
        use_language_attr = True

    attr_training_data, attr_test_data = None, None
    from data.attribute_data import Attribute_Data
    attr_training_data = Attribute_Data(data_path=os.path.join(cfg.dataset_dir, 'training_data.pickle'),
                                        attr_info=cfg.attr_info,
                                        use_language_attr=use_language_attr,
                                        is_video_data=cfg.attr_info.is_video_data,
                                        encoder_path=cfg.attr_func.encoder_path,
                                        frame_stack=cfg.frame_stack,
                                        image_size=cfg.image_size,
                                        subset_selector=cfg.attr_training_subset)
    if read_test_data:
        attr_test_data = Attribute_Data(data_path=os.path.join(cfg.dataset_dir, 'test_data.pickle'),
                                        attr_info=cfg.attr_info,
                                        use_language_attr=use_language_attr,
                                        is_video_data=cfg.attr_info.is_video_data,
                                        encoder_path=cfg.attr_func.encoder_path,
                                        frame_stack=cfg.frame_stack,
                                        image_size=cfg.image_size,
                                        subset_selector=cfg.attr_test_subset)
    return attr_training_data, attr_test_data

