from data import build_scenario


semantic_segmentation_scenario = build_scenario(
    source_datasets_name=['GTA5', 'SuperviselyPerson'],
    target_datasets_order=['Cityscapes', 'BaiduPerson'] * 15,
    da_mode='close_set',
    data_dirs={
        'GTA5': '/data/zql/datasets/GTA-ls-copy/GTA5',
        'SuperviselyPerson': '/data/zql/datasets/supervisely_person/Supervisely Person Dataset',
        'Cityscapes': '/data/zql/datasets/cityscape/',
        'BaiduPerson': '/data/zql/datasets/baidu_person/clean_images/'
    },
    transforms={
        'GTA5': None,
        'SuperviselyPerson': None,
        'Cityscapes': None,
        'BaiduPerson': None
    }
)