from data import build_scenario


scenario = build_scenario(
    source_datasets_name=['MultiWoz_MultiDomains'],
    target_datasets_order=['MultiWoz_taxi', 'MultiWoz_hotel'] * 15,
    da_mode='close_set',
    data_dirs={
        'MultiWoz_MultiDomains': '/data/zql/datasets/multiwoz',
        'MultiWoz_taxi': '/data/zql/datasets/multiwoz',
        'MultiWoz_hotel': '/data/zql/datasets/multiwoz',
    },
    transforms={
        'MultiWoz_MultiDomains': None,
        'MultiWoz_taxi': None,
        'MultiWoz_hotel': None
    }
)