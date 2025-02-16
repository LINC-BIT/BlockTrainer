from data import build_scenario


# pos_tagging_scenario = build_scenario(
#     source_datasets_name=[i + '-TokenCls-OPT' for i in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
#                             'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']],
#     target_datasets_order=[i + '-TokenCls-OPT' for i in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker', 
#                            'Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
#                            'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
#                            'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100', 
#                            'SemEval-Laptop', 'SemEval-Rest'] * 2 + ['Liu3Domains-Computer', 'Liu3Domains-Router']],
#     da_mode='close_set',
#     data_dirs={
#         **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
#             for k in [i + '-TokenCls-OPT' for i in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
#                             'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']]},
        
#         **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing3Domains/asc/{k.split("-")[1]}' 
#             for k in [i + '-TokenCls-OPT'  for i in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker']]},
        
#         **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc/{k.split("-")[1]}' 
#             for k in [i + '-TokenCls-OPT'  for i in ['Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
#                            'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
#                            'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100']]},
        
#         **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/XuSemEval/asc/14/{k.split("-")[1].lower()}' 
#             for k in [i + '-TokenCls-OPT'  for i in ['SemEval-Laptop', 'SemEval-Rest']]},
#     },
#     transforms={
#         n: None for n in [i + '-TokenCls-OPT' for i in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
#                             'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']] + [i + '-TokenCls-OPT' for i in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker', 
#                            'Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
#                            'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
#                            'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100', 
#                            'SemEval-Laptop', 'SemEval-Rest'] * 2 + ['Liu3Domains-Computer', 'Liu3Domains-Router']]
#     }
# )

# o = [i + '-TokenCls-OPT' for i in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
#                             'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']] + [i + '-TokenCls-OPT' for i in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker', 
#                            'Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
#                            'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
#                            'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100', 
#                            'SemEval-Laptop', 'SemEval-Rest'] * 2 + ['Liu3Domains-Computer', 'Liu3Domains-Router']]
# import random
# random.shuffle(o)


pos_tagging_scenario = build_scenario(
    source_datasets_name=['WMT14-TokenCls-OPT'],
    target_datasets_order=['Ding9Domains-ipod-TokenCls-OPT', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB-TokenCls-OPT', 'Ding9Domains-CanonPowerShotSD500-TokenCls-OPT', 'Ding9Domains-CanonS100-TokenCls-OPT', 'Ding9Domains-LinksysRouter-TokenCls-OPT', 'Ding9Domains-MicroMP3-TokenCls-OPT', 'Ding9Domains-HitachiRouter-TokenCls-OPT', 'Ding9Domains-Nokia6600-TokenCls-OPT', 'Liu3Domains-Computer-TokenCls-OPT', 'Liu3Domains-Speaker-TokenCls-OPT', 'Liu3Domains-Computer-TokenCls-OPT', 'Ding9Domains-CanonS100-TokenCls-OPT', 'Ding9Domains-DiaperChamp-TokenCls-OPT', 'Liu3Domains-Router-TokenCls-OPT', 'Liu3Domains-Speaker-TokenCls-OPT', 'Ding9Domains-ipod-TokenCls-OPT', 'SemEval-Laptop-TokenCls-OPT', 'HL5Domains-ApexAD2600Progressive-TokenCls-OPT', 'SemEval-Rest-TokenCls-OPT', 'Ding9Domains-DiaperChamp-TokenCls-OPT', 'Liu3Domains-Router-TokenCls-OPT', 'HL5Domains-CanonG3-TokenCls-OPT', 'Liu3Domains-Router-TokenCls-OPT', 'HL5Domains-NikonCoolpix4300-TokenCls-OPT', 'Ding9Domains-LinksysRouter-TokenCls-OPT', 'Ding9Domains-CanonPowerShotSD500-TokenCls-OPT', 'Liu3Domains-Computer-TokenCls-OPT', 'Ding9Domains-HitachiRouter-TokenCls-OPT', 'Ding9Domains-Norton-TokenCls-OPT', 'SemEval-Rest-TokenCls-OPT', 'SemEval-Laptop-TokenCls-OPT', 'HL5Domains-Nokia6610-TokenCls-OPT', 'Ding9Domains-MicroMP3-TokenCls-OPT', 'Ding9Domains-Nokia6600-TokenCls-OPT', 'Ding9Domains-Norton-TokenCls-OPT'][0: 30],
    # target_datasets_order=[i + '-TokenCls-OPT' for i in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
    #                         'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']] + [i + '-TokenCls-OPT' for i in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker', 
    #                        'Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
    #                        'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
    #                        'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100', 
    #                        'SemEval-Laptop', 'SemEval-Rest'] * 2 + ['Liu3Domains-Computer', 'Liu3Domains-Router']],
    da_mode='close_set',
    data_dirs={
        'WMT14-TokenCls-OPT': f'/data/zql/datasets/wmt14/de-en',
        'Opus-TokenCls-OPT': f'/data/zql/datasets/opus_books/de-en',
        'News-TokenCls-OPT': f'/data/zql/datasets/news_commentary/de-en',
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing5Domains/asc/{k.split("-")[1]}' 
            for k in [i + '-TokenCls-OPT' for i in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                            'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']]},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing3Domains/asc/{k.split("-")[1]}' 
            for k in [i + '-TokenCls-OPT'  for i in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker']]},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/Bing9Domains/asc/{k.split("-")[1]}' 
            for k in [i + '-TokenCls-OPT'  for i in ['Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
                           'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
                           'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100']]},
        
        **{k: f'/data/zql/datasets/nlp_asc_19_domains/dat/absa/XuSemEval/asc/14/{k.split("-")[1].lower()}' 
            for k in [i + '-TokenCls-OPT'  for i in ['SemEval-Laptop', 'SemEval-Rest']]},
    },
    transforms={
        'WMT14-TokenCls-OPT': None,
        'Opus-TokenCls-OPT': None,
        'News-TokenCls-OPT': None,
        **{n: None for n in [i + '-TokenCls-OPT' for i in ['HL5Domains-ApexAD2600Progressive', 'HL5Domains-CanonG3', 'HL5Domains-CreativeLabsNomadJukeboxZenXtra40GB',
                            'HL5Domains-NikonCoolpix4300', 'HL5Domains-Nokia6610']] + [i + '-TokenCls-OPT' for i in ['Liu3Domains-Computer', 'Liu3Domains-Router', 'Liu3Domains-Speaker', 
                           'Ding9Domains-DiaperChamp', 'Ding9Domains-Norton', 'Ding9Domains-LinksysRouter', 
                           'Ding9Domains-MicroMP3', 'Ding9Domains-Nokia6600', 'Ding9Domains-CanonPowerShotSD500', 
                           'Ding9Domains-ipod', 'Ding9Domains-HitachiRouter', 'Ding9Domains-CanonS100', 
                           'SemEval-Laptop', 'SemEval-Rest'] * 2 + ['Liu3Domains-Computer', 'Liu3Domains-Router']]}
    },
)