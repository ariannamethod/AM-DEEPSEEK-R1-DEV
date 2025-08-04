
# aguvis json file with mobile action space
MOBILE_FILE = [
    "android_control.json",
    "gui-odyssey-l1.json",
    "aitw-l3.json",
    "coat.jsonamex-l2.json",
    "amex-l1.json",
    "amex-l3.json",
    "gui-odyssey-l3.json",
    "aitw-l1.json",
    "aitw-l2.json",
    "gui-odyssey-l2.json",
]

# Processing:  guienv
# 
# Duplicate images in guienv.json. difference: 257578
# len(images_path): 327972
# len(images_set_path): 70394
# user/assistant by image: 3.6590902633747193
# 
# 
# Processing:  ricosca
# 
# Duplicate images in ricosca.json. difference: 155066
# len(images_path): 173212
# len(images_set_path): 18146
# user/assistant by image: 8.54546456519343
# 
# 
# Processing:  ui_refexp
# 
# Duplicate images in ui_refexp.json. difference: 10978
# len(images_path): 15624
# len(images_set_path): 4646
# user/assistant by image: 2.3628928110202323
# 
# Processing:  widget_captioning
# 
# Duplicate images in widget_captioning.json. difference: 87017
# len(images_path): 101426
# len(images_set_path): 14409
# user/assistant by image: 6.039072801721146

config_dict_stage_1 = [
    {
        "json_path": "guienv.json",
        "images_folder": "guienvs/images/",
    },
    {
        "json_path": "omniact.json",
        "images_folder": "omniact/images/",
    },
    {
        "json_path": "ricoig16k.json",
        "images_folder": "ricoig16k/images/",
    },
    {
        "json_path": "ricosca.json",
        "images_folder": "ricosca/images/",
    },
    {
        "json_path": "seeclick.json",
        "images_folder": "seeclick/seeclick_web_imgs/",
    },
    {
        "json_path": "webui350k.json",
        "images_folder": "webui350k/images/",
    },
    {
        "json_path": "ui_refexp.json",
        "images_folder": "ui_refexp/images/",
    },
    {
        "json_path": "widget_captioning.json",
        "images_folder": "widget_captioning/images/",
    },
    
]


# Processing:  guiact-web-single
# 
# Duplicate images in guiact-web-single.json. difference: 54134
# len(images_path): 67396
# len(images_set_path): 13262
# user/assistant by image: 4.081888101342181
# 
# Processing:  guiact-web-multi-l3
# 
# Duplicate images in guiact-web-multi-l3.json. difference: 24
# len(images_path): 16704
# len(images_set_path): 16680
# user/assistant by image: 0.0014388489208633094
# 
# Processing:  miniwob-l3
# 
# Duplicate images in miniwob-l3.json. difference: 161
# len(images_path): 9826
# len(images_set_path): 9665
# user/assistant by image: 0.016658044490429385
# 
# Processing:  gui-odyssey-l3
# 
# Duplicate images in gui-odyssey-l3.json. difference: 24
# len(images_path): 118282
# len(images_set_path): 118258
# user/assistant by image: 0.0002029461008980365


config_dict_stage_2 = [
    {
        "json_path": "mind2web-l3.json",
        "images_folder": "mind2web/",
    },
    {
        "json_path": "guiact-web-single.json",
        "images_folder": "guiact-web-single/images/",
    },
    {
        "json_path": "guiact-web-multi-l3.json",
        "images_folder": "guiact-web-multi-v2/images",
    },
    {
        "json_path": "miniwob-l3.json",
        "images_folder": "images",
    },
    {
        "json_path": "coat.json",
        "images_folder": "coat/images/",
    },
    {
        "json_path": "android_control.json",
        "images_folder": "android_control/images/",
    },
    {
        "json_path": "gui-odyssey-l3.json",
        "images_folder": "gui-odyssey/images/",
    },
    {
        "json_path": "amex-l3.json",
        "images_folder": "amex/images/",
    },
    {
        "json_path": "aitw-l3.json",
        "images_folder": "aitw-v1/images/",
    },
]
