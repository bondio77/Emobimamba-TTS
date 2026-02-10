import argparse
import os
import yaml
import torch
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm
from inference.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model


def main(configs):
    emo_dict = {
        "Neutral": 0,
        "Angry": 1,
        "Sad": 2,
        "Surprise": 4,
        "Happy": 3
    }
    language_num = 1
    acoustic_dir = f"/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/model_checkpoint/context_EGCA_mamba1_8/1000epoch.pth"
    output_dir = f"/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/model_checkpoint/context_EGCA_mamba1_8/test1000"
    voco_dir = f"./vocoder/generator_only.pth"

    symbol_path = ('/home/lee/Desktop/INSUNG/intensity_fs2/ESD_preprocess_vocs_1/symbols.json')

    text2speech = Text2Speech(
        train_configs=configs,
        model_dir=acoustic_dir,
        device='cuda:1',
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        backward_window=1,
        forward_window=3,
        symbol_path=symbol_path,
        dict_unit='phone',
        w_quantize=False,
    )

    preprocess_config, _ = configs
    vocoder = load_model(voco_dir).to('cuda:1').eval()

    intensity_level = ['min', 'med', 'max']

    with open('/home/lee/Desktop/INSUNG/intensity_fs2/ESD_preprocess_vocs_1/speakers.json', 'r') as speaker_file:
        speakers = json.load(speaker_file)

    inverted_speakers = {value: key for key, value in speakers.items()}
    val_list = open('../test.txt', 'r').readlines()

    for j, i in tqdm(enumerate(val_list)):
        wav_name, spk, phone, txt, emotion = i.split('|')
        emotion = emotion.replace('\n', '')
        for strength in intensity_level:
            with torch.no_grad():
                emo_id = emo_dict[emotion]
                spk_id = speakers[spk]
                output_dict = text2speech(
                    txt,
                    intensity_level=f'{strength}',
                    emotions=torch.from_numpy(np.array(emo_id)).long().unsqueeze(0),
                    sids=torch.from_numpy(np.array(spk_id)).long().unsqueeze(0),
                    lids=torch.from_numpy(np.array(language_num)).long().unsqueeze(0),
                )

                wav = vocoder.inference(output_dict["feat_gen"])

                save_path = output_dir + '/' + spk + '/' + strength + '/' + emotion.replace('\n', '') + '/' + wav_name + '.wav'

                txt_path = save_path.replace('wav', 'txt')
                if os.path.isdir(os.path.dirname(save_path)) == 0:
                    os.makedirs(os.path.dirname(save_path))

                f = open(txt_path, 'w')
                f.write(i.replace('\n', ''))
                f.close()
                sf.write(save_path, wav.cpu().numpy(), 24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, default="./configs/esd/preprocess.yaml", help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default="./configs/esd/model_6L.yaml", help="path to model.yaml"
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config)

    main(configs)
