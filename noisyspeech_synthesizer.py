# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/microsoft_dns/')

import glob
import argparse
import configparser as CP

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

import random
from random import shuffle
from microsoft_dns.noisyspeech_synthesizer_singleprocess import \
    add_pyreverb, build_audio, gen_audio
from microsoft_dns.audiolib import (
    audioread, audiowrite, normalize_segmental_rms, active_rms, EPS,
    activitydetector, is_clipped, add_clipping
)
from microsoft_dns import utils

MAXTRIES = 50
MAXFILELEN = 100


np.random.seed(5)
random.seed(5)


def segmental_snr_mixer(params, clean, noise, snr,
                        target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))
    clean = clean / (max(abs(clean)) + EPS)
    noise = noise / (max(abs(noise)) + EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean,
                                    target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise,
                                    target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr / 20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize
    # noisyspeech with that value.
    # There is a chance of clipping that might happen with very less
    # probability, which is not a major issue. 
    noisy_rms_level = np.random.randint(params['target_level_lower'],
                                        params['target_level_upper'])
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1.
    # If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def synthesize(params):
    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    clean_index = params['fileindex_start']
    noise_index = params['fileindex_start']
    file_num = params['fileindex_start']

    while file_num <= params['fileindex_end']:
        try:
            # generate clean speech
            clean, clean_sf, clean_cf, clean_laf, clean_index = \
                gen_audio(is_clean=True, params=params, index=clean_index)
            # generate noise
            noise, noise_sf, noise_cf, noise_laf, noise_index = \
                gen_audio(is_clean=False, params=params, index=noise_index,
                          audio_samples_length=len(clean))
            clean_clipped_files += clean_cf
            clean_low_activity_files += clean_laf
            noise_clipped_files += noise_cf
            noise_low_activity_files += noise_laf

            snr = np.random.randint(params['snr_lower'], params['snr_upper'])

            clean_snr, noise_snr, noisy_snr, target_level = \
                segmental_snr_mixer(params=params, clean=clean, noise=noise, snr=snr)
        except ValueError as e:
            print('Found exception')
            print(str(e))
            print('Trying again')
            clean_index += 1
            noise_index += 1
            continue

        # unexpected clipping
        if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            print("Warning: File #" + str(file_num) + " has unexpected clipping, " + \
                  "returning without writing audio to disk")
            continue

        clean_source_files += clean_sf
        noise_source_files += noise_sf

        # write resultant audio streams to files
        hyphen = '-'
        clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_sf]
        clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
        noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_sf]
        noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]

        noisyfilename = clean_files_joined + '_' + noise_files_joined + '_snr' + \
                        str(snr) + '_tl' + str(target_level) + '_fileid_' + str(file_num) + '.wav'
        cleanfilename = 'clean_fileid_'+str(file_num)+'.wav'
        noisefilename = 'noise_fileid_'+str(file_num)+'.wav'

        dir = 'validation_set/' if params['is_test_set'] else 'training_set/'
        noisypath = os.path.join(params['root'] + dir + params['noisy_speech_dir'], noisyfilename)
        cleanpath = os.path.join(params['root'] + dir + params['clean_speech_dir'], cleanfilename)
        noisepath = os.path.join(params['root'] + dir + params['noise_dir'], noisefilename)

        audio_signals = [noisy_snr, clean_snr, noise_snr]
        file_paths = [noisypath, cleanpath, noisepath]

        file_num += 1
        for i in range(len(audio_signals)):
            try:
                audiowrite(file_paths[i], audio_signals[i], params['fs'])
            except Exception as e:
                print(str(e))


    return clean_source_files, clean_clipped_files, clean_low_activity_files, \
           noise_source_files, noise_clipped_files, noise_low_activity_files



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fs', type=int, default=16000, help='sampling frequency')
    parser.add_argument('-root', type=str, default='./', help='root dataset directory')
    parser.add_argument('-audio_length', type=float, default=30, help='minimum length of each audio clip (s)')
    parser.add_argument('-silence_length', type=float, default=0.2, help='duration of silence introduced between clean speech utterances (s)')
    parser.add_argument('-total_hours', type=float, default=500, help='total number of hours of data required (hrs)')
    parser.add_argument('-snr_lower', type=int, default=-5, help='lower bound of SNR required (dB)')
    parser.add_argument('-snr_upper', type=int, default=20, help='upper bound of SNR required (dB)')
    parser.add_argument('-target_level_lower', type=float, default=-35, help='lower bound of target audio (dB)')
    parser.add_argument('-target_level_upper', type=float, default=-15, help='upper bound of target audio (dB)')
    parser.add_argument('-target_snr_levels', type=int, default=21, help='total number of snr levels')
    parser.add_argument('-clean_activity_threshold', type=float, default=0.6, help='activity threshold for clean speech')
    parser.add_argument('-noise_activity_threshold', type=float, default=0.0, help='activity threshold for noise')
    parser.add_argument('-is_validation_set', type=bool, default=False, help='validation data flag')
    # parser.add_argument('-use_singing_data', type=bool, default=False, help='use singing data')
    # parser.add_argument('-use_emotion_data', type=bool, default=False, help='use emotion data')
    # parser.add_argument('-use_mandarin_data', type=bool, default=False, help='use mandarin data')
    # parser.add_argument('-reverb_table', type=str, default='RIR_table_simple.csv', help='reverberation table data')
    # parser.add_argument('-lower_t60', type=float, default=0.3, help='lower bound of t60 range in seconds')
    # parser.add_argument('-upper_t60', type=float, default=1.3, help='upper bound of t60 range in seconds')
    parser.add_argument('-noisy_speech_dir', type=str, default='noisy', help='noisy speech directory')
    parser.add_argument('-clean_speech_dir', type=str, default='clean', help='clean speech directory')
    parser.add_argument('-noise_dir', type=str, default='noise', help='noise directory')
    parser.add_argument('-log_dir', type=str, default='log', help='log directory')
    parser.add_argument('-fileindex_start', type=int, default=0, help='start file idx')

    params = vars(parser.parse_args())

    root = params['root']
    params['is_test_set'] = params['is_validation_set']

    params['num_files'] = int(params['total_hours'] * 3600 / params['audio_length'])
    # params['fileindex_start'] = 0
    params['fileindex_end'] = params['fileindex_start'] + params['num_files'] - 1
    print('Number of files to be synthesized:', params['num_files'])
    print('Start idx:', params['fileindex_start'])
    print('Stop idx:', params['fileindex_end'])
    print(f'Generating synthesized data in {root}')

    clean_dir = root + 'datasets_fullband/clean_fullband'
    noise_dir = root + 'datasets_fullband/noise_fullband'
    clean_filenames = glob.glob(clean_dir + '/**/*.wav', recursive=True)
    noise_filenames = glob.glob(noise_dir + '/**/*.wav', recursive=True)

    shuffle(clean_filenames)
    shuffle(noise_filenames)

    params['cleanfilenames'] = clean_filenames
    params['num_cleanfiles'] = len(clean_filenames)

    params['noisefilenames'] = noise_filenames
    params['num_noisefiles'] = len(noise_filenames)

    # Call synthesize() to generate audio
    clean_source_files, clean_clipped_files, clean_low_activity_files, \
    noise_source_files, noise_clipped_files, noise_low_activity_files = synthesize(params)

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = params['log_dir'] + '/'
    os.makedirs(log_dir, exist_ok=True)

    utils.write_log_file(log_dir, 'source_files.csv', clean_source_files + noise_source_files)
    utils.write_log_file(log_dir, 'clipped_files.csv', clean_clipped_files + noise_clipped_files)
    utils.write_log_file(log_dir, 'low_activity_files.csv', \
                         clean_low_activity_files + noise_low_activity_files)

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = len(clean_source_files) + len(clean_clipped_files) + len(clean_low_activity_files)
    total_noise = len(noise_source_files) + len(noise_clipped_files) + len(noise_low_activity_files)
    pct_clean_clipped = round(len(clean_clipped_files)/total_clean*100, 1)
    pct_noise_clipped = round(len(noise_clipped_files)/total_noise*100, 1)
    pct_clean_low_activity = round(len(clean_low_activity_files)/total_clean*100, 1)
    pct_noise_low_activity = round(len(noise_low_activity_files)/total_noise*100, 1)

    print("Of the " + str(total_clean) + " clean speech files analyzed, " + \
          str(pct_clean_clipped) + "% had clipping, and " + str(pct_clean_low_activity) + \
          "% had low activity " + "(below " + str(params['clean_activity_threshold']*100) + \
          "% active percentage)")
    print("Of the " + str(total_noise) + " noise files analyzed, " + str(pct_noise_clipped) + \
          "% had clipping, and " + str(pct_noise_low_activity) + "% had low activity " + \
          "(below " + str(params['noise_activity_threshold']*100) + "% active percentage)")