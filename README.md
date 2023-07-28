# Readme

![solution_structure_2023-01-24](https://user-images.githubusercontent.com/29907126/225791642-b7888797-1202-4141-8580-63cde1278b98.png)

The [Intel Neuromorphic Deep Noise Suppression Challenge (Intel N-DNS Challenge)](https://arxiv.org/abs/2303.09503) is a contest to help neuromorphic and machine learning researchers create high-quality and low-power real-time audio denoising systems. The Intel N-DNS challenge is inspired by the [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge), and it re-uses the Microsoft DNS Challenge noisy and clean speech datasets. This repository contains the challenge information, code, and documentation to get started with Intel N-DNS Challenge.

A solution to the Intel N-DNS Challenge consists of an audio **encoder**, a **neuromorphic denoiser**, and an audio **decoder**. Noisy speech is input to the encoder, which converts the audio waveform into a form suitable for processing in the neuromorphic denoiser. The neuromorphic denoiser takes this input and removes noise from the signal. Finally, the decoder converts the output of the neuromorphic denoiser into a clean output audio waveform. The Intel N-DNS Challenge consists of two tracks:

**Track 1 (Algorithmic)** aims to encourage algorithmic innovation that leads to a higher denoising performance while being efficient when implemented as a neuromorphic system. The encoder, decoder, and neuromorphic denoiser all run on CPU.

**Track 2 (Loihi 2)** aims to realize the algorithmic innovation in Track 1 on actual neuromorphic hardware and demonstrate a real-time denoising system. The encoder and decoder run on CPU and the neuromorphic denoiser runs on Loihi 2.

Solutions submitted to the Intel N-DNS challenge are evaluated in terms of an audio quality metric (denoising task performance) and computational resource usage metrics, which measure the efficiency of the solution as a system; submissions also include source code and a short write-up. Solutions will be holistically considered (metrics, write-up, innovativeness, commercial relevance, etc.) by an Intel committee for a monetary prize (details below).

Please see our [arXiv paper](https://arxiv.org/abs/2303.09503) for a more detailed overview of the challenge.

## Table of Contents
* [How to participate?](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#how-to-participate)
* [Install instructions](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#install-instructions)
* [Dataset](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#dataset)
* [Dataloader](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#dataloader)
* [Baseline Solution](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#baseline-solution)
* [Evaluation Metrics](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#evaluation-metrics)
* [Metricsboard](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#metricsboard)
* [Challenge Rules](Intel_NDNS_Challenge_Rules.pdf)

## How to participate?

Follow the [registration instructions](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#1-registration) below to participate. The overview of the challenge timeline is shown below.

```mermaid
gantt
    Title Neuromorphic DNS Challenge
    dateFormat  MM-DD-YYYY
    axisFormat  %m-%d-%Y

    Challenge start :milestone, s0, 03-16-2023, 0d

    section Track 1
    Track 1 solution development :active, t0, after s0, 138d
    Test Set 1 release           :milestone, after t0
    Track 1 submission           :t2, after t0, 30d
    Model freeze                 :crit, t2, after t0, 30d
    Track 1 evaluation           :t3, after t2, 15d
    Track 1 winner announcement  :crit, milestone, aftet t3
    
    section Track 2
    Track 2 solution development :tt0, after t0, 182d
    Test Set 2 release           :milestone, after tt0
    Track 2 submission           :tt2, after tt0, 30d
    Model freeze                 :crit, tt2, after tt0, 30d
    Track 2 evaluation           :tt3, after tt2, 15d
    Challenge winner announcement  :crit, milestone, aftet t3

```

### Important dates

|Phase|Date|
|-|-:|
|Challenge start             | Mar 16, 2023|
|Test set 1 release          | _On or about_ July 28, 2023|
|Track 1 submission deadline | _On or about_ Aug 28, 2023|
|Track 1 winner announcement | Sep 14, 2023|
|Test set 2 release          | _On or about_ Jan 28, 2024|
|Track 2 submission deadline | _On or about_ Feb 28, 2024|
|Track 2 winner announcement | Mar 14, 2024|
> Challenge dates are subject to change. Registered participants shall be notified of any changes in the dates or fixation of _on or about_ dates.


### 1. Registration
1. Create your challenge github repo (public or private) and provide access to `lava-nc-user` user.
2. Register for the challenge [here](https://intel.az1.qualtrics.com/jfe/form/SV_agZwvhRHlMw1y86).
3. You will receive a registration confirmation.

Once registered, you will receive updates about different phases of the challenges.

> Participation for Track 2 will need Loihi system cloud access which needs an Intel Neuromorphic Research Collaboration agreement. Please see [Join the INRC](https://intel-ncl.atlassian.net/wiki/spaces/INRC/pages/1784807425/Join+the+INRC) or drop an email to [inrc_interest@intel.com](mailto:inrc_interest@intel.comm). This process might take a while, so it is recommended to initiate this process as early as possible if you want to participate in Track 2.

### 2. Test Set 1 Release
Once the _test set 1_ for Track 1 is released, we will enter _Track 1 model freeze phase_. The details on test set 1 will be updated later.
* Participants shall not change their model during this phase.
* Participants shall evaluate their model on _test set 1_, measure all the necessary metrics on an Intel Core i5 quad-core machine clocked at 2.4 GHz or weaker, and submit their metrics along with a solution writeup.

> __Important:__ At least one valid metric board entry must have been submitted before _Track 1 model freeze phase_. Metricboard entries will be randomly verified.

### 3. Track 1 Winner
A committee of Intel employees will evaluate the Track 1 solutions to decide the winners, making a holistic evaluation including audio quality, computational resource usage, solution write-up quality, innovativeness, and commercial relevance.

> __Important:__ Intel reserves the right to consider and evaluate submissions at its discretion. Implementation and management of this challenge and associated prizes are subject to change at any time without notice to contest participants or winners and is at the complete discretion of Intel.

### 4. Test set 2 Release
Once the _test set 2_ for Track 2 is released, we will enter _Track 2 model freeze phase_. The details on test set 2 will be updated later.
* Participants shall not change their model during this phase.
* Participants shall evaluate their model on _test set 2_, measure all the necessary metrics on Loihi, and submit their metrics along with a solution writeup.

> __Important:__ At least one valid metric board entry must have been submitted before _Track 2 model freeze phase_. Metricboard entries will be randomly verified.

### 5. Track 2 Winner (Challenge Winner)
A committee of Intel employees will evaluate the Track 2 solutions to decide the winners, making a holistic evaluation including audio quality, computational resource usage, solution write-up quality, innovativeness, and commercial relevance.

> __Important:__ Intel reserves the right to consider and evaluate submissions at its discretion. Implementation and management of this challenge and associated prizes are subject to change at any time without notice to contest participants or winners and is at the complete discretion of Intel.


### Prize
There will be two prizes awarded
* __Track 1 winner__: fifteen thousand dollars (US \$15,000.00) or the equivalent in grant money to the winner of Track 1

and six months later,
* __Track 2 winner__: forty thousand dollars (US \$40,000.00) or the equivalent in grant money to the winner of Track 2.

These awards will be made based on the judging of the Intel committee. Where the winner is a resident from one of the named countries in the Intel N-DNS Challenge Rules and not a government employee, Intel can directly award the prize money to the winner. Where the winner is a government employee to which Intel can administer academic grant funding (regardless of whether the winner resides in one of the named countries in the Intel N-DNS Challenge Rules); a research grant in the amount for the appropriate track will be awarded to the university where the researcher/government employee is from, and in the researcher's name. Where the winner does not fall into the above categories, Intel will publicly recognize the winner, but the winner is not eligible to receive a prize. Limit of one prize per submission. 

> **Important**:
> * Researchers affiliated with universities worldwide, not restricted to the countries listed in the N-DNS Challenge Rules, are also eligible to receive prizes that Intel will administer. This includes, but is not limited to, government employees such as professors, research associates, postdoctoral research fellows, and research scientists employed by a state-funded university or research institution. Where possible, Intel will provide unrestricted gift funding to the awardee's department or group. However, universities in countries under U.S. embargo are not eligible to receive award funding.
> * Other individuals that do not fall into the above categories, but wish to enter this Contest, may do so. However, they are not eligible for any prize, but will be publicly recognized if they win. See Prizes under N-DNS Challenge Rules for further details. 
> * For avoidance of doubt, Intel has the sole discretion to determine the category of the entries to the N-DNS Award contest.

### Solution Writeup
We also ask that challenge participants submit a short (one or two page) write-up that explains the thought process that went into developing their solution. Please include what worked, what did not work, and why certain strategies were chosen versus others. While audio quality and power are key metrics for evaluating solutions, the overarching goal of this challenge is to drive neuromorphic algorithm innovation, and challenge participant learnings are extremely valuable.

This write-up can be submitted directly to Intel to maintain privacy before the track deadline, but for the write-up to be considered in the holistic evaluation of the solution for the monetary prize, we require that it be shared publicly within 14 days after the deadline for each track. Naturally, however, we encourage participants to share their write-ups publicly at any time, to help inspire others' solutions.

Additionally, we plan to invite a select group of challenge participants to present their solutions at a future Intel Neuromorphic Research Community (INRC) forum, based on their algorithmic innovation and metricsboard results as judged by the Intel committee, to share their learnings and participate in a discussion on developing new and improved neuromorphic computing challenges.

For ease of comparison, we ask that your solution write-up include a clear table with the evaluation metrics for your solution akin to the Table in the [Metricsboard](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#metricsboard).

For your writeup, please use a single-column Word document or Latex template with 1-inch margins, single-spacing, reasonable font size (11pt or 12pt; default font like Times New Roman), and up to two US letter-size or A4 pages. Please submit a PDF.

### Source code
Challenge participants must provide the source code used in the creation of their solution (model definition, final trained model, training scripts, inference scripts, etc.) with MIT or BSD3 license.

Challenge participant source code for Track 1 will be publicly released after the Track 1 winner is announced. Likewise for Track 2.

## Install Instructions
```bash
pip install -r requirements.txt
python -c "import os; from distutils.sysconfig import get_python_lib; open(get_python_lib() + os.sep + 'ndns.pth', 'a').write(os.getcwd())"
```

## Uninstall Instructions
```bash
python -c "import os; from distutils.sysconfig import get_python_lib; pth = get_python_lib() + os.sep + 'ndns.pth'; os.remove(pth) if os.path.exists(pth) else None;"
```

## Dataset

### 1. Download steps
- Edit `microsoft_dns/download-dns-challenge-4.sh` to point the desired download location and downloader
- `bash microsoft_dns/download-dns-challenge-4.sh`
- Extract all the `*.tar.bz2` files.

### 2. Download verification
- Download SHA2 [checksums](https://dns4public.blob.core.windows.net/dns4archive/dns4-datasets-files-sha1.csv.bz2) and extract it.
- Run the following to verify dataset validity.
    ```python
    import pandas as pd
    import hashlib

    def sha1_hash(file_name: str) -> str:
        file_hash = hashlib.sha1()
        with open(file_name, 'rb') as f: fb = f.read()
        file_hash.update(fb)
        return file_hash.hexdigest()

    sha1sums = pd.read_csv("dns4-datasets-files-sha1.csv.bz2", names=["size", "sha1", "path"])
    file_not_found = []
    for idx in range(len(sha1sums)):
        try:
            if sha1_hash(sha1sums['path'][idx]) != sha1sums['sha1'][idx]:
                print(sha1sums['path'][idx], 'is corrupted')
        except FileNotFoundError as e:
            file_not_found.append(sha1sums['path'][idx])

    # 336494 files
    with open('missing.log', 'wt') as f:
        f.write('\n'.join(file_not_found))
    ```

### 3. Training/Validation data synthesization
- Training dataset: `python noisyspeech_synthesizer.py -root <your dataset folder>`
- Validation dataset: `python noisyspeech_synthesizer.py -root <your dataset folder> -is_validation_set true`

### 4. Testing data
- Testing data with similar statistics as the validation dataset generated from the script above will be made available towards the end of each track. We will initiate a model freeze before the release, meaning the participants will not be able to change their trained model after that.

## Dataloader
```python
from audio_dataloader import DNSAudio

train_set = DNSAudio(root=<your dataset folder> + 'training_set/')
validation_set = DNSAudio(root=<your dataset folder> + 'validation_set/')
```

## Baseline Solution

The baseline solution is described in the [Intel N-DNS Challenge paper](https://arxiv.org/abs/2303.09503). 

The code for training and running the baseline solution can be found in this directory: `baseline_solution/sdnn_delays`. 

The training script `baseline_solution/sdnn_delays/train_sdnn.py` is run as follows:
```bash
python train_sdnn.py # + optional arguments
```

## Evaluation Metrics

The N-DNS solution will be evaluated based on multiple different metrics.
1. **SI-SNR** of the solution
2. **SI-SNRi** of the solution (improvement against both _noisy data_ and _encode+decode_ processing).
1. **DNSMOS** quality of the solution (overall, signal, background)
3. **Latency** of the solution (encode & decode latency + data buffer latency +  DNS network latency)
4. **Power** of the N-DNS network (proxy for Track 1)
5. **Power Delay Product (PDP)** of the N-DNS solution (proxy for Track 1)

### SI-SNR
This repo provides SI-SNR module which can be used to evaluate SI-SNR and SI-SNRi metrics.

$\displaystyle\text{SI-SNR} = 10\ \log_{10}\frac{\Vert s_\text{target}\Vert ^2}{\Vert e_\text{noise}\Vert ^2}$

>where\
$s = \text{zero mean target signal}$\
$\hat{s} = \text{zero mean estimate signal}$\
$s_\text{target} = \displaystyle\frac{\langle\hat s, s\rangle\,s}{\Vert s \Vert ^2}$\
$e_\text{noise} = \hat s - s_\text{target}$

- **In Code Evaluation**
    ```python
    from snr import si_snr
    score = si_snr(clean, noisy)
    ```

### DNSMOS (MOS)
This repo provides DNSMOS module which is wrapped from Microsoft DNS challenge. The resulting array is a DNSMOS score (overall, signal, noisy). It also supports batched evaluation.
- **In Code Evaluation**
    ```python
    from dnsmos import DNSMOS
    dnsmos = DNSMOS()
    quality = dnsmos(noisy)  # It is in order [ovrl, sig, bak]
    ```


Other metrics are specific to the N-DNS solution system. For reference, a detailed walkthrough of the evaluation of the baseline solution is described in [`baseline_solution/sdnn_delays/evaluate_network.ipynb`](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/blob/main/baseline_solution/sdnn_delays/evaluate_network.ipynb).

Please refer to the [Intel N-DNS Challenge paper](https://arxiv.org/abs/2303.09503) for more details about the metrics.

## Metricsboard
The evaluation metrics for participant solutions will be listed below and updated at regular intervals.

Submitting to the metricsboard will help you meaure the progress of your solution against other participating teams. Earlier submissions are encouraged.

To submit to the metricsboard, please create a ```.yml``` file with contents akin to the table below in the top level of the Github repository that you share with Intel so that we can import your metrics and update them on the public metricsboard. Please use [```example_metricsboard_writeout.py```](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/blob/main/example_metricsboard_writeout.py) as an example for how to generate a valid ```.yml``` file with standard key names. For the Track 1 validation set, name the ```.yml``` file ```metricsboard_track_1_validation.yml```.


**Track 1 (Validation Set)**
| Entry| <sub>$\text{SI-SNR}$ <sup>(dB)| <sub>$\text{SI-SNRi}$ <sup>data (dB)| <sub>$\text{SI-SNRi}$ <sup>enc+dec (dB)| <sub>$\text{MOS}$ <sup>(ovrl)| <sub>$\text{MOS}$ <sup>(sig)| <sub>$\text{MOS}$ <sup>(bak)| <sub>$\text{latency}$ <sup>enc+dec (ms)| <sub>$\text{latency}$ <sup>total (ms)| <sub>$\text{Power}$ $\text{proxy}$ <sup>(M-Ops/s) | <sub>$\text{PDP}$ $\text{proxy}$ <sup>(M-Ops)| <sub>$\text{Params}$ <sup>($\times 10^3$)|<sub>$\text{Size}$ <sup>(KB)|
|:-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| Team xyz (mm/dd/yyyy)              |       |      |      |      |      |      |       |        |        |      |       |
|   NoiCE Spiking Conv (07/27/2023)  | 13.15 | 5.53 | 5.53 |  2.8 | 3.22 | 3.64 | 0.082 |  8.082 | 6,110.78 | 49.09 | 2,100 | 8,209 |
| Microsoft NsNet2 (02/20/2023)      | 11.89 | 4.26 | 4.26 | 2.95 | 3.27 | 3.94 | 0.024 | 20.024 | 136.13 | 2.72 | 2,681 |10,500|
| Intel proprietary DNS (02/28/2023) | 12.71 | 5.09 | 5.09 | 3.09 | 3.35 | 4.08 | 0.036 |  8.036 |    -   |   -  | 1,901 | 3,802|
| Baseline SDNN solution (02/20/2023)| 12.50 | 4.88 | 4.88 | 2.71 | 3.21 | 3.46 | 0.036 |  8.036 |  11.59 | 0.09 |   525 |   465|
| Validation set                     |  7.62 |   -  |   -  | 2.45 | 3.19 | 2.72 |   -   |    -   |    -   |   -  |   -   |   -  |

**Track 2**
| Entry| <sub>$\text{SI-SNR}$ <sup>(dB)| <sub>$\text{SI-SNRi}$ <sup>data (dB)| <sub>$\text{SI-SNRi}$ <sup>enc+dec (dB)| <sub>$\text{MOS}$ <sup>(ovrl)| <sub>$\text{MOS}$ <sup>(sig)| <sub>$\text{MOS}$ <sup>(bak)| <sub>$\text{latency}$ <sup>enc+dec (ms)| <sub>$\text{latency}$ <sup>total (ms)| <sub>$\text{Power}$ <sup>(W) | <sub>$\text{PDP}$ <sup>(Ws)| <sub>$\text{Cores}$|
|:-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| Team xyz (mm/dd/yyyy) |        |         |         |         |             |           |   | | | | |

> **Note:**
> * An Intel committee will determine the challenge winner using a holistic evaluation (not one particular metric). We encourage challenge participants to strive for top performance in all metrics. 
> * Metrics shall be taken as submitted by the participants. There will be a verification process during the contest winner evaluation.


For any additional clarifications, please refer to the challenge [FAQ](faq.md) or [Rules](Intel_NDNS_Challenge_Rules.pdf) or ask questions in the [discussions](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/discussions) or email us at [ndns@intel.com](mailto:ndns@intel.com).

