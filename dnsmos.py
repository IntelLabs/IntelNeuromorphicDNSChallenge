import os
from typing import Dict, Tuple, List
import numpy as np
import onnxruntime as ort

# import tensorrt_utils.engine as eng
# import tensorrt_utils.inference as inf
# from onnx import ModelProto
# import tensorrt as trt 

INPUT_LENGTH = 9.01
root = os.path.dirname(os.path.abspath(__file__)) + '/'

class DNSMOS:
    """DNSMOS score evaluation module for audio signal.

    Parameters
    ----------
    model_path : str, optional
        The path of DNSMOS onnx description, by default
        'microsoft_dns/DNSMOS/DNSMOS/sig_bak_ovr.onnx'.
    sampling_rate : int, optional
        Audio sampling rate, by default 16000.
    providers : list of str
        Sequence of onnx runtime providers precedence. Default is
        ['TensorrtExecutionProvider',
        'CUDAExecutionProvider',
        'CPUExecutionProvider']. Refer to onnx runtime to get more details
        about providers.
    """
    def __init__(
        self,
        model_path: str = 'microsoft_dns/DNSMOS/DNSMOS/sig_bak_ovr.onnx',
        sampling_rate: int = 16000,
        providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ) -> None:
        self.onnx_sess = ort.InferenceSession(root + model_path,
                                              providers=providers)
        self.sampling_rate = sampling_rate

    def get_polyfit_val(self,
                        sig: float,
                        bak: float,
                        ovr: float) -> Tuple[float, float, float]:
        """Polynomial fit of raw score to DNSMOS value.

        Parameters
        ----------
        sig : float
            Raw signal score.
        bak : float
            Raw background score.
        ovr : float
            Raw overall score.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Rescaled DNSMOS score (signal, background, overall).
        """
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.00524390])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def full_score(self, audio: np.ndarray) -> Dict[str, float]:
        """Evaluate the full DNSMOS score metric.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal to evaluate.

        Returns
        -------
        Dict[str, float]
            DNSMOS evaluation scores.
        """
        len_samples = int(INPUT_LENGTH * self.sampling_rate)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / self.sampling_rate)
                       - INPUT_LENGTH) + 1
        hop_len_samples = self.sampling_rate
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples):
                              int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(
                audio_seg).astype('float32')[np.newaxis, :]
            oi = {'input_1': input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(
                None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        return clip_dict

    def __call__(self, audio: np.ndarray) -> float:
        """Overall DNSMOS evalution score.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal to evaluate.

        Returns
        -------
        float
            Tuple of overall, signal, and background DNSMOS score.
        """
        if len(audio.shape) == 1:
            full_score = self.full_score(audio)
            return np.array([full_score['OVRL'],
                             full_score['SIG'],
                             full_score['BAK']])
        elif len(audio.shape) == 2:
            batch_scores = []
            for b in range(audio.shape[0]):
                full_score = self.full_score(audio[b])
                batch_scores.append([full_score['OVRL'],
                                     full_score['SIG'],
                                     full_score['BAK']])
            return np.array(batch_scores)
        raise RuntimeError(f'Found unsupported audio of shape {audio.shape}'
                           'Expected it to be 1-D or 2-D array.')
